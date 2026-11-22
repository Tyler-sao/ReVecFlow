"""
Custard (SAM .gv) -> Comal ComalGraph protobuf converter for the C+S baseline.

    python cs_gv2proto.py <graph.gv> <out.bin> [--pb-dir DIR] [--name NAME] [--report out.json]

Custard (SAM's TACO fork, `taco ... --print-sam-graph=X.gv`) emits a SAM dataflow graph as graphviz text; Comal only reads
the tortilla ComalGraph protobuf that FuseFlow writes. This converter maps the 12 SAM node types Custard prints onto the
Comal operations one to one; nothing is optimised or fused differently from what Custard emitted.

Stream model. Every gv edge becomes one Comal stream id (one sender, one receiver). Broadcast nodes become Broadcast
operations of the edge type. An output port with more than one consumer and no broadcast node gets a Broadcast.

The one structural difference: SAM has a separate RepeatSignalGenerator (crd -> repsig) feeding Repeat; Comal's Repeat
generates the signal itself from a stream in the REF map (proto field rep_ref). Comal's RepeatSigGen turns every value and
every empty token into one repeat (templates/repeat.rs), so the ref stream that the producer of the repsiggen's crd emits
alongside it (same level, token for token aligned: FiberLookup's ref, a Joiner's output ref) drives Repeat identically.
The converter connects that ref (adding a Broadcast output) and drops the repsiggen. A crd whose producer has no aligned
ref stream is refused (no silent substitute).

Roots: a fiberlookup or repeat with root=true and no incoming ref gets its own Generator (Comal's root: one value, then D).
"""
import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ATTR = re.compile(r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|[^\s,\]]+)')
STMT = re.compile(r'^\s*(\w+)\s*(?:->\s*(\w+)\s*)?\[(.*?)\]\s*;?\s*$', re.S | re.M)


class ConvertError(Exception):
    pass


def attrs_of(s):
    out = {}
    for k, v in ATTR.findall(s):
        out[k] = v[1:-1] if v.startswith('"') else v
    return out


def parse_gv(text):
    body = text[text.index("{") + 1:text.rindex("}")]
    graph_comment = None
    m = re.search(r'^\s*comment\s*=\s*"([^"]*)"', body, re.M)
    if m:
        graph_comment = m.group(1)
    nodes, edges = {}, []
    # statements may span lines (labels contain raw newlines); find each "... [ ... ]" block
    for st in re.finditer(r'(\w+)\s*(?:->\s*(\w+)\s*)?\[((?:[^\]"]|"(?:[^"\\]|\\.)*")*)\]', body):
        src, dst, a = st.group(1), st.group(2), attrs_of(st.group(3))
        if dst is None:
            if "type" not in a:
                raise ConvertError(f"node {src} has no type: {a}")
            nodes[src] = a
        else:
            if "type" not in a:
                raise ConvertError(f"edge {src}->{dst} has no type: {a}")
            edges.append({"src": src, "dst": dst, "type": a["type"], "comment": a.get("comment", ""),
                          "label": a.get("label", "")})
    for e in edges:
        for end in ("src", "dst"):
            if e[end] not in nodes:
                raise ConvertError(f"edge endpoint {e[end]} is not a node")
    return graph_comment, nodes, edges


def load_pb(pb_dir):
    sys.path.insert(0, str(pb_dir))
    import comal_pb2, ops_pb2, tortilla_pb2, stream_pb2  # noqa: E401
    return comal_pb2, ops_pb2, tortilla_pb2, stream_pb2


class Converter:
    def __init__(self, nodes, edges, pb, tensor_order=()):
        self.nodes, self.edges = nodes, edges
        self.tensor_order = list(tensor_order)
        self.comal_pb2, self.ops, self.tort, self.stream = pb
        self.ids = itertools.count(1)
        for e in edges:
            e["id"] = next(self.ids)
        self.ins = defaultdict(list)
        self.outs = defaultdict(list)
        for e in edges:
            self.ins[e["dst"]].append(e)
            self.outs[e["src"]].append(e)
        self.operators = []
        self.extra_bcast_out = defaultdict(list)    # edge id of a broadcast's input -> extra output stream ids
        self.extra_port_out = defaultdict(list)     # (node, type, comment) -> extra consumer stream ids
        self.report = {"nodes": defaultdict(int), "operators": defaultdict(int), "repeat_rewired": [], "product_order": []}

    # ---- stream helpers ----
    def S(self, kind, sid):
        cls = {"crd": self.stream.CrdStream, "ref": self.stream.RefStream, "val": self.stream.ValStream,
               "repsig": self.stream.RepSigStream}[kind]
        s = cls()
        s.id.id = sid
        return s

    def node_type(self, n):
        return self.nodes[n]["type"]

    def one(self, lst, what):
        if len(lst) != 1:
            raise ConvertError(f"expected exactly one {what}, found {len(lst)}: {[ (e['src'], e['dst'], e['type'], e['comment']) for e in lst]}")
        return lst[0]

    def in_edges(self, n, typ, comment=None):
        return [e for e in self.ins[n] if e["type"] == typ and (comment is None or e["comment"] == comment)]

    def out_ports(self, n):
        """outgoing edges grouped by (type, source-side port). Source-side port = type for single-output nodes;
        for joiners the ref outputs are told apart by their 'out-T' comment."""
        ports = defaultdict(list)
        for e in self.outs[n]:
            key = (e["type"], e["comment"] if e["comment"].startswith("out-") else "")
            ports[key].append(e)
        return ports

    def port_stream(self, n, typ, comment=""):
        """The single stream id leaving node n on port (typ, comment); several consumers -> a Broadcast is added and
        its INPUT id is returned. None if the port has no consumer."""
        key = (typ, comment)
        consumers = [e for e in self.out_ports(n).get(key, []) if self.node_type(e["dst"]) != "repsiggen"]
        extra = self.extra_port_out.get((n, typ, comment), [])
        if not consumers and not extra:
            return None
        if len(consumers) == 1 and not extra:
            return consumers[0]["id"]
        src_id = next(self.ids)
        self.add_broadcast(typ, src_id, [e["id"] for e in consumers] + extra, label=f"auto_bcast_{n}_{typ}")
        return src_id

    def add_broadcast(self, typ, in_id, out_ids, label):
        op = self.ops.Operation(name="broadcast", id=next(self.ids))
        b = op.broadcast
        b.label = label
        bundle = {"crd": b.crd, "ref": b.ref, "val": b.val, "repsig": b.repsig}[typ]
        bundle.input.CopyFrom(self.S(typ, in_id))
        for o in out_ids:
            bundle.outputs.append(self.S(typ, o))
        self.push(op, "Broadcast")

    def push(self, op, kind):
        self.operators.append(op)
        self.report["operators"][kind] += 1

    def generator(self, label, tensor=""):
        sid = next(self.ids)
        op = self.ops.Operation(name="root", id=next(self.ids))
        op.root.label = label
        op.root.tensor = tensor
        op.root.output_ref.CopyFrom(self.S("ref", sid))
        self.push(op, "Generator")
        return sid

    # ---- repsiggen -> aligned ref ----
    def trace_crd_producer(self, edge):
        """follow a crd edge back through broadcast nodes to the node that produced the coordinates"""
        e = edge
        seen = 0
        while self.node_type(e["src"]) == "broadcast":
            e = self.one(self.ins[e["src"]], f"input of broadcast {e['src']}")
            seen += 1
            if seen > 64:
                raise ConvertError("broadcast cycle")
        return e

    def aligned_ref_for(self, crd_edge, rep_node):
        prod_edge = self.trace_crd_producer(crd_edge)
        p = prod_edge["src"]
        t = self.node_type(p)
        if t == "fiberlookup":
            key = ("ref", "")
        elif t in ("intersect", "union"):
            refs = sorted({k for k in self.out_ports(p) if k[0] == "ref"}, key=lambda k: k[1])
            if not refs:
                raise ConvertError(f"repeat {rep_node}: joiner {p} has no output ref to drive the repeat signal")
            key = refs[0]
        else:
            raise ConvertError(f"repeat {rep_node}: repsiggen crd comes from a {t} node ({p}), which has no aligned ref "
                               f"stream; Comal's Repeat needs one (refusing to substitute)")
        new_id = next(self.ids)
        # if that ref port feeds a gv broadcast node, add the new output there; else register an extra port consumer
        cons = self.out_ports(p).get(key, [])
        if len(cons) == 1 and self.node_type(cons[0]["dst"]) == "broadcast":
            self.extra_bcast_out[cons[0]["id"]].append(new_id)
        else:
            self.extra_port_out[(p, key[0], key[1])].append(new_id)
        self.report["repeat_rewired"].append({"repeat": rep_node, "crd_producer": p, "producer_type": t,
                                              "ref_port": key[1] or "ref"})
        return new_id

    # ---- conversion (driven by convert() below) ----
    def unknown(self, n, a):
        raise ConvertError(f"node {n}: SAM node type {a['type']!r} is not handled by the converter")

    def c_repsiggen(self, n, a):
        pass                                   # folded into the Repeat (see module docstring)

    def c_broadcast(self, n, a):
        e_in = self.one(self.ins[n], f"input of broadcast {n}")
        if e_in["type"] == "repsig":
            return                             # repsig fan-out: every Repeat makes its own signal (see docstring)
        outs = [e["id"] for e in self.outs[n] if not (e_in["type"] == "crd" and self.node_type(e["dst"]) == "repsiggen")]
        outs += self.extra_bcast_out.get(e_in["id"], [])
        if any(e["type"] != e_in["type"] for e in self.outs[n]):
            raise ConvertError(f"broadcast {n} mixes stream types")
        if not outs:
            return
        self.add_broadcast(e_in["type"], e_in["id"], outs, label=f"bcast_{n}")

    def c_fiberlookup(self, n, a):
        op = self.ops.Operation(name="fiberlookup", id=int(n) + 1_000_000)
        f = op.fiber_lookup
        ins = self.in_edges(n, "ref")
        if ins:
            f.input_ref.CopyFrom(self.S("ref", self.one(ins, f"ref input of fiberlookup {n}")["id"]))
        elif a.get("root") == "true":
            f.input_ref.CopyFrom(self.S("ref", self.generator(f"root_{n}", a.get("tensor", ""))))
        else:
            raise ConvertError(f"fiberlookup {n}: no input ref and not a root")
        crd = self.port_stream(n, "crd")
        ref = self.port_stream(n, "ref")
        if crd is None:                        # only fed a dropped repsiggen: sunk in convert()
            crd = next(self.ids)
        if ref is None:
            raise ConvertError(f"fiberlookup {n}: output ref has no consumer (Comal has no ref sink)")
        f.output_crd.CopyFrom(self.S("crd", crd))
        f.output_ref.CopyFrom(self.S("ref", ref))
        f.index, f.tensor, f.mode = a["index"], a["tensor"], int(a["mode"])
        f.format, f.src, f.root = a["format"], a.get("src") == "true", a.get("root") == "true"
        f.label = f"FiberLookup {a['index']}: {a['tensor']}{a['mode']}"
        self.push(op, "FiberLookup")

    def c_joiner(self, n, a, kind):
        op = self.ops.Operation(name=kind, id=int(n) + 1_000_000)
        j = op.joiner
        j.join_type = self.ops.Joiner.INTERSECT if kind == "intersect" else self.ops.Joiner.UNION
        j.index, j.label = a["index"], f"{kind} {a['index']}"
        tensors = sorted({e["comment"][3:] for e in self.ins[n] if e["comment"].startswith("in-")})
        if len(tensors) < 2:
            raise ConvertError(f"{kind} {n}: fewer than two input tensors {tensors}")
        for T in tensors:
            pair = j.input_pairs.add()
            pair.crd.CopyFrom(self.S("crd", self.one(self.in_edges(n, "crd", f"in-{T}"), f"crd in-{T} of {kind} {n}")["id"]))
            pair.in_ref.ref_stream.CopyFrom(self.S("ref", self.one(self.in_edges(n, "ref", f"in-{T}"), f"ref in-{T} of {kind} {n}")["id"]))
        for T in tensors:                      # output_refs[i] belongs to input_pairs[i]
            sid = self.port_stream(n, "ref", f"out-{T}")
            if sid is None:                    # unused output: give it a stream that nothing reads -> sink it
                sid = next(self.ids)
                self.sink_ref(sid, f"{kind}_{n}_out_{T}")
            j.output_refs.add().ref_stream.CopyFrom(self.S("ref", sid))
        crd = self.port_stream(n, "crd")
        if crd is None:                        # only fed a dropped repsiggen: a sink is attached in convert()
            crd = next(self.ids)
        j.output_crd.CopyFrom(self.S("crd", crd))
        self.push(op, "Joiner")

    def c_intersect(self, n, a):
        self.c_joiner(n, a, "intersect")

    def c_union(self, n, a):
        self.c_joiner(n, a, "union")

    def sink_ref(self, sid, label):
        raise ConvertError(f"{label}: joiner output ref with no consumer (Comal has no ref sink); graph not supported")

    def c_repeat(self, n, a):
        op = self.ops.Operation(name="repeat", id=int(n) + 1_000_000)
        r = op.repeat
        ins = self.in_edges(n, "ref")
        if ins:
            r.in_ref.CopyFrom(self.S("ref", self.one(ins, f"ref input of repeat {n}")["id"]))
        elif a.get("root") == "true":
            r.in_ref.CopyFrom(self.S("ref", self.generator(f"root_rep_{n}", a.get("tensor", ""))))
        else:
            raise ConvertError(f"repeat {n}: no input ref and not a root")
        r.rep_ref.CopyFrom(self.S("ref", self.rep_ref[n]))
        out = self.port_stream(n, "ref")
        if out is None:
            raise ConvertError(f"repeat {n}: output ref unused")
        r.out_ref.CopyFrom(self.S("ref", out))
        r.label, r.index, r.root, r.tensor = f"Repeat {a['index']}: {a.get('tensor', '')}", a["index"], a.get("root") == "true", a.get("tensor", "")
        self.push(op, "Repeat")

    def c_arrayvals(self, n, a):
        op = self.ops.Operation(name="arrayvals", id=int(n) + 1_000_000)
        r = op.array
        r.tensor, r.label = a["tensor"], f"Array Vals: {a['tensor']}"
        r.input_ref.CopyFrom(self.S("ref", self.one(self.in_edges(n, "ref"), f"ref input of arrayvals {n}")["id"]))
        out = self.port_stream(n, "val")
        if out is None:
            raise ConvertError(f"arrayvals {n}: output unused")
        r.output_val.CopyFrom(self.S("val", out))
        self.push(op, "ArrayVals")

    def alu(self, n, a, opcode):
        op = self.ops.Operation(name=a["type"], id=int(n) + 1_000_000)
        u = op.alu
        vin = self.in_edges(n, "val")
        if len(vin) != 2:
            raise ConvertError(f"{a['type']} {n}: {len(vin)} value inputs (expected 2)")
        if opcode == self.ops.ALU.SUB:
            # Custard prints x = b - y as add(sub=1); which operand is the minuend is not encoded in the gv: refuse
            raise ConvertError(f"add {n} with sub=1: operand order is not recoverable from the gv")
        if opcode == self.ops.ALU.MUL:
            vin = self.order_product_inputs(n, vin)
        for e in vin:
            u.vals.inputs.append(self.S("val", e["id"]))
        out = self.port_stream(n, "val")
        if out is None:
            raise ConvertError(f"{a['type']} {n}: output unused")
        u.vals.output.CopyFrom(self.S("val", out))
        st = u.stages.add()
        st.inputs.extend([0, 1])
        st.output = 2
        st.op = opcode
        u.output_val = 2
        u.label = a["type"]
        self.push(op, "ALU")

    def val_tensor(self, e):
        """tensor read by the arrayvals that produced value edge e (through broadcasts), else None"""
        while self.node_type(e["src"]) == "broadcast":
            e = self.one(self.ins[e["src"]], f"input of broadcast {e['src']}")
        a = self.nodes[e["src"]]
        return a.get("tensor") if a["type"] == "arrayvals" else None

    def order_product_inputs(self, n, vin):
        """Comal's true-block MUL is a tile matrix product (val1 . val2), so the operands must follow the expression:
        both inputs read straight from arrayvals -> ordered as the tensors appear in the graph comment (Custard writes
        it in expression order: output, then operands left to right). Otherwise the edge order is kept, and a product
        whose order cannot be established is reported."""
        t = [self.val_tensor(e) for e in vin]
        if all(x in self.tensor_order for x in t):
            vin = sorted(vin, key=lambda e: self.tensor_order.index(self.val_tensor(e)))
            self.report["product_order"].append({"mul": n, "inputs": [self.val_tensor(e) for e in vin]})
        else:
            self.report["product_order"].append({"mul": n, "inputs": t, "note": "edge order (not both arrayvals)"})
        return vin

    def c_mul(self, n, a):
        self.alu(n, a, self.ops.ALU.MUL)

    def c_add(self, n, a):
        self.alu(n, a, self.ops.ALU.SUB if a.get("sub") == "1" else self.ops.ALU.ADD)

    def c_reduce(self, n, a):
        op = self.ops.Operation(name="reduce", id=int(n) + 1_000_000)
        r = op.reduce
        r.input_val.CopyFrom(self.S("val", self.one(self.in_edges(n, "val"), f"val input of reduce {n}")["id"]))
        out = self.port_stream(n, "val")
        if out is None:
            raise ConvertError(f"reduce {n}: output unused")
        r.output_val.CopyFrom(self.S("val", out))
        r.label, r.reduce_type = "Reduce", self.ops.Reduce.ADD
        self.push(op, "Reduce")

    @staticmethod
    def crd_name(comment):
        for p in ("in-", "outer-", "inner-", "out-"):
            if comment.startswith(p):
                return comment[len(p):]
        return comment

    def crd_manager(self, n, a, kind):
        """Custard's crddrop/crdhold: inputs named by the coordinate in the edge comment ("i", "in-k", "outer-j"),
        outputs "outer-<outer>" / "inner-<inner>". An output nobody reads gets a sink (convert())."""
        op = self.ops.Operation(name=kind, id=int(n) + 1_000_000)
        m = op.coord_drop if kind == "crddrop" else op.coord_hold
        outer, inner = a["outer"], a["inner"]
        ins = self.in_edges(n, "crd")
        io = [e for e in ins if self.crd_name(e["comment"]) == outer]
        ii = [e for e in ins if self.crd_name(e["comment"]) == inner]
        if len(io) + len(ii) != len(ins):
            raise ConvertError(f"{kind} {n}: input comments {[e['comment'] for e in ins]} do not name {outer}/{inner}")
        m.input_outer_crd.CopyFrom(self.S("crd", self.one(io, f"outer ({outer}) crd input of {kind} {n}")["id"]))
        m.input_inner_crd.CopyFrom(self.S("crd", self.one(ii, f"inner ({inner}) crd input of {kind} {n}")["id"]))
        for side, name, field in (("outer", outer, m.output_outer_crd), ("inner", inner, m.output_inner_crd)):
            sid = self.port_stream_named(n, f"{side}-{name}")
            field.CopyFrom(self.S("crd", sid if sid is not None else next(self.ids)))
        extra = [e for e in self.outs[n] if e["comment"] not in (f"outer-{outer}", f"inner-{inner}")]
        if extra:
            raise ConvertError(f"{kind} {n}: unexpected outputs {[e['comment'] for e in extra]}")
        m.outer_crd, m.inner_crd, m.label = outer, inner, f"{kind} {outer},{inner}"
        self.push(op, "CoordDrop" if kind == "crddrop" else "CoordHold")

    def port_stream_named(self, n, comment, typ="crd"):
        """like port_stream, for outputs told apart by their comment (crddrop/crdhold/spacc)"""
        cons = [e for e in self.outs[n] if e["type"] == typ and e["comment"] == comment
                and self.node_type(e["dst"]) != "repsiggen"]
        if not cons:
            return None
        if len(cons) == 1:
            return cons[0]["id"]
        sid = next(self.ids)
        self.add_broadcast(typ, sid, [e["id"] for e in cons], f"auto_bcast_{n}_{comment}")
        return sid

    def c_crddrop(self, n, a):
        self.crd_manager(n, a, "crddrop")

    def c_crdhold(self, n, a):
        self.crd_manager(n, a, "crdhold")

    def c_spaccumulator(self, n, a):
        """Custard: type=spaccumulator,order=1,in0=<inner>,in1=<outer>. Inputs: crd named <inner> (the coordinate of
        the accumulated fiber), crd named <outer> (the reduced coordinate), val. Outputs: crd "out-<inner>", val.
        Comal Spacc1: in_crd_inner, in_crd_outer, in_val -> out_crd_inner, out_val."""
        order = int(a["order"])
        if order != 1:
            raise ConvertError(f"spaccumulator {n}: order {order}; only order 1 seen in Custard output so far")
        inner, outer = a["in0"], a["in1"]
        op = self.ops.Operation(name="spaccumulator", id=int(n) + 1_000_000)
        sp = op.spacc
        sp.order, sp.inner_crd, sp.label = 1, inner, f"spacc1 {inner} over {outer}"
        sp.outer_crds.append(outer)
        ins = self.in_edges(n, "crd")
        ci = [e for e in ins if self.crd_name(e["comment"]) == inner]
        co = [e for e in ins if self.crd_name(e["comment"]) == outer]
        if len(ci) + len(co) != len(ins):
            raise ConvertError(f"spaccumulator {n}: input comments {[e['comment'] for e in ins]} do not name {inner}/{outer}")
        sp.input_inner_crd.CopyFrom(self.S("crd", self.one(ci, f"inner ({inner}) crd of spacc {n}")["id"]))
        sp.input_outer_crds.append(self.S("crd", self.one(co, f"outer ({outer}) crd of spacc {n}")["id"]))
        sp.input_val.CopyFrom(self.S("val", self.one(self.in_edges(n, "val"), f"val input of spacc {n}")["id"]))
        oc = self.port_stream_named(n, f"out-{inner}")
        ov = self.port_stream(n, "val")
        if oc is None or ov is None:
            raise ConvertError(f"spaccumulator {n}: outputs out-{inner}/val not both consumed")
        sp.output_inner_crd.CopyFrom(self.S("crd", oc))
        sp.output_val.CopyFrom(self.S("val", ov))
        self.push(op, "SparseAccumulator")

    def c_fiberwrite(self, n, a):
        if a.get("mode") == "vals":
            op = self.ops.Operation(name="valwrite", id=int(n) + 1_000_000)
            w = op.val_write
            w.input_val.CopyFrom(self.S("val", self.one(self.in_edges(n, "val"), f"val input of fiberwrite {n}")["id"]))
            w.label, w.tensor, w.sink = f"vals_{a['tensor']}", a["tensor"], True
            w.crdsize, w.segsize = a.get("size", ""), ""
            self.push(op, "ValWrite")
        else:
            op = self.ops.Operation(name="fiberwrite", id=int(n) + 1_000_000)
            w = op.fiber_write
            w.input_crd.CopyFrom(self.S("crd", self.one(self.in_edges(n, "crd"), f"crd input of fiberwrite {n}")["id"]))
            w.index, w.tensor, w.mode_val, w.format = a["index"], a["tensor"], a["mode"], a["format"]
            w.segsize, w.crdsize, w.sink = a.get("segsize", ""), a.get("crdsize", ""), True
            w.label = f"FiberWrite {a['index']}: {a['tensor']}{a['mode']}"
            self.push(op, "FiberWrite")


def drop_reduce_before_spacc(nodes, edges):
    """Opt-in repair (--drop-reduce-before-spacc). Custard prints a Reduce between the product and an order-1
    SparseAccumulator for Gustavson orders (e.g. X(i,j)=A(i,k)*B(k,j), reorder(i,k,j)); that Reduce sums the inner (j)
    fiber, so the accumulator receives one value per k instead of one per (k, j) (Comal: "in_icrd and in_val don't match
    types"). SAM's reference wiring of the same kernel (sam/sim/test/final-apps/test_matmul_ikj_FINAL.py) feeds the
    product straight into the accumulator. Removed: every reduce whose only value output goes into a spaccumulator."""
    dropped = []
    for n, a in list(nodes.items()):
        if a["type"] != "reduce":
            continue
        ins = [e for e in edges if e["dst"] == n]
        outs = [e for e in edges if e["src"] == n]
        if len(ins) == 1 and len(outs) == 1 and nodes[outs[0]["dst"]]["type"] == "spaccumulator":
            ins[0]["dst"] = outs[0]["dst"]
            edges.remove(outs[0])
            del nodes[n]
            dropped.append(n)
    return dropped


def convert(gv_path, out_path, pb_dir, name=None, repair_reduce=False):
    text = Path(gv_path).read_text()
    gcomment, nodes, edges = parse_gv(text)
    repairs = {"dropped_reduce_before_spacc": drop_reduce_before_spacc(nodes, edges)} if repair_reduce else {}
    pb = load_pb(pb_dir)
    torder = [part.split("=")[0] for part in (gcomment or "").split(",") if "=" in part]
    c = Converter(nodes, edges, pb, torder)
    rep_pre = {}
    c.rep_ref = rep_pre
    # first pass registers repeat signal sources
    order = sorted(nodes, key=lambda n: int(n) if n.isdigit() else n)
    for n in order:
        if nodes[n]["type"] == "repeat":
            rs = c.one(c.in_edges(n, "repsig"), f"repsig input of repeat {n}")
            while nodes[rs["src"]]["type"] == "broadcast":       # one repsiggen shared by several repeats
                rs = c.one(c.ins[rs["src"]], f"input of repsig broadcast {rs['src']}")
            if nodes[rs["src"]]["type"] != "repsiggen":
                raise ConvertError(f"repeat {n}: repsig from {nodes[rs['src']]['type']}")
            rsg_in = c.one(c.in_edges(rs["src"], "crd"), f"crd input of repsiggen {rs['src']}")
            rep_pre[n] = c.aligned_ref_for(rsg_in, n)
    for n in order:
        c.report["nodes"][nodes[n]["type"]] += 1
        getattr(c, "c_" + nodes[n]["type"], c.unknown)(n, nodes[n])
    comal_pb2, ops_pb2, tort_pb2, _ = pb
    g = comal_pb2.ComalGraph(name=name or Path(gv_path).stem, channel_size=1024)
    g.graph.name = g.name
    for op in c.operators:
        g.graph.operators.append(op)
    # streams whose only consumer in the gv was a dropped repsiggen: give them a sink (Comal needs every channel read).
    # crd -> FiberWrite, val -> ValWrite (tensor "_sink", ignored when outputs are read back); a dangling ref is refused.
    dangling = check_streams(g, sinks=True)
    sinks = []
    for sid, k in sorted(dangling.items()):
        op = g.graph.operators.add(name="sink", id=9_000_000 + sid)
        if k == "CrdStream":
            op.fiber_write.input_crd.id.id = sid
            op.fiber_write.tensor, op.fiber_write.index, op.fiber_write.mode_val = "_sink", "_", "0"
            op.fiber_write.format, op.fiber_write.sink, op.fiber_write.label = "compressed", True, f"sink_{sid}"
        elif k == "ValStream":
            op.val_write.input_val.id.id = sid
            op.val_write.tensor, op.val_write.label, op.val_write.sink = "_sink", f"sink_{sid}", True
        else:
            raise ConvertError(f"stream {sid} ({k}) is written but never read; no sink for that type")
        sinks.append({"stream": sid, "type": k})
    check_streams(g)
    Path(out_path).write_bytes(g.SerializeToString())
    rep = {"gv": str(gv_path), "proto": str(out_path), "formats": gcomment, "edges": len(edges),
           "nodes": dict(c.report["nodes"]), "operators": dict(c.report["operators"]),
           "repeat_rewired": c.report["repeat_rewired"], "sinks": sinks, "repairs": repairs,
           "product_order": c.report["product_order"]}
    return rep


def is_repeated(fd):
    """protobuf >= 6 removed FieldDescriptor.label (the server has 7.x); older versions lack is_repeated"""
    r = getattr(fd, "is_repeated", None)
    if r is not None:
        return r() if callable(r) else bool(r)
    return fd.label == fd.LABEL_REPEATED


def check_streams(g, sinks=None):
    """every stream id has exactly one writer and one reader. A stream is written when any field on its path inside the
    operation is an output field (output*, out_*), else read."""
    writers, readers, kind = defaultdict(int), defaultdict(int), {}

    def walk(msg, is_out):
        for fd, val in msg.ListFields():
            vals = val if is_repeated(fd) else [val]
            out = is_out or fd.name.startswith("output") or fd.name.startswith("out_")
            for v in vals:
                if not hasattr(v, "ListFields"):
                    continue
                if v.DESCRIPTOR.name.endswith("Stream"):
                    (writers if out else readers)[v.id.id] += 1
                    kind[v.id.id] = v.DESCRIPTOR.name
                else:
                    walk(v, out)

    for op in g.graph.operators:
        walk(getattr(op, op.WhichOneof("op")), False)
    bad = {s: (writers[s], readers[s]) for s in set(writers) | set(readers) if (writers[s], readers[s]) != (1, 1)}
    if sinks is not None:
        dangling = {s: kind[s] for s, (w, r) in bad.items() if (w, r) == (1, 0)}
        return dangling
    if bad:
        raise ConvertError(f"stream check failed (id: writers, readers): {dict(sorted(bad.items())[:20])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gv")
    ap.add_argument("out")
    ap.add_argument("--pb-dir", default="/fuseflow-artifact/tortilla-visualizer",
                    help="directory with comal_pb2/ops_pb2/tortilla_pb2/stream_pb2 (FuseFlow's generated bindings)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--drop-reduce-before-spacc", action="store_true",
                    help="opt-in repair of Custard's Gustavson-order graphs (see drop_reduce_before_spacc)")
    a = ap.parse_args()
    try:
        rep = convert(a.gv, a.out, a.pb_dir, a.name, a.drop_reduce_before_spacc)
    except ConvertError as e:
        print(f"CONVERT ERROR: {e}", file=sys.stderr)
        return 2
    print(json.dumps(rep, indent=1))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
