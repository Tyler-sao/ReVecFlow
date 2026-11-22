#!/usr/bin/env python3
"""
New Figure 12 baselines: run ONE kernel exactly as Figure 12 runs it (run_end_to_end.py, same arguments as
run_figure12_benchmarks.run_single_mlir), capture the output tensor from the same simulation, and check it
against an independent reference computed from the very tensor files the kernel was simulated with.

    cd /fuseflow-artifact && /fuseflow-v3/venv/bin/python scripts/newmodels_kernel.py \
        --mlir sgc_unfused/sgc_prop1.mlir --dataset KarateClub KarateClub --out results/figure12_newmodels/runs/X

- Must run with the /fuseflow-v3/venv Python: its Comal has run_graph_outputs (writer-output patch 0003). Inside
  this process comal.run_graph is replaced by a wrapper around run_graph_outputs, so run_end_to_end.py's own
  call returns the usual (passed, cycles) and the writers' arrays are kept: one simulation gives both the
  cycles and the checked values. Nothing in run_end_to_end.py is changed.
- Settings as Figure 12's GNN runs: -sp 0.9 (ignored by the GNN generator classes), -par 1, --useGen,
  stream level 1, block mode 1 (scalar), HBM off (COMAL_ENABLE_HBM=0).
- After the run the data directory (data/<model dir>/<kernel>_b64) is moved into OUT/data, so later runs of the
  same kernel on another dataset cannot overwrite it and the reference can be recomputed.
- Reference: f32, sums in index order (SciPy CSR products), plus f64 for scale; tolerance atol = rtol = 1e-4,
  fixed in advance. OUT/result.json: status ok | sim-failed | malformed-output | mismatch | error.
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from itertools import permutations
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ARTIFACT = Path("/fuseflow-artifact")
MODELS = ARTIFACT / "fuseflow-compiler" / "tests" / "models"
BUILD = Path("/fuseflow-v3/toolchain/compiler/build")     # compiler with the locate fix (0016); decision 2026-10-13
ATOL = RTOL = 1e-4
SETTINGS = {"sp": 0.9, "par": 1, "streamlevel": 1, "blockmode": 1, "hbm": False, "block": 64,
            "simulator": "FuseFlow with the locate fix: compiler 0016, Comal 0015 (+0003 output capture)"}


# ------------------------------------------------------------------ reading tensors
def _ints(path):
    return np.fromfile(path, sep="\n", dtype=np.int64)


def read_tensor(prefix, t, shape):
    """Logical array of a header tensor t (formats per stored level, mode order), from its files. A 2-D tensor with
    both levels compressed (adjacency, weights) is returned as a SciPy CSR matrix, so that large graphs fit."""
    order = t["mode_order"]
    store_shape = [shape[d] for d in order]
    vals = np.fromfile(f"{prefix}_mode_vals", sep="\n", dtype=np.float64).astype(np.float32)
    coords, parents = [], 1
    for lvl, fmt in enumerate(t["format"]):
        if fmt == "compressed":
            seg = _ints(f"{prefix}_mode_{lvl}_seg")
            crd = _ints(f"{prefix}_mode_{lvl}_crd")
        elif fmt == "dense":
            seg = np.arange(parents + 1) * store_shape[lvl]
            crd = np.tile(np.arange(store_shape[lvl]), parents)
        else:
            raise ValueError(f"{prefix}: level format {fmt}")
        if len(seg) != parents + 1:
            raise ValueError(f"{prefix}: level {lvl} seg has {len(seg)} entries for {parents} parents")
        owner = np.repeat(np.arange(parents), np.diff(seg))
        coords = [c[owner] for c in coords] + [crd]
        parents = len(crd)
    if parents != len(vals):
        raise ValueError(f"{prefix}: {len(vals)} values for {parents} coordinates")
    idx = [None] * len(shape)
    for lvl, d in enumerate(order):
        idx[d] = coords[lvl]
    if len(shape) == 2 and t["format"] == ["compressed", "compressed"]:
        m = sp.csr_matrix((vals, (idx[0], idx[1])), shape=tuple(shape))
        m.sort_indices()
        return m
    out = np.zeros(shape, dtype=np.float32)
    out[tuple(idx)] = vals
    return out


def output_dense(fibers, vals, shape):
    """Writer arrays -> dense output. Level order: the unique chain of seg/crd lengths. Dimensions: outermost
    stored level = dimension 0 unless the coordinates do not fit (then the other assignment); a square output
    with both assignments possible keeps the row-major one."""
    if len(vals) != 1:
        raise ValueError(f"{len(vals)} value writers")
    v = np.asarray(vals[0][1], dtype=np.float64)
    chains = [o for o in permutations(fibers) if len(o[0][3]) == 2 and len(o[-1][4]) == len(v)
              and all(len(o[i][3]) == len(o[i - 1][4]) + 1 for i in range(1, len(o)))]
    if len(chains) != 1:
        raise ValueError(f"{len(chains)} level orders for writers {[(f[1], f[2], len(f[3]), len(f[4])) for f in fibers]}")
    order = chains[0]
    coords, parents = [], 1
    for f in order:
        seg, crd = np.asarray(f[3], dtype=np.int64), np.asarray(f[4], dtype=np.int64)
        owner = np.repeat(np.arange(parents), np.diff(seg))
        coords = [c[owner] for c in coords] + [crd]
        parents = len(crd)
    for dims in permutations(range(len(shape))):
        if all(len(c) == 0 or c.max() < shape[d] for c, d in zip(coords, dims)):
            out = np.zeros(shape)
            idx = [None] * len(shape)
            for c, d in zip(coords, dims):
                idx[d] = c
            out[tuple(idx)] = v
            return out, [f[1] for f in order], list(dims)
    raise ValueError(f"output coordinates do not fit {shape}")


# ------------------------------------------------------------------ references (f32 index order and f64)
def spmm(a, b, dtype):
    """a sparse/dense @ b dense, sums over k in index order (CSR row products)."""
    return np.asarray(sp.csr_matrix(a, dtype=dtype) @ np.asarray(b, dtype=dtype))


def dense_sparse(p, w, dtype):
    """p dense @ w (sparse weight): column j of the result sums over k in index order."""
    return np.asarray((sp.csr_matrix(w, dtype=dtype).T.tocsr() @ np.asarray(p, dtype=dtype).T).T)


def softmax(z):
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)


BN_EPS = np.float32(9.99999974e-6)
KEEP, ALPHA = 0.9, 0.1


def relu(z):
    return np.where(z > 0, z, 0).astype(z.dtype)


def batchnorm(z, mu, invstd, gamma, beta, dtype):
    """As the MLIR: (z - mu) * s * gamma + beta, s = 1/sqrt(var + eps) precomputed offline (the kernel's input)."""
    return ((z - mu.astype(dtype)[None, :]) * invstd.astype(dtype)[None, :]) * gamma.astype(dtype)[None, :]         + beta.astype(dtype)[None, :]


def gin_layer(h, a, p, last, dtype, self_term=True):
    """p: [W1, b1, mu, s, gamma, beta, W2, b2, (mu, s, gamma, beta)] as the kernel's arguments."""
    s = spmm(a, h, dtype) + (h.astype(dtype) if self_term else 0)
    z = dense_sparse(s, p[0], dtype) + p[1].astype(dtype)[None, :]
    z = relu(batchnorm(z, *p[2:6], dtype))
    z = dense_sparse(z, p[6], dtype) + p[7].astype(dtype)[None, :]
    if not last:
        z = relu(batchnorm(z, *p[8:12], dtype))
    return z


def appnp_step(z, a, h, dtype):
    return dtype(KEEP) * spmm(a, z, dtype) + dtype(ALPHA) * h.astype(dtype)


def appnp_mlp(x, w1, w2, dtype):
    return dense_sparse(relu(dense_sparse(x, w1, dtype)), w2, dtype)


def reference_gin_appnp(kernel, ins, dtype):
    """GIN and APPNP kernels (all parameters are arguments, in the order emit_gin_appnp.py wrote them)."""
    p = lambda lo, hi: [ins[i] for i in range(lo, hi)]
    k = kernel[:-len("_logits")] if kernel.endswith("_logits") else kernel
    sm = (lambda z: z) if kernel.endswith("_logits") else softmax
    if k == "gin_sparse":
        h1 = gin_layer(ins[0], ins[1], p(2, 14), False, dtype)
        return sm(gin_layer(h1, ins[1], p(14, 22), True, dtype))
    if k == "one_layer_gin":
        return gin_layer(ins[0], ins[1], p(2, 14), False, dtype)
    if k in ("gin_diag_l2_noself", "gin_diag_l2_loops"):   # loops: the A + I file is the kernel's adjacency
        return gin_layer(ins[0], ins[1], p(2, 10), True, dtype, self_term=False)
    if k == "gin_diag_l2_hidden":
        return gin_layer(ins[0], ins[1], p(2, 10), True, dtype)
    if k.startswith("appnp_diag_mlp_"):
        h = appnp_mlp(ins[0], ins[2], ins[3], dtype)
        z = h
        for _ in range(int(k.split("_")[-1][0])):
            z = appnp_step(z, ins[1], h, dtype)
        return z
    if k == "appnp_diag_2steps":
        return appnp_step(appnp_step(ins[0], ins[1], ins[2], dtype), ins[1], ins[2], dtype)
    if k == "one_layer_gin2":
        return sm(gin_layer(ins[0], ins[1], p(2, 10), True, dtype))
    if k.startswith("gin_adj_h") or k == "appnp_prop":
        return spmm(ins[0], ins[1], dtype)
    if k.startswith("gin_selfadd"):
        return ins[0].astype(dtype) + ins[1].astype(dtype)
    if k.endswith("_mul") and (k.startswith("gin_mlp") or k.startswith("appnp_lin")):
        return dense_sparse(ins[0], ins[1], dtype)
    if k.endswith("_bias") and k.startswith("gin_mlp"):
        return ins[0].astype(dtype) + ins[1].astype(dtype)[None, :]
    if k.endswith("_bn1") or k == "gin_bn_out1":
        return batchnorm(ins[0].astype(dtype), *p(1, 5), dtype)
    if "relu" in k:
        return relu(ins[0].astype(dtype))
    if k in ("gin_softmax", "appnp_softmax"):
        return softmax(ins[0].astype(dtype))
    if k == "appnp_sparse":
        h = appnp_mlp(ins[0], ins[2], ins[3], dtype)
        z = h
        for _ in range(10):
            z = appnp_step(z, ins[1], h, dtype)
        return sm(z)
    if k == "appnp_mlp":
        return appnp_mlp(ins[0], ins[1], ins[2], dtype)
    if k == "appnp_step":
        return appnp_step(ins[0], ins[1], ins[2], dtype)
    if k == "appnp_step_last":
        return sm(appnp_step(ins[0], ins[1], ins[2], dtype))
    if k == "appnp_mix":
        return dtype(KEEP) * ins[0].astype(dtype) + dtype(ALPHA) * ins[1].astype(dtype)
    raise ValueError(f"no reference for {kernel}")


class NoReference(ValueError):
    pass


class _Done(Exception):
    """End of a paired / capture-only check (no reference to compare with)."""


def csr_source(kernel):
    """A CSR layer copy <k>_csr computes exactly what its source kernel <k> computes (make_csr_kernels.py)."""
    repair = {"gin_repair_l1": "one_layer_gin", "gin_repair_l2": "one_layer_gin2", "gin_repair_l2_logits": "one_layer_gin2_logits"}
    if kernel in repair:                          # identical copies, A's empty rows padded (GIN MAG repair)
        return repair[kernel]
    if not kernel.endswith("_csr"):
        return kernel
    base = kernel[:-len("_csr")]
    return {"one_layer_graphsage": "sage_repair_l1"}.get(base, base)


def reference(kernel, ins, w, b, dtype):
    """ins: {arg number: array}; w: 2-D constants; b: 1-D constants (header order). A kernel named <k>_logits is
    the check-only copy of <k> that returns the values before the softmax."""
    kernel = csr_source(kernel)
    if kernel == "one_layer_gcn" or kernel.endswith("_orig"):
        raise NoReference(f"no independent reference for {kernel}; checked against its paired run (--compare-with)")
    if kernel.startswith(("gin", "one_layer_gin", "appnp")):
        return reference_gin_appnp(kernel, ins, dtype)
    if kernel.startswith("sage_repair"):
        # FuseFlow's one_layer_graphsage(2): relu(X.W_self + b_self + (A.X).W_neigh + b_neigh), in the MLIR's order;
        # W_self = argument 3, b_neigh = argument 2, W_neigh = the 2-D constant, b_self = the 1-D constant
        if len(w) != 1 or len(b) != 1:
            raise ValueError(f"{kernel}: expected one 2-D and one 1-D constant, got {len(w)} and {len(b)}")
        x = ins[0]
        s_ = dense_sparse(x, ins[3], dtype) + b[0].astype(dtype)[None, :]
        n_ = dense_sparse(spmm(ins[1], x, dtype), w[0], dtype) + ins[2].astype(dtype)[None, :]
        return relu(s_ + n_)
    # SGC partially fused plans (all parameters are arguments)
    if kernel == "sgc_prop_classify":
        return softmax(dense_sparse(spmm(ins[0], ins[1], dtype), ins[2], dtype) + ins[3].astype(dtype)[None, :])
    if kernel == "sgc_prop_linear":
        return dense_sparse(spmm(ins[0], ins[1], dtype), ins[2], dtype) + ins[3].astype(dtype)[None, :]
    if kernel == "sgc_prop_mul":
        return dense_sparse(spmm(ins[0], ins[1], dtype), ins[2], dtype)
    if kernel == "sgc_bias_softmax":
        return softmax(ins[0].astype(dtype) + ins[1].astype(dtype)[None, :])
    if kernel == "sgc_linear":
        return dense_sparse(ins[0], ins[1], dtype) + ins[2].astype(dtype)[None, :]
    if kernel.endswith("_logits"):
        base = kernel[:-len("_logits")]
        if base == "sgc_classify":
            return dense_sparse(ins[0], w[0], dtype) + b[0].astype(dtype)[None, :]
        if base == "sgc_sparse":
            p = spmm(ins[1], spmm(ins[1], ins[0], dtype), dtype)
            return dense_sparse(p, w[0], dtype) + b[0].astype(dtype)[None, :]
        raise ValueError(f"no reference for {kernel}")
    if kernel in ("sgc_prop1", "sgc_prop2"):
        return spmm(ins[0], ins[1], dtype)
    if kernel == "sgc_propagate":
        return spmm(ins[1], spmm(ins[1], ins[0], dtype), dtype)
    if kernel == "sgc_linear_mul":
        return dense_sparse(ins[0], ins[1], dtype)
    if kernel == "sgc_linear_bias":
        return (ins[0].astype(dtype) + ins[1].astype(dtype)[None, :])
    if kernel == "sgc_softmax":
        return softmax(ins[0].astype(dtype))
    if kernel == "sgc_classify":
        return softmax(dense_sparse(ins[0], w[0], dtype) + b[0].astype(dtype)[None, :])
    if kernel == "sgc_sparse":
        p = spmm(ins[1], spmm(ins[1], ins[0], dtype), dtype)
        return softmax(dense_sparse(p, w[0], dtype) + b[0].astype(dtype)[None, :])
    raise ValueError(f"no reference for {kernel}")


def compare(out, ref):
    err = np.abs(out - ref)
    bad = err > ATOL + RTOL * np.abs(ref)
    rel = err / np.maximum(np.abs(ref), ATOL)
    return {"elements": int(ref.size), "failing": int(bad.sum()), "max_abs_err": float(err.max()),
            "max_rel_err": float(rel.max()), "ref_max_abs": float(np.abs(ref).max()),
            "ref_nonzeros": int(np.count_nonzero(ref))}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", required=True, help="path below tests/models, e.g. sgc_unfused/sgc_prop1.mlir")
    ap.add_argument("--dataset", nargs=2, required=True, metavar=("TYPE", "NAME"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", default=str(BUILD), help="compiler build directory (default: the locate-fixed copy)")
    ap.add_argument("--capture-only", action="store_true",
                    help="no reference: keep the output for a paired check (status ok, recorded as unchecked)")
    ap.add_argument("--compare-with", default=None,
                    help="run directory of a kernel computing the same function on the same values; its output.npz "
                         "is the check (used where no independent reference exists: GCN layer 1)")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    mlir = MODELS / args.mlir
    kernel = mlir.stem
    result = {"kernel": kernel, "mlir": str(mlir), "dataset": args.dataset, "settings": SETTINGS,
              "tolerance": {"atol": ATOL, "rtol": RTOL}, "status": "running", "python": sys.executable}
    write = lambda: (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    write()
    os.environ.setdefault("DATA_PATH", "/tmp/data")
    os.environ["COMAL_ENABLE_HBM"] = "0"
    os.environ["TORCH_WEIGHTS_ONLY_LOAD"] = "0"
    os.chdir(ARTIFACT)
    sys.path.insert(0, str(ARTIFACT / "scripts"))
    start = time.time()
    try:
        import comal
        if not hasattr(comal, "run_graph_outputs"):
            raise SystemExit(f"{comal.__file__}: no run_graph_outputs; run with /fuseflow-v3/venv/bin/python")
        captured = {}

        def run_graph_capture(op, data):
            passed, cycles, fibers, vals = comal.run_graph_outputs(op, data)
            captured.update(passed=passed, cycles=cycles, fibers=fibers, vals=vals, op=op, data=data)
            return passed, cycles

        comal.run_graph = run_graph_capture
        import run_end_to_end as r
        from utils import get_tensors_properties
        result["build"] = args.build
        argv = ["--infile", str(mlir), "--build", args.build, "-sp", str(SETTINGS["sp"]), "-par", str(SETTINGS["par"]),
                "--useGen", "-inDataset", args.dataset[0], "-inData", args.dataset[1]]
        rr = r.run_fuseflow_all(r.get_argparser().parse_args(argv))
        result["wall_seconds"] = round(time.time() - start, 1)
        sim = rr.get("comal") if isinstance(rr, dict) else None
        result["comal"] = list(sim) if sim else None
        if not captured or not captured["passed"] or not captured["cycles"]:
            result.update(status="sim-failed", reason=f"Comal returned {sim}")
            write()
            return 1
        result["cycles"] = captured["cycles"]
        data_dir = Path(captured["data"])
        sam = mlir.parent / f"{kernel}_sam.mlir"
        header = get_tensors_properties(sam)
        shutil.copy(sam, out / "sam.mlir")
        shutil.move(str(data_dir), str(out / "data"))
        data_dir = out / "data"
        info = json.loads((data_dir / "preprocess.json").read_text())
        result["preprocess_seconds"] = info.get("preprocess_seconds")
        result["preprocess"] = {k: v for k, v in info.items() if k not in ("tensors", "sizes")}
        place = {1767: "N", 50: "F", 121: "C", 64: "H"}
        if "sizes" in info:
            lshape = lambda t: [info["sizes"][place[int(s)]] for s in t["shape"]]
        else:   # FuseFlow's own generator classes (GraphSAGE repair): shapes from the data files
            lshape = lambda t: [int(v) for v in (data_dir / f"tensor_{[n for n, u in header.items() if u is t][0]}_mode_shape").read_text().split()]
        ins, w, b = {}, [], []
        for name, t in header.items():
            if not (data_dir / f"tensor_{name}_mode_vals").exists():
                continue                                   # e.g. the %argout buffer, never read
            a = read_tensor(str(data_dir / f"tensor_{name}"), t, lshape(t))
            if isinstance(t["Arg"], int):
                ins.setdefault(t["Arg"], a)
            elif len(t["mode_order"]) == 2:
                w.append(a)
            else:
                b.append(a)
        result["inputs"] = {str(k): list(v.shape) for k, v in ins.items()} | {"weights": [list(x.shape) for x in w],
                                                                             "biases": [list(x.shape) for x in b]}
        try:
            ref32 = reference(kernel, ins, w, b, np.float32).astype(np.float64)
            ref64 = reference(kernel, ins, w, b, np.float64)
        except ValueError as e:     # NoReference, or a source kernel without a reference ("no reference for ...")
            if not (isinstance(e, NoReference) or str(e).startswith("no reference")) or not (args.capture_only or args.compare_with):
                raise
            # output shape: rows = the adjacency's, columns from the writers (same rule in both paired runs)
            n = [v.shape[0] for v in ins.values() if v.ndim == 2 and v.shape[0] == v.shape[1]]
            # columns from the inner level only: the outermost writer (one segment, seg of length 2) holds the row
            # coordinates (up to N), which made an N x N output (CSR run, Collab: 415 GiB) - fixed 2026-10-14
            inner = [f for f in captured["fibers"] if len(f[3]) != 2] or captured["fibers"]
            cols = max(int(np.max(f[4])) + 1 for f in inner if len(f[4])) if inner else 0
            got, levels, dims = output_dense(captured["fibers"], captured["vals"], [n[0], cols] if n else [0, cols])
            np.savez_compressed(out / "output.npz", out=got.astype(np.float32))
            result["output"] = {"shape": list(got.shape), "writer_levels": levels, "dims": dims}
            result["reference"] = str(e)
            if args.compare_with:
                other = np.load(Path(args.compare_with) / "output.npz")["out"].astype(np.float64)
                if other.shape != got.shape:
                    result.update(status="mismatch", reason=f"output shape {got.shape} vs paired {other.shape}")
                else:
                    result["vs_paired"] = compare(got, other) | {"paired_run": str(args.compare_with)}
                    result["status"] = "ok" if result["vs_paired"]["failing"] == 0 else "mismatch"
            else:
                result.update(status="ok", checked=False)
            raise _Done()
        got, levels, dims = output_dense(captured["fibers"], captured["vals"], list(ref32.shape))
        np.savez_compressed(out / "output.npz", out=got.astype(np.float32), ref32=ref32.astype(np.float32))
        result["output"] = {"shape": list(ref32.shape), "writer_levels": levels, "dims": dims}
        result["vs_ref_f32"] = compare(got, ref32)
        result["ref_f32_vs_f64"] = compare(ref32, ref64)
        result["status"] = "ok" if result["vs_ref_f32"]["failing"] == 0 else "mismatch"
        if result["status"] == "mismatch":
            # NumPy/SciPy in this container gave different results on identical inputs (APPNP step, Cora, 2026-10-13):
            # recompute both references twice; the kernel passes only if a recomputed f32 reference matches the output
            # with the unchanged tolerance AND agrees with its recomputed f64 reference. Every attempt is recorded.
            result["reference_attempts"] = [{"vs_ref_f32": result["vs_ref_f32"], "ref_f32_vs_f64": result["ref_f32_vs_f64"]}]
            for _ in range(2):
                r32 = reference(kernel, ins, w, b, np.float32).astype(np.float64)
                r64 = reference(kernel, ins, w, b, np.float64)
                att = {"vs_ref_f32": compare(got, r32), "ref_f32_vs_f64": compare(r32, r64)}
                result["reference_attempts"].append(att)
                if att["vs_ref_f32"]["failing"] == 0 and att["ref_f32_vs_f64"]["failing"] == 0:
                    result.update(status="ok", vs_ref_f32=att["vs_ref_f32"], ref_f32_vs_f64=att["ref_f32_vs_f64"],
                                  reference_recomputed=True)
                    np.savez_compressed(out / "output.npz", out=got.astype(np.float32), ref32=r32.astype(np.float32))
                    break
    except _Done:
        pass
    except Exception as e:
        result.update(status="error", reason=f"{type(e).__name__}: {e}"[:600], trace=traceback.format_exc()[-2000:])
    result["wall_seconds_total"] = round(time.time() - start, 1)
    write()
    print(json.dumps({k: result.get(k) for k in ("kernel", "dataset", "status", "cycles", "vs_ref_f32", "reason")}))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
