#!/usr/bin/env python3
"""
FuseFlow V3, Step B2: how Comal charges a block token. One program (default GPT-3's unfused Q/K/V projection,
a pure product), the same compiled graph run in scalar mode and in true-block mode (Comal 0020/0021) at the
given block sizes, HBM off, outputs checked.

    cd /fuseflow-v3 && /fuseflow-v3/venv/bin/python /fuseflow-v3/scripts/b2_block_timing.py [--model M.mlir] [--data DIR] \
        [--blocks 16 32 64] [--out /fuseflow-v3/check/b2]

V3 Stage B (2026-10-15): the V3 Step B2 script, run in fuseflow-v3 with Comal 0020-0022 (0022: tile accumulation) and
the V3 compiler; compile/translate/read helpers from v3tools.py (V2's run_candidate.py is not used). Ends with
"STAGE B2 PASS" only if every block size's output is within tolerance of the f64 product.

- The model is compiled once (development sam-opt, stream level 1, parallel factor 1); its header tensors are
  bound by name to the Figure 12 data in --data (FuseFlow's own generated inputs).
- Blocked inputs (input preparation): every input is cut into N x N tiles over its last two dimensions; its
  coordinate shape is divided by N there; a compressed level stores the tiles holding a nonzero (empty fibers
  dropped, as FuseFlow's generator does); each stored tile is written as N*N values, row-major.
- Every run's output is rebuilt from Comal's writers and compared with the f64 product of the inputs
  (atol = rtol = 1e-4) and with the scalar run.
- Reported per run: cycles; the number of multiplications the program performs (scalar: pairs of stored
  values meeting at the same k; blocked: pairs of stored tiles), cycles per multiplication and MACs per cycle.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

V3 = Path("/fuseflow-v3")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3tools import (Failed, compile_candidate, coordinates, get_tensors_properties, read_input,  # noqa: E402,F401
                     reconstruct, translate)

SIM = {"hbm": False, "stream_level": 1, "par_factor": 1}


def comal_call(fn, op_bin, data, log):
    """Runs a Comal entry point with its prints in log; returns (passed, cycles, fibers, vals)."""
    sys.stdout.flush(); sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(log, "w") as f:
        os.dup2(f.fileno(), 1); os.dup2(f.fileno(), 2)
        try:
            return fn(str(op_bin), str(data))
        finally:
            os.dup2(saved[0], 1); os.dup2(saved[1], 2); os.close(saved[0]); os.close(saved[1])


def write_blocked(a, formats, prefix, n):
    """a: dense array; tiles of n x n over the last two dimensions. Returns (coordinate shape, stored tiles)."""
    *lead, r, c = a.shape
    if r % n or c % n:
        raise Failed("binding-failed", f"{prefix}: last two dimensions {r} x {c} not divisible by {n}")
    t = a.reshape(*lead, r // n, n, c // n, n)
    t = np.moveaxis(t, -3, -2)                      # (..., r/n, c/n, n, n)
    cshape = [*lead, r // n, c // n]
    nz = t.any(axis=(-1, -2))
    parents = [()]
    for d, fmt in enumerate(formats):
        seg, crd, nxt = [0], [], []
        for p in parents:
            coords = range(cshape[d]) if fmt == "dense" else np.flatnonzero(
                nz[p] if nz[p].ndim == 1 else nz[p].any(axis=tuple(range(1, nz[p].ndim))))
            crd.extend(int(x) for x in coords)
            seg.append(len(crd))
            nxt.extend(p + (int(x),) for x in coords)
        if fmt == "compressed":
            np.savetxt(f"{prefix}_mode_{d}_seg", seg, fmt="%d")
            np.savetxt(f"{prefix}_mode_{d}_crd", crd, fmt="%d")
        elif fmt != "dense":
            raise Failed("binding-failed", f"{prefix}: level format {fmt} not handled")
        parents = nxt
    vals = np.stack([t[p] for p in parents]).astype(np.float32).reshape(-1)
    np.savetxt(f"{prefix}_mode_vals", vals, fmt="%.9g")
    Path(f"{prefix}_mode_shape").write_text("".join(f"{s}\n" for s in cshape))
    return cshape, len(parents)


def unblock(fibers, vals, out_levels, cshape, n):
    """Writer arrays of a true-block run -> dense logical array."""
    nblocks = len(vals[0][1]) // (n * n)
    if nblocks * n * n != len(vals[0][1]):
        raise Failed("malformed-output", f"{len(vals[0][1])} values is not a multiple of {n * n}")
    ids = [(vals[0][0], list(range(1, nblocks + 1)))]
    idx, *_ = reconstruct({"output": {"levels": out_levels, "shape": cshape}}, fibers, ids)
    blocks = np.asarray(vals[0][1], dtype=np.float64).reshape(nblocks, n, n)
    full = np.zeros((*cshape, n, n))
    pos = np.nonzero(idx)
    full[pos] = blocks[idx[pos].astype(np.int64) - 1]
    full = np.moveaxis(full, -2, -3)                # (..., r/n, n, c/n, n)
    return full.reshape(*cshape[:-2], cshape[-2] * n, cshape[-1] * n)


def writer_names(levels, fibers):
    """The writers name index variables d<n>, the header i<n>: map by the number, refuse anything else."""
    wnames = [f[1] for f in fibers]
    num = lambda v: v.lstrip("abcdefghijklmnopqrstuvwxyz")
    if sorted(wnames) == sorted(levels):
        return levels
    if sorted(num(v) for v in wnames) != sorted(num(v) for v in levels):
        raise SystemExit(f"output writers {wnames} do not correspond to the output indices {levels}")
    mapped = [next(w for w in wnames if num(w) == num(v)) for v in levels]
    print("output index variables as the writers name them:", mapped)
    return mapped


def pairs(px, pw):
    """Multiplications of X[..., k] . W[k, j] over stored entries: sum over k of (X entries in column k) x
    (W entries in row k). px, pw: boolean patterns (scalars or tiles)."""
    return int(px.reshape(-1, px.shape[-1]).sum(axis=0).astype(np.int64) @ pw.sum(axis=1).astype(np.int64))


def compare(out, ref):
    err = np.abs(out - ref)
    bad = err > 1e-4 + 1e-4 * np.abs(ref)
    return {"elements": int(ref.size), "failing": int(bad.sum()), "max_abs_err": float(err.max()),
            "ref_max_abs": float(np.abs(ref).max())}


def main():
    ap = argparse.ArgumentParser()
    m = Path("/fuseflow-artifact/fuseflow-compiler/tests/models/gpt-3_unfused/projection_mul.mlir")
    ap.add_argument("--model", default=str(m))
    ap.add_argument("--data", default="/fuseflow-artifact/data/gpt-3_unfused/projection_mul_b64")
    ap.add_argument("--blocks", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--out", default=str(V3 / "check" / "b2"))
    ap.add_argument("--no-scalar", action="store_true", help="skip the scalar run (blocks are checked against f64)")
    args = ap.parse_args()
    os.environ["COMAL_ENABLE_HBM"] = "0"
    import comal
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    comp = out / "compiled"
    comp.mkdir(exist_ok=True)
    compile_candidate(SIM | {"model": args.model}, comp)
    op_bin = translate(comp)
    header = get_tensors_properties(comp / "sam.mlir")
    data = Path(args.data)
    data_keys = sorted(f.name[len("tensor_"):-len("_mode_shape")] for f in data.glob("tensor_*_mode_shape"))
    print("header:", {k: (h["format"], h["shape"]) for k, h in header.items()})
    print("data:", data_keys)
    outs = [k for k in header if k.startswith("tOut")]
    ins = [k for k in header if not k.startswith("tOut")]
    if len(outs) > 1 or sorted(ins) != sorted(k for k in data_keys if k in header):
        raise SystemExit(f"binding by name failed: header inputs {ins}, outputs {outs}, data {data_keys}")
    arrays = {}
    for k in ins:
        shape = [int(s) for s in (data / f"tensor_{k}_mode_shape").read_text().split()]
        arrays[k] = read_input({"data": str(data / f"tensor_{k}"), "format": header[k]["format"], "shape": shape})
        print(f"input {k}: shape {shape} format {header[k]['format']} nonzeros {np.count_nonzero(arrays[k])}")
    if len(ins) != 2:
        raise SystemExit("this script handles one product of two inputs")
    # the product the program computes: last index of the first operand contracted with the first of the second
    # (projection: X[b,i,k] . W[k,j]); checked against the scalar run below
    kx_, kw_ = sorted(ins, key=lambda k: -arrays[k].ndim)    # X (activation, higher rank), W (2-D weight)
    x, w = arrays[kx_], arrays[kw_]
    if arrays[kw_].ndim != 2 or x.shape[-1] != w.shape[0]:
        raise SystemExit(f"expected X[..., k] . W[k, j]; got {x.shape} and {w.shape}")
    ref = np.einsum("...k,kj->...j", x, w)
    oshape = list(ref.shape)
    if outs:
        levels = header[outs[0]]["vars"]
    else:
        # no output line in the header (0014 emits it only for models with origin names): the product's
        # output indices are X's free indices followed by W's free index
        xv, wv = header[kx_]["vars"], header[kw_]["vars"]
        shared = set(xv) & set(wv)
        levels = [v for v in xv if v not in shared] + [v for v in wv if v not in shared]
    print("output index variables:", levels)
    # scalar run
    sd = out / "scalar_data"
    sd.mkdir(exist_ok=True)
    for k in ins:
        for f in data.glob(f"tensor_{k}_mode_*"):
            dst = sd / f.name
            if not dst.exists():
                os.symlink(f.resolve(), dst)
    if args.no_scalar:
        passed, cycles, fibers, vals = None, None, None, None
    else:
        passed, cycles, fibers, vals = comal_call(comal.run_graph_outputs, op_bin, sd, out / "scalar.log")
    if fibers:
        levels = writer_names(levels, fibers)
    dense = reconstruct({"output": {"levels": levels, "shape": oshape}}, fibers, vals)[0] if fibers else None
    kx, kw = (x != 0), (w != 0)
    scalar_mults = pairs(kx, kw)
    res = {"scalar": {"passed": passed, "cycles": cycles, "multiplications": scalar_mults,
                      "vs_f64": compare(dense, ref) if dense is not None else None}}
    print("scalar", json.dumps(res["scalar"]))
    for n in args.blocks:
        bd = out / f"block{n}_data"
        bd.mkdir(exist_ok=True)
        cshape = {}
        for k in ins:
            cshape[k], stored = write_blocked(arrays[k], header[k]["format"], str(bd / f"tensor_{k}"), n)
            print(f"block {n}: {k} coordinate shape {cshape[k]}, {stored} tiles stored")
        ocshape = [*oshape[:-2], oshape[-2] // n, oshape[-1] // n]
        fn = getattr(comal, f"run_graph_true_block{n}_outputs")
        try:
            passed, cycles, fibers, vals = comal_call(fn, op_bin, bd, out / f"block{n}.log")
            raw = np.asarray(vals[0][1], dtype=np.float64)
            print(f"block{n} diagnostics: writers {[(f[1], f[2], len(f[3]), len(f[4])) for f in fibers]}, "
                  f"{raw.size} values ({raw.size // (n * n)} tiles), nonzero {np.count_nonzero(raw)}, "
                  f"sum {raw.sum():.6g} (reference sum {ref.sum():.6g})")
            levels = writer_names(levels, fibers)
            got = unblock(fibers, vals, levels, ocshape, n)
            np.save(out / f"block{n}_out.npy", got)
            bt = lambda a: np.swapaxes(a.reshape(*a.shape[:-2], a.shape[-2] // n, n, a.shape[-1] // n, n), -1, -3
                                       ).reshape(a.shape)            # transpose every tile in place
            hyp = {"as computed": got, "output tiles transposed": bt(got)}
            alt = {"W tiles transposed": np.einsum("...k,kj->...j", x, bt(w)),
                   "X tiles transposed": np.einsum("...k,kj->...j", bt(x), w),
                   "both transposed": np.einsum("...k,kj->...j", bt(x), bt(w))}
            diag = {h: compare(g, ref)["failing"] for h, g in hyp.items()}
            diag.update({f"got vs product with {h}": compare(got, a)["failing"] for h, a in alt.items()})
            diag.update({f"tiles-transposed got vs product with {h}": compare(bt(got), a)["failing"] for h, a in alt.items()})
            print(f"block{n} hypotheses (failing elements): {json.dumps(diag)}")
            print(f"block{n} first tile corner, got:")
            print(np.array2string(got[0, :4, :4], precision=4))
            print("reference:")
            print(np.array2string(ref[0, :4, :4], precision=4))
            bx = x.reshape(*x.shape[:-2], x.shape[-2] // n, n, x.shape[-1] // n, n).any(axis=(-1, -3))
            bw = w.reshape(w.shape[0] // n, n, w.shape[1] // n, n).any(axis=(-1, -3))
            mults = pairs(bx, bw)
            r = {"passed": passed, "cycles": cycles, "tile_multiplications": mults,
                 "cycles_per_tile_multiplication": round(cycles / mults, 3) if mults else None,
                 "macs_per_cycle": round(mults * n ** 3 / cycles, 2) if cycles else None,
                 "speedup_vs_scalar": round(res["scalar"]["cycles"] / cycles, 3) if cycles and res["scalar"]["cycles"] else None,
                 "vs_f64": compare(got, ref), "vs_scalar": compare(got, dense) if dense is not None else None}
        except Exception as e:                      # recorded per block size; the other sizes still run
            r = {"error": f"{type(e).__name__}: {e}"[:400]}
        res[f"block{n}"] = r
        print(f"block{n}", json.dumps(r))
    (out / "b2_result.json").write_text(json.dumps(res, indent=1))
    ok = all(((res.get(f"block{n}") or {}).get("vs_f64") or {}).get("failing") == 0 for n in args.blocks)
    print("STAGE B2", "PASS" if ok else "FAIL", {f"block{n}": (res[f"block{n}"].get("cycles"),
          (res[f"block{n}"].get("vs_f64") or {}).get("failing"), res[f"block{n}"].get("error")) for n in args.blocks})


if __name__ == "__main__":
    main()
