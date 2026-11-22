"""
Shared pieces of the C+S baseline (Custard + Stardust, compiled by Custard, simulated in Comal).

Used by cs_kernel.py. Same logic as the smoke test (cs_smoke.py), kept here so that every C+S run uses one copy.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import scipy.sparse as sp

TACO = "/fuseflow-v3/third_party/custard/build/bin/taco"
HERE = Path(__file__).resolve().parent
FIX = "--drop-reduce-before-spacc"


def parse_formats(comment):
    """Custard graph comment 'X=ss01,A=ss01,B=dd10' -> {T: (levels 'ss', storage mode order (0, 1))}"""
    out = {}
    for part in comment.split(","):
        t, f = part.split("=")
        if f == "none":
            out[t] = ("", ())
            continue
        k = len(f) // 2
        out[t] = (f[:k], tuple(int(x) for x in f[k:]))
    return out


def save(path, arr, ints):
    np.savetxt(path, np.asarray(arr), fmt="%d" if ints else "%.9g")


def write_tensor(d, name, arr, levels, order):
    """SAM/Comal layout of a Custard graph: per stored level l (tensor mode order[l]) tensor_<T>_mode_<order[l]>_seg/_crd
    if compressed, nothing if dense; values in storage order; tensor_<T>_mode_shape = dims in MODE order.
    (Custard's fiberlookup 'mode' is the tensor dimension, Comal opens tensor_<T>_mode_<mode>_*.)"""
    p = Path(d) / f"tensor_{name}"
    shape = arr.shape
    Path(f"{p}_mode_shape").write_text("".join(f"{s}\n" for s in shape))
    if len(shape) == 1:
        v = arr.toarray().ravel() if sp.issparse(arr) else np.asarray(arr)
        if levels == "d":
            save(f"{p}_mode_vals", v.astype(np.float32), False)
        elif levels == "s":
            nz = np.flatnonzero(v)
            save(f"{p}_mode_0_seg", [0, nz.size], True)
            save(f"{p}_mode_0_crd", nz, True)
            save(f"{p}_mode_vals", v[nz].astype(np.float32), False)
        else:
            raise ValueError(levels)
        return
    if len(shape) != 2:
        raise ValueError("2-D or 1-D only")
    M = arr if order == (0, 1) else arr.T
    if levels == "dd":
        dense = M.toarray() if sp.issparse(M) else np.asarray(M)
        save(f"{p}_mode_vals", dense.astype(np.float32).ravel(), False)
        return
    S = sp.csr_matrix(M, dtype=np.float32)
    S.sort_indices()
    S.eliminate_zeros()
    m0, m1 = order
    if levels == "ss":
        rows = np.flatnonzero(np.diff(S.indptr))
        save(f"{p}_mode_{m0}_seg", [0, rows.size], True)
        save(f"{p}_mode_{m0}_crd", rows, True)
        sub = S[rows]
        save(f"{p}_mode_{m1}_seg", sub.indptr, True)
        save(f"{p}_mode_{m1}_crd", sub.indices, True)
        save(f"{p}_mode_vals", sub.data, False)
    elif levels == "ds":
        save(f"{p}_mode_{m1}_seg", S.indptr, True)
        save(f"{p}_mode_{m1}_crd", S.indices, True)
        save(f"{p}_mode_vals", S.data, False)
    else:
        raise ValueError(f"levels {levels} not handled")


def dense_output(fibers, vals, name, levels, order, oidx, shape):
    """rebuild a 2-D 'ss' output of a Custard graph from its writers (FiberWrite per level, ValWrite 'vals_<T>')"""
    if levels != "ss":
        raise ValueError(f"output levels {levels}: only ss handled")
    idx_of_level = [oidx[m] for m in order]
    lv = {}
    for t, index, fmt, seg, crd in fibers:
        if t == name:
            lv[index] = (np.asarray(seg, dtype=np.int64), np.asarray(crd, dtype=np.int64))
    v = [np.asarray(x, dtype=np.float64) for lab, x in vals if lab == f"vals_{name}"]
    if len(v) != 1 or set(lv) != set(idx_of_level):
        raise ValueError(f"writers: fibers {sorted(lv)} vals {[lab for lab, _ in vals]}")
    v = v[0]
    (s0, c0), (s1, c1) = lv[idx_of_level[0]], lv[idx_of_level[1]]
    if v.size != c1.size:
        raise ValueError(f"{v.size} values for {c1.size} level-1 coordinates")
    out = np.zeros(shape if order == (0, 1) else shape[::-1])
    for p_, r in enumerate(c0):
        a, b = s1[p_], s1[p_ + 1]
        out[r, c1[a:b]] = v[a:b]
    return out if order == (0, 1) else out.T


def dense_output_blocked(fibers, vals, name, order, oidx, n, cshape):
    """rebuild a 2-D 'ss' output of a true-block run (n x n tiles; level coordinates are tile coordinates; each stored
    tile = n*n values row-major) -> dense array of shape cshape * n (caller slices off the padding)"""
    if tuple(order) != (0, 1):
        raise ValueError("blocked output rebuild: natural storage order only")
    lv = {t_index: (np.asarray(seg, dtype=np.int64), np.asarray(crd, dtype=np.int64))
          for t, t_index, fmt, seg, crd in fibers if t == name}
    v = [np.asarray(x, dtype=np.float64) for lab, x in vals if lab == f"vals_{name}"]
    if len(v) != 1 or set(lv) != {oidx[0], oidx[1]}:
        raise ValueError(f"writers: fibers {sorted(lv)} vals {[lab for lab, _ in vals]}")
    v = v[0]
    (s0, c0), (s1, c1) = lv[oidx[0]], lv[oidx[1]]
    if v.size != c1.size * n * n:
        raise ValueError(f"{v.size} values for {c1.size} stored tiles of {n}x{n}")
    tiles = v.reshape(-1, n, n)
    out = np.zeros((cshape[0] * n, cshape[1] * n))
    for p_, r in enumerate(c0):
        for q in range(s1[p_], s1[p_ + 1]):
            c = c1[q]
            out[r * n:(r + 1) * n, c * n:(c + 1) * n] = tiles[q]
    return out


def compare(out, ref):
    err = np.abs(out - ref)
    bad = err > 1e-4 + 1e-4 * np.abs(ref)
    return {"elements": int(ref.size), "failing": int(bad.sum()), "max_abs_err": float(err.max()) if err.size else 0.0,
            "ref_max_abs": float(np.abs(ref).max()) if ref.size else 0.0}


def _comal_child(proto, data, out, entry, log):
    import pickle
    os.environ["COMAL_ENABLE_HBM"] = "0"
    import comal
    if log:                                             # Comal's prints (counts, cycles) into a file
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    try:
        res = ("ok", getattr(comal, entry)(proto, data))
    except BaseException as e:                          # Comal panics surface as PanicError
        res = ("error", f"{type(e).__name__}: {e}")
    with open(out, "wb") as f:
        pickle.dump(res, f)


def comal_isolated(proto, data, out, timeout, entry="run_graph_outputs", log=None):
    """comal.run_graph_outputs (scalar mode, HBM off as in H) in a forked child: a panic inside Comal can leave its other
    contexts blocked forever, so the parent waits at most `timeout` seconds and kills the child.
    entry: run_graph_outputs (scalar) or run_graph_true_block{16,32,64}_outputs (true-block mode).
    Returns ("ok", (passed, cycles, fibers, vals)) | ("error", msg) | ("hang", None)."""
    import multiprocessing as mp
    import pickle
    out = Path(out)
    if out.exists():
        out.unlink()
    p = mp.get_context("fork").Process(target=_comal_child, args=(str(proto), str(data), str(out), entry,
                                                                  str(log) if log else None))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return ("hang", None)
    if not out.exists():
        return ("error", f"simulation process exited with code {p.exitcode} and no result")
    with open(out, "rb") as f:
        res = pickle.load(f)
    out.unlink()
    return res


def custard_graph(expr, fmts, order, gv, conv_flags, pb_dir, python):
    """Custard (taco --print-sam-graph) then cs_gv2proto. Returns (proto path, converter report) or raises RuntimeError."""
    gv = Path(gv)
    sched = [f"-s=reorder({order})"] if order else []
    p = subprocess.run([TACO, expr, *[f"-f={f}" for f in fmts], *sched, f"--print-sam-graph={gv}"],
                       capture_output=True, text=True)
    if p.returncode != 0 or not gv.exists():
        raise RuntimeError(f"custard: {p.stderr[-1500:]}")
    proto = gv.with_suffix(".bin")
    rep = gv.with_suffix(".convert.json")
    p = subprocess.run([python, str(HERE / "cs_gv2proto.py"), str(gv), str(proto), "--pb-dir", pb_dir,
                        "--report", str(rep), *conv_flags], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"convert: {(p.stderr or p.stdout)[-1500:]}")
    return proto, json.loads(rep.read_text())
