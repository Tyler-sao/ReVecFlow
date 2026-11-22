"""
C+S smoke test: Custard graph -> cs_gv2proto -> Comal (scalar mode), checked against float64 numpy on small random inputs.

    /fuseflow-v3/venv/bin/python -u /fuseflow-v3/cs/scripts/cs_smoke.py [--cases spmm_ikj spmm_ijk gcn_layer_iklj]

Each case: run Custard (`taco <expr> -f=... -s=reorder(...) --print-sam-graph`), convert the .gv with cs_gv2proto.py, write
the inputs in the SAM/Comal file layout of the formats Custard chose (graph comment, e.g. "B=dd10": dense/dense, stored
mode 1 first), run comal.run_graph_outputs (writer-output patch 0003) with HBM off as in H, rebuild the output tensor
from its writers and compare with float64 (atol = rtol = 1e-4).
Nothing outside /fuseflow-v3/cs/work/smoke_run and /fuseflow-v3/cs/results/smoke is written.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
TACO = "/fuseflow-v3/third_party/custard/build/bin/taco"
WORK = Path("/fuseflow-v3/cs/work/smoke_run")
RES = Path("/fuseflow-v3/cs/results/smoke")

rng = np.random.default_rng(20261025)


def rand_sparse(r, c, dens, empty_rows=3):
    m = sp.random(r, c, density=dens, format="csr", random_state=rng, data_rvs=lambda n: rng.uniform(-1, 1, n))
    m = m.tolil()
    for i in rng.choice(r, empty_rows, replace=False):
        m[i, :] = 0
    m = sp.csr_matrix(m, dtype=np.float32)
    m.eliminate_zeros()
    return m


A = rand_sparse(40, 30, 0.12)
B = rng.uniform(-1, 1, (30, 8)).astype(np.float32)
H = rng.uniform(-1, 1, (30, 6)).astype(np.float32)
W = rng.uniform(-1, 1, (6, 5)).astype(np.float32)
A2 = rand_sparse(40, 40, 0.08)     # square adjacency for chained cases
bj = rng.uniform(-1, 1, 8).astype(np.float32)
W1 = rng.uniform(-1, 1, (6, 8)).astype(np.float32)
Hn = rng.uniform(-1, 1, (40, 6)).astype(np.float32)   # node features of the 40 output rows (GraphSAGE self term)
FIX = ["--drop-reduce-before-spacc"]

CASES = {
    "spmm_ikj": dict(expr="X(i,j)=A(i,k)*B(k,j)", fmts=["X:ss", "A:ss", "B:dd"], order="i,k,j",
                     ins={"A": A, "B": B}, out="X", oidx="ij",
                     ref=lambda: A.astype(np.float64) @ B.astype(np.float64)),
    "spmm_ijk": dict(expr="X(i,j)=A(i,k)*B(k,j)", fmts=["X:ss", "A:ss", "B:dd:1,0"], order="i,j,k",
                     ins={"A": A, "B": B}, out="X", oidx="ij",
                     ref=lambda: A.astype(np.float64) @ B.astype(np.float64)),
    # the same with the opt-in repair: Custard's extra Reduce before the sparse accumulator removed
    "spmm_ikj_fixed": dict(expr="X(i,j)=A(i,k)*B(k,j)", fmts=["X:ss", "A:ss", "B:dd"], order="i,k,j",
                           ins={"A": A, "B": B}, out="X", oidx="ij", conv=["--drop-reduce-before-spacc"],
                           ref=lambda: A.astype(np.float64) @ B.astype(np.float64)),
    # one GCN layer as one expression, inner-product order (reductions k, l innermost: Reduce only, no accumulator)
    "gcn_layer_ijkl": dict(expr="X(i,j)=A(i,k)*H(k,l)*W(l,j)", fmts=["X:ss", "A:ss", "H:dd", "W:dd:1,0"],
                           order="i,j,k,l", ins={"A": A, "H": H, "W": W}, out="X", oidx="ij",
                           ref=lambda: A.astype(np.float64) @ H.astype(np.float64) @ W.astype(np.float64)),
    # dense GEMM (the linear layer X.W of every model), both orders
    "gemm_ijk": dict(expr="Y(i,j)=H(i,k)*W(k,j)", fmts=["Y:ss", "H:dd", "W:dd:1,0"], order="i,j,k",
                     ins={"H": H, "W": W}, out="Y", oidx="ij",
                     ref=lambda: H.astype(np.float64) @ W.astype(np.float64)),
    "gemm_ikj_fixed": dict(expr="Y(i,j)=H(i,k)*W(k,j)", fmts=["Y:ss", "H:dd", "W:dd"], order="i,k,j",
                           ins={"H": H, "W": W}, out="Y", oidx="ij", conv=FIX,
                           ref=lambda: H.astype(np.float64) @ W.astype(np.float64)),
    # epilogue fusion inside one expression (a C+S rewrite candidate): aggregation + bias
    "spmm_bias_ijk": dict(expr="X(i,j)=A(i,k)*B(k,j)+b(j)", fmts=["X:ss", "A:ss", "B:dd:1,0", "b:d"], order="i,j,k",
                          ins={"A": A, "B": B, "b": bj}, out="X", oidx="ij",
                          ref=lambda: A.astype(np.float64) @ B.astype(np.float64) + bj.astype(np.float64)[None, :]),
    "spmm_bias_ikj_fixed": dict(expr="X(i,j)=A(i,k)*B(k,j)+b(j)", fmts=["X:ss", "A:ss", "B:dd", "b:d"], order="i,k,j",
                                ins={"A": A, "B": B, "b": bj}, out="X", oidx="ij", conv=FIX,
                                ref=lambda: A.astype(np.float64) @ B.astype(np.float64) + bj.astype(np.float64)[None, :]),
    # GraphSAGE-style sum of two products with different reduction variables (a C+S rewrite candidate)
    "sage_sum_ikj_fixed": dict(expr="X(i,j)=Hn(i,l)*W1(l,j)+A(i,k)*B(k,j)", fmts=["X:ss", "Hn:dd", "W1:dd", "A:ss", "B:dd"],
                               order="i,k,l,j", ins={"Hn": Hn, "W1": W1, "A": A, "B": B}, out="X", oidx="ij", conv=FIX,
                               ref=lambda: Hn.astype(np.float64) @ W1.astype(np.float64) + A.astype(np.float64) @ B.astype(np.float64)),
    # the same two expressions in TACO's default order (an expression with a sum of terms is not a perfect loop nest,
    # so reorder() cannot apply; the default nest is i, j outermost with each term's reduction inside -> inner product,
    # so the dense right operands are stored column-major)
    "spmm_bias_default": dict(expr="X(i,j)=A(i,k)*B(k,j)+b(j)", fmts=["X:ss", "A:ss", "B:dd:1,0", "b:d"], order=None,
                              ins={"A": A, "B": B, "b": bj}, out="X", oidx="ij",
                              ref=lambda: A.astype(np.float64) @ B.astype(np.float64) + bj.astype(np.float64)[None, :]),
    "sage_sum_default": dict(expr="X(i,j)=Hn(i,l)*W1(l,j)+A(i,k)*B(k,j)", fmts=["X:ss", "Hn:dd", "W1:dd:1,0", "A:ss", "B:dd:1,0"],
                             order=None, ins={"Hn": Hn, "W1": W1, "A": A, "B": B}, out="X", oidx="ij",
                             ref=lambda: Hn.astype(np.float64) @ W1.astype(np.float64) + A.astype(np.float64) @ B.astype(np.float64)),
    # the C+S rewrite of one GCN layer (aggregation and linear in one expression), Custard's full fusion
    "gcn_layer_iklj": dict(expr="X(i,j)=A(i,k)*H(k,l)*W(l,j)", fmts=["X:ss", "A:ss", "H:dd", "W:dd"], order="i,k,l,j",
                           ins={"A": A, "H": H, "W": W}, out="X", oidx="ij",
                           ref=lambda: A.astype(np.float64) @ H.astype(np.float64) @ W.astype(np.float64)),
}


def parse_formats(comment):
    """'X=ss01,A=ss01,B=dd10' -> {T: (levels 'ss', storage mode order (0, 1))}"""
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
    """SAM/Comal layout: per stored level l (tensor mode order[l]) tensor_<T>_mode_<order[l]>_seg/_crd if compressed,
    nothing if dense; values in storage order; tensor_<T>_mode_shape = dims in MODE order."""
    p = d / f"tensor_{name}"
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
    """rebuild a 2-D 'ss' output from its writers (FiberWrite per level, ValWrite 'vals_<T>')"""
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


def compare(out, ref):
    err = np.abs(out - ref)
    bad = err > 1e-4 + 1e-4 * np.abs(ref)
    return {"elements": int(ref.size), "failing": int(bad.sum()), "max_abs_err": float(err.max()),
            "ref_max_abs": float(np.abs(ref).max())}


TIMEOUT = 120


def _comal_child(proto, data, out):
    import pickle
    os.environ["COMAL_ENABLE_HBM"] = "0"
    import comal
    try:
        res = ("ok", comal.run_graph_outputs(proto, data))
    except BaseException as e:                          # Comal panics surface as PanicError
        res = ("error", f"{type(e).__name__}: {e}")
    with open(out, "wb") as f:
        pickle.dump(res, f)


def comal_isolated(proto, data, out, timeout):
    """run_graph_outputs in a forked child: a panic inside Comal can leave its other contexts blocked forever (the
    process never returns), so the parent waits at most `timeout` seconds and kills the child."""
    import multiprocessing as mp
    import pickle
    out = Path(out)
    if out.exists():
        out.unlink()
    p = mp.get_context("fork").Process(target=_comal_child, args=(proto, data, str(out)))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return ("hang", None)
    if not out.exists():
        return ("error", f"simulation process exited with code {p.exitcode} and no result")
    with open(out, "rb") as f:
        return pickle.load(f)


def run_case(name, c, pb_dir, conv_python):
    d = WORK / name
    if d.exists():
        subprocess.run(["rm", "-rf", str(d)], check=True)
    (d / "data").mkdir(parents=True)
    r = {"case": name, "expr": c["expr"], "formats": c["fmts"], "order": c["order"]}
    gv = d / f"{name}.gv"
    sched = [f"-s=reorder({c['order']})"] if c["order"] else []     # None: Custard/TACO's default loop order
    cmd = [TACO, c["expr"], *[f"-f={f}" for f in c["fmts"]], *sched, f"--print-sam-graph={gv}"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not gv.exists():
        r.update(status="custard_error", stderr=p.stderr[-2000:])
        return r
    proto = d / f"{name}.bin"
    p = subprocess.run([conv_python, str(HERE / "cs_gv2proto.py"), str(gv), str(proto), "--pb-dir", pb_dir,
                        "--report", str(d / "convert.json"), *c.get("conv", [])], capture_output=True, text=True)
    if p.returncode != 0:
        r.update(status="convert_error", stderr=(p.stderr or p.stdout)[-2000:])
        return r
    conv = json.loads((d / "convert.json").read_text())
    r["convert"] = {k: conv[k] for k in ("formats", "operators", "repeat_rewired", "sinks", "repairs")}
    fm = parse_formats(conv["formats"])
    for t, arr in c["ins"].items():
        write_tensor(d / "data", t, arr, *fm[t])
    t0 = time.time()
    res = comal_isolated(str(proto), str(d / "data"), d / "comal_out.pkl", TIMEOUT)
    if res[0] == "hang":
        r.update(status="comal_hang", error=f"no result after {TIMEOUT} s (a panicked context leaves the others "
                                            f"waiting forever); the simulation process was killed")
        return r
    if res[0] == "error":
        r.update(status="comal_error", error=res[1][-2000:])
        return r
    passed, cycles, fibers, vals = res[1]
    r.update(passed=bool(passed), cycles=int(cycles), sim_s=round(time.time() - t0, 3))
    if not passed:                                      # a context panicked (Comal prints it); outputs are partial
        r.update(status="comal_failed", error="Comal reported passed=false (see the panic printed above)")
        return r
    ref = c["ref"]()
    try:
        got = dense_output(fibers, vals, c["out"], *fm[c["out"]], c["oidx"], ref.shape)
    except (ValueError, IndexError) as e:
        r.update(status="output_error", error=f"{type(e).__name__}: {e}")
        return r
    r["check"] = compare(got, ref)
    r["status"] = "exact" if r["check"]["failing"] == 0 else "MISMATCH"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--pb-dir", default="/fuseflow-artifact/tortilla-visualizer")
    ap.add_argument("--conv-python", default=sys.executable, help="python with google.protobuf for the converter")
    ap.add_argument("--timeout", type=int, default=120, help="seconds per Comal run before it counts as a hang")
    a = ap.parse_args()
    global TIMEOUT
    TIMEOUT = a.timeout
    import comal
    if not hasattr(comal, "run_graph_outputs"):
        sys.exit(f"{comal.__file__}: no run_graph_outputs (need the Comal with writer-output patch 0003)")
    print("comal:", comal.__file__, flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    results = []
    for name in a.cases:
        try:
            r = run_case(name, CASES[name], a.pb_dir, a.conv_python)
        except Exception as e:                          # one broken case must not hide the others
            r = {"case": name, "status": "script_error", "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        short = {k: r.get(k) for k in ("status", "passed", "cycles", "check")}
        print(f"{name:16} {json.dumps(short)}", flush=True)
        if r.get("status") not in ("exact",):
            print("   ", json.dumps({k: r[k] for k in ("stderr", "error") if k in r})[:1500], flush=True)
    (RES / "smoke_results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"written {RES / 'smoke_results.json'}")


if __name__ == "__main__":
    main()
