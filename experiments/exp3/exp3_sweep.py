"""
Experiment 3: sensitivity sweep on synthetic graphs (one variable at a time), H / H+G / H+V / H+G+V.

    cd /fuseflow-v3 && /fuseflow-v3/venv/bin/python -u /fuseflow-v3/exp3/scripts/exp3_sweep.py --capture
    cd /fuseflow-v3 && /fuseflow-v3/venv/bin/python -u /fuseflow-v3/exp3/scripts/exp3_sweep.py --probe
    cd /fuseflow-v3 && /fuseflow-v3/venv/bin/python -u /fuseflow-v3/exp3/scripts/exp3_sweep.py [--sweeps density F C pattern] [--resume]

Layer studied: SGC's, S.S.X.W (S = D^-1/2 (A + I) D^-1/2), the only layer whose input width F and output width C are
both free (GCN's hidden width 16 is fixed in its kernels). Kernels (FuseFlow's / H_G's, compiled once):
  aggF = sgc_unfused/sgc_prop1.mlir  (S . X, width F)       gemm = sgc_unfused/sgc_linear_mul.mlir (X . W, F -> C)
  aggC = hg/hg_propnorm_c.mlir       (S . Z, width C)
  o2 (FuseFlow's order) = 2 aggF + gemm;  o3 (re-associated) = gemm + 2 aggC, generated only when C < F (H_G's rule).
Per kernel, as in H: the fastest of the scalar run and blocked 16/32/64 whose output passed; with V the aggregation
kernels also have their vector-16 run (Comal patch 0027). Totals:
  H = o2;  H+G = min(o2, o3);  H+V = o2 with V;  H+G+V = min(o2, o3) with V.
bias and softmax are left out (the same kernels, at width C, in all four variants: a constant); fused kernels (o1) are
left out (experiment 2b: no gain). Reported cycles are therefore the layer's aggregation + update part.
Mechanics: each kernel is compiled once by FuseFlow on Cora (--capture: program + tensor header saved); for every
point the inputs are written directly in FuseFlow's layouts with h_kernel.py's own writers (scalar = blocked layout at
n = 1, blocked n = 16/32/64, vector 16) and the Comal entry points are called in a separate process with a timeout;
every output is checked against float64 numpy (atol = rtol = 1e-4).
Sweeps (one variable at a time from BASE): N with the average degree fixed (density = 10 / N), density, F, C, pattern.
Graphs: uniform (random pairs), powerlaw (Chung-Lu, gamma 2.5), blockdiag (communities of 100 nodes, 90 % of edges
inside), symmetric, no duplicate edges, then + I and symmetric normalization. Seeded.
Outputs: /fuseflow-v3/exp3/results/sweep_results.json (+ captured programs in /fuseflow-v3/exp3/captured/).
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "/fuseflow-v3/cs/scripts")      # comal_isolated (read only)
sys.path.insert(1, "/fuseflow-v3/scripts")         # H's writers, unblock/unvec, compare (read only)
import cs_common as cs  # noqa: E402
import h_kernel as hk  # noqa: E402

ARTIFACT = Path("/fuseflow-artifact")
MODELS = ARTIFACT / "fuseflow-compiler" / "tests" / "models"
BUILD = "/fuseflow-v3/toolchain/compiler/build"
ROOT = Path("/fuseflow-v3/exp3")
CAP = ROOT / "captured"
WORK = ROOT / "work"
RES = ROOT / "results" / "sweep_results.json"
KERNELS = {"aggF": "sgc_unfused/sgc_prop1.mlir", "aggC": "hg/hg_propnorm_c.mlir", "gemm": "sgc_unfused/sgc_linear_mul.mlir"}
BASE = {"N": 2000, "density": 0.005, "F": 256, "C": 8, "pattern": "powerlaw"}
SWEEPS = {"N": [1000, 4000, 16000, 64000], "density": [0.001, 0.003, 0.01, 0.03, 0.1], "F": [16, 64, 256, 1024, 2048],
          "C": [4, 16, 64, 256, 512], "pattern": ["uniform", "powerlaw", "blockdiag"]}
AVG_DEG = BASE["density"] * BASE["N"]          # the N sweep keeps the average degree (about 10), not the density
BLOCKS = [16, 32, 64]
TIMEOUT = 7200


# ------------------------------------------------------------------ capture (once per kernel)
def capture():
    os.environ.setdefault("DATA_PATH", "/tmp/data")
    os.environ["COMAL_ENABLE_HBM"] = "0"
    os.environ["TORCH_WEIGHTS_ONLY_LOAD"] = "0"
    os.chdir(ARTIFACT)
    sys.path.insert(0, str(ARTIFACT / "scripts"))
    import comal
    captured = {}
    comal.run_graph = lambda op, data: (captured.update(op=op, data=data), (True, 0))[1]
    import run_end_to_end as r
    from utils import get_tensors_properties
    from newmodels_kernel import read_tensor
    for name, path in KERNELS.items():
        d = CAP / name
        if (d / "header.json").exists():
            print(f"{name}: already captured", flush=True)
            continue
        captured.clear()
        mlir = MODELS / path
        argv = ["--infile", str(mlir), "--build", BUILD, "-sp", "0.9", "-par", "1", "--useGen",
                "-inDataset", "Planetoid", "-inData", "Cora"]
        r.run_fuseflow_all(r.get_argparser().parse_args(argv))
        data = Path(captured["data"])
        header = get_tensors_properties(mlir.parent / f"{mlir.stem}_sam.mlir")
        shapes = {k: [int(v) for v in (data / f"tensor_{k}_mode_shape").read_text().split()]
                  for k in header if (data / f"tensor_{k}_mode_vals").exists()}
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(captured["op"], d / "op.bin")
        hdr = {k: {"format": header[k]["format"], "Arg": header[k].get("Arg"), "mode_order": header[k].get("mode_order"),
                   "shape_cora": shapes[k]} for k in shapes}
        (d / "header.json").write_text(json.dumps(hdr, indent=1) + "\n")
        # a sanity read of the captured inputs (the readers accept the header as FuseFlow wrote it)
        for k in shapes:
            read_tensor(str(data / f"tensor_{k}"), header[k], shapes[k])
        print(f"{name}: captured {path} -> {d}  tensors {[(k, v['format'], v['shape_cora']) for k, v in hdr.items()]}", flush=True)


# ------------------------------------------------------------------ synthetic graphs
def graph(N, density, pattern, seed):
    rng = np.random.default_rng(seed)
    m = max(1, int(round(density * N * N / 2)))
    if pattern == "uniform":
        w = np.ones(N)
    elif pattern == "powerlaw":
        w = (np.arange(1, N + 1, dtype=np.float64)) ** (-1.0 / 1.5)       # Chung-Lu, gamma = 2.5
    elif pattern == "blockdiag":
        w = None
    else:
        raise ValueError(pattern)
    edges = set()
    tries = 0
    while len(edges) < m and tries < 50:
        k = 2 * (m - len(edges)) + 16
        if w is not None:
            p = w / w.sum()
            i, j = rng.choice(N, k, p=p), rng.choice(N, k, p=p)
        else:
            i = rng.integers(0, N, k)
            inside = rng.random(k) < 0.9
            blk = (i // 100) * 100
            j = np.where(inside, blk + rng.integers(0, 100, k), rng.integers(0, N, k))
            j = np.minimum(j, N - 1)
        for a, b in zip(i.tolist(), j.tolist()):
            if a != b:
                edges.add((min(a, b), max(a, b)))
                if len(edges) >= m:
                    break
        tries += 1
    e = np.array(sorted(edges), dtype=np.int64)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(N, N))
    A = (A + A.T).tocsr()
    A = A + sp.eye(N, format="csr")
    d = np.asarray(A.sum(axis=1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(d))
    S = (Dm @ A @ Dm).tocsr().astype(np.float32)
    S.sort_indices()
    return S


# ------------------------------------------------------------------ one kernel at one point
def run_kernel(name, arrays, ref, workdir, vector):
    """scalar (n = 1 layout), blocked 16/32/64 and optionally vector 16; returns per-mode records"""
    d = CAP / name
    op = d / "op.bin"
    hdr = json.loads((d / "header.json").read_text())
    from newmodels_kernel import output_dense
    recs = {}
    scalar_got = None
    modes = [("scalar", 1)] + [(f"b{n}", n) for n in BLOCKS] + ([("v16", 16)] if vector else [])
    for mode, n in modes:
        wd = workdir / f"{name}_{mode}"
        wd.mkdir(parents=True)
        rec = {}
        try:
            for k, h in hdr.items():
                a = arrays[k]
                fmt = h["format"]
                pre = str(wd / f"tensor_{k}")
                if mode == "v16":
                    vec = a.shape[-1] == ref.shape[-1]
                    if sp.issparse(a) and not vec:
                        hk.write_vec_sparse(a, fmt, pre, 16)
                    else:
                        hk.write_vec_dense(a.toarray() if sp.issparse(a) else a, fmt, pre, 16, vec)
                elif sp.issparse(a):
                    hk.write_blocked_sparse(a, fmt, pre, n)
                else:
                    hk.write_blocked_dense(np.asarray(a, dtype=np.float32), fmt, pre, n)
            entry = {"scalar": "run_graph_outputs", "v16": "run_graph_vector16_outputs"}.get(mode, f"run_graph_true_block{n}_outputs")
            t0 = time.time()
            res = cs.comal_isolated(op, wd, workdir / f"{name}_{mode}.pkl", TIMEOUT, entry=entry,
                                    log=workdir / f"{name}_{mode}.log")
            rec["seconds"] = round(time.time() - t0, 1)
            if res[0] != "ok":
                rec.update(status=f"comal_{res[0]}", error=str(res[1])[-500:])
            else:
                passed, cyc, fibers, vals = res[1]
                rec.update(passed=bool(passed), cycles=int(cyc))
                if not passed:
                    rec["status"] = "sim-failed"
                else:
                    if mode == "scalar":
                        got = output_dense(fibers, vals, list(ref.shape))[0]
                    elif mode == "v16":
                        got = hk.unvec(fibers, vals, list(ref.shape), 16)
                    else:
                        got = hk.unblock(fibers, vals, [-(-s // n) for s in ref.shape], n)[:ref.shape[0], :ref.shape[1]]
                    rec["check"] = hk.compare(got, ref)
                    rec["status"] = "ok" if rec["check"]["failing"] == 0 else "mismatch"
                    if mode == "scalar":
                        scalar_got = got
                    elif rec["status"] == "mismatch" and scalar_got is not None:
                        # H's tolerance rule (2026-10-17): accepted only where FuseFlow's scalar output fails too
                        rec["check"]["baseline"] = hk.baseline_accept(got, ref, scalar_got)
                        if rec["check"]["baseline"]["accepted"]:
                            rec["status"] = "ok"
        except Exception as e:
            rec.update(status="error", error=f"{type(e).__name__}: {e}"[-800:])
        shutil.rmtree(wd, ignore_errors=True)
        recs[mode] = rec
        print(f"      {name:5} {mode:6} {rec.get('status'):9} {rec.get('cycles')}  {rec.get('seconds')} s  {rec.get('error') or ''}",
              flush=True)
    return recs


def best(recs, with_v):
    c = [r["cycles"] for m, r in recs.items() if r.get("status") == "ok" and (with_v or m != "v16")]
    return min(c) if c else None


def run_point(p, seed=7):
    N, F, C = p["N"], p["F"], p["C"]
    S = graph(N, p["density"], p["pattern"], seed)
    rng = np.random.default_rng(seed + 1)
    X = rng.uniform(-1, 1, (N, F)).astype(np.float32)
    W = rng.uniform(-1, 1, (F, C)).astype(np.float32)
    Z = (X.astype(np.float64) @ W.astype(np.float64)).astype(np.float32)
    S64 = S.astype(np.float64)
    roles = {}
    for name in KERNELS:
        hdr = json.loads((CAP / name / "header.json").read_text())
        nc = max(max(h["shape_cora"]) for h in hdr.values())          # Cora's N is the largest dimension
        m = {}
        for k, h in hdr.items():
            s = tuple(h["shape_cora"])
            if s == (nc, nc):
                m[k] = S
            elif name == "gemm" and len(s) == 2 and s[0] == nc:
                m[k] = X
            elif name == "gemm":
                m[k] = sp.csr_matrix(W)
            elif name == "aggF":
                m[k] = X
            else:
                m[k] = Z
        roles[name] = m
    refs = {"aggF": np.asarray(S64 @ X.astype(np.float64)), "aggC": np.asarray(S64 @ Z.astype(np.float64)),
            "gemm": X.astype(np.float64) @ W.astype(np.float64)}
    wd = WORK / f"pt_{int(time.time())}"
    wd.mkdir(parents=True)
    out = {"params": p, "nnz": int(S.nnz), "kernels": {}}
    use_o3 = C < F
    for name in (["aggF", "gemm"] + (["aggC"] if use_o3 else [])):
        out["kernels"][name] = run_kernel(name, roles[name], refs[name], wd, vector=name != "gemm")
    shutil.rmtree(wd, ignore_errors=True)
    k = out["kernels"]
    hv = {n: (best(r, False), best(r, True)) for n, r in k.items()}
    o2 = lambda v: (2 * hv["aggF"][v] + hv["gemm"][0]) if hv["aggF"][v] and hv["gemm"][0] else None
    o3 = lambda v: (hv["gemm"][0] + 2 * hv["aggC"][v]) if use_o3 and hv["aggC"][v] and hv["gemm"][0] else None
    mn = lambda *x: min((y for y in x if y), default=None)
    out["totals"] = {"H": o2(0), "H+G": mn(o2(0), o3(0)), "H+V": o2(1), "H+G+V": mn(o2(1), o3(1)),
                     "o3_generated": use_o3, "kernel_values": {n: {"H": a, "V": b} for n, (a, b) in hv.items()}}
    return out


def key(p):
    return f"N{p['N']}_d{p['density']}_F{p['F']}_C{p['C']}_{p['pattern']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--probe", action="store_true", help="only the base point")
    ap.add_argument("--sweeps", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.capture:
        capture()
        return 0
    os.environ["COMAL_ENABLE_HBM"] = "0"
    os.chdir(ARTIFACT)
    sys.path.insert(0, str(ARTIFACT / "scripts"))
    for name in KERNELS:
        if not (CAP / name / "header.json").exists():
            sys.exit(f"{name} not captured: run with --capture first")
    doc = json.loads(RES.read_text()) if RES.exists() else {"metadata": {"definition": __doc__, "base": BASE}, "points": {}}
    RES.parent.mkdir(parents=True, exist_ok=True)
    pts = [dict(BASE)] if a.probe else [
        dict(BASE, **({"N": v, "density": round(AVG_DEG / v, 8)} if var == "N" else {var: v}))
        for var in a.sweeps for v in SWEEPS[var]]
    for p in pts:
        kk = key(p)
        if a.resume and kk in doc["points"] and "totals" in doc["points"][kk]:
            print(f"{kk}: kept", flush=True)
            continue
        print(f"== {kk}", flush=True)
        t0 = time.time()
        try:
            r = run_point(p)
        except Exception as e:
            r = {"params": p, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        r["seconds"] = round(time.time() - t0, 1)
        doc["points"][kk] = r
        RES.write_text(json.dumps(doc, indent=1) + "\n")
        t = r.get("totals") or {}
        h = t.get("H")
        g = lambda x: f"{h / x:6.2f}x" if h and x else "   -  "
        print(f"   H {h}  H+G {g(t.get('H+G'))}  H+V {g(t.get('H+V'))}  H+G+V {g(t.get('H+G+V'))}  "
              f"({r['seconds']} s) {r.get('error') or ''}", flush=True)
    print(f"written {RES}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
