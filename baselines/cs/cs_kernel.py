"""
C+S baseline, one matrix-multiplication kernel of FuseFlow's unfused configuration on one dataset.

    cd /fuseflow-v3 && /fuseflow-v3/venv/bin/python -u /fuseflow-v3/cs/scripts/cs_kernel.py \
        --mlir gcn_unfused/gcn_adj_x2.mlir --dataset Planetoid Cora --out /fuseflow-v3/cs/results/runs/cora/gcn_adj_x2 \
        [--orders ikj ijk] [--blocks 16 32 64] [--timeout 21600]

Definition (decision 2026-10-26, option a): C+S compiles with Custard every expression Custard compiles correctly.
Custard's own graphs were checked in the smoke test (/fuseflow-v3/cs/results/smoke): single matrix products are correct
(Gustavson order i,k,j only with the documented repair --drop-reduce-before-spacc); expressions with a sum of terms
(bias add, GraphSAGE's two products) or two reductions (A.H.W) come out wrong. So C+S rewrite = C+S unfused: Custard for
every matmul kernel, FuseFlow's own kernels for everything else.

Steps, all on the inputs FuseFlow's generator writes for this kernel (as h_kernel.py, Figure 12's arguments):
  1. FuseFlow: compile the kernel (V3 compiler), capture its program and data, read its inputs;
  2. reference: float64 product of the two inputs (as read, float32 values);
  3. FuseFlow paired run: its own program on its own files, scalar Comal, HBM off (= H's setting), checked;
  4. C+S: per loop order, Custard expression X(i,j)=A(i,k)*B(k,j) with FuseFlow's formats for A and B (compressed ->
     s, dense -> d), output X:ss; the graph converted (cs_gv2proto.py), the inputs written in Custard's layout, scalar
     Comal, HBM off, checked (atol = rtol = 1e-4).
     ikj (default): Gustavson, natural storage, with the repair; ijk: inner product, B stored column-major. The inner
     product scans a dense B column for every output element (N x N x F intersections on an adjacency product), so it
     is only run when asked.
  5. C+S blocked (the comparison against H, decision 2026-10-26): the same Custard ikj program in Comal's
     true-block mode at n = 16/32/64 (Comal's MUL is the n x n tile product, the sparse accumulator adds tiles), the
     inputs written by H's blocked writers (h_kernel.py), the output checked against float64 with H's tolerance rule
     (2026-10-17: beyond-tolerance elements accepted only if FuseFlow's own scalar output fails there too, with no
     smaller error). As in H, on Collab and MAG a kernel reading the N x N adjacency is not run blocked.
     C+S per kernel ("cs_H") = the fastest of the C+S scalar run and every block size whose output passed.
Results: OUT/result.json. Inputs and programs stay under OUT (removed with --clean).
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, "/fuseflow-v3/scripts")       # H's blocked writers and tolerance rule (read only)
import cs_common as cs  # noqa: E402
import h_kernel as hk  # noqa: E402
BIG = {"ogbl-collab", "ogbn-mag"}

ARTIFACT = Path("/fuseflow-artifact")
MODELS = ARTIFACT / "fuseflow-compiler" / "tests" / "models"
BUILD = "/fuseflow-v3/toolchain/compiler/build"
PB_DIR = "/fuseflow-artifact/tortilla-visualizer"
LETTER = {"compressed": "s", "dense": "d"}


class _SkipScalar(Exception):
    pass


def adjacency_kernel(L):
    return L.shape[0] == L.shape[1] and sp.issparse(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir", required=True)
    ap.add_argument("--dataset", nargs=2, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orders", nargs="+", default=["ikj"], choices=["ikj", "ijk"])
    ap.add_argument("--timeout", type=int, default=21600, help="seconds per Comal run")
    ap.add_argument("--blocks", type=int, nargs="*", default=[16, 32, 64], choices=[16, 32, 64])
    ap.add_argument("--no-fuseflow", action="store_true", help="skip the paired FuseFlow run")
    ap.add_argument("--clean", action="store_true", help="remove the written inputs after the run")
    ap.add_argument("--blocked-only", action="store_true",
                    help="skip both scalar simulations (FuseFlow paired and C+S); convert the Custard ikj program and run it "
                         "only in true-block mode. For GEMM kernels whose scalar run would exceed the timeout (MAG): on "
                         "Cora/Cora-ML/DBLP/Collab every GEMM's C+S scalar run matched FuseFlow's and was 100-1000x slower "
                         "than its blocked runs, so it never decides cs_H")
    ap.add_argument("--simulate-big-adjacency", action="store_true",
                    help="on Collab/MAG, simulate the adjacency SpMM anyway instead of recording the N^2 lower bound")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    mlir = MODELS / a.mlir
    kernel = mlir.stem
    result = {"kernel": kernel, "mlir": str(mlir), "dataset": a.dataset, "status": "running", "hbm": False,
              "definition": "C+S option a: Custard for matmul kernels (see cs_kernel.py docstring)"}
    write = lambda: (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    write()
    os.environ.setdefault("DATA_PATH", "/tmp/data")
    os.environ["COMAL_ENABLE_HBM"] = "0"
    os.environ["TORCH_WEIGHTS_ONLY_LOAD"] = "0"
    t_start = time.time()
    try:
        if not re.search(r"linalg\.matmul", mlir.read_text()):
            raise SystemExit(f"{a.mlir}: no linalg.matmul; C+S runs only the matmul kernels (option a)")
        # ---- 1. FuseFlow's program and inputs (exactly as h_kernel.py captures them)
        os.chdir(ARTIFACT)
        sys.path.insert(0, str(ARTIFACT / "scripts"))
        import comal
        captured = {}

        def intercept(op, data):
            captured.update(op=op, data=data)
            return True, 0

        comal.run_graph = intercept
        import run_end_to_end as r
        from utils import get_tensors_properties
        from newmodels_kernel import output_dense, read_tensor
        argv = ["--infile", str(mlir), "--build", BUILD, "-sp", "0.9", "-par", "1", "--useGen",
                "-inDataset", a.dataset[0], "-inData", a.dataset[1]]
        r.run_fuseflow_all(r.get_argparser().parse_args(argv))
        if not captured:
            raise RuntimeError("run_end_to_end did not reach the simulation")
        op = out / "fuseflow_op.bin"
        shutil.copy(captured["op"], op)
        data = Path(captured["data"])
        sam = mlir.parent / f"{kernel}_sam.mlir"
        header = get_tensors_properties(sam)
        shapes = {k: [int(v) for v in (data / f"tensor_{k}_mode_shape").read_text().split()]
                  for k in header if (data / f"tensor_{k}_mode_vals").exists()}
        ins = {k: read_tensor(str(data / f"tensor_{k}"), header[k], shapes[k]) for k in shapes}
        two = [(k, v) for k, v in ins.items() if v.ndim == 2]
        if len(two) != 2:
            raise RuntimeError(f"expected two 2-D inputs, found {[(k, v.shape) for k, v in ins.items()]}")
        pairs = [(L, R) for L in two for R in two if L[0] != R[0] and L[1].shape[1] == R[1].shape[0]]
        if len(pairs) == 2:      # square operands: keep FuseFlow's argument order
            pairs.sort(key=lambda p: header[p[0][0]].get("Arg") if isinstance(header[p[0][0]].get("Arg"), int) else 99)
        if not pairs:
            raise RuntimeError(f"no operand order fits a product: {[(k, v.shape) for k, v in two]}")
        (kl, L), (kr, R) = pairs[0]
        fl = "".join(LETTER[f] for f in header[kl]["format"])
        fr = "".join(LETTER[f] for f in header[kr]["format"])
        result["operands"] = {"A": {"fuseflow": kl, "shape": list(L.shape), "format": fl,
                                    "nnz": int(L.nnz) if sp.issparse(L) else int(np.count_nonzero(L))},
                              "B": {"fuseflow": kr, "shape": list(R.shape), "format": fr,
                                    "nnz": int(R.nnz) if sp.issparse(R) else int(np.count_nonzero(R))}}
        dense = lambda m: m.toarray() if sp.issparse(m) else np.asarray(m)
        adjacency = L.shape[0] == L.shape[1] and sp.issparse(L)
        if a.dataset[1] in BIG and adjacency and not a.simulate_big_adjacency:
            # Custard's ikj SpMM intersects every row of A with B's DENSE k level (no locate), so its scalar run costs
            # at least N^2 cycles: measured N^2 x 1.0000-1.0004 on Cora, Cora-ML and DBLP whenever N^2 exceeds the
            # product work. Collab: N^2 = 5.6e10 cycles (about 15 h of simulation), MAG: 5.4e11 (about 150 h). H's rule
            # does not run these kernels blocked either. Recorded as a lower bound, not simulated.
            n = int(L.shape[0])
            result.update(status="done", cs_mode="lower_bound", cs_scalar_lower_bound=n * n, cs_H=None,
                          cs_best=None, note="C+S scalar >= N^2 (dense-k intersection per row); not simulated, "
                                             "see cs_kernel.py; rerun with --simulate-big-adjacency to simulate")
            raise SystemExit(f"lower bound recorded: N^2 = {n * n}")
        # sparse x dense stays sparse-aware (a dense Collab adjacency would need 415 GiB)
        if sp.issparse(L):
            ref = np.asarray(L.astype(np.float64) @ dense(R).astype(np.float64))
        else:
            ref = dense(L).astype(np.float64) @ dense(R).astype(np.float64)
        result["reference"] = {"shape": list(ref.shape), "how": "float64 A @ B of FuseFlow's generated inputs"}
        write()

        # ---- 3. FuseFlow's own program, paired, same settings
        ff_out = None
        if a.blocked_only:
            result["cs_mode"] = "blocked_only"
            if adjacency_kernel(L):
                raise RuntimeError("--blocked-only is for GEMM kernels; an adjacency SpMM's scalar run decides cs_H")
        if not a.no_fuseflow and not a.blocked_only:
            t0 = time.time()
            res = cs.comal_isolated(op, data, out / "ff.pkl", a.timeout, log=out / "fuseflow_scalar.log")
            ff = {"seconds": round(time.time() - t0, 1)}
            if res[0] == "ok":
                passed, cyc, fibers, vals = res[1]
                ff.update(passed=bool(passed), cycles=int(cyc))
                if passed:
                    try:
                        ff_out = output_dense(fibers, vals, list(ref.shape))[0]
                        ff["check"] = cs.compare(ff_out, ref)
                    except Exception as e:
                        ff["check_error"] = f"{type(e).__name__}: {e}"
            else:
                ff["status"] = res[0]
                ff["error"] = res[1]
            result["fuseflow"] = ff
            write()
            print("fuseflow", json.dumps(ff), flush=True)

        # ---- 4. C+S, per loop order
        result["cs"] = {}
        for order in a.orders:
            od = out / f"cs_{order}"
            (od / "data").mkdir(parents=True)
            rb = "1,0" if order == "ijk" else None
            fmts = ["X:ss", f"A:{fl}", f"B:{fr}" + (f":{rb}" if rb else "")]
            sched = {"ikj": "i,k,j", "ijk": "i,j,k"}[order]
            flags = [cs.FIX] if order == "ikj" else []
            rec = {"expr": "X(i,j)=A(i,k)*B(k,j)", "formats": fmts, "order": sched, "repairs_flag": flags}
            try:
                proto, conv = cs.custard_graph(rec["expr"], fmts, sched, od / f"{kernel}_{order}.gv", flags, PB_DIR,
                                               sys.executable)
                rec["convert"] = {k: conv[k] for k in ("formats", "operators", "repeat_rewired", "sinks", "repairs",
                                                       "product_order")}
                rec["proto"] = str(proto)
                fm = cs.parse_formats(conv["formats"])
                if a.blocked_only:
                    rec["status"] = "not_simulated"
                    raise _SkipScalar()
                cs.write_tensor(od / "data", "A", L, *fm["A"])
                cs.write_tensor(od / "data", "B", R, *fm["B"])
                t0 = time.time()
                res = cs.comal_isolated(proto, od / "data", od / "cs.pkl", a.timeout, log=od / "cs_scalar.log")
                rec["seconds"] = round(time.time() - t0, 1)
                if res[0] != "ok":
                    rec.update(status=f"comal_{res[0]}", error=res[1])
                else:
                    passed, cyc, fibers, vals = res[1]
                    rec.update(passed=bool(passed), cycles=int(cyc))
                    if not passed:
                        rec["status"] = "comal_failed"
                    else:
                        got = cs.dense_output(fibers, vals, "X", *fm["X"], "ij", ref.shape)
                        rec["check"] = cs.compare(got, ref)
                        rec["status"] = "exact" if rec["check"]["failing"] == 0 else "MISMATCH"
            except _SkipScalar:
                pass
            except Exception as e:
                rec.update(status="error", error=f"{type(e).__name__}: {e}"[-2000:])
            if a.clean:
                shutil.rmtree(od / "data", ignore_errors=True)
            result["cs"][order] = rec
            write()
            print(f"cs {order}", json.dumps({k: rec.get(k) for k in ("status", "cycles", "check", "seconds", "error")}),
                  flush=True)
        ok = [v["cycles"] for v in result["cs"].values() if v.get("status") == "exact"]
        result["cs_best"] = min(ok) if ok else None
        if result.get("fuseflow", {}).get("cycles") and result["cs_best"]:
            result["cs_over_fuseflow"] = round(result["cs_best"] / result["fuseflow"]["cycles"], 4)

        # ---- 5. C+S blocked: the ikj program in true-block mode
        result["cs_blocks"] = {}
        adjacency = L.shape[0] == L.shape[1] and sp.issparse(L)
        if a.dataset[1] in BIG and adjacency:
            result["cs_blocks_skipped"] = "H rule: on Collab/MAG a kernel reading the N x N adjacency is not run blocked"
            a.blocks = []
        ikj = result["cs"].get("ikj")
        if a.blocks and not (ikj and ikj.get("proto") and ikj.get("status") in ("exact", "not_simulated")):
            result["cs_blocks_skipped"] = "no exact scalar ikj program to run blocked"
            a.blocks = []
        for n in a.blocks:
            bd = out / f"cs_ikj_block{n}_data"
            bd.mkdir()
            rec = {}
            try:
                for name, m, lv in (("A", L, fl), ("B", R, fr)):
                    fmt = ["compressed" if x == "s" else "dense" for x in lv]
                    if fmt == ["compressed", "compressed"]:
                        rec[name] = hk.write_blocked_sparse(sp.csr_matrix(m), fmt, str(bd / f"tensor_{name}"), n)
                    else:
                        rec[name] = hk.write_blocked_dense(np.asarray(dense(m), dtype=np.float32), fmt,
                                                           str(bd / f"tensor_{name}"), n)
                t0 = time.time()
                res = cs.comal_isolated(ikj["proto"], bd, out / f"b{n}.pkl", a.timeout,
                                        entry=f"run_graph_true_block{n}_outputs", log=out / f"cs_ikj_block{n}.log")
                rec = {"inputs": rec, "seconds": round(time.time() - t0, 1)}
                if res[0] != "ok":
                    rec.update(status=f"comal_{res[0]}", error=res[1])
                else:
                    passed, cyc, fibers, vals = res[1]
                    rec.update(passed=bool(passed), cycles=int(cyc))
                    if not passed:
                        rec["status"] = "comal_failed"
                    else:
                        cshape = [-(-x // n) for x in ref.shape]
                        got = cs.dense_output_blocked(fibers, vals, "X", (0, 1), "ij", n, cshape)
                        got = got[:ref.shape[0], :ref.shape[1]]
                        rec["check"] = cs.compare(got, ref)
                        rec["status"] = "ok" if rec["check"]["failing"] == 0 else "mismatch"
                        if rec["status"] == "mismatch" and ff_out is not None:
                            rec["check"]["baseline"] = hk.baseline_accept(got, ref, ff_out)
                            if rec["check"]["baseline"]["accepted"]:
                                rec["status"] = "ok"
            except Exception as e:
                rec.update(status="error", error=f"{type(e).__name__}: {e}"[-2000:])
            shutil.rmtree(bd, ignore_errors=True)
            result["cs_blocks"][str(n)] = rec
            write()
            print(f"cs block{n}", json.dumps({k: rec.get(k) for k in ("status", "cycles", "check", "seconds", "error")}),
                  flush=True)
        cand = ok + [v["cycles"] for v in result["cs_blocks"].values() if v.get("status") == "ok"]
        result["cs_H"] = min(cand) if cand else None
        result["status"] = "done" if (ok or (a.blocked_only and result["cs_H"])) else "no_valid_cs"
    except SystemExit as e:
        if result.get("cs_mode") == "lower_bound":
            result["reason"] = str(e)
        else:
            result.update(status="skipped", reason=str(e))
    except Exception as e:
        result.update(status="error", reason=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    result["seconds"] = round(time.time() - t_start, 1)
    write()
    print(json.dumps({k: result.get(k) for k in ("kernel", "status", "cs_best", "cs_over_fuseflow", "cs_H", "reason")}),
          flush=True)
    return 0 if result["status"] == "done" else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # the first Cora run finished (result written) but its process never exited: FuseFlow's generator leaves
    # non-daemon threads behind. Leave without waiting for them.
    os._exit(code)
