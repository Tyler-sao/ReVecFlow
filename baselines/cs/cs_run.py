"""
C+S baseline driver: cs_kernel.py for every matmul kernel of FuseFlow's unfused configuration, per dataset.

    cd /fuseflow-v3 && /fuseflow-v3/venv/bin/python -u /fuseflow-v3/cs/scripts/cs_run.py \
        [--datasets cora cora_ml dblp collab mag] [--models gcn graphsage gin sgc] [--only gcn_adj_x1 ...] [--resume]

Kernels: h0_inventory.json (the H baseline's inventory), configuration "unfused", MLIR containing linalg.matmul
(13 files for the four models; GraphSAGE's live in their own directory and are run separately). Each kernel file and
dataset is run once, sequentially, at nice 10; a finished run (result.json status done) is kept with --resume, any other
earlier attempt is moved to /fuseflow-v3/cs/archive/ first (never deleted).
Runs: /fuseflow-v3/cs/results/runs/<dataset>/<model>__<kernel>/ (+ .log). Summary: /fuseflow-v3/cs/results/cs_runs.json.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

PY = "/fuseflow-v3/venv/bin/python"
KERNEL = "/fuseflow-v3/cs/scripts/cs_kernel.py"
INV = Path("/fuseflow-v3/h_baseline/h0/h0_inventory.json")
MODELS = Path("/fuseflow-artifact/fuseflow-compiler/tests/models")
RUNS = Path("/fuseflow-v3/cs/results/runs")
ARCH = Path("/fuseflow-v3/cs/archive")
SUMMARY = Path("/fuseflow-v3/cs/results/cs_runs.json")
DATASETS = {"cora": ("Planetoid", "Cora"), "cora_ml": ("CitationFull", "cora_ml"), "dblp": ("CitationFull", "dblp"),
            "collab": ("ogbl", "ogbl-collab"), "mag": ("ogbn", "ogbn-mag")}


def matmul_kernels(models, only):
    inv = json.loads(INV.read_text())
    out = {}
    for key, r in sorted(inv.items()):
        if r.get("config") != "unfused" or r.get("model") not in models or not r.get("path"):
            continue
        if only and r["kernel"] not in only:
            continue
        f = MODELS / r["path"]
        if f.exists() and re.search(r"linalg\.matmul", f.read_text()):
            out[r["path"]] = {"model": r["model"], "kernel": r["kernel"], "key": key}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora"], choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=["gcn", "graphsage", "gin", "sgc"])
    ap.add_argument("--only", nargs="*", default=None, help="kernel names to restrict to")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--timeout", type=int, default=21600)
    ap.add_argument("--blocked-only", action="store_true", help="pass --blocked-only to cs_kernel.py (GEMM kernels "
                    "too large to simulate in scalar mode; adjacency kernels on Collab/MAG record their lower bound)")
    a = ap.parse_args()
    ks = matmul_kernels(a.models, a.only)
    print(f"{len(ks)} matmul kernels: {[v['model'] + '/' + v['kernel'] for v in ks.values()]}", flush=True)
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    for ds in a.datasets:
        for path, k in ks.items():
            name = f"{k['model']}__{k['kernel']}"
            rd = RUNS / ds / name
            res = rd / "result.json"
            if a.resume and res.exists() and json.loads(res.read_text()).get("status") == "done":
                print(f"{ds:7} {name:34} kept (done)", flush=True)
            else:
                if rd.exists():
                    dst = ARCH / f"runs_{time.strftime('%Y%m%d_%H%M%S')}" / ds
                    dst.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["mv", str(rd), str(dst / name)], check=True)
                    log = rd.with_suffix(".log")
                    if log.exists():
                        subprocess.run(["mv", str(log), str(dst / f"{name}.log")], check=True)
                rd.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                with open(rd.with_suffix(".log"), "w") as f:
                    subprocess.run(["nice", "-n", "10", PY, "-u", KERNEL, "--mlir", path, "--dataset", *DATASETS[ds],
                                    "--out", str(rd), "--timeout", str(a.timeout),
                                    *(["--blocked-only"] if a.blocked_only else [])],
                                   cwd="/fuseflow-v3", stdout=f, stderr=subprocess.STDOUT)
                print(f"{ds:7} {name:34} ran {time.time() - t0:7.0f} s", flush=True)
            r = json.loads(res.read_text()) if res.exists() else {"status": "no result.json"}
            summary.setdefault(ds, {})[name] = {
                "path": path, "status": r.get("status"), "fuseflow_scalar": (r.get("fuseflow") or {}).get("cycles"),
                "cs_scalar": r.get("cs_best"), "cs_blocks": {n: (v.get("status"), v.get("cycles"))
                                                             for n, v in (r.get("cs_blocks") or {}).items()},
                "cs_H": r.get("cs_H"), "cs_blocks_skipped": r.get("cs_blocks_skipped"), "reason": r.get("reason"),
                "cs_mode": r.get("cs_mode", "simulated"), "cs_scalar_lower_bound": r.get("cs_scalar_lower_bound")}
            SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
            s = summary[ds][name]
            print(f"        fuseflow {s['fuseflow_scalar']}  cs {s['cs_scalar']}  cs_H {s['cs_H']}  "
                  f"blocks {s['cs_blocks']}  {s['status']} {s['cs_mode']} {s['cs_scalar_lower_bound'] or ''}", flush=True)
    print(f"summary {SUMMARY}")


if __name__ == "__main__":
    main()
