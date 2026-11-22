"""
GRANII-dagger runs (tiled comparison, decision 2026-10-27: every method in the cross comparison is tiled).

    cd /fuseflow-v3 && /fuseflow-v3/venv/bin/python -u /fuseflow-v3/granii/scripts/grn_run.py         [--datasets cora cora_ml dblp] [--resume] [--big-scalar]

Only one new measurement is needed: GIN's (A + I) composition, the SpMM over A + I (make_grn_kernels.py:
grn_adjloops_f, width F; grn_adjloops_64, width 64), through H's harness h_kernel.py, blocked 16/32/64 only (no scalar
run), every output checked against float64 (GIN's own reference via --ref-kernel; H's tolerance rule). Everything else GRANII's composition space needs is
already measured (H, H_G) or provably dominated (granii_assemble.py).
Collab/MAG: H's rule does not run adjacency kernels blocked there (H uses their scalar values), so these two kernels are
skipped on Collab/MAG unless --big-scalar (scalar-only, the same treatment H gives its own adjacency kernels).
Sequential, nice 10. Runs: /fuseflow-v3/granii/runs/<dataset>/h__<stem>/ (+ .log); an unfinished earlier run is moved
to /fuseflow-v3/granii/archive/ first. Summary: /fuseflow-v3/granii/results/grn_runs.json.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

PY = "/fuseflow-v3/venv/bin/python"
H_KERNEL = "/fuseflow-v3/scripts/h_kernel.py"
RUNS = Path("/fuseflow-v3/granii/runs")
ARCH = Path("/fuseflow-v3/granii/archive")
SUMMARY = Path("/fuseflow-v3/granii/results/grn_runs.json")
DATASETS = {"cora": ("Planetoid", "Cora"), "cora_ml": ("CitationFull", "cora_ml"), "dblp": ("CitationFull", "dblp"),
            "collab": ("ogbl", "ogbl-collab"), "mag": ("ogbn", "ogbn-mag")}
BIG = {"collab", "mag"}
NEW = ["grn/grn_adjloops_f.mlir", "grn/grn_adjloops_64.mlir"]


def run(kind, path, ds, resume):
    stem = Path(path).stem
    rd = RUNS / ds / f"{kind}__{stem}"
    res = rd / "result.json"
    if resume and res.exists() and json.loads(res.read_text()).get("status") == "done":
        return json.loads(res.read_text()), "kept"
    if rd.exists():
        dst = ARCH / f"runs_{time.strftime('%Y%m%d_%H%M%S')}" / ds
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(["mv", str(rd), str(dst / rd.name)], check=True)
    rd.parent.mkdir(parents=True, exist_ok=True)
    # float64 reference: the GIN kernel this is a copy of (same computation A . H, here on A + I); without it h_kernel
    # finds no single-operation reference (fill + matmul) and could not check a blocked-only run
    ref = {"grn_adjloops_f": "gin_adj_h1", "grn_adjloops_64": "gin_adj_h2"}[stem]
    flags = ["--ref-kernel", ref] + (["--scalar-only"] if ds in BIG else ["--blocks", "16", "32", "64"])
    cmd = [PY, "-u", H_KERNEL, "--mlir", path, "--dataset", *DATASETS[ds], "--out", str(rd), *flags]
    cwd = "/fuseflow-artifact"
    with open(f"{rd}.log", "w") as f:
        subprocess.run(["nice", "-n", "10", *cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                       env={**__import__("os").environ, "DATA_PATH": "/tmp/data"})
    return (json.loads(res.read_text()) if res.exists() else {"status": "no result.json"}), "ran"


def value(kind, r):
    """H's per-kernel rule: fastest of the scalar run and every block size whose output passed"""
    c = [(r.get("paired_scalar") or {}).get("cycles") if (r.get("paired_scalar") or {}).get("passed") else None]
    c += [b.get("cycles") for b in (r.get("blocks") or {}).values() if b.get("status") == "ok"]
    c = [x for x in c if x]
    return min(c) if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "cora_ml", "dblp"], choices=list(DATASETS))
    ap.add_argument("--big-scalar", action="store_true", help="also Collab/MAG, scalar only (H's rule for adjacency)")
    ap.add_argument("--only", nargs="*", default=None, help="kernel stems to restrict to")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets:
        if ds in BIG and not a.big_scalar:
            print(f"{ds:7} skipped (H runs adjacency kernels scalar-only there; use --big-scalar)", flush=True)
            continue
        jobs = [("h", p) for p in NEW]
        for kind, path in jobs:
            stem = Path(path).stem
            if a.only and stem not in a.only:
                continue
            t0 = time.time()
            r, how = run(kind, path, ds, a.resume)
            rec = {"path": path, "status": r.get("status"), "value": value(kind, r), "reason": r.get("reason"),
                   "scalar": (r.get("paired_scalar") or {}).get("cycles"),
                   "blocks": {n: (b.get("status"), b.get("cycles")) for n, b in (r.get("blocks") or {}).items()},
                   "check": r.get("check")}
            summary.setdefault(ds, {})[f"{kind}__{stem}"] = rec
            SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
            print(f"{ds:7} {kind:3} {stem:20} {how} {time.time() - t0:6.0f} s  value {rec['value']}  scalar {rec['scalar']}  "
                  f"blocks {rec['blocks']}  {rec['status']} {rec['reason'] or ''}", flush=True)
    print(f"summary {SUMMARY}")


if __name__ == "__main__":
    main()
