#!/usr/bin/env python3
"""
FuseFlow V3, full ablation over two techniques on top of H (FuseFlow with tiling): G = re-association + per-layer kernel
boundaries (computation graph), V = vectorized feature dimension (operator: the aggregation walks A nonzero by nonzero
and processes the dense operand's feature row as 16-lane vectors; Comal patch 0027). Cells: H, H+G, H+V, H+G+V.

    cd /fuseflow-v3 && /fuseflow-v3/venv/bin/python -u /fuseflow-v3/scripts/h_v_run.py \
        [--datasets cora cora_ml dblp collab mag] [--models gcn graphsage sgc gin] [--resume] [--assemble-only]

1. New runs (h_kernel.py --vector 16), each output checked against the kernel's float64 reference (H's checks, the
   tolerance rule of 2026-10-17): the aggregation kernels A . dense of every model (gcn_adj_x1/x2, gin_adj_h1/h2,
   sgc_prop1/2) and H_G's A.(N x C) kernels (not on MAG: C = 349, H_G does not use them).
   All five datasets: the vector mode does not tile A, so H's A rule for Collab/MAG does not apply.
   Timing (patch 0027): Comal's element-wise convention, latency = elements processed (16 per vector), one vector per cycle.
2. Assembly: for every kernel of every graph variant, candidate values tagged with the techniques they need -
     {}  H's value        {V} its vector run
   A cell S uses, per kernel, the cheapest candidate whose tags are in S. Without G the cell has FuseFlow's
   configurations (unfused = o2 in every layer, partially fused = o1); with G, H_G's per-layer choice among o1, o2, o3
   (LAYERS of h_g_run.py). Fully fused = H's value in every cell.
   Built-in check: the cells H and H+G must reproduce h_results and h_g_results.
Results: results/h_vec/cells/<cell>.json (plot schema, one per cell), results/h_vec/h_vec_summary.json, runs in results/h_vec/runs/.
"""
import argparse
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PY = "/fuseflow-v3/venv/bin/python"
KERNEL = "/fuseflow-v3/scripts/h_kernel.py"
HERE = Path(__file__).resolve().parent
INV = Path("/fuseflow-v3/h_baseline/h0/h0_inventory.json")
MODELS = Path("/fuseflow-artifact/fuseflow-compiler/tests/models")
FILES = {"H": Path("/fuseflow-v3/h_baseline/h_results.json"),
         "H_G": Path("/fuseflow-v3/results/h_g/h_g_results.json")}
OUT = Path("/fuseflow-v3/results/h_vec")
AGG = re.compile(r"^(gcn_adj_x[12]|gin_adj_h[12]|sgc_prop[12])$")   # aggregation kernels A . dense


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hg = load("h_g_run")
LAYERS, DATASETS, BIG = hg.LAYERS, hg.DATASETS, hg.BIG
CELLS = {"".join(sorted(s)) or "-": set(s) for k in range(3) for s in itertools.combinations("GV", k)}
NAME = lambda s: "H" + ("+" + "+".join(sorted(s)) if s else "")


def run_v(path, ds, flags, resume):
    rd = OUT / "runs" / ds / Path(path).stem
    res = rd / "result.json"
    if resume and res.exists() and json.loads(res.read_text()).get("status") == "done":
        return json.loads(res.read_text())
    if rd.exists():
        subprocess.run(["rm", "-rf", str(rd)], check=True)
    rd.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", KERNEL, "--mlir", path, "--dataset", *DATASETS[ds], "--out", str(rd), "--vector", "16", *flags]
    with open(f"{rd}.log", "w") as f:
        subprocess.run(cmd, cwd="/fuseflow-artifact", stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "DATA_PATH": "/tmp/data"})
    return json.loads(res.read_text()) if res.exists() else {"status": "error", "reason": f"no result.json, see {rd}.log"}


def best_block(r):
    ok = {n: b.get("cycles") for n, b in (r or {}).get("blocks", {}).items() if b.get("status") == "ok" and b.get("cycles")}
    return min(ok.values()) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=list(LAYERS), choices=list(LAYERS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--assemble-only", action="store_true", help="no runs: assemble from the runs present")
    args = ap.parse_args()
    inv = json.loads(INV.read_text())
    D = {k: json.loads(p.read_text()) for k, p in FILES.items()}
    H = D["H"]
    fname = lambda r: Path(r["path"]).name if len(set((r.get("copies") or {"x": 0}).values())) == 1 else r["path"]
    reads_a = lambda r: any(t.get("shape") == ["1767", "1767"] for t in (r.get("tensors") or {}).values())
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"metadata": {"cells": {c: NAME(s) for c, s in CELLS.items()}}, "runs": {}, "best": {}, "checks": []}
    R = {}  # (ds, path) -> vector-mode result
    for ds in args.datasets:
        jobs = {}  # path -> flags
        for key, r in inv.items():
            m, cfg, stem = key.split("/")
            if m in args.models and cfg in ("unfused", "partially_fused") and r.get("path") and AGG.match(stem):
                jobs.setdefault(r["path"], [])
        if ds != "mag":
            for name in hg.NEW:
                jobs.setdefault(hg.NEW[name], [])
        seen = {}
        for path, flags in sorted(jobs.items()):
            k = Path(path).name
            same = k in seen and (MODELS / path).read_bytes() == (MODELS / seen[k]).read_bytes()
            if same:                                       # byte-identical copies (e.g. gcn_adj_x1 for GCN and GraphSAGE)
                R[(ds, path)] = R[(ds, seen[k])]
                continue
            seen.setdefault(k, path)
            rd = OUT / "runs" / ds / Path(path).stem
            if args.assemble_only:
                R[(ds, path)] = json.loads((rd / "result.json").read_text()) if (rd / "result.json").exists() else {}
                continue
            t = time.time()
            r = run_v(path, ds, flags, args.resume)
            R[(ds, path)] = r
            summary["runs"].setdefault(ds, {})[path] = {
                "status": r.get("status"), "reason": r.get("reason"), "check": r.get("check"), "seconds": round(time.time() - t, 1),
                "blocks": {n: (b.get("cycles"), b.get("status")) for n, b in r.get("blocks", {}).items()},
                "peak_rss_gb": r.get("peak_rss_gb")}
            print(f"{ds:7} V {path:50} {json.dumps(summary['runs'][ds][path]['blocks'])}  {r.get('status')} {r.get('reason') or ''}", flush=True)
            (OUT / "h_vec_summary.json").write_text(json.dumps(summary, indent=1))
        # ------------------------------------------------------------------ assembly
        path_of = {key: r.get("path") for key, r in inv.items()}
        for m in args.models:
            def cands(k, variant):
                """[(cycles, tags, held)] for kernel k of a variant."""
                out = []
                if k.startswith("new:"):
                    name = k[4:]
                    v = ((D["H_G"].get("new_kernels", {}).get(name) or {}).get(ds) or {}).get("value")
                    if v:
                        out.append((v, set(), ds in BIG))
                    rb = best_block(R.get((ds, hg.NEW[name])))
                    if rb:
                        out.append((rb, {"V"}, False))
                    return out
                key = f"{m}/{k}"
                hr = (H["kernels"].get(key) or {}).get(ds) or {}
                cnt = hr.get("count") or 1
                held = hr.get("block") == "not run" or hr.get("status") == "running"
                if hr.get("H"):
                    out.append((hr["H"], set(), held))
                rb = best_block(R.get((ds, path_of.get(key))))
                if rb:
                    out.append((rb * cnt, {"V"}, False))
                return out

            def layer_opts(cell):
                layers = []
                for layer in LAYERS[m]:
                    opts = {}
                    for v, ks in layer.items():
                        if v == "o3" and ("G" not in cell or any(k.startswith("new:") and not hg.new_needed(k[4:], m, ds)[0] for k in ks)):
                            continue
                        picked = []
                        for k in ks:
                            c = [x for x in cands(k, v) if x[1] <= cell]
                            if not c:
                                break
                            picked.append((k,) + min(c, key=lambda x: x[0]))
                        else:
                            opts[v] = {"cycles": sum(p[1] for p in picked),
                                       "kernels": [f"{k} = {c:,} ({'+'.join(sorted(t)) or 'H'})" for k, c, t, _ in picked],
                                       "held": [k for k, _, _, h in picked if h]}
                    layers.append(opts)
                return layers

            fused = (H["configs"].get(f"{m}/fully_fused") or {}).get(ds) or {}
            for c, cell in CELLS.items():
                layers = layer_opts(cell)
                if "G" in cell:
                    plans = {"unfused": ("o2", "o3"), "partially_fused": ("o1", "o2", "o3")}
                else:
                    plans = {"unfused": ("o2",), "partially_fused": ("o1",)}
                res = {}
                for cfg, allowed in plans.items():
                    ch = [min(((v, o) for v, o in opts.items() if v in allowed), key=lambda x: x[1]["cycles"], default=(None, None))
                          for opts in layers]
                    ok = all(o for _, o in ch)
                    res[cfg] = {"cycles": sum(o["cycles"] for _, o in ch) if ok else None,
                                "H": (H["configs"].get(f"{m}/{cfg}") or {}).get(ds, {}).get("H"),
                                "S": (H["configs"].get(f"{m}/{cfg}") or {}).get(ds, {}).get("S"),
                                "layers": [{"variant": v, **(o or {})} for v, o in ch],
                                "held": [f"layer {i + 1} {k}" for i, (_, o) in enumerate(ch) if o for k in o["held"]]}
                res["fully_fused"] = {"cycles": fused.get("H"), "H": fused.get("H"), "S": fused.get("S"), "kernels": fused.get("kernels")}
                cf = OUT / "cells" / f"{NAME(cell).replace('+', '_')}.json"
                cf.parent.mkdir(parents=True, exist_ok=True)
                doc = json.loads(cf.read_text()) if cf.exists() else {"metadata": {"cell": NAME(cell)}, "configs": {}, "kernels": H["kernels"]}
                for cfg, rec in res.items():
                    doc["configs"].setdefault(f"{m}/{cfg}", {})[ds] = rec
                cf.write_text(json.dumps(doc))
                best = min((r["cycles"] for r in res.values() if r.get("cycles")), default=None)
                summary["best"].setdefault(m, {}).setdefault(ds, {})[NAME(cell)] = best
                # built-in check against the earlier experiments
                ref = {"-": "H", "G": "H_G"}.get(c)
                if ref:
                    for cfg in ("unfused", "partially_fused"):
                        want = ((D[ref]["configs"].get(f"{m}/{cfg}") or {}).get(ds) or {})
                        want = want.get("cycles", want.get("H"))
                        got = res[cfg]["cycles"]
                        if want and got and abs(want - got) > max(10, 1e-6 * want):
                            summary["checks"].append(f"{NAME(cell)} {m}/{ds}/{cfg}: assembled {got:,} but {ref} has {want:,}")
            b = summary["best"][m][ds]
            g = lambda s: (b["H"] / b[s]) if b.get("H") and b.get(s) else 0
            line = "  ".join(f"{s} {g(s):.2f}x" for s in ("H+G", "H+V", "H+G+V"))
            prod = lambda x, y: g(x) * g(y)
            print(f"== {ds} {m}: best H {b.get('H') or 0:,}  | gain over H: {line}  | G+V vs G*V {g('H+G+V') / max(prod('H+G', 'H+V'), 1e-9):.2f}  G+V vs max {g('H+G+V') / max(g('H+G'), g('H+V'), 1e-9):.2f}", flush=True)
            (OUT / "h_vec_summary.json").write_text(json.dumps(summary, indent=1))
    print("\nbuilt-in checks:", "all reproduced" if not summary["checks"] else "\n  " + "\n  ".join(summary["checks"]))
    (OUT / "h_vec_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
