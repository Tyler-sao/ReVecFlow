#!/usr/bin/env python3
"""
FuseFlow V3, H baseline = Figure 12 under tiling (decisions 2026-10-16: H is the main baseline; per kernel the best
of scalar and block 16/32/64; S stays as the published-FuseFlow column).

    cd /fuseflow-v3 && /fuseflow-v3/venv/bin/python -u /fuseflow-v3/scripts/h_run.py --datasets cora \
        [--models gcn graphsage sgc gin] [--blocks 16 32 64] [--resume] [--out /fuseflow-v3/h_baseline]

For every kernel of every Figure 12 configuration (from h0_inventory.json; GCN's and GraphSAGE's fully fused kernels
added from Figure 12's results) and dataset:
  - S: the trusted scalar cycles from the baselines (not rerun);
  - block: h_kernel.py at 16/32/64 on the same generated inputs, every output checked; a kernel without a reference is
    rerun with --paired (its scalar run on the same files is the check);
  - softmax kernels are run like every other kernel since Comal patch 0025 (block max reduction); the output check
    decides;
  - not blocked, with the reason recorded: on Collab and MAG, kernels reading the N x N adjacency (a stored tile of A holds about one nonzero but is written as
    n*n values - about 1.3 billion values on MAG at n = 16 - and its blocked cycles are estimated at about the scalar
    ones; measured on Cora, Cora-ML and DBLP instead);
  - H per kernel = the fastest of S and every block size whose output passed; H per configuration = the sum over its
    kernels, valid only if every kernel has a value.
One kernel file and dataset is run once, even when several configurations use it (same generator, same data).
Results: OUT/h_results.json and a table per model and dataset (S and H per configuration, and the best of each).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PY = "/fuseflow-v3/venv/bin/python"
KERNEL = "/fuseflow-v3/scripts/h_kernel.py"
INV = Path("/fuseflow-v3/h_baseline/h0/h0_inventory.json")
B = Path("/fuseflow-v3/baselines")
DATASETS = {"cora": ("Planetoid", "Cora"), "cora_ml": ("CitationFull", "cora_ml"), "dblp": ("CitationFull", "dblp"),
            "collab": ("ogbl", "ogbl-collab"), "mag": ("ogbn", "ogbn-mag")}
BIG = {"collab", "mag"}
FULLY = {"gcn": "gcn/gcn_sparse.mlir", "graphsage": "graphsage/graphsage_sparse.mlir"}


def kernels(models):
    inv = json.loads(INV.read_text())
    f12 = json.loads((B / "figure12_results.json").read_text())["results"]
    ks = {}
    for key, r in inv.items():
        if r["model"] in models and r.get("path"):
            ks[key] = r
    for m, f in FULLY.items():
        if m in models:
            cyc = {ds: (rec.get("fully_fused") or {}).get("cycles") if (rec.get("fully_fused") or {}).get("success") else None
                   for ds, rec in f12.get(m, {}).items()}
            ks[f"{m}/fully_fused/{Path(f).stem}"] = {"model": m, "config": "fully_fused", "kernel": Path(f).stem, "path": f,
                                                     "S_cycles": cyc, "tensors": {}, "class": "scalar-only"}
    return ks


def reads_adjacency(r):
    return any(t.get("shape") == ["1767", "1767"] for t in r.get("tensors", {}).values()) or "adj" in r["kernel"] \
        or r["config"] != "unfused"


def run(path, ds, rd, blocks, paired, resume):
    res = rd / "result.json"
    if resume and res.exists():
        r = json.loads(res.read_text())
        if r.get("status") in ("done",) or (r.get("status") == "unchecked" and not paired):
            return r
    if rd.exists():
        subprocess.run(["rm", "-rf", str(rd)], check=True)
    rd.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", KERNEL, "--mlir", path, "--dataset", *DATASETS[ds], "--out", str(rd), "--blocks", *map(str, blocks)]
    if paired:
        cmd.append("--paired")
    with open(f"{rd}.log", "w") as f:
        subprocess.run(cmd, cwd="/fuseflow-artifact", stdout=f, stderr=subprocess.STDOUT,
                       env={**os.environ, "DATA_PATH": "/tmp/data"})
    return json.loads(res.read_text()) if res.exists() else {"status": "error", "reason": f"no result.json, see {rd}.log"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora"], choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=["gcn", "graphsage", "sgc", "gin"])
    ap.add_argument("--blocks", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="/fuseflow-v3/h_baseline")
    ap.add_argument("--keep", default=None, help="regex on the kernel key (model/config/kernel): such kernels are not "
                    "run; their record already in h_results.json is kept (with --keep-reason added), else recorded not run")
    ap.add_argument("--keep-reason", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jf = out / "h_results.json"
    doc = json.loads(jf.read_text()) if jf.exists() else {"kernels": {}, "configs": {}}
    ks = kernels(args.models)
    # Figure 12 counts some kernels more than once in a configuration (GraphSAGE unfused: gcn_linear1_mul, gcn_linear1_bias,
    # gcn_linear2_mul, gcn_linear2_bias - the node's and the neighbours' linear layers - as one breakdown entry holding
    # exactly twice the cycles of the same file elsewhere). A blocked run is one execution, so it is multiplied by the
    # same count: count = S of this occurrence / the smallest S of the same file on the same dataset (an integer, else 1)
    # (matched by file name; only where every copy of that file is byte-identical, per the inventory's md5s)
    fkey = lambda r: Path(r["path"]).name if len(set((r.get("copies") or {"x": 0}).values())) == 1 else r["path"]
    single = {}
    for r in ks.values():
        for d, c in (r.get("S_cycles") or {}).items():
            if c:
                single[(fkey(r), d)] = min(c, single.get((fkey(r), d), c))
    done = {}
    for ds in args.datasets:
        for key, r in sorted(ks.items()):
            m, cfg, k = key.split("/")
            s = (r.get("S_cycles") or {}).get(ds)
            rec = {"S": s}
            old = (doc["kernels"].get(key) or {}).get(ds)
            if args.keep and re.search(args.keep, key):
                rec = dict(old) if old else {"S": s, "block": "not run"}
                rec["S"] = s
                rec["reason"] = args.keep_reason or rec.get("reason")
            elif r["path"] in done.get(ds, {}):
                rec.update(done[ds][r["path"]])
            elif ds in BIG and reads_adjacency(r):
                rec.update(block="not run", reason="reads the N x N adjacency: blocked layout of A not practical at this "
                                                   "size, blocked cycles estimated at about scalar (measured on Cora, Cora-ML, DBLP)")
            else:
                rd = out / "runs" / ds / r["path"].replace("/", "__").replace(".mlir", "")
                t = time.time()
                res = run(r["path"], ds, rd, args.blocks, False, args.resume)
                if res.get("status") == "unchecked":
                    res = run(r["path"], ds, rd, args.blocks, True, False)
                best = {n: b["cycles"] for n, b in res.get("blocks", {}).items() if b.get("status") == "ok"}
                rec.update(block={n: {"status": b.get("status"), "cycles": b.get("cycles"),
                                      "failing": (b.get("check") or {}).get("failing"), "reason": b.get("reason"),
                                      "baseline": (b.get("check") or {}).get("baseline")}
                                  for n, b in res.get("blocks", {}).items()},
                           check=res.get("check"), status=res.get("status"), reason=res.get("reason"),
                           paired_scalar=res.get("paired_scalar"), seconds=round(time.time() - t, 1), best_block=best)
                done.setdefault(ds, {})[r["path"]] = {x: rec[x] for x in rec if x != "S"}
            bb = rec.get("best_block") or {}
            mult = 1
            if s and single.get((fkey(r), ds)):
                q = s / single[(fkey(r), ds)]
                if round(q) >= 2 and abs(q - round(q)) < 1e-3:
                    mult = round(q)
            rec["count"] = mult
            cand = {"scalar": s} | {f"block{n}": c * mult for n, c in bb.items()}
            cand = {a: c for a, c in cand.items() if c}
            rec["H"] = min(cand.values()) if cand else None
            rec["H_choice"] = min(cand, key=cand.get) if cand else None
            doc["kernels"].setdefault(key, {})[ds] = rec
            jf.write_text(json.dumps(doc, indent=1))
            print(f"{ds:7} {key:55} S {s or 0:>14,}  H {rec['H'] or 0:>14,} ({rec['H_choice']})  "
                  f"[{rec.get('status') or rec.get('block')}] blocks {json.dumps({n: (b.get('cycles'), b.get('status')) for n, b in (rec.get('block') if isinstance(rec.get('block'), dict) else {}).items()})} "
                  f"{rec.get('reason') or ''} {rec.get('check') or ''}"
                  + "".join(f"  [block{n}: {x['baseline']['blocked_failing']} element(s) accepted, also failing for FuseFlow's scalar]"
                            for n, x in (rec.get('block') if isinstance(rec.get('block'), dict) else {}).items()
                            if (x.get('baseline') or {}).get('accepted')), flush=True)
        # per configuration totals
        for m in args.models:
            for cfg in ("unfused", "partially_fused", "fully_fused"):
                members = [k for k in ks if k.startswith(f"{m}/{cfg}/")]
                if not members:
                    continue
                vals = [doc["kernels"][k][ds] for k in members]
                tot = lambda key: sum(v[key] for v in vals) if all(v.get(key) for v in vals) else None
                doc["configs"].setdefault(f"{m}/{cfg}", {})[ds] = {"S": tot("S"), "H": tot("H"),
                                                                  "kernels": {k.split("/")[2]: (v.get("S"), v.get("H"), v.get("H_choice")) for k, v in zip(members, vals)}}
            per = {c.split("/")[1]: v[ds] for c, v in doc["configs"].items() if c.startswith(f"{m}/") and ds in v}
            sb = {c: v["S"] for c, v in per.items() if v["S"]}
            hb = {c: v["H"] for c, v in per.items() if v["H"]}
            line = "  ".join(f"{c}: S {v['S'] or 0:,} H {v['H'] or 0:,}" for c, v in per.items())
            print(f"== {ds} {m}: {line}  | best S {min(sb, key=sb.get) if sb else '-'}  best H {min(hb, key=hb.get) if hb else '-'} "
                  f"{(min(sb.values()) / min(hb.values())) if sb and hb else 0:.2f}x", flush=True)
        jf.write_text(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
