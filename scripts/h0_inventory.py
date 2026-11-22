#!/usr/bin/env python3
"""
FuseFlow V3, H0 (read-only): inventory of every Figure 12 kernel the H baseline has to run - "Figure 12 under tiling".

    cd /fuseflow-v3 && /fuseflow-v3/venv/bin/python -u /fuseflow-v3/scripts/h0_inventory.py [--out /fuseflow-v3/h_baseline/h0]

For each model (GCN, GraphSAGE, SGC, GIN, SAE, GPT-3) and Figure 12 configuration (unfused, partially fused, fully fused;
SGC's partially fused = partial_p1, decision 2026-10-13), from the read-only baseline copies in /fuseflow-v3/baselines:
  - the kernels of the configuration (the breakdown's files) and their scalar cycles per dataset (S, trusted, not rerun);
  - where the kernel file is in this container (every copy, and whether the copies are identical);
  - compiled with the V3 compiler (sam-opt only, no simulation): its header tensors (rank, formats, shape, argument);
  - the operations in the MLIR: matmuls, element-wise bodies, reductions (iterator "reduction", or max/exp/log/div);
  - a class: "block-candidate" (products and element-wise only: true-block mode computes the same function) or
    "scalar-only" (a reduction over a row - softmax, row max/sum - which block mode computes wrongly); and whether a
    1-D tensor (bias, BatchNorm vector) is read, whose block layout is untested;
  - whether an independent reference exists (newmodels_kernel.reference) to check a blocked output, or a paired scalar
    run is needed.
Writes OUT/h0_inventory.json and prints one line per kernel. Nothing is simulated or changed.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/fuseflow-artifact/scripts")
from v3tools import Failed, compile_candidate, get_tensors_properties  # noqa: E402

M = Path("/fuseflow-artifact/fuseflow-compiler/tests/models")
B = Path("/fuseflow-v3/baselines")
CONFIG = {"gcn": ("figure12", {"unfused": "unfused", "partially_fused": "partially_fused", "fully_fused": "fully_fused"}),
          "graphsage": ("figure12", {"unfused": "unfused", "partially_fused": "partially_fused", "fully_fused": "fully_fused"}),
          "sae": ("figure12", {"unfused": "unfused", "partially_fused": "partially_fused", "fully_fused": "fully_fused"}),
          "gpt-3": ("figure12", {"unfused": "unfused", "partially_fused": "partially_fused", "fully_fused": "fully_fused"}),
          "sgc": ("newmodels", {"unfused": "unfused", "partially_fused": "partial_p1", "fully_fused": "fully_fused"}),
          "gin": ("newmodels", {"unfused": "unfused", "partially_fused": "partially_fused", "fully_fused": "fully_fused"})}
REDUCE_OPS = ("arith.maximumf", "math.exp", "math.log", "arith.divf")


def find(file, model):
    name = Path(file).name
    hits = [h for h in sorted(M.glob(f"**/{name}")) if h.parent.name not in ("csr_layers",)]
    if "/" in file:
        hits = [h for h in hits if str(h).endswith(file)] or hits
    pref = [h for d in (f"{model}_partially_fused", f"{model}_unfused", model) for h in hits if h.parent.name == d]
    chosen = (pref or hits or [None])[0]
    md5 = {str(h.relative_to(M)): hashlib.md5(h.read_bytes()).hexdigest()[:8] for h in hits}
    return chosen, md5


def ops(text):
    body = text
    return {"matmul": body.count("linalg.matmul"), "batch_matmul": body.count("linalg.batch_matmul"),
            "generic": body.count("linalg.generic"), "reduction_iterators": body.count('"reduction"'),
            **{o.split(".")[1]: body.count(o) for o in REDUCE_OPS + ("arith.addf", "arith.mulf", "arith.subf", "arith.cmpf", "math.rsqrt")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/fuseflow-v3/h_baseline/h0")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f12 = json.loads((B / "figure12_results.json").read_text())["results"]
    new = json.loads((B / "figure12_newmodels_results.json").read_text())["results"]
    print("models in figure12_results.json:", sorted(f12), "| in figure12_newmodels_results.json:", sorted(new), flush=True)
    sys.argv = [sys.argv[0]]
    import newmodels_kernel as nk
    inv = {}
    for model, (src, cfgs) in CONFIG.items():
        res = (f12 if src == "figure12" else new).get(model, {})
        for cfg, key in cfgs.items():
            kernels = {}
            for ds, r in res.items():
                rec = (r or {}).get(key) or {}
                for e in rec.get("breakdown", []):
                    k = e.get("file") or e.get("kernel")
                    kernels.setdefault(k, {})[ds] = e.get("cycles") if e.get("success", e.get("valid", True)) else None
            for k, cyc in kernels.items():
                file = k if k.endswith(".mlir") else k + ".mlir"
                path, copies = find(file, model)
                rec = {"model": model, "config": cfg, "kernel": Path(file).stem, "S_cycles": cyc, "copies": copies,
                       "path": str(path.relative_to(M)) if path else None}
                if path is None:
                    rec["error"] = "kernel file not found"
                else:
                    text = path.read_text()
                    o = ops(text)
                    rec["ops"] = {a: b for a, b in o.items() if b}
                    d = out / "compiled" / model / cfg / Path(file).stem
                    d.mkdir(parents=True, exist_ok=True)
                    try:
                        compile_candidate({"model": str(path), "stream_level": 1, "par_factor": 1}, d)
                        hdr = get_tensors_properties(d / "sam.mlir")
                        rec["tensors"] = {t: {"rank": len(h["format"]), "format": h["format"], "shape": h["shape"], "arg": h.get("Arg")}
                                          for t, h in hdr.items()}
                    except Failed as e:
                        rec["error"] = f"{e.status}: {e.reason}"[:300]
                    reduce = o["reduction_iterators"] > 0 or any(o[x.split(".")[1]] for x in REDUCE_OPS)
                    rec["class"] = "scalar-only" if reduce else "block-candidate"
                    rec["reads_1d"] = any(t["rank"] == 1 for t in rec.get("tensors", {}).values())
                    rec["reads_3d_or_more"] = any(t["rank"] > 2 for t in rec.get("tensors", {}).values())
                    try:
                        nk.reference(Path(file).stem, {}, [], [], float)
                        rec["reference"] = "yes"
                    except nk.NoReference:
                        rec["reference"] = "paired scalar run needed"
                    except ValueError as e:
                        rec["reference"] = "paired scalar run needed" if str(e).startswith("no reference") else "yes"
                    except Exception:
                        rec["reference"] = "yes"
                inv[f"{model}/{cfg}/{Path(file).stem}"] = rec
                print(f"{model:9} {cfg:16} {Path(file).stem:34} {rec.get('class', '-'):16} 1-D {str(rec.get('reads_1d', '-')):5} "
                      f"ref {rec.get('reference', '-'):25} ops {rec.get('ops')} S {cyc} {rec.get('error', '')}", flush=True)
    (out / "h0_inventory.json").write_text(json.dumps(inv, indent=1))
    n = len(inv)
    print(f"\n{n} kernel entries; block-candidates {sum(r.get('class') == 'block-candidate' for r in inv.values())}, "
          f"scalar-only {sum(r.get('class') == 'scalar-only' for r in inv.values())}, "
          f"needing a paired scalar run {sum(r.get('reference') == 'paired scalar run needed' and r.get('class') == 'block-candidate' for r in inv.values())}, "
          f"errors {sum('error' in r for r in inv.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
