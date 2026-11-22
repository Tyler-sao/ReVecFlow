#!/usr/bin/env python3
"""
FuseFlow V3, H_G = H + graph-level choices (re-association and per-layer kernel boundaries), four GNN models; built on
H (not on H_CSR), so that its contribution is separate; H_V3 combines all.

    cd /fuseflow-v3 && /fuseflow-v3/venv/bin/python -u /fuseflow-v3/scripts/h_g_run.py \
        [--datasets cora cora_ml dblp collab mag] [--models gcn graphsage sgc gin] [--blocks 16 32 64] [--resume]

Per layer, three graph variants (the layers may choose differently - per-layer kernel boundaries):
  o1 fused        FuseFlow's per-layer kernel (partially fused configuration)              H values
  o2 unfused      FuseFlow's unfused kernels, original order A.X then .W                  H values
  o3 re-assoc.    unfused, X.W first then A.(XW): the aggregation at the narrower width    H values of kernels of the
                  same shape, plus two new kernels (make_hg_kernels.py):
    GCN, GraphSAGE  layer 1: gcn_adj_x1 (A . N x F) -> the value of gcn_adj_x2 (A . N x hidden; same A, same shape)
                    layer 2: gcn_adj_x2 (A . N x hidden) -> hg_adj_c (A . N x C), only where C < hidden
    GIN             layer 1: gin_adj_h1 -> gin_adj_h2 (A . N x 64), gin_selfadd1 -> gin_selfadd2 (width 64)
                    ((1+eps)X + A X) W1 = (1+eps) X W1 + A (X W1); layer 2 keeps its width (64 -> 64): no o3
    SGC             S.S.X.W -> X.W first, then hg_propnorm_c twice (S . N x C), only where C < F
  FuseFlow's unfused methodology times every kernel on generator inputs of its shape (random dense intermediates); the
  re-associated graphs are timed the same way. Each kernel's output is checked (H's checks; the new kernels against a
  float64 matmul reference); the end-to-end output of a re-associated graph differs from the original order by float32
  rounding only (Stage D: 200 of 3.8M elements beyond 1e-4 on Collab GCN layer 1, reported there).
New kernels: scalar (paired run: its cycles, output checked) and blocks on Cora, Cora-ML, DBLP; scalar only on Collab
(they read A: H's rule); not run on MAG (C = 349 is not narrower than the widths they would replace).
Configurations written (same schema as h_results.json, "cycles" = H_G):
  unfused          the unfused graph, per layer the cheaper of o2 and o3 (re-association alone)
  partially_fused  per layer the cheapest of o1, o2, o3 (re-association + per-layer kernel boundaries) = H_G's choice
  fully_fused      H's value (one kernel for the whole model; no graph choice inside it)
Results: /fuseflow-v3/results/h_g/h_g_results.json, runs in /fuseflow-v3/results/h_g/runs/<dataset>/.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PY = "/fuseflow-v3/venv/bin/python"
KERNEL = "/fuseflow-v3/scripts/h_kernel.py"
MODELS = Path("/fuseflow-artifact/fuseflow-compiler/tests/models")
H_RESULTS = Path("/fuseflow-v3/h_baseline/h_results.json")
OUT = Path("/fuseflow-v3/results/h_g")
DATASETS = {"cora": ("Planetoid", "Cora"), "cora_ml": ("CitationFull", "cora_ml"), "dblp": ("CitationFull", "dblp"),
            "collab": ("ogbl", "ogbl-collab"), "mag": ("ogbn", "ogbn-mag")}
BIG = {"collab", "mag"}
WIDTH = {"cora": (1433, 7), "cora_ml": (2879, 7), "dblp": (1639, 4), "collab": (128, 2), "mag": (128, 349)}  # (F, C)
HIDDEN = {"gcn": 16, "graphsage": 16, "gin": 64}
NEW = {"hg_adj_c": "hg/hg_adj_c.mlir", "hg_propnorm_c": "hg/hg_propnorm_c.mlir"}
# per model: layers, each {variant: [kernel, ...]}; a kernel is "<config>/<stem>" (an H value) or "new:<name>"
LAYERS = {
    "gcn": [
        {"o1": ["partially_fused/one_layer_gcn"],
         "o2": ["unfused/gcn_adj_x1", "unfused/gcn_linear1_mul", "unfused/gcn_linear1_bias", "unfused/gcn_relu"],
         "o3": ["unfused/gcn_linear1_mul", "unfused/gcn_adj_x2", "unfused/gcn_linear1_bias", "unfused/gcn_relu"]},
        {"o1": ["partially_fused/one_layer_gcn2"],
         "o2": ["unfused/gcn_adj_x2", "unfused/gcn_linear2_mul", "unfused/gcn_linear2_bias", "unfused/gcn_softmax"],
         "o3": ["unfused/gcn_linear2_mul", "new:hg_adj_c", "unfused/gcn_linear2_bias", "unfused/gcn_softmax"]}],
    "graphsage": [
        {"o1": ["partially_fused/one_layer_graphsage"],
         "o2": ["unfused/gcn_adj_x1", "unfused/gcn_linear1_mul", "unfused/gcn_linear1_bias", "unfused/graphsage_linear_adds",
                "unfused/gcn_relu"],
         "o3": ["unfused/gcn_linear1_mul", "unfused/gcn_adj_x2", "unfused/gcn_linear1_bias", "unfused/graphsage_linear_adds",
                "unfused/gcn_relu"]},
        {"o1": ["partially_fused/one_layer_graphsage2"],
         "o2": ["unfused/gcn_adj_x2", "unfused/gcn_linear2_mul", "unfused/gcn_linear2_bias"],
         "o3": ["unfused/gcn_linear2_mul", "new:hg_adj_c", "unfused/gcn_linear2_bias"]}],
    "sgc": [
        {"o1": ["partially_fused/sgc_prop1", "partially_fused/sgc_prop_classify"],
         "o2": ["unfused/sgc_prop1", "unfused/sgc_prop2", "unfused/sgc_linear_mul", "unfused/sgc_linear_bias", "unfused/sgc_softmax"],
         "o3": ["unfused/sgc_linear_mul", "new:hg_propnorm_c", "new:hg_propnorm_c", "unfused/sgc_linear_bias", "unfused/sgc_softmax"]}],
    "gin": [
        {"o1": ["partially_fused/one_layer_gin"],
         "o2": ["unfused/gin_adj_h1", "unfused/gin_selfadd1", "unfused/gin_mlp1_lin1_mul", "unfused/gin_mlp1_lin1_bias",
                "unfused/gin_mlp1_bn1", "unfused/gin_mlp1_relu1", "unfused/gin_mlp1_lin2_mul", "unfused/gin_mlp1_lin2_bias",
                "unfused/gin_bn_out1", "unfused/gin_relu_out1"],
         "o3": ["unfused/gin_mlp1_lin1_mul", "unfused/gin_adj_h2", "unfused/gin_selfadd2", "unfused/gin_mlp1_lin1_bias",
                "unfused/gin_mlp1_bn1", "unfused/gin_mlp1_relu1", "unfused/gin_mlp1_lin2_mul", "unfused/gin_mlp1_lin2_bias",
                "unfused/gin_bn_out1", "unfused/gin_relu_out1"]},
        {"o1": ["partially_fused/one_layer_gin2"],
         "o2": ["unfused/gin_adj_h2", "unfused/gin_selfadd2", "unfused/gin_mlp2_lin1_mul", "unfused/gin_mlp2_lin1_bias",
                "unfused/gin_mlp2_bn1", "unfused/gin_mlp2_relu1", "unfused/gin_mlp2_lin2_mul", "unfused/gin_mlp2_lin2_bias",
                "unfused/gin_softmax"]}],
}


def new_needed(name, model, ds):
    F, C = WIDTH[ds]
    if name == "hg_adj_c":
        return C < HIDDEN[model], f"C = {C} not below the hidden width {HIDDEN[model]}"
    return C < F, f"C = {C} not below F = {F}"


def run_new(name, ds, blocks, resume):
    rd = OUT / "runs" / ds / name
    res = rd / "result.json"
    if resume and res.exists() and json.loads(res.read_text()).get("status") == "done":
        return json.loads(res.read_text())
    if rd.exists():
        subprocess.run(["rm", "-rf", str(rd)], check=True)
    rd.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", KERNEL, "--mlir", NEW[name], "--dataset", *DATASETS[ds], "--out", str(rd)]
    cmd += ["--scalar-only"] if ds in BIG else ["--paired", "--blocks", *map(str, blocks)]
    with open(f"{rd}.log", "w") as f:
        subprocess.run(cmd, cwd="/fuseflow-artifact", stdout=f, stderr=subprocess.STDOUT,
                       env={**os.environ, "DATA_PATH": "/tmp/data"})
    return json.loads(res.read_text()) if res.exists() else {"status": "error", "reason": f"no result.json, see {rd}.log"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=list(LAYERS), choices=list(LAYERS))
    ap.add_argument("--blocks", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    missing = [p for p in NEW.values() if not (MODELS / p).exists()]
    if missing:
        sys.exit(f"STOP: run make_hg_kernels.py first ({missing} missing)")
    H = json.loads(H_RESULTS.read_text())
    # every unfused / partially fused kernel of H must be placed in exactly the layers above
    for m in args.models:
        for cfg in ("unfused", "partially_fused"):
            listed = {k.split("/")[1] for layer in LAYERS[m] for v in ("o1", "o2") for k in layer.get(v, []) if k.startswith(cfg + "/")}
            have = {k.split("/")[2] for k in H["kernels"] if k.startswith(f"{m}/{cfg}/")}
            if listed != have:
                sys.exit(f"STOP {m}/{cfg}: layer lists {sorted(listed)} differ from H's kernels {sorted(have)}")
    OUT.mkdir(parents=True, exist_ok=True)
    jf = OUT / "h_g_results.json"
    doc = {"metadata": {"variant": "H_G", "h_results": str(H_RESULTS),
                        "rule": "per layer the cheapest of FuseFlow's fused kernel (o1), its unfused kernels (o2) and the "
                                "re-associated unfused graph (o3); built on H"},
           "kernels": json.loads(json.dumps(H["kernels"])), "configs": {}, "new_kernels": {}, "choices": {}}
    for ds in args.datasets:
        new_val = {}
        for name in NEW:
            users = [m for m in args.models if any(f"new:{name}" in layer.get("o3", []) for layer in LAYERS[m])]
            need = [m for m in users if new_needed(name, m, ds)[0]]
            if not need:
                doc["new_kernels"].setdefault(name, {})[ds] = {"status": "not run",
                                                               "reason": "; ".join(sorted({new_needed(name, m, ds)[1] for m in users})) or "no model uses it"}
                continue
            r = run_new(name, ds, args.blocks, args.resume)
            ps = r.get("paired_scalar") or {}
            scalar_ok = ps.get("passed") and (ps.get("vs_ref") or {}).get("failing") == 0
            cand = ({"scalar": ps.get("cycles")} if scalar_ok else {}) | \
                   {f"block{n}": b.get("cycles") for n, b in r.get("blocks", {}).items() if b.get("status") == "ok"}
            cand = {a: c for a, c in cand.items() if c}
            choice = min(cand, key=cand.get) if cand else None
            new_val[name] = cand.get(choice)
            held = ds in BIG
            doc["new_kernels"].setdefault(name, {})[ds] = {
                "status": r.get("status"), "reason": r.get("reason"), "check": r.get("check"), "paired_scalar": ps,
                "blocks": {n: {"status": b.get("status"), "cycles": b.get("cycles"), "failing": (b.get("check") or {}).get("failing")}
                           for n, b in r.get("blocks", {}).items()},
                "value": new_val[name], "choice": choice, "peak_rss_gb": r.get("peak_rss_gb"),
                "held_scalar": "reads the N x N adjacency: scalar only on Collab/MAG (H's rule)" if held else None}
            print(f"{ds:7} new {name:14} scalar {ps.get('cycles') or 0:>14,} ({'ok' if scalar_ok else 'not used'})  "
                  f"blocks {json.dumps({n: (b.get('cycles'), b.get('status')) for n, b in r.get('blocks', {}).items()})}  "
                  f"-> {new_val[name] or 0:,} ({choice})", flush=True)
        for m in args.models:
            def val(k):
                """(cycles, held) of one kernel in a variant; held = counted at scalar cost by a harness rule."""
                if k.startswith("new:"):
                    v = new_val.get(k[4:])
                    return v, ds in BIG
                kr = (doc["kernels"].get(f"{m}/{k}") or {}).get(ds) or {}
                return kr.get("H"), kr.get("block") == "not run" or kr.get("status") == "running"

            layers = []
            for li, layer in enumerate(LAYERS[m]):
                opts = {}
                for v, ks in layer.items():
                    if v == "o3" and any(k.startswith("new:") and not new_needed(k[4:], m, ds)[0] for k in ks):
                        continue
                    vals = [val(k) for k in ks]
                    if all(c for c, _ in vals):
                        opts[v] = {"cycles": sum(c for c, _ in vals), "kernels": ks,
                                   "held": [k for k, (_, h) in zip(ks, vals) if h]}
                layers.append(opts)
            pick = lambda allowed: [min(((v, o) for v, o in opts.items() if v in allowed), key=lambda x: x[1]["cycles"], default=(None, None))
                                    for opts in layers]
            fused = (H["configs"].get(f"{m}/fully_fused") or {}).get(ds) or {}
            out = {}
            for cfg, allowed in (("unfused", ("o2", "o3")), ("partially_fused", ("o1", "o2", "o3"))):
                ch = pick(allowed)
                ok = all(o for _, o in ch)
                out[cfg] = {"cycles": sum(o["cycles"] for _, o in ch) if ok else None,
                            "H": (H["configs"].get(f"{m}/{cfg}") or {}).get(ds, {}).get("H"),
                            "S": (H["configs"].get(f"{m}/{cfg}") or {}).get(ds, {}).get("S"),
                            "layers": [{"variant": v, **(o or {})} for v, o in ch],
                            "held": [f"layer {i + 1} {k}" for i, (_, o) in enumerate(ch) if o for k in o["held"]]}
            out["fully_fused"] = {"cycles": fused.get("H"), "H": fused.get("H"), "S": fused.get("S"),
                                  "kernels": fused.get("kernels")}
            for cfg, rec in out.items():
                doc["configs"].setdefault(f"{m}/{cfg}", {})[ds] = rec
            doc["choices"].setdefault(m, {})[ds] = {f"layer {i + 1}": {v: o["cycles"] for v, o in opts.items()} for i, opts in enumerate(layers)}
            jf.write_text(json.dumps(doc, indent=1))
            hb = [(c, r.get("H")) for c, r in out.items() if r.get("H")]
            gb = [(c, r.get("cycles")) for c, r in out.items() if r.get("cycles")]
            bh, bg = min(hb, key=lambda x: x[1], default=None), min(gb, key=lambda x: x[1], default=None)
            sb = [r.get("S") for r in out.values() if r.get("S")]
            print(f"== {ds} {m}: " + "  ".join(f"{c}: H {r.get('H') or 0:,} H_G {r.get('cycles') or 0:,}" for c, r in out.items())
                  + f"  | layers {[(v, o['cycles'] if o else None) for v, o in pick(('o1', 'o2', 'o3'))]}"
                  + f"  | H_G vs H {(bh[1] / bg[1]) if bh and bg else 0:.3f}x  vs S {(min(sb) / bg[1]) if sb and bg else 0:.2f}x", flush=True)
    jf.write_text(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
