"""
GALA-dagger: GALA (Lenadora et al., OOPSLA'25) ported as-is onto the common substrate H (FuseFlow + tiling, Comal cycles).
Decision 2026-10-27: strict port - GALA's own transformations with GALA's own decision rules, nothing of ours,
no cycle-based (oracle) choice.

    /fuseflow-v3/venv/bin/python -u /fuseflow-v3/gala/scripts/gala_assemble.py [--datasets ...] [--models ...]

What is ported (GALA paper section in brackets):
  - complexity-based operator reordering [6.2.4]: per layer, the update (X.W) is moved before the aggregation (A.X) when
    that lowers the aggregation's width, i.e. when the layer's output width is below its input width; decided from the
    dimensions only. On our layer table (h_g_run.LAYERS) this is variant o3 (re-associated, unfused) if out < in, else o2
    (FuseFlow's unfused order). Widths: h_g_run.WIDTH (F, C) and HIDDEN; GIN's first MLP width 64.
  - sparsity-aware rewrite [6.2.2]: GALA keeps the precomputed weighted (normalized) adjacency for comparatively sparse
    graphs, which is what FuseFlow's kernels already read; all five graphs are sparse,
    so the rewrite leaves the program unchanged (GALA decides by density E/N^2; ours are all below 1e-2). Recorded only.
  - intra-operator schedules [5.2]: col_tile -> the substrate's tiling (H's per-kernel best block size, shared by every
    method); thread coarsening is a GPU-thread transformation with no counterpart on SAM dataflow -> not ported.
  - not applicable: training-specific transformations (inference only), unweighted-graph kernels (A is weighted),
    sampling (not semantically equivalent).
  - no operator fusion (GALA has none, section 10): GALA-dagger has the unfused configuration only.
Kernel values: H's per-kernel H (h_results.json) and H_G's new kernels (o3's hg_adj_c / hg_propnorm_c), i.e. exactly the
values the cell H+G uses; nothing is simulated.
Diagnostics (not GALA): the cycle-best of o2/o3 per layer (to show where GALA's rule leaves cycles), and built-in
checks that the replicated layer sums reproduce H (o2 everywhere) and the cell H+G (unfused).
Outputs: /fuseflow-v3/gala/results/gala_results.json and cells/GALA.json (plot schema).
"""
import argparse
import importlib.util
import json
from pathlib import Path

V3 = Path("/fuseflow-v3/scripts")
H_FILE = Path("/fuseflow-v3/h_baseline/h_results.json")
HG_FILE = Path("/fuseflow-v3/results/h_g/h_g_results.json")
CELLS = Path("/fuseflow-v3/results/h_vec/cells")
OUT = Path("/fuseflow-v3/gala/results")
GIN_MLP = 64          # width of GIN's first MLP linear (gin_mlp1_lin1_mul: F -> 64)


def load(name):
    spec = importlib.util.spec_from_file_location(name, V3 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def widths(hg, m, ds, layer):
    """(input width, output width) of a layer's linear update"""
    F, C = hg.WIDTH[ds]
    if m == "sgc":
        return F, C
    hid = GIN_MLP if m == "gin" else hg.HIDDEN[m]
    return (F, hid) if layer == 0 else (hid, C)


def cell_best(name, m, ds):
    f = CELLS / f"{name}.json"
    if not f.exists():
        return None
    cfgs = json.loads(f.read_text()).get("configs", {})
    return min((c[ds]["cycles"] for k, c in cfgs.items() if k.startswith(f"{m}/") and (c.get(ds) or {}).get("cycles")),
               default=None)


def main():
    hg = load("h_g_run")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(hg.DATASETS))
    ap.add_argument("--models", nargs="+", default=list(hg.LAYERS))
    a = ap.parse_args()
    H = json.loads(H_FILE.read_text())
    HG = json.loads(HG_FILE.read_text())
    doc = {"metadata": {"definition": __doc__}, "models": {}, "checks": []}
    cell = {"metadata": {"cell": "GALA-dagger (strict port, unfused only)"}, "configs": {}, "kernels": {}}

    for m in a.models:
        for ds in a.datasets:
            def value(k):
                if k.startswith("new:"):
                    return ((HG.get("new_kernels", {}).get(k[4:]) or {}).get(ds) or {}).get("value")
                return ((H["kernels"].get(f"{m}/{k}") or {}).get(ds) or {}).get("H")

            layers, missing = [], []
            for i, layer in enumerate(hg.LAYERS[m]):
                opts = {}
                for v, ks in layer.items():
                    if v == "o1":
                        continue                       # fused: GALA has no operator fusion
                    if v == "o3" and any(k.startswith("new:") and not hg.new_needed(k[4:], m, ds)[0] for k in ks):
                        continue                       # re-association not applicable here (H_G's rule)
                    vals = [value(k) for k in ks]
                    if all(vals):
                        opts[v] = sum(vals)
                    else:
                        missing.append(f"layer {i + 1} {v}: {[k for k, x in zip(ks, vals) if not x]}")
                w_in, w_out = widths(hg, m, ds, i)
                rule = "o3" if (w_out < w_in and "o3" in opts) else "o2"
                if rule not in opts:
                    rule = None
                best = min(opts, key=opts.get) if opts else None
                layers.append({"layer": i + 1, "widths": [w_in, w_out], "options": opts, "gala_choice": rule,
                               "cycle_best": best, "gala_cycles": opts.get(rule) if rule else None,
                               "regret": (opts[rule] - opts[best]) if rule and best else None})
            gala = sum(l["gala_cycles"] for l in layers) if all(l["gala_cycles"] for l in layers) else None
            oracle = sum(l["options"][l["cycle_best"]] for l in layers) if all(l["cycle_best"] for l in layers) else None
            o2 = sum(l["options"]["o2"] for l in layers) if all("o2" in l["options"] for l in layers) else None
            h_unf = ((H["configs"].get(f"{m}/unfused") or {}).get(ds) or {}).get("H")
            hcfg = {c.split("/")[1]: (v.get(ds) or {}).get("H") for c, v in H["configs"].items() if c.startswith(f"{m}/")}
            h_best = min((x for x in hcfg.values() if x), default=None)
            hg_cell = json.loads((CELLS / "H_G.json").read_text()) if (CELLS / "H_G.json").exists() else {}
            hg_unf = ((hg_cell.get("configs", {}).get(f"{m}/unfused") or {}).get(ds) or {}).get("cycles")
            # built-in checks: the replicated sums reproduce H (o2 everywhere) and the cell H+G (unfused = best of o2/o3)
            for what, got, want in (("o2 everywhere vs H unfused", o2, h_unf), ("best o2/o3 vs cell H+G unfused", oracle, hg_unf)):
                if got and want and abs(got - want) > max(10, 1e-6 * want):
                    doc["checks"].append(f"{m}/{ds} {what}: {got:,} vs {want:,}")
            rec = {"GALA": gala, "layers": layers, "missing": missing, "H_unfused": h_unf, "H_best": h_best,
                   "unfused_cycle_best_o2_o3": oracle, "H+G": cell_best("H_G", m, ds), "H+G+V": cell_best("H_G_V", m, ds),
                   "sparsity_rewrite": "keeps the precomputed weighted adjacency (sparse graph): no change"}
            doc["models"].setdefault(m, {})[ds] = rec
            cell["configs"].setdefault(f"{m}/unfused", {})[ds] = {"cycles": gala, "layers": layers}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gala_results.json").write_text(json.dumps(doc, indent=1) + "\n")
    (OUT / "cells").mkdir(exist_ok=True)
    (OUT / "cells" / "GALA.json").write_text(json.dumps(cell, indent=1) + "\n")

    f = lambda v: f"{v:,}".rjust(15) if isinstance(v, int) else str(v).rjust(15)
    r_ = lambda x, y: f"{x / y:7.2f}x" if x and y else "    -   "
    print(f"{'model':9} {'dataset':8} {'H best':>15} {'GALA':>15} {'H+G':>15} {'H+G+V':>15}   GALA vs H best | "
          f"H+G+V vs GALA | GALA choices (rule / cycle-best)")
    for m, per in doc["models"].items():
        for ds, r in per.items():
            ch = " ".join(f"L{l['layer']}:{l['gala_choice']}/{l['cycle_best']}" + ("*" if l["gala_choice"] != l["cycle_best"] else "")
                          for l in r["layers"])
            print(f"{m:9} {ds:8} {f(r['H_best'])} {f(r['GALA'])} {f(r['H+G'])} {f(r['H+G+V'])}   "
                  f"{r_(r['H_best'], r['GALA'])}     | {r_(r['GALA'], r['H+G+V'])}      | {ch}"
                  + (f"   MISSING {r['missing']}" if r["missing"] else ""))
    print("\n* = GALA's dimension rule picks a different variant than the cycle-best one")
    print("built-in checks:", "all reproduced" if not doc["checks"] else "\n  " + "\n  ".join(doc["checks"]))
    print(f"written {OUT / 'gala_results.json'}, {OUT / 'cells' / 'GALA.json'}")


if __name__ == "__main__":
    main()
