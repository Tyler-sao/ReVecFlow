"""
GRANII-dagger: GRANII (Lenadora et al., CGO'26) ported as-is onto the common substrate H (FuseFlow + tiling, Comal
cycles). Decisions 2026-10-27: strict port, tiled comparison, Collab/MAG adjacency kernels scalar as in H.

    /fuseflow-v3/venv/bin/python -u /fuseflow-v3/granii/scripts/granii_assemble.py [--datasets ...] [--models ...]

GRANII: enumerate the sparse/dense primitive compositions of every layer obtained by matrix re-association (its
association trees), and select, per layer, the composition with the smallest SUM of per-primitive costs. Non-linear
operators are barriers; one layer at a time, decisions chained; no operator fusion (kernels as the framework provides).
Port:
  - cost model: the measured cycles of each kernel on H (ideal additive cost model). GRANII's XGBoost models can only do
    worse, so GRANII-dagger is an upper bound of GRANII on this substrate (conservative for the comparison).
  - kernel values: H (h_results.json, per-kernel H incl. GraphSAGE's count), H_G's new kernels (hg_adj_c,
    hg_propnorm_c), and the (A + I) SpMM kernels of grn_run.py (grn_runs.json).
  - Ñ (normalized adjacency) and A + I are graph preprocessing, taken as inputs as for every method.
Composition space per layer (unfused only):
  GCN / GraphSAGE  o2 (aggregate, then update), o3 (update, then aggregate) - precomputed Ñ.
                   Dynamic normalization (D . A~ . D with GRANII's unweighted SpMM and two row broadcasts) is dominated
                   in the tiled comparison without measurement: a stored tile of A~ holds a 0/1 pattern whose product with
                   a tile of the features is still a tile matrix product, so the unweighted SpMM costs the same as the
                   weighted one, and every dynamic composition = the precomputed composition of the same order + at least
                   two row-broadcast kernels of positive cost. GRANII's minimum-sum selection never picks it.
                   (On Collab/MAG, where H runs adjacency kernels scalar, the same argument is used; recorded.)
  SGC              o2, o3, and the mixed order A.((A.X).W) (sgc_prop1, sgc_linear_mul, hg_propnorm_c), where H_G's
                   re-association applies (C < F); dynamic normalization dominated as above.
  GIN              layer 1: (A.H + H).W [o2], A.(H.W) + H.W [o3], (A + I).H then .W, (A + I).(H.W) [new kernels];
                   layer 2: the same four with the first MLP linear (64 -> 64).
Built-in checks: o2 everywhere = H unfused; GRANII-dagger <= GALA-dagger (its candidate set contains GALA's choice).
Outputs: /fuseflow-v3/granii/results/granii_results.json and cells/GRANII.json (plot schema).
"""
import argparse
import importlib.util
import json
from pathlib import Path

V3 = Path("/fuseflow-v3/scripts")
H_FILE = Path("/fuseflow-v3/h_baseline/h_results.json")
HG_FILE = Path("/fuseflow-v3/results/h_g/h_g_results.json")
GRN_FILE = Path("/fuseflow-v3/granii/results/grn_runs.json")
GALA_FILE = Path("/fuseflow-v3/gala/results/gala_results.json")
CELLS = Path("/fuseflow-v3/results/h_vec/cells")
OUT = Path("/fuseflow-v3/granii/results")


def load(name):
    spec = importlib.util.spec_from_file_location(name, V3 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cell_best(name, m, ds):
    f = CELLS / f"{name}.json"
    if not f.exists():
        return None
    cfgs = json.loads(f.read_text()).get("configs", {})
    return min((c[ds]["cycles"] for k, c in cfgs.items() if k.startswith(f"{m}/") and (c.get(ds) or {}).get("cycles")),
               default=None)


def compositions(hg, m, ds):
    """[layer] -> {name: [kernel, ...]}; kernels: 'unfused/<stem>' (H), 'new:<name>' (H_G), 'grn:<stem>' (grn_run)"""
    layers = []
    for i, layer in enumerate(hg.LAYERS[m]):
        c = {}
        o2 = layer["o2"]
        c["o2"] = o2
        if "o3" in layer and not any(k.startswith("new:") and not hg.new_needed(k[4:], m, ds)[0] for k in layer["o3"]):
            c["o3"] = layer["o3"]
        if m == "sgc" and hg.new_needed("hg_propnorm_c", m, ds)[0]:
            c["mixed"] = ["unfused/sgc_prop1", "unfused/sgc_linear_mul", "new:hg_propnorm_c"] + o2[3:]
        if m == "gin":
            if i == 0:
                lin, rest = "unfused/gin_mlp1_lin1_mul", o2[3:]
                c["o3"] = layer["o3"]
                c["A+I"] = ["grn:grn_adjloops_f", lin] + rest
                c["A+I_o3"] = [lin, "grn:grn_adjloops_64"] + rest
            else:
                adj, sa, lin, rest = o2[0], o2[1], o2[2], o2[3:]
                c["o3"] = [lin, adj, sa] + rest
                c["A+I"] = ["grn:grn_adjloops_64", lin] + rest
                c["A+I_o3"] = [lin, "grn:grn_adjloops_64"] + rest
        layers.append(c)
    return layers


def main():
    hg = load("h_g_run")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(hg.DATASETS))
    ap.add_argument("--models", nargs="+", default=list(hg.LAYERS))
    a = ap.parse_args()
    H = json.loads(H_FILE.read_text())
    HG = json.loads(HG_FILE.read_text())
    GRN = json.loads(GRN_FILE.read_text()) if GRN_FILE.exists() else {}
    GALA = json.loads(GALA_FILE.read_text()).get("models", {}) if GALA_FILE.exists() else {}
    doc = {"metadata": {"definition": __doc__}, "models": {}, "checks": []}
    cell = {"metadata": {"cell": "GRANII-dagger (strict port, ideal additive cost model, unfused only)"}, "configs": {},
            "kernels": {}}
    for m in a.models:
        for ds in a.datasets:
            def value(k):
                if k.startswith("new:"):
                    return ((HG.get("new_kernels", {}).get(k[4:]) or {}).get(ds) or {}).get("value")
                if k.startswith("grn:"):
                    return ((GRN.get(ds) or {}).get(f"h__{k[4:]}") or {}).get("value")
                return ((H["kernels"].get(f"{m}/{k}") or {}).get(ds) or {}).get("H")

            layers, missing = [], []
            for i, comps in enumerate(compositions(hg, m, ds)):
                costs = {}
                for name, ks in comps.items():
                    vals = [value(k) for k in ks]
                    if all(vals):
                        costs[name] = sum(vals)
                    else:
                        missing.append(f"layer {i + 1} {name}: {[k for k, v in zip(ks, vals) if not v]}")
                pick = min(costs, key=costs.get) if costs else None
                layers.append({"layer": i + 1, "compositions": costs, "choice": pick,
                               "cycles": costs.get(pick) if pick else None})
            total = sum(l["cycles"] for l in layers) if all(l["cycles"] for l in layers) else None
            o2 = sum(l["compositions"]["o2"] for l in layers) if all("o2" in l["compositions"] for l in layers) else None
            h_unf = ((H["configs"].get(f"{m}/unfused") or {}).get(ds) or {}).get("H")
            if o2 and h_unf and abs(o2 - h_unf) > max(10, 1e-6 * h_unf):
                doc["checks"].append(f"{m}/{ds} o2 everywhere vs H unfused: {o2:,} vs {h_unf:,}")
            gala = ((GALA.get(m) or {}).get(ds) or {}).get("GALA")
            if total and gala and total > gala + 10:
                doc["checks"].append(f"{m}/{ds} GRANII {total:,} above GALA {gala:,} (its candidates contain GALA's)")
            hcfg = [((v.get(ds) or {}).get("H")) for c, v in H["configs"].items() if c.startswith(f"{m}/")]
            rec = {"GRANII": total, "layers": layers, "missing": missing, "GALA": gala, "H_unfused": h_unf,
                   "H_best": min((x for x in hcfg if x), default=None), "H+G+V": cell_best("H_G_V", m, ds)}
            doc["models"].setdefault(m, {})[ds] = rec
            cell["configs"].setdefault(f"{m}/unfused", {})[ds] = {"cycles": total, "layers": layers}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "granii_results.json").write_text(json.dumps(doc, indent=1) + "\n")
    (OUT / "cells").mkdir(exist_ok=True)
    (OUT / "cells" / "GRANII.json").write_text(json.dumps(cell, indent=1) + "\n")

    f = lambda v: f"{v:,}".rjust(15) if isinstance(v, int) else str(v).rjust(15)
    r_ = lambda x, y: f"{x / y:7.2f}x" if x and y else "    -   "
    print(f"{'model':9} {'dataset':8} {'H best':>15} {'GALA':>15} {'GRANII':>15} {'H+G+V':>15}   GRANII vs H best | "
          f"GRANII vs GALA | H+G+V vs GRANII | choices")
    for m, per in doc["models"].items():
        for ds, r in per.items():
            ch = " ".join(f"L{l['layer']}:{l['choice']}" for l in r["layers"])
            print(f"{m:9} {ds:8} {f(r['H_best'])} {f(r['GALA'])} {f(r['GRANII'])} {f(r['H+G+V'])}   "
                  f"{r_(r['H_best'], r['GRANII'])}      | {r_(r['GALA'], r['GRANII'])}      | {r_(r['GRANII'], r['H+G+V'])}       | {ch}"
                  + (f"   MISSING {r['missing']}" if r["missing"] else ""))
    print("built-in checks:", "all passed" if not doc["checks"] else "\n  " + "\n  ".join(doc["checks"]))
    print(f"written {OUT / 'granii_results.json'}, {OUT / 'cells' / 'GRANII.json'}")


if __name__ == "__main__":
    main()
