"""
C+S baseline: assemble per-configuration totals from the per-kernel runs, next to H and H+G+V.

    /fuseflow-v3/venv/bin/python -u /fuseflow-v3/cs/scripts/cs_assemble.py [--datasets ...] [--models ...]

C+S has one configuration, unfused (Custard cannot fuse across expressions; option a, cs_kernel.py docstring). Per kernel
of FuseFlow's unfused configuration (H's h_results.json, same kernel list and the same GraphSAGE count multiplier):
  - matmul kernels (runs/<ds>/<model>__<kernel>/result.json):
      untiled  = C+S scalar cycles x count;
      tiled    = cs_H x count (fastest of C+S scalar and every block size whose output passed);
      Collab/MAG adjacency SpMM: lower bound N^2 x count in both (cs_mode lower_bound; the total is then ">=");
      blocked_only GEMM (MAG): untiled = FuseFlow's S of that kernel, marked assumed (on every simulated dataset the
      C+S scalar GEMM equals FuseFlow's within 0.6-1.0x).
  - every other kernel: FuseFlow's own kernel, S (untiled) and H (tiled) from h_results.json.
Also a per-kernel check: C+S blocked cycles vs H's blocked cycles at every block size.
Outputs (/fuseflow-v3/cs/results/): cs_assembled.json (everything), cells/C_S_untiled.json and cells/C_S_tiled.json
(plot schema of plot_v3_figure12.py: configs[<model>/unfused][<dataset>] = {"cycles", ...}).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cs_run import matmul_kernels  # noqa: E402  (the same kernel list the runs used)

H_RESULTS = Path("/fuseflow-v3/h_baseline/h_results.json")
HGV = Path("/fuseflow-v3/results/h_vec/cells/H_G_V.json")   # the H+G+V cell (plot schema), every dataset
RUNS = Path("/fuseflow-v3/cs/results/runs")
OUT = Path("/fuseflow-v3/cs/results")
DATASETS = ["cora", "cora_ml", "dblp", "collab", "mag"]
MODELS = ["gcn", "graphsage", "gin", "sgc"]


def load_run(ds, m, k):
    f = RUNS / ds / f"{m}__{k}" / "result.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    a = ap.parse_args()
    H = json.loads(H_RESULTS.read_text())
    hgv = json.loads(HGV.read_text()).get("configs", {}) if HGV.exists() else {}
    mm = {(v["model"], v["kernel"]) for v in matmul_kernels(a.models, None).values()}
    doc = {"metadata": {"definition": __doc__}, "models": {}, "kernel_check": []}
    cells = {name: {"metadata": {"cell": name}, "configs": {}, "kernels": {}} for name in ("C+S untiled", "C+S tiled")}
    for m in a.models:
        for ds in a.datasets:
            keys = sorted(k for k in H["kernels"] if k.startswith(f"{m}/unfused/") and ds in H["kernels"][k])
            if not keys:
                continue
            tot_u, tot_t, bound, assumed, missing, rows = 0, 0, [], [], [], {}
            for key in keys:
                kname = key.split("/")[2]
                h = H["kernels"][key][ds]
                cnt = h.get("count") or 1
                r = load_run(ds, m, kname)
                if (m, kname) not in mm:                        # not a matmul kernel: FuseFlow's own
                    u, t, src = h.get("S"), h.get("H"), "fuseflow kernel"
                elif r is None:
                    u = t = None
                    src = "C+S run missing"
                elif r.get("status") != "done":
                    u = t = None
                    src = f"C+S run {r.get('status')}: {r.get('reason')}"
                elif r.get("cs_mode") == "lower_bound":
                    u = t = r["cs_scalar_lower_bound"] * cnt
                    bound.append(kname)
                    src = "C+S lower bound N^2"
                else:
                    t = r["cs_H"] * cnt if r.get("cs_H") else None
                    if r.get("cs_mode") == "blocked_only":
                        u = h.get("S")
                        assumed.append(kname)
                        src = "C+S blocked only; untiled = FuseFlow S (assumed)"
                    else:
                        u = r["cs_best"] * cnt if r.get("cs_best") else None
                        src = "C+S"
                    # per-kernel check against H's blocked runs
                    hb = h.get("block") if isinstance(h.get("block"), dict) else {}
                    for n, cb in (r.get("cs_blocks") or {}).items():
                        hc = (hb.get(n) or {}).get("cycles")
                        if cb.get("cycles") and hc:
                            doc["kernel_check"].append({"model": m, "dataset": ds, "kernel": kname, "block": n,
                                                        "cs": cb["cycles"], "H": hc, "diff": cb["cycles"] - hc,
                                                        "rel": round((cb["cycles"] - hc) / hc, 6)})
                if u is None or t is None:
                    missing.append(f"{kname} ({src})")
                tot_u = tot_u + u if (tot_u is not None and u) else None
                tot_t = tot_t + t if (tot_t is not None and t) else None
                rows[kname] = {"count": cnt, "source": src, "untiled": u, "tiled": t, "S": h.get("S"), "H": h.get("H"),
                               "H_choice": h.get("H_choice")}
            hcfg = {c.split("/")[1]: (v.get(ds) or {}) for c, v in H["configs"].items() if c.startswith(f"{m}/")}
            h_best = min(((v.get("H"), c) for c, v in hcfg.items() if v.get("H")), default=(None, None))
            s_best = min(((v.get("S"), c) for c, v in hcfg.items() if v.get("S")), default=(None, None))
            ours = min((c[ds]["cycles"] for k, c in hgv.items()
                        if k.startswith(f"{m}/") and (c.get(ds) or {}).get("cycles")), default=None)
            rec = {"C+S_untiled": tot_u, "C+S_tiled": tot_t, "lower_bound_kernels": bound, "assumed_kernels": assumed,
                   "missing": missing, "S_unfused": (hcfg.get("unfused") or {}).get("S"),
                   "H_unfused": (hcfg.get("unfused") or {}).get("H"), "S_best": s_best, "H_best": h_best,
                   "H+G+V": ours, "kernels": rows}
            doc["models"].setdefault(m, {})[ds] = rec
            for name, v in (("C+S untiled", tot_u), ("C+S tiled", tot_t)):
                cells[name]["configs"].setdefault(f"{m}/unfused", {})[ds] = {
                    "cycles": v, "lower_bound": bool(bound), "held": [f"{k} (lower bound N^2)" for k in bound],
                    "kernels": {k: [x["S"], x["untiled" if name.endswith("untiled") else "tiled"]] for k, x in rows.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cs_assembled.json").write_text(json.dumps(doc, indent=1) + "\n")
    (OUT / "cells").mkdir(exist_ok=True)
    for name, c in cells.items():
        (OUT / "cells" / f"{name.replace('+', '_').replace(' ', '_')}.json").write_text(json.dumps(c, indent=1) + "\n")

    # ---- report
    kc = doc["kernel_check"]
    if kc:
        worst = max(kc, key=lambda x: abs(x["rel"]))
        print(f"C+S blocked vs H blocked, {len(kc)} (kernel, dataset, block) pairs: max |diff| "
              f"{max(abs(x['diff']) for x in kc)} cycles, max |rel| {abs(worst['rel']):.2e} "
              f"({worst['model']}/{worst['kernel']} {worst['dataset']} block{worst['block']})")
        off = [x for x in kc if abs(x["rel"]) > 1e-3]
        for x in off:
            print("   differs >0.1%:", x)
    f = lambda v, ge="": f"{ge}{v:,}".rjust(18) if isinstance(v, int) else str(v).rjust(18)
    print(f"\n{'model':9} {'dataset':8} {'S unfused':>18} {'C+S untiled':>18} {'H unfused':>18} {'C+S tiled':>18} "
          f"{'H best':>18} {'H+G+V':>18}  H+G+V over C+S tiled / over H best   (>= : lower bound)")
    for m, per in doc["models"].items():
        for ds, r in per.items():
            ge = ">=" if r["lower_bound_kernels"] else "  "
            ratio = lambda x, y: f"{x / y:6.2f}x" if x and y else "   -  "
            ge = ge.strip()
            print(f"{m:9} {ds:8} {f(r['S_unfused'])} {f(r['C+S_untiled'], ge)} {f(r['H_unfused'])} "
                  f"{f(r['C+S_tiled'], ge)} {f(r['H_best'][0])} {f(r['H+G+V'])}  "
                  f"{(ge + ratio(r['C+S_tiled'], r['H+G+V']).strip()).rjust(11)} / {ratio(r['H_best'][0], r['H+G+V'])}"
                  + (f"   assumed: {r['assumed_kernels']}" if r["assumed_kernels"] else "")
                  + (f"   MISSING: {r['missing']}" if r["missing"] else ""))
    print(f"\nwritten {OUT / 'cs_assembled.json'}, {OUT / 'cells'}/C_S_untiled.json, C_S_tiled.json")


if __name__ == "__main__":
    main()
