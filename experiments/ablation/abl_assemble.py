"""
Experiment 2 (ablation), from the existing per-kernel results - nothing is simulated.

    /fuseflow-v3/venv/bin/python -u /fuseflow-v3/ablation/scripts/abl_assemble.py [--datasets ...] [--models ...]

Per layer candidates (h_g_run.LAYERS): o1 = FuseFlow's layer-fused kernel (partially fused), o2 = FuseFlow's unfused
kernels in the model's order, o3 = re-associated unfused kernels (where H_G's rule applies: the width shrinks).
Kernel values:
  no V : H's per-kernel value (h_results.json; GraphSAGE's count included), H_G's new kernels (hg_*)
  with V: min(no-V value, vector run x count) for the aggregation kernels and H_G's new kernels (h_vec runs), exactly the
          candidates of the cell H+G+V (no CSR); o1 has no vector run (fused kernel), so V leaves it unchanged.
A model total is the minimum over the configurations it may use, always including FuseFlow's fully fused (H's value,
untouched by G and V) and, through H, FuseFlow's uniform configurations:
  H                 min(sum o2, sum o1, fully fused)                       = H
  2b reassoc only   min(H, sum_layers min(o2, o3))                         re-association, no per-layer fusion choice
  2b fusion only    min(H, sum_layers min(o1, o2))                         per-layer fusion boundary, no re-association
  2b full G         min(H, sum_layers min(o1, o2, o3))                     = H+G
  2a H+V            min(sum o2^V, sum o1, fully fused)                     = H+V
  2c joint (ours)   min(sum_layers min(o2^V, o3^V), sum_layers min(o1, o2^V, o3^V), fully fused)   = H+G+V
  2c select-then-V  each configuration's per-layer choice is made with the no-V costs, then evaluated with V:
                    min(sum_layers [argmin_noV(o2, o3)]^V, sum_layers [argmin_noV(o1, o2, o3)]^V, fully fused)
Built-in checks: the recomputed H, H+G, H+V, H+G+V equal the cells in results/h_vec/cells (and H's h_results).
Outputs: /fuseflow-v3/ablation/results/ablation_results.json.
"""
import argparse
import importlib.util
import json
from pathlib import Path

V3 = Path("/fuseflow-v3/scripts")
H_FILE = Path("/fuseflow-v3/h_baseline/h_results.json")
HG_FILE = Path("/fuseflow-v3/results/h_g/h_g_results.json")
INV = Path("/fuseflow-v3/h_baseline/h0/h0_inventory.json")
VRUNS = Path("/fuseflow-v3/results/h_vec/runs")
CELLS = Path("/fuseflow-v3/results/h_vec/cells")
OUT = Path("/fuseflow-v3/ablation/results")


def load(name):
    spec = importlib.util.spec_from_file_location(name, V3 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vec_value(ds, path):
    f = VRUNS / ds / Path(path).stem / "result.json"
    if not path or not f.exists():
        return None
    r = json.loads(f.read_text())
    ok = [b.get("cycles") for b in r.get("blocks", {}).values() if b.get("status") == "ok" and b.get("cycles")]
    return min(ok) if ok else None


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
    inv = json.loads(INV.read_text())
    doc = {"metadata": {"definition": __doc__}, "models": {}, "checks": []}
    for m in a.models:
        for ds in a.datasets:
            def v0(k):
                if k.startswith("new:"):
                    return ((HG.get("new_kernels", {}).get(k[4:]) or {}).get(ds) or {}).get("value")
                return ((H["kernels"].get(f"{m}/{k}") or {}).get(ds) or {}).get("H")

            def v1(k):
                if k.startswith("new:"):
                    c = [v0(k), vec_value(ds, hg.NEW[k[4:]])]
                else:
                    hr = (H["kernels"].get(f"{m}/{k}") or {}).get(ds) or {}
                    vv = vec_value(ds, (inv.get(f"{m}/{k}") or {}).get("path"))
                    c = [hr.get("H"), vv * (hr.get("count") or 1) if vv else None]
                c = [x for x in c if x]
                return min(c) if c else None

            layers = []
            for layer in hg.LAYERS[m]:
                o0, o1v = {}, {}
                for v, ks in layer.items():
                    if v == "o3" and any(k.startswith("new:") and not hg.new_needed(k[4:], m, ds)[0] for k in ks):
                        continue
                    a0, a1 = [v0(k) for k in ks], [v1(k) for k in ks]
                    if all(a0):
                        o0[v] = sum(a0)
                    if all(a1):
                        o1v[v] = sum(a1)
                layers.append((o0, o1v))
            ff = ((H["configs"].get(f"{m}/fully_fused") or {}).get(ds) or {}).get("H")

            def pick(costs, allowed):
                c = {v: x for v, x in costs.items() if v in allowed}
                return min(c, key=c.get) if c else None

            def total(sel_costs, eval_costs, allowed):
                """per layer: choose with sel_costs among allowed, evaluate with eval_costs"""
                s = 0
                for (c0, c1), (e0, e1) in zip(sel_costs, eval_costs):
                    v = pick(c0, allowed)
                    if v is None or v not in e1:
                        return None
                    s += e1[v]
                return s

            noV = [(o0, o0) for o0, _ in layers]       # select and evaluate without V
            withV = [(o1v, o1v) for _, o1v in layers]
            selnoV = [(o0, o0) for o0, _ in layers]
            mn = lambda *xs: min((x for x in xs if x), default=None)
            sum_o = lambda which, k: (sum(l[which][k] for l in layers) if all(k in l[which] for l in layers) else None)

            Hh = mn(sum_o(0, "o2"), sum_o(0, "o1"), ff)
            reassoc = mn(Hh, total(noV, noV, ("o2", "o3")))
            fusion = mn(Hh, total(noV, noV, ("o1", "o2")))
            fullG = mn(Hh, total(noV, noV, ("o1", "o2", "o3")))
            HV = mn(sum_o(1, "o2"), sum_o(1, "o1"), ff)
            joint = mn(total(withV, withV, ("o2", "o3")), total(withV, withV, ("o1", "o2", "o3")), ff)
            thenV = mn(total(selnoV, withV, ("o2", "o3")), total(selnoV, withV, ("o1", "o2", "o3")), ff)
            changed = []
            for i, (o0, o1v) in enumerate(layers):
                for cfg, allowed in (("unfused", ("o2", "o3")), ("partially_fused", ("o1", "o2", "o3"))):
                    b0, b1 = pick(o0, allowed), pick(o1v, allowed)
                    if b0 != b1:
                        changed.append(f"layer {i + 1} {cfg}: {b0} -> {b1}")
            for name, got, cell in (("H", Hh, "H"), ("H+G", fullG, "H_G"), ("H+V", HV, "H_V"), ("H+G+V", joint, "H_G_V")):
                want = cell_best(cell, m, ds)
                if got and want and abs(got - want) > max(10, 1e-6 * want):
                    doc["checks"].append(f"{m}/{ds} {name}: recomputed {got:,} vs cell {want:,}")
            doc["models"].setdefault(m, {})[ds] = {
                "H": Hh, "reassoc_only": reassoc, "fusion_only": fusion, "full_G": fullG, "H+V": HV,
                "joint_H+G+V": joint, "select_then_V": thenV, "choices_changed_by_V": changed,
                "layers_noV": [o0 for o0, _ in layers], "layers_withV": [o1v for _, o1v in layers], "fully_fused": ff}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation_results.json").write_text(json.dumps(doc, indent=1) + "\n")
    g = lambda h, x: f"{h / x:6.2f}x" if h and x else "   -   "
    print("2a/2b: speedup over H        reassoc-only  fusion-only  full G  |  H+V   H+G+V  | (G+V)/max(G,V)")
    for m, per in doc["models"].items():
        for ds, r in per.items():
            h = r["H"]
            gG, gV, gGV = (h / r["full_G"] if r["full_G"] else 0), (h / r["H+V"] if r["H+V"] else 0), \
                (h / r["joint_H+G+V"] if r["joint_H+G+V"] else 0)
            print(f"{m:9} {ds:8}  {g(h, r['reassoc_only'])}   {g(h, r['fusion_only'])}   {g(h, r['full_G'])} | "
                  f"{g(h, r['H+V'])} {g(h, r['joint_H+G+V'])} | {gGV / max(gG, gV, 1e-9):5.2f}")
    print("\n2c: joint selection vs select-then-V (cycles; ratio > 1 = joint is faster)")
    for m, per in doc["models"].items():
        for ds, r in per.items():
            j, t = r["joint_H+G+V"], r["select_then_V"]
            print(f"{m:9} {ds:8}  joint {j or 0:>13,}  select-then-V {t or 0:>13,}  {g(t, j)}  "
                  f"changed: {'; '.join(r['choices_changed_by_V']) or '-'}")
    print("\nbuilt-in checks:", "all reproduced" if not doc["checks"] else "\n  " + "\n  ".join(doc["checks"]))
    print(f"written {OUT / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
