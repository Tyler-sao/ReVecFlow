"""
Experiment 4: overhead and resources, from the existing runs - nothing is simulated.

    /fuseflow-v3/venv/bin/python -u /fuseflow-v3/exp4/scripts/exp4_assemble.py [--datasets ...] [--models ...]

Per kernel (H's rule): the fastest passing mode among scalar / block 16,32,64 (h_results H_choice; H_G new kernels:
paired scalar or blocks), and with V also the vector-16 run of the aggregation kernels (h_vec runs; x count).
Per layer candidates as in experiment 2 (h_g_run.LAYERS): o1 fused, o2 unfused, o3 re-associated (when C < width).

4.1 work and data volume, UNFUSED configuration (every kernel has full records; fused kernels' inputs are not recorded):
    H_unf = o2 per layer, H+G = best of o2/o3, H+V = o2 with V, H+G+V = best of o2/o3 with V.
    useful MACs (mode independent): aggregation nnz x w; X.W N x F x d; element-wise N x w (one op per element).
    stored values (the chosen mode's input layout, as written by H's writers): scalar = nnz (sparse) / all elements
    (dense, incl. compressed weights); block n = stored tiles x n^2 (zeros inside tiles included); vector 16 = nnz for A
    (one scalar per non-zero, broadcast over the lanes) + stored vectors x 16 for the dense operand.
    HBM is off in every run: this is the number of values streamed by the dataflow, a proxy for memory traffic.
4.2 cycle breakdown, each variant's best configuration (as in experiment 2, incl. fused kernels): aggregation, X.W,
    element-wise (bias, ReLU, softmax, BN, self-add, adds), fused (layer-fused o1, fully fused).
4.3 selection cost: candidate kernels per model and dataset, runs per kernel (scalar + 3 block sizes, + vector 16 for
    aggregations), and the simulation time / peak memory recorded by the runs G and V add (H_G new kernels, vector runs).
Outputs: /fuseflow-v3/exp4/results/exp4_results.json.
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path

V3 = Path("/fuseflow-v3/scripts")
H_FILE = Path("/fuseflow-v3/h_baseline/h_results.json")
HG_FILE = Path("/fuseflow-v3/results/h_g/h_g_results.json")
INV = Path("/fuseflow-v3/h_baseline/h0/h0_inventory.json")
HRUNS = Path("/fuseflow-v3/h_baseline/runs")
GRUNS = Path("/fuseflow-v3/results/h_g/runs")
VRUNS = Path("/fuseflow-v3/results/h_vec/runs")
CSRUNS = Path("/fuseflow-v3/cs/results/runs")
OUT = Path("/fuseflow-v3/exp4/results")
AGG = re.compile(r"^(gcn_adj_x[12]|gin_adj_h[12]|sgc_prop[12])$")


def load(name):
    spec = importlib.util.spec_from_file_location(name, V3 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jload(p):
    return json.loads(p.read_text()) if p.exists() else None


def category(k):
    if k.startswith("new:"):
        return "aggregation"
    cfg, stem = k.split("/")
    if cfg != "unfused":
        return "fused"
    if AGG.match(stem):
        return "aggregation"
    if stem.endswith("_mul"):
        return "X.W"
    return "element-wise"


def main():
    hg = load("h_g_run")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(hg.DATASETS))
    ap.add_argument("--models", nargs="+", default=list(hg.LAYERS))
    a = ap.parse_args()
    H = json.loads(H_FILE.read_text())
    HG = json.loads(HG_FILE.read_text())
    inv = json.loads(INV.read_text())
    doc = {"metadata": {"definition": __doc__}, "models": {}}
    nnz = {}
    for ds in a.datasets:
        r = jload(CSRUNS / ds / "gcn__gcn_adj_x1" / "result.json") or {}
        nnz[ds] = ((r.get("operands") or {}).get("A") or {}).get("nnz")

    for m in a.models:
        for ds in a.datasets:
            info = {}

            def kernel(k):
                """{noV: (cycles, mode, run), V: (cycles, mode, run)} for one kernel occurrence"""
                if k in info:
                    return info[k]
                if k.startswith("new:"):
                    name = k[4:]
                    rec = (HG.get("new_kernels", {}).get(name) or {}).get(ds) or {}
                    run = jload(GRUNS / ds / name / "result.json")
                    cand = {}
                    ps = rec.get("paired_scalar") or {}
                    if ps.get("passed") and ps.get("cycles"):
                        cand["scalar"] = ps["cycles"]
                    for n, b in (rec.get("blocks") or {}).items():
                        if b.get("status") == "ok" and b.get("cycles"):
                            cand[f"block{n}"] = b["cycles"]
                    v0 = rec.get("value")
                    mode0 = min(cand, key=cand.get) if cand else None
                    vrun = jload(VRUNS / ds / Path(hg.NEW[name]).stem / "result.json")
                    cnt = 1
                else:
                    hr = (H["kernels"].get(f"{m}/{k}") or {}).get(ds) or {}
                    v0, mode0, cnt = hr.get("H"), hr.get("H_choice"), hr.get("count") or 1
                    path = (inv.get(f"{m}/{k}") or {}).get("path") or ""
                    run = jload(HRUNS / ds / path.replace("/", "__").replace(".mlir", "") / "result.json")
                    vrun = jload(VRUNS / ds / Path(path).stem / "result.json") if path and AGG.match(Path(path).stem) else None
                vb = ((vrun or {}).get("blocks") or {}).get("v16") or {}
                vv = vb.get("cycles") * cnt if vb.get("status") == "ok" and vb.get("cycles") else None
                res = {"noV": (v0, mode0, run, cnt), "V": (vv, "vector16", vrun, cnt) if vv and v0 and vv < v0 else (v0, mode0, run, cnt),
                       "vrun": vrun}
                info[k] = res
                return res

            def cs_shapes(k):
                """[(shape, format)] of a matmul kernel's inputs from its C+S run (fallback when H has no run)"""
                if k.startswith("new:"):
                    return None
                r = jload(CSRUNS / ds / f"{m}__{k.split('/')[1]}" / "result.json") or {}
                o = r.get("operands") or {}
                if "A" not in o:
                    return None
                fmt = lambda f: ["compressed" if c == "s" else "dense" for c in f]
                return [(o["A"]["shape"], fmt(o["A"]["format"])), (o["B"]["shape"], fmt(o["B"]["format"]))]

            def in_shapes(k, run):
                if run and run.get("inputs"):
                    return [(t["shape"], t["format"]) for t in run["inputs"].values()]
                return cs_shapes(k)

            def stored(k, which):
                """values in the chosen mode's input layout (None if not recorded)"""
                cyc, mode, run, cnt = kernel(k)[which]
                if not mode or (not run and mode != "scalar"):
                    return None
                if mode == "vector16":
                    v = (run.get("blocks") or {}).get("v16", {}).get("inputs") or {}
                    tot = 0
                    for x in v.values():
                        tot += x["nonzeros"] if "nonzeros" in x else x.get("stored_vectors", 0) * 16
                    return tot * cnt
                if mode.startswith("block"):
                    n = int(mode[5:])
                    b = ((run.get("blocks") or {}).get(str(n)) or {}).get("inputs") or {}
                    tot = sum(x.get("stored_tiles", 0) for x in b.values() if isinstance(x, dict)) * n * n
                    return tot * cnt if tot else None
                shp = in_shapes(k, run) if mode == "scalar" else None
                if mode == "scalar" and shp is None:
                    return None
                tot = 0
                for s, fm in shp:
                    if len(s) == 2 and s[0] == s[1] and fm == ["compressed", "compressed"]:
                        if not nnz.get(ds):
                            return None
                        tot += nnz[ds]
                    else:
                        p = 1
                        for d in s:
                            p *= d
                        tot += p
                return tot * cnt

            def macs(k):
                """useful multiply-adds of the kernel occurrence (mode independent)"""
                c, _, run, cnt = kernel(k)["noV"]
                shp = in_shapes(k, run) or in_shapes(k, kernel(k)["vrun"]) or []
                shapes = [s for s, _ in shp]
                if category(k) == "aggregation":
                    w = [s[1] for s in shapes if len(s) == 2 and not (s[0] == s[1])]
                    if not w or not nnz.get(ds):
                        return None
                    return nnz[ds] * w[0] * cnt
                if category(k) == "X.W":
                    x = sorted((s for s in shapes if len(s) == 2), key=lambda s: -s[0])
                    if len(x) != 2:
                        return None
                    X, W = x
                    return X[0] * X[1] * W[1] * cnt
                if category(k) == "element-wise":
                    big = max((s for s in shapes if len(s) == 2), key=lambda s: s[0] * s[1], default=None)
                    return big[0] * big[1] * cnt if big else None
                return None

            layers = hg.LAYERS[m]

            def options(layer):
                o = {}
                for v, ks in layer.items():
                    if v == "o3" and any(k.startswith("new:") and not hg.new_needed(k[4:], m, ds)[0] for k in ks):
                        continue
                    o[v] = ks
                return o

            def pick(which, allowed):
                """per layer best option among allowed, by cycles with/without V; returns kernel list or None"""
                ks_all = []
                for layer in layers:
                    best = None
                    for v, ks in options(layer).items():
                        if v not in allowed:
                            continue
                        cs = [kernel(k)[which][0] for k in ks]
                        if all(cs) and (best is None or sum(cs) < best[0]):
                            best = (sum(cs), ks)
                    if best is None:
                        return None
                    ks_all += best[1]
                return ks_all

            ff = ((H["configs"].get(f"{m}/fully_fused") or {}).get(ds) or {}).get("H")
            variants = {"H": ("noV", ("o2",), ("o1",)), "H+G": ("noV", ("o2", "o3"), ("o1", "o2", "o3")),
                        "H+V": ("V", ("o2",), ("o1",)), "H+G+V": ("V", ("o2", "o3"), ("o1", "o2", "o3"))}
            rec = {"work_unfused": {}, "breakdown": {}, "selection": {}}
            for name, (which, unf, part) in variants.items():
                # 4.1: unfused configuration
                ks = pick(which, unf)
                if ks:
                    mac = [macs(k) for k in ks]
                    sto = [stored(k, which) for k in ks]
                    rec["work_unfused"][name] = {
                        "cycles": sum(kernel(k)[which][0] for k in ks),
                        "useful_macs": sum(mac) if all(x is not None for x in mac) else None,
                        "stored_values": sum(sto) if all(x is not None for x in sto) else None,
                        "by_category_stored": {c: sum(s for k, s in zip(ks, sto) if category(k) == c and s) for c in
                                               ("aggregation", "X.W", "element-wise")},
                        "modes": {k: kernel(k)[which][1] for k in ks}}
                # 4.2: best configuration
                confs = {}
                for cfg, allowed in (("unfused", unf), ("partially_fused", part)):
                    ks2 = pick(which, allowed)
                    if ks2:
                        confs[cfg] = ks2
                tots = {c: sum(kernel(k)[which][0] for k in ks2) for c, ks2 in confs.items()}
                if ff:
                    tots["fully_fused"] = ff
                best = min(tots, key=tots.get) if tots else None
                brk = {"aggregation": 0, "X.W": 0, "element-wise": 0, "fused": 0}
                if best == "fully_fused":
                    brk["fused"] = ff
                elif best:
                    for k in confs[best]:
                        brk[category(k)] += kernel(k)[which][0]
                rec["breakdown"][name] = {"config": best, "cycles": tots.get(best), "by_category": brk}
            # 4.3: selection cost (candidates and the runs G and V add)
            cand = sorted({k for layer in layers for v, ks in options(layer).items() for k in ks})
            g_new = [k for k in cand if k.startswith("new:")]
            aggs = [k for k in cand if category(k) == "aggregation"]
            secs, rss = 0.0, 0.0
            for k in g_new:
                r = jload(GRUNS / ds / k[4:] / "result.json") or {}
                secs += r.get("wall_seconds") or 0
                rss = max(rss, r.get("peak_rss_gb") or 0)
            seen = set()
            for k in aggs:
                vr = kernel(k)["vrun"]
                key = json.dumps((vr or {}).get("mlir"))
                if vr and key not in seen:
                    seen.add(key)
                    secs += vr.get("wall_seconds") or 0
                    rss = max(rss, vr.get("peak_rss_gb") or 0)
            rec["selection"] = {"layers": len(layers), "options_per_layer": [len(options(l)) for l in layers],
                                "candidate_kernels": len(cand), "new_kernels_for_G": len(g_new),
                                "vector_runs_for_V": len(seen),
                                "runs_per_kernel": "scalar + block 16/32/64 (+ vector 16 for aggregations)",
                                "added_simulation_seconds": round(secs, 1), "peak_rss_gb_added_runs": rss}
            doc["models"].setdefault(m, {})[ds] = rec

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp4_results.json").write_text(json.dumps(doc, indent=1) + "\n")
    f = lambda v: f"{v:,}" if isinstance(v, int) else str(v)
    print("4.1 unfused configuration: useful MACs and stored values (ratio to H_unf)")
    for m, per in doc["models"].items():
        for ds, r in per.items():
            w = r["work_unfused"]
            h = w.get("H") or {}
            line = []
            for n in ("H", "H+G", "H+V", "H+G+V"):
                x = w.get(n) or {}
                rm = (x.get("useful_macs") / h["useful_macs"]) if x.get("useful_macs") and h.get("useful_macs") else None
                rs = (x.get("stored_values") / h["stored_values"]) if x.get("stored_values") and h.get("stored_values") else None
                line.append(f"{n}: macs {f(x.get('useful_macs'))} ({rm:.3f})" if rm else f"{n}: macs {f(x.get('useful_macs'))}")
                line[-1] += f" stored {f(x.get('stored_values'))} ({rs:.3f})" if rs else f" stored {f(x.get('stored_values'))}"
            print(f"{m:9} {ds:8} " + " | ".join(line))
    print("\n4.2 cycle breakdown of the best configuration (share of total)")
    for m, per in doc["models"].items():
        for ds, r in per.items():
            parts = []
            for n in ("H", "H+G", "H+V", "H+G+V"):
                b = r["breakdown"].get(n) or {}
                t = b.get("cycles") or 0
                c = b.get("by_category") or {}
                parts.append(f"{n}[{b.get('config')}] " + " ".join(f"{k[:4]} {v / t:.0%}" for k, v in c.items() if t and v))
            print(f"{m:9} {ds:8} " + " | ".join(parts))
    print("\n4.3 selection cost")
    for m, per in doc["models"].items():
        for ds, r in per.items():
            s = r["selection"]
            print(f"{m:9} {ds:8} options/layer {s['options_per_layer']} candidate kernels {s['candidate_kernels']} "
                  f"(new for G {s['new_kernels_for_G']}, vector runs for V {s['vector_runs_for_V']})  "
                  f"added simulation {s['added_simulation_seconds']:,} s  peak RSS {s['peak_rss_gb_added_runs']} GB")
    print(f"\nwritten {OUT / 'exp4_results.json'}")


if __name__ == "__main__":
    main()
