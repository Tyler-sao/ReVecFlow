#!/usr/bin/env python3
"""
New Figure 12 model baselines (SGC, GIN; APPNP dropped 2026-10-13): runs the configurations kernel by kernel and accounts them as
Figure 12 does (fully fused = its one kernel; partially fused and unfused = the sum of their kernels, valid only if
every kernel is valid). Not part of the original Figure 12.

    cd /fuseflow-artifact && /fuseflow-v3/venv/bin/python -u scripts/run_figure12_newmodels.py \
        --models sgc --datasets cora [--fusion unfused partially_fused fully_fused] [--kernels K ...] \
        [--workers 1] [--timeout 86400] [--resume] [--keep-data] [--repeat-mode all|once]

Each kernel is one run of scripts/newmodels_kernel.py (one simulation; cycles and checked output from the same
simulator: FuseFlow with the locate fix, HBM off, parallel factor 1, block mode 1) in its own directory
results/figure12_newmodels/runs/<model>/<dataset>/<fusion>/<kernel>[_r<i>]. A kernel is valid if its status is ok
(exit 0, Comal passed, cycles parsed, output within atol = rtol = 1e-4 of the f32 reference).
- checks: the check-only kernels (values before a softmax) of a configuration; they must be valid for the
  configuration to be valid, and their cycles are not counted.
- repeats: APPNP's step kernels run 10 (unfused) or 9 + 1 (partially fused) times, each a separate kernel run as
  Figure 12 runs each operator; --repeat-mode once runs one and multiplies (a labelled deviation; with the fixed
  generator seed every repeat sees the same data, so the cycles are identical either way).
- skips (artifact's medium mode): fully fused is not run on MAG, recorded as "skipped (medium-mode rule)".
- known failures are run and recorded like any other kernel.
- MAG repairs (2026-10-14): on OGB-MAG, whose A has 118,465 empty rows, the partially fused GIN and GraphSAGE kernels
  fail in Comal ("Stop tokens don't match" / "Unexpected end of stream"). For (model, partially_fused, mag) the runner
  runs REPAIRS instead: identical copies of the kernels (tests/models/gin_repair, tests/models/graphsage_repair) whose
  data differ only in A's empty rows, padded with one explicit zero each. The record carries a "repair" field, and its
  breakdown names the original kernels (with "repaired_copy"). GraphSAGE is run only for this case (its other baselines
  are FuseFlow's own Figure 12 results); its repaired record is also written into figure12_results.json and
  figure12_graphsage.json (backed up once to <file>.before_graphsage_mag_repair_<date>). Earlier repair runs in
  results/figure12_newmodels/{gin,graphsage}_repair/mag/ are reused (copied) instead of simulated again.
Outputs (host-visible through the results/ mount): results/figure12_newmodels/figure12_newmodels_results.json,
results/figure12_newmodels/summary.csv. --resume reuses kernel runs whose result.json has status ok. Unless
--keep-data, a kernel's data directory is deleted after its check (result.json and output.npz stay).
"""
import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARTIFACT = Path("/fuseflow-artifact")
OUT = ARTIFACT / "results" / "figure12_newmodels"
PY = Path("/fuseflow-v3/venv/bin/python")
KERNEL = ARTIFACT / "scripts" / "newmodels_kernel.py"

DATASETS = {"cora": ("Planetoid", "Cora"), "cora_ml": ("CitationFull", "cora_ml"), "dblp": ("CitationFull", "dblp"),
            "collab": ("ogbl", "ogbl-collab"), "mag": ("ogbn", "ogbn-mag")}

U = lambda m, *ks: [f"{m}_unfused/{k}" for k in ks]
CONFIGS = {
    "sgc": {
        "fully_fused": {"kernels": ["sgc/sgc_sparse"], "checks": ["sgc/sgc_sparse_logits"]},
        # the guide's grouping (propagation stage | classifier): both propagations in one group
        "partially_fused": {"kernels": U("sgc", "sgc_propagate", "sgc_classify"), "checks": U("sgc", "sgc_classify_logits")},
        # partially fused plans with the two propagations in separate groups (chain S.X | S.P1 | .W | +b | softmax)
        "partial_p1": {"kernels": U("sgc", "sgc_prop1", "sgc_prop_classify"), "checks": U("sgc", "sgc_prop_linear")},
        "partial_p2": {"kernels": U("sgc", "sgc_prop1", "sgc_prop_linear", "sgc_softmax")},
        "partial_p3": {"kernels": U("sgc", "sgc_prop1", "sgc_prop_mul", "sgc_bias_softmax"), "checks": U("sgc", "sgc_linear_bias")},
        "partial_p4": {"kernels": U("sgc", "sgc_prop1", "sgc_prop_mul", "sgc_linear_bias", "sgc_softmax")},
        "partial_p5": {"kernels": U("sgc", "sgc_prop1", "sgc_prop2", "sgc_classify"), "checks": U("sgc", "sgc_classify_logits")},
        "partial_p6": {"kernels": U("sgc", "sgc_prop1", "sgc_prop2", "sgc_linear", "sgc_softmax")},
        "partial_p7": {"kernels": U("sgc", "sgc_prop1", "sgc_prop2", "sgc_linear_mul", "sgc_bias_softmax"), "checks": U("sgc", "sgc_linear_bias")},
        "unfused": {"kernels": U("sgc", "sgc_prop1", "sgc_prop2", "sgc_linear_mul", "sgc_linear_bias", "sgc_softmax")},
    },
    "gin": {
        "fully_fused": {"kernels": ["gin/gin_sparse"], "checks": ["gin/gin_sparse_logits"]},
        "partially_fused": {"kernels": U("gin", "one_layer_gin", "one_layer_gin2"), "checks": U("gin", "one_layer_gin2_logits")},
        "unfused": {"kernels": U("gin", "gin_adj_h1", "gin_selfadd1", "gin_mlp1_lin1_mul", "gin_mlp1_lin1_bias", "gin_mlp1_bn1",
                                 "gin_mlp1_relu1", "gin_mlp1_lin2_mul", "gin_mlp1_lin2_bias", "gin_bn_out1", "gin_relu_out1",
                                 "gin_adj_h2", "gin_selfadd2", "gin_mlp2_lin1_mul", "gin_mlp2_lin1_bias", "gin_mlp2_bn1",
                                 "gin_mlp2_relu1", "gin_mlp2_lin2_mul", "gin_mlp2_lin2_bias", "gin_softmax")},
    },
}
# GraphSAGE: only its repaired partially fused run on MAG (REPAIRS); every other GraphSAGE number is FuseFlow's
CONFIGS["graphsage"] = {"partially_fused": {"kernels": [], "checks": []}}
_WHY = ("OGB-MAG's A has 118,465 empty rows; the kernel fails in Comal on them (reproduced on Cora by emptying rows of A: "
        "fails; padding them: exact, with the original kernel's cycles)")
_HOW = ("identical copy of the kernel, same generator class and values, A's empty rows padded with one explicit zero each "
        "(input preparation)")
REPAIRS = {
    ("gin", "partially_fused", "mag"): {
        "kernels": ["gin_repair/gin_repair_l1", "gin_repair/gin_repair_l2"], "checks": ["gin_repair/gin_repair_l2_logits"],
        "names": {"gin_repair_l1": "one_layer_gin", "gin_repair_l2": "one_layer_gin2", "gin_repair_l2_logits": "one_layer_gin2_logits"},
        "legacy": "gin_repair/mag"},
    ("graphsage", "partially_fused", "mag"): {
        "kernels": ["graphsage_repair/sage_repair_l1", "graphsage_repair/sage_repair_l2"], "checks": [],
        "names": {"sage_repair_l1": "one_layer_graphsage", "sage_repair_l2": "one_layer_graphsage2"},
        "legacy": "graphsage_repair/mag"},
}
FUSIONS = ["unfused", "partially_fused", "partial_p1", "partial_p2", "partial_p3", "partial_p4", "partial_p5",
           "partial_p6", "partial_p7", "fully_fused"]
SKIP = {("fully_fused", "mag"): "skipped (medium-mode rule: fully fused is not run on MAG)"}
CSV_FIELDS = ["model", "dataset", "fusion", "kernel", "role", "repeat_index", "status", "reason", "cycles", "valid",
              "max_abs_err", "max_rel_err", "wall_seconds", "padded_weight_rows", "preprocess_seconds", "data_dir", "log_file"]

_locks, _locks_guard = {}, threading.Lock()
_done, _done_guard = {}, threading.Lock()   # kernel runs of this invocation, by directory


def kernel_lock(name):
    """run_end_to_end.py writes <kernel>_sam.mlir next to the MLIR and data/<dir>/<kernel>_b64: one run per kernel
    name at a time."""
    with _locks_guard:
        return _locks.setdefault(name, threading.Lock())


def run_kernel(mlir, dataset, run_dir, timeout, resume, keep_data):
    res_file = run_dir / "result.json"
    if resume and res_file.exists():
        r = json.loads(res_file.read_text())
        if r.get("status") == "ok":
            return r | {"resumed": True}
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    log = run_dir.parent / f"{run_dir.name}.log"
    start = time.time()
    with kernel_lock(Path(mlir).stem), open(log, "w") as f:
        p = subprocess.Popen([str(PY), "-u", str(KERNEL), "--mlir", f"{mlir}.mlir", "--dataset", *DATASETS[dataset], "--out", str(run_dir)],
                             cwd=ARTIFACT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, "DATA_PATH": os.environ.get("DATA_PATH", "/tmp/data")})
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return {"status": "timeout", "reason": f"wall-clock budget {timeout} s exceeded", "cycles": None,
                    "wall_seconds_total": round(time.time() - start, 1), "log": str(log)}
    r = json.loads(res_file.read_text()) if res_file.exists() else {"status": "error", "reason": f"no result.json (exit {p.returncode})"}
    if r.get("status") == "running":
        r.update(status="error", reason=f"process ended (exit {p.returncode}) before finishing; see {log}")
    if not r.get("reason") and r.get("status") not in ("ok",):
        tail = log.read_text(errors="replace").strip().splitlines()[-3:]
        r["reason"] = " | ".join(tail)[:400]
    pre = run_dir / "data" / "preprocess.json"
    if pre.exists():
        r["padded_weight_rows"] = json.loads(pre.read_text()).get("padded_weight_rows")
    if not keep_data and (run_dir / "data").exists():
        shutil.rmtree(run_dir / "data")
        r["data_deleted"] = True
    r["log"] = str(log)
    if res_file.exists():
        res_file.write_text(json.dumps(r, indent=1) + "\n")     # keeps the additions for --resume
    return r


def expand(entries):
    out = []
    for e in entries:
        mlir, n = (e, 1) if isinstance(e, str) else e
        out += [(mlir, i, n) for i in range(n)]
    return out


def run_config(model, dataset, fusion, args):
    repair = REPAIRS.get((model, fusion, dataset))
    cfg = repair or CONFIGS[model][fusion]
    rec = {"success": False, "valid": False, "cycles": None, "breakdown": [], "checks": []}
    if (fusion, dataset) in SKIP:
        return rec | {"status": "skipped", "reason": SKIP[(fusion, dataset)]}
    jobs = []
    for role, entries in (("kernel", cfg["kernels"]), ("check", cfg.get("checks", []))):
        for mlir, i, n in expand(entries):
            name = Path(mlir).stem
            if args.kernels and name not in args.kernels:
                continue
            once = args.repeat_mode == "once" and n > 1
            if once and i > 0:
                continue
            jobs.append((role, mlir, i, n, once))
    results = []

    def one(job):
        role, mlir, i, n, once = job
        # one run per kernel (and repeat) and dataset, shared by every configuration: a kernel's data depend only on
        # its name and the dataset (fixed seed), so the same kernel in two plans is the same simulation
        rd = OUT / "runs" / model / dataset / (Path(mlir).stem + (f"_r{i}" if n > 1 and not once else ""))
        with _done_guard:
            lock = _done.setdefault(str(rd), threading.Lock())
        with lock:
            if str(rd) + "#result" in _done:
                return job, rd, _done[str(rd) + "#result"]
            legacy = OUT / repair["legacy"] / Path(mlir).stem if repair else None
            if legacy and not (rd / "result.json").exists() and (legacy / "result.json").exists() \
                    and json.loads((legacy / "result.json").read_text()).get("status") == "ok":
                rd.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(legacy, rd)            # an earlier repair run of the same kernel and data: reused
            r = run_kernel(mlir, dataset, rd, args.timeout, args.resume, args.keep_data)
            _done[str(rd) + "#result"] = r
        return job, rd, r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(one, jobs))
    total, bad = 0, []
    for (role, mlir, i, n, once), rd, r in results:
        cmp_ = r.get("vs_ref_f32") or {}
        stem = Path(mlir).stem
        orig = (repair or {}).get("names", {}).get(stem)
        entry = {"file": mlir, "kernel": orig or stem, **({"repaired_copy": stem} if orig else {}), "repeat_index": i, "repeats": n, "status": r.get("status"),
                 "valid": r.get("status") == "ok", "success": r.get("status") == "ok", "cycles": r.get("cycles"),
                 "reason": r.get("reason"), "wall_seconds": r.get("wall_seconds_total"),
                 "max_abs_err": cmp_.get("max_abs_err"), "max_rel_err": cmp_.get("max_rel_err"),
                 "preprocess_seconds": r.get("preprocess_seconds"), "padded_weight_rows": r.get("padded_weight_rows"),
                 "multiplied_by": n if once else 1, "run_dir": str(rd), "log": r.get("log")}
        (rec["breakdown"] if role == "kernel" else rec["checks"]).append(entry)
        if not entry["valid"]:
            bad.append(f"{entry['kernel']}{'#' + str(i) if n > 1 else ''}: {entry['status']}")
        elif role == "kernel":
            total += entry["cycles"] * entry["multiplied_by"]
    complete = not args.kernels
    rec["success"] = rec["valid"] = complete and not bad
    rec["cycles"] = total if rec["valid"] else None
    rec["status"] = "ok" if rec["valid"] else ("partial-run" if not complete else "invalid")
    if bad:
        rec["reason"] = "; ".join(bad)[:1000]
    if repair:
        rec["repair"] = {"date": "2026-10-14", "why": _WHY, "how": _HOW, "kernels": cfg["kernels"] + cfg["checks"]}
    if args.repeat_mode == "once" and any(e["multiplied_by"] > 1 for e in rec["breakdown"]):
        rec["deviation"] = "repeated kernels run once and multiplied (--repeat-mode once)"
    return rec


def write_graphsage_mag_to_figure12(rec):
    """The repaired GraphSAGE partially fused MAG record, in FuseFlow's format, into figure12_results.json and
    figure12_graphsage.json (decision 2026-10-13: repaired results go where the results are kept)."""
    entry = {"success": True, "cycles": rec["cycles"],
             "breakdown": [{"file": f"{e['kernel']}.mlir", "success": True, "cycles": e["cycles"]} for e in rec["breakdown"]],
             "repair": rec["repair"] | {"runs": [e["run_dir"] for e in rec["breakdown"]],
                                        "checks": {e["kernel"]: {"max_abs_err": e["max_abs_err"]} for e in rec["breakdown"]},
                                        "simulator": "FuseFlow with the locate fix (compiler 0016, Comal 0015) and output capture (0003)"}}
    for f in (ARTIFACT / "figure12_results.json", ARTIFACT / "figure12_graphsage.json"):
        if not f.exists():
            continue
        doc = json.loads(f.read_text())
        res = doc.get("results", doc)
        old = res.get("graphsage", {}).get("mag", {}).get("partially_fused") or {}
        if old.get("success") and old.get("cycles") == entry["cycles"]:
            print(f"   {f.name}: graphsage/mag/partially_fused already {entry['cycles']:,}")
            continue
        backup = f.with_name(f"{f.name}.before_graphsage_mag_repair_{time.strftime('%Y%m%d')}")
        if not backup.exists():
            shutil.copy2(f, backup)
        res.setdefault("graphsage", {}).setdefault("mag", {})["partially_fused"] = entry
        f.write_text(json.dumps(doc, indent=2))
        print(f"   {f.name}: graphsage/mag/partially_fused {old.get('cycles')} -> {entry['cycles']:,} (backup {backup.name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--fusion", nargs="+", default=FUSIONS, choices=FUSIONS)
    ap.add_argument("--kernels", nargs="+", default=None, help="only these kernel names (retries); configurations stay partial")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=86400)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-data", action="store_true")
    ap.add_argument("--repeat-mode", choices=["all", "once"], default="all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jf = OUT / "figure12_newmodels_results.json"
    doc = json.loads(jf.read_text()) if jf.exists() else {"results": {}}
    doc["metadata"] = {
        "label": "new model baselines on FuseFlow (not part of the original Figure 12)",
        "simulator": "FuseFlow with the locate fix (compiler 0016, Comal 0015) and output capture (0003)",
        "hbm_enabled": False, "parfactor": 1, "streamlevel": 1, "blockmode": 1, "sp_argument": 0.9,
        "weight_sparsity": "MatrixGenerator default (0.6); -sp is ignored by the GNN generator classes, as in Figure 12",
        "seed": 4, "tolerance": {"atol": 1e-4, "rtol": 1e-4}, "repeat_mode": args.repeat_mode,
        "input_preparation": ["normalized adjacencies (SGC, APPNP) computed offline",
                              "BatchNorm factor 1/sqrt(var + eps) computed offline (GIN)",
                              "every compressed weight row stored; empty rows padded with one explicit zero"],
        "datasets": {k: list(v) for k, v in DATASETS.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for dataset in args.datasets:
        for model in args.models:
            fusions = [f for f in args.fusion if f in CONFIGS[model]
                       and (model != "graphsage" or (model, f, dataset) in REPAIRS)]
            if not fusions:
                continue
            for fusion in fusions:
                print(f"== {model} {dataset} {fusion}", flush=True)
                rec = run_config(model, dataset, fusion, args)
                doc["results"].setdefault(model, {}).setdefault(dataset, {})[fusion] = rec
                jf.write_text(json.dumps(doc, indent=1))
                print(f"   {rec['status']} cycles {rec['cycles']} {rec.get('reason', '')[:300]}"
                      f"{' (repaired run)' if rec.get('repair') else ''}", flush=True)
                if model == "graphsage" and rec.get("valid"):
                    write_graphsage_mag_to_figure12(rec)
            r = doc["results"][model][dataset]
            un = (r.get("unfused") or {}).get("cycles")
            for fusion in FUSIONS:
                c = (r.get(fusion) or {}).get("cycles")
                if fusion in r:
                    r[fusion]["speedup_vs_unfused"] = round(un / c, 4) if un and c else None
            valid = {f: v["cycles"] for f, v in r.items() if isinstance(v, dict) and v.get("valid")}
            if valid:
                best = min(valid, key=valid.get)
                r["best"] = {"configuration": best, "cycles": valid[best], "speedup_vs_unfused": r[best].get("speedup_vs_unfused")}
            jf.write_text(json.dumps(doc, indent=1))
            print(f"   best {model} {dataset}: {r.get('best')}", flush=True)
    rows = []
    for model, per in doc["results"].items():
        for dataset, cfgs in per.items():
            for fusion, rec in cfgs.items():
                if fusion == "best":
                    continue
                if rec.get("status") == "skipped":
                    rows.append({"model": model, "dataset": dataset, "fusion": fusion, "status": "skipped", "reason": rec["reason"]})
                for role in ("breakdown", "checks"):
                    for e in rec.get(role, []):
                        rows.append({"model": model, "dataset": dataset, "fusion": fusion, "kernel": e["kernel"],
                                     "role": "kernel" if role == "breakdown" else "check", "repeat_index": e["repeat_index"],
                                     "status": e["status"], "reason": e["reason"], "cycles": e["cycles"], "valid": e["valid"],
                                     "max_abs_err": e["max_abs_err"], "max_rel_err": e["max_rel_err"],
                                     "wall_seconds": e["wall_seconds"], "padded_weight_rows": json.dumps(e["padded_weight_rows"]),
                                     "preprocess_seconds": e["preprocess_seconds"], "data_dir": e["run_dir"], "log_file": e["log"]})
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("written:", jf, OUT / "summary.csv")


if __name__ == "__main__":
    sys.exit(main())
