#!/usr/bin/env python3
"""
FuseFlow V3, H_G: the two new kernels the re-associated graphs need, and their data generators. New files only (plus
one import line appended to this container's data_generator.py, backup first, never twice).

    docker exec -w /fuseflow-artifact fuseflow-v3 /fuseflow-v3/venv/bin/python /fuseflow-v3/scripts/make_hg_kernels.py [--check-only]

  hg/hg_adj_c.mlir       A . Z with Z N x C (C = classes): FuseFlow's gcn_unfused/gcn_adj_x1.mlir with the dense operand's
                         placeholder N x F (1767x50) changed to N x C (1767x121); nothing else changed.
                         Layer 2 of GCN / GraphSAGE re-associated: H1.W2 first, then A.(H1 W2).
  hg/hg_propnorm_c.mlir  S . Z with Z N x C: sgc_unfused/sgc_prop1.mlir, same change. SGC re-associated: X.W first,
                         then the two propagations at width C.
Data (scripts/data_generator_hg.py): FuseFlow's role-based generator (data_generator_newmodels._RoleKernel), as for every
GIN/SGC kernel: hg_adj_c = [adj, rand] (A as gin_adj_h1 / gcn_adj_x1 use it - same cycles in Figure 12), hg_propnorm_c =
[adj_norm, rand] (S as sgc_prop1); "rand" = a random dense input of the placeholder shape, as FuseFlow's unfused kernels
take. Sizes from the placeholders: 1767 = N, 121 = C.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

A = Path("/fuseflow-artifact")
M = A / "fuseflow-compiler" / "tests" / "models"
OUT = M / "hg"
GEN = A / "scripts" / "data_generator_hg.py"
DG = A / "scripts" / "data_generator.py"
LINE = "from data_generator_hg import *  # V3 H_G kernels (hg/), role-based generator classes"
KERNELS = {"hg_adj_c": ("gcn_unfused/gcn_adj_x1.mlir", ["adj", "rand"]),
           "hg_propnorm_c": ("sgc_unfused/sgc_prop1.mlir", ["adj_norm", "rand"])}
GEN_SRC = '''"""V3 H_G kernels (models/hg/): generator classes from FuseFlow's role-based generator (written by make_hg_kernels.py)."""
from data_generator_newmodels import _RoleKernel

HG_ROLES = {roles}
__all__ = []
for _name, _roles in HG_ROLES.items():
    globals()[_name] = type(_name, (_RoleKernel,), {{"ROLES": _roles}})
    __all__.append(_name)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not args.check_only:
        OUT.mkdir(exist_ok=True)
    for name, (src, roles) in KERNELS.items():
        text = (M / src).read_text()
        n = text.count("1767x50xf32")
        if n < 2 or "x50" in text.replace("1767x50xf32", ""):
            sys.exit(f"STOP {src}: expected the N x F placeholder 1767x50xf32 (found {n}) and no other x50")
        new = (f"// V3 H_G KERNEL (new file, 2026-10-19): {src} with the dense operand N x F (1767x50) changed to N x C "
               f"(1767x121); nothing else changed\n") + text.replace("1767x50xf32", "1767x121xf32")
        print(f"== {name}: {src} -> hg/{name}.mlir ({n} placeholder uses changed), roles {roles}")
        for line in new.splitlines():
            if "func.func" in line or "linalg." in line:
                print("   ", line.strip()[:200])
        if not args.check_only:
            if (OUT / f"{name}.mlir").exists() and (OUT / f"{name}.mlir").read_text() != new:
                sys.exit(f"STOP: hg/{name}.mlir exists with other content")
            (OUT / f"{name}.mlir").write_text(new)
    if args.check_only:
        return 0
    GEN.write_text(GEN_SRC.format(roles=repr({k: r for k, (_, r) in KERNELS.items()})))
    text = DG.read_text()
    if LINE not in text:
        shutil.copy(DG, f"{DG}.before_hg_{time.strftime('%Y%m%d_%H%M%S')}")
        DG.write_text(text.rstrip("\n") + "\n" + LINE + "\n")
        print("import line appended to data_generator.py (backup taken)")
    else:
        print("import line already present")
    chk = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, 'scripts'); import data_generator as dg; "
                          f"print([(k, hasattr(dg, k)) for k in {list(KERNELS)}])"], cwd=A, capture_output=True, text=True)
    print("generator classes:", chk.stdout.strip() or chk.stderr[-500:])
    if chk.returncode or "False" in chk.stdout:
        sys.exit("STOP: the generator classes do not import")
    return 0


if __name__ == "__main__":
    sys.exit(main())
