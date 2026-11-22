#!/usr/bin/env python3
"""
GRANII-dagger (strict port of GRANII, CGO'26): the new kernels its primitive compositions need, and their data generators.
New files only (plus one import line appended to this container's data_generator.py, backup first, never twice) - the
same mechanism make_hg_kernels.py used for H_G.

    docker exec -w /fuseflow-artifact fuseflow-v3 /fuseflow-v3/venv/bin/python /fuseflow-v3/granii/scripts/make_grn_kernels.py [--check-only]

  (GRANII's row-broadcast kernels are not generated: in the tiled comparison the dynamic-normalization compositions
   are dominated without measurement, see granii_assemble.py. ROWBCAST below is kept for reference only.)
  grn/grn_adjloops_f.mlir / _64.mlir    SpMM over A + I (GIN's composition (A + I) . H: the self term folded into the
                                        graph, no selfadd kernel): gin_unfused/gin_adj_h1.mlir / gin_adj_h2.mlir unchanged
                                        except the comment; data [adj_loops, x] / [adj_loops, rand] (FuseFlow's role
                                        generator: adj_loops = A + I).
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

A = Path("/fuseflow-artifact")
M = A / "fuseflow-compiler" / "tests" / "models"
OUT = M / "grn"
GEN = A / "scripts" / "data_generator_grn.py"
DG = A / "scripts" / "data_generator.py"
LINE = "from data_generator_grn import *  # GRANII-dagger kernels (grn/), role-based generator classes"

ROWBCAST = """#map = affine_map<(d0, d1) -> (d0, d1)>
#mapr = affine_map<(d0, d1) -> (d0)>
#SPARSE = #sparse_tensor.encoding<{{
  map = (i,j) -> (i : compressed, j : compressed)
}}>
#SPARSE1D = #sparse_tensor.encoding<{{
  map = (i) -> (i : dense)
}}>

// GRANII-dagger KERNEL (new file, 2026-10-27): row broadcast Y(i,j) = X(i,j) * d(i), width {w}
func.func @matmul1 (%arg0: tensor<1767x{w}xf32>, %arg1: tensor<1767xf32>, %argout: tensor<1767x{w}xf32>) -> tensor<1767x{w}xf32> {{
    %v1 = tensor.empty() : tensor<1767x{w}xf32>
    %v2 = linalg.generic {{indexing_maps = [#map, #mapr, #map], iterator_types = ["parallel", "parallel"]}} ins(%arg0, %arg1 : tensor<1767x{w}xf32>, tensor<1767xf32>) outs(%v1 : tensor<1767x{w}xf32>) {{
    ^bb0(%in: f32, %d: f32, %out: f32):
      %p = arith.mulf %in, %d : f32
      linalg.yield %p : f32
    }} -> tensor<1767x{w}xf32>
    return %v2 : tensor<1767x{w}xf32>
}}
"""

KERNELS = {}   # row-broadcast kernels not needed in the tiled comparison (see granii_assemble.py)
KERNELS["grn_adjloops_f"] = (("gin_unfused/gin_adj_h1.mlir",
                              "// GRANII-dagger KERNEL (new file, 2026-10-27): gin_adj_h1.mlir unchanged, data A + I\n"),
                             ["adj_loops", "x"])
KERNELS["grn_adjloops_64"] = (("gin_unfused/gin_adj_h2.mlir",
                               "// GRANII-dagger KERNEL (new file, 2026-10-27): gin_adj_h2.mlir unchanged, data A + I\n"),
                              ["adj_loops", "rand"])

GEN_SRC = '''"""GRANII-dagger kernels (models/grn/): generator classes from FuseFlow's role-based generator (make_grn_kernels.py)."""
from data_generator_newmodels import _RoleKernel

GRN_ROLES = {roles}
__all__ = []
for _name, _roles in GRN_ROLES.items():
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
        if isinstance(src, tuple):
            text = (M / src[0]).read_text()
            new = src[1] + text
        else:
            new = src
        print(f"== {name}: roles {roles}")
        for line in new.splitlines():
            if "func.func" in line or "linalg." in line:
                print("   ", line.strip()[:200])
        if not args.check_only:
            f = OUT / f"{name}.mlir"
            if f.exists() and f.read_text() != new:
                sys.exit(f"STOP: grn/{name}.mlir exists with other content")
            f.write_text(new)
    if args.check_only:
        return 0
    GEN.write_text(GEN_SRC.format(roles=repr({k: r for k, (_, r) in KERNELS.items()})))
    text = DG.read_text()
    if LINE not in text:
        shutil.copy(DG, f"{DG}.before_grn_{time.strftime('%Y%m%d_%H%M%S')}")
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
