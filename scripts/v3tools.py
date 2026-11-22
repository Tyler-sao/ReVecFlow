#!/usr/bin/env python3
"""
FuseFlow V3: compile, translate and read helpers for the scripts in /fuseflow-v3/scripts, for the V3 toolchain
(/fuseflow-v3/toolchain/compiler = baseline compiler + 0016, without V2's 0001/0002 options).

- compile_candidate: sam-opt --linalg-to-sam -cse --stream-parallelizer=..., the generated program after the marker
  (FuseFlow's own invocation; a loop-order table printed before the marker is dropped);
- translate: sam-translate --emit-proto, which writes the baseline's fixed /tmp/op.bin; copied to OUT/op.bin (one
  translation at a time: callers run sequentially);
- coordinates, to_dense, read_input, reconstruct: copied unchanged from V2's run_candidate.py (archived), so the output
  rebuilding is the code validated in V2.
"""
import os
import shutil
import subprocess
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

TOOLS = Path(os.environ.get("V3_TOOLS", "/fuseflow-v3/toolchain/compiler/build/tools"))
MARK = "// BEGIN GENERATED MLIR CODE"
sys.path.insert(0, "/fuseflow-artifact/scripts")
from utils import get_tensors_properties  # noqa: E402,F401  (the parser run_end_to_end.py uses)


class Failed(Exception):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


def compile_candidate(spec, out):
    cmd = [str(TOOLS / "sam-opt"), "--linalg-to-sam", "-cse",
           f"--stream-parallelizer=stream-level={spec['stream_level']} par-factor={spec['par_factor']}", spec["model"]]
    p = subprocess.run(cmd, cwd=TOOLS.parent, capture_output=True, text=True)
    if p.returncode != 0 or MARK not in p.stdout:
        raise Failed("compile-failed", f"sam-opt exit {p.returncode}: {p.stderr.strip()[-500:]}")
    (out / "sam.mlir").write_text(p.stdout[p.stdout.index(MARK):])


def translate(out):
    fixed = Path("/tmp/op.bin")
    fixed.unlink(missing_ok=True)
    with open(out / "proto.txt", "w") as proto:
        p = subprocess.run([str(TOOLS / "sam-translate"), str(out / "sam.mlir"), "--emit-proto"], cwd=TOOLS.parent,
                           stdout=proto, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0 or not fixed.exists():
        raise Failed("translate-failed", f"sam-translate exit {p.returncode}: {p.stderr.strip()[-500:]}")
    shutil.copy(fixed, out / "op.bin")
    return out / "op.bin"


def coordinates(levels, nvals):
    """levels: [(seg, crd)] from the outermost stored level inwards. Returns one coordinate array
    per level, each of length nvals, or raises if the arrays do not form a tree."""
    coords, parents = [], 1
    for depth, (seg, crd) in enumerate(levels):
        seg, crd = np.asarray(seg, dtype=np.int64), np.asarray(crd, dtype=np.int64)
        if len(seg) != parents + 1 or seg[0] != 0 or seg[-1] != len(crd) or np.any(np.diff(seg) < 0):
            raise Failed("malformed-output", f"level {depth}: seg has {len(seg)} entries for {parents} parent "
                         f"positions, starts {seg[:1]}, ends {seg[-1:]}, {len(crd)} coordinates")
        owner = np.repeat(np.arange(parents), np.diff(seg))
        coords = [c[owner] for c in coords] + [crd]
        parents = len(crd)
    if parents != nvals:
        raise Failed("malformed-output", f"{nvals} values for {parents} coordinates")
    return coords


def to_dense(level_arrays, dims, vals, shape):
    """level_arrays: [(seg, crd)] outermost first; dims[i]: tensor dimension of stored level i."""
    coords = coordinates(level_arrays, len(vals))
    for d, c in zip(dims, coords):
        if len(c) and (c.min() < 0 or c.max() >= shape[d]):
            raise Failed("malformed-output", f"dimension {d}: coordinates outside [0, {shape[d]})")
    flat = np.ravel_multi_index([coords[dims.index(d)] for d in range(len(shape))], shape)
    if len(np.unique(flat)) != len(flat):
        raise Failed("malformed-output", "repeated coordinates")
    dense = np.zeros(shape, dtype=np.float64)
    dense.flat[flat] = vals
    return dense


def read_input(b):
    """Dense array of a bound input, read the way Comal reads it: compressed levels through
    seg/crd, dense levels positionally (identity mode order, checked in bind_inputs)."""
    p, shape = b["data"], b["shape"]
    vals = np.loadtxt(f"{p}_mode_vals", ndmin=1).astype(np.float32).astype(np.float64)  # Comal parses f32
    levels, parents = [], 1
    for d, fmt in enumerate(b["format"]):
        if fmt == "compressed":
            seg = np.loadtxt(f"{p}_mode_{d}_seg", dtype=np.int64, ndmin=1)
            crd = np.loadtxt(f"{p}_mode_{d}_crd", dtype=np.int64, ndmin=1)
        elif fmt == "dense":
            seg = np.arange(parents + 1) * shape[d]
            crd = np.tile(np.arange(shape[d]), parents)
        else:
            raise Failed("binding-failed", f"{p}: level format {fmt} not handled")
        levels.append((seg, crd))
        parents = len(crd)
    return to_dense(levels, list(range(len(shape))), vals, shape)


def reconstruct(spec, fibers, vals):
    want = spec["output"]["levels"]
    if len(vals) != 1:
        raise Failed("malformed-output", f"{len(vals)} value writers; one output is supported")
    if sorted(f[1] for f in fibers) != sorted(want):
        raise Failed("malformed-output", f"writers {[(f[0], f[1], f[2]) for f in fibers]} do not match "
                     f"output levels {want}")
    nvals = len(vals[0][1])
    # storage order: the order in which the seg arrays chain (one root fiber, then one fiber per
    # coordinate of the level above, values for the innermost coordinates)
    chains = [order for order in permutations(fibers)
              if len(order[0][3]) == 2 and len(order[-1][4]) == nvals
              and all(len(order[i][3]) == len(order[i - 1][4]) + 1 for i in range(1, len(order)))]
    if len(chains) != 1:
        raise Failed("malformed-output", f"{len(chains)} consistent level orders for writers "
                     f"{[(f[1], len(f[3]), len(f[4])) for f in fibers]} and {nvals} values")
    order = chains[0]
    dims = [want.index(f[1]) for f in order]
    dense = to_dense([(f[3], f[4]) for f in order], dims, np.asarray(vals[0][1], dtype=np.float64),
                     spec["output"]["shape"])
    raw = {f"level{i}_{part}": np.asarray(f[3 if part == "seg" else 4], dtype=np.uint32)
           for i, f in enumerate(order) for part in ("seg", "crd")}
    return dense, dims, raw, np.asarray(vals[0][1], dtype=np.float32), order[0][0]
