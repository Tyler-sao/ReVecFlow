# ReVecFlow

Joint graph-level and operator-level optimization of sparse GNN inference on streaming dataflow, built on
[FuseFlow](https://github.com/lrubens/fuseflow-artifact) and evaluated in the cycle-accurate simulator Comal.

- **H**: FuseFlow with tiling. Per kernel, the fastest of scalar and 16/32/64 blocks.
- **G**: matrix re-association and per-layer kernel boundaries (computation graph level).
- **V**: 16-lane vectorization of the feature dimension in the aggregation kernels (operator level, Comal patch 0027).

This repository holds our patches, kernels and scripts. FuseFlow, its compiler and Comal are not
redistributed; they are fetched from their own repositories and patched.

## Method notes

- All simulations run with Comal's HBM model disabled (`COMAL_ENABLE_HBM=0`, set by the scripts).
- Simulated outputs are checked against a float64 reference (atol = rtol = 1e-4). A blocked or vector output with elements
  beyond the tolerance is accepted only where FuseFlow's scalar output also fails, with no smaller error.
- **GALA†** and **GRANII†** are ports of the decision rules of GALA (OOPSLA'25) and GRANII (CGO'26) to the same substrate
  (FuseFlow + tiling, Comal cycles). They are not the authors' implementations and say nothing about the originals' running
  time. GRANII† selects with measured kernel cycles in place of its learned cost model.
- **C+S** (`baselines/cs`) compiles the matrix products with Custard (SAM's TACO fork). On ogbl-collab and ogbn-mag its
  adjacency kernels are recorded as an N² lower bound, not simulated.
- On ogbl-collab and ogbn-mag, H runs the adjacency kernels in scalar mode, because the blocked layout of A is not
  practical at that size.
- The block and vector modes of Comal have no HBM hooks. Enabling HBM charges memory time to scalar runs only, so HBM-on
  numbers of this code are not comparable across modes.

## Layout

| Path | Content |
|---|---|
| `patches/comal` | Comal patches 0003, 0015, 0020-0027 and the `Cargo.lock` used |
| `patches/compiler` | FuseFlow compiler patch 0016 |
| `patches/artifact` | Changes to the artifact's `data_generator.py` and `run_end_to_end.py` |
| `artifact_scripts` | New generator and reference modules installed into the artifact's `scripts/` |
| `kernels` | New MLIR kernels (GIN, SGC, re-associated, A+I) |
| `scripts` | H, H+G and H+V runners, output checks, assembly of the ablation cells |
| `baselines` | C+S, GALA†, GRANII† |
| `experiments` | Ablation, sensitivity sweep (`exp3`), cost analysis (`exp4`) |
| `tools` | `install.sh`, tool versions of our container (`versions.json`) |

## Running

1. Build the FuseFlow artifact container as described in its README, at the pinned versions below. The artifact lists one
   submodule with an SSH address; the first line makes git use HTTPS:

   ```bash
   git config --global url."https://github.com/".insteadOf git@github.com:
   git clone https://github.com/lrubens/fuseflow-artifact.git
   cd fuseflow-artifact && git checkout 36cb1c6 && git submodule update --init --recursive
   docker build -t fuseflow-artifact .
   docker run -d -it -v /path/to/ReVecFlow:/ReVecFlow --name revecflow fuseflow-artifact bash
   ```

2. Install inside the container (builds the patched compiler and Comal, needs network access for Cargo):

   ```bash
   docker exec revecflow bash /ReVecFlow/tools/install.sh
   ```

3. Produce the scalar baselines of the Figure 12 configurations and the kernel inventory (inside the container):

   ```bash
   cd /fuseflow-artifact
   python3 scripts/run_figure12_benchmarks.py --mode medium --no-hbm --output figure12_results.json
   /fuseflow-v3/venv/bin/python -u scripts/run_figure12_newmodels.py --models sgc gin --resume
   cp figure12_results.json results/figure12_newmodels/figure12_newmodels_results.json /fuseflow-v3/baselines/
   /fuseflow-v3/venv/bin/python /fuseflow-v3/scripts/h0_inventory.py
   ```

   On ogbn-mag, FuseFlow's partially fused GraphSAGE kernels fail on empty rows of A. Before the `cp`,
   `run_figure12_newmodels.py --models graphsage --datasets mag --fusion partially_fused` runs the padded copies in
   `kernels/graphsage_repair` and updates `figure12_results.json`.

4. Run inside the container, sequentially (the kernels share `/tmp/op.bin` and `/tmp/data`):

   ```bash
   PY=/fuseflow-v3/venv/bin/python
   $PY /fuseflow-v3/scripts/h_run.py --datasets cora --resume
   $PY /fuseflow-v3/scripts/h_g_run.py --datasets cora --resume
   $PY /fuseflow-v3/scripts/h_v_run.py --datasets cora --resume
   $PY /fuseflow-v3/gala/scripts/gala_assemble.py --datasets cora
   $PY /fuseflow-v3/granii/scripts/grn_run.py --datasets cora --resume
   $PY /fuseflow-v3/granii/scripts/granii_assemble.py --datasets cora
   ```

   Datasets: `cora cora_ml dblp collab mag`. They are downloaded by PyTorch Geometric and OGB and are not included here.
   C+S needs Custard:
   `baselines/cs/cs_build_custard.sh`, then `cs_run.py` and `cs_assemble.py`. The sweep is `exp3_sweep.py --capture`, then
   `exp3_sweep.py --resume`. On ogbl-collab and ogbn-mag, `grn_run.py` needs `--big-scalar`.

Large datasets take hours to days and up to tens of GB of memory.

## Pinned versions

| Component | Commit |
|---|---|
| lrubens/fuseflow-artifact | `36cb1c6` |
| lrubens/fuseflow-compiler | `076b1035404d4a0db1ac95d7e689fe5ebc812533` |
| fuseflow-authors/comal | `0edc9aa01ff06985d8b877bb3e765371935b50cc` |

Tool versions of our container: `tools/versions.json`.

## License

Our code is under the MIT license (`LICENSE`). The patches modify third-party code that keeps its own terms, see `NOTICE`.
