"""
Data generator classes for the new Figure 12 model baselines (SGC first; GIN and APPNP follow).

Imported at the end of data_generator.py by one appended line (`from data_generator_newmodels import *`), so that
run_end_to_end.py finds a class by MLIR file name exactly as for GCN/GraphSAGE. Only the names in __all__ are
exported, so no existing name of data_generator is shadowed.

Conventions copied from the existing GNN classes (data_generator.py):
- dataset loading: load_dataset_memory_efficient; real features dumped dense ("UNC") through MatrixGenerator
  with tensor=features, as data_gen does;
- the raw adjacency is built exactly as data_gen builds it: dense 0/1 when N*N*4 bytes <= 10 GB, otherwise a
  SciPy COO matrix whose duplicate edges are summed (Collab);
- weights: MatrixGenerator(..., format="CSF") with its default sparsity (data_generator's GNN classes take no
  weight_sparsity argument, so -sp has no effect, as in Figure 12); biases dense; synthetic intermediates of
  later kernels are random MatrixGenerator tensors, as one_layer_gcn2 / gcn_linear1_bias do;
- the global numpy/random seeds set at data_generator's import (4) are used as they are.
New:
- normalized adjacencies (SGC: (D+I)^-1/2 (A+I) (D+I)^-1/2, degrees = row sums of A+I) are computed offline
  from that raw A and written in the same CSF layout as dump_sparse_csr_fast, values as %.9g (exact f32);
  the time is written to <data dir>/preprocess.json;
- tensor shapes are taken from the SAM header's placeholder shapes, mapped to the dataset's sizes
  (placeholders: 1767 = N nodes, 50 = F features, 121 = C classes, 64 = H hidden width), so every tensor gets
  the orientation the compiled program expects.
"""
import json
import os
import time

import numpy as np
import scipy.sparse as sp

from data_generator import MatrixGenerator, load_dataset_memory_efficient

__all__ = ["sgc_sparse", "sgc_propagate", "sgc_classify", "sgc_prop1", "sgc_prop2", "sgc_linear_mul",
           "sgc_linear_bias", "sgc_softmax", "sgc_sparse_logits", "sgc_classify_logits"]  # GIN and APPNP classes are appended to __all__ below

PLACEHOLDER = {1767: "N", 50: "F", 121: "C", 64: "H"}


def _argno(a):
    return a if isinstance(a, int) else None


class _NewModelBase:
    """Shared parts: dataset, sizes, header bookkeeping, writers. Subclasses define ROLES and generate_data."""
    hidden = 64

    def __init__(self, args):
        self.dataset_name, self.name = args.inDataset, args.inData
        self.root_dir = os.environ["DATA_PATH"]
        self.data, self.num_classes = load_dataset_memory_efficient(self.dataset_name, self.root_dir, self.name)
        x = self.data.x.numpy() if hasattr(self.data.x, "numpy") else self.data.x
        self.features = np.asarray(x, dtype=np.float32)
        edge = self.data.edge_index.numpy()
        self.edge_index = edge
        n = self.data.num_nodes
        self.N = n if n is not None and n >= int(edge.max()) + 1 else max(int(edge.max()) + 1, self.features.shape[0])
        self.sizes = {"N": self.N, "F": self.features.shape[1], "C": self.num_classes, "H": self.hidden}
        self.tensors = []          # (name, arg, mode, header shape, format), in header order
        self.file_path = None      # set by run_end_to_end.py before generate_data

    def collect_names(self, tensor_name, input_arg, mode, shape=None, format=None):
        self.tensors.append({"name": tensor_name, "arg": _argno(input_arg), "mode": [int(m) for m in mode],
                             "shape": [int(s) for s in shape], "format": format})

    # ---- header helpers
    def shape_of(self, t):
        try:
            return tuple(self.sizes[PLACEHOLDER[s]] for s in t["shape"])
        except KeyError:
            raise ValueError(f"{t['name']}: header shape {t['shape']} has a size that is not a placeholder "
                             f"{sorted(PLACEHOLDER)}")

    def arg(self, i):
        found = [t for t in self.tensors if t["arg"] == i]
        if not found:
            raise ValueError(f"no header tensor for argument {i}: {self.tensors}")
        return found

    def consts(self, rank):
        return [t for t in self.tensors if t["arg"] is None and len(t["mode"]) == rank]

    def dump_dir(self):
        return os.getcwd() + self.file_path

    # ---- writers
    def write_dense(self, t, array=None):
        shape = self.shape_of(t)
        if array is not None and tuple(array.shape) != shape:
            raise ValueError(f"{t['name']}: array {array.shape} but header asks {shape}")
        kw = {"tensor": array} if array is not None else {}
        m = MatrixGenerator(t["name"], shape, mode_ordering=t["mode"], dump_dir=self.dump_dir(), format="UNC", **kw)
        m.dump_outputs("UNC")

    def write_weight(self, t):
        """Random CSF weight (MatrixGenerator, its default sparsity; same random draws as before). Written with every
        row stored and one explicit zero (column 0) in each row without nonzeros: Comal mishandles empty fibers
        (a fused group with an H x C weight having empty rows fails, KarateClub 2026-10-13), so the weight layout
        is padded for every kernel of the new models. Values are unchanged; padded rows are recorded."""
        m = MatrixGenerator(t["name"], self.shape_of(t), mode_ordering=t["mode"], dump_dir=self.dump_dir(), format="CSF")
        w = np.asarray(m.get_matrix(), dtype=np.float32).reshape(self.shape_of(t))
        padded = self.write_sparse_matrix(t, sp.csr_matrix(w), pad_empty_rows=True)
        self.padded_rows = getattr(self, "padded_rows", {}) | {t["name"]: padded}

    def write_bias(self, t):
        m = MatrixGenerator(t["name"], self.shape_of(t), [0], dump_dir=self.dump_dir(), format="UNC")
        m.dump_outputs("UNC")

    def write_sparse_matrix(self, t, csr, pad_empty_rows=False):
        """Same layout as data_generator.dump_sparse_csr_fast (compressed rows, compressed columns), in the
        header's mode order, values %.9g. pad_empty_rows: every stored row of the (mode-ordered) matrix gets at least
        one entry, an explicit zero at column 0. Returns the number of padded rows."""
        if t["mode"] == [1, 0]:
            csr = csr.T
        elif t["mode"] != [0, 1]:
            raise ValueError(f"{t['name']}: mode order {t['mode']}")
        csr = sp.csr_matrix(csr)
        csr.eliminate_zeros()
        csr.sort_indices()
        padded = 0
        if pad_empty_rows:
            empty = np.flatnonzero(np.diff(csr.indptr) == 0)
            padded = len(empty)
            if padded:
                coo = csr.tocoo()
                rows = np.concatenate([coo.row, empty])
                cols = np.concatenate([coo.col, np.zeros(padded, dtype=coo.col.dtype)])
                vals = np.concatenate([coo.data, np.zeros(padded, dtype=coo.data.dtype)])
                order = np.lexsort((cols, rows))
                counts = np.bincount(rows, minlength=csr.shape[0])
                indptr = np.concatenate([[0], np.cumsum(counts)])
                csr = sp.csr_matrix((vals[order], cols[order], indptr), shape=csr.shape)   # keeps the explicit zeros
        d, name = self.dump_dir(), "tensor_" + t["name"]
        os.makedirs(d, exist_ok=True)
        rows, cols = csr.shape
        row_nnz = np.diff(csr.indptr)
        nonempty = np.flatnonzero(row_nnz > 0)
        with open(d + name + "_mode_shape", "w") as f:
            f.write(f"{rows}\n{cols}\n")
        np.savetxt(d + name + "_mode_vals", csr.data.astype(np.float32), fmt="%.9g")
        np.savetxt(d + name + "_mode_0_seg", [0, len(nonempty)], fmt="%d")
        np.savetxt(d + name + "_mode_0_crd", nonempty, fmt="%d")
        seg1 = np.zeros(len(nonempty) + 1, dtype=np.int64)
        seg1[1:] = np.cumsum(row_nnz[nonempty])
        np.savetxt(d + name + "_mode_1_seg", seg1, fmt="%d")
        np.savetxt(d + name + "_mode_1_crd", csr.indices, fmt="%d")
        return padded

    # ---- graphs
    def raw_adjacency(self):
        """A exactly as data_gen builds it (0/1 dense path; summed duplicates on the large-graph path)."""
        e, n = self.edge_index, self.N
        if n * n * 4 / 1024 ** 3 > 10:
            return sp.coo_matrix((np.ones(e.shape[1], dtype=np.float32), (e[0], e[1])), shape=(n, n)).tocsr()
        a = sp.coo_matrix((np.ones(e.shape[1], dtype=np.float32), (e[0], e[1])), shape=(n, n)).tocsr()
        a.data[:] = 1.0                                     # dense path: adj[i, j] = 1, duplicates collapse
        return a

    def normalized_adjacency(self):
        """(D+I)^-1/2 (A+I) (D+I)^-1/2, degrees = row sums of A+I (SGC AugNormAdj; APPNP calc_A_hat)."""
        start = time.time()
        a = (self.raw_adjacency() + sp.eye(self.N, dtype=np.float32, format="csr")).tocsr()
        deg = np.asarray(a.sum(axis=1)).ravel().astype(np.float64)
        dinv = np.where(deg > 0, deg ** -0.5, 0.0)
        s = (sp.diags(dinv) @ a.astype(np.float64) @ sp.diags(dinv)).tocsr().astype(np.float32)
        self.preprocess_seconds = time.time() - start
        return s

    def record(self, **info):
        with open(self.dump_dir() + "preprocess.json", "w") as f:
            json.dump({"model_class": type(self).__name__, "dataset": [self.dataset_name, self.name],
                       "sizes": self.sizes, "tensors": self.tensors, **info}, f, indent=1)


# ---------------------------------------------------------------- SGC
class sgc_sparse(_NewModelBase):
    """Fully fused: arg0 X (real), arg1 S; W (F x C, CSF), b (C)."""
    def generate_data(self):
        self.write_dense(self.arg(0)[0], self.features)
        s = self.normalized_adjacency()
        for t in self.arg(1):
            self.write_sparse_matrix(t, s)
        for t in self.consts(2):
            self.write_weight(t)
        for t in self.consts(1):
            self.write_bias(t)
        self.record(preprocess_seconds=self.preprocess_seconds)


class sgc_propagate(_NewModelBase):
    """Partially fused stage 1: arg0 X (real), arg1 S."""
    def generate_data(self):
        self.write_dense(self.arg(0)[0], self.features)
        s = self.normalized_adjacency()
        for t in self.arg(1):
            self.write_sparse_matrix(t, s)
        self.record(preprocess_seconds=self.preprocess_seconds)


class sgc_classify(_NewModelBase):
    """Partially fused stage 2: arg0 P (synthetic N x F, as Figure 12's layer-2 kernels); W, b."""
    def generate_data(self):
        self.write_dense(self.arg(0)[0])
        for t in self.consts(2):
            self.write_weight(t)
        for t in self.consts(1):
            self.write_bias(t)
        self.record()


class sgc_prop1(_NewModelBase):
    """Unfused propagation 1: arg0 S, arg1 X (real), as gcn_adj_x1."""
    def generate_data(self):
        s = self.normalized_adjacency()
        for t in self.arg(0):
            self.write_sparse_matrix(t, s)
        self.write_dense(self.arg(1)[0], self.features)
        self.record(preprocess_seconds=self.preprocess_seconds)


class sgc_prop2(_NewModelBase):
    """Unfused propagation 2: arg0 S, arg1 P1 (synthetic N x F)."""
    def generate_data(self):
        s = self.normalized_adjacency()
        for t in self.arg(0):
            self.write_sparse_matrix(t, s)
        self.write_dense(self.arg(1)[0])
        self.record(preprocess_seconds=self.preprocess_seconds)


class sgc_linear_mul(_NewModelBase):
    """Unfused classifier product: arg0 P2 (synthetic N x F), arg1 W (F x C, CSF)."""
    def generate_data(self):
        self.write_dense(self.arg(0)[0])
        self.write_weight(self.arg(1)[0])
        self.record()


class sgc_linear_bias(_NewModelBase):
    """Unfused classifier bias: arg0 (synthetic N x C), arg1 b (C)."""
    def generate_data(self):
        self.write_dense(self.arg(0)[0])
        self.write_bias(self.arg(1)[0])
        self.record()


class sgc_softmax(_NewModelBase):
    """Unfused softmax: arg0 (synthetic N x C)."""
    def generate_data(self):
        self.write_dense(self.arg(0)[0])
        self.record()


# check-only kernels (not part of any configuration): the same programs returning the values before the softmax,
# so that a saturated softmax cannot hide a wrong product; same generation order, hence the same data
class sgc_sparse_logits(sgc_sparse):
    pass


class sgc_classify_logits(sgc_classify):
    pass


# ---------------------------------------------------------------- GIN and APPNP
# Every parameter of these kernels is a numbered function argument; KERNEL_ROLES gives the role of each argument
# (written by emit_gin_appnp.py together with the MLIR). The %argout buffer of unfused kernels has no role and no data.
#   x        real features (dense)                  adj       raw A, as data_gen builds it (GIN)
#   adj_loops  A + I (diagnostic kernels only)
#   adj_norm (D+I)^-1/2 (A+I) (D+I)^-1/2 (APPNP)     rand      synthetic input of a later kernel (random, dense)
#   weight   random CSF, MatrixGenerator default     bias      random dense vector, MatrixGenerator default
#   bn_mu, bn_gamma, bn_beta: random dense vectors (MatrixGenerator default)
#   bn_invstd: s = 1/sqrt(var + 1e-5) with var ~ uniform(0.5, 2.0), computed offline (the variances are kept in
#              <data dir>/bn_var_<tensor>.npy for the record)
KERNEL_ROLES = {
    'gin_sparse': ['x', 'adj', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta', 'weight', 'bias'],
    'gin_sparse_logits': ['x', 'adj', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta', 'weight', 'bias'],
    'one_layer_gin': ['x', 'adj', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta'],
    'one_layer_gin2': ['rand', 'adj', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta', 'weight', 'bias'],
    'one_layer_gin2_logits': ['rand', 'adj', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta', 'weight', 'bias'],
    'gin_adj_h1': ['adj', 'x'],
    'gin_selfadd1': ['rand', 'x'],
    'gin_mlp1_lin1_mul': ['rand', 'weight'],
    'gin_mlp1_lin1_bias': ['rand', 'bias'],
    'gin_mlp1_bn1': ['rand', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta'],
    'gin_mlp1_relu1': ['rand'],
    'gin_mlp1_lin2_mul': ['rand', 'weight'],
    'gin_mlp1_lin2_bias': ['rand', 'bias'],
    'gin_bn_out1': ['rand', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta'],
    'gin_relu_out1': ['rand'],
    'gin_adj_h2': ['adj', 'rand'],
    'gin_selfadd2': ['rand', 'rand'],
    'gin_mlp2_lin1_mul': ['rand', 'weight'],
    'gin_mlp2_lin1_bias': ['rand', 'bias'],
    'gin_mlp2_bn1': ['rand', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta'],
    'gin_mlp2_relu1': ['rand'],
    'gin_mlp2_lin2_mul': ['rand', 'weight'],
    'gin_mlp2_lin2_bias': ['rand', 'bias'],
    'gin_softmax': ['rand'],
    'appnp_sparse': ['x', 'adj_norm', 'weight', 'weight'],
    'appnp_sparse_logits': ['x', 'adj_norm', 'weight', 'weight'],
    'appnp_mlp': ['x', 'weight', 'weight'],
    'appnp_step': ['rand', 'adj_norm', 'rand'],
    'appnp_step_last': ['rand', 'adj_norm', 'rand'],
    'appnp_step_last_logits': ['rand', 'adj_norm', 'rand'],
    'appnp_lin1_mul': ['x', 'weight'],
    'appnp_relu': ['rand'],
    'appnp_lin2_mul': ['rand', 'weight'],
    'appnp_prop': ['adj_norm', 'rand'],
    'appnp_mix': ['rand', 'rand'],
    'appnp_softmax': ['rand'],
    'gin_diag_l2_noself': ['rand', 'adj', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta', 'weight', 'bias'],
    'gin_diag_l2_loops': ['rand', 'adj_loops', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta', 'weight', 'bias'],
    'gin_diag_l2_hidden': ['rand', 'adj', 'weight', 'bias', 'bn_mu', 'bn_invstd', 'bn_gamma', 'bn_beta', 'weight', 'bias'],
    'appnp_diag_mlp_1step': ['x', 'adj_norm', 'weight', 'weight'],
    'appnp_diag_mlp_2step': ['x', 'adj_norm', 'weight', 'weight'],
    'appnp_diag_2steps': ['rand', 'adj_norm', 'rand'],
    'sgc_prop_classify': ['adj_norm', 'rand', 'weight', 'bias'],
    'sgc_prop_linear': ['adj_norm', 'rand', 'weight', 'bias'],
    'sgc_prop_mul': ['adj_norm', 'rand', 'weight'],
    'sgc_bias_softmax': ['rand', 'bias'],
    'sgc_linear': ['rand', 'weight', 'bias'],
}


class _RoleKernel(_NewModelBase):
    ROLES = []

    def generate_data(self):
        info = {}
        for i, role in enumerate(self.ROLES):
            ts = self.arg(i)
            if role == "x":
                self.write_dense(ts[0], self.features)
            elif role in ("adj", "adj_norm", "adj_loops"):
                a = {"adj": self.raw_adjacency, "adj_norm": self.normalized_adjacency,
                     "adj_loops": lambda: (self.raw_adjacency() + sp.eye(self.N, dtype=np.float32, format="csr")).tocsr()}[role]()
                for t in ts:
                    self.write_sparse_matrix(t, a)
                if role == "adj_norm":
                    info["preprocess_seconds"] = self.preprocess_seconds
            elif role == "rand":
                self.write_dense(ts[0])
            elif role == "weight":
                self.write_weight(ts[0])
            elif role in ("bias", "bn_mu", "bn_gamma", "bn_beta"):
                self.write_bias(ts[0])
            elif role == "bn_invstd":
                t = ts[0]
                var = np.random.uniform(0.5, 2.0, self.shape_of(t)).astype(np.float32)
                invstd = (np.float32(1) / np.sqrt(var + np.float32(9.99999974e-6))).astype(np.float32)
                np.save(self.dump_dir() + f"bn_var_{t['name']}.npy", var)
                m = MatrixGenerator(t["name"], self.shape_of(t), [0], dump_dir=self.dump_dir(), format="UNC", tensor=invstd)
                m.dump_outputs("UNC")
            else:
                raise ValueError(f"{type(self).__name__}: unknown role {role}")
        self.record(roles=self.ROLES, padded_weight_rows=getattr(self, "padded_rows", {}), **info)


for _name, _roles in KERNEL_ROLES.items():
    globals()[_name] = type(_name, (_RoleKernel,), {"ROLES": _roles})
    __all__.append(_name)


# ---------------------------------------------------------------- GraphSAGE partially fused on MAG: repair (2026-10-13)
# FuseFlow's one_layer_graphsage / one_layer_graphsage2 fail on OGB-MAG in Figure 12 (Comal: "Unexpected end of
# stream"); MAG's directed citation graph has 118,465 nodes without out-edges, so A has empty rows, and the layer
# walks the rows of X and A together. These classes reuse FuseFlow's generator classes unchanged and only
# post-process A's files: optionally empty rows 0..99 (diagnostic, to reproduce the failure on a small graph),
# optionally store one explicit zero in every empty row (the repair). The kernels are identical copies of FuseFlow's.
from data_generator import one_layer_graphsage as _fuseflow_sage_l1, one_layer_graphsage2 as _fuseflow_sage_l2  # noqa: E402


def _rewrite_adjacency(gen, holes, pad):
    d = os.getcwd() + gen.file_path
    prefix = d + "tensor_" + gen.adjname
    ints = lambda f: np.fromfile(f, sep="\n", dtype=np.int64)
    rows, cols = [int(v) for v in open(prefix + "_mode_shape").read().split()]
    vals = np.fromfile(prefix + "_mode_vals", sep="\n", dtype=np.float64).astype(np.float32)
    crd0, seg1, crd1 = ints(prefix + "_mode_0_crd"), ints(prefix + "_mode_1_seg"), ints(prefix + "_mode_1_crd")
    a = sp.csr_matrix((vals, (np.repeat(crd0, np.diff(seg1)), crd1)), shape=(rows, cols))
    before = int((np.diff(a.indptr) == 0).sum())
    if holes:
        a = a.tolil()
        a[:holes, :] = 0
        a = a.tocsr()
    a.eliminate_zeros()
    empty = int((np.diff(a.indptr) == 0).sum())
    shim = _NewModelBase.__new__(_NewModelBase)
    shim.dump_dir = lambda: d
    padded = shim.write_sparse_matrix({"name": gen.adjname, "mode": [0, 1]}, a, pad_empty_rows=pad)
    with open(d + "preprocess.json", "w") as f:
        json.dump({"model_class": type(gen).__name__, "fuseflow_class": type(gen).__mro__[1].__name__,
                   "adjacency": gen.adjname, "empty_rows_generated": before, "rows_emptied": holes,
                   "empty_rows_written": empty, "padded_rows": padded}, f, indent=1)


class sage_repair_l1(_fuseflow_sage_l1):
    def generate_data(self):
        super().generate_data()
        _rewrite_adjacency(self, 0, True)


class sage_repair_l1_holes(_fuseflow_sage_l1):
    def generate_data(self):
        super().generate_data()
        _rewrite_adjacency(self, 100, False)


class sage_repair_l1_holes_pad(_fuseflow_sage_l1):
    def generate_data(self):
        super().generate_data()
        _rewrite_adjacency(self, 100, True)


class sage_repair_l2(_fuseflow_sage_l2):
    def generate_data(self):
        super().generate_data()
        _rewrite_adjacency(self, 0, True)


__all__ += ["sage_repair_l1", "sage_repair_l1_holes", "sage_repair_l1_holes_pad", "sage_repair_l2"]


# ---------------------------------------------------------------- CSR layer kernels (2026-10-14)
# scripts/csr_layers.json (written by make_csr_kernels.py) names, for each kernel in tests/models/csr_layers/, the
# source kernel it copies. Its generator class subclasses the source kernel's generator class, so the data are generated
# exactly as for the source (same calls, same random draws); afterwards the files are rewritten in the layout the
# kernel's SAM header asks for:
#   (dense, compressed) tensors (X and the weights of a _csr kernel): rows dense, columns compressed;
#   (compressed, compressed) 2-D tensors (the adjacency): unchanged except that empty rows are padded;
# in both, a row without nonzeros stores one explicit zero at column 0 (Comal handles no empty fiber). Values are
# unchanged (read back as float32, written %.9g). The _orig copy (GCN) gets the same padding of A and keeps FuseFlow's
# formats, so its output checks the _csr copy's on identical values.
def _read_stored_matrix(prefix):
    """(rows, cols) matrix of a mode-(0,1) tensor's files as a CSR matrix with explicit zeros removed."""
    rows, cols = [int(v) for v in open(prefix + "_mode_shape").read().split()[:2]]
    vals = np.fromfile(prefix + "_mode_vals", sep="\n", dtype=np.float64).astype(np.float32)
    ints = lambda f: np.fromfile(f, sep="\n", dtype=np.int64)
    if os.path.exists(prefix + "_mode_0_seg"):
        crd0, seg1, crd1 = ints(prefix + "_mode_0_crd"), ints(prefix + "_mode_1_seg"), ints(prefix + "_mode_1_crd")
        m = sp.csr_matrix((vals, (np.repeat(crd0, np.diff(seg1)), crd1)), shape=(rows, cols))
    elif os.path.exists(prefix + "_mode_1_seg"):
        seg1, crd1 = ints(prefix + "_mode_1_seg"), ints(prefix + "_mode_1_crd")
        m = sp.csr_matrix((vals, crd1, seg1), shape=(rows, cols))
    else:
        if vals.size != rows * cols:
            raise ValueError(f"{prefix}: {vals.size} dense values for shape {rows} x {cols}")
        m = sp.csr_matrix(vals.reshape(rows, cols))
    m.eliminate_zeros()
    m.sort_indices()
    return m


def _write_csr_layout(prefix, m):
    """Rows dense, columns compressed; empty rows padded with an explicit zero at column 0. Returns padded rows."""
    rows, cols = m.shape
    counts = np.diff(m.indptr)
    empty = np.flatnonzero(counts == 0)
    seg = np.zeros(rows + 1, dtype=np.int64)
    new_counts = np.maximum(counts, 1)
    seg[1:] = np.cumsum(new_counts)
    crd = np.zeros(seg[-1], dtype=np.int64)
    vals = np.zeros(seg[-1], dtype=np.float32)
    keep = np.repeat(seg[:-1], counts) + (np.arange(m.nnz) - np.repeat(m.indptr[:-1], counts))
    crd[keep] = m.indices
    vals[keep] = m.data
    for f in (prefix + "_mode_0_seg", prefix + "_mode_0_crd"):
        if os.path.exists(f):
            os.remove(f)
    with open(prefix + "_mode_shape", "w") as f:
        f.write(f"{rows}\n{cols}\n")
    np.savetxt(prefix + "_mode_vals", vals, fmt="%.9g")
    np.savetxt(prefix + "_mode_1_seg", seg, fmt="%d")
    np.savetxt(prefix + "_mode_1_crd", crd, fmt="%d")
    return int(len(empty))


class _CSRLayout:
    """Mixin: records the header formats, generates the source kernel's data, rewrites the files."""
    def collect_names(self, *a, **k):
        self._csr_hdr = getattr(self, "_csr_hdr", [])
        name, mode = a[0], list(a[2]) if len(a) > 2 else list(k.get("mode"))
        fmt = a[4] if len(a) > 4 else k.get("format")
        self._csr_hdr.append({"name": name, "mode": [int(x) for x in mode], "format": list(fmt) if fmt else None})
        return super().collect_names(*a, **k)

    def generate_data(self, *a, **k):
        r = super().generate_data(*a, **k)
        d = os.getcwd() + self.file_path
        done = {}
        for t in self._csr_hdr:
            prefix = d + "tensor_" + t["name"]
            if not os.path.exists(prefix + "_mode_vals") or not t["format"] or len(t["format"]) != 2:
                continue
            if t["format"] not in (["dense", "compressed"], ["compressed", "compressed"]):
                continue
            if t["mode"] != [0, 1]:
                raise ValueError(f"{t['name']}: mode order {t['mode']} not handled for the CSR layout")
            m = _read_stored_matrix(prefix)
            if t["format"] == ["dense", "compressed"]:
                done[t["name"]] = {"layout": "csr", "nonzeros": int(m.nnz), "padded_rows": _write_csr_layout(prefix, m)}
            else:
                shim = _NewModelBase.__new__(_NewModelBase)
                shim.dump_dir = lambda: d
                done[t["name"]] = {"layout": "compressed-compressed", "nonzeros": int(m.nnz),
                                   "padded_rows": shim.write_sparse_matrix({"name": t["name"], "mode": [0, 1]}, m, pad_empty_rows=True)}
        info_file = d + "preprocess.json"
        info = json.load(open(info_file)) if os.path.exists(info_file) else {"model_class": type(self).__name__}
        info["csr_layout"] = {"source_class": type(self).__mro__[2].__name__, "tensors": done}
        with open(info_file, "w") as f:
            json.dump(info, f, indent=1)
        return r


def _csr_layer_classes():
    import data_generator as _dg
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csr_layers.json")
    if not os.path.exists(path):
        return []
    made = []
    for kernel, spec in json.load(open(path)).items():
        base = globals().get(spec["source"]) or getattr(_dg, spec["source"])
        globals()[kernel] = type(kernel, (_CSRLayout, base), {})
        made.append(kernel)
    return made


__all__ += _csr_layer_classes()


# ---------------------------------------------------------------- GIN partially fused on OGB-MAG: repair (2026-10-14)
# Stage 3: one_layer_gin and one_layer_gin2 fail on MAG with Comal "Stop tokens don't match" (binary.rs:106), the
# failure reproduced for GraphSAGE by emptying rows of A on Cora; MAG's A has 118,465 empty rows and the GIN layer
# combines A.h with h row by row. The copies in tests/models/gin_repair/ are identical to the kernels; their data are
# generated by the same classes (same values), then A's empty rows are padded with one explicit zero each
# (_CSRLayout on an unchanged kernel: no (dense, compressed) tensor, so only the padding applies; the weights are
# already written padded).
class gin_repair_l1(_CSRLayout, globals()["one_layer_gin"]):
    pass


class gin_repair_l2(_CSRLayout, globals()["one_layer_gin2"]):
    pass


class gin_repair_l2_logits(_CSRLayout, globals()["one_layer_gin2_logits"]):
    pass


__all__ += ["gin_repair_l1", "gin_repair_l2", "gin_repair_l2_logits"]
