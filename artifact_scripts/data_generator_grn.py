"""GRANII-dagger kernels (models/grn/): generator classes from FuseFlow's role-based generator (make_grn_kernels.py)."""
from data_generator_newmodels import _RoleKernel

GRN_ROLES = {'grn_adjloops_f': ['adj_loops', 'x'], 'grn_adjloops_64': ['adj_loops', 'rand']}
__all__ = []
for _name, _roles in GRN_ROLES.items():
    globals()[_name] = type(_name, (_RoleKernel,), {"ROLES": _roles})
    __all__.append(_name)
