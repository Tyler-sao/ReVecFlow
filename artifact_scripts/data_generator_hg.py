"""V3 H_G kernels (models/hg/): generator classes from FuseFlow's role-based generator (written by make_hg_kernels.py)."""
from data_generator_newmodels import _RoleKernel

HG_ROLES = {'hg_adj_c': ['adj', 'rand'], 'hg_propnorm_c': ['adj_norm', 'rand']}
__all__ = []
for _name, _roles in HG_ROLES.items():
    globals()[_name] = type(_name, (_RoleKernel,), {"ROLES": _roles})
    __all__.append(_name)
