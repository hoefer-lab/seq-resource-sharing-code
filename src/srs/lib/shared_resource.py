# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Resource models for the activated-fraction ODE.

Each function returns the total available resource :math:`R_0` given
the scarcity parameter :math:`\\zeta` and the current system state.
All are Numba-JIT compiled for use inside the simulation loop.
"""

# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
from numba import njit


@njit
def constant_resource(zeta):
    """Return value of constant resource."""
    return zeta


@njit
def growing_count_resource(zeta, count):
    """Return value of resource linear to system size."""
    return zeta * count


@njit
def pcna1_like_resource(t, zeta):
    """Return value of a PCNA1-like growing resource."""
    # Fitting PCNA1 data with sigmoidal curve results inhere:
    t0 = (-8.30040304e02 + -6.73730254e02) / 2
    k = (5.30145076e-03 + 1.03882609e-02) / 2
    # Klaus data provide typical duration (data_exp['schizo'].mean().sum())
    t_first_to_egress = 853.6363636363636
    result = zeta / (1 + np.exp(-k * (t - (t0 + t_first_to_egress))))
    return result


@njit(error_model="numpy")
def get_resource(resource_type, zeta, count, time):
    """Return current catalysis-like resource.

    Parameters
    ----------
    resource_type : str
        Indicating the resource type (see enum ResourceType)
    zeta : float
        Sparsity factor
    count : float
        Number of current system size:
            resource_type='zeta_n': count corresponds to nuclear number
            resource_type='zeta_g': count corresponds to genome count

    Returns
    -------
    r0 : float
        Catalysis-like resource

    """
    # choose resource type
    if resource_type == "const":
        return constant_resource(zeta)
    elif resource_type in ["zeta_n", "zeta_g"]:
        return growing_count_resource(zeta, count)
    elif resource_type == "pcna1":
        return pcna1_like_resource(time, zeta)
    return 0
