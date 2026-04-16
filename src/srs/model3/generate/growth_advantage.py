# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Growth and utilisation advantage calculations.

Computes pairwise advantages between sequential and parallel
resource-sharing simulations.  A positive value means the sequential
mode outperforms the parallel mode.

Functions
---------
get_advantages
    Aggregate advantage over all pairs of sequential/parallel runs.
get_growth_advantage_i
    Growth-rate advantage for a single (seq, par) pair.
get_utilization_advantage_i
    Time-integrated utilisation advantage for a single pair.
"""

import itertools

import numpy as np
from scipy.integrate import simpson

from .._tools import get_mean_std_dict

N_ADVANTAGE = 20


def get_utilization_advantage_i(r_seq, r_par):
    """Compute the time-integrated utilisation advantage for one pair.

    The utilisation advantage is the difference of the time-integrals of
    :math:`\\eta(t)` between the sequential and parallel runs, averaged
    over the last *N_ADVANTAGE* time points.

    Parameters
    ----------
    r_seq, r_par : dict
        Single branching-process results with ``'time'`` and ``'eta'``.

    Returns
    -------
    advantage : float
    """
    t_max = np.min([r_seq["time"].max(), r_par["time"].max()])
    x = np.linspace(0, t_max, 1000)
    utilization_advantage = np.zeros(N_ADVANTAGE)
    for i, t in enumerate(x[-N_ADVANTAGE:]):
        x_masked = x[x <= t]
        y_seq = np.interp(x=x_masked, xp=r_seq["time"], fp=r_seq["eta"])
        y_par = np.interp(x=x_masked, xp=r_par["time"], fp=r_par["eta"])
        utilization_advantage[i] = simpson(y=y_seq, x=x_masked) - simpson(
            y=y_par, x=x_masked
        )
    return np.mean(utilization_advantage)


def get_advantages(y_label, res_seq, res_par):
    """Compute the mean advantage over all (seq, par) pairs.

    For ``y_label='eta'`` the utilisation advantage is computed; for
    ``'n'`` or ``'g'`` the growth-rate advantage is used.

    Parameters
    ----------
    y_label : str
        Observable label: ``'n'``, ``'g'``, or ``'eta'``.
    res_seq : list of dict
        Sequential-sharing simulation results.
    res_par : list of dict
        Parallel-sharing simulation results.

    Returns
    -------
    stats : dict
        ``{'mean': float, 'std': float}``
    """

    def get_advantage_i(r_seq, r_par):
        if y_label == "eta":
            return get_utilization_advantage_i(r_seq, r_par)
        else:
            return get_growth_advantage_i(y_label, r_seq, r_par)

    advantages = np.zeros(len(res_seq) ** 2)
    for i, (r_seq, r_par) in enumerate(itertools.product(res_seq, res_par)):
        advantages[i] = get_advantage_i(r_seq, r_par)
    return get_mean_std_dict(advantages)


def get_growth_advantage_i(y_label, r_seq, r_par):
    """Compute the growth-rate advantage for one (seq, par) pair.

    The advantage is defined as the mean difference of
    ``log(y_seq) - log(y_par)`` over the last *N_ADVANTAGE* time points.

    Parameters
    ----------
    y_label : str
        Observable label (``'n'`` or ``'g'``).
    r_seq, r_par : dict
        Single branching-process results.

    Returns
    -------
    advantage : float
    """
    t_max = np.min([r_seq["time"].max(), r_par["time"].max()])
    x = np.linspace(0, t_max, 1000)
    y = np.interp(x, r_seq["time"], np.log(r_seq[y_label])) - np.interp(
        x, r_par["time"], np.log(r_par[y_label])
    )
    return np.mean(y[-N_ADVANTAGE:])
