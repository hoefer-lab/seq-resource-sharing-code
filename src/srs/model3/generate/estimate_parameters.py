# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Extract steady-state parameters and advantage metrics from simulation output.

Combines :mod:`.steady_state` (per-run growth rate, utilisation, …) with
:mod:`.growth_advantage` (pairwise advantage comparisons) into a single
parameter dictionary suitable for serialisation.

Functions
---------
estimate_parameter
    Build complete result dictionary from sequential and parallel runs.
"""

from .growth_advantage import get_advantages
from .steady_state import get_steady_state_parameters


def estimate_parameter(res_seq, res_par, tau, zeta):
    """Compute all derived quantities for one ``(zeta, tau)`` parameter point.

    Parameters
    ----------
    res_seq : list of dict
        Simulation results under sequential sharing (low ``k_u``).
    res_par : list of dict
        Simulation results under parallel sharing (high ``k_u``).
    tau : float
        Minimal S-phase duration.
    zeta : float
        Resource-scarcity factor.

    Returns
    -------
    results : dict
        Merged dictionary containing growth/utilisation advantages and
        steady-state parameters for both sharing modes.
    """
    results = {
        "growth_advantage_n": get_advantages("n", res_seq, res_par),
        "growth_advantage_g": get_advantages("g", res_seq, res_par),
        "utilization_advantage": get_advantages("eta", res_seq, res_par),
    }

    results_seq = get_steady_state_parameters(
        res=res_seq, tau=tau, zeta=zeta, suffix="_seq"
    )
    results_par = get_steady_state_parameters(
        res=res_par, tau=tau, zeta=zeta, suffix="_par"
    )

    return results | results_seq | results_par
