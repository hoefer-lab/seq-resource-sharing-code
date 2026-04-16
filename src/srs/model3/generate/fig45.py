# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Run branching-process simulations and save results for Figs. 4 and 5.

This is the main entry-point for generating the simulation data that
underlies Figs. 4 and 5 of the paper.  For every ``(zeta, tau)`` pair it
launches two batches of simulations — one in the sequential and one in
the parallel resource-sharing regime — extracts steady-state parameters,
and writes the results to a newline-delimited JSON file via :mod:`.save`.

Functions
---------
run_simulation_for_fig45
    Execute a full parameter point and persist the result.
"""

import numpy as np

from srs.lib.enums import ResourceType

from .._core.wrapper import simulate_time_wrapper
from ..default_params import F_THRESHOLD
from .estimate_parameters import estimate_parameter
from .save import save_data


def run_simulation_for_fig45(zeta, tau, n_sample=480, n_gen=15, std=0.1):
    """Simulate data used for Figs. 4 and 5.

    Two sets of *n_sample* branching-process realisations are run:
    one with a low unbinding rate (sequential sharing, ``k_u = 0.01``)
    and one with a high unbinding rate (parallel sharing, ``k_u = 100``).

    The extracted steady-state parameters and advantages are written to
    a text file via :func:`.save.save_data`.

    Parameters
    ----------
    zeta : float
        Resource-scarcity factor.
    tau : float
        Minimal S-phase duration.  Converted to ``rho = 1 / tau`` for
        the simulation wrapper.
    n_sample : int, default=480
        Number of simulated branching processes per sharing mode.
    n_gen : int, default=15
        Approximate number of generations to simulate.
    std : float, default=0.1
        Standard deviation of the D-phase duration distribution.
    """
    resource_type = ResourceType.GROWING_G
    k_b = 1e6
    kus = np.array([1e-2, 1e2])
    rho = 1.0 / tau

    res_seq = simulate_time_wrapper(
        n_sample=n_sample,
        resource_type=resource_type,
        zeta=zeta,
        k_u=kus[0],
        k_b=k_b,
        std=std,
        n_gen=n_gen,
        rho=rho,
        f_threshold=F_THRESHOLD,
    )
    res_par = simulate_time_wrapper(
        n_sample=n_sample,
        resource_type=resource_type,
        zeta=zeta,
        k_u=kus[1],
        k_b=k_b,
        std=std,
        n_gen=n_gen,
        rho=rho,
        f_threshold=F_THRESHOLD,
    )

    parameters = estimate_parameter(res_seq, res_par, tau, zeta)
    parameters["n_gen"] = n_gen
    parameters["std"] = std
    parameters["zeta"] = zeta
    parameters["tau"] = tau
    parameters["k_b"] = k_b
    parameters["resource_type"] = resource_type.value
    parameters["kus"] = [kus[0], kus[1]]
    save_data(
        parameters=parameters,
        n_sample=n_sample,
        resource_type=resource_type,
        n_gen=n_gen,
    )
