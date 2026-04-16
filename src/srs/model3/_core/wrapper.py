# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Parallel / serial wrapper for branching-process simulations.

Provides :func:`simulate_time_wrapper`, which runs multiple independent
realisations of :func:`~model3.core.simulation.simulate_time` — either in
parallel via *joblib* or sequentially — and collects the results into a
list.
"""

# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import multiprocessing

import numpy as np
from joblib import Parallel, delayed

from srs.lib.enums import ODEMethod, ResourceType

from .simulation import simulate_time


# @time
def simulate_time_wrapper(
    n_sample,
    k_u,
    k_b,
    std,
    f_threshold,
    rho=2.0,
    zetas=None,
    zeta=1,
    zeta_std=0,
    resource_type=ResourceType.GROWING_G,
    n_stop=None,
    n_gen=None,
    save_f=False,
    ode_method=ODEMethod.EULER,
    acceleration=1,
    parallel=True,
    scaling=1,
):
    """Run multiple branching-process simulations with varying scarcity.

    Parameters
    ----------
    n_sample : int
        Number of simulations.
    k_u : float
        Unbinding rate [/cycle].
    k_b : float
        Binding rate [/cycle].
    std : float
        Standard deviation of D-phase duration.
    f_threshold : float
        Activated-fraction threshold for the s*-phase definition.
    rho : float, default=2.0
        Inverse S-phase duration (``rho = 1 / tau_s``).
    zetas : ndarray, optional
        Pre-specified scarcity factors (one per simulation).  If *None*,
        values are drawn from ``Uniform(zeta - zeta_std, zeta + zeta_std)``.
    zeta : float, default=1.0
        Mean scarcity factor (used when *zetas* is not supplied).
    zeta_std : float, default=0
        Half-width of the uniform scarcity distribution.
    resource_type : ResourceType, default=ResourceType.GROWING_G
        Resource-growth model.
    n_stop : int, optional
        Number of nuclei at which each simulation stops.
        Either *n_stop* or *n_gen* must be specified.
    n_gen : int, optional
        Approximate number of generations to simulate.
        Ignored when *n_stop* is given.
    save_f : bool, default=False
        Whether to store per-nucleus activated fractions.
    ode_method : ODEMethod, default=ODEMethod.EULER
        ODE solver for the activated-fraction dynamics.
    acceleration : float, default=1
        Time-acceleration factor for the resource model.
    parallel : bool, default=True
        If True, use *joblib* for parallel execution.
    scaling : float, default=1
        Additional time-scaling factor for the resource model.

    Returns
    -------
    results : list of dict
        List of simulation result dictionaries from each run.
    """

    # sample zetas if not given
    if zetas is None:
        rng = np.random.default_rng()
        zetas = rng.uniform(zeta - zeta_std, zeta + zeta_std, n_sample)

    if parallel:
        # set the number of parallel jobs
        if n_sample > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = n_sample

        results = Parallel(n_jobs=n_jobs)(
            delayed(simulate_time)(
                k_u=k_u,
                k_b=k_b,
                std=std,
                rho=rho,
                zeta=z,
                resource_type=resource_type,
                ode_method=ode_method,
                n_stop=n_stop,
                n_gen=n_gen,
                save_f=save_f,
                acceleration=acceleration,
                f_threshold=f_threshold,
                scaling=scaling,
            )
            for z in zetas
        )
    else:
        results = []
        for z in zetas:
            results.append(
                simulate_time(
                    k_u=k_u,
                    k_b=k_b,
                    std=std,
                    rho=rho,
                    zeta=z,
                    resource_type=resource_type,
                    ode_method=ode_method,
                    n_stop=n_stop,
                    n_gen=n_gen,
                    save_f=save_f,
                    acceleration=acceleration,
                    f_threshold=f_threshold,
                    scaling=scaling,
                )
            )

    return results
