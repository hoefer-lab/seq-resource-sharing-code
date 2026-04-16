# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Compute steady-state growth parameters from simulation output.

For each branching-process realisation the time-dependent growth rate
(:math:`\\lambda`), utilisation (:math:`\\eta`), and activated fraction
(:math:`\\phi`) are estimated via weighted polynomial fits, low-pass
filtered, and evaluated near the end of the simulation to obtain a
steady-state value.

Functions
---------
get_steady_state_parameters
    Aggregate steady-state values over an ensemble of realisations.
steady_tau
    Approximate nuclear-cycle duration from ``tau`` and ``zeta``.
get_s_phase_mean_and_std
    Mean and standard deviation of S-phase durations.
"""

import numpy as np

from .._tools import filtered_signal, get_mean_std_dict
from .polyfit import get_polyfit, get_polyfits


def steady_tau(tau, zeta):
    """Rough estimate for the duration of a nuclear cycle.

    In steady state the cycle duration is
    :math:`\\sigma = \\ln 2 / \\lambda_{\\infty}` where
    :math:`\\lambda_{\\infty}` depends on the resource regime:

    * **Saturated** (``zeta / tau > ln 2``): S-phases run at full speed,
      :math:`\\lambda_{\\infty} = \\ln 2`.
    * **Limited** (``zeta / tau <= ln 2``): S-phases are slowed by resource
      scarcity, :math:`\\lambda_{\\infty} = \\zeta / \\tau`.

    Parameters
    ----------
    tau : float
        Minimal S-phase duration.
    zeta : float
        Resource-scarcity factor.

    Returns
    -------
    cycle_duration : float
        Estimated steady-state cycle duration.
    """
    if (zeta / tau) > np.log(2):  # saturated regime
        lambda_inf = np.log(2)
    else:  # limited regime
        lambda_inf = zeta / tau
    # assume that the lifetime distribution is such narrow that it can be seen
    # as a common duration among all nuclei -> tau = np.log(2) / lambda
    cycle_duration = np.log(2) / lambda_inf
    return cycle_duration


def get_s_phase_mean_and_std(res):
    """Return mean and standard deviation of the last 200 S-phases.

    Parameters
    ----------
    res : list of dict
        Simulation results; each dict must contain an ``'s_phases'`` array
        of shape ``(N, 2)`` with start/end times.

    Returns
    -------
    stats : dict
        ``{'mean': float, 'std': float}``
    """
    s_phases = []
    for rs in res:
        # ensure that only nuclei that finished their S-phase are taken into
        # account
        mask = rs["s_phases"][:, 1] != 0
        s_phases_raw = rs["s_phases"][mask][-200:]
        # [:, 0] and [:, 1] encode the beginning and ending of the S-phase
        # respectively; their difference equals the S-phase duration
        s_phases.append(s_phases_raw[:, 1] - s_phases_raw[:, 0])
    s_phases = np.concatenate(s_phases)
    return get_mean_std_dict(s_phases)


def get_steady_state_parameters(res, tau, zeta, suffix=""):
    """Extract steady-state growth rate, utilisation, and activation.

    For every realisation in *res* the observables are computed on a
    time grid via weighted polynomial fits, low-pass filtered, and
    evaluated at ``t_max - 3 * sigma`` (three cycle durations before
    the end).

    Parameters
    ----------
    res : list of dict
        Ensemble of simulation results.
    tau : float
        Minimal S-phase duration.
    zeta : float
        Resource-scarcity factor.
    suffix : str, default=''
        Appended to each key in the returned dictionary (e.g. ``'_seq'``).

    Returns
    -------
    params : dict
        Keys ``'lambda_<suffix>'``, ``'phi_<suffix>'``,
        ``'eta_<suffix>'``, ``'s-phase_duration<suffix>'`` each mapping
        to ``{'mean': float, 'std': float}``.
    """
    # choose sigma such that it corresponds to the expected cycle duration
    sigma = steady_tau(tau, zeta)

    lambda_vals, eta_vals, phi_vals = np.zeros((3, len(res)))
    for i, rs in enumerate(res):
        t_max = np.max(rs["time"])
        times = np.linspace(0, t_max, 100)
        lambda_t, eta_t, phi_t = np.zeros((3, len(times)))
        for j, t in enumerate(times):
            lambda_t[j] = get_polyfit(
                deg=1, y_label="g", sigma=sigma, res=rs, x_max=t_max, x0=t
            )
            eta_t[j], phi_t[j] = get_polyfits(
                deg=0,
                y_labels=["eta", "phi"],
                sigma=sigma,
                res=rs,
                x_max=t_max,
                x0=t,
            )

        # remove oscillating noise
        lambda_t = filtered_signal(y=lambda_t)
        eta_t = filtered_signal(y=eta_t)
        phi_t = filtered_signal(y=phi_t)

        lambda_vals[i], eta_vals[i], phi_vals[i] = [
            np.interp(t_max - 3 * sigma, times, y)
            for y in [lambda_t, eta_t, phi_t]
        ]
    return {
        "lambda" + suffix: get_mean_std_dict(lambda_vals),
        "phi" + suffix: get_mean_std_dict(phi_vals),
        "eta" + suffix: get_mean_std_dict(eta_vals),
        "s-phase_duration" + suffix: get_s_phase_mean_and_std(res),
    }
