# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
r"""ODE solvers for the activated-fraction dynamics.

The central equation is

.. math::
    \frac{\mathrm{d} f_i}{\mathrm{d} t}
    = k_b \bigl(R_0 - \sum_j f_j\bigr)(1 - f_i) - k_u f_i

where :math:`f_i` is the fraction of actively replicating forks in
nucleus *i* and :math:`R_0` is the available (dimensionless) resource.

Two propagators are provided:

* :func:`get_activated_fraction_euler` — adaptive forward-Euler with
  mass-conservation and steady-state clamping safeguards.
* :func:`get_activated_fraction_nlsoda` — high-order LSODA via
  *numbalsoda*, with automatic Euler fallback.
"""

# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numba as nb
import numpy as np
from numbalsoda import lsoda, lsoda_sig


@nb.njit(error_model="numpy")
def model(k_u, k_b, r_0, f_0):
    r"""Return df/dt for the activated-fraction ODE.

    .. math::
        \frac{df_i}{dt} = k_b (r_0 - \sum_j f_j)(1 - f_i) - k_u f_i

    Parameters
    ----------
    k_u : float
        Unbinding rate [/cycle].
    k_b : float
        Binding rate [/cycle].
    r_0 : float
        Available (dimensionless) resource.
    f_0 : ndarray
        Fraction of actively replicating complexes per nucleus.

    Returns
    -------
    dfdt : ndarray
        Time derivative of each activated fraction.
    """
    return k_b * (r_0 - f_0.sum()) * (1.0 - f_0) - k_u * f_0


@nb.njit(error_model="numpy")
def get_activated_fraction_euler(k_u, k_b, r_0, f_0, delta_t0):
    """Propagate *f_0* by one adaptive-Euler step.

    The step size is chosen so that the largest component changes by at
    most 0.01; it is then clipped to ``[5e-8, delta_t0]``.

    Parameters
    ----------
    k_u : float
        Unbinding rate [/cycle].
    k_b : float
        Binding rate [/cycle].
    r_0 : float
        Available (dimensionless) resource.
    f_0 : ndarray
        Fraction of actively replicating complexes per nucleus.
    delta_t0 : float
        Maximum allowed step size.

    Returns
    -------
    delta_t : float
        Actual step size used.
    f_1 : ndarray
        Updated activated fractions.
    """
    # Evaluate RHS once and reuse for step-size selection and Euler step
    dfdt = model(k_u, k_b, r_0, f_0)
    delta_t = 0.01 / np.abs(dfdt).max()
    # clip delta_t to ensure a minimal stepsize
    delta_t = np.clip(np.array([delta_t]), 5e-8, delta_t0)[0]

    delta_f = delta_t * dfdt
    f_1 = f_0 + delta_f

    # Clamp to steady state to prevent overshoot
    f_eq = model_steady_state(r_0, k_u / k_b, len(f_0))
    exceed_eq = ((f_0 <= f_eq) & (f_1 > f_eq)) | ((f_0 >= f_eq) & (f_1 < f_eq))
    f_1[exceed_eq] = f_eq

    # Enforce mass conservation: sum(f) <= r_0
    if f_1.sum() > r_0:
        delta_f = f_1 - f_0
        total_growth = delta_f[delta_f > 0].sum()
        if total_growth > 0:
            total_shrinking = delta_f[delta_f < 0].sum()
            scaling_factor = (
                0.9999 * (r_0 - f_0.sum() + total_shrinking) / total_growth
            )
            delta_f[delta_f > 0] *= scaling_factor
            f_1 = f_0 + delta_f
    # Ensure f_i in [0, 1]
    f_1 = np.clip(f_1, 0.0, 1.0)
    return delta_t, f_1


@nb.njit(error_model="numpy")
def model_steady_state(r_0, k, n_nuclei):
    """Return the symmetric steady-state activated fraction.

    Solves ``df/dt = 0`` assuming all nuclei share the same fraction,
    yielding a quadratic whose smaller root is the physical solution.

    Parameters
    ----------
    r_0 : float
        Available (dimensionless) resource.
    k : float
        Dimensionless dissociation constant (``k_u / k_b``).
    n_nuclei : int
        Number of in-S nuclei.

    Returns
    -------
    f_eq : float
        Steady-state activated fraction per nucleus.
    """
    a = (r_0 + k) / n_nuclei + 1
    f_eq = (a - np.sqrt(a**2 - 4 * r_0 / n_nuclei)) / 2
    return f_eq


@nb.cfunc(lsoda_sig)
def model_nlsoda(t, f_0, df, args):
    """LSODA C-callback for the activated-fraction ODE.

    Parameters (via ``args`` array)
    -------------------------------
    args[0] : k_u — unbinding rate [/cycle].
    args[1] : k_b — binding rate [/cycle].
    args[2] : r_0 — available (dimensionless) resource.
    args[3] : n   — number of in-S nuclei (cast to int).
    """
    args_ = nb.carray(args, (4,))
    k_u, k_b, r_0, n = args_
    f_0_ = nb.carray(f_0, (np.int32(n),))

    f_sum = f_0_.sum()
    for i in range(np.int32(n)):
        df[i] = k_b * (r_0 - f_sum) * (1 - f_0_[i]) - k_u * f_0_[i]


model_ptr = model_nlsoda.address


@nb.njit(boundscheck=False, error_model="numpy")
def get_activated_fraction_nlsoda(k_u, k_b, r_0, f_0, delta_t0):
    """Propagate *f_0* by one LSODA step (via numbalsoda).

    Parameters
    ----------
    k_u : float
        Unbinding rate [/cycle].
    k_b : float
        Binding rate [/cycle].
    r_0 : float
        Available (dimensionless) resource.
    f_0 : ndarray
        Fraction of actively replicating complexes per nucleus.
    delta_t0 : float
        Step size to propagate in time.

    Returns
    -------
    delta_t : float
        Step size used (always *delta_t0* for LSODA).
    f_1 : ndarray
        Updated activated fractions.
    """
    args = np.array([k_u, k_b, r_0, np.float64(len(f_0))])
    t_eval = np.array([0.0, delta_t0])
    f_1, success = lsoda(model_ptr, f_0, t_eval, args, mxstep=1e5)

    if not success:
        # Fall back to Euler if LSODA fails to converge
        return get_activated_fraction_euler(
            k_u=k_u, k_b=k_b, r_0=r_0, f_0=f_0, delta_t0=delta_t0
        )

    # Ensure f_i in [0, 1]
    f_1 = np.clip(f_1[-1], 0.0, 1.0)
    return delta_t0, f_1
