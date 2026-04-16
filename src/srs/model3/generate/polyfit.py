# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Weighted polynomial fitting with bump-function kernels.

Provides localised polynomial regression using a smooth bump-function
weight that is box-like with exponential tails.  Used to extract
instantaneous growth rates and utilisation values from noisy simulation
time series.
"""

import numpy as np


def get_polyfits(deg, y_labels, res, x0, x_max, sigma):
    """Compute weighted polynomial coefficients for multiple observables.

    Parameters
    ----------
    deg : int
        Polynomial degree.
    y_labels : sequence of str
        Observable names (keys into *res*).
    res : dict
        Simulation result with ``'time'`` and observable arrays.
    x0 : float
        Centre of the weight kernel.
    x_max : float
        Right edge beyond which the weight is zero.
    sigma : float
        Width of the bump-function kernel.

    Returns
    -------
    params : ndarray, shape (len(y_labels),)
        Leading coefficient for each observable.
    """

    params = np.zeros(len(y_labels))
    for i, y_label in enumerate(y_labels):
        params[i] = get_polyfit(
            deg=deg, y_label=y_label, res=res, x0=x0, x_max=x_max, sigma=sigma
        )
    return params


def get_polyfit(deg, y_label, res, x0, x_max, sigma):
    """Compute the leading weighted polynomial coefficient for one observable.

    For ``'n'`` and ``'g'`` labels the data is log-transformed before
    fitting, so the returned coefficient approximates the instantaneous
    exponential growth rate.

    Parameters
    ----------
    deg : int
        Polynomial degree.
    y_label : str
        Observable name (key into *res*).
    res : dict
        Simulation result with ``'time'`` and observable arrays.
    x0 : float
        Centre of the weight kernel.
    x_max : float
        Right edge beyond which the weight is zero.
    sigma : float
        Width of the bump-function kernel.

    Returns
    -------
    coeff : float
        Leading polynomial coefficient.
    """
    if y_label == "n" or y_label == "g":
        y = np.log(res[y_label])
    else:
        y = res[y_label]
    weights = get_weights(x=res["time"], x0=x0, x_max=x_max, width=sigma)
    return np.polyfit(x=res["time"], y=y, deg=deg, w=weights)[0]


def get_weights(x, x0, width, x_min=0, x_max=np.inf):
    """Compute bump-function weights centred at *x0* with smooth exponential tails.

    Parameters
    ----------
    x : np.ndarray
        Sample points.
    x0 : float
        Centre of the kernel.
    width : float
        Half-width of the support region.
    x_min, x_max : float
        Hard lower/upper bounds outside which the weight is zero.

    Returns
    -------
    np.ndarray
        Normalised weight vector (sums to 1).
    """
    # Centered variable z
    z = x - x0
    z_max = x_max - x0
    z_min = x_min - x0

    with np.errstate(divide="ignore", over="ignore"):
        weights = np.where(
            (np.abs(z) >= width) | (z > z_max) | (z < z_min),
            np.zeros_like(x),
            np.exp(width**2 / (z**2 - width**2)),
        )
    return weights / weights.sum()
