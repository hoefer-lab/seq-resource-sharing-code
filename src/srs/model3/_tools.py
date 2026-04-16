# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Small numerical helper functions.

Includes binary-name conversion, sliding-mean computation, signal
filtering, and time-series utilities used throughout the model-3
package.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def get_mean_std_dict(x):
    """Return dict with mean and std of given array."""
    return {"mean": x.mean(axis=0), "std": x.std(axis=0)}


def filtered_signal(y):
    """Apply a 4th-order low-pass Butterworth filter to *y*.

    Parameters
    ----------
    y : ndarray
        1-D signal to filter.

    Returns
    -------
    y_filtered : ndarray
        Filtered signal (same length as *y*).
    """
    sos = butter(4, 0.125, output="sos")
    y_filtered = sosfiltfilt(sos, y)
    return y_filtered


def get_sliding_mean(x, y, x_max=None):
    """Compute a sliding-bin average of *y* over *x*.

    The *x* range ``[0, x_max]`` is divided into ~50×x_max bins and
    the mean of *y* is computed in each bin.

    Parameters
    ----------
    x, y : ndarray
        Data coordinates (both 1-D, same length).
    x_max : float, optional
        Right edge of the averaging window (defaults to ``x.max()``).

    Returns
    -------
    x_space : ndarray
        Bin centres.
    y_mean : ndarray
        Mean of *y* in each bin.
    """
    if not x_max:
        x_max = x.max()

    x_space, delta_x = np.linspace(
        0, x_max, num=(x_max * 50).astype(np.int32), retstep=True
    )
    y_mean = np.zeros_like(x_space)

    for i, x1 in enumerate(x_space):
        y_mean[i] = np.mean(y[(x < x1 + delta_x / 2) & (x >= x1 - delta_x / 2)])
    return x_space, y_mean


def get_t_max(results):
    """Return the minimum final time across all realisations.

    Useful for truncating time axes so that all traces contribute to
    every time point.

    Parameters
    ----------
    results : list of list of dict
        Nested simulation results (outer: sharing modes; inner:
        realisations).

    Returns
    -------
    t_max : float
    """
    t_max = np.inf
    for res in results:
        for rs in res:
            t_max = np.min([t_max, np.max(rs["time"])])
    return t_max
