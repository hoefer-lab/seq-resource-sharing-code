# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Gamma-distribution sampling for D-phase durations.

D-phase lengths are drawn from a Gamma distribution whose mean and
variance are set by the biological parameters.  Three convenience
functions are provided:

* :func:`get_gamma` — sample from Gamma(mean, std).
* :func:`get_d_cv` — sample from Gamma(mean, cv).
* :func:`get_scale_and_shape` — convert (mean, cv) to Gamma (scale, shape)
  parameters.
"""

# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
from numpy.random import gamma


def get_gamma(std, mu, size=1):
    """Return d-phase duration.

    Parameters
    ----------
    std : float
        Standard deviation
    mu : float
        Mean value
    size : int, default=1
        Length of returned ndarray

    Returns
    -------
    sample : float
        Gamma-distributed samples

    """
    # For cv == 0 sample duration is deterministic and given by the mean
    if std == 0:
        return mu * np.ones(size)
    else:
        shape = (mu / std) ** 2
        scale = std**2 / mu
        samples = gamma(scale=scale, shape=shape, size=size)
        return samples


def get_d_cv(cv, tau_d, size=1):
    """Return d-phase duration.

    Parameters
    ----------
    cv : float
        Coefficient of variation of d
    tau_d : float
        Mean of D-phase [/cycle]
    size : int, default=1
        Length of returned ndarray

    Returns
    -------
    d : float
        Durations of D-phases

    """
    # For cv == 0 d-phase duration is deterministic and given by the mean
    if cv == 0:
        return tau_d * np.ones(size)
    else:
        theta, k = get_scale_and_shape(mean=tau_d, cv=cv)
        d_phases = gamma(scale=theta, shape=k, size=size)
        return d_phases


def get_scale_and_shape(mean, cv):
    """Return scale and shape for Gamma distr. according to given mean and cv.

    This function return the shape and scale parameter of a Gamma distribution

        p(x, k, theta) = x^(k-1) e^(-x / theta) / [Gamma(k) theta^k]

    Parameters
    ----------
    mean : float
        mean

    cv : float
        Coefficient of variation

    Returns
    -------
    k : float
        shape parameter of Gamma distribution
    theta : float
        scale parameter of Gamma distribution
    """
    # For a Gamma distribution the cv is given by 1 / \sqrt(k)
    k = cv ** (-2)
    # By using  mean = k * theta we get
    theta = mean / k
    return theta, k
