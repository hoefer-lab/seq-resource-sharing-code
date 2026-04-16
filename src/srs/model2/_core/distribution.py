# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Percentile-based sampling of correlated S/D-phase durations.

Draws phase durations from the empirical distribution using conditional
multivariate normals, preserving the kinship correlation structure
estimated in :mod:`lib.correlation_matrix`.
"""

# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
from scipy import stats

from .multivariate_normal import MultivariateNormal


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_single_time(datas):
    """Draw one random duration from each dataset."""
    rng = np.random.default_rng()
    durations = [rng.choice(data) for data in datas]
    return np.rint(durations).astype(int)


def sd_to_p(datas, sds):
    """Return for a given s the corresponding p value."""
    ps = np.array(
        [
            stats.percentileofscore(data, sd, "mean") / 100
            for data, sd in zip(datas, sds)
        ]
    )
    # # if percentileofscore is 1 set it to a slightly lower value as otherwise
    # # it results in x=inf
    # if np.isclose(ps, 1).any() or np.isclose(ps, 0).any():
    #     n_max = np.max([len(data) for data in datas])
    #     ps[np.isclose(ps, 1)] = 1 - 1 / (2 * n_max)
    #     ps[np.isclose(ps, 0)] = 1 / (2 * n_max)
    #     # ps[ps == 1] = np.array([1 - .1 / len(data) for data in datas[ps == 1]])

    return ps


def p_to_sd(datas, ps):
    """
    Computes the percentile-based values for multiple datasets using the specified probabilities.
    This function takes a collection of data arrays and a corresponding collection of probability values,
    where each probability is interpreted as a fraction (between 0 and 1) specifying the desired percentile.
    For each dataset in `datas` and its corresponding probability in `ps`, the function computes the percentile
    value at 100 * probability using NumPy's 'closest_observation' method. The result is a list of these calculated
    percentile values.

    Parameters:
    -----------
        datas (Iterable[array-like]): A collection of datasets, where each element is an array-like object
            containing numerical values.
        ps (Iterable[float]): A collection of probability values (each between 0 and 1) corresponding to the
            datasets in `datas`. Each probability indicates the percentile to compute for the respective dataset.

    Returns:
    --------
        List[float]: A list of percentile values computed for each dataset based on the supplied probabilities.

    Example:
        >>> import numpy as np
        >>> data1 = np.arange(100)
        >>> data2 = np.linspace(0, 1, 50)
        >>> result = p_to_sd([data1, data2], [0.5, 0.25])
        >>> # result[0] is the 50th percentile of data1, and result[1] is the 25th percentile of data2.
    """

    sd = [
        np.percentile(data, 100 * p, method="closest_observation")
        for data, p in zip(datas, ps)
    ]
    return sd


def get_s_and_d(
    data, corr, s_pre=None, d_pre=None, is_first=False, is_second=False
):
    """Return replication duration r and division+gap time d.

    For s_pre == None, the ancestor with d = 0 is modelled and for
    d_pre == None, the first prolonged d2-phase with s2.

    Note: Here in our model corresponds d to G2MG1 and s to the S-phase from
    the well-known cell cycle.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe used to estimate d and s
    s_pre : float
        s-phase of parent
    d_pre : float
        d-phase of parent

    Returns
    -------
    d : int
        D-phase duration
    s : int
        S-phase duration

    """
    # For s_pre is None: model initial s phase
    if is_first and d_pre is None:
        current_data = {
            "s": data["s"],
            "d": None,
            "sx": data["sx"],
            "dx": data["dx"],
        }
        corr = corr[1:, 1:]
    elif is_second:
        current_data = {
            "s": data["sx"],
            "d": data["dx"],
            "sx": data["sx"],
            "dx": data["dxx"],
        }
    else:
        current_data = {
            "s": data["sx"],
            "d": data["dxx"],
            "sx": data["sx"],
            "dx": data["dxx"],
        }
    s0, s1, d0, d1 = draw_s_d(
        data=current_data, corr=corr, s_pre=s_pre, d_pre=d_pre
    )

    # sort d phase such that '0' always starts earlier than its
    # sibling '1'
    if d1 < d0:
        d1, d0 = d0, d1
        s1, s0 = s0, s1

    return {
        "s": s_pre,
        "d": d_pre,
        "s0": s0,
        "d0": d0,
        "s1": s1,
        "d1": d1,
    }


def draw_s_d(data, corr, s_pre, d_pre):
    """Draw S- and D-phase durations for two daughters via conditional normal."""
    if d_pre is None:
        p_pre = sd_to_p(datas=[data["s"]], sds=[s_pre])
        # corr = corr[1:, 1:]
        k = 1
    else:
        p_pre = sd_to_p(datas=[data["d"], data["s"]], sds=[d_pre, s_pre])
        k = 2

    # Estimate x for mean=0 and sigma=1
    xs_pre = stats.norm().ppf(q=p_pre)

    # estimate correlation
    multin = MultivariateNormal(np.zeros(corr.shape[0]), corr)
    # create beta
    multin.partition(k=k)
    mean, cov = multin.cond_dist(1, xs_pre)
    # draw new xs
    xs = np.random.default_rng().multivariate_normal(mean, cov)
    # transform p_s back to s
    ps = stats.norm().cdf(xs)

    sd = p_to_sd(datas=[data["sx"], data["sx"], data["dx"], data["dx"]], ps=ps)

    return np.rint(sd).astype(int)
