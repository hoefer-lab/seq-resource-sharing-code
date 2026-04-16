# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Parameter inference for model 3 via Approximate Bayesian Computation.

This module implements an ABC-SMC workflow (using *pyabc*) that fits the
branching-process model to experimental replication-timing data.  The
Wasserstein distance between normalised summary statistics (synchrony,
S-phase duration, D-phase duration) serves as the discrepancy measure.

The main entry point is :func:`model3_infer_params`.  Helper classes and
functions include :class:`ConstrainedPrior`, :func:`get_synthetic_data`,
:func:`find_map_estimate`, and pre-computed MAP parameter dictionaries at
module level.
"""

# ~~~ IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import tempfile
import warnings

import numpy as np
import pyabc
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, wasserstein_distance_nd

from srs.lib.data_loader import convert_to_df, get_data
from srs.lib.enums import DataSet, DropData, ResourceType

from ._convert import convert_rs, get_sync_s_d
from ._core.wrapper import simulate_time_wrapper
from .default_params import F_THRESHOLD
from .plotting.fig6 import plot_fig_inference_si

# Suppress RuntimeWarnings from scipy's covariance module that fire when
# pyABC's MultivariateNormalTransition encounters a near-singular covariance
# matrix.  pyABC handles this internally via regularisation, so the warnings
# are safe to silence.  A module-level filter is needed because
# warnings.catch_warnings() does not propagate to child processes or
# work reliably in IPython / notebook environments.
warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul.*",
    category=RuntimeWarning,
    module=r"scipy\.stats\._covariance",
)

# ~~~ CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MIN_VALID_SAMPLES = 10

LIMITS = {
    "ku": [-4, 2],
    "kb": [0.5, 2.5],
    "zeta_low": [0.5, 2.0],
    "zeta_high": [0.75, 3.5],
    "std": [0.0, 0.2],
    "rho": [1 / 0.4, 1 / 0.175],
    "scaling": [60, 120],
}


# ~~~ ABC COMPONENTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def distance(x, y):
    """Wasserstein distance between two sample dictionaries."""
    if x["samples"].size == 0 or y["samples"].size == 0:
        return np.inf
    return wasserstein_distance_nd(x["samples"], y["samples"])


def get_samples(sync, s_phase, d_phase, sync_std, s_phase_std, d_phase_std):
    """Normalise sync / S-phase / D-phase into a (N, 3) sample matrix.

    Each column is divided by its standard deviation so that all three
    observables contribute equally to the Wasserstein distance.

    Parameters
    ----------
    sync, s_phase, d_phase : ndarray
        Raw observable arrays.
    sync_std, s_phase_std, d_phase_std : float
        Standard deviations of the experimental reference for normalisation.

    Returns
    -------
    samples : ndarray, shape (N, 3)
    """
    samples = np.column_stack(
        [
            sync / sync_std,
            s_phase / s_phase_std,
            d_phase / d_phase_std,
        ]
    )
    if not np.all(np.isfinite(samples)):
        raise ValueError("Non-finite values in normalised samples")
    return samples


def get_synthetic_data(
    n=50,
    std=0.1,
    k_b=50,
    resource_type=ResourceType.PCNA1,
    zeta_low=1.2,
    zeta_high=2.2,
    rho=1 / 0.275,
    k_u=1e-2,
    acceleration=1,
    n_stop=24,
    scaling=1,
    f_threshold=F_THRESHOLD,
    parallel=False,
):
    """Run *n* branching-process simulations and return summary observables.

    Parameters
    ----------
    n : int
        Number of simulations.
    std : float
        D-phase standard deviation.
    k_b, k_u : float
        Binding / unbinding rates.
    resource_type : ResourceType
        Resource model.
    zeta_low, zeta_high : float
        Bounds of the uniform scarcity distribution.
    rho : float
        Inverse S-phase duration.
    acceleration, scaling : float
        Time-scaling factors for the resource model.
    n_stop : int
        Number of nuclei at which each simulation stops.
    f_threshold : float
        Threshold defining the effective S-phase.
    parallel : bool
        Whether to run simulations in parallel.

    Returns
    -------
    sync, s_phase, d_phase : ndarray
        Summary observables with invalid entries removed.
    """
    n_trees = n // 4 + 2
    zetas = np.random.default_rng().uniform(zeta_low, zeta_high, n_trees)

    res = simulate_time_wrapper(
        parallel=parallel,
        n_sample=n_trees,
        resource_type=resource_type,
        zetas=zetas,
        k_u=k_u,
        k_b=k_b,
        std=std,
        n_stop=n_stop,
        acceleration=acceleration,
        save_f=True,
        rho=rho,
        f_threshold=f_threshold,
        scaling=scaling,
    )

    res = {
        f"RBC-{i + 1}": convert_rs(rs, f_threshold) for i, rs in enumerate(res)
    }
    df = convert_to_df(res)
    df = df.copy() * scaling

    return get_sync_s_d(df, n=n)


class ConstrainedPrior(pyabc.DistributionBase):
    """Uniform prior with constraints ``ku < kb - min_diff`` and
    ``zeta_low < zeta_high``.

    Rejection sampling in :meth:`rvs` guarantees that every drawn sample
    satisfies both constraints; :meth:`pdf` returns zero for violating
    points.
    """

    def __init__(self):
        self.ku = pyabc.RV(
            "uniform", LIMITS["ku"][0], LIMITS["ku"][1] - LIMITS["ku"][0]
        )
        self.kb = pyabc.RV(
            "uniform", LIMITS["kb"][0], LIMITS["kb"][1] - LIMITS["kb"][0]
        )
        self.zeta_low = pyabc.RV(
            "uniform",
            LIMITS["zeta_low"][0],
            LIMITS["zeta_low"][1] - LIMITS["zeta_low"][0],
        )
        self.zeta_high = pyabc.RV(
            "uniform",
            LIMITS["zeta_high"][0],
            LIMITS["zeta_high"][1] - LIMITS["zeta_high"][0],
        )
        self.std = pyabc.RV(
            "uniform", LIMITS["std"][0], LIMITS["std"][1] - LIMITS["std"][0]
        )
        self.rho = pyabc.RV(
            "uniform", LIMITS["rho"][0], LIMITS["rho"][1] - LIMITS["rho"][0]
        )
        self.scaling = pyabc.RV(
            "uniform",
            LIMITS["scaling"][0],
            LIMITS["scaling"][1] - LIMITS["scaling"][0],
        )
        self.min_diff = 1.0

    def rvs(self, *args, **kwargs):
        """Draw a sample satisfying all constraints."""
        ku, kb = self.ku.rvs(), self.kb.rvs()
        while ku > kb - self.min_diff:
            ku, kb = self.ku.rvs(), self.kb.rvs()

        zeta_low, zeta_high = self.zeta_low.rvs(), self.zeta_high.rvs()
        while zeta_high < zeta_low:
            zeta_low, zeta_high = self.zeta_low.rvs(), self.zeta_high.rvs()

        return pyabc.Parameter(
            ku=ku,
            kb=kb,
            zeta_low=zeta_low,
            zeta_high=zeta_high,
            std=self.std.rvs(),
            rho=self.rho.rvs(),
            scaling=self.scaling.rvs(),
        )

    def pdf(self, x):
        """Return the prior density, zero for constraint-violating points."""
        if (x["ku"] > x["kb"] - self.min_diff) or (
            x["zeta_high"] < x["zeta_low"]
        ):
            return 0.0
        return (
            self.ku.pdf(x["ku"])
            * self.kb.pdf(x["kb"])
            * self.zeta_low.pdf(x["zeta_low"])
            * self.zeta_high.pdf(x["zeta_high"])
            * self.std.pdf(x["std"])
            * self.rho.pdf(x["rho"])
            * self.scaling.pdf(x["scaling"])
        )


# ~~~ MAP ESTIMATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def find_map_estimate(dframe, w):
    """Find the MAP estimate from weighted ABC posterior samples.

    Fits a Gaussian KDE to the normalised posterior and maximises it
    using :func:`scipy.optimize.minimize`.

    Parameters
    ----------
    dframe : DataFrame
        Posterior samples with one column per parameter.
    w : ndarray
        Sample weights.

    Returns
    -------
    parameters : dict
        MAP estimate keyed by parameter name.
    log_pdf : float
        Log-density at the MAP point.
    """
    par_ids = list(dframe.columns)
    data = dframe.to_numpy()
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    normalised = (data - means) / stds

    kde = gaussian_kde(normalised.T, weights=w)
    x0 = np.median(normalised, axis=0)
    result = minimize(lambda x: -kde.logpdf(x), x0)

    map_normalised = result.x
    map_original = map_normalised * stds + means
    log_pdf = kde.logpdf(map_normalised)

    parameters = {pid: map_original[i] for i, pid in enumerate(par_ids)}
    return parameters, log_pdf


# ~~~ PUBLIC API ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def model3_infer_params(
    drop_data=DropData.SUBTREE,
    max_nr_populations=15,
    population_size=10000,
    saveprefix="fig6_pi",
):
    """Run ABC-SMC inference for model 3 and visualise the posterior.

    Parameters
    ----------
    drop_data : DropData
        Which sub-trees to drop from the experimental data.
    max_nr_populations : int
        Maximum number of ABC-SMC generations.
    population_size : int
        Number of accepted particles per generation.
    saveprefix : str
        Prefix for saved figure filenames.

    Returns
    -------
    abc : pyabc.ABCSMC
        The fitted ABC-SMC object (with accessible history).
    """
    # --- Load experimental reference data ---
    data_exp = get_data(dataset=DataSet.KLAUS, drop_data=drop_data)
    sync_exp, s_phase_exp, d_phase_exp = get_sync_s_d(data_exp["sd"])

    sync_std = np.std(sync_exp)
    s_phase_std = np.std(s_phase_exp)
    d_phase_std = np.std(d_phase_exp)

    # --- Define ABC model (closure over experimental std's) ---
    def custom_model(parameters):
        sync, s_phase, d_phase = get_synthetic_data(
            n=len(sync_exp),
            k_b=10 ** parameters["kb"],
            k_u=10 ** parameters["ku"],
            zeta_low=parameters["zeta_low"],
            zeta_high=parameters["zeta_high"],
            std=parameters["std"],
            rho=parameters["rho"],
            scaling=parameters["scaling"],
            n_stop=12,
        )
        return {
            "samples": get_samples(
                sync, s_phase, d_phase, sync_std, s_phase_std, d_phase_std
            ),
        }

    data = {
        "samples": get_samples(
            sync_exp,
            s_phase_exp,
            d_phase_exp,
            sync_std,
            s_phase_std,
            d_phase_std,
        ),
    }

    # --- Run ABC-SMC ---
    abc = pyabc.ABCSMC(
        custom_model,
        ConstrainedPrior(),
        distance,
        population_size=population_size,
    )
    db_path = "sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db")
    abc.new(db_path, data)
    abc.run(
        minimum_epsilon=0.1,
        max_nr_populations=max_nr_populations,
        min_acceptance_rate=0.01,
    )

    # --- MAP estimate ---
    dframe, w = abc.history.get_distribution()
    parameters, log_pdf = find_map_estimate(dframe, w)

    plot_fig_inference_si(
        df=dframe,
        w=w,
        limits=LIMITS,
        mle=parameters,
        savename_prefix=saveprefix,
    )

    # --- Print summary ---
    print("\n  MAP parameter estimates")
    print("  " + "-" * 36)
    print(f"  {'k_b':>12s} = {10 ** parameters['kb']:.4f}")
    print(f"  {'k_u':>12s} = {10 ** parameters['ku']:.6f}")
    print(f"  {'scaling':>12s} = {parameters['scaling']:.2f}")
    print(f"  {'std':>12s} = {parameters['std']:.4f}")
    print(f"  {'rho':>12s} = {parameters['rho']:.4f}")
    print(f"  {'tau':>12s} = {1 / parameters['rho']:.4f}")
    print(f"  {'zeta_low':>12s} = {parameters['zeta_low']:.4f}")
    print(f"  {'zeta_high':>12s} = {parameters['zeta_high']:.4f}")
    print(f"  {'log-pdf':>12s} = {log_pdf}")

    # --- Confidence intervals (95%) ---
    print("\n  95% confidence intervals")
    print("  " + "-" * 36)
    param_names = ["kb", "ku", "scaling", "std", "rho", "zeta_low", "zeta_high"]
    for param in param_names:
        lower = dframe[param].quantile(0.025, interpolation="linear")
        upper = dframe[param].quantile(0.975, interpolation="linear")
        if param in ["kb", "ku"]:
            lower = 10**lower
            upper = 10**upper
        print(f"  {param:>12s}   ({lower:.6f}, {upper:.6f})")

    return abc
