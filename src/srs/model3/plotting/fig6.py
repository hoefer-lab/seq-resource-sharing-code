# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Paper Figure 6 — simulated vs. experimental replication timing.

This module produces the main data-comparison figure (Fig. 6) and several
supplementary panels that overlay simulated branching-process output on
experimental replication-timing data (Airyscan and Klaus datasets).

The main entry point is :func:`plot_fig6`.
The ABC posterior corner plot is provided by :func:`plot_fig_inference_si`.
"""

import matplotlib.pyplot as plt
import numpy as np
import prettypyplot as pplt

from srs.lib.data_loader import (
    convert_to_df,
    get_data,
)
from srs.lib.enums import DataSet, DropData, ResourceType
from srs.lib.plotting import (
    compute_bulk_statistics,
    plot_duration_comparison,
    plot_kinship_correlation,
    plot_phase_distribution,
    plot_replicating_nuclei,
    plot_resource_by_delay,
    plot_resource_fitting,
    plot_seq_vs_par_comparison,
    plot_sphase_vs_delay,
    plot_sphase_vs_delay_fast_slow,
)
from srs.lib.style import FIGORDNER, set_size, use_pplt_style

from .._convert import convert_rs
from .._core.wrapper import simulate_time_wrapper


def get_sim_data_model3(
    std,
    k_b,
    resource_type,
    zeta_low,
    zeta_high,
    rho,
    f_threshold,
    n=30000,
    k_u=1e-2,
    n_stop=24,
    acceleration=1,
    scaling=1,
):
    """Run branching-process simulations and return summary DataFrames.

    Draws *n* scarcity values from ``Uniform(zeta_low, zeta_high)`` and
    runs one realisation per value.  Results are converted to a phase
    dictionary, a tidy DataFrame, and bulk summary statistics.

    Parameters
    ----------
    std : float
        D-phase standard deviation.
    k_b : float
        Binding rate.
    resource_type : ResourceType
        Resource-growth model.
    zeta_low, zeta_high : float
        Bounds of the uniform scarcity distribution.
    rho : float
        Inverse S-phase duration.
    f_threshold : float
        Activated-fraction threshold for the s*-phase.
    n : int, default=30000
        Number of realisations.
    k_u : float, default=1e-2
        Unbinding rate.
    n_stop : int, default=24
        Number of nuclei per realisation.
    acceleration : float, default=1
        Time-acceleration factor.
    scaling : float, default=1
        Time-scaling factor.

    Returns
    -------
    data : dict
        ``{'sd': DataFrame, 'sc': dict, 'bulk': dict}``.
    """
    zetas = np.random.default_rng().uniform(zeta_low, zeta_high, n)
    res = simulate_time_wrapper(
        n_sample=n,
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
        f"RBC-{i + 1}": convert_rs(rs, f_threshold=f_threshold)
        for i, rs in enumerate(res)
    }
    df = convert_to_df(res)

    data = {
        "sd": df,
        "sc": res,
        "bulk": compute_bulk_statistics(res, model="model3"),
    }

    return data


def plot_fig6(
    params,
    f_threshold,
    n=500,
    drop_data=DropData.SUBTREE,
):
    """Assemble and save paper Figure 6 and supplementary panels.

    Compares simulated branching-process output (using the MAP parameter
    estimates in *params*) against experimental Klaus and Airyscan data.
    Produces four PDF figures:

    * ``fig6_n_{n}.pdf`` — main Figure 6.
    * ``figs7_n_{n}.pdf`` — Airyscan SI figure.
    * ``figs4_n_{n}.pdf`` — fitting diagnostics.

    Parameters
    ----------
    params : dict
        MAP parameter estimates (keys: ``'ku'``, ``'kb'``, ``'std'``,
        ``'rho'``, ``'scaling'``, ``'zeta_low'``, ``'zeta_high'``).
    f_threshold : float
        Activated-fraction threshold for the s*-phase.
    n : int, default=500
        Number of simulated realisations.
    drop_data : DropData, default=DropData.SUBTREE
        Which sub-trees to drop from experimental data.
    """
    use_pplt_style()
    acceleration_airy = np.mean([1.2181, 1.1303, 1.3235, 1.3425])

    def get_sim_data(params, parallel=False, airy=False):
        if parallel:
            ku = params["ku"] * (1e2 / params["ku"])
            kb = params["kb"] * (1e2 / params["ku"])
        else:
            ku = params["ku"]
            kb = params["kb"]

        if airy:
            acceleration = acceleration_airy
            n_stop = 16
            ku = ku / acceleration
            kb = kb / acceleration
            std = params["std"] / acceleration

        else:
            acceleration = 1
            n_stop = 24
            ku = ku
            kb = kb
            std = params["std"]

        return get_sim_data_model3(
            n=n,
            std=std,
            k_b=kb,
            zeta_low=params["zeta_low"],
            zeta_high=params["zeta_high"],
            rho=params["rho"],
            k_u=ku,
            scaling=params["scaling"] * acceleration,
            resource_type=ResourceType.PCNA1,
            f_threshold=f_threshold,
            n_stop=n_stop,
        )

    def get_masks_slow_fast(df_sim, df_exp, scaling=1):
        df_x_exp = df_exp.s
        x_mother_exp = df_x_exp.dropna().to_numpy()
        x_thresh_exp = np.percentile(
            x_mother_exp, q=50, method="closest_observation"
        )
        q_exp = (x_mother_exp <= x_thresh_exp).sum() / len(x_mother_exp) * 100
        df_x_sim = df_sim.s * scaling
        x_mother_sim = df_x_sim.dropna().to_numpy()
        # x_mother = (x_mother / 5).round() * 5.0  # ensure binning as in exp data
        x_thresh_sim = np.percentile(
            x_mother_sim,
            q=50,  # q_exp,
            method="closest_observation",
        )

        print(
            f"  Fast/slow split — "
            f"exp threshold S = {x_thresh_exp:.1f} min  "
            f"({q_exp:.0f}% fast)"
        )
        mask_sim = {
            "fast": df_x_sim <= x_thresh_sim,
            "slow": df_x_sim > x_thresh_sim,
        }
        mask_exp = {
            "fast": df_x_exp <= x_thresh_exp,
            "slow": df_x_exp > x_thresh_exp,
        }
        return mask_sim, mask_exp

    data_exp = get_data(dataset=DataSet.KLAUS, drop_data=drop_data)
    data_exp_airy = get_data(dataset=DataSet.AIRY, drop_data=drop_data)

    data_sim = get_sim_data(params)
    data_sim_parallel = get_sim_data(params, parallel=True)

    mask_sim, mask_exp = get_masks_slow_fast(
        df_sim=data_sim["sd"],
        scaling=params["scaling"],
        df_exp=data_exp["sd"],
    )

    data_sim_airy = get_sim_data(params, airy=True)
    # --------------------------- Main Fig 6 -----------------------------------

    # --------------------------- Main Fig 6 -----------------------------------
    fig6, grid6 = plt.subplots(
        figsize=np.array(set_size("nature_wide", subplot=[2, 3])),
        nrows=2,
        ncols=3,
        gridspec_kw={"height_ratios": (1.3, 1.0)},
        constrained_layout=True,
    )

    bins, delta_bins = plot_sphase_vs_delay(
        df_exp=data_exp["sd"],
        dataset_exp=DataSet.KLAUS,
        df_sim=data_sim["sd"],
        dataset_sim=DataSet.RESMODEL,
        ax=grid6[0, 0],
        gen="2nd",
        model="model3",
        scaling=params["scaling"],
    )

    counts_slow, counts_fast = plot_sphase_vs_delay_fast_slow(
        df=data_exp["sd"],
        dataset=DataSet.KLAUS,
        ax=grid6[0, 2],
        scaling=params["scaling"],
        bins=bins,
        delta_bins=delta_bins,
        mask_fast=mask_exp["fast"],
        mask_slow=mask_exp["slow"],
        cax_ylim=[0, 24],
        exp_data=True,
    )

    plot_sphase_vs_delay_fast_slow(
        df=data_sim["sd"],
        dataset=DataSet.RESMODEL,
        ax=grid6[0, 1],
        scaling=params["scaling"],
        bins=bins,
        delta_bins=delta_bins,
        mask_fast=mask_sim["fast"],
        mask_slow=mask_sim["slow"],
        cax_ylim=[0, 24],
        exp_data=False,
        counts_fast=counts_fast,
        counts_slow=counts_slow,
    )

    plot_resource_by_delay(
        df_exp=data_exp["sd"],
        data_sim=data_sim,
        ax1=grid6[1, 0],
        scaling=params["scaling"],
    )

    plot_replicating_nuclei(
        dataset_exp=DataSet.KLAUS,
        data_exp=data_exp,
        dataset_sim=DataSet.RESMODEL,
        data_sim=data_sim,
        ax=grid6[1, 1],
        model="model3",
        scaling=params["scaling"],
    )

    plot_seq_vs_par_comparison(
        dataset_exp=DataSet.KLAUS,
        data_exp=data_exp,
        data_sim_seq=data_sim,
        data_sim_par=data_sim_parallel,
        ax=grid6[1, 2],
        scaling=params["scaling"],
    )
    fig6.align_xlabels()
    fig6.align_ylabels()

    fig6.get_layout_engine().set(
        w_pad=6 / 72, h_pad=6 / 72, hspace=0.2, wspace=0.15
    )
    fig6.savefig(fname=FIGORDNER + f"fig6_n_{n}.pdf")

    # ------------------------------ Fig S7 ------------------------------------
    figs7, grids7 = plt.subplot_mosaic(
        [["a", "b", "c"], ["a", "d", "."]],
        figsize=np.array(set_size("nature_wide", subplot=[2, 3])),
        gridspec_kw={
            "wspace": 0,
            "height_ratios": (1.5, 1.0),
            "hspace": 0.1,
        },
        constrained_layout=True,
    )

    plot_sphase_vs_delay(
        df_exp=data_exp_airy["sd"],
        dataset_exp=DataSet.AIRY,
        df_sim=data_sim_airy["sd"],
        dataset_sim=DataSet.RESMODEL,
        ax=grids7["b"],
        gen="2nd",
        model="model3",
        scaling=params["scaling"] * acceleration_airy,
    )

    plot_replicating_nuclei(
        dataset_exp=DataSet.AIRY,
        data_exp=data_exp_airy,
        dataset_sim=DataSet.RESMODEL,
        data_sim=data_sim_airy,
        ax=grids7["d"],
        model="model3",
        scaling=params["scaling"] * acceleration_airy,
    )

    plot_kinship_correlation(
        data_exp_airy["sd"], data_sim_airy["sd"], ax=grids7["c"]
    )

    plot_duration_comparison(ax=grids7["a"])

    figs7.get_layout_engine().set(
        w_pad=6 / 72, h_pad=6 / 72, hspace=0.2, wspace=0.15
    )
    figs7.savefig(fname=FIGORDNER + f"figs7_n_{n}.pdf")

    # --------------------------- Fig S4 Fitting -------------------------------
    figs4, grids4 = plt.subplots(
        figsize=np.array(set_size("nature_wide", subplot=[2, 3])),
        nrows=2,
        ncols=3,
        constrained_layout=True,
    )

    for ax, phase in zip(grids4[1, 0:2], ["s_phase", "d_phase"]):
        plot_phase_distribution(
            df_exp=data_exp["sd"],
            dataset_exp=DataSet.KLAUS,
            df_sim=data_sim["sd"],
            dataset_sim=DataSet.RESMODEL,
            ax=ax,
            phase=phase,
            scaling=params["scaling"],
        )

    plot_resource_fitting(
        data_exp=data_exp,
        axs=grids4[0, :],
        zeta_low=params["zeta_low"],
        zeta_high=params["zeta_high"],
    )
    plot_kinship_correlation(data_exp["sd"], data_sim["sd"], ax=grids4[1, 2])

    figs4.get_layout_engine().set(
        w_pad=6 / 72, h_pad=6 / 72, hspace=0.2, wspace=0.15
    )
    figs4.savefig(fname=FIGORDNER + f"figs4_n_{n}.pdf")


def plot_fig_inference_si(df, w, limits=None, mle=None, savename_prefix=""):
    """Plot an ABC posterior corner plot (1-D marginals + 2-D histograms).

    Each diagonal panel shows a weighted 1-D histogram; off-diagonal panels
    show weighted 2-D histograms with colour-coded density.  The MAP
    estimate (if given) is marked with a red line / cross.

    Parameters
    ----------
    df : DataFrame
        Posterior samples with one column per parameter.
    w : ndarray
        Sample weights (normalised internally by ``np.histogram``).
    limits : dict of {str: [lo, hi]}, optional
        Axis limits per parameter.
    mle : dict of {str: float}, optional
        MAP estimates to overlay.
    savename_prefix : str, default=""
        Prefix prepended to the output filename.
    """
    use_pplt_style()

    n_par = df.shape[1]
    par_ids = list(df.columns.values)

    def hist_plot(x, w, ax):
        n_bins = 30
        if limits is not None:
            bins = np.linspace(
                limits[x][0],
                limits[x][1] + (np.diff(limits[x])[0] / (n_bins - 1)),
                n_bins + 1,
            )
        else:
            bins = n_bins

        ax.hist(
            df[x],
            weights=w,
            bins=bins,
            density=True,
            facecolor="pplt:gray",
            alpha=0.25,
        )

        # Add step plot to draw the outer shape
        counts, bin_edges = np.histogram(
            df[x], bins=bins, weights=w, density=True
        )
        ax.step(
            bin_edges[:-1], counts, where="post", color="pplt:gray", linewidth=1
        )

        if limits is not None:
            ax.set_xlim(limits[x])
        if mle is not None:
            ax.axvline(mle[x], color="pplt:red", lw=1.5)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ylim = [ylim[0], ylim[1] * 1.1]
        ax.set_ylim(ylim[0], ylim[1])
        dx_dy = np.diff(xlim)[0] / np.diff(ylim)[0]
        ax.set_aspect(dx_dy)

    def hist2d_pot(x, y, ax, w):
        n_bins = 20
        if limits is not None:
            bins_x = np.linspace(
                limits[x][0],
                limits[x][1],
                n_bins,
            )
            bins_y = np.linspace(
                limits[y][0],
                limits[y][1],
                n_bins,
            )
            bins = (bins_x, bins_y)
        else:
            bins = n_bins

        hist, xedges, yedges = np.histogram2d(
            df[x], df[y], bins=bins, density=True, weights=w
        )
        dx_dy = np.diff(xedges)[0] / np.diff(yedges)[0]
        hist[hist == 0] = None
        im = pplt.imshow(
            hist.T,
            extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()],
            ax=ax,
            aspect=dx_dy,
            vmax=np.nanmax(hist),
            vmin=0,
        )
        pplt.colorbar(im, ax=ax, label="relative count")
        if mle is not None:
            ax.scatter(mle[x], mle[y], color="pplt:red", marker="x", lw=2)

    fig_si, grid_si = plt.subplots(
        figsize=2 * [0.95 * np.array(set_size("nature_wide"))[0]],
        nrows=n_par,
        ncols=n_par,
        sharex=False,
        sharey=False,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
        constrained_layout=True,
    )

    labels = {
        "ku": r"$\log_{10}(k_u)$",
        "kb": r"$\log_{10}(k_b)$",
        "zeta_low": r"$\xi_{\text{low}}$",
        "zeta_high": r"$\xi_{\text{high}}$",
        "std": r"$\sigma_{\text{D}^\ast}$",
        "tau": r"$\tau_{\text{S}^\ast}$",
        "rho": r"$\rho$",
        "scaling": r"$\tau$",
        "f_threshold": r"$c_{i, \text{min}}$",
    }

    ticks_dict = {
        "ku": [-3, -1, 1],
        "kb": [0.7, 1.4, 2.1],
        "zeta_low": [0.75, 1.25, 1.75],
        "zeta_high": [1, 2.0, 3.0],
        "std": [0.03, 0.1, 0.18],
        "tau": [0.2, 0.25, 0.3],
        "rho": [3, 4, 5],
        "scaling": [70, 90, 110],
        "f_threshold": [0.06, 0.12, 0.18],
    }

    for i in range(n_par):
        for j in range(n_par):
            if i < j:
                grid_si[i, j].remove()
                continue
            elif i == j:
                hist_plot(
                    x=par_ids[i],
                    w=w,
                    ax=grid_si[i, j],
                )
            else:
                hist2d_pot(
                    y=par_ids[i],
                    x=par_ids[j],
                    ax=grid_si[i, j],
                    w=w,
                )
                grid_si[i, j].set_yticks(ticks_dict[par_ids[i]])
            grid_si[i, j].set_xticks(ticks_dict[par_ids[j]])

            if j == 0:
                grid_si[i, j].set_ylabel(labels[par_ids[i]])
            elif i == j:
                grid_si[i, j].yaxis.tick_right()
                grid_si[i, j].tick_params(axis="y", direction="in", pad=-22)
            else:
                grid_si[i, j].tick_params(
                    axis="y", direction="in", labelleft=False
                )

            if i == n_par - 1:
                grid_si[i, j].set_xlabel(labels[par_ids[j]])
            else:
                grid_si[i, j].tick_params(
                    axis="x", direction="in", labelbottom=False
                )

    fig_si.savefig(fname=FIGORDNER + savename_prefix + "fig6_v2_si.pdf")
