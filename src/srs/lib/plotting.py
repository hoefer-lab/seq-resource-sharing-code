# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Plotting functions for all paper figures.

Includes bulk-statistics computation, kinship-correlation heatmaps,
violin / scatter comparisons, nuclei-count time series, and
resource-fitting diagnostics used by both model 2 and model 3.
"""

import itertools
import warnings

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prettypyplot as pplt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import MaxNLocator, MultipleLocator, ScalarFormatter
from mpl_toolkits import axes_grid1 as mpl_axes_grid1
from scipy.stats import pearsonr

from .correlation_matrix import get_corr_matrix_plot_pooled
from .data_loader import *
from .enums import DataSet, DropData
from .style import (
    COLOR_SEQ,
    COLOR_SEQ_SHADE,
    SCATTER_KWARGS,
    colors_fast_slow_dict,
    colors_violin_dict,
    colors_violin_shade_dict,
)


def plot_kinship_correlation(df_exp, df_sim, ax, show_cbar=True, two_exp=False):
    """Plot dual half-circle kinship correlation heatmap (data vs. simulation).

    Parameters
    ----------
    df_exp, df_sim : pd.DataFrame
        Phase-duration DataFrames for experimental and simulated data.
    ax : matplotlib.axes.Axes
        Target axes.
    show_cbar : bool
        Whether to show the colour bar.
    two_exp : bool
        If *True*, use the simulation p-values from ``df_sim``;
        otherwise use the experimental ones for both halves.
    """
    correlation_res_sim = get_corr_matrix_plot_pooled(df=df_sim)
    correlation_res_exp = get_corr_matrix_plot_pooled(df=df_exp)

    # print('simulation:')
    # print(correlation_res_sim['corr'])
    # print('\n\ndata:')
    # print(correlation_res_exp['corr'])
    if two_exp:
        pval_sim = correlation_res_sim["pval"]
    else:
        pval_sim = correlation_res_exp["pval"]

    correlationplot_dual(
        corr_sim=correlation_res_sim["corr"],
        corr_data=correlation_res_exp["corr"],
        pval_sim=pval_sim,
        pval_data=correlation_res_exp["pval"],
        x_columns=correlation_res_sim["columns"],
        y_columns=correlation_res_sim["rows"],
        cbarlabel=r"Gaussian rank correlation",  # r"Spearman's $\rho$",
        ax=ax,
        cmap=None,
        show_cbar=show_cbar,
        show_legend=False,
    )
    for label in ax.get_xticklabels():
        label.set_rotation(90)


def plot_phase_identity(dataset_exp, df_exp, axs):
    """Plot 2-D histograms of sister S-phase identity (s0 vs. s1).

    Parameters
    ----------
    dataset_exp : DataSet
        Experimental dataset enum.
    df_exp : pd.DataFrame
        Phase-duration DataFrame.
    axs : tuple[Axes, Axes]
        Pair of axes ``(ax_model, ax_data)``.
    """
    ax_model, ax_data = axs

    df_data = df_exp[["s0", "s1"]].dropna().copy()

    x = df_data["s0"].to_numpy()
    y = df_data["s1"].to_numpy()

    bins = np.arange(20, 90, 5)
    hist2d_data, xedges, yedges = np.histogram2d(x, y, bins=[bins, bins])
    hist2d_data[hist2d_data == 0] = (
        np.nan
    )  # set zero values to NaN for better visualization

    r, p = pearsonr(x, y)
    # add noise
    dx = 5
    x += np.random.uniform(-dx / 4, dx / 4, size=x.shape)
    y += np.random.uniform(-dx / 4, dx / 4, size=y.shape)

    # Model plot
    s_phase = df_exp[["s0", "s1"]].to_numpy().flatten()
    s_phase = s_phase[~np.isnan(s_phase)]
    n = 100000000
    s0, s1 = np.random.choice(s_phase, size=(2, n), replace=True)

    # Plot density on a 5min grid for the model
    hist2d, xedges, yedges = np.histogram2d(s0, s1, bins=[bins, bins])
    hist2d[hist2d == 0] = (
        np.nan
    )  # set zero values to NaN for better visualization
    hist2d *= len(s_phase) / n
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    # Determine the common vmin and vmax for both plots
    vmin = np.nanmin([np.nanmin(hist2d_data), np.nanmin(hist2d)])
    vmax = np.nanmax([np.nanmax(hist2d_data), np.nanmax(hist2d)])

    black_alpha_cmap = ListedColormap(
        [(0.2, 0.2, 0.2, 1 - alpha) for alpha in np.linspace(1, 0, 10)]
    )

    blue_alpha_cmap = ListedColormap(
        [
            (0.2, 0.3843137254901961, 0.6901960784313725, 1 - alpha)
            for alpha in np.linspace(1, 0, 10)
        ]
    )

    pcm = ax_data.pcolormesh(
        X,
        Y,
        np.transpose(hist2d_data),
        vmin=vmin,
        vmax=vmax,
        cmap=black_alpha_cmap,
    )
    pplt.colorbar(pcm, ax=ax_data, label="count", pad=0.05, position="right")

    pcm = ax_model.pcolormesh(
        X, Y, hist2d, vmin=vmin, vmax=vmax, cmap=blue_alpha_cmap
    )
    pplt.colorbar(pcm, ax=ax_model, label="", pad=0.05, position="right")

    # Data plot
    ax_data.scatter(x, y, color="black", **SCATTER_KWARGS)
    if p < 0.001:
        p = "$p$ < 0.001"
    else:
        p = f"$p$ = {p:.3f}"
    pplt.text(
        x=0.05,
        y=0.95,
        s=f"$r$ = {r:.2f}\n{p}",
        ha="left",
        va="top",
        ax=ax_data,
        fontsize=6,
        transform=ax_data.transAxes,
    )

    # pplt.colorbar(im=sc)
    for ax in axs:
        ax.plot(
            [x.min(), x.max()],
            [x.min(), x.max()],
            "--",
            zorder=0,
            color="pplt:gray",
        )
        ax.lines[0].set_linewidth(0.5 * ax.lines[0].get_linewidth())
        ax.set_xlabel(r"S$_1$-phase (min)")
        ax.set_aspect("equal", anchor="SW")
        ax.set_xticks([25, 50, 75])
        ax.set_yticks([25, 50, 75])
    ax_model.set_ylabel(r"S$_2$-phase (min)")

    xlim, ylim = ax_data.get_xlim(), ax_data.get_ylim()
    ax_data.set_xlim([xlim[0] - 0.05 * (xlim[1] - xlim[0]), None])
    ax_data.set_ylim([ylim[0] - 0.05 * (ylim[1] - ylim[0]), None])
    xlim, ylim = ax_data.get_xlim(), ax_data.get_ylim()
    ax_model.set_xlim(xlim)
    ax_model.set_ylim(ylim)


def get_delay_and_sphase(df, gen, mask=None):
    """Extract replication delay and S-phase duration arrays from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Phase-duration DataFrame with columns s0, s1, d0, d1, etc.
    gen : {'2nd', '3rd'}
        Nuclear generation to extract.
    mask : np.ndarray, optional
        Boolean mask applied to filter the rows.

    Returns
    -------
    sync : np.ndarray
        Replication delay normalised by mean S-phase.
    s_phase : np.ndarray
        Corresponding S-phase durations.
    """
    if gen == "2nd":
        s_phase = np.concatenate([df.s0, df.s1])
        sync = np.concatenate(2 * [(df.d1 - df.d0) / np.nanmean(df.s0)])

    elif gen == "3rd":
        s_phase = np.concatenate([df.s00, df.s01, df.s10, df.s11])
        s_mean = np.nanmean(np.concatenate([df.s00, df.s10]))
        sync = np.concatenate(
            [
                (df.d01 - df.d00) / s_mean,
                (df.d01 - df.d00) / s_mean,
                (df.d11 - df.d10) / s_mean,
                (df.d11 - df.d10) / s_mean,
            ]
        )
    if mask is not None:
        if gen == "2nd":
            mask = np.concatenate([mask, mask])
        elif gen == "3rd":
            mask = np.concatenate([mask, mask, mask, mask])
        s_phase, sync = s_phase[mask], sync[mask]

    remove_nan = ~(np.isnan(s_phase) | np.isnan(sync))
    s_phase, sync = s_phase[remove_nan], sync[remove_nan]

    return sync, s_phase


def plot_phase_distribution(
    df_exp, dataset_exp, df_sim, dataset_sim, ax, phase="s_phase", scaling=1
):
    """Plot overlaid histograms of S- or D-phase distributions (data vs. model).

    Parameters
    ----------
    df_exp, df_sim : pd.DataFrame
        Experimental / simulated phase-duration DataFrames.
    dataset_exp, dataset_sim : DataSet
        Dataset enums (used for colour lookup).
    ax : matplotlib.axes.Axes
        Target axes.
    phase : {'s_phase', 'd_phase'}
        Which phase to plot.
    scaling : float
        Time-scaling factor applied to simulated data.
    """

    def get_phases(df, phase):
        if phase == "s_phase":
            phases = np.concatenate([df.s0, df.s1])
        elif phase == "d_phase":
            phases = np.concatenate([df.d00, df.d01, df.d10, df.d11])
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return phases[~np.isnan(phases)]

    phase_exp = get_phases(df_exp, phase)
    phase_sim = get_phases(df_sim, phase) * scaling
    weights = [
        np.ones_like(phase_exp),
        len(phase_exp) / len(phase_sim) * np.ones_like(phase_sim),
    ]

    # phase_min = min(np.nanmin(phase_exp), np.nanmin(phase_sim))
    # phase_max = max(np.nanmax(phase_exp), np.nanmax(phase_sim))
    # print(phase_min, phase_max)
    # bins = np.arange(phase_min - 5, phase_max + 10, 5)
    bins = np.arange(25, 135, 5)
    ax.hist(
        [phase_exp, phase_sim],
        bins=bins,
        stacked=False,
        color=[
            colors_violin_shade_dict[dataset_exp.value],
            colors_violin_shade_dict[dataset_sim.value],
        ],
        linewidth=0.6,
        edgecolor="pplt:gray",
        # density=True,
        rwidth=0.6,
        weights=weights,
    )
    if phase == "s_phase":
        ax.set_xlabel(r"S$_i$-phase (min)")
    elif phase == "d_phase":
        ax.set_xlabel(r"D$_{ij}$-phase (min)")
    ax.set_ylabel("count")


def plot_sphase_vs_delay(
    df_exp,
    dataset_exp,
    df_sim,
    dataset_sim,
    ax,
    model="model2",
    gen="2nd",
    n_bootstrap=10000,
    scaling=1,
    delta_bins=None,
    bins=None,
    mask_exp=None,
    mask_sim=None,
    cax_ylim=None,
):
    """Plot S-phase duration vs. replication delay with binned bootstrap means.

    The upper panel shows mean S-phase per delay bin with bootstrap CIs;
    the lower panel shows the delay histogram.

    Parameters
    ----------
    df_exp, df_sim : pd.DataFrame
        Phase-duration DataFrames for experiment and simulation.
    dataset_exp, dataset_sim : DataSet
        Dataset enums for colour lookup.
    ax : matplotlib.axes.Axes
        Target axes (a sub-axes for the histogram is created automatically).
    model : {'model2', 'model3'}
        Model label (affects delay-bin defaults).
    gen : {'2nd', '3rd'}
        Nuclear generation.
    n_bootstrap : int
        Number of bootstrap resamples for CI estimation.
    scaling : float
        Time-scaling factor for simulated data.
    delta_bins, bins : float or array-like, optional
        Bin width / edges for the delay histogram.
    mask_exp, mask_sim : np.ndarray, optional
        Boolean masks to filter data rows.
    cax_ylim : tuple, optional
        y-limits for the histogram sub-axes.
    """
    # generate divider
    divider = mpl_axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("bottom", "66%", pad="20%")

    def get_label(dataset):
        if dataset.value in [DataSet.KLAUS.value, DataSet.AIRY.value]:
            label = "data"
        else:
            label = "model"
        return label

    if dataset_exp.equal(DataSet.KLAUS):
        if gen == "2nd":
            bin_max = 2.0
        else:
            bin_max = 2  # 1.
    else:
        if gen == "2nd":
            bin_max = 1.2
        else:
            bin_max = 2.4

    sync_exp, s_phase_exp = get_delay_and_sphase(df_exp, gen, mask=mask_exp)
    sync_sim, s_phase_sim = get_delay_and_sphase(df_sim, gen, mask=mask_sim)
    if delta_bins is None:
        delta_bins = np.min(np.diff(np.unique(sync_exp))) * 2
    if bins is None:
        bins = np.arange(0, np.ceil(bin_max / delta_bins) + 1) * delta_bins

    bins_return = bins
    weights = [
        np.ones_like(sync_exp),
        len(sync_exp) / len(sync_sim) * np.ones_like(sync_sim),
    ]
    cax.hist(
        [sync_exp, sync_sim],
        bins=bins,
        stacked=False,
        color=[
            colors_violin_shade_dict[dataset_exp.value],
            colors_violin_shade_dict[dataset_sim.value],
        ],
        linewidth=0.6,
        edgecolor="pplt:gray",
        # density=True,
        rwidth=0.6,
        weights=weights,
    )

    # add errorbars for data-hist
    choices = np.random.choice(
        np.arange(len(sync_exp)),
        size=(n_bootstrap, len(sync_exp)),
        replace=True,
    )
    sync_exp_bs_samples = sync_exp[choices]
    hist_exp_samples = np.array(
        [
            np.histogram(sync_exp_sample, bins=bins)[0]
            for sync_exp_sample in sync_exp_bs_samples
        ]
    )
    hist_exp = np.mean(hist_exp_samples, axis=0)
    hist_exp_std = np.std(hist_exp_samples, axis=0)
    for y, yerr, x in zip(hist_exp, hist_exp_std, bins[:-1]):
        cax.errorbar(
            x + (0.2 + 0.6 / 4) * delta_bins,
            y,
            yerr=yerr,
            fmt="none",
            color="pplt:gray",
        )

    # redefine bins such that they match to
    bins = bins[:-1] + delta_bins / 2

    for s_phase, sync, dataset in zip(
        [s_phase_exp, s_phase_sim],
        [sync_exp, sync_sim],
        [dataset_exp, dataset_sim],
    ):
        if dataset.equal(DataSet.RESMODEL) or dataset.equal(
            DataSet.RESPARMODEL
        ):
            rescaling = 1 / scaling
        else:
            rescaling = 1

        for bin_i in bins:
            if len(sync) > 10000:  # to avoid plotting a super flat long tail
                if bin_i > np.percentile(sync, 99):
                    continue
            mask = (sync >= bin_i - delta_bins / 2) & (
                sync < bin_i + delta_bins / 2
            )
            y_sample = s_phase[mask] / rescaling
            if len(y_sample) == 0:
                continue
            y_mean = np.mean(y_sample)

            choices = np.random.choice(
                np.arange(len(y_sample)),
                size=(n_bootstrap, len(y_sample)),
                replace=True,
            )
            y_bs_samples = y_sample[choices]

            y_mean_bs = np.mean(y_bs_samples, axis=1)
            y_std = np.std(y_mean_bs)

            x = [bin_i - delta_bins / 2, bin_i + delta_bins / 2]

            ax.plot(
                x,
                [y_mean] * 2,
                color=colors_violin_dict[dataset.value],
                label=get_label(dataset),
                solid_capstyle="butt",
            )

            # Fill the area between the rolling mean and the standard deviation
            ax.fill_between(
                x,
                [y_mean - y_std] * 2,
                [y_mean + y_std] * 2,
                facecolor=colors_violin_shade_dict[dataset.value],
                zorder=1,
            )

    # if model == 'model2':
    # ---- plot observed data
    x, y = sync_exp, s_phase_exp
    # add noise
    dx = np.diff(np.unique(x)).min()
    dy = np.diff(np.unique(y)).min()
    x += np.random.uniform(0, dx / 3, size=x.shape)  # + dx/4
    y += np.random.uniform(0, dy / 3, size=y.shape)
    ax.scatter(
        x,
        y,
        color=colors_violin_shade_dict[dataset_exp.value],
        label="data",
        **SCATTER_KWARGS,
    )

    if gen == "2nd":
        s = r"2-nuclei stage"
    else:
        s = r"4-nuclei stage"
    pplt.text(
        x=0.5,
        y=1,
        s=s,
        ha="center",
        va="top",
        ax=ax,
        fontsize=8,
        transform=ax.transAxes,
    )

    ax.set_xticks([])
    # ax.minorticks_on()
    cax.tick_params(axis="x", which="major", direction="out", labelbottom=True)
    cax.tick_params(
        axis="x",
        which="minor",
        direction="inout",
        labelbottom=False,
        length=ax.xaxis.majorTicks[0].tick1line.get_markersize(),
    )
    # cax.tick_params(axis='x', which='minor', bottom=True, labelbottom=False, direction='in')
    # cax.tick_params(axis='x', which='major', bottom=True, labelbottom=True)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel(" \n" r"S-phase (min)")
    cax.set_ylabel("count")  # , rotation=0, labelpad=15)
    cax.xaxis.set_major_locator(MultipleLocator(1))
    cax.xaxis.set_minor_locator(MultipleLocator(delta_bins))
    cax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    cax.set_xlabel(r"replication delay")
    # if model == 'model2':
    if dataset_exp.equal(DataSet.KLAUS):
        ax.set_ylim([15, 79])
        ax.set_yticks([25, 50, 75])
        ax.set_xlim([-0.14398222825992282, 2.3542184779708837])
        cax.set_xlim([-0.14398222825992282, 2.3542184779708837])
        if cax_ylim is None:
            cax.set_ylim([0, 48])
            cax.set_yticks([0, 24, 48])
        else:
            cax.set_ylim(cax_ylim)
            cax.set_yticks([0, 12, 24])
    else:
        ax.set_ylim([20, 105])
        ax.set_xlim([-0.12469635627530358, 1.5])  # 2.6249397580614087])
        cax.set_xlim([-0.12469635627530358, 1.5])  # 2.6249397580614087])
        cax.set_yticks([0, 15, 30])
        cax.set_ylim([0, 30])
    return bins_return, delta_bins


def plot_sphase_vs_delay_fast_slow(
    df,
    dataset,
    ax,
    n_bootstrap=10000,
    scaling=1,
    delta_bins=None,
    bins=None,
    mask_slow=None,
    mask_fast=None,
    cax_ylim=None,
    exp_data=False,
    counts_fast=None,
    counts_slow=None,
):
    """Plot S-phase vs. delay split by fast/slow mother replication.

    Same layout as :func:`plot_sphase_vs_delay` but the data are split
    into two groups (fast vs. slow mother) with separate colours.

    Parameters
    ----------
    df : pd.DataFrame
        Phase-duration DataFrame.
    dataset : DataSet
        Dataset enum for colour lookup.
    ax : matplotlib.axes.Axes
        Target axes.
    n_bootstrap : int
        Bootstrap resamples for CI estimation.
    scaling : float
        Time-scaling factor.
    delta_bins, bins : float or array-like, optional
        Bin width / edges.
    mask_slow, mask_fast : np.ndarray, optional
        Boolean masks selecting slow/fast mother rows.
    cax_ylim : tuple, optional
        y-limits for histogram sub-axes.
    exp_data : bool
        If *True*, overlay raw scatter points.
    counts_fast, counts_slow : int, optional
        Pre-computed histogram counts for weighting.
    """
    # generate divider
    divider = mpl_axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("bottom", "66%", pad="20%")

    def get_label(dataset):
        if exp_data:
            label = "data"
        else:
            label = "model"
        return label

    bin_max = 2.0

    sync_slow, s_phase_slow = get_delay_and_sphase(
        df, gen="2nd", mask=mask_slow
    )
    sync_fast, s_phase_fast = get_delay_and_sphase(
        df, gen="2nd", mask=mask_fast
    )
    if delta_bins is None:
        sync, s_phase = get_delay_and_sphase(df, gen="model3")
        delta_bins = np.min(np.diff(np.unique(sync))) * 2
    if bins is None:
        bins = np.arange(0, np.ceil(bin_max / delta_bins) + 1) * delta_bins

    if counts_slow is not None and counts_fast is not None:
        weights = [
            counts_slow / len(sync_slow) * np.ones_like(sync_slow),
            counts_fast / len(sync_fast) * np.ones_like(sync_fast),
        ]
    else:
        weights = [
            np.ones_like(sync_slow),
            np.ones_like(sync_fast),
        ]
    cax.hist(
        [sync_slow, sync_fast],
        bins=bins,
        stacked=False,
        color=[
            colors_fast_slow_dict[dataset.value]["slow_shade"],
            colors_fast_slow_dict[dataset.value]["fast_shade"],
        ],
        linewidth=0.6,
        edgecolor="pplt:gray",
        # density=True,
        rwidth=0.6,
        weights=weights,
    )

    # add errorbars for data-hist
    if exp_data:
        for i, sync in enumerate([sync_slow, sync_fast]):
            choices = np.random.choice(
                np.arange(len(sync)),
                size=(n_bootstrap, len(sync)),
                replace=True,
            )
            sync_exp_bs_samples = sync[choices]
            hist_exp_samples = np.array(
                [
                    np.histogram(sync_exp_sample, bins=bins)[0]
                    for sync_exp_sample in sync_exp_bs_samples
                ]
            )
            hist_exp = np.mean(hist_exp_samples, axis=0)
            hist_exp_std = np.std(hist_exp_samples, axis=0)
            for y, yerr, x in zip(hist_exp, hist_exp_std, bins[:-1]):
                cax.errorbar(
                    x + (0.2 + (0.6 + 1.2 * i) / 4) * delta_bins,
                    y,
                    yerr=yerr,
                    fmt="none",
                    color="pplt:gray",
                )

    # redefine bins such that they match to
    bins = bins[:-1] + delta_bins / 2

    for s_phase, sync, mode in zip(
        [s_phase_slow, s_phase_fast],
        [sync_slow, sync_fast],
        ["slow", "fast"],
    ):
        if not exp_data:
            rescaling = 1 / scaling
        else:
            rescaling = 1

        for bin_i in bins:
            if len(sync) > 10000:  # to avoid plotting a super flat long tail
                if bin_i > np.percentile(sync, 99):
                    continue
            mask = (sync >= bin_i - delta_bins / 2) & (
                sync < bin_i + delta_bins / 2
            )
            y_sample = s_phase[mask] / rescaling
            if len(y_sample) == 0:
                continue
            y_mean = np.mean(y_sample)

            choices = np.random.choice(
                np.arange(len(y_sample)),
                size=(n_bootstrap, len(y_sample)),
                replace=True,
            )
            y_bs_samples = y_sample[choices]

            y_mean_bs = np.mean(y_bs_samples, axis=1)
            y_std = np.std(y_mean_bs)

            x = [bin_i - delta_bins / 2, bin_i + delta_bins / 2]

            ax.plot(
                x,
                [y_mean] * 2,
                color=colors_fast_slow_dict[dataset.value][mode],
                label=get_label(dataset),
                solid_capstyle="butt",
            )

            # Fill the area between the rolling mean and the standard deviation
            ax.fill_between(
                x,
                [y_mean - y_std] * 2,
                [y_mean + y_std] * 2,
                facecolor=colors_fast_slow_dict[dataset.value][mode],
                alpha=0.2,
            )
    if exp_data:
        # # ---- plot observed data
        for sync, s_phase, mode in zip(
            [sync_slow, sync_fast],
            [s_phase_slow, s_phase_fast],
            ["slow", "fast"],
        ):
            x, y = sync, s_phase
            # add noise
            dx = np.diff(np.unique(x)).min()
            dy = np.diff(np.unique(y)).min()
            x += np.random.uniform(0, dx / 3, size=x.shape)  # + dx/4
            y += np.random.uniform(0, dy / 3, size=y.shape)
            ax.scatter(
                x,
                y,
                color=colors_fast_slow_dict[dataset.value][f"{mode}_shade"],
                **SCATTER_KWARGS,
            )

    ax.set_xticks([])
    cax.tick_params(axis="x", which="major", direction="out", labelbottom=True)
    cax.tick_params(
        axis="x",
        which="minor",
        direction="inout",
        labelbottom=False,
        length=ax.xaxis.majorTicks[0].tick1line.get_markersize(),
    )
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel(" \n" r"S-phase (min)")
    cax.set_ylabel("count")  # , rotation=0, labelpad=15)
    cax.xaxis.set_major_locator(MultipleLocator(1))
    cax.xaxis.set_minor_locator(MultipleLocator(delta_bins))
    cax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    cax.set_xlabel(r"replication delay")
    ax.set_ylim([15, 79])
    ax.set_yticks([25, 50, 75])
    ax.set_xlim([-0.14398222825992282, 2.3542184779708837])
    cax.set_xlim([-0.14398222825992282, 2.3542184779708837])
    if cax_ylim is None:
        cax.set_ylim([0, 48])
        cax.set_yticks([0, 24, 48])
    else:
        cax.set_ylim(cax_ylim)
        cax.set_yticks([0, 12, 24])

    counts_fast = weights[1].sum()
    counts_slow = weights[0].sum()
    return counts_slow, counts_fast


def convert_phases_to_dict(df):
    """Convert a phase-duration DataFrame to the dict format expected by :func:`get_delay_and_sphase`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``s``, ``s0``, ``s1``, ``d0``, ``d1``,
        ``d00``, ``d01``, ``d10``, ``d11``.

    Returns
    -------
    dict
        Keys ``'s'``, ``'sx'``, ``'dx'``, ``'dxx'`` mapping to flat arrays.
    """

    keys = ["s", "sx", "dx", "dxx"]
    columns = [["s"], ["s0", "s1"], ["d0", "d1"], ["d00", "d01", "d10", "d11"]]

    result = {
        key: df[column].dropna().values.flatten()
        for key, column in zip(keys, columns)
    }
    return result


def plot_replicating_nuclei(
    dataset_exp, data_exp, dataset_sim, data_sim, ax, model="model2", scaling=1
):
    """Plot mean number of replicating nuclei over time (data vs. model).

    Parameters
    ----------
    dataset_exp, dataset_sim : DataSet
        Dataset enums.
    data_exp, data_sim : dict
        Data dictionaries containing ``'bulk'`` with time and ns traces.
    ax : matplotlib.axes.Axes
        Target axes.
    model : str
        Model identifier.
    scaling : float
        Time-scaling factor for the simulation.
    """
    for dataset, data in zip([dataset_exp, dataset_sim], [data_exp, data_sim]):
        t, ns = data["bulk"]["t"], data["bulk"]["ns"]["mean"]
        if dataset.equal(DataSet.BARMODEL):
            ns[t >= 600] = np.nan
        if dataset.equal(DataSet.RESMODEL) or dataset.equal(
            DataSet.RESPARMODEL
        ):
            rescaling = 1 / scaling
            t = t / rescaling
        ax.plot(
            t,
            ns,
            color=colors_violin_dict[dataset.value],
            # solid_capstyle='butt',
        )

        # Fill the area between the rolling mean and the standard deviation
        if dataset.equal(dataset_exp):
            ax.fill_between(
                t,
                data["bulk"]["ns"]["mean"] - data["bulk"]["ns"]["mean_std"],
                data["bulk"]["ns"]["mean"] + data["bulk"]["ns"]["mean_std"],
                facecolor=colors_violin_dict[dataset.value],
                alpha=0.2,
            )

    ax.set_ylabel(r"# replicating nuclei $n_S$")
    ax.set_xlabel(r"time $t$ (min)")
    ax.set_xlim([-400, 800])

    ax.set_ylim([None, 4.9])

    ax.set_xticks([-400, -200, 0, 200, 400, 600, 800])
    new_label = [
        "-400",
        "-200",
        r"t$_{\text{S}_1}$",
        "+200",
        "+400",
        "+600",
        "+800",
    ]
    ax.set_xticklabels(new_label)


def plot_nuclei_count(dataset, data, ax):
    """Plot total nuclei count on a log-scale y-axis with individual traces.

    Parameters
    ----------
    dataset : DataSet
        Dataset enum for colour lookup.
    data : dict
        Data dictionary with ``'sc'`` (single-cell traces) and ``'bulk'``.
    ax : matplotlib.axes.Axes
        Target axes.
    """
    for sc in data["sc"].values():
        ax.plot(
            sc["t"], sc["n"], color=colors_violin_dict[dataset.value], alpha=0.2
        )

    t_bulk = data["bulk"]["t"]
    n_bulk = data["bulk"]["n"]["mean"]
    mask = t_bulk < 500

    ax.plot(t_bulk[mask], n_bulk[mask], color=colors_violin_dict[dataset.value])
    ax.set_ylabel("number of nuclei $n$")
    ax.set_xlabel(r"time $t$ (min)")
    ax.set_xlim([-200, 800])

    # ax.yaxis.set_major_locator(FixedLocator(2**np.arange(1,6)))
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8, 16])
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    # ax.yaxis.set_major_locator(MultipleLocator(2))
    # Format the ticklabel to be 2 raised to the power of `x`
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(2**x)))
    ax.set_xticks([-200, 0, 200, 400, 600, 800])
    ax.set_xticklabels(["-200", "t$_{S_1}$", "+200", "+400", "+600", "+800"])


def plot_resource_fitting(
    data_exp,
    zeta_low,
    zeta_high,
    axs,
):
    """Fit and plot sigmoidal resource curves (PCNA1 / mCherry) with zeta bounds.

    Each signal is normalised, fitted to a logistic, and overlaid with
    individual traces. The last axes panel shows the inferred resource
    envelope ``r(t) = zeta * sigmoid(t)``.

    Parameters
    ----------
    data_exp : dict
        Experimental data containing ``'pcna1'`` sub-dict.
    zeta_low, zeta_high : float
        Lower / upper bounds of the scarcity parameter zeta.
    axs : array-like of Axes
        At least 3 axes (PCNA1, mCherry, resource overlay).
    """

    # --- Fit logistic with shared t0 across all traces ---
    # Each trace i has its own r_0_i, r_final_i, k_i, but t0 is shared.
    # Parameter vector: [t0, r_0_0, r_final_0, k_0, r_0_1, r_final_1, k_1, ...]
    from scipy.optimize import curve_fit

    labels = ["pcna1", "mCherry"]
    t12, k = np.empty((2, len(labels)))

    def resource(t, *params):
        t12, k = params
        return 1 / (1 + np.exp(-k * (t - t12)))

    for i, (ax, label) in enumerate(zip(axs, labels)):
        # Plot aligned data (subtract each trace's fitted r_0 so growth starts at 0)
        time = data_exp["pcna1"]["t"]
        t_first_to_egress = 853.6363636363636
        mask = (time < 0) & (time > -t_first_to_egress)

        # Collect all single-cell traces
        traces = np.array(
            [
                data_exp["pcna1"][label][key]
                for key in data_exp["pcna1"][label].keys()
            ]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=RuntimeWarning)
            mean_trace = np.nanmean(traces, axis=0)
        if label == "pcna1":
            color = "#008B00"
            color_shade = "#98DD98"
        else:
            color = "#AC0000"
            color_shade = "#C68181"

        ax.plot(
            time,
            traces.T,
            color=color_shade,
            lw=0.5,
            zorder=0,
        )

        # add merged fit
        ax.plot(
            time[mask],
            mean_trace[mask],
            color=color,
            zorder=1,
        )

        ylim = ax.get_ylim()
        ax.vlines(
            0,
            ylim[0],
            ylim[1] * 0.95,
            color="black",
            linestyle="--",
            lw=0.5,
        )
        ax.text(
            0,
            ylim[1],
            r"egress",
            color="black",
            ha="center",
            va="bottom",
            fontsize=6,
        )

        popt, pcov = curve_fit(
            resource,
            time[mask],
            mean_trace[mask],
            p0=[-500.0, 0.01],
            # bounds=(lower, upper),
            maxfev=50000,
        )
        ax.plot(
            time[mask],
            resource(time[mask], *popt),
            color="black",
            zorder=4,
            linestyle="--",
            lw=1,
        )
        # ax.plot(
        #     time,
        #     resource(time, *popt),
        #     color=COLOR_SEQ_SHADE,
        #     # lw=0.5,
        #     zorder=4,
        # )
        t12[i] = popt[0]
        k[i] = popt[1]

        # Mark the shared t0
        ax.vlines(
            popt[0],
            ylim[0],
            ylim[1] * 0.95,
            color="black",
            linestyle="--",
            lw=0.5,
            zorder=1,
        )
        ax.text(
            popt[0],
            ylim[1],
            r"$t_{{1/2}}$",
            color="black",
            ha="center",
            va="bottom",
            fontsize=6,
        )
        ax.set_ylim(ylim)
        ylabel = "PCNA1" if label == "pcna1" else "mCherry"
        ax.set_ylabel(ylabel + r" intensity")

    t12_mean = t12.mean()
    k_mean = k.mean()
    zeta = (zeta_low + zeta_high) / 2
    zeta_std = (zeta_high - zeta_low) / np.sqrt(
        12
    )  # std of uniform distribution
    time = np.linspace(-t_first_to_egress, 1000 - t_first_to_egress, 300)
    axs[-1].fill_between(
        time,
        (zeta - zeta_std) * resource(time, *[t12_mean, k_mean]),
        (zeta + zeta_std) * resource(time, *[t12_mean, k_mean]),
        color=COLOR_SEQ_SHADE,
        zorder=2,
    )
    axs[-1].plot(
        time,
        zeta * resource(time, *[t12_mean, k_mean]),
        color=COLOR_SEQ,
        zorder=4,
    )
    axs[-1].plot(
        time,
        zeta_low * resource(time, *[t12_mean, k_mean]),
        color=COLOR_SEQ,
        zorder=4,
        linestyle="--",
    )
    axs[-1].plot(
        time,
        zeta_high * resource(time, *[t12_mean, k_mean]),
        color=COLOR_SEQ,
        zorder=4,
        linestyle="--",
    )
    axs[-1].text(
        time[-1] + 25,
        zeta_low,
        r"$\xi_{low}$",
        color="black",
        ha="left",
        va="center",
        fontsize=6,
    )
    axs[-1].text(
        time[-1] + 25,
        zeta_high,
        r"$\xi_{high}$",
        color="black",
        ha="left",
        va="center",
        fontsize=6,
    )
    axs[-1].set_ylim([0, None])
    # axs[-1].set_yticks(np.array([0, 0.5, 1.0, 1.5, zeta_low, zeta_high]))
    # axs[-1].set_yticklabels([0, 0.5, 1.0, 1.5, r"$\xi_{low}$", r"$\xi_{high}$"])

    axs[-1].set_ylabel(r"resource $r(t)$")

    x_ticks = [
        -t_first_to_egress,
        250 - t_first_to_egress,
        500 - t_first_to_egress,
        750 - t_first_to_egress,
        1000 - t_first_to_egress,
    ]
    x_ticks_labels = ["t$_{S}$", "+250", "+500", "+750", "+1000"]
    for ax in axs:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_labels)
        ax.set_xlabel(r"time $t$ (min)")


def plot_seq_vs_par_comparison(
    dataset_exp, data_exp, data_sim_seq, data_sim_par, ax, scaling=1
):
    """Box-plot comparing nuclear multiplication duration across data, sequential, and parallel models.

    Parameters
    ----------
    dataset_exp : DataSet
        Experimental dataset enum.
    data_exp, data_sim_seq, data_sim_par : dict
        Data dictionaries with ``'bulk'['2nd to last S-phase']``.
    ax : matplotlib.axes.Axes
        Target axes.
    scaling : float
        Time-scaling factor for simulated data.
    """
    rescaling = 1 / scaling
    ylabel = "nuclear multiplication\n" + r"2$^\text{nd}$ to last S-phase (min)"
    df = pd.DataFrame(
        {
            ylabel: np.concatenate(
                [
                    data_exp["bulk"]["2nd to last S-phase"],
                    data_sim_seq["bulk"]["2nd to last S-phase"] / rescaling,
                    data_sim_par["bulk"]["2nd to last S-phase"] / rescaling,
                ]
            ),
            "model": np.concatenate(
                [
                    ["data"] * len(data_exp["bulk"]["2nd to last S-phase"]),
                    ["sequential"]
                    * len(data_sim_seq["bulk"]["2nd to last S-phase"]),
                    ["parallel"]
                    * len(data_sim_par["bulk"]["2nd to last S-phase"]),
                ]
            ),
        }
    )

    ratio = np.nanmean(
        data_sim_par["bulk"]["2nd to last S-phase"]
    ) / np.nanmean(data_sim_seq["bulk"]["2nd to last S-phase"])
    diff_min = (
        np.nanmean(data_sim_par["bulk"]["2nd to last S-phase"])
        - np.nanmean(data_sim_seq["bulk"]["2nd to last S-phase"])
    ) / rescaling
    sns.boxplot(
        x="model",
        y=ylabel,
        hue="model",
        data=df,
        showfliers=False,
        ax=ax,
        width=0.5,
        legend=False,
        palette={
            "data": colors_violin_shade_dict[dataset_exp.value],
            "sequential": colors_violin_shade_dict[DataSet.RESMODEL.value],
            "parallel": colors_violin_shade_dict[DataSet.RESPARMODEL.value],
        },
    )


def plot_duration_comparison(ax):
    """Violin plot comparing S/D phase durations between Klaus and Airyscan datasets."""
    coln = [["s"], ["d0", "d1"], ["s0", "s1"], ["d00", "d01", "d10", "d11"]]
    key_coln = ["S", "D$_i$", "S$_i$", "D$_{ij}$"]
    phases, datasets, durations = [], [], []
    for c, key in zip(coln, key_coln):
        mean = []
        for dataset in [DataSet.KLAUS, DataSet.AIRY]:
            data = get_data(dataset=dataset, drop_data=DropData.SUBTREE)
            duration = data["sd"][c].dropna().to_numpy().flatten()
            phases.append(len(duration) * [key])
            datasets.append(len(duration) * [dataset.value])
            durations.append(duration)
            mean.append(np.mean(duration))

    df = pd.DataFrame(
        {
            "duration (min)": np.concatenate(durations),
            "phase": np.concatenate(phases),
            "dataset": np.concatenate(datasets),
        }
    )

    sns.violinplot(
        x="phase",
        y="duration (min)",
        hue="dataset",
        split=True,
        data=df,
        ax=ax,
        inner="quartile",
        gap=0.1,
        width=1,
        cut=0,
        # scale='count',
        palette=[
            colors_violin_shade_dict[DataSet.KLAUS.value],
            colors_violin_shade_dict[DataSet.AIRY.value],
        ],
    )


def compute_bulk_statistics(data_sc, model="model2"):
    """Aggregate single-cell traces into bulk summary statistics.

    Aligns all traces to the second S-phase onset, interpolates onto a
    common time grid, and computes mean, std, percentiles of replicating-nuclei
    count, resource level, and event durations.

    Parameters
    ----------
    data_sc : dict
        Mapping of cell ID to single-cell trace dictionaries.
    model : {'model2', 'model3'}
        Model identifier (affects alignment and interpolation).

    Returns
    -------
    dict
        Keys: ``'t'``, ``'ns'``, ``'r'``, ``'ns_binary'``,
        ``'1st to last S-phase'``, ``'2nd to last S-phase'``.
    """
    first_to_last, second_to_last = np.zeros((2, len(data_sc)))
    #
    if model == "model3":
        data_sc = data_sc.copy()
        for i, (key, d) in enumerate(data_sc.items()):
            idx = np.where(np.diff(d["replicating nuclei at t"]) > 0)[0][1] + 1
            t_offset = d["time"][idx]
            data_sc[key]["time"] = d["time"] - t_offset

            # Find the last index where ns drops to 0 (end of last S-phase)
            ns_trace = d["replicating nuclei at t"]
            # Last index where ns > 0, then +1 is where it drops to 0
            nonzero_indices = np.where(ns_trace > 0)[0]
            last_s_end_idx = nonzero_indices[-1]
            # Use the first zero AFTER the last non-zero (where trace
            # actually reaches 0 via interpolation) so that the duration
            # is consistent with the plotted trace.
            last_s_zero_idx = min(last_s_end_idx + 1, len(ns_trace) - 1)
            # First index where ns > 0 (start of first S-phase)
            first_s_start_idx = nonzero_indices[0]

            # Time from first S-phase start to last S-phase end
            first_to_last[i] = (
                d["time"][last_s_zero_idx] - d["time"][first_s_start_idx]
            )

            # For second_to_last: find the start of the 2nd S-phase
            # The 2nd S-phase starts at the 2nd positive transition in ns
            # Use idx (= pos_transitions[1] + 1) to be consistent with
            # the trace alignment (t=0 at idx).
            pos_transitions = np.where(np.diff(ns_trace) > 0)[0]
            if len(pos_transitions) >= 2:
                second_s_start_idx = pos_transitions[1] + 1
            else:
                second_s_start_idx = first_s_start_idx
            second_to_last[i] = (
                d["time"][last_s_zero_idx] - d["time"][second_s_start_idx]
            )
            # first_to_last[i] = (
            #     d["s*-phase-dict"].max() - d["s*-phase-dict"].min()
            # )
            # second_to_last[i] = (
            #     d["s*-phase-dict"].max() - d["s*-phase-dict"][1:3].min()
            # )
    else:
        first_to_last, second_to_last = np.nan, np.nan
    # process data so that plotting routine is simplified
    # 1. Get t: Get tmin and tmax
    tmin, tmax = np.swapaxes(
        np.array(
            [[d["time"].min(), d["time"].max()] for d in data_sc.values()]
        ),
        0,
        1,
    )
    tmin, tmax = np.min(tmin), np.max(tmax)
    if model == "model2":
        t = np.arange(tmin, 1 + tmax)
    else:
        t = np.linspace(tmin, tmax, 1000)

    # 2. estimate ns and n (take mean for the beginning)
    ns = np.empty((len(data_sc), len(t)))
    ns.fill(np.nan)

    r = np.empty((len(data_sc), len(t)))
    r.fill(np.nan)

    if model == "model2":
        for i, d in enumerate(data_sc.values()):
            offset = np.argmax(t >= d["time"][0])
            ns[i][offset : offset + len(d["replicating nuclei at t"])] = d[
                "replicating nuclei at t"
            ]
            ns[i][0:offset] = 0
    else:
        for i, d in enumerate(data_sc.values()):

            def get_ns(t):
                y = np.interp(
                    t,
                    d["time"],
                    d["replicating nuclei at t"],
                    left=0.0,
                    right=0.0,
                )
                return y

            ns[i] = get_ns(t)

            r[i] = np.interp(t, d["time"][1:], d["r"][1:], left=0.0)

    ns_binary = ns.copy()
    ns_binary[ns > 1] = 1

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        return {
            "t": t,
            "ns": {
                "mean": np.nanmean(ns, axis=0),
                "std": np.nanstd(ns, axis=0),
                "median": np.nanpercentile(ns, q=50, axis=0, method="linear"),
                "iqr_low": np.nanpercentile(ns, q=25, axis=0, method="linear"),
                "iqr_high": np.nanpercentile(ns, q=75, axis=0, method="linear"),
                "traces": ns,
            },
            "r": {
                "mean": np.nanmean(r, axis=0),
                "std": np.nanstd(r, axis=0),
                "median": np.nanpercentile(r, q=50, axis=0, method="linear"),
                "iqr_low": np.nanpercentile(r, q=25, axis=0, method="linear"),
                "iqr_high": np.nanpercentile(r, q=75, axis=0, method="linear"),
                "min": np.nanmin(r, axis=0),
                "max": np.nanmax(r, axis=0),
            },
            "ns_binary": {
                "mean": np.nanmean(ns_binary, axis=0),
                "std": np.nanstd(ns_binary, axis=0),
            },
            "1st to last S-phase": first_to_last,
            "2nd to last S-phase": second_to_last,
        }


def plot_resource_by_delay(df_exp, data_sim, ax1, scaling=1):
    """Plot simulated resource concentration dynamics split by low vs. high replication delay.

    Parameters
    ----------
    df_exp : pd.DataFrame
        Experimental phase-duration DataFrame (used to set delay bins).
    data_sim : dict
        Simulation data with ``'sd'`` DataFrame and ``'sc'`` single-cell traces.
    ax1 : matplotlib.axes.Axes
        Target axes; a second axes is appended to the right.
    scaling : float
        Time-scaling factor.
    """

    # estimate delay range depending on the exp. bins
    sync_exp, _ = get_delay_and_sphase(df_exp, gen="2nd")
    delta_bins = np.min(np.diff(np.unique(sync_exp))) * 2
    d_ranges = [[0, delta_bins], [3 * delta_bins, 4 * delta_bins]]

    # estimate replication delay
    df = data_sim["sd"]
    delay = (df.d1 - df.d0) / np.nanmean(df.s0)

    timepoints = np.linspace(-0.2, 1.2, 100)
    c1, c2 = np.swapaxes(
        np.array(
            [
                [
                    np.interp(
                        timepoints,
                        rbc["time"],
                        rbc["f"][:, rbc["names"] == 2].flatten(),
                    ),
                    np.interp(
                        timepoints,
                        rbc["time"],
                        rbc["f"][:, rbc["names"] == 3].flatten(),
                    ),
                ]
                for rbc in data_sim["sc"].values()
            ]
        ),
        0,
        1,
    )

    # sort c1 and c2, such that c1 always starts first
    mask = np.argmax(c1 > 0.001, axis=1) > np.argmax(c2 > 0.001, axis=1)
    c1[mask], c2[mask] = c2[mask], c1[mask]

    divider = mpl_axes_grid1.make_axes_locatable(ax1)
    ax2 = divider.append_axes("right", "100%", pad="20%")

    timepoints = timepoints * scaling
    for ax, (d_min, d_max) in zip([ax1, ax2], d_ranges):
        mask_delay = (delay < d_max) & (delay >= d_min)
        (line,) = ax.plot(
            timepoints,
            np.median(c1[mask_delay], axis=0),
            color=colors_violin_dict[DataSet.RESMODEL.value],
            label="nucleus 1",
        )
        ax.fill_between(
            timepoints,
            np.percentile(c1[mask_delay], q=25, axis=0),
            np.percentile(c1[mask_delay], q=75, axis=0),
            facecolor=line.get_color(),
            alpha=0.2,
        )
        # ax.fill_between(
        #     timepoints,
        #     np.mean(c1[mask_delay], axis=0) - np.std(c1[mask_delay], axis=0),
        #     np.mean(c1[mask_delay], axis=0) + np.std(c1[mask_delay], axis=0),
        #     facecolor=line.get_color(),
        #     alpha=0.2)
        (line,) = ax.plot(
            timepoints,
            np.median(c2[mask_delay], axis=0),
            color=colors_violin_dict[DataSet.RESMODEL.value],
            linestyle=":",
            label="nucleus 2",
        )

        ax.fill_between(
            timepoints,
            np.percentile(c2[mask_delay], q=25, axis=0),
            np.percentile(c2[mask_delay], q=75, axis=0),
            facecolor=line.get_color(),
            alpha=0.2,
        )
        # ax.fill_between(
        #     timepoints,
        #     np.mean(c2[mask_delay], axis=0) - np.std(c2[mask_delay], axis=0),
        #     np.mean(c2[mask_delay], axis=0) + np.std(c2[mask_delay], axis=0),
        #     facecolor=line.get_color(),
        #     alpha=0.2)
        ax.set_xlabel(r"time $t$ (min)")

        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([0, 40, 80])
        new_label = [r"t$_{\text{S}_1}$", "+40", "+80"]
        ax.set_xticklabels(new_label)
        ax.set_ylim([-0.1, 1.2])
        ax.set_xlim([-20, 90])
    ax1.set_ylabel(r"concentration $c(t)$")
    pplt.legend(ax1, frameon=False)
    ax1.set_ylabel(r"concentration $c(t)$")
    ax2.set_yticklabels([])


# ---------------------------------------------------------------------------
# Correlation heatmap plots (merged from correlation_plot.py)
# ---------------------------------------------------------------------------


def correlationplot(
    corr,
    pval,
    x_columns,
    y_columns,
    ax,
    cmap=None,
    cbarlabel="",
    show_legend=None,
    show_cbar=True,
):
    """Create a correlation heatmap with circle sizes encoding p-values.

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix indexed/columned by *x_columns* and *y_columns*.
    pval : pd.DataFrame
        Corresponding p-value matrix.
    x_columns, y_columns : list[str]
        Labels for the x- and y-axes.
    ax : matplotlib.axes.Axes
        Target axes.
    cmap : matplotlib.colors.Colormap, optional
        Colour map; a blue-white-red diverging map is used when *None*.
    cbarlabel : str
        Label for the colour bar.
    show_legend : bool, optional
        If *True*, add a legend mapping circle size to significance levels.
    show_cbar : bool
        Whether to draw the colour bar.
    """
    if cmap is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        clist = [cycle[3], (0.97, 0.97, 0.97), cycle[2]]
        cmap = LinearSegmentedColormap.from_list("custom_blue_red", clist)

    bounds = np.linspace(-1, 1, 11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    im = pplt.imshow(
        corr, origin="upper", norm=norm, cmap=cmap, aspect="equal", ax=ax
    )
    if show_cbar:
        pplt.colorbar(im, label=cbarlabel, position="right")
    im.remove()
    ax.set_xticks(np.arange(len(x_columns)))
    ax.set_yticks(np.arange(len(y_columns)))
    ax.set_xticklabels(x_columns)
    ax.set_yticklabels(y_columns)
    ax.set_xlim([-0.75, len(x_columns) - 0.25])
    ax.set_ylim([len(y_columns) - 0.25, -0.75])

    def get_color(corr):
        if not np.isfinite(corr).all():
            return "black"
        index = bounds[:-1] <= corr
        bounds_to_color = np.linspace(0, 1, bounds.shape[0] - 1)
        color = np.array(cmap(bounds_to_color[index][-1]))
        return color

    circles = [
        plt.Circle(
            (i, j),
            radius=_get_size_of_circle(pval[x_columns[i]][y_columns[j]]),
            linewidth=0,
            fill=True,
        )
        for i, j in itertools.product(
            np.arange(len(x_columns)), np.arange(len(y_columns))
        )
    ]
    colors = [
        get_color(corr[x][y])
        for x, y in itertools.product(x_columns, y_columns)
    ]
    c = mpl.collections.PatchCollection(circles, facecolors=colors)
    ax.add_collection(c)
    scatterkwargs = {
        "color": "pplt:gray",
        "marker": "o",
        "alpha": 0.75,
        "linewidth": 0.0,
    }
    if show_legend:
        size4 = (0.85 * _linewidth_from_data_units(0.95, ax)) ** 2
        size3 = (0.85 * _linewidth_from_data_units(0.70, ax)) ** 2
        size2 = (0.85 * _linewidth_from_data_units(0.45, ax)) ** 2
        size1 = (0.85 * _linewidth_from_data_units(0.20, ax)) ** 2

        ax.scatter(-100, 0, s=size4, label="$p<0.001$", **scatterkwargs)
        ax.scatter(-100, 0, s=size3, label="$p<0.01$", **scatterkwargs)
        ax.scatter(-100, 0, s=size2, label="$p<0.05$", **scatterkwargs)
        ax.scatter(-100, 0, s=size1, label="n.s.", **scatterkwargs)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.6, 0.5),
            handletextpad=1.15,
            labelspacing=1.15,
            frameon=False,
        )


def correlationplot_dual(
    corr_data,
    pval_data,
    corr_sim,
    pval_sim,
    x_columns,
    y_columns,
    ax,
    cmap=None,
    cbarlabel="",
    show_legend=None,
    show_cbar=True,
):
    """Create a dual half-circle correlation heatmap (data vs. simulation).

    Each cell shows two half-circles (upper = data, lower = simulation) whose
    size encodes the p-value and colour encodes the Pearson correlation.

    Parameters
    ----------
    corr_data, corr_sim : pd.DataFrame
        Correlation matrices for data and simulation.
    pval_data, pval_sim : pd.DataFrame
        Corresponding p-value matrices.
    x_columns, y_columns : list[str]
        Axis labels.
    ax : matplotlib.axes.Axes
        Target axes.
    cmap : matplotlib.colors.Colormap, optional
        Diverging colour map (default: blue-white-red).
    cbarlabel : str
        Colour-bar label.
    show_legend, show_cbar : bool
        Toggle legend / colour bar.
    """
    if cmap is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        clist = [cycle[3], (0.97, 0.97, 0.97), cycle[2]]
        cmap = LinearSegmentedColormap.from_list("custom_blue_red", clist)

    bounds = np.linspace(-1, 1, 21)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    im = pplt.imshow(
        corr_data, origin="upper", norm=norm, cmap=cmap, aspect="equal", ax=ax
    )
    if show_cbar:
        cbar = pplt.colorbar(
            im, label=cbarlabel, position="right", ticks=[-1, -0.5, 0, 0.5, 1]
        )
        cbar.ax.minorticks_off()
    im.remove()
    ax.set_xticks(np.arange(len(x_columns)))
    ax.set_yticks(np.arange(len(y_columns)))
    ax.set_xticklabels(x_columns)
    ax.set_yticklabels(y_columns)
    ax.set_xlim([-0.75, len(x_columns) - 0.25])
    ax.set_ylim([len(y_columns) - 0.25, -0.75])

    for corr, pval, angles in zip(
        [corr_data, corr_sim], [pval_data, pval_sim], [[180, 360], [0, 180]]
    ):
        circles = [
            mpatches.Wedge(
                (i, j),
                _get_size_of_circle(pval[x_columns[i]][y_columns[j]]),
                angles[0],
                angles[1],
                fill=True,
            )
            for i, j in itertools.product(
                np.arange(len(x_columns)), np.arange(len(y_columns))
            )
        ]

        colors = [
            _get_color_of_corr(corr[x][y], bounds=bounds, cmap=cmap)
            for x, y in itertools.product(x_columns, y_columns)
        ]
        c = mpl.collections.PatchCollection(
            circles, fc=colors, ec="pplt:gray", lw=0.8
        )
        ax.add_collection(c)
    scatterkwargs = {
        "edgecolor": "pplt:gray",
        "facecolor": "#BDBFC2",
        "marker": "o",
    }
    if show_legend:
        scaling_factor = 2.3
        size4 = (
            _linewidth_from_data_units(
                scaling_factor * _get_size_of_circle(0), ax
            )
            ** 2
        )
        size3 = (
            _linewidth_from_data_units(
                scaling_factor * _get_size_of_circle(0.005), ax
            )
            ** 2
        )
        size2 = (
            _linewidth_from_data_units(
                scaling_factor * _get_size_of_circle(0.02), ax
            )
            ** 2
        )
        size1 = (
            _linewidth_from_data_units(
                scaling_factor * _get_size_of_circle(1), ax
            )
            ** 2
        )

        ax.scatter(-100, 0, s=size4, label="$p<0.001$", **scatterkwargs)
        ax.scatter(-100, 0, s=size3, label="$p<0.01$", **scatterkwargs)
        ax.scatter(-100, 0, s=size2, label="$p<0.05$", **scatterkwargs)
        ax.scatter(-100, 0, s=size1, label="n.s.", **scatterkwargs)
        ax.legend(
            loc="upper right",
            handletextpad=1.2,
            labelspacing=2,
            frameon=False,
            ncol=2,
        )
    ax.set_aspect("equal", anchor="SW")


def _get_size_of_circle(pval):
    """Map a p-value to a circle radius for correlation heatmaps."""
    if not np.isfinite(pval).all():
        return 0
    elif pval >= 0.05:
        return 0.275
    elif pval >= 0.01:
        return 0.325
    elif pval >= 0.001:
        return 0.375
    else:
        return 0.425


def _get_color_of_corr(corr, bounds, cmap):
    """Map a correlation value to a colour using discretised bins."""
    if not np.isfinite(corr).all():
        return "black"
    index = bounds[:-1] <= corr
    bounds_to_color = np.linspace(0, 1, bounds.shape[0] - 1)
    color = np.array(cmap(bounds_to_color[index][-1]))
    return color


def _linewidth_from_data_units(linewidth, axis, reference="y"):
    """Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth : float
        Linewidth in data units of the respective reference-axis.
    axis : matplotlib.axes.Axes
        Axis used for the data-to-display transformation.
    reference : {'x', 'y'}
        Which axis to use as reference (default ``'y'``).

    Returns
    -------
    float
        Linewidth in points.
    """
    fig = axis.get_figure()
    if reference == "x":
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())[0]
    elif reference == "y":
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.abs(np.diff(axis.get_ylim())[0])
    length *= 72
    return linewidth * (length / value_range)
