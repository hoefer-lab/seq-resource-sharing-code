# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Model-2 statistical diagnostics — phase distributions and pooled correlations.

Provides :func:`plot_phases_share_dist` (violin plots with KS tests)
and :func:`plot_corr_pooling` (subtree-resolved correlation panels).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prettypyplot as pplt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits import axes_grid1 as mpl_axes_grid1
from scipy.stats import ks_2samp, pearsonr

from srs.lib.correlation_matrix import normal_transformation_df
from srs.lib.plotting import convert_phases_to_dict
from srs.lib.style import colors_violin_dict, colors_violin_shade_dict

ROW_COL = ["d", "s", "dx", "dy", "sx", "sy"]
SUBTREES = ["1", "2+3", "pooled"]  # , '2', '3'

ALL_PHASES = [
    "s",
    "s0",
    "s1",
    "s00",
    "s01",
    "s10",
    "s11",
    "d0",
    "d1",
    "d00",
    "d01",
    "d10",
    "d11",
]
SX_ALL = ["s0", "s1", "s00", "s01", "s10", "s11"]
DXX = ["d00", "d01", "d10", "d11"]


def plot_phases_share_dist(df_exp, fig, dataset_exp, grid_full, n_bs=50000):
    """Plot phase-duration distributions with bootstrap confidence bands.

    Compares mean S- and D-phase durations across subtree generations,
    overlaying bootstrap resampled means and KDE density estimates.

    Parameters
    ----------
    df_exp : pd.DataFrame
        Experimental phase-duration DataFrame.
    fig : matplotlib.figure.Figure
        Figure for adding axes.
    dataset_exp : DataSet
        Dataset enum for colour lookup.
    grid_full : np.ndarray
        Grid of subplot axes.
    n_bs : int
        Number of bootstrap resamples.
    """

    grid = grid_full[:, :3]
    gs = grid_full[0, 3].get_gridspec()
    divider = mpl_axes_grid1.make_axes_locatable(grid_full[0, 3])
    cax1 = divider.append_axes("bottom", "100%", pad="40%")
    divider = mpl_axes_grid1.make_axes_locatable(grid_full[1, 3])
    cax2 = divider.append_axes("bottom", "100%", pad="40%")
    axs_common = [grid_full[0, 3], cax1, grid_full[1, 3], cax2]

    cmap = plt.get_cmap("summertimes")
    edgecolors = cmap(np.linspace(0, 1, 6))

    x_labels = [
        ["d0", "d1"],
        ["d00", "d01", "d10", "d11"],
        ["s0", "s1", "s00", "s01", "s10", "s11"],
    ]

    titles = [
        [r"D$_1$", r"D$_2$"],
        [r"D$_{i1}$", r"D$_{i2}$"],
        [r"S$_i$ and S$_{ij}$"],
    ]
    labels = [
        ["", ""],
        [r"$i=1$", r"$i=2$"],
        [r"$i=1$", r"$i=2$", r"$ij=11$", r"$ij=12$", r"$ij=21$", r"$ij=22$"],
    ]
    for i, (x_label, title, label) in enumerate(zip(x_labels, titles, labels)):
        # apply 'subtree' dropna
        df_x = df_exp[x_label].copy()
        x = df_x.to_numpy()
        bins = np.arange(np.nanmin(x) - 5, np.nanmax(x) + 10, 5).astype(int)

        # define common distribution
        merged_distr = x[~np.isnan(x)]
        m = len(x_label)
        x_c = np.random.choice(merged_distr, (m, n_bs), replace=True)
        if x_label[0].startswith("d"):
            # sort such that sister '0' starts always first, i.e. x_0 <= x_1
            x_c = np.sort(x_c.reshape(m // 2, 2, n_bs), axis=1).reshape(m, n_bs)

        if i == 1:
            colors = edgecolors[2 : len(x_label) + 2]
        else:
            colors = edgecolors[: len(x_label)]
        # split plots in case of dx and dxx
        if x_label[0].startswith("d"):
            x_sublabels = [x_label[::2], x_label[1::2]]
            x_model = [x_c[::2], x_c[1::2]]
            axs = grid[:, i]
            colors_sub = [
                colors[: len(x_sublabels[0])],
                colors[len(x_sublabels[0]) :],
            ]
            x_data = [x[:, ::2], x[:, 1::2]]
        else:
            x_sublabels = [x_label]
            x_model = [x_c]
            gs = grid[0, i].get_gridspec()
            for ax in grid[:, i]:
                ax.remove()
            axbig = fig.add_subplot(gs[0:2, i])
            axs = [axbig]
            colors_sub = [colors]
            x_data = [x]
        for xi_model, xi, xi_label, ax, colors, t in zip(
            x_model, x_data, x_sublabels, axs, colors_sub, title
        ):
            # len_data = len(xi[~np.isnan(xi)])
            len_data = np.max(
                [len(xij[~np.isnan(xij)]) for xij in np.swapaxes(xi, 0, 1)]
            )
            len_model = len(xi_model[~np.isnan(xi_model)])
            ax.hist(
                xi_model.flatten(),
                bins=bins,
                histtype="barstacked",
                fill=True,
                color=colors_violin_shade_dict[dataset_exp.value],
                align="left",
                weights=[np.ones(len_model) * len_data / len_model],
            )
            if len(xi_label) < 2:
                ax.hist(
                    xi,
                    bins=bins,
                    histtype="bar",
                    fill=True,
                    rwidth=0.4,
                    color=colors,
                    align="left",
                )
            else:
                ax.hist(
                    xi,
                    bins=bins,
                    histtype="bar",
                    stacked=False,
                    fill=True,
                    color=colors,
                    align="left",
                )

            pplt.text(
                x=0.5,
                y=1,
                s=t,
                ha="center",
                va="top",
                ax=ax,
                fontsize=8,
                transform=ax.transAxes,
            )
            for i, (z_c, c, lab) in enumerate(zip(xi_model, colors, label)):
                z_e = xi[:, i]
                z_e = z_e[~np.isnan(z_e)]
                ks = ks_2samp(z_e, z_c)
                if lab != "":
                    lab = f"{lab}, "
                pplt.text(
                    x=1,
                    y=1 - 0.1 * i,
                    s=f"{lab}$p = ${ks[1]:.2f}",
                    color=c,
                    ha="right",
                    va="top",
                    ax=ax,
                    fontsize=6,
                    transform=ax.transAxes,
                )
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
            ax.set_ylabel("count")

            ax.xaxis.set_minor_locator(
                MultipleLocator(np.diff(ax.get_xticks()).min() / 2)
            )
            ax.tick_params(
                axis="x",
                which="minor",
                direction="out",
                labelbottom=False,
                length=ax.xaxis.majorTicks[0].tick1line.get_markersize(),
            )
        axs[-1].set_xlabel("phase (min)")

    ax_lims = []
    for i, (ax, x) in enumerate(zip(axs_common, ["s", "sx", "dx", "dxx"])):
        data = convert_phases_to_dict(df_exp)[x]
        xmin = np.min(data)
        xmax = np.max(data)
        ax.hist(
            data,
            bins=np.arange(xmin / 5, xmax / 5 + 1) * 5,
            color=colors_violin_shade_dict[dataset_exp.value],
        )
        if i < 3:
            ax.set_xticklabels("")
        else:
            ax.set_xlabel("phase (min)")
            ax.set_ylabel("count")

        ax_lims.append(ax.get_xlim())
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))

        if x == "s":
            xlabel = r"S"
        elif x == "sx":
            xlabel = r"S$_{{i}}$"
        elif x == "dx":
            xlabel = r"D$_{{i}}$"
        else:
            xlabel = r"D$_{{ij}}$"
        pplt.text(
            x=0.75,
            y=1,
            s=xlabel,
            ha="center",
            va="top",
            ax=ax,
            fontsize=7,
            transform=ax.transAxes,
        )

    x_lim = [np.min(ax_lims), np.max(ax_lims)]
    for ax in axs_common:
        ax.set_xlim(x_lim)
        ax.xaxis.set_minor_locator(
            MultipleLocator(np.diff(ax.get_xticks()).min() / 2)
        )
        ax.tick_params(
            axis="x",
            which="minor",
            direction="out",
            labelbottom=False,
            length=ax.xaxis.majorTicks[0].tick1line.get_markersize(),
        )


def plot_corr_pooling(df_exp, grid, dataset_exp, n_bs=50000):
    """Plot kinship-correlation pooling analysis across subtree depths.

    Shows how sister, mother-daughter, etc. correlations evolve when
    pooling successive generations, with bootstrap confidence intervals.

    Parameters
    ----------
    df_exp : pd.DataFrame
        Experimental phase-duration DataFrame.
    grid : np.ndarray
        Grid of subplot axes.
    dataset_exp : DataSet
        Dataset enum for colour lookup.
    n_bs : int
        Number of bootstrap resamples.
    """
    df_normal = normal_transformation_df(df_exp)

    corr_colors = np.array(
        [
            "pplt:red",
            "pplt:green",
            colors_violin_dict[dataset_exp.value],
        ]
    )

    for ax in grid[:, :2].flatten():
        ax.remove()
    grid = grid[:, 2:]

    rng = np.random.default_rng()

    cols = ["d", "d", "d", "s", "dx", "dx", "dx", "s", "sx"]
    rows = ["s", "dx", "sx", "dx", "sy", "sx", "dy", "sx", "sy"]
    for i, (col, row, ax) in enumerate(zip(cols, rows, grid.flatten())):
        corrs = {}
        mean, low, high = np.zeros((3, len(SUBTREES)))
        for j, key in enumerate(SUBTREES):
            xy = get_xy_subtree(df_normal, subtree=key, row=row, col=col)
            if xy.shape[0] < 2:
                corrs[key] = np.nan
                mean[j] = np.nan
                low[j] = np.nan
                high[j] = np.nan
            else:
                xys = rng.choice(xy, (n_bs, xy.shape[0]))
                corrs[key] = pearsonr(xys[:, :, 0], xys[:, :, 1], axis=1)[0]
                mean[j] = pearsonr(xy[:, 0], xy[:, 1])[0]
                corr_bs = pearsonr(xys[:, :, 0], xys[:, :, 1], axis=1)[0]
                low[j] = np.nanpercentile(corr_bs, 2.5)
                high[j] = np.nanpercentile(corr_bs, 97.5)
        for k, (m, lo, h, c) in enumerate(zip(mean, low, high, corr_colors)):
            ax.errorbar(
                x=[m],
                y=[k],
                xerr=[[m - lo], [h - m]],
                marker="o",
                markersize=4,
                capsize=2,
                linestyle="none",
                color=c,
            )
        ax.fill_between(
            [low[-1], high[-1]],
            -2,
            6,
            facecolor=colors_violin_dict[dataset_exp.value],
            alpha=0.15,
            zorder=0,
        )
        pplt.text(
            x=0.05,
            y=0.95,
            s=f"{map_col_row(col)}$ - ${map_col_row(row)}",
            ha="left",
            va="top",
            ax=ax,
            fontsize=7,
            transform=ax.transAxes,
        )
        ax.vlines(0, -2, 6, color="pplt:gray", zorder=0, lw=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks(np.arange(len(SUBTREES)))
        if i % 3 == 0:
            ax.set_yticklabels(SUBTREES)
        else:
            ax.set_yticklabels("")
    for ax in grid[-1, :]:
        ax.set_xlabel("correlation\n")
    for ax in grid[:-1, :].flatten():
        ax.set_xticklabels("")
    for ax in grid.flatten():
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.tick_params(
            axis="x",
            which="minor",
            direction="out",
            labelbottom=False,
            length=ax.xaxis.majorTicks[0].tick1line.get_markersize(),
        )


def map_col_row(x):
    if x == "s":
        return r"S"
    elif x == "sx":
        return r"S$_{{a}}$"
    elif x == "sy":
        return r"S$_{{b}}$"
    elif x == "dx":
        return r"D$_{{a}}$"
    elif x == "dy":
        return r"D$_{{b}}$"
    elif x == "d":
        return r"D"
    return x


def get_xy_subtree(df_exp, subtree, row, col):
    df = df_exp.copy()
    df["nan"] = np.nan

    if subtree == "1":
        raw_clmns = np.array(["nan", "s", "d0", "d1", "s0", "s1"])
    elif subtree == "2":
        raw_clmns = np.array(["d0", "s0", "d00", "d01", "s00", "s01"])
    elif subtree == "3":
        raw_clmns = np.array(["d1", "s1", "d10", "d11", "s10", "s11"])
    elif subtree == "2+3":
        raw_clmns = np.array(
            [
                ["d0", "s0", "d00", "d01", "s00", "s01"],
                ["d1", "s1", "d10", "d11", "s10", "s11"],
            ]
        )
    elif subtree == "pooled":
        raw_clmns = np.array(
            [
                ["nan", "s", "d0", "d1", "s0", "s1"],
                ["d0", "s0", "d00", "d01", "s00", "s01"],
                ["d1", "s1", "d10", "d11", "s10", "s11"],
            ]
        )

    if row[:-1] != col[:-1]:
        if subtree == "1":
            raw_clmns = np.array(
                [raw_clmns, ["nan", "s", "d1", "d0", "s1", "s0"]]
            )
        elif subtree == "2":
            raw_clmns = np.array(
                [raw_clmns, ["d0", "s0", "d01", "d00", "s01", "s00"]]
            )
        elif subtree == "3":
            raw_clmns = np.array(
                [raw_clmns, ["d1", "s1", "d11", "d10", "s11", "s10"]]
            )
        elif subtree == "2+3":
            raw_clmns = np.array(
                [
                    ["d0", "s0", "d00", "d01", "s00", "s01"],
                    ["d0", "s0", "d01", "d00", "s01", "s00"],
                    ["d1", "s1", "d10", "d11", "s10", "s11"],
                    ["d1", "s1", "d11", "d10", "s11", "s10"],
                ]
            )
        elif subtree == "pooled":
            raw_clmns = np.array(
                [
                    ["nan", "s", "d0", "d1", "s0", "s1"],
                    ["nan", "s", "d1", "d0", "s1", "s0"],
                    ["d0", "s0", "d00", "d01", "s00", "s01"],
                    ["d0", "s0", "d01", "d00", "s01", "s00"],
                    ["d1", "s1", "d10", "d11", "s10", "s11"],
                    ["d1", "s1", "d11", "d10", "s11", "s10"],
                ]
            )

    if len(raw_clmns.shape) > 1:
        df_list = []
        for raw_clmn in raw_clmns:
            df_list.append(
                df[raw_clmn]
                .copy()
                .rename(
                    columns={
                        raw_clmn[i]: ROW_COL[i] for i in range(len(raw_clmn))
                    }
                )
            )
        df_x = pd.concat(df_list, ignore_index=True)
    else:
        df_x = (
            df[raw_clmns]
            .copy()
            .rename(
                columns={
                    raw_clmns[i]: ROW_COL[i] for i in range(len(raw_clmns))
                }
            )
        )

    return df_x[[row, col]].dropna().to_numpy()
