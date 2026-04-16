# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Paper Figures 4 and 5 — growth dynamics and resource utilisation.

This module assembles the multi-panel figures that show:

* **Fig. 4** — DNA-content dynamics, growth rate and resource
  utilisation as functions of scarcity :math:`\\zeta`, plus
  collapsed scaling laws.
* **Fig. 5** — Growth advantage of sequential over parallel sharing,
  intermediate-regime scan, and three-mode tree examples.

The main entry point is :func:`create_fig45`.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import prettypyplot as pplt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from srs.lib.enums import ResourceType
from srs.lib.style import (
    COLOR_DICT,
    COLORS,
    ETALIM,
    FIGORDNER,
    ZETALIM,
    get_colors_paulaner,
    set_size,
    use_pplt_style,
)

from .._core.wrapper import simulate_time_wrapper
from .._read_data import (
    FILENAME_DEFAULT,
    FILENAME_SCAN_REGIME,
    read_data,
)
from .._tools import get_sliding_mean, get_t_max
from .tree import plot_single_tree

scattersize = 4**2


def power2ticks(x, pos):
    """Tick formatter that displays values as powers of 2."""
    if x in [1, 2]:
        return f"${x}$"
    else:
        return rf"$2^{int(np.log2(x))}$"


def fakepower2ticks(x, pos):
    """Tick formatter that maps linear values to :math:`2^x` labels."""
    return rf"${2**x}$"


def set_power2ticks(ax):
    """Set a log-2 y-axis with power-of-two tick labels on *ax*."""
    y_max = ax.get_ylim()[1]
    ax.set_yscale("log")
    ax.set_yticks(2 ** np.arange(0, int(np.log2(y_max)) + 1)[::2])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(power2ticks))
    ax.minorticks_off()
    for label in ax.get_yticklabels():
        label.set_ha("left")
    ax.tick_params(axis="y", which="major", pad=10)


def set_fakepower2ticks(ax):
    """Set a linear y-axis whose labels read :math:`2^0, 2^1, 2^2` on *ax*."""
    ax.set_yticks(np.arange(0, 3))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fakepower2ticks))
    for label in ax.get_yticklabels():
        label.set_ha("left")
    ax.tick_params(axis="y", which="major", pad=10)
    ax.set_ylim([None, 2.6])


def fig4bd(
    axs,
    n_gen,
    tau=0.3,
    f_threshold=0.15,
    cv_d=0.1,
    k_b=1e6,
    n_sample=480,
    kus=np.array([1e-2, 1e2]),
    resource_type=ResourceType.GROWING_G,
):
    """Create Fig. 4b (DNA content) and Fig. 4d (utilisation) plus tree insets.

    Runs simulations for a sequential (*low k_u*) and a parallel (*high k_u*)
    sharing mode and fills three axes:

    * ``axs[0]`` — DNA-content dynamics (Fig. 4b).
    * ``axs[1]`` — Resource utilisation dynamics (Fig. 4d).
    * ``axs[2]`` — Replication-tree examples (split into two sub-axes).

    Parameters
    ----------
    axs : list of matplotlib.axes.Axes
        Three axes to draw into.
    n_gen : int
        Number of generations to simulate.
    tau : float, default=0.3
        S-phase fraction of the cell cycle.
    f_threshold : float, default=0.15
        Activated-fraction threshold for the s*-phase.
    cv_d : float, default=0.1
        Standard deviation of the D-phase duration.
    k_b : float, default=1e6
        Binding rate.
    n_sample : int, default=480
        Number of independent realisations per sharing mode.
    kus : ndarray, default=[1e-2, 1e2]
        Unbinding rates for sequential and parallel modes.
    resource_type : ResourceType
        Resource-growth model.

    Returns
    -------
    axs : list of matplotlib.axes.Axes
    """
    zeta = 0.8 * np.log(2) * tau
    results = []

    for k_u in kus:
        res = simulate_time_wrapper(
            n_sample=n_sample,
            resource_type=resource_type,
            zeta=zeta,
            k_u=k_u,
            k_b=k_b,
            std=cv_d,
            n_gen=n_gen,
            rho=1 / tau,
            f_threshold=f_threshold,
        )
        results.append(res)

    plot_fig4b(ax=axs[0], results=results)
    plot_fig4d(ax=axs[1], results=results)

    # plot trees
    results = []
    for k_u in kus[::-1]:
        res = simulate_time_wrapper(
            n_sample=1,
            resource_type=resource_type,
            zeta=zeta,
            k_u=k_u,
            k_b=k_b,
            std=cv_d,
            n_stop=32,
            save_f=True,
            rho=1 / tau,
            f_threshold=f_threshold,
        )
        results.append(res)
    truncate_x = 5
    ax = axs[2]
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size="100%", pad=0)
    for a in [ax, ax2]:
        a.set_ylim(-0.5, 2.5)
        a.set_xlim(-0.05 * truncate_x, 1.05 * truncate_x)

    ax.set_axis_off()
    ax2.set_xlabel(r"time $t$ (generation)")

    plot_single_tree(
        ax=ax, res=results[0][0], truncate_x=truncate_x, truncate_gen=4
    )
    plot_single_tree(
        ax=ax2, res=results[1][0], truncate_x=truncate_x, truncate_gen=4
    )

    return axs


def plot_fig4b(ax, results):
    """Plot DNA-content dynamics (Fig. 4b).

    Individual traces are shown as faint lines; a sliding-mean curve
    is overlaid for each sharing mode.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot into.
    results : list of list of dict
        Simulation results for each sharing mode.
    """

    t_max = get_t_max(results)
    color_shades = [COLOR_DICT["seq_shade"], COLOR_DICT["par_shade"]]
    colors = [COLOR_DICT["seq"], COLOR_DICT["par"]]
    for res, color, color_shade in zip(results, colors, color_shades):
        timepoints, counts = np.array([]), np.array([])
        for rs in res:
            mask = rs["time"] <= t_max
            pplt.plot(
                rs["time"][mask],
                rs["g"][mask],
                color=color_shade,
                alpha=5 / 480,
                zorder=0,
                ax=ax,
            )
            timepoints = np.append(timepoints, rs["time"])
            counts = np.append(counts, rs["g"])
        time, n_mean = get_sliding_mean(x=timepoints, y=counts, x_max=t_max)
        (line,) = pplt.plot(time, n_mean, color=color, zorder=2, ax=ax)

    set_power2ticks(ax)

    ax.set_xlabel(r"time $t$ (generation)")
    ax.set_ylabel(r"total DNA content $(g)$")


def plot_fig4d(ax, results):
    """Plot Fig. 4d."""
    t_max = get_t_max(results)
    color_shades = [COLOR_DICT["seq_shade"], COLOR_DICT["par_shade"]]
    colors = [COLOR_DICT["seq"], COLOR_DICT["par"]]
    for i, (res, color, color_shade) in enumerate(
        zip(results, colors, color_shades)
    ):
        timepoints, etas = np.array([]), np.array([])
        for rs in res:
            pplt.plot(
                rs["time"][rs["time"] <= t_max],
                1.5 * i + rs["eta"][rs["time"] <= t_max],
                color=color_shade,
                alpha=10 / 196,
                zorder=0,
                ax=ax,
            )
            timepoints = np.append(timepoints, rs["time"])
            etas = np.append(etas, rs["eta"])
        time, eta_mean = get_sliding_mean(x=timepoints, y=etas, x_max=t_max)
        pplt.plot(time, 1.5 * i + eta_mean, color=color, zorder=2, ax=ax)

    ax.set_xlabel(r"time $t$ (generation)")
    ax.set_ylim([-0.25, 2.75])
    ax.set_yticks([0, 1, 1.5, 2.5], ["0", "1", "0", "1"])
    ax.set_ylabel("resource utilization \n" + r"level $\eta$")


def fig5_middle_bd(
    axs,
    n_gen,
    tau=0.3,
    f_threshold=0.15,
    cv_d=0.1,
    k_b=1e6,
    n_sample=200,
    kus=np.array([1e-2, 3 / 0.3, 1e2]),
    resource_type=ResourceType.GROWING_G,
):
    """Create panels for the intermediate-regime figure (Fig. 5 middle).

    Compares three sharing modes (sequential, intermediate, parallel)
    with tree examples and resource-utilisation traces.

    Parameters
    ----------
    axs : list of matplotlib.axes.Axes
        Two axes: ``axs[0]`` for tree examples, ``axs[1]`` for utilisation.
    n_gen : int
    tau : float, default=0.3
    f_threshold : float, default=0.15
    cv_d : float, default=0.1
    k_b : float, default=1e6
    n_sample : int, default=200
    kus : ndarray
        Three unbinding rates (sequential, intermediate, parallel).
    resource_type : ResourceType

    Returns
    -------
    axs : list of matplotlib.axes.Axes
    """
    zeta = 0.8 * np.log(2) * tau
    results = []

    for k_u in kus:
        res = simulate_time_wrapper(
            n_sample=n_sample,
            resource_type=resource_type,
            zeta=zeta,
            k_u=k_u,
            k_b=k_b,
            std=cv_d,
            n_gen=n_gen,
            rho=1 / tau,
            f_threshold=f_threshold,
        )
        results.append(res)

    colors_3 = [COLOR_DICT["seq"], COLORS[12], COLOR_DICT["par"]]
    color_shades_3 = [
        COLOR_DICT["seq_shade"],
        COLORS[13],
        COLOR_DICT["par_shade"],
    ]

    # --- tree examples for 3 ku values (replaces DNA content panel) ---
    tree_results = []
    for k_u in kus[::-1]:
        res = simulate_time_wrapper(
            n_sample=1,
            resource_type=resource_type,
            zeta=zeta,
            k_u=k_u,
            k_b=k_b,
            std=cv_d,
            n_stop=32,
            save_f=True,
            rho=1 / tau,
            f_threshold=f_threshold,
        )
        tree_results.append(res)

    truncate_x = 5
    ax_tree = axs[0]
    divider = make_axes_locatable(ax_tree)
    ax_tree2 = divider.append_axes("bottom", size="100%", pad=0)
    ax_tree3 = divider.append_axes("bottom", size="100%", pad=0)
    for a in [ax_tree, ax_tree2, ax_tree3]:
        a.set_ylim(-0.5, 2.5)
        a.set_xlim(-0.05 * truncate_x, 1.05 * truncate_x)

    ax_tree.set_axis_off()
    ax_tree2.set_axis_off()
    ax_tree3.set_xlabel(r"time $t$ (generation)")

    plot_single_tree(
        ax=ax_tree,
        res=tree_results[0][0],
        truncate_x=truncate_x,
        truncate_gen=4,
    )
    plot_single_tree(
        ax=ax_tree2,
        res=tree_results[1][0],
        truncate_x=truncate_x,
        truncate_gen=4,
    )
    plot_single_tree(
        ax=ax_tree3,
        res=tree_results[2][0],
        truncate_x=truncate_x,
        truncate_gen=4,
    )

    plot_fig5_middle_d(
        ax=axs[1], results=results, colors=colors_3, color_shades=color_shades_3
    )

    return axs


def plot_fig5_middle_b(ax, results, colors, color_shades):
    """Plot DNA-content dynamics for three *k_u* values (intermediate regime).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    results : list of list of dict
    colors, color_shades : list of colour-like
    """
    t_max = get_t_max(results)
    for res, color, color_shade in zip(results, colors, color_shades):
        timepoints, counts = np.array([]), np.array([])
        for rs in res:
            mask = rs["time"] <= t_max
            pplt.plot(
                rs["time"][mask],
                rs["g"][mask],
                color=color_shade,
                alpha=5 / 480,
                zorder=0,
                ax=ax,
            )
            timepoints = np.append(timepoints, rs["time"])
            counts = np.append(counts, rs["g"])
        time, n_mean = get_sliding_mean(x=timepoints, y=counts, x_max=t_max)
        pplt.plot(time, n_mean, color=color, zorder=2, ax=ax)

    set_power2ticks(ax)
    ax.set_xlabel(r"time $t$ (generation)")
    ax.set_ylabel(r"total DNA content $(g)$")


def plot_fig5_middle_d(ax, results, colors, color_shades):
    """Plot resource utilisation for three *k_u* values (intermediate regime).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    results : list of list of dict
    colors, color_shades : list of colour-like
    """
    t_max = get_t_max(results)
    n = len(results)
    for i, (res, color, color_shade) in enumerate(
        zip(results, colors, color_shades)
    ):
        timepoints, etas = np.array([]), np.array([])
        for rs in res:
            pplt.plot(
                rs["time"][rs["time"] <= t_max],
                1.5 * i + rs["eta"][rs["time"] <= t_max],
                color=color_shade,
                alpha=10 / 196,
                zorder=0,
                ax=ax,
            )
            timepoints = np.append(timepoints, rs["time"])
            etas = np.append(etas, rs["eta"])
        time, eta_mean = get_sliding_mean(x=timepoints, y=etas, x_max=t_max)
        pplt.plot(time, 1.5 * i + eta_mean, color=color, zorder=2, ax=ax)

    ax.set_xlabel(r"time $t$ (generation)")
    ytick_pos = [1.5 * i + v for i in range(n) for v in [0, 1]]
    ytick_lab = ["0", "1"] * n
    ax.set_ylim([-0.25, 1.5 * (n - 1) + 1.25])
    ax.set_yticks(ytick_pos, ytick_lab)
    ax.set_ylabel("resource utilization \n" + r"level $\eta$")


def fig4ce(ax, data, y_name, ylabel, ylim=None):
    """Plot a quantity vs. resource availability :math:`\\zeta` (Fig. 4c/e style).

    Lines are coloured by :math:`\\tau` and shaded error bands show
    \u00b1 1 std.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : dict
        Pre-computed scan data from :func:`~model3.read_data.read_data`.
    y_name : str
        Key prefix in *data* (e.g. ``'lambda'``, ``'eta'``).
    ylabel : str
        Axes label.
    ylim : tuple of float, optional
        y-axis limits.
    """
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"resource availability $\zeta$")
    ax.set_xlim(ZETALIM)
    if ylim is not None:
        ax.set_ylim(ylim)

    zeta = data["zeta"]
    tau = data["tau"]
    # get all values of chi
    tau_int = np.unique(tau)
    colors_paulaner = get_colors_paulaner(len(tau_int))

    for tau_0, color in zip(tau_int, colors_paulaner):
        mask = tau == tau_0

        x = zeta[mask]
        # sort with respect to x
        order = x.argsort()
        x = x[order]
        if y_name + "_seq" in data:
            y_seq = data[y_name + "_seq"]["mean"][:][mask][order]
            y_seq_std = data[y_name + "_seq"]["std"][:][mask][order]
            y_par = data[y_name + "_par"]["mean"][:][mask][order]
            y_par_std = data[y_name + "_par"]["std"][:][mask][order]
            y = (y_seq + y_par) / 2
            y_std = np.sqrt(
                y_seq_std**2 + y_par_std**2 + 0.25 * (y_seq - y_par) ** 2
            )
        else:
            y = data[y_name]["mean"][:][mask][order]
            y_std = data[y_name]["std"][:][mask][order]

            # convert growth advantage to log2 base
            if y_name == "growth_advantage_g":
                y /= np.log(2)
                y_std /= np.log(2)

        (line,) = ax.plot(x, y, color=color, zorder=2)
        ax.fill_between(
            x, y - y_std, y + y_std, facecolor=line.get_color(), alpha=0.25
        )

    if y_name == "growth_advantage_g":
        set_fakepower2ticks(ax)
    return ax


def fig5d_add_model_prediction(ax, data):
    """Overlay analytic model prediction on the growth-advantage panel (Fig. 5d).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : dict
        Pre-computed scan data.
    """
    zeta = data["zeta"]
    tau = data["tau"]
    # get all values of chi
    tau_int = np.unique(tau)
    colors_paulaner = get_colors_paulaner(len(tau_int))
    y_name = "utilization_advantage"

    zeta_selection = 1 - np.arange(20) / 20
    for tau_0, color in zip(tau_int, colors_paulaner):
        mask = (tau == tau_0) & np.array(
            [np.round(z, 3) in np.round(zeta_selection, 2) for z in zeta]
        )

        x = zeta[mask]
        utilization_advantage = data[y_name]["mean"][:][mask]
        growth_advantage = zeta[mask] / tau[mask] * utilization_advantage

        SCATTER_KWARGS = {"s": 12, "linewidths": 1, "marker": "o"}
        ax.scatter(
            x,
            growth_advantage / np.log(2),
            edgecolor=color,
            facecolor="none",
            zorder=10,
            **SCATTER_KWARGS,
        )
    return ax


def fig4f(ax, data, ylabel):
    """Plot steady growth rate :math:`\\lambda` vs. rescaled resource (Fig. 4f).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : dict
    ylabel : str
    """
    lamb = np.concatenate(
        [data["lambda_seq"]["mean"][:], data["lambda_par"]["mean"][:]]
    )
    zeta = np.concatenate([data["zeta"], data["zeta"]])
    tau = np.concatenate([data["tau"], data["tau"]])

    x = zeta / tau
    y = lamb
    ax.scatter(
        x,
        y,
        marker="x",
        color="pplt:gray",
        s=scattersize,
    )
    # add model prediction
    x_space = np.linspace(0, 5, 1000)

    def model(x):
        return np.where(x > np.log(2), np.log(2), x)

    (line,) = ax.plot(x_space, model(x_space), color=COLOR_DICT["model"])

    ax.vlines(
        np.log(2), 0, 1.1, color="pplt:lightgray", lw=line.get_linewidth() / 2
    )
    xlim = ax.get_xlim()
    ax.set_xticks(list(ax.get_xticks()) + [np.log(2)])
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i, label in enumerate(labels):
        if label == "0.693":
            labels[i] = r"$\zeta_c$"
        else:
            labels[i] = label.split(".")[0]
    ax.set_xticklabels(labels)
    ax.set_xlim(xlim)

    ax.set_xlabel(r"rescaled resource availability $\zeta / \tau_S$")
    ax.set_ylabel(ylabel)
    ax.set_ylim([0, 0.9])
    return ax


def fig4g(ax, data, ylim):
    """Plot steady resource utilisation :math:`\\eta` vs. rescaled resource (Fig. 4g).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : dict
    ylim : tuple of float
    """
    eta = np.concatenate(
        [data["eta_seq"]["mean"][:], data["eta_par"]["mean"][:]]
    )
    zeta = np.concatenate([data["zeta"], data["zeta"]])
    tau = np.concatenate([data["tau"], data["tau"]])

    x = zeta / tau
    y = eta
    # Add all data to fig4c and fig4d
    ax.scatter(x, y, marker="x", color="pplt:gray", s=scattersize, zorder=0)

    # add model prediction
    x_space = np.linspace(0, 5, 1000)

    def model(x):
        return np.where(
            x > np.log(2),
            np.divide(np.log(2), x, where=x > 0, out=np.ones_like(x)),
            1,
        )

    (line,) = ax.plot(x_space, model(x_space), color=COLOR_DICT["model"])

    ax.vlines(
        np.log(2), 0, 1.1, color="pplt:lightgray", lw=line.get_linewidth() / 2
    )
    xlim = ax.get_xlim()
    ax.set_xticks(list(ax.get_xticks()) + [np.log(2)])
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i, label in enumerate(labels):
        if label == "0.693":
            labels[i] = r"$\zeta_c$"
        else:
            labels[i] = label.split(".")[0]
    ax.set_xticklabels(labels)
    ax.set_xlim(xlim)

    ax.set_ylabel("steady resource \n" + r"utilization $\eta$")
    ax.set_xlabel(r"rescaled resource availability $\zeta / \tau_S$")
    ax.set_ylim(ylim)
    return ax


def fig5_middle_scan_regime(ax, data):
    """Plot maximal growth advantage vs. intermediate-regime unbinding rate.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : dict
        Scan-regime data from :func:`~model3.read_data.read_data`.
    """
    kus = np.array(data["kus"].tolist())
    ku_low = kus[:, 0]
    y = data["growth_advantage_g"]["mean"] / np.log(2)
    y_std = data["growth_advantage_g"]["std"] / np.log(2)

    order = ku_low.argsort()
    ku_low = ku_low[order]
    y = y[order]
    y_std = y_std[order]

    (line,) = ax.plot(ku_low, y, color="pplt:gray", zorder=2)
    ax.fill_between(
        ku_low, y - y_std, y + y_std, facecolor=line.get_color(), alpha=0.25
    )
    ylim = ax.get_ylim()
    rho = 1 / data["tau"][0]
    ax.vlines(
        rho,
        ylim[0],
        ylim[1],
        color="pplt:lightgray",
        lw=line.get_linewidth() / 2,
    )
    ax.set_xscale("log")
    ax.set_xlabel("intermediate regime \n" + r"unbinding rate $k_u$")
    ax.set_ylabel("maximal growth \n" + r"advantage $g_{interm} / g_{par}$")
    set_fakepower2ticks(ax)
    return ax


def plot_fig45(n_gen=8, filename=FILENAME_DEFAULT):
    """Assemble and save paper Figures 4 and 5.

    Parameters
    ----------
    n_gen : int, default=8
        Number of generations to simulate.
    filename : str, default=FILENAME_DEFAULT
        Path to the data file.

    Reads pre-computed parameter-scan data and generates PDF files:
    ``fig4.pdf``, ``fig5.pdf``, ``figs3.pdf``.

    """
    use_pplt_style()
    figsize = np.array(set_size("nature_wide", subplot=[2, 3]))
    fig4, grid4 = plt.subplots(
        nrows=2, ncols=3, figsize=figsize, constrained_layout=True
    )
    figs3, grids3 = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=figsize * np.array([1, 0.66]),
        constrained_layout=True,
    )
    # read data
    data = read_data(filename=filename)

    fig5, grid5 = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=figsize * np.array([1, 0.5]),
        constrained_layout=True,
    )
    fig4bd(axs=[grid4[0, 0], grid5[1], grid5[0]], n_gen=n_gen)
    fig4ce(
        ax=grid4[0, 1],
        data=data,
        y_name="lambda",
        ylabel=r"steady growth rate $\lambda$",
        ylim=[0, 0.9],
    )
    fig4ce(
        ax=grid4[0, 2],
        data=data,
        y_name="eta",
        ylabel="steady resource \n " + r"utilization $\eta$",
        ylim=ETALIM,
    )
    fig4f(
        ax=grid4[1, 1],
        data=data,
        ylabel=r"steady growth rate $\lambda$",
    )
    fig4g(ax=grid4[1, 0], data=data, ylim=ETALIM)
    grid4[1, 2].set_axis_off()

    fig4.get_layout_engine().set(
        w_pad=6 / 72, h_pad=6 / 72, hspace=0.2, wspace=0.15
    )
    fig4.savefig(FIGORDNER + "fig4.pdf")

    fig4ce(
        ax=grid5[2],
        data=data,
        y_name="growth_advantage_g",
        ylabel="eventual growth \n " + r"advantage $g_{seq} / g_{par}$",
    )
    fig5d_add_model_prediction(ax=grid5[2], data=data)
    fig5.get_layout_engine().set(
        w_pad=6 / 72, h_pad=6 / 72, hspace=0.2, wspace=0.15
    )
    fig5.savefig(FIGORDNER + "fig5.pdf")

    # --- figs3: tree examples, utilization, and growth advantage (seq vs intermediate) ---
    fig5_middle_bd(axs=[grids3[0], grids3[1]], n_gen=n_gen)
    data_scan = read_data(filename=FILENAME_SCAN_REGIME)
    fig5_middle_scan_regime(ax=grids3[2], data=data_scan)
    figs3.get_layout_engine().set(
        w_pad=6 / 72, h_pad=6 / 72, hspace=0.2, wspace=0.15
    )
    figs3.savefig(FIGORDNER + "figs3.pdf")
