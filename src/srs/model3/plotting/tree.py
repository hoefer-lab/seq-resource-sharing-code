# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Replication-tree visualisation.

Renders a single branching-process realisation as a heatmap where the
x-axis is time and each row represents a nucleus.  S-phase colour
encodes the activated fork-fraction *f*, while the D-phase backbone is
shown in a separate colour map.

The main entry points are :func:`plot_sharing_mode` (generates a
full figure comparing sequential vs. parallel sharing) and
:func:`plot_single_tree` (draws one tree into a given Axes).
"""

# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import matplotlib.pyplot as plt
import numpy as np
import prettypyplot as pplt

from srs.lib.enums import ResourceType
from srs.lib.style import (
    CMAP_D,
    CMAP_S,
    CMAP_S_CLIPPED,
    COLOR_PAR,
    COLOR_SEQ,
    FIGORDNER,
    set_size,
    use_pplt_style,
)

from .._core.wrapper import simulate_time_wrapper
from ..default_params import F_THRESHOLD


def plot_sharing_mode(nstop=8, zeta=1, cv_d=0.02):
    """Generate a figure comparing sequential vs. parallel resource sharing.

    Simulates one branching process for each extreme sharing mode
    (high vs. low *k_u*) and plots the resulting replication trees
    side by side, with optional clipped views.

    Parameters
    ----------
    nstop : int, default=8
        Number of nuclei at which each simulation stops.
    zeta : float, default=1
        Scarcity factor.
    cv_d : float, default=0.02
        Standard deviation of D-phase duration.

    Returns
    -------
    results : list of dict
        Simulation result dictionaries for the two sharing modes.
    """

    k_b = 1e6
    kus = np.array([1e2, 1e-2])
    resource_type = ResourceType.CONST
    colors = [COLOR_PAR, COLOR_SEQ]

    results = []
    xmax = 0
    for i in range(len(kus)):  # [1/s]
        res = simulate_time_wrapper(
            n_sample=1,
            resource_type=resource_type,
            zeta=zeta,
            n_stop=nstop,
            k_u=kus[i],  # to convert from sec tfo min
            std=cv_d,
            k_b=k_b,
            tau=0.3,
            save_f=True,
        )[0]
        results.append(res)
        xmax = np.max([xmax, res["time"].max()])

    use_pplt_style()
    savename = "2024_paper_sharing_mode"
    figsize = np.array(set_size("nature_wide")) / 4

    for clipped in [False, True]:
        fig, axs = plt.subplots(
            2,
            1,
            sharex=True,
            sharey=True,
            figsize=figsize,
            gridspec_kw={"hspace": 0.6, "wspace": 0},
        )

        fig2, axs2 = plt.subplots(
            2,
            1,
            sharex=True,
            sharey=True,
            figsize=figsize,
            gridspec_kw={"hspace": 0.6, "wspace": 0},
        )
        for i, (res, ax, ax2) in enumerate(zip(results, axs, axs2)):
            plot_single_tree(
                ax=ax,
                ax2=ax2,
                res=res,
                clip=clipped,
                linecolor=colors[i],
                show_cbar=True,
            )

        ax.set_xlim([-0.05 * xmax, 1.05 * xmax])
        ax.set_ylim(-0.1, 2.1)
        ax.set_xlabel("time $t$ (generation)")

        ax2.set_xlim([-0.05 * xmax, 1.05 * xmax])
        ax2.set_xlabel("time $t$ (generation)")
        ax2.set_ylabel("# replicating nuclei")

        if clipped:
            s = savename + "_clipped"
            fig2.savefig(FIGORDNER + s + "_n.pdf")
        else:
            s = savename
        fig.savefig(FIGORDNER + s + ".pdf")

    return results


def plot_single_tree(
    ax,
    res,
    linecolor=None,
    ax2=None,
    show_cbar=False,
    clip=False,
    truncate_x=None,
    truncate_gen=None,
):
    """Render a single replication tree as a heatmap on *ax*.

    Each row of the heatmap is a nucleus; colour encodes the activated
    fork-fraction *f* during S-phase, while D-phase periods are shown
    in a separate colour map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes for the tree heatmap.
    res : dict
        Simulation result dictionary from
        :func:`~model3.core.simulation.simulate_time`.
    linecolor : colour-like, optional
        Colour for the replicating-nuclei count line on *ax2*.
    ax2 : matplotlib.axes.Axes, optional
        If given, plot the number of simultaneously replicating nuclei.
    show_cbar : bool, default=False
        Whether to add a colour bar.
    clip : bool, default=False
        If True, binarise *f* using :data:`F_THRESHOLD` (on/off view).
    truncate_x : float, optional
        If given, crop the tree at this time value.
    truncate_gen : int, optional
        Unused; reserved for future generation-based truncation.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    t, f, names = res["time"], np.moveaxis(res["f"], 0, 1), res["names"]
    s_phases = res["s_phases"]
    # convert names to binary names (1 -> 10, 11; 10 -> 100, 101; ...)
    names = np.array([int(bin(name)[2:]) for name in names])

    if clip:
        f[f < F_THRESHOLD] = 0
        f[f >= F_THRESHOLD] = 1
        cmap = CMAP_S_CLIPPED
    else:
        cmap = CMAP_S

    y = np.array([convert_name_to_y_value(name) for name in names])
    y -= y.min() / 2
    x = t

    # sort y and f according to the y value
    idx_sort = y.argsort()
    y = y[idx_sort]
    f = f[idx_sort]
    names = names[idx_sort]
    s_phases = s_phases[idx_sort]

    # plot d_phases
    f_backbone = get_f_backbone(names=names, f=f, s_phases=s_phases, t=t)

    if truncate_x:
        mask = x <= truncate_x
        x, f_backbone, f = x[mask], f_backbone[:, mask], f[:, mask]

    im = ax.pcolormesh(
        x,
        y,
        f_backbone,
        vmin=0,
        vmax=1,
        cmap=CMAP_D,
        edgecolors="face",
        linewidth=0.1,
    )

    # plot s-phases
    f[f == 0] = np.nan
    im = ax.pcolormesh(
        x, y, f, vmin=0, vmax=1, cmap=cmap, edgecolors="face", linewidth=0.1
    )

    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    if show_cbar:
        cbar = pplt.colorbar(
            im,
            label="replication speed \n"
            r"active fork-fraction $f$",
            position="right",
            pad="5%",
        )
        # set the color of the lines
        cbar.solids.set_edgecolor("face")

    if ax2 is not None:
        # plot 'No. nuclei in S-phase'
        # apply threshold
        f[f < F_THRESHOLD] = 0
        f[f >= F_THRESHOLD] = 1
        n = np.nansum(f, axis=0)
        if linecolor is not None:
            ax2.plot(t, n, color=linecolor)
        else:
            ax2.plot(t, n)
        ax2.spines[["right", "top"]].set_visible(False)
    return ax


def convert_name_to_y_value(name):
    """Convert a binary name string to a vertical position for the tree plot.

    Each bit after the leading '1' shifts the y-value up or down by a
    decreasing amount, producing a balanced binary-tree layout.

    Parameters
    ----------
    name : int
        Binary-encoded name as a base-10 integer (e.g. 101 → binary
        expansion ``1-0-1``).

    Returns
    -------
    y_value : float
        Vertical coordinate in the range roughly [0, 2].
    """
    y_value = 1
    digits = [int(d) for d in f"{name}"[1:]]
    for i, d in enumerate(digits):
        y_value += (2 * d - 1) / 2 ** (i + 1)
    return y_value


def get_f_backbone(names, f, t, s_phases):
    """Build a heatmap mask for D-phase backbone intervals.

    For each nucleus the D-phase spans from birth (end of mother's
    S-phase) to the start of the nucleus's own S-phase.  The returned
    array has ``1`` during D-phase and ``NaN`` elsewhere, suitable for
    overlay plotting with :func:`matplotlib.axes.Axes.pcolormesh`.

    Parameters
    ----------
    names : ndarray of int
        Binary-encoded nucleus names.
    f : ndarray, shape (n_nuclei, n_time)
        Per-nucleus activated fractions.
    t : ndarray, shape (n_time,)
        Simulation timeline.
    s_phases : ndarray, shape (n_nuclei, 2)
        Start/end times of each nucleus's S-phase.

    Returns
    -------
    f_backbone : ndarray, shape (n_nuclei, n_time)
        D-phase mask (1 during D-phase, NaN otherwise).
    """
    f_backbone = np.zeros_like(f)
    for i, (f_row, name) in enumerate(zip(f, names)):
        if name == 1:
            tob = 0  # time of birth
        else:
            mask_mother = int(f"{name}"[:-1]) == names
            # time of birth
            tob = s_phases[mask_mother][0, 1]
        # time of starting s-phase
        # tos = s_phases[name == names][0, 0]
        # as in the clipped case, this will not coincide with entering s-phase
        # we use f > 0 instead
        tos = t[np.argmax(f[i] > 0)]

        mask_d_phase = (t <= tos) & (t >= tob)
        f_backbone[i][mask_d_phase] = 1

    f_backbone[f_backbone == 0] = np.nan
    return f_backbone
