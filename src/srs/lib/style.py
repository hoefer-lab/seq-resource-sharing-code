# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Style settings — colour palettes, rcParams presets, and figure sizing.

Defines colour maps (sequential/parallel S-/D-phase), per-dataset colour
dictionaries, the shared figure-output directory, and helper functions
for Nature-style figure dimensions and *prettypyplot* initialisation.
"""

# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import prettypyplot as pplt
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap

from .enums import DataSet

colors = ["#058AB0", "#70C9E1", "#EE456D", "#FD9DB5", "#CBCDCF"]
colors_paulaner = ["#fec21f", "#ed6a0c", "#df0712", "#df017b", "#4a287d"]
FIGORDNER = str(Path(__file__).resolve().parent.parent / "fig") + "/"
ZETALIM = [-0.05, 1.05]
ETALIM = [-0.05, 1.15]

COLORS = [
    "#8cb369",
    "#caddbb",
    "#33658a",
    "#97bcd8",
    "#CBCDCF",
    "#94989C",
    "#D6D8D9",
    "#963756",
    "#af506f",
    "#c86888",
    "#DDA1B6",
    "#f4e285",
    "#f2882c",
    "#f6b379",
]


COLOR_SEQ = COLORS[2]
COLOR_SEQ_SHADE = COLORS[3]
COLOR_PAR = COLORS[0]
COLOR_PAR_SHADE = COLORS[1]
COLOR_D = COLORS[4]
COLOR_LIGHT = COLORS[5]
COLOR_LIGHT_SHADE = COLORS[6]
COLOR_S_100 = COLORS[7]
COLOR_S_75 = COLORS[8]
COLOR_S_50 = COLORS[9]
COLOR_S_0 = COLORS[10]

COLOR_DICT = {
    "seq": COLOR_SEQ,
    "seq_shade": COLOR_SEQ_SHADE,
    "par": COLOR_PAR,
    "par_shade": COLOR_PAR_SHADE,
    "d": COLOR_D,
    "s": COLOR_S_0,
    "s_50": COLOR_S_50,
    "s_75": COLOR_S_75,
    "s_100": COLOR_S_100,
    "model": COLORS[-1],
    "light": COLOR_LIGHT,
    "light_shade": COLOR_LIGHT_SHADE,
}

CMAP_S = LinearSegmentedColormap.from_list(
    "cmap_s", [COLOR_S_0, COLOR_S_100], N=20
)
CMAP_S_CLIPPED = LinearSegmentedColormap.from_list(
    "cmap_s_clipped", [COLOR_S_0, COLOR_S_0], N=2
)
CMAP_D = LinearSegmentedColormap.from_list(
    "cmap_d",
    ["#CBCDCF", "#CBCDCF"],
    N=2,
)

# ---------------------------------------------------------------------------
# Per-dataset colour dictionaries
# ---------------------------------------------------------------------------

COLOR_AIRY = "#A5668B"
COLOR_AIRY_SHADE = "#C398B1"

colors_violin_dict = {
    DataSet.KLAUS.value: COLOR_LIGHT,
    DataSet.AIRY.value: COLOR_AIRY,
    DataSet.MODEL1.value: COLOR_SEQ,
    DataSet.BARMODEL.value: COLORS[12],
    DataSet.RESMODEL.value: COLOR_SEQ,
    DataSet.RESPARMODEL.value: COLOR_PAR,
}

colors_violin_shade_dict = {
    DataSet.KLAUS.value: COLOR_D,
    DataSet.AIRY.value: COLOR_AIRY_SHADE,
    DataSet.MODEL1.value: COLOR_SEQ_SHADE,
    DataSet.BARMODEL.value: COLORS[13],
    DataSet.RESMODEL.value: COLOR_SEQ_SHADE,
    DataSet.RESPARMODEL.value: COLOR_PAR_SHADE,
}

COLOR_EXP_FAST = "#6B7280"
COLOR_EXP_FAST_SHADE = "#C4C8CF"
COLOR_EXP_SLOW = "#374151"
COLOR_EXP_SLOW_SHADE = "#8B9099"

COLOR_SIM_FAST = "#5B9BD5"
COLOR_SIM_FAST_SHADE = "#B4D4F0"
COLOR_SIM_SLOW = "#1E5A8A"
COLOR_SIM_SLOW_SHADE = "#6099BF"

colors_fast_slow_dict = {
    DataSet.KLAUS.value: {
        "fast": COLOR_EXP_FAST,
        "fast_shade": COLOR_EXP_FAST_SHADE,
        "slow": COLOR_EXP_SLOW,
        "slow_shade": COLOR_EXP_SLOW_SHADE,
    },
    DataSet.RESMODEL.value: {
        "fast": COLOR_SIM_FAST,
        "fast_shade": COLOR_SIM_FAST_SHADE,
        "slow": COLOR_SIM_SLOW,
        "slow_shade": COLOR_SIM_SLOW_SHADE,
    },
}

SCATTER_KWARGS = {"s": 10, "linewidths": 0.8, "alpha": 1, "marker": "x"}


def use_pplt_style():
    """Activate prettypyplot style with Arial font and publication-ready settings."""
    pplt.use_style(true_black=True)
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})
    rc("text", usetex="false")
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.it"] = "Arial:italic"
    plt.rcParams["mathtext.rm"] = "Arial"
    plt.rcParams["mathtext.default"] = "it"
    plt.rcParams["font.serif"] = "Arial"
    plt.rcParams["font.family"] = "Arial"
    fontsize = 8
    nice_fonts = {
        # # Use LaTeX to write all text
        # 'text.usetex': True,
        "pdf.fonttype": 42,
        "font.size": fontsize,
        # Make the legend/label fonts a little smaller
        "axes.labelsize": fontsize - 0,
        "legend.fontsize": fontsize - 2,
        "xtick.labelsize": fontsize - 2,
        "ytick.labelsize": fontsize - 2,
    }
    plt.rcParams.update(nice_fonts)


def set_size(width, fraction=1, subplot=np.array([1, 1])):
    """Compute figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width : float or str
        Width in points, or a preset name (``'nature_wide'``, ``'pnas'``, etc.).
    fraction : float
        Fraction of the width to use.
    subplot : array-like of shape (2,)
        ``[n_rows, n_cols]`` — adjusts height by the row/column ratio.

    Returns
    -------
    tuple[float, float]
        ``(width_inches, height_inches)``.
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "thesiscolumn":
        width_pt = 135.2125
    elif width == "thesiscolumnbroad":
        width_pt = 303.3069
    elif width == "beamer":
        width_pt = 307.28987
    elif width == "pnas":
        width_pt = 246.09686
    elif width == "pnasbothcolumn":
        width_pt = 505.69374
    elif width == "poster_sixths":
        width_pt = 340.157
    elif width == "nature_wide":
        width_pt = 510.24
    elif width == "naturecolumn":
        width_pt = 250.12
    else:
        width_pt = width
    # Width of figure
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplot[0] / subplot[1])

    fig_dim = (fig_width_in, 1.2 * fig_height_in)

    return fig_dim


def get_colors_paulaner(n):
    """Return *n* colours from the Paulaner gradient palette."""
    if n < len(colors_paulaner):
        return colors_paulaner[:n]
    else:
        cmap = LinearSegmentedColormap.from_list("", colors_paulaner)
        return [cmap(i) for i in np.linspace(0, 0.99, n)]
