# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Paper Figure 2 — correlated branching process (models 1 and 2).

Compares experimental replication-timing data with simulated output from
the correlated branching-process model.  Supports both the full
kinship-correlated model (model 2, ``DataSet.BARMODEL``) and the
independent null model (model 1, ``DataSet.MODEL1``).

The main entry point is :func:`plot_fig23`.
"""

import matplotlib.pyplot as plt
import numpy as np

from srs.lib.correlation_matrix import get_model1_corr, get_model2_corr
from srs.lib.data_loader import convert_to_df, get_data
from srs.lib.enums import DataSet, DropData, StoppingCriterion
from srs.lib.plotting import (
    plot_kinship_correlation,
    plot_nuclei_count,
    plot_phase_identity,
    plot_sphase_vs_delay,
)
from srs.lib.style import FIGORDNER, set_size, use_pplt_style

from .._core.wrapper import simulation_sd_wrapper
from .statistics import plot_corr_pooling, plot_phases_share_dist


def get_sim_data(
    dataset,
    df_exp,
    n=30000,
    stoppingcriterion=StoppingCriterion.COUNTER,
    n_max=24,
):
    """Run branching process simulation and return simulated data.

    Supports ``DataSet.BARMODEL`` (model 2 — full kinship correlations)
    and ``DataSet.MODEL1`` (model 1 — independent; only D-sister
    correlation retained).
    """
    assert dataset.value in (DataSet.BARMODEL.value, DataSet.MODEL1.value), (
        "Invalid sim dataset: Only DataSet.BARMODEL and DataSet.MODEL1 "
        "are valid options"
    )

    if dataset.equal(DataSet.MODEL1):
        corr = get_model1_corr(df=df_exp)
    else:
        corr = get_model2_corr(df=df_exp)

    res = simulation_sd_wrapper(
        df=df_exp.copy(),
        corr=corr,
        n=n,
        stoppingcriterion=stoppingcriterion,
        n_max=n_max,
    )
    data = convert_to_df(res)

    return data, res


def plot_fig2(
    n=1000,
    drop_data=DropData.SUBTREE,
):
    """Generate Figures 2, S1 and S2 comparing experimental and simulated data."""
    use_pplt_style()

    data_exp = get_data(dataset=DataSet.KLAUS, drop_data=drop_data)
    df_exp = data_exp["sd"]
    df_sim, res_sim = get_sim_data(dataset=DataSet.BARMODEL, df_exp=df_exp, n=n)

    # -------------------------------- Fig 2 -----------------------------------
    fig2, grid2 = plt.subplots(
        figsize=np.array(set_size("naturecolumn") * np.array([1, 2])) * 0.95,
        nrows=3,
        ncols=2,
        gridspec_kw={"wspace": 0, "height_ratios": (1, 1, 1.3), "hspace": 0.2},
        constrained_layout=True,
    )

    # correlation plots
    plot_phase_identity(dataset_exp=DataSet.KLAUS, df_exp=df_exp, axs=grid2[0])
    # plot_res_corr(df_exp, df_sim, ax=grid2[0, 1])
    plot_kinship_correlation(df_exp, df_sim, ax=grid2[1, 1])

    plot_sphase_vs_delay(
        df_exp=df_exp,
        dataset_exp=DataSet.KLAUS,
        df_sim=df_sim,
        dataset_sim=DataSet.BARMODEL,
        ax=grid2[2, 1],
        gen="2nd",
    )
    grid2[2, 0].remove()
    grid2[1, 0].remove()
    fig2.savefig(fname=FIGORDNER + f"fig2_n_{n}.pdf")

    # -------------------------------- Fig S2 ----------------------------------
    figs2, grids2 = plt.subplots(
        figsize=np.array(set_size("nature_wide", subplot=[2, 3])),
        nrows=2,
        ncols=3,
        constrained_layout=True,
    )
    for ax in grids2.flatten():
        plot_nuclei_count(
            data=get_data(dataset=DataSet.AIRY, drop_data=drop_data),
            dataset=DataSet.AIRY,
            ax=ax,
        )
    for ax in grids2.flatten()[1:]:
        ax.remove()
    figs2.savefig(fname=FIGORDNER + "figs2.pdf")

    # -------------------------------- Fig S1 ----------------------------------
    figsize = np.array(set_size("nature_wide", subplot=[2, 2.5])) * 1.2
    figs1 = plt.figure(figsize=figsize)
    subfigs = figs1.subfigures(
        nrows=2, ncols=1, hspace=0.05, height_ratios=[2, 3], wspace=0
    )
    # plot results
    fig_kwargs_dist = {
        "nrows": 2,
        "ncols": 4,
        "sharey": "col",
        "sharex": "col",
        "gridspec_kw": {"wspace": 0.3, "hspace": 0.2},
        "width_ratios": [1, 1, 1, 0.5],
    }
    fig_kwargs_corr = {
        "nrows": 3,
        "ncols": 5,
        "gridspec_kw": {"wspace": 0.2, "hspace": 0.2},
    }
    grid_dist = subfigs[0].subplots(**fig_kwargs_dist)
    grid_corr = subfigs[1].subplots(**fig_kwargs_corr)
    plot_phases_share_dist(
        df_exp,
        fig=subfigs[0],
        grid_full=grid_dist,
        n_bs=50000,
        dataset_exp=DataSet.KLAUS,
    )
    plot_corr_pooling(
        df_exp, grid=grid_corr, n_bs=50000, dataset_exp=DataSet.KLAUS
    )

    figs1.savefig(fname=FIGORDNER + "figs1.pdf")
