# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Single-realisation simulation of the correlated branching process (model 2).

Given an empirical phase-duration DataFrame and its correlation matrix,
this module draws correlated S- and D-phase durations to build a
complete replication tree.  Setting the correlation matrix to the
model-1 (independent) form yields the null model without kinship
inheritance.
"""

# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

from srs.lib.enums import StoppingCriterion

from .distribution import get_s_and_d, get_single_time


def simulation_sd(
    df,
    corr,
    stoppingcriterion=StoppingCriterion.TIMER,
    t_max=800,
    n_max=2**8,
):
    """Simulate replication dynamic with two phases (S and G2MG1).

    Args:
        df (DataFrame): Data; default unfiltered data defined in read_data.py
        stop_criterion (StopCriterion): Criterion to stop simulation (count,
            time time_fixed)
        method (String): Method to fit data ('erlang', 'sample')

    Returns:
        rpl_dict (dict): dict with timepoints of all s-phases

    """

    rpl_dict = nucleus_division(
        df=df,
        corr=corr,
        stoppingcriterion=stoppingcriterion,
        t_max=t_max,
        n_max=n_max,
    )
    # choosing t_max as the last recoreded event
    # t_max = np.max(rpl_dict[max(rpl_dict)])
    t_max = max(i for v in rpl_dict.values() for i in v)

    rpl_dict = add_time_count_and_total_number(rpl_dict=rpl_dict, t_max=t_max)

    # align time
    # corresponds to alignment at start of S1
    shift = np.min([rpl_dict["s0"][0], rpl_dict["s1"][0]])
    for k, v in rpl_dict.items():
        if ("s" in k) or ("d" in k) or ("time" in k):
            rpl_dict[k] = v - shift

    return rpl_dict


def add_time_count_and_total_number(rpl_dict, t_max):
    """Add time, replicating nuclei and nuclei number at t to dict."""
    time_points = np.arange(0, t_max + 1)
    replicating_nuclei = np.zeros_like(time_points)
    nuclei_number = np.ones_like(time_points)

    for key, [start_s, end_s] in rpl_dict.items():
        if key.startswith("s"):
            replicating_nuclei[start_s:end_s] += 1
            nuclei_number[end_s:] += 1

    rpl_dict["time"] = time_points
    rpl_dict["replicating nuclei at t"] = replicating_nuclei
    rpl_dict["nuclei number at t"] = nuclei_number
    return rpl_dict


def nucleus_division(stoppingcriterion, df, t_max, n_max, corr, position="s"):
    """Perform nucleus division and return dict with replication times.

    Args:
        df (DataFrame): replication data
        stop_criterion (StopCriterion): Criterion to stop simulation (count,
            time time_fixed)
        final_nuclei_count ()
    Returns:
        rpl_dict (dict): dict with timepoints of all s-phases

    """

    def is_finished(end_s, n):
        if stoppingcriterion.equal(StoppingCriterion.TIMER):
            return end_s <= t_max
        else:
            return n < n_max

    # convert df to a dict with each key corresponding to the data for a
    # given phase, e.g. S, Dx, Sx, Dxx
    data = convert_df(df=df)

    # Dict to track all s-phases
    rpl_dict = {}

    dict_sd = get_s_and_d(
        data=data,
        s_pre=get_single_time(datas=[data["s"]]).item(),
        d_pre=None,
        corr=corr,
        is_first=True,
    )
    # add first nuclei to list
    nuclei_list = [{"t": 0, "name": position} | dict_sd]
    n = 1
    # Loop as long as a nuclei are inside the list
    while nuclei_list:
        if len(nuclei_list) > 1:
            min_value = min(nuclei_list, key=lambda x: x["t"])
            min_index = nuclei_list.index(min_value)
        else:
            min_index = 0
        # pop the nucleus which replicates next
        active_nuclei = nuclei_list.pop(min_index)

        start_s = active_nuclei["t"]
        end_s = start_s + active_nuclei["s"]

        # ensure that only if a nucleus would finish replication before
        # t_rep, it produces two daughter nuclei
        if is_finished(end_s=end_s, n=n):  # len(rpl_dict) + 1):
            rpl_dict[active_nuclei["name"]] = np.array([start_s, end_s])
            n += 1
            for daughter in ["0", "1"]:
                name = active_nuclei["name"] + daughter

                dict_sd = get_s_and_d(
                    data=data,
                    corr=corr,
                    s_pre=active_nuclei[f"s{daughter}"],
                    d_pre=active_nuclei[f"d{daughter}"],
                    is_second=n == 2,
                )

                # add d-phase to results
                rpl_dict[f"d{name[1:]}"] = np.array(
                    [end_s, end_s + dict_sd["d"]]
                )

                nuclei_list.append(
                    {"t": end_s + dict_sd["d"], "name": name} | dict_sd
                )
    return rpl_dict


def convert_df(df):
    """Convert of to a dict with the respective arrays needed for get_s_and_d"""

    keys = ["s", "sx", "dx", "dxx"]
    columns = [["s"], ["s0", "s1"], ["d0", "d1"], ["d00", "d01", "d10", "d11"]]

    result = {
        key: df[column].dropna().values.flatten()
        for key, column in zip(keys, columns)
    }
    return result
