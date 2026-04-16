# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Conversion utilities for branching-process simulation output.

This module turns the raw NumPy arrays returned by
:func:`~model3.core.simulation.simulate_time` into phase-keyed
dictionaries that downstream plotting and inference code expects.

Functions
---------
convert_int_to_key
    Map a binary-tree integer name to a human-readable phase key.
convert_rs
    Convert a single simulation result dict into an ``{s01: [t0, t1], …}``
    phase dictionary.
get_sync_s_d
    Extract synchrony, S-phase and D-phase duration arrays from a
    DataFrame of phase durations.
"""

import numpy as np

# ~~~ BINARY-TREE NAME CONVERSION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@np.vectorize
def convert_int_to_key(i, key=""):
    """Convert a binary-tree integer name to a phase-key string.

    The ancestor (1) maps to ``'s'``; each subsequent bit encodes a
    left (0) or right (1) daughter, e.g. 5 (binary 101) → ``'s01'``.

    Parameters
    ----------
    i : int
        Binary-tree name produced by the simulation (1 = root, 2/3 = first
        daughters, 4–7 = granddaughters, etc.).
    key : str
        Accumulator for recursive calls (leave at default ``""``).

    Returns
    -------
    phase_key : str
        String of the form ``'s'`` followed by a binary suffix, e.g.
        ``'s'``, ``'s0'``, ``'s01'``.
    """
    if i == 1:
        return f"s{key}"
    return convert_int_to_key(i // 2, key=f"{i % 2}{key}")


# ~~~ SISTER-BRANCH SORTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _sort_sister_phases(res_dict):
    """Sort sister branches so that the shorter D-phase comes first.

    Swaps keys recursively through the entire sub-tree when a swap is
    performed, ensuring a consistent left/right ordering.
    Modifies *res_dict* in place.

    Parameters
    ----------
    res_dict : dict
        Phase dictionary (keys like ``'s0'``, ``'d01'``, etc.).
    """

    def _swap_keys(suffix1, suffix2):
        did_swap = False
        for prefix in ("d", "s"):
            key1 = prefix + suffix1
            key2 = prefix + suffix2
            if key1 in res_dict and key2 in res_dict:
                res_dict[key1], res_dict[key2] = (
                    res_dict[key2],
                    res_dict[key1],
                )
                did_swap = True
        if did_swap:
            _swap_keys(suffix1 + "0", suffix2 + "0")
            _swap_keys(suffix1 + "1", suffix2 + "1")

    def _sort_and_swap(key):
        sister_0 = f"d{key}0"
        sister_1 = f"d{key}1"
        if sister_0 in res_dict and sister_1 in res_dict:
            if np.diff(res_dict[sister_0]) > np.diff(res_dict[sister_1]):
                _swap_keys(sister_0[1:], sister_1[1:])
                _sort_and_swap(sister_0)
                _sort_and_swap(sister_1)

    for key in np.unique([k[1:] for k in res_dict]):
        _sort_and_swap(key)


# ~~~ RESULT CONVERSION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def convert_rs(rs, f_threshold):
    """Convert a single simulation result dict into a phase dictionary.

    Maps each nucleus to a key like ``'s01'`` (S-phase) and infers the
    corresponding D-phase intervals (e.g. ``'d01'``) from the gap between
    a mother's S-phase end and a daughter's S-phase start.

    Sentinel values of ``-1`` in ``s*_phases`` (meaning "not yet
    recorded") are replaced by ``NaN`` so that downstream filters work
    correctly.

    Parameters
    ----------
    rs : dict
        Result dictionary returned by
        :func:`~model3.core.simulation.simulate_time`.
    f_threshold : float
        Activated-fraction threshold used for the s*-phase definition.

    Returns
    -------
    res_dict : dict
        Dictionary keyed by phase strings (``'s0'``, ``'d01'``, …) with
        ``[start, end]`` arrays as values, plus metadata entries:

        - ``'time'`` — simulation timeline shifted so that s0 starts at 0.
        - ``'replicating nuclei at t'`` — number of nuclei above
          *f_threshold* at each time point.
        - ``'nuclei number at t'`` — total nucleus count over time.
        - ``'s-phase-dict'``, ``'s*-phase-dict'`` — raw phase arrays.
        - ``'r'`` — shared resource over time.
        - ``'f'`` — per-nucleus activated fractions (if saved).
        - ``'names'`` — binary-tree names.
        - ``'t_s*_2'`` — s*-phase duration from the second nucleus onward
          (if available in *rs*).
    """
    # Replace -1 sentinels with NaN so downstream NaN filters can properly
    # remove invalid entries instead of computing durations of zero.
    s_star = rs["s*_phases"].astype(float).copy()
    s_star[s_star < 0] = np.nan
    res_dict = {
        key: s for s, key in zip(s_star, convert_int_to_key(rs["names"]))
    }

    # Infer D-phase intervals: gap between mother's end and daughter's start
    keys = list(res_dict.keys())
    for key in keys:
        mother_key = key[:-1]
        if mother_key in keys:
            res_dict["d" + key[1:]] = np.array(
                [res_dict[mother_key][1], res_dict[key][0]]
            )

    # Sort sister branches so that the shorter D-phase comes first
    _sort_sister_phases(res_dict)

    # Append trace-level metadata
    res_dict["time"] = rs["time"] - res_dict["s0"][0]
    res_dict["replicating nuclei at t"] = (rs["f"] >= f_threshold).sum(axis=1)
    res_dict["nuclei number at t"] = rs["n"]
    res_dict["s-phase-dict"] = rs["s_phases"]
    res_dict["s*-phase-dict"] = rs["s*_phases"]
    res_dict["r"] = rs["r_0"]
    res_dict["f"] = rs["f"]
    res_dict["names"] = rs["names"]
    if "t_s*_2" in rs:
        res_dict["t_s*_2"] = rs["t_s*_2"]
    return res_dict


# ~~~ SUMMARY STATISTICS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_sync_s_d(df, n=None):
    """Extract synchrony, S-phase and D-phase durations from a DataFrame.

    Concatenates the four sister-pair combinations ``(s0, d00)``,
    ``(s0, d01)``, ``(s1, d10)``, ``(s1, d11)`` and removes rows that
    contain ``NaN`` or ``Inf`` values.

    Parameters
    ----------
    df : DataFrame
        Phase-duration DataFrame produced by
        :func:`~lib.data_loader.convert_to_df`.
    n : int, optional
        If given, truncate the returned arrays to at most *n* entries.

    Returns
    -------
    sync : ndarray
        Synchrony measure (D-phase difference normalised by mean S-phase).
    s_phase : ndarray
        S-phase durations.
    d_phase : ndarray
        D-phase durations.
    """
    s_phase = np.concatenate([df.s0, df.s0, df.s1, df.s1])
    d_phase = np.concatenate([df.d00, df.d01, df.d10, df.d11])
    sync = np.concatenate(4 * [(df.d1 - df.d0) / np.nanmean(df.s0)])

    bad = (
        np.isnan(s_phase)
        | np.isnan(sync)
        | np.isnan(d_phase)
        | np.isinf(s_phase)
        | np.isinf(sync)
        | np.isinf(d_phase)
    )
    return sync[~bad][:n], s_phase[~bad][:n], d_phase[~bad][:n]
