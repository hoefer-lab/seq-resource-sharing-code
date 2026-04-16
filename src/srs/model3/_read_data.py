# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""I/O for pre-computed branching-process parameter-scan results.

The simulation parameter scans (growth rate, utilisation, etc. as
functions of :math:`\\zeta` and :math:`\\tau`) are stored as
newline-delimited JSON text files in ``data/``.  This module reads
them back into dictionaries of NumPy arrays.
"""

import json
import os

import numpy as np

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data"
)

FILENAME_DEFAULT = os.path.join(_DATA_DIR, "param_scan_fig45.jsonl")
FILENAME_SCAN_REGIME = os.path.join(
    _DATA_DIR,
    "param_scan_figs3.jsonl",
)


def read_data(filename=FILENAME_DEFAULT):
    """Read branching-process parameter-scan results from a JSON text file.

    Each line of the file is a JSON object with per-parameter-point results.
    The returned dictionary contains NumPy arrays keyed by observable name.

    Parameters
    ----------
    filename : str, default=FILENAME_DEFAULT
        Path to the newline-delimited JSON file.

    Returns
    -------
    data : dict
        Scalar keys (``'zeta'``, ``'tau'``, …) map to 1-D arrays;
        observable keys (``'lambda_seq'``, ``'eta_par'``, …) map to
        ``{'mean': ndarray, 'std': ndarray}`` sub-dicts.
    """
    # Opening JSON file
    with open(filename, "r") as f:
        # Reading from json file
        raw_data = f.read()
        raw_data = (
            raw_data.replace("\\u03b7_", "eta_")
            .replace("\\u03c6_", "phi_")
            .replace("\\u03bb_", "lambda_")
        )
        raw_data = raw_data.split("\n")

    keys_single = [
        "s-phase_duration_seq",
        "s-phase_duration_par",
        "n_gen",
        "std",
        "zeta",
        "tau",
        "k_b",
        "resource_type",
        "kus",
    ]
    keys_dict = [
        "growth_advantage_n",
        "utilization_advantage",
        "growth_advantage_g",
        "lambda_seq",
        "phi_seq",
        "eta_seq",
        "lambda_par",
        "phi_par",
        "eta_par",
    ]
    data_single = {key: [] for key in keys_single}
    data_dict = {key: {"mean": [], "std": []} for key in keys_dict}
    data = {**data_single, **data_dict}

    for rw_dt in raw_data:
        if len(rw_dt) > 0:
            try:
                rw_dt_dict = json.loads(rw_dt)
            except json.JSONDecodeError:
                continue
            for key in keys_single:
                data[key].append(rw_dt_dict[key])
            for key in keys_dict:
                for k in data[key].keys():
                    data[key][k].append(rw_dt_dict[key][k])
    # convert to numpy
    for key in keys_single:
        data[key] = np.array(data[key])
    for key in keys_dict:
        for k in data[key].keys():
            data[key][k] = np.array(data[key][k])

    return data
