# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""JSON serialisation of simulation results with NumPy support.

Functions
---------
save_data
    Append a single parameter-point result as one JSON line.
"""

import json
import os

import numpy as np

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"
)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy arrays to Python lists."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_data(parameters, n_sample, resource_type, n_gen):
    """Append simulation results for one parameter point to a JSON file.

    Each call appends a single newline-delimited JSON record to the
    output file in ``data/``.

    Parameters
    ----------
    parameters : dict
        The result dictionary to serialise.
    n_sample : int
        Number of samples (used in the filename).
    resource_type : ResourceType
        Resource-growth model (its ``.value`` appears in the filename).
    n_gen : int
        Number of generations (used in the filename).
    """
    filename = os.path.join(
        _DATA_DIR,
        f"param_scan_n_{n_sample}_{resource_type.value}_n_gen_{n_gen}.jsonl",
    )
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n")
        json.dump(parameters, f, cls=_NumpyEncoder)
