# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Parallel wrapper for the correlated branching-process simulation.

Runs many independent realisations of
:func:`~model2.core.simulation.simulation_sd` via *joblib* and collects
results into an ``{RBC-i: tree}`` dictionary.
"""

# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import multiprocessing

import numpy as np
from joblib import Parallel, delayed

from .simulation import simulation_sd


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def simulation_sd_wrapper(
    df,
    corr,
    n,
    stoppingcriterion,
    num_cores=multiprocessing.cpu_count(),
    n_max=2**8,
):
    """Execute simulation_r_d n times."""
    results = np.array(
        Parallel(n_jobs=num_cores)(  # , verbose=1
            delayed(simulation_sd)(
                df=df,
                corr=corr,
                stoppingcriterion=stoppingcriterion,
                n_max=n_max,
            )
            for i in range(n)
        )
    )
    dict_of_nuclei = {f"RBC-{i + 1}": v for i, v in enumerate(results)}

    return dict_of_nuclei
