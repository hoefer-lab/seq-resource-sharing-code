# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Data-generation routines for Figs. 4 and 5.

Modules
-------
fig45
    Entry-point script that runs the branching-process simulations and
    saves results to disk.
estimate_parameters
    Extract steady-state growth parameters and advantage metrics from
    raw simulation output.
steady_state
    Compute steady-state growth rate, utilisation, and activation for
    individual simulation runs using weighted polynomial fits.
growth_advantage
    Calculate growth and utilisation advantages between sequential and
    parallel resource-sharing modes.
save
    JSON serialisation of simulation results with NumPy support.
polyfit
    Weighted polynomial fitting with bump-function kernels, used by
    :mod:`.steady_state`.
"""
