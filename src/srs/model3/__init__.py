# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Model 3 — Sequential resource-sharing branching process.

This package implements a stochastic branching-process model where nuclei
compete for a shared, finite replication resource.  Each nucleus undergoes
a D-phase (delay) followed by an S-phase (DNA replication) whose speed
depends on the fraction of resource it has captured.

Subpackages
-----------
_core
    Simulation engine: ODE solvers, distribution sampling, and the
    branching-process loop.
plotting
    Visualisation routines for paper figures (trees, Figs. 4–6).
generate
    Data-generation scripts for Figs. 4 and 5 — runs parameter scans
    and saves results.

Modules
-------
_inference
    Approximate Bayesian Computation (ABC-SMC) parameter inference.
default_params
    MAP parameter estimates and the canonical ``F_THRESHOLD`` constant.
_convert
    Shared utilities for converting raw simulation output into
    phase-keyed dictionaries.
_tools
    Small numerical helpers (sliding mean, filtering, binary names).
_read_data
    I/O for pre-computed simulation-scan results stored as JSON text
    files.
"""

__all__ = ["plotting", "generate", "default_params"]
