# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Models 1 and 2 — Correlated branching process.

This package implements a stochastic branching-process model where sister
nuclei draw correlated S-phase and D-phase durations from the empirical
distribution using conditional multivariate normals.

**Model 2** (``DataSet.BARMODEL``) uses the full empirical kinship
correlation matrix, including mother–daughter and sister–sister
correlations.

**Model 1** (``DataSet.MODEL1``) is the independent null model: all
off-diagonal correlations are removed except the D-phase sister
correlation (d0 ↔ d1), which reflects the shared division event.
Model 1 is a strict subset of model 2.

Subpackages
-----------
_core
    Simulation engine: multivariate-normal sampling, distribution
    transforms, and the branching-process loop.
plotting
    Visualisation routines for paper figures (Fig. 2 and SI).
"""

__all__ = ["plotting"]
