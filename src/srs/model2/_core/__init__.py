# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Core simulation engine for the correlated branching-process model.

Modules
-------
simulation
    Main simulation loop and helper functions for a single
    branching-process realisation.
wrapper
    Parallel wrapper that runs many realisations via *joblib*.
distribution
    Percentile-based sampling of correlated S/D-phase durations
    using conditional multivariate normals.
multivariate_normal
    Multivariate normal distribution with partitioning and
    conditional-distribution methods.
"""
