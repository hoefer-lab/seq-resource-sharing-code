# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Shared library — enums, plotting helpers, data loaders and style settings.

Notation mapping (code → manuscript)
-------------------------------------
``d``            →  D*-phase        (delay / gap between replications)
``s``            →  S*-phase        (DNA replication)
``std``          →  σ_{D*}          (D-phase duration standard deviation)
``scaling``      →  ⟨τ_min⟩        (mean minimal S-phase duration)
``f_threshold``  →  c_{i,min}       (minimal activated fraction)

Modules
-------
enums
    ``EnhancedEnum`` base class and domain enumerations (``DataSet``,
    ``Alignment``, ``DropData``, etc.).
style
    Colour palettes, rcParams presets, and figure-sizing utilities.
data_loader
    Unified data loader for Klaus and Airyscan experiments plus
    simulation-result conversion helpers.
plotting
    All paper-figure plotting functions (bulk statistics, kinship
    correlations, violin plots, etc.).
correlation_matrix
    Kinship correlation computation and pooled-correlation plots.
shared_resource
    Numba-accelerated ODE resource model.
"""
