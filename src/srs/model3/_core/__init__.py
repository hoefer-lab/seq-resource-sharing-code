# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Core simulation engine for the branching-process model.

Modules
-------
simulation
    Main simulation loop and helper functions for a single
    branching-process realisation.
wrapper
    Parallel / serial wrapper that runs many realisations via *joblib*.
ode
    ODE right-hand side and solvers (Euler, LSODA) for the
    activated-fraction dynamics.
distributions
    Gamma-distribution sampling for D-phase durations.
"""
