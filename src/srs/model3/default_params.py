# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Default model parameters and shared constants.

The MAP parameter estimates were obtained from ABC-SMC inference
(15 populations, 10 000 particles, ``c_min = 0.15``) and are used by the
plotting routines to generate Fig. 6 without re-running the inference.

Notation mapping (code → manuscript)
-------------------------------------
``d``            →  D\*-phase  (delay / gap between replications)
``s``            →  S\*-phase  (DNA replication)
``std``          →  :math:`\\sigma_{D^*}`     (D-phase duration std. dev.)
``scaling``      →  :math:`\\langle\\tau_{\\min}\\rangle`  (mean minimal S-phase duration)
``F_THRESHOLD``  →  :math:`c_{i,\\min}`       (minimal activated fraction)
"""

F_THRESHOLD = 0.15
"""Activated-fraction threshold :math:`c_{i,\\min}` defining the effective S*-phase."""

DEFAULT_PARAMS = {
    "kb": 20.3666,
    "ku": 0.004220,
    "scaling": 88.17,
    "std": 0.0934,
    "rho": 4.7706,
    "tau": 0.2096,
    "zeta_low": 0.7820,
    "zeta_high": 1.4980,
}
"""MAP parameter estimates from ABC-SMC inference.

Keys
----
kb : float
    Binding rate (linear scale; ABC prior is on log10).
ku : float
    Unbinding rate (linear scale).
scaling : float
    Time-scaling factor [min / model time unit] (manuscript: :math:`\\langle\\tau_{\\min}\\rangle`).
std : float
    D-phase duration standard deviation (manuscript: :math:`\\sigma_{D^*}`).
rho : float
    Inverse S-phase duration (``rho = 1 / tau``).
tau : float
    Minimal S-phase duration.
zeta_low, zeta_high : float
    Bounds of the uniform scarcity-factor distribution.
"""
