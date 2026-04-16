# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Domain enumerations used across all packages.

Notation mapping (code → manuscript)
-------------------------------------
``d``            →  D\*-phase  (delay / gap between replications)
``s``            →  S\*-phase  (DNA replication)
``std``          →  :math:`\\sigma_{D^*}`     (D-phase duration std. dev.)
``scaling``      →  :math:`\\langle\\tau_{\\min}\\rangle`  (mean minimal S-phase duration)
``f_threshold``  →  :math:`c_{i,\\min}`       (minimal activated fraction)
"""

from enum import Enum


class EnhancedEnum(Enum):
    """Enum with function to compare their strings."""

    def equal(self, enum):
        """Compare strings (workaround for two Enums from diff. instances)."""
        return self.value == enum.value

    def equal_str(self, string):
        """Compare enum value with given string."""
        return self.value == string

    # overload == (equal to) operator
    def __eq__(self, point_ov):
        """Compare two enums."""
        return self.value == point_ov.value


class StoppingCriterion(EnhancedEnum):
    """Criterion for terminating a branching-process simulation."""

    TIMER = "timer"  # stops after reaching a predefined time
    COUNTER = "counter"  # stops after reaching a predefined number of nuclei


class DataSet(EnhancedEnum):
    """Identifier for experimental and simulated datasets.

    Experimental
    ------------
    KLAUS : spinning-disk data from Klaus *et al.* 2022 (Figs. 2 & 6).
    AIRY  : Airyscan-2 data (higher resolution, ~33 % slower cycles due to
            phototoxicity; used for Fig. S7).

    Simulated — model 1 & 2  (correlated branching process)
    --------------------------------------------------------
    MODEL1   : model 1 — independent branching process (no kinship
               inheritance; D-sister correlations only).  Subset of
               model 2 with off-diagonal correlations removed.
    BARMODEL : model 2 — full kinship-correlated branching process.

    Simulated — model 3  (resource-sharing branching process)
    ---------------------------------------------------------
    RESMODEL    : model 3, sequential resource sharing.
    RESPARMODEL : model 3, parallel resource sharing.
    """

    KLAUS = "klaus"
    AIRY = "airyscan"
    MODEL1 = "indep model"
    BARMODEL = "inh model"
    RESMODEL = "res seq model"
    RESPARMODEL = "res par model"


class ODEMethod(EnhancedEnum):
    """Numerical method for the activated-fraction ODE."""

    EULER = "euler"
    LSODA = "lsoda"


class ResourceType(EnhancedEnum):
    """Resource-growth mode for model 3 (maps to :math:`R_0`)."""

    # constant resource, i.e. r_tot = zeta
    CONST = "const"
    # growing resource depending on current nuclei number, i.e. r_tot = zeta * n
    GROWING_N = "zeta_n"
    # growing resource depending on current genome count, i.e. r_tot = zeta * g
    GROWING_G = "zeta_g"
    # PCNA1-like resource, i.e. r_tot grows similar to the PCNA1 signal
    PCNA1 = "pcna1"


class DropData(EnhancedEnum):
    """Filtering mode for incomplete S/D-phase observations."""

    # use all raw data
    NONE = "none"
    # only keep phases observed in both sisters
    SUBTREE = "subtree"
    # Only use phases observed in all nuclei
    ALL = "all"
