# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Single-realisation branching-process simulation.

This module implements the core simulation loop in which nuclei undergo
alternating D-phases (delay) and S-phases (DNA replication).  During
S-phase each nucleus competes for a shared resource that determines its
replication speed (activated fraction *f*).  When a nucleus finishes
S-phase it spawns two daughter nuclei that enter new D-phases.

The main entry point is :func:`simulate_time`; all heavy-lifting helpers
are Numba-JIT compiled for performance.
"""

# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
from numba import njit

from srs.lib.enums import ODEMethod, ResourceType, StoppingCriterion
from srs.lib.shared_resource import get_resource

from .distributions import get_gamma
from .ode import get_activated_fraction_euler, get_activated_fraction_nlsoda

EULER = ODEMethod.EULER.value
LSODA = ODEMethod.LSODA.value
COUNT = StoppingCriterion.COUNTER.value
TIME = StoppingCriterion.TIMER.value


# ~~~ LEAF HELPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@njit(error_model="numpy")
def stopping_sim(stopping_criterion, is_finished, is_active, size):
    """Check whether the simulation should be stopped.

    Parameters
    ----------
    stopping_criterion : str
        ``'counter'`` stops after ``size`` nuclei have finished;
        ``'timer'`` stops once ``size`` nuclei are active or finished.
    is_finished : ndarray of bool, shape (size,)
        True for nuclei that completed their S-phase.
    is_active : ndarray of bool, shape (size,)
        True for nuclei currently in D- or S-phase.
    size : int
        Target number of nuclei.

    Returns
    -------
    bool
    """
    stop_count = stopping_criterion == COUNT and is_finished.sum() >= size
    stop_time = (
        stopping_criterion == TIME
        and (is_finished.sum() + is_active.sum()) >= size
    )
    return stop_count or stop_time


@njit(error_model="numpy")
def get_activated_fraction(ode_method, r_0, f_0, delta_t0, k_u, k_b):
    """Dispatch ODE solve for the activated fraction to the chosen method.

    Parameters
    ----------
    ode_method : str
        ``'euler'`` or ``'lsoda'``.
    r_0 : float
        Current shared resource.
    f_0 : ndarray
        Activated fractions of in-S nuclei.
    delta_t0 : float
        Maximum time step.
    k_u, k_b : float
        Unbinding / binding rates.

    Returns
    -------
    delta_t : float
        Actual time step used.
    f_1 : ndarray
        Updated activated fractions.
    """
    if ode_method == LSODA:
        return get_activated_fraction_nlsoda(
            r_0=r_0, f_0=f_0, delta_t0=delta_t0, k_u=k_u, k_b=k_b
        )
    return get_activated_fraction_euler(
        r_0=r_0, f_0=f_0, delta_t0=delta_t0, k_u=k_u, k_b=k_b
    )


@njit(error_model="numpy")
def spawn_daughters(names, is_active, is_finished, is_finishing):
    """Activate up to two daughter nuclei for each finishing mother.

    Daughters are placed into the first free slots (neither active nor
    finished) and named according to the binary tree convention:
    mother *m* produces daughters *2m* and *2m + 1*.

    Parameters
    ----------
    names : ndarray of int32, shape (size,)
        Binary tree names (modified in place).
    is_active, is_finished : ndarray of bool, shape (size,)
        Nucleus state flags (modified in place).
    is_finishing : ndarray of bool, shape (size,)
        Mask for nuclei that just completed S-phase.
    """
    for name in names[is_finishing]:
        free_slots = np.where(~is_active & ~is_finished)[0]
        if len(free_slots) == 0:
            break
        n_new = min(len(free_slots), 2)
        for i in range(n_new):
            names[free_slots[i]] = 2 * name + i
            is_active[free_slots[i]] = True


# ~~~ BUILDING BLOCKS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@njit(error_model="numpy")
def step_s_phase(
    ode_method,
    r_0,
    activated_fraction,
    s_progress,
    s_phases,
    s_phases_star,
    is_active,
    is_finished,
    names,
    in_s,
    delta_t0,
    delta_s,
    k_u,
    k_b,
    f_threshold,
    time,
):
    """Advance all S-phase nuclei by one time step.

    1. Records S-phase and s*-phase start times for newly entering nuclei.
    2. Solves the ODE for the activated fraction.
    3. Advances S-phase progress.
    4. Marks finishing nuclei and spawns their daughters.

    Parameters
    ----------
    ode_method : str
        ODE solver identifier (``'euler'`` or ``'lsoda'``).
    r_0 : float
        Current shared resource.
    activated_fraction : ndarray, shape (size,)
        Fraction of active replication complexes per nucleus (modified in place).
    s_progress : ndarray, shape (size,)
        Replication progress in [0, 1] (modified in place).
    s_phases : ndarray, shape (size, 2)
        Start / end times of each nucleus's S-phase (modified in place).
    s_phases_star : ndarray, shape (size, 2)
        Start / end times of the *effective* S-phase (f >= f_threshold).
        Initialised to -1; entries < 0 mean "not yet recorded".
    is_active, is_finished : ndarray of bool, shape (size,)
        Nucleus state flags (modified in place).
    names : ndarray of int32, shape (size,)
        Binary tree names (modified in place when daughters are spawned).
    in_s : ndarray of bool, shape (size,)
        Mask for nuclei currently in S-phase.
    delta_t0 : float
        Maximum allowed time step.
    delta_s : float
        S-phase rate (= rho).
    k_u, k_b : float
        Unbinding / binding rates.
    f_threshold : float
        Activated-fraction threshold that defines the s*-phase start.
    time : float
        Current simulation time.

    Returns
    -------
    delta_t : float
        Actual time step used (may be smaller than *delta_t0* for stability).
    f_0 : ndarray
        Updated activated fractions for in-S nuclei.
    """
    # Record start times for nuclei newly entering S-phase
    s_phases[(s_progress == 0) & in_s, 0] = time
    # Record s* start for nuclei first surpassing f >= f_threshold.
    # s_phases_star is initialised to -1; entries < 0 mean "not yet set".
    s_phases_star[
        (
            (s_phases_star[:, 0] < 0)
            & (activated_fraction >= f_threshold)
            & in_s
        ),
        0,
    ] = time

    # Solve ODE for activated fraction
    delta_t, f_0 = get_activated_fraction(
        ode_method=ode_method,
        r_0=r_0,
        f_0=activated_fraction[in_s],
        delta_t0=delta_t0,
        k_u=k_u,
        k_b=k_b,
    )
    activated_fraction[in_s] = f_0
    activated_fraction[~in_s] = 0.0

    # Advance S-phase progress
    s_progress[in_s] += delta_t * delta_s * activated_fraction[in_s]
    s_progress[:] = np.clip(s_progress, 0, 1)

    # Detect nuclei finishing S-phase (progress == 1 AND was in S)
    is_finishing = ((s_progress >= 1) + in_s.astype(np.int8)) == 2
    if is_finishing.any():
        s_phases[is_finishing, 1] = time
        # Only record s* end for nuclei whose s* start was recorded
        has_star_start = s_phases_star[:, 0] >= 0
        s_phases_star[is_finishing & has_star_start, 1] = time

        is_active[is_finishing] = False
        is_finished[is_finishing] = True
        activated_fraction[is_finishing] = 0.0

        spawn_daughters(names, is_active, is_finished, is_finishing)

    return delta_t, f_0


@njit(error_model="numpy")
def record_snapshot(
    timeline,
    resource,
    dna_content,
    count,
    eta,
    phi,
    active_resource,
    time,
    r_0,
    s_progress,
    is_finished,
    f_0,
    in_s,
    save_f,
    activated_fraction,
):
    """Append current system state to the time-series accumulators.

    Parameters
    ----------
    timeline, resource, dna_content, count, eta, phi : ndarray
        Growing 1-D arrays that accumulate simulation history.
    active_resource : ndarray
        Flat buffer collecting per-nucleus activated fractions (only
        appended when *save_f* is True).
    time : float
        Current simulation time.
    r_0 : float
        Current shared resource.
    s_progress : ndarray, shape (size,)
        Replication progress of each nucleus.
    is_finished : ndarray of bool, shape (size,)
        Finished-nucleus flags.
    f_0 : ndarray
        Activated fractions for in-S nuclei (from last ODE step).
    in_s : ndarray of bool, shape (size,)
        Mask for nuclei currently in S-phase.
    save_f : bool
        Whether to store per-nucleus activated fractions.
    activated_fraction : ndarray, shape (size,)
        Full activated-fraction vector.

    Returns
    -------
    timeline, resource, dna_content, count, eta, phi, active_resource
        The same arrays with one new entry appended.
    """
    timeline = np.append(timeline, time)
    resource = np.append(resource, r_0)
    dna_content = np.append(dna_content, 1 + s_progress.sum())
    count = np.append(count, is_finished.sum() + 1)
    if in_s.any():
        eta = np.append(eta, f_0.sum() / r_0)
        phi = np.append(phi, f_0.sum() / in_s.sum())
    else:
        eta = np.append(eta, 0)
        phi = np.append(phi, 1)
    if save_f:
        active_resource = np.append(active_resource, activated_fraction)
    return timeline, resource, dna_content, count, eta, phi, active_resource


# ~~~ MAIN SIMULATION LOOP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@njit(error_model="numpy")
def propagate_time(
    resource_type,
    stopping_criterion,
    zeta,
    k_u,
    k_b,
    size,
    delta_s,
    delta_d,
    delta_t0,
    save_f,
    ode_method,
    acceleration,
    f_threshold,
    scaling,
):
    """Propagate the branching process in time.

    Starting from a single ancestor nucleus the simulation loop repeats:

    1. Compute the current shared resource *r_0*.
    2. Advance S-phase nuclei (ODE + progress + daughter spawning) via
       :func:`step_s_phase`.
    3. Advance D-phase counters.
    4. Periodically record a snapshot via :func:`record_snapshot`.

    Parameters
    ----------
    resource_type : str
        Resource model identifier (e.g. ``'zeta_g'``, ``'zeta_n'``).
    stopping_criterion : str
        ``'counter'`` or ``'timer'``.
    zeta : float
        Scarcity factor.
    k_u, k_b : float
        Unbinding / binding rates.
    size : int
        Maximum number of nuclei tracked.
    delta_s : float
        S-phase rate (= rho).
    delta_d : ndarray, shape (size,)
        Per-nucleus D-phase rates (reciprocal of gamma-sampled durations).
    delta_t0 : float
        Maximum time step.
    save_f : bool
        Store per-nucleus activated fractions in the output.
    ode_method : str
        ``'euler'`` or ``'lsoda'``.
    acceleration, scaling : float
        Factors applied to time when evaluating the resource model.
    f_threshold : float
        Threshold for the s*-phase start.

    Returns
    -------
    names : ndarray of int32
    s_phases, s_phases_star : ndarray, shape (size, 2)
    timeline, resource, eta, phi, dna_content, count : ndarray
    active_resource : ndarray
    """
    # Minimum interval between recorded snapshots (at least 100 per cycle)
    time_steps = 0.01

    # --- state arrays ---
    s_progress, d_progress, activated_fraction = np.zeros((3, size))
    is_active, is_finished = np.zeros((2, size), dtype=np.bool_)
    names = np.zeros(size, dtype=np.int32)
    s_phases = np.zeros((size, 2))
    s_phases_star = -np.ones((size, 2))  # -1 sentinel = "not yet recorded"

    # --- time-series accumulators ---
    timeline, resource, eta = np.zeros((3, 1))
    dna_content, count, phi = np.ones((3, 1))
    active_resource = np.zeros(size)

    # First nucleus starts directly in S-phase
    is_active[0] = True
    d_progress[0] = 1  # skip initial D-phase
    time = 0
    # Binary naming: ancestor=1, daughters 10/11, 100/101/110/111, ...
    names[0] = 1
    f_0 = np.zeros(1)

    while not stopping_sim(
        stopping_criterion=stopping_criterion,
        is_finished=is_finished,
        is_active=is_active,
        size=size,
    ):
        # Compute shared resource
        if resource_type == "zeta_n":
            system_size = 1 + is_finished.sum()
        else:
            system_size = 1 + s_progress.sum()

        r_0 = get_resource(
            resource_type,
            zeta=zeta,
            count=system_size,
            time=time * acceleration * scaling,
        )

        # Determine which nuclei are in D- vs S-phase
        in_d = (d_progress < 1) & is_active
        in_s = (d_progress >= 1) & is_active
        delta_t = delta_t0

        # Advance S-phase nuclei
        if in_s.any():
            delta_t, f_0 = step_s_phase(
                ode_method,
                r_0,
                activated_fraction,
                s_progress,
                s_phases,
                s_phases_star,
                is_active,
                is_finished,
                names,
                in_s,
                delta_t0,
                delta_s,
                k_u,
                k_b,
                f_threshold,
                time,
            )

        # Advance time and D-phase progress
        time += delta_t
        d_progress[in_d] += delta_t * delta_d[in_d]

        # Record snapshot periodically or at final step
        if (time - timeline[-1] >= time_steps) or stopping_sim(
            stopping_criterion=stopping_criterion,
            is_finished=is_finished,
            is_active=is_active,
            size=size,
        ):
            (
                timeline,
                resource,
                dna_content,
                count,
                eta,
                phi,
                active_resource,
            ) = record_snapshot(
                timeline,
                resource,
                dna_content,
                count,
                eta,
                phi,
                active_resource,
                time,
                r_0,
                s_progress,
                is_finished,
                f_0,
                in_s,
                save_f,
                activated_fraction,
            )

    return (
        names,
        s_phases,
        timeline,
        resource,
        eta,
        phi,
        active_resource,
        dna_content,
        count,
        s_phases_star,
    )


# ~~~ PUBLIC API ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def simulate_time(
    k_u,
    k_b,
    std,
    rho=2.0,
    zeta=1,
    save_f=False,
    resource_type=ResourceType.GROWING_G,
    n_stop=None,
    n_gen=None,
    ode_method=ODEMethod.EULER,
    acceleration=1,
    f_threshold=0.2,
    scaling=1,
):
    """Simulate a branching process where each nucleus undergoes S- and D-phase.

    Parameters
    ----------
    k_u : float
        Unbinding rate [/cycle].
    k_b : float
        Binding rate [/cycle].
    std : float
        Standard deviation of the D-phase duration.
    rho : float, default=2.0
        Inverse S-phase duration, ``rho = 1 / tau_s`` where
        ``tau_s + tau_d = 1``.
    zeta : float, default=1.0
        Scarcity factor describing amount of resource available.
    save_f : bool, default=False
        Whether to include per-nucleus activated fractions in the output.
    resource_type : ResourceType, default=ResourceType.GROWING_G
        Determines how the shared resource scales with system size.
    n_stop : int, optional
        Number of nuclei where the simulation is stopped (strict counter).
        Either *n_stop* or *n_gen* must be specified.
    n_gen : int, optional
        Approximate number of generations to simulate.
        Ignored when *n_stop* is given.
    ode_method : ODEMethod, default=ODEMethod.EULER
        Solver for the activated-fraction ODE.
    acceleration : float, default=1
        Factor applied to time when evaluating the resource model.
    f_threshold : float, default=0.2
        Activated-fraction threshold defining the start of the effective
        S-phase (s*-phase).
    scaling : float, default=1
        Additional time-scaling factor for the resource model.

    Returns
    -------
    results : dict
        Dictionary with the following keys:

        - ``'names'`` — binary tree names of each nucleus (ndarray of int32).
        - ``'s_phases'`` — S-phase [start, end] times, shape (size, 2).
        - ``'s*_phases'`` — effective S-phase [start, end] times,
          shape (size, 2); entries of -1 indicate unrecorded values.
        - ``'time'`` — simulation time series.
        - ``'r_0'`` — shared resource over time.
        - ``'eta'`` — total activated fraction / resource over time.
        - ``'phi'`` — mean activated fraction per in-S nucleus over time.
        - ``'g'`` — genome count (DNA content) over time.
        - ``'n'`` — nucleus count over time.
        - ``'t_s*'`` — overall s*-phase duration (start of first s* to
          end of last s*); -1 if fewer than one valid s*-phase exists.
        - ``'t_s*_2'`` — s*-phase duration from start of the *second*
          s* to end of last s*; -1 if fewer than two valid s*-phases exist.
        - ``'f'`` — *(only when save_f=True)* per-nucleus activated
          fractions at each snapshot, shape (n_snapshots, size).
    """
    # check if n_stop is provided, if so force stopping-criterion to be a
    # strict counter mechanism
    if n_stop:
        stopping_criterion = StoppingCriterion.COUNTER
        size = n_stop - 1
    elif n_gen:
        stopping_criterion = StoppingCriterion.TIMER
        # number corresponding to n_gen generations
        size = 2**n_gen - 1
    else:
        raise ValueError("Specify either n_stop or n_gen")

    # S-phase rate: rho = 1 / tau_s
    delta_s = rho
    # D-phase durations are gamma distributed
    delta_d = 1 / get_gamma(std=std, mu=1 - 1 / rho, size=size)
    # set maximal step-size in time
    delta_t0 = 0.01

    (
        names,
        s_phases,
        timeline,
        resource,
        eta,
        phi,
        active_resource,
        dna_content,
        count,
        s_phases_star,
    ) = propagate_time(
        resource_type=resource_type.value,
        stopping_criterion=stopping_criterion.value,
        ode_method=ode_method.value,
        zeta=zeta,
        k_u=k_u,
        k_b=k_b,
        size=size,
        delta_s=delta_s,
        delta_d=delta_d,
        delta_t0=delta_t0,
        save_f=save_f,
        acceleration=acceleration,
        f_threshold=f_threshold,
        scaling=scaling,
    )

    # Compute overall s* durations from valid entries (start >= 0, end >= 0)
    valid = (s_phases_star[:, 0] >= 0) & (s_phases_star[:, 1] >= 0)
    if valid.sum() >= 1:
        t_star_first = s_phases_star[valid, 0].min()
        # Todo: This is not exactly the time from first s* start to last s* end, but is need to avoid a to early ending of the s* phase due to all nuclei dropping below the f_threshold at some point.
        t_star_last = s_phases[1:].max()
        t_s_star = t_star_last - t_star_first
    else:
        t_s_star = -1.0

    if valid.sum() >= 2:
        starts_sorted = np.sort(s_phases_star[valid, 0])
        t_s_star_2 = t_star_last - starts_sorted[1]
    else:
        t_s_star_2 = -1.0

    results = {
        "names": names,
        "s_phases": s_phases,
        "s*_phases": s_phases_star,
        "time": timeline,
        "r_0": resource,
        "eta": eta,
        "phi": phi,
        "g": dna_content,
        "n": count,
        "t_s*": t_s_star,
        "t_s*_2": t_s_star_2,
    }

    if save_f:
        results["f"] = active_resource.reshape(
            active_resource.size // size, size
        )
    return results
