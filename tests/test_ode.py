# -*- coding: utf-8 -*-
"""Tests for model3/_core/ode.py — ODE solvers and steady-state."""

import numpy as np

from srs.model3._core.ode import (
    get_activated_fraction_euler,
    model,
    model_steady_state,
)


class TestModel:
    """df/dt = k_b (r_0 - sum(f))(1-f_i) - k_u f_i"""

    def test_zero_fraction_positive_derivative(self):
        """All f_i = 0 and r_0 > 0  ⟹  df/dt > 0."""
        dfdt = model(k_u=1.0, k_b=10.0, r_0=2.0, f_0=np.zeros(3))
        assert (dfdt > 0).all()

    def test_equilibrium_derivative_vanishes(self):
        """At steady state df/dt should be ≈ 0."""
        n = 4
        r_0, k_u, k_b = 2.0, 1.0, 10.0
        f_eq = model_steady_state(r_0, k_u / k_b, n)
        f_0 = np.full(n, f_eq)
        dfdt = model(k_u, k_b, r_0, f_0)
        np.testing.assert_allclose(dfdt, 0, atol=1e-10)

    def test_single_nucleus(self):
        """Manual check for single nucleus:
        df/dt = k_b(r_0 - f)(1-f) - k_u f"""
        f_0 = np.array([0.3])
        k_u, k_b, r_0 = 0.5, 2.0, 1.0
        expected = k_b * (r_0 - 0.3) * (1 - 0.3) - k_u * 0.3
        dfdt = model(k_u, k_b, r_0, f_0)
        np.testing.assert_allclose(dfdt[0], expected)


class TestModelSteadyState:
    def test_single_nucleus_analytic(self):
        """For n=1, k_u/k_b=k, steady state f solves
        f^2 - (r_0+k+1)f + r_0 = 0, taking the smaller root."""
        r_0, k = 2.0, 0.1
        a = r_0 + k + 1
        expected = (a - np.sqrt(a**2 - 4 * r_0)) / 2
        np.testing.assert_allclose(model_steady_state(r_0, k, 1), expected)

    def test_large_resource_approaches_one(self):
        """With abundant resource and fast binding, f_eq → 1."""
        f_eq = model_steady_state(r_0=1000.0, k=0.001, n_nuclei=1)
        assert f_eq > 0.99

    def test_scarce_resource(self):
        """With little resource and many nuclei, f_eq should be small."""
        f_eq = model_steady_state(r_0=0.1, k=1.0, n_nuclei=10)
        assert f_eq < 0.05

    def test_symmetry(self):
        """Steady state for n nuclei sharing r_0 equals that for 1 nucleus
        sharing r_0/n only when k=0 (perfect binding)."""
        f_n = model_steady_state(r_0=2.0, k=0.0, n_nuclei=4)
        # With k=0: f = (r_0/n + 1 - sqrt((r_0/n+1)^2 - 4r_0/n))/2
        # = (r_0/n + 1 - |r_0/n - 1|)/2 = min(r_0/n, 1)
        np.testing.assert_allclose(f_n, min(2.0 / 4, 1.0), atol=1e-10)


class TestEulerIntegrator:
    def test_fraction_increases_from_zero(self):
        """Starting at f=0 with positive resource, fractions should increase."""
        f_0 = np.zeros(2)
        dt, f_1 = get_activated_fraction_euler(
            k_u=1.0, k_b=20.0, r_0=2.0, f_0=f_0, delta_t0=1.0
        )
        assert dt > 0
        assert (f_1 >= f_0).all()
        assert (f_1 > 0).all()

    def test_mass_conservation(self):
        """sum(f) should not exceed r_0 after one step."""
        f_0 = np.array([0.4, 0.3, 0.2])
        _, f_1 = get_activated_fraction_euler(
            k_u=0.1, k_b=50.0, r_0=1.0, f_0=f_0, delta_t0=1.0
        )
        assert f_1.sum() <= 1.0 + 1e-10

    def test_clamp_to_unit_interval(self):
        """Activated fractions must stay in [0, 1]."""
        f_0 = np.array([0.99, 0.005])
        _, f_1 = get_activated_fraction_euler(
            k_u=0.01, k_b=100.0, r_0=5.0, f_0=f_0, delta_t0=10.0
        )
        assert (f_1 >= 0).all()
        assert (f_1 <= 1).all()

    def test_relaxation_towards_equilibrium(self):
        """After many small steps, fractions should converge to steady state."""
        n, r_0, k_u, k_b = 3, 2.0, 1.0, 20.0
        f = np.zeros(n)
        for _ in range(5000):
            _, f = get_activated_fraction_euler(k_u, k_b, r_0, f, delta_t0=0.5)
        f_eq = model_steady_state(r_0, k_u / k_b, n)
        np.testing.assert_allclose(f, f_eq, rtol=0.02)
