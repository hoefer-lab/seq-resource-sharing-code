# -*- coding: utf-8 -*-
"""Tests for model3/_core/distributions.py — Gamma distribution helpers."""

import numpy as np
import pytest

from srs.model3._core.distributions import get_d_cv, get_gamma, get_scale_and_shape


class TestGetScaleAndShape:
    def test_known_values(self):
        """cv = 0.5, mean = 10  ⟹  k = 4, theta = 2.5."""
        theta, k = get_scale_and_shape(mean=10.0, cv=0.5)
        np.testing.assert_allclose(k, 4.0)
        np.testing.assert_allclose(theta, 2.5)

    def test_mean_equals_k_times_theta(self):
        theta, k = get_scale_and_shape(mean=7.0, cv=0.3)
        np.testing.assert_allclose(k * theta, 7.0)

    def test_cv_equals_one_over_sqrt_k(self):
        cv = 0.4
        theta, k = get_scale_and_shape(mean=5.0, cv=cv)
        np.testing.assert_allclose(1 / np.sqrt(k), cv)

    def test_small_cv_gives_large_shape(self):
        _, k = get_scale_and_shape(mean=1.0, cv=0.01)
        assert k == pytest.approx(10000.0)


class TestGetGamma:
    def test_deterministic_when_std_zero(self):
        samples = get_gamma(std=0, mu=5.0, size=10)
        np.testing.assert_array_equal(samples, 5.0)

    def test_output_shape(self):
        samples = get_gamma(std=1.0, mu=5.0, size=100)
        assert samples.shape == (100,)

    def test_positive_samples(self):
        np.random.seed(42)
        samples = get_gamma(std=2.0, mu=5.0, size=1000)
        assert (samples > 0).all()

    def test_mean_close_to_mu(self):
        np.random.seed(0)
        samples = get_gamma(std=1.0, mu=10.0, size=50000)
        np.testing.assert_allclose(samples.mean(), 10.0, rtol=0.02)


class TestGetDCv:
    def test_deterministic_when_cv_zero(self):
        samples = get_d_cv(cv=0, tau_d=3.0, size=5)
        np.testing.assert_array_equal(samples, 3.0)

    def test_mean_close_to_tau_d(self):
        np.random.seed(1)
        samples = get_d_cv(cv=0.3, tau_d=8.0, size=50000)
        np.testing.assert_allclose(samples.mean(), 8.0, rtol=0.02)
