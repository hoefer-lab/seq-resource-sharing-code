# -*- coding: utf-8 -*-
"""Tests for model3/_tools.py — numerical helpers."""

import numpy as np
import pytest

from srs.model3._tools import (
    filtered_signal,
    get_mean_std_dict,
    get_sliding_mean,
    get_t_max,
)


class TestGetMeanStdDict:
    def test_1d(self):
        x = np.array([2.0, 4.0, 6.0])
        d = get_mean_std_dict(x)
        np.testing.assert_allclose(d["mean"], 4.0)
        np.testing.assert_allclose(d["std"], np.std([2, 4, 6]))

    def test_2d(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        d = get_mean_std_dict(x)
        np.testing.assert_allclose(d["mean"], [2.0, 3.0])
        np.testing.assert_allclose(d["std"], [1.0, 1.0])


class TestFilteredSignal:
    def test_preserves_length(self):
        y = np.random.default_rng(0).normal(size=200)
        assert filtered_signal(y).shape == y.shape

    def test_reduces_high_freq_noise(self):
        t = np.linspace(0, 1, 500)
        low = np.sin(2 * np.pi * 2 * t)
        noise = 0.5 * np.sin(2 * np.pi * 100 * t)
        yf = filtered_signal(low + noise)
        # Filtered signal should be much closer to the low-freq component
        residual = np.std(yf - low)
        assert residual < 0.3

    def test_constant_signal_unchanged(self):
        y = 5.0 * np.ones(100)
        np.testing.assert_allclose(filtered_signal(y), y, atol=1e-10)


class TestGetSlidingMean:
    def test_constant_function(self):
        x = np.linspace(0, 10, 500)
        y = 3.0 * np.ones_like(x)
        x_out, y_out = get_sliding_mean(x, y, x_max=np.float64(10))
        np.testing.assert_allclose(y_out, 3.0, atol=1e-10)

    def test_linear_function(self):
        x = np.linspace(0, 5, 1000)
        y = 2 * x
        x_out, y_out = get_sliding_mean(x, y, x_max=np.float64(5))
        # Each bin mean should ≈ 2*x_center
        np.testing.assert_allclose(y_out, 2 * x_out, atol=0.3)


class TestGetTMax:
    def test_simple(self):
        results = [
            [{"time": np.array([0, 1, 2, 10])}, {"time": np.array([0, 1, 5])}],
            [{"time": np.array([0, 1, 2, 3, 7])}],
        ]
        assert get_t_max(results) == pytest.approx(5.0)

    def test_single_realisation(self):
        results = [[{"time": np.array([0.0, 0.5, 1.0])}]]
        assert get_t_max(results) == pytest.approx(1.0)
