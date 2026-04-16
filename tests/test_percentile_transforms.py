# -*- coding: utf-8 -*-
"""Tests for model2/_core/distribution.py — percentile transforms."""

import numpy as np
import pytest

from srs.model2._core.distribution import p_to_sd, sd_to_p


class TestSdToP:
    def test_median_gives_half(self):
        data = np.arange(1, 101, dtype=float)
        ps = sd_to_p([data], np.array([50.0]))
        # percentileofscore('mean') for 50 in 1..100 → ~50%
        assert 0.45 < ps[0] < 0.55

    def test_minimum_gives_low_percentile(self):
        data = np.arange(1, 101, dtype=float)
        ps = sd_to_p([data], np.array([1.0]))
        assert ps[0] < 0.05

    def test_maximum_gives_high_percentile(self):
        data = np.arange(1, 101, dtype=float)
        ps = sd_to_p([data], np.array([100.0]))
        assert ps[0] > 0.95


class TestPToSd:
    def test_median_returns_middle(self):
        data = np.arange(1.0, 101.0)
        sd = p_to_sd([data], [0.5])
        assert sd[0] == pytest.approx(50.0, abs=1)

    def test_low_percentile_returns_low_value(self):
        data = np.arange(1.0, 101.0)
        sd = p_to_sd([data], [0.1])
        assert sd[0] <= 15

    def test_high_percentile_returns_high_value(self):
        data = np.arange(1.0, 101.0)
        sd = p_to_sd([data], [0.9])
        assert sd[0] >= 85


class TestRoundTrip:
    def test_sd_to_p_to_sd_approx_identity(self):
        """Converting sd→p→sd should approximately recover the original."""
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=10, size=500)
        original = np.percentile(data, 30)
        ps = sd_to_p([data], np.array([original]))
        recovered = p_to_sd([data], ps)
        np.testing.assert_allclose(recovered[0], original, atol=1.0)
