# -*- coding: utf-8 -*-
"""Tests for lib/shared_resource.py — resource model functions."""

import numpy as np
import pytest

from srs.lib.shared_resource import (
    constant_resource,
    get_resource,
    growing_count_resource,
    pcna1_like_resource,
)


class TestConstantResource:
    def test_returns_zeta(self):
        assert constant_resource(0.5) == 0.5

    def test_zero(self):
        assert constant_resource(0.0) == 0.0


class TestGrowingCountResource:
    def test_linear(self):
        assert growing_count_resource(2.0, 3.0) == 6.0

    def test_zero_count(self):
        assert growing_count_resource(1.0, 0.0) == 0.0

    def test_zero_zeta(self):
        assert growing_count_resource(0.0, 10.0) == 0.0


class TestPcna1LikeResource:
    def test_sigmoid_range(self):
        """Output should be between 0 and zeta for all t."""
        zeta = 2.0
        for t in np.linspace(-5000, 5000, 100):
            val = pcna1_like_resource(t, zeta)
            assert 0 <= val <= zeta + 1e-10

    def test_monotonically_increasing(self):
        zeta = 1.0
        ts = np.linspace(-2000, 2000, 200)
        vals = [pcna1_like_resource(t, zeta) for t in ts]
        assert all(b >= a - 1e-15 for a, b in zip(vals, vals[1:]))

    def test_approaches_zeta_at_large_t(self):
        zeta = 3.0
        val = pcna1_like_resource(1e6, zeta)
        np.testing.assert_allclose(val, zeta, rtol=1e-5)


class TestGetResource:
    def test_const_dispatch(self):
        assert get_resource("const", 0.7, 0, 0) == pytest.approx(0.7)

    def test_zeta_n_dispatch(self):
        assert get_resource("zeta_n", 0.5, 4.0, 0) == pytest.approx(2.0)

    def test_zeta_g_dispatch(self):
        assert get_resource("zeta_g", 0.3, 10.0, 0) == pytest.approx(3.0)

    def test_unknown_type_returns_zero(self):
        assert get_resource("nonexistent", 1.0, 1.0, 1.0) == 0
