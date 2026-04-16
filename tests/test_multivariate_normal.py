# -*- coding: utf-8 -*-
"""Tests for model2/_core/multivariate_normal.py — conditional MVN."""

import numpy as np
import pytest

from srs.model2._core.multivariate_normal import MultivariateNormal


class TestMultivariateNormal:
    @pytest.fixture()
    def mvn_2d(self):
        """Simple 2-D correlated normal: ρ = 0.6."""
        mean = [1.0, 2.0]
        cov = [[1.0, 0.6], [0.6, 1.0]]
        mvn = MultivariateNormal(mean, cov)
        mvn.partition(1)
        return mvn

    @pytest.fixture()
    def mvn_4d(self):
        """4-D with block structure."""
        mean = [0.0, 1.0, 2.0, 3.0]
        cov = np.eye(4)
        cov[0, 2] = cov[2, 0] = 0.5
        cov[1, 3] = cov[3, 1] = 0.3
        mvn = MultivariateNormal(mean, cov)
        mvn.partition(2)
        return mvn

    def test_partition_shapes(self, mvn_4d):
        assert mvn_4d.means[0].shape == (2,)
        assert mvn_4d.means[1].shape == (2,)
        assert mvn_4d.covs[0][0].shape == (2, 2)
        assert mvn_4d.covs[0][1].shape == (2, 2)

    def test_cond_mean_uncorrelated(self):
        """If off-diag blocks are zero, conditional mean == marginal mean."""
        mvn = MultivariateNormal([1, 2, 3], np.eye(3))
        mvn.partition(1)
        mean_hat, cov_hat = mvn.cond_dist(0, np.array([5.0, 10.0]))
        np.testing.assert_allclose(mean_hat, [1.0])
        np.testing.assert_allclose(cov_hat, [[1.0]])

    def test_cond_mean_2d(self, mvn_2d):
        """z1 | z2 = 3 should shift the mean by β(z2 - μ2)."""
        z2 = np.array([3.0])
        mean_hat, _ = mvn_2d.cond_dist(0, z2)
        # β = Σ12 Σ22⁻¹ = 0.6/1 = 0.6
        expected = 1.0 + 0.6 * (3.0 - 2.0)
        np.testing.assert_allclose(mean_hat, [expected])

    def test_cond_cov_2d(self, mvn_2d):
        """Conditional variance = σ11 - β σ21 = 1 - 0.6*0.6 = 0.64."""
        _, cov_hat = mvn_2d.cond_dist(0, np.array([3.0]))
        np.testing.assert_allclose(cov_hat, [[0.64]])

    def test_cond_dist_symmetry(self, mvn_4d):
        """Conditioning on z2 then z1 should give different means."""
        z_test = np.array([1.0, 1.0])
        mean0, _ = mvn_4d.cond_dist(0, z_test)
        mean1, _ = mvn_4d.cond_dist(1, z_test)
        assert not np.allclose(mean0, mean1)

    def test_cond_cov_positive_semidefinite(self, mvn_4d):
        """Conditional covariance matrix must have non-negative eigenvalues."""
        _, cov_hat = mvn_4d.cond_dist(0, np.array([1.0, 1.0]))
        eigvals = np.linalg.eigvalsh(cov_hat)
        assert (eigvals >= -1e-12).all()
