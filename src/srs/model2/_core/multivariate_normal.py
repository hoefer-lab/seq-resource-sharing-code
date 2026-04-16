# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Multivariate normal with partitioning and conditional-distribution methods."""

import numpy as np


class MultivariateNormal:
    """
    Class of multivariate normal distribution.

    Parameters
    ----------
    mean: ndarray(float, dim=1)
        the mean of z, N by 1
    cov: ndarray(float, dim=2)
        the covarianece matrix of z, N by 1

    Arguments
    ---------
    mean, cov:
        see parameters
    means: list(ndarray(float, dim=1))
        list of mean vectors mean1 and mean2 in order
    covs: list(list(ndarray(float, dim=2)))
        2 dimensional list of covariance matrices
        cov11, cov12, cov21, cov22 in order
    betas: list(ndarray(float, dim=1))
        list of regression coefficients beta1 and beta2 in order
    """

    def __init__(self, mean, cov):
        """Initialize."""
        self.mean = np.array(mean)
        self.cov = np.atleast_2d(cov)

    def partition(self, k):
        """Partition vector z.

        Given k, partition the random vector z into a size k vector z1
        and a size N-k vector z2. Partition the mean vector mean into
        mean1 and mean2, and the covariance matrix cov into cov11, cov12,
        cov21, cov22 correspondingly. Compute the regression coefficients
        beta1 and beta2 using the partitioned arrays.

        """
        mean = self.mean
        cov = self.cov

        self.means = [mean[:k], mean[k:]]
        self.covs = [[cov[:k, :k], cov[:k, k:]], [cov[k:, :k], cov[k:, k:]]]

        self.betas = [
            self.covs[0][1] @ np.linalg.inv(self.covs[1][1]),
            self.covs[1][0] @ np.linalg.inv(self.covs[0][0]),
        ]

    def cond_dist(self, ind, z):
        """Compute conditional distribution given z.

        Compute the conditional distribution of z1 given z2, or reversely.
        Argument ind determines whether we compute the conditional
        distribution of z1 (ind=0) or z2 (ind=1).

        Returns
        ---------
        mean_hat: ndarray(float, ndim=1)
            The conditional mean of z1 or z2.
        cov_hat: ndarray(float, ndim=2)
            The conditional covariance matrix of z1 or z2.
        """
        beta = self.betas[ind]
        means = self.means
        covs = self.covs

        mean_hat = means[ind] + beta @ (z - means[1 - ind])
        cov_hat = covs[ind][ind] - beta @ covs[1 - ind][1 - ind] @ beta.T

        return mean_hat, cov_hat
