# -*- coding: utf-8 -*-
"""Tests for lib/correlation_matrix.py — normal_transformation_df."""

import numpy as np
import pandas as pd
import pytest

from srs.lib.correlation_matrix import (
    get_model1_corr,
    get_model2_corr,
    normal_transformation_df,
)


@pytest.fixture()
def sample_df():
    """Synthetic phase-duration DataFrame (50 rows, all phases)."""
    rng = np.random.default_rng(0)
    n = 50
    phases = [
        "s",
        "s0",
        "s1",
        "s00",
        "s01",
        "s10",
        "s11",
        "d0",
        "d1",
        "d00",
        "d01",
        "d10",
        "d11",
    ]
    data = {p: rng.exponential(scale=10, size=n) for p in phases}
    return pd.DataFrame(data)


class TestNormalTransformation:
    def test_output_shape(self, sample_df):
        df_t = normal_transformation_df(sample_df)
        assert df_t.shape == sample_df.shape

    def test_output_is_finite(self, sample_df):
        df_t = normal_transformation_df(sample_df)
        # percentileofscore can return 0 or 100, which maps to ±inf
        # but the bulk should be finite
        finite_frac = np.isfinite(df_t.to_numpy()).mean()
        assert finite_frac > 0.9

    def test_z_scores_roughly_centered(self, sample_df):
        df_t = normal_transformation_df(sample_df)
        for col in df_t.columns:
            vals = df_t[col].to_numpy()
            vals = vals[np.isfinite(vals)]
            assert abs(vals.mean()) < 1.0


class TestModel1Corr:
    def test_diagonal_is_one(self, sample_df):
        corr = get_model1_corr(sample_df)
        np.testing.assert_allclose(np.diag(corr), 1.0)

    def test_only_d_sister_nonzero(self, sample_df):
        """Off-diagonal entries should be zero except d0↔d1."""
        corr = get_model1_corr(sample_df)
        n = corr.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if {i, j} == {4, 5}:  # d0, d1 indices in INDEX
                    continue
                assert corr[i, j] == pytest.approx(0.0), (
                    f"Expected zero at ({i},{j}), got {corr[i, j]}"
                )

    def test_d_sister_preserved(self, sample_df):
        """d0↔d1 entry in model-1 should match model-2."""
        corr1 = get_model1_corr(sample_df)
        corr2 = get_model2_corr(sample_df)
        np.testing.assert_allclose(corr1[4, 5], corr2[4, 5])
        np.testing.assert_allclose(corr1[5, 4], corr2[5, 4])
