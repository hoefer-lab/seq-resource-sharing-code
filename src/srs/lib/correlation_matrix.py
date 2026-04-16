# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Kinship correlation matrices for S- and D-phase durations.

Computes Pearson correlation matrices (with shuffle correction for
D-sister pairs) from experimental phase-duration DataFrames.  Provides
the full model-2 correlation matrix (:func:`get_model2_corr`), the
model-1 independent subset (:func:`get_model1_corr`), and pooled
correlation plotting helpers.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr

s = ["s"]
s2 = 2 * s
s4 = 4 * s
sx = ["s0", "s1"]
sxj = ["s1", "s0"]
sx2 = ["s0", "s0", "s1", "s1"]
sxx = ["s00", "s01", "s10", "s11"]

dx = ["d0", "d1"]
dxj = ["d1", "d0"]
dx2 = ["d0", "d0", "d1", "d1"]
dxj2 = ["d1", "d1", "d0", "d0"]
dxx = ["d00", "d01", "d10", "d11"]
dxxj = ["d01", "d00", "d11", "d10"]

ALL_PHASES = [
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
INDEX = ["d", "s", "s0", "s1", "d0", "d1"]


def get_raw_corr_matrix(df, normal_transformed=True, dropna=True):
    """Compute the full kinship correlation and p-value matrices.

    Parameters
    ----------
    df : pd.DataFrame
        Phase-duration DataFrame with columns s, s0, …, d0, d1, …
    normal_transformed : bool
        Apply rank-based normal transformation before computing correlations.
    dropna : bool
        Drop rows with NaN before computing each pair.

    Returns
    -------
    dict
        ``'corr'``, ``'corr_sig'`` (significant only), ``'pval'`` DataFrames.
    """
    index = [
        "s",
        "dx",
        "dxj",
        "sx",
        "dxx",
        "dxxj",
        "sxx",
        "sister",
        "mother",
        "cousins",
        "aunts",
        "grand-\nmother",
    ]
    df_corr = pd.DataFrame(
        np.zeros((len(index), len(index))) * np.nan, index=index, columns=index
    )

    df = df.copy()
    # In case of limited number of trees, use pooled log-transformed data
    if normal_transformed:
        df = normal_transformation_df(df)
    df["nan"] = np.nan

    df_pval = df_corr.copy()
    forloop_dict = {
        "s": [
            ["dx", s2, dx],
            ["sx", s2, sx],
            ["dxx", s4, dxx],
            ["sxx", s4, sxx],
        ],
        "dx": [
            ["dx", ["d0"], ["d1"]],
            ["sx", dx, sx],
            ["dxx", dx2, dxx],
            ["sxx", dx2, sxx],
            ["sister", ["d0"], ["d1"]],
        ],
        "sx": [
            ["sx", ["s0"], ["s1"]],
            ["dxx", sx2, dxx],
            ["sxx", sx2, sxx],
            ["sister", ["s0"], ["s1"]],
            ["mother", s2, sx],
        ],
        "dxx": [
            ["dxx", ["d00", "d10"], ["d01", "d11"]],
            ["sxx", dxx, sxx],
            ["sister", ["d00", "d10"], ["d01", "d11"]],
            ["mother", dx2, dxx],
            [
                "cousins",
                ["d00", "d00", "d01", "d01"],
                ["d10", "d11", "d10", "d11"],
            ],
            ["aunts", ["d1", "d1", "d0", "d0"], dxx],
        ],
        "sxx": [
            ["sxx", ["s00", "s10"], ["s01", "s11"]],
            ["sister", ["s00", "s10"], ["s01", "s11"]],
            ["mother", sx2, sxx],
            [
                "cousins",
                ["s00", "s00", "s01", "s01"],
                ["s10", "s11", "s10", "s11"],
            ],
            ["aunts", ["s1", "s1", "s0", "s0"], sxx],
            ["grand-\nmother", s4, sxx],
        ],
        "dxj": [
            ["s", s2, dxj],
            ["dx", ["nan"], ["nan"]],
            ["sx", sx, dxj],
            ["dxx", ["nan"], ["nan"]],
            ["sxx", sxx, dxj2],
        ],
        "dxxj": [
            ["s", s4, dxxj],
            ["dx", dx2, dxxj],
            ["sx", sx2, dxxj],
            ["dxx", ["nan"], ["nan"]],
            ["sxx", sxx, dxxj],
        ],
    }

    for key_x, values in forloop_dict.items():
        for key_y, x_list, y_list in values:
            if dropna:
                df_dropped = df[
                    np.unique(np.concatenate([x_list, y_list]))
                ].dropna()
                x = df_dropped[x_list].values.flatten()
                y = df_dropped[y_list].values.flatten()
            else:
                x = df[x_list].values.flatten()
                y = df[y_list].values.flatten()
                mask = np.isnan(x) | np.isnan(y)
                x, y = x[~mask], y[~mask]

            if len(x) < 2 or len(y) < 2:
                corr, pval = 0, 1
            else:
                corr, pval = pearsonr(x, y)

            # in case of d-d correlations, the correlation has a strong bias
            # towards higher r, due to the effect of sorting. To get ride of it
            # correct corr and pval for x_list in [['d0'], ['d00', 'd10']]
            # if
            if (x_list in [["d0"], ["d00", "d10"]]) and (
                y_list in [["d1"], ["d01", "d11"]]
            ):
                corr, pval = get_shuffled_corr(x, y)

            df_corr.at[key_y, key_x] = corr
            df_corr.at[key_x, key_y] = corr
            df_pval.at[key_y, key_x] = pval
            df_pval.at[key_x, key_y] = pval
    df_corr_sig = df_corr.copy()
    df_corr_sig[df_pval >= 0.05] = 0
    return {
        "corr": df_corr,
        "corr_sig": df_corr_sig,
        "pval": df_pval,
    }


def get_corr_matrix_plot(df, relatives=False, dropna=False):
    """Compute a 4×4 (or 4×5 for relatives) correlation sub-matrix for plotting.

    Parameters
    ----------
    df : pd.DataFrame
        Phase-duration DataFrame.
    relatives : bool
        If *True*, columns are kinship relations; otherwise phase-phase.
    dropna : bool
        Drop NaN rows.

    Returns
    -------
    dict
        ``'corr'``, ``'pval'`` DataFrames, plus ``'rows'`` and ``'columns'``.
    """
    res = get_raw_corr_matrix(df, normal_transformed=True, dropna=dropna)
    raw_rows = ["dx", "sx", "dxx", "sxx"]
    rows = [r"D$_{i}$", r"S$_{i}$", r"D$_{ij}$", r"S$_{ij}$"]

    if relatives:
        # Create empty DataFrame filled with NaNs
        columns = ["sister", "mother", "cousins", "aunts", "grand-\nmother"]
    else:
        columns = ["s", "dx", "sx", "dxx"]

    df_corr = res["corr"].loc[raw_rows, columns]
    df_pval = res["pval"].loc[raw_rows, columns]

    df_corr.index = rows
    df_pval.index = rows

    if not relatives:
        columns = [r"S", r"D$_{i}$", r"S$_{i}$", r"D$_{ij}$"]
        df_corr.columns = columns
        df_pval.columns = columns
        for df_tmp in [df_corr, df_pval]:
            df_tmp.values[np.triu_indices_from(df_tmp, k=1)] = np.nan
    return {"corr": df_corr, "pval": df_pval, "rows": rows, "columns": columns}


def get_corr_matrix_plot_pooled(df, dropna=False):
    """Compute a 2×5 correlation matrix using pooled (subtree) data.

    Rows: ["S", "D"]
    Columns: ["sister", "mother", "cousins", "aunts", "grand-\\nmother"]

    For sister and mother the data is pooled across subtrees via get_xy_pairs.
    For cousins, aunts, and grandmother the pairs are the same as in
    get_raw_corr_matrix (i.e. from the deepest observed generation only).
    """
    df_transformed = normal_transformation_df(df)
    df_transformed["nan"] = np.nan

    rows = [r"S", r"D"]
    columns = ["sister", "mother", "cousins", "aunts", "grand-\nmother"]

    df_corr = pd.DataFrame(
        np.full((len(rows), len(columns)), np.nan),
        index=rows,
        columns=columns,
    )
    df_pval = df_corr.copy()

    # --- helper to compute correlation from explicit column lists ----------
    def _corr_from_cols(x_list, y_list, shuffle=False):
        if dropna:
            df_dropped = df_transformed[
                np.unique(np.concatenate([x_list, y_list]))
            ].dropna()
            x = df_dropped[x_list].values.flatten()
            y = df_dropped[y_list].values.flatten()
        else:
            x = df_transformed[x_list].values.flatten()
            y = df_transformed[y_list].values.flatten()
            mask = np.isnan(x) | np.isnan(y)
            x, y = x[~mask], y[~mask]
        if len(x) < 2 or len(y) < 2:
            return 0.0, 1.0
        if shuffle:
            return get_shuffled_corr(x, y)
        return pearsonr(x, y)

    # --- helper to compute correlation from pooled subtree pairs -----------
    def _corr_from_xy_pairs(row_idx, col_idx, shuffle=False):
        x, y = get_xy_pairs(df_transformed, row_idx, col_idx)
        if len(x) < 2 or len(y) < 2:
            return 0.0, 1.0
        if shuffle:
            return get_shuffled_corr(x, y)
        return pearsonr(x, y)

    # ---- S row ------------------------------------------------------------
    # sister  (pooled): s0↔s1, s00↔s01, s10↔s11
    c, p = _corr_from_xy_pairs("s0", "s1")
    df_corr.at[r"S", "sister"] = c
    df_pval.at[r"S", "sister"] = p

    # mother  (pooled): s↔s0/s1, s0↔s00/s01, s1↔s10/s11
    c, p = _corr_from_xy_pairs("s", "s0")
    df_corr.at[r"S", "mother"] = c
    df_pval.at[r"S", "mother"] = p

    # cousins: same as sxx-cousins in get_raw_corr_matrix
    c, p = _corr_from_cols(
        ["s00", "s00", "s01", "s01"], ["s10", "s11", "s10", "s11"]
    )
    df_corr.at[r"S", "cousins"] = c
    df_pval.at[r"S", "cousins"] = p

    # aunts: same as sxx-aunts
    c, p = _corr_from_cols(
        ["s1", "s1", "s0", "s0"], ["s00", "s01", "s10", "s11"]
    )
    df_corr.at[r"S", "aunts"] = c
    df_pval.at[r"S", "aunts"] = p

    # grandmother: same as sxx-grandmother
    c, p = _corr_from_cols(["s", "s", "s", "s"], ["s00", "s01", "s10", "s11"])
    df_corr.at[r"S", "grand-\nmother"] = c
    df_pval.at[r"S", "grand-\nmother"] = p

    # ---- D row ------------------------------------------------------------
    # sister  (pooled, with shuffle correction): d0↔d1, d00↔d01, d10↔d11
    c, p = _corr_from_xy_pairs("d0", "d1", shuffle=True)
    df_corr.at[r"D", "sister"] = c
    df_pval.at[r"D", "sister"] = p

    # mother  (pooled): d0↔d00/d01, d1↔d10/d11
    c, p = _corr_from_xy_pairs("d", "d0")
    df_corr.at[r"D", "mother"] = c
    df_pval.at[r"D", "mother"] = p

    # cousins: same as dxx-cousins
    c, p = _corr_from_cols(
        ["d00", "d00", "d01", "d01"], ["d10", "d11", "d10", "d11"]
    )
    df_corr.at[r"D", "cousins"] = c
    df_pval.at[r"D", "cousins"] = p

    # aunts: same as dxx-aunts
    c, p = _corr_from_cols(
        ["d1", "d1", "d0", "d0"], ["d00", "d01", "d10", "d11"]
    )
    df_corr.at[r"D", "aunts"] = c
    df_pval.at[r"D", "aunts"] = p

    # grandmother: not available for D phases
    # df_corr / df_pval remain NaN

    # Transpose so kinship relations are rows and phases are columns
    df_corr = df_corr.T
    df_pval = df_pval.T
    rows, columns = columns, rows

    return {"corr": df_corr, "pval": df_pval, "rows": rows, "columns": columns}


def get_shuffled_corr(x, y):
    """
    Calculate the correlation between shuffled pairs of x and y.

    This function shuffles the pairs of x and y, then calculates the Pearson
    correlation coefficient and p-value for the shuffled data.

    Parameters:
        x (array-like): First input array.
        y (array-like): Second input array.

    Returns:
        tuple: Pearson correlation coefficient and p-value.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    rng = np.random.default_rng()
    m = len(x)
    if m < 1000:
        n = 10000
    else:
        n = 1000

    # estimate mean correlation of shuffled xi-yi pairs
    xy = np.swapaxes(np.array([x, y]), 0, 1)
    xys = rng.choice(xy, (n, m))
    # shuffle xi-yi pairs
    swap_mask = np.random.choice([True, False], (n, m))
    xys[swap_mask] = xys[swap_mask][:, ::-1]
    corr = pearsonr(xys[:, :, 0], xys[:, :, 1], axis=1)[0].mean()

    # estimate mean correlation of randomly assigned x and y (break xi-yi pairs)
    xys_random = np.random.choice(xy.flatten(), (n, m, 2))
    corr_distr = pearsonr(xys_random[:, :, 0], xys_random[:, :, 1], axis=1)[0]
    pval = (np.sum(corr_distr >= corr) + 1) / (n + 1)

    return corr, pval


def normal_transformation_df(df):
    """Quantile-transforms each column of a DataFrame to standard normal
    distribution.

        The function groups columns of the DataFrame based on predefined phases ('s', 'd0', 'd1', 's0', 's1', 's00', 's01', 's10', 's11', 'd00', 'd01', 'd10', 'd11') and calculates the percentile of each value within its respective group. It then converts this percentile to a z-score using the standard normal distribution's percent point function (PPF).

        Args:
            df (pd.DataFrame): The input DataFrame to be transformed.  The columns of the DataFrame should correspond to the phases used for grouping.

        Returns:
            pd.DataFrame: A new DataFrame with the same index as the input, where each value has been transformed to a z-score representing its position within a standard normal distribution, based on its original percentile within its group.

        Raises:
            AssertionError: If an unknown phase is encountered in the DataFrame columns.
    """
    s = ["s"]
    dx = ["d0", "d1"]
    sx = ["s0", "s1", "s00", "s01", "s10", "s11"]
    dxx = ["d00", "d01", "d10", "d11"]

    df_dict = {}
    for x in ALL_PHASES:
        if x == "s":
            x_group = s
        elif x in dx:
            x_group = dx
        elif x in sx:
            x_group = sx
        elif x in dxx:
            x_group = dxx
        else:
            assert False, f"Unknown phase {x}"

        p = (
            stats.percentileofscore(
                score=df[x].to_numpy(),
                a=df[x_group].to_numpy().flatten(),
                kind="mean",
                nan_policy="omit",
            )
            / 100
        )
        x_normal = stats.norm().ppf(q=p)
        df_dict[x] = x_normal
    df_normal = pd.DataFrame(df_dict, index=df.index)
    return df_normal


def get_model2_corr(df):
    """Compute a 6×6 phase correlation matrix for model-2 comparison.

    Applies normal transformation, then computes Pearson correlations
    (with shuffle correction for D-D sister pairs) across the INDEX
    phases ``['d', 's', 's0', 's1', 'd0', 'd1']``.

    Parameters
    ----------
    df : pd.DataFrame
        Phase-duration DataFrame.

    Returns
    -------
    np.ndarray
        Symmetric 6×6 correlation matrix (insignificant entries set to 0).
    """
    df = normal_transformation_df(df)
    corr_matrix = np.zeros((len(INDEX), len(INDEX)))
    for i in range(len(INDEX)):
        for j in np.arange(i, len(INDEX)):
            x, y = get_xy_pairs(df, INDEX[i], INDEX[j])
            if len(x) < 2 or len(y) < 2:
                corr, pval = 0, 1
            else:
                corr, pval = pearsonr(x, y)

            # in case of d-d correlations, the correlation has a strong bias
            # towards higher r, due to the effect of sorting. To get ride of it
            # correct corr and pval for x_list in [['d0'], ['d00', 'd10']]
            # if
            if (INDEX[i] in ["d0", "d1"]) and (INDEX[j] in ["d0", "d1"]):
                corr, pval = get_shuffled_corr(x, y)

            # set all unsignificant correlations to 0
            if pval >= 0.05:
                corr = 0

            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    return corr_matrix


def get_model1_corr(df):
    """Compute the model-1 (independent) correlation matrix.

    Model 1 is a subset of model 2 where kinship inheritance is removed:
    all off-diagonal correlations are zeroed out *except* the D-sister
    correlation (d0 ↔ d1), which arises from the shared biological
    constraint that sisters must pass through a common division event.

    Parameters
    ----------
    df : pd.DataFrame
        Phase-duration DataFrame (same format as for :func:`get_model2_corr`).

    Returns
    -------
    np.ndarray
        Symmetric 6×6 correlation matrix (identity + D-sister entry).
    """
    corr_full = get_model2_corr(df)
    corr_matrix = np.eye(len(INDEX))

    # Keep only the D-sister correlation (d0 ↔ d1).
    i_d0 = INDEX.index("d0")
    i_d1 = INDEX.index("d1")
    corr_matrix[i_d0, i_d1] = corr_full[i_d0, i_d1]
    corr_matrix[i_d1, i_d0] = corr_full[i_d1, i_d0]

    return corr_matrix


def get_xy_pairs(df_exp, row, col):
    """Extract pooled (x, y) pairs across all observed subtree generations.

    Pools matching phase pairs from generation 0 (s↔s0/s1), generation 1
    (s0↔s00/s01, …), etc., so that correlation estimates use all available
    data rather than only the deepest generation.

    Parameters
    ----------
    df_exp : pd.DataFrame
        Normal-transformed phase-duration DataFrame.
    row, col : str
        Phase identifiers from INDEX (e.g. ``'s0'``, ``'d1'``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Paired arrays ``(x, y)`` with NaN rows dropped.
    """
    df = df_exp.copy()
    df["nan"] = np.nan

    raw_clmns = np.array(
        [
            ["nan", "s", "s0", "s1", "d0", "d1"],
            ["d0", "s0", "s00", "s01", "d00", "d01"],
            ["d1", "s1", "s10", "s11", "d10", "d11"],
        ]
    )

    if row[:-1] != col[:-1]:
        raw_clmns = np.array(
            [
                ["nan", "s", "s0", "s1", "d0", "d1"],
                ["nan", "s", "s1", "s0", "d1", "d0"],
                ["d0", "s0", "s00", "s01", "d00", "d01"],
                ["d0", "s0", "s01", "s00", "d01", "d00"],
                ["d1", "s1", "s10", "s11", "d10", "d11"],
                ["d1", "s1", "s11", "s10", "d11", "d10"],
            ]
        )

    df_list = []
    for raw_clmn in raw_clmns:
        df_list.append(
            df[raw_clmn]
            .copy()
            .rename(
                columns={raw_clmn[i]: INDEX[i] for i in range(len(raw_clmn))}
            )
        )
    df_x = pd.concat(df_list, ignore_index=True)

    return np.swapaxes(df_x[[row, col]].dropna().to_numpy(), 0, 1)
