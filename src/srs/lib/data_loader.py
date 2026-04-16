# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2026, Patrick Binder
# All rights reserved.
"""Unified data-loading module.

Combines loading routines for the two experimental datasets:

* **Klaus** — spinning-disk confocal data from Klaus *et al.* 2022
  (Figs. 2 and 6).
* **Airyscan** — Zeiss LSM900 / Airyscan 2 time-lapse data (Fig. S7).
  Nuclear cycles are ~33 %% slower than in the Klaus dataset, likely
  due to increased phototoxicity of the Airyscan setup.

Also provides colour dictionaries, ``convert_to_df`` for simulation
output, and the main ``get_data`` dispatcher.
"""

import os
import warnings

import numpy as np
import pandas as pd

from .correlation_matrix import ALL_PHASES
from .enums import DataSet, DropData

DATAPATH = os.path.join(os.path.dirname(__file__), "..", "data") + os.sep


# ---------------------------------------------------------------------------
# Klaus data helpers
# ---------------------------------------------------------------------------


def _klaus_get_total_pcna1_growth():
    """Load and normalise PCNA1-GFP and NLS-mCherry intensity traces from Klaus et al.

    Returns
    -------
    dict
        ``'t'``: time array, ``'pcna1'``/``'mCherry'``: dicts of normalised traces.
    """
    data_pcna1 = pd.read_excel(
        DATAPATH + "klaus_pcna1_mcherry_traces.xlsm",
        sheet_name="PCNA1-GFP Egress aligned",
    )
    data_mCherry = pd.read_excel(
        DATAPATH + "klaus_pcna1_mcherry_traces.xlsm",
        sheet_name="NLS-mCherry Egress aligned",
    )
    t = data_pcna1["Time"].to_numpy()

    def normalize_signal(signal):
        background = np.min(
            [
                np.nanpercentile(signal[t > 0], 15),
                np.nanpercentile(signal[t <= 0], 15),
            ]
        )
        signal_backgroundcorrected = signal - background
        signal_normalized = signal_backgroundcorrected / np.nanpercentile(
            signal_backgroundcorrected[t <= 0], 85
        )
        return signal_normalized

    def isgrowing(signal, t=t):
        signal = normalize_signal(signal)
        mask = ~np.isnan(signal)
        signal, t = signal[mask], t[mask]
        fold_change = np.nanpercentile(signal[t < 0], 85) / np.max(
            [0.01, signal[:6].mean()]
        )
        return fold_change > 1.25

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        pcna1 = {
            col: normalize_signal(data_pcna1[col].to_numpy())
            for col in data_pcna1.columns
            if f"{col}"[0].isdigit() and isgrowing(data_pcna1[col], t)
        }

        mCherry = {
            col: normalize_signal(data_mCherry[col].to_numpy())
            for col in data_mCherry.columns
            if f"{col}"[0].isdigit() and isgrowing(data_mCherry[col], t)
        }

    return {
        "t": t,
        "pcna1": pcna1,
        "mCherry": mCherry,
    }


def _klaus_get_merozoite_count():
    """Return array of observed merozoite counts from Klaus et al."""
    return np.array(
        [
            24,
            16,
            17,
            28,
            24,
            23,
            24,
            23,
            25,
            21,
            18,
            24,
            26,
            20,
            22,
            26,
            22,
            28,
            30,
            19,
            26,
        ]
    )


def _klaus_add_d_and_s_phases():
    """Merge both Klaus S/D-phase sheets into a single DataFrame."""
    df2 = _klaus_read_data_sd()
    df = _klaus_read_data_sd2()
    dframe = pd.concat([df, df2])
    return dframe.dropna(how="all")


def _klaus_read_data_schizo():
    """Load, rename and return Duration of Schizogony data."""
    data = pd.read_csv(
        DATAPATH + "klaus_schizogony_durations.csv",
        sep=None,
        engine="python",
    )
    last_r_to_egress = (
        data["first replication to egress"] - data["first to last replication"]
    )
    first_to_last_r = data["first to last replication"]
    return pd.DataFrame(
        {
            "last r to egress": last_r_to_egress,
            "first to last r": first_to_last_r,
        }
    )


def _klaus_read_data_sd():
    """Read klaus_sd_phase_timing.csv and rename columns."""
    data = pd.read_csv(
        DATAPATH + "klaus_sd_phase_timing.csv", sep=None, engine="python"
    )

    r = (data["End 1st Replication"] - data["Start 1st replication"]) * 5
    r0 = (
        data["End 2nd round of replication"] - data["Start 2nd replication"]
    ) * 5
    r1 = (
        data["End 3rd round of replication "]
        - data["Start 3rd round of replication"]
    ) * 5
    r00 = (data["End 4th"] - data["Start 4th"]) * 5
    r01 = (data["End 5th"] - data["Start 5th"]) * 5

    d0 = (data["Start 2nd replication"] - data["End 1st Replication"]) * 5
    d1 = (
        data["Start 3rd round of replication"] - data["End 1st Replication"]
    ) * 5
    d00 = (data["Start 4th"] - data["End 2nd round of replication"]) * 5
    d01 = (data["Start 5th"] - data["End 2nd round of replication"]) * 5

    ds0 = (data["Start 2nd replication"] - data["1st Division"]) * 5
    ds1 = (data["Start 3rd round of replication"] - data["1st Division"]) * 5
    ds00 = (data["Start 4th"] - data["2nd Divison"]) * 5
    ds01 = (data["Start 5th"] - data["2nd Divison"]) * 5

    data_dict = {
        "s": r,
        "s0": r0,
        "s1": r1,
        "s00": r00,
        "s01": r01,
        "d0": d0,
        "d1": d1,
        "d00": d00,
        "d01": d01,
        "ds0": ds0,
        "ds1": ds1,
        "ds00": ds00,
        "ds01": ds01,
        "initial fluorescence": data[
            "Fluorescence Signal/Area before replication"
        ],
    }
    index = pd.Index(
        [f"{a:.0f}_{p}" for a, p in zip(data["analysis"], data["parasite"])]
    )
    return pd.DataFrame(data_dict).set_index(index)


def _klaus_read_data_sd2():
    """Read klaus_sd_phase_timing_long.csv and rename columns."""
    data = pd.read_csv(
        DATAPATH + "klaus_sd_phase_timing_long.csv", sep=None, engine="python"
    )

    r = data["Lenght 1st R"]
    r0 = data["Lenght 2nd R"]
    r1 = data["Lenght 3rd R"]
    r00 = data["Lenght 4th R"]
    r01 = data["Lenght 5th R"]
    r10 = data["Lenght 6th R"]
    r11 = data["Lenght 7th R"]
    r41 = data["Lenght 8th R"]

    d0 = (data["Start 2nd R"] - data["End 1st R"]) * 5
    d1 = (data["Start 3rd R"] - data["End 1st R"]) * 5
    d00 = (data["Start 4th R"] - data["End 2nd R"]) * 5
    d01 = (data["Start 5th R"] - data["End 2nd R"]) * 5
    d10 = (data["Start 6th R"] - data["End 3rd R"]) * 5
    d11 = (data["Start 7th R"] - data["End 3rd R"]) * 5

    ds0 = (data["Start 2nd R"] - data["1st D"]) * 5
    ds1 = (data["Start 3rd R"] - data["1st D"]) * 5
    ds00 = (data["Start 4th R"] - data["2nd D"]) * 5
    ds01 = (data["Start 5th R"] - data["2nd D"]) * 5
    ds10 = (data["Start 6th R"] - data["3rd D"]) * 5
    ds11 = (data["Start 7th R"] - data["3rd D"]) * 5

    sd = data["End 1st R to D"]
    sd0 = (data["2nd D"] - data["End 2nd R"]) * 5
    sd1 = (data["3rd D"] - data["End 3rd R"]) * 5

    s_overlapp = (data["End 2nd R"] - data["Start 3rd R"]) * 5

    data_dict = {
        "s": r,
        "s0": r0,
        "s1": r1,
        "s00": r00,
        "s01": r01,
        "s10": r10,
        "s11": r11,
        "s41": r41,
        "sd": sd,
        "sd0": sd0,
        "sd1": sd1,
        "d0": d0,
        "d1": d1,
        "d00": d00,
        "d01": d01,
        "d10": d10,
        "d11": d11,
        "ds0": ds0,
        "ds1": ds1,
        "ds00": ds00,
        "ds01": ds01,
        "ds10": ds10,
        "ds11": ds11,
        "s_overlapp": s_overlapp,
        "initial fluorescence": data[
            "Fluorescence Signal/Area before replication"
        ],
    }
    index = pd.Index([f"20200802_{d}" for d in data["20200802"]])
    return pd.DataFrame(data_dict).set_index(index)


def _klaus_get_data_longitudinal():
    """Load all longitudinal csv files from Severina.

    Returns
    -------
    dict
        Single-cell dicts keyed by parasite identifier.
    """
    df = pd.read_csv(
        DATAPATH + "klaus_longitudinal_traces.csv",
        sep=None,
        engine="python",
    )

    parasites = np.unique(
        [d[2:] for d in df.columns if d.startswith(("r_", "c_", "g_"))]
    )
    time = df["Time"].to_numpy()
    data_sc = {}
    for parasite in parasites:
        if "c_" + parasite not in df.columns:
            continue
        ns = df["c_" + parasite].to_numpy()

        if (ns == -1).any():
            ns[np.nanargmax(ns == -1) :] = 0
            is_finished = True
        else:
            is_finished = False

        if np.argwhere(np.diff(ns) > 0)[0] < np.argwhere(np.diff(ns) < 0)[0]:
            t_begin_s2 = time[int(np.argwhere(np.diff(ns) > 0)[1][0]) + 1]
        else:
            t_begin_s2 = time[int(np.argwhere(np.diff(ns) > 0)[0][0]) + 1]

        t = time - t_begin_s2

        replication = np.zeros_like(ns)
        replication[ns >= 1] = 1
        if is_finished:
            data_sc[parasite] = {
                "t": t,
                "ns": ns,
                "replication": replication,
                "is_finished": is_finished,
            }
    return data_sc


def _klaus_add_bulk_data(data_sc):
    """Aggregate single-cell longitudinal traces into bulk summary statistics.

    Returns
    -------
    dict
        Keys ``'t'``, ``'ns'`` (with mean/std/percentile sub-dicts),
        ``'ns_binary'``, ``'2nd to last S-phase'``.
    """
    tmin, tmax = np.swapaxes(
        np.array([[d["t"].min(), d["t"].max()] for d in data_sc.values()]), 0, 1
    )
    tmin, tmax = np.min(tmin), np.max(tmax)
    t = np.arange(tmin / 5, 1 + tmax / 5) * 5

    ns = np.empty((len(data_sc), len(t)))
    ns.fill(np.nan)
    second_to_last = np.zeros(len(data_sc))
    for i, d in enumerate(data_sc.values()):
        offset = np.argmax(t == d["t"][0])
        ns[i][offset : offset + len(d["ns"])] = d["ns"]
        ns[i][0:offset] = 0
        if d["is_finished"]:
            ns[i][offset + len(d["ns"]) :] = 0
            second_to_last[i] = d["t"][d["ns"] > 0][-1]
        else:
            second_to_last[i] = np.nan

    bs_samples = np.random.choice(
        np.arange(len(ns)), (10000, len(ns)), replace=True
    )

    ns_binary = ns.copy()
    ns_binary[ns > 1] = 1

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        ns_mean_bs = np.nanmean(ns[bs_samples], axis=1)
        return {
            "t": t,
            "ns": {
                "mean": np.nanmean(ns, axis=0),
                "mean_std": np.nanstd(ns_mean_bs, axis=0),
                "mean_low": np.nanpercentile(
                    ns_mean_bs, q=2.5, axis=0, method="linear"
                ),
                "mean_high": np.nanpercentile(
                    ns_mean_bs, q=97.5, axis=0, method="linear"
                ),
                "std": np.nanstd(ns, axis=0),
                "median": np.nanpercentile(ns, q=50, axis=0, method="linear"),
                "iqr_low": np.nanpercentile(ns, q=25, axis=0, method="linear"),
                "iqr_high": np.nanpercentile(ns, q=75, axis=0, method="linear"),
                "traces": ns,
            },
            "ns_binary": {
                "mean": np.nanmean(ns_binary, axis=0),
                "std": np.nanstd(ns_binary, axis=0),
            },
            "2nd to last S-phase": second_to_last,
        }


def get_data_klaus():
    """Load all Klaus et al. experimental data.

    Returns
    -------
    dict
        Keys ``'sc'``, ``'bulk'``, ``'sd'``, ``'schizo'``, ``'merozoite_count'``,
        ``'pcna1'``.
    """
    data = {}
    data["sc"] = _klaus_get_data_longitudinal()
    data["bulk"] = _klaus_add_bulk_data(data["sc"])
    data["sd"] = _klaus_add_d_and_s_phases()
    data["schizo"] = _klaus_read_data_schizo()
    data["merozoite_count"] = _klaus_get_merozoite_count()
    data["pcna1"] = _klaus_get_total_pcna1_growth()
    return data


# ---------------------------------------------------------------------------
# Airyscan data helpers
# ---------------------------------------------------------------------------


def _airy_add_bulk_data(data_sc):
    """Aggregate single-cell Airyscan traces into bulk summary statistics.

    Returns
    -------
    dict
        Keys ``'t'``, ``'ns'``, ``'ns_binary'``, ``'n'``, ``'2nd to last S-phase'``.
    """
    tmin, tmax = np.swapaxes(
        np.array([[d["t"].min(), d["t"].max()] for d in data_sc.values()]), 0, 1
    )
    tmin, tmax = np.min(tmin), np.max(tmax)
    t = np.arange(tmin / 5, 1 + tmax / 5) * 5

    ns, n = np.empty((2, len(data_sc), len(t)))
    ns.fill(np.nan)
    n.fill(np.nan)
    second_to_last = np.zeros(len(data_sc))
    for i, d in enumerate(data_sc.values()):
        offset = np.argmax(t == d["t"][0])
        ns[i][offset : offset + len(d["ns"])] = d["ns"]
        ns[i][0:offset] = 0
        n[i][offset : offset + len(d["n"])] = d["n"]
        n[i][0:offset] = 1
        if d["is_finished"]:
            ns[i][offset + len(d["ns"]) :] = 0
            second_to_last[i] = d["t"][d["ns"] > 0][-1]
        else:
            second_to_last[i] = np.nan

    bs_samples = np.random.choice(
        np.arange(len(ns)), (10000, len(ns)), replace=True
    )

    ns_binary = ns.copy()
    ns_binary[ns > 1] = 1

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        ns_mean_bs = np.nanmean(ns[bs_samples], axis=1)
        return {
            "t": t,
            "ns": {
                "mean": np.nanmean(ns, axis=0),
                "mean_std": np.nanstd(ns_mean_bs, axis=0),
                "std": np.nanstd(ns, axis=0),
                "median": np.nanpercentile(ns, q=50, axis=0, method="linear"),
                "iqr_low": np.nanpercentile(ns, q=25, axis=0, method="linear"),
                "iqr_high": np.nanpercentile(ns, q=75, axis=0, method="linear"),
            },
            "ns_binary": {
                "mean": np.nanmean(ns_binary, axis=0),
                "std": np.nanstd(ns_binary, axis=0),
            },
            "n": {
                "mean": np.nanmean(n, axis=0),
                "std": np.nanstd(n, axis=0),
            },
            "2nd to last S-phase": second_to_last,
        }


def _airy_split_ranges(x):
    """Split a range string or number into [start, end]."""
    if isinstance(x, (int, np.integer, float)):
        return [x, x]
    elif x.isdigit():
        return [int(x), int(x)]
    elif x == "unresolvable":
        return [np.nan, np.nan]
    else:
        splitted_range = [
            int(s) for s in x.replace("/", "-").split("-") if s.isdigit()
        ]
        if len(splitted_range) == 2:
            return splitted_range
        else:
            return [np.nan, np.nan]


def _airy_convert_raw_data(df):
    """Convert one Airyscan Excel sheet into a processed single-cell dict.

    Returns
    -------
    dict
        Keys ``'t'``, ``'replication'``, ``'ns'``, ``'ns_error'``, ``'n'``,
        ``'n_error'``, ``'n_exact'``, ``'is_finished'``.
    """

    def get_upper_and_lower_bound(label, mask):
        raw_ns = df[label].to_numpy()[mask]
        ns_low, ns_high = np.swapaxes(
            np.array([_airy_split_ranges(ns) for ns in raw_ns]),
            0,
            1,
        )
        return ns_low, ns_high

    t = df["Minutes"].to_numpy()
    mask = ~np.isnan(t)
    t = t[mask]

    raw_n_label = next(
        (
            label
            for label in [
                "Nuclei no. Estimation",
                "Nuclei no. estimation",
                "Nuclei No. estim.",
            ]
            if label in df
        ),
        None,
    )
    raw_ns_label = "No. Nuclei in S-phase"

    n_exact_label = next(
        (
            label
            for label in [
                "Nuclei No. Probably",
                "Nuclei no. probable",
                "nuclei no. Probable",
            ]
            if label in df
        ),
        "Nuclei no. Probable",
    )

    if (df[raw_ns_label] == -1).any():
        is_finished = True
    else:
        is_finished = False

    ns_low, ns_high = get_upper_and_lower_bound(raw_ns_label, mask)
    n_low, n_high = get_upper_and_lower_bound(raw_n_label, mask)

    if (
        np.argwhere(np.diff(ns_low) > 0)[0]
        < np.argwhere(np.diff(ns_low) < 0)[0]
    ):
        t_begin_s2 = t[int(np.argwhere(np.diff(ns_low) > 0)[1]) + 1]
    else:
        t_begin_s2 = t[int(np.argwhere(np.diff(ns_low) > 0)[0]) + 1]

    if n_exact_label in df:
        n_exact = df[n_exact_label].to_numpy()[mask]
    else:
        n_exact = (n_low + n_high) / 2
    return {
        "t": t - t_begin_s2,
        "replication": pd.to_numeric(
            df["Replc._Egress"], errors="coerce"
        ).to_numpy()[mask],
        "ns": (ns_low + ns_high) / 2,
        "ns_error": {
            "low": ns_low,
            "high": ns_high,
        },
        "n": (n_low + n_high) / 2,
        "n_error": {
            "low": n_low,
            "high": n_high,
        },
        "n_exact": n_exact,
        "is_finished": is_finished,
    }


def _airy_add_d_and_s_phases(raw_data):
    """Extract S- and D-phase durations from all Airyscan sheets.

    Returns
    -------
    pd.DataFrame
        Phase durations with one row per parasite.
    """
    data_sd = {
        "s": [],
        "s0": [],
        "s1": [],
        "s00": [],
        "s01": [],
        "s10": [],
        "s11": [],
        "d0": [],
        "d1": [],
        "d00": [],
        "d01": [],
        "d10": [],
        "d11": [],
        "ds0": [],
        "ds1": [],
        "ds00": [],
        "ds01": [],
        "ds10": [],
        "ds11": [],
        "sd": [],
        "sd0": [],
        "sd1": [],
        "index": [],
        "s_overlapp": [],
    }
    for key, rw_dt in raw_data.items():
        t = rw_dt["Minutes"].to_numpy()

        s_phases = {}
        for x in ["N3.2", "N3.1", "N2.1", "N1", "N2.2", "N3.3", "N3.4"]:
            series = rw_dt[x]
            n = pd.to_numeric(series, errors="coerce").to_numpy()

            if np.isnan(n).all():
                s_phases[x] = {
                    "nuclear_start": np.nan,
                    "start": np.nan,
                    "end": np.nan,
                    "duration": np.nan,
                }
            else:
                nuclear_start = t[np.nanargmax(n == 0)]
                s_start = t[int(np.nanargmax(np.diff(n))) + 1]
                s_end = t[int(np.nanargmin(np.diff(n))) + 1]
                s_phases[x] = {
                    "nuclear_start": nuclear_start,
                    "start": s_start,
                    "end": s_end,
                    "duration": s_end - s_start,
                }

        data_sd["index"].append(key)
        data_sd["s"].append(s_phases["N1"]["duration"])
        data_sd["s0"].append(s_phases["N2.1"]["duration"])
        data_sd["d0"].append(s_phases["N2.1"]["start"] - s_phases["N1"]["end"])
        data_sd["s1"].append(s_phases["N2.2"]["duration"])
        data_sd["d1"].append(s_phases["N2.2"]["start"] - s_phases["N1"]["end"])
        data_sd["s00"].append(s_phases["N3.1"]["duration"])
        data_sd["d00"].append(
            s_phases["N3.1"]["start"] - s_phases["N2.1"]["end"]
        )
        data_sd["s01"].append(s_phases["N3.2"]["duration"])
        data_sd["d01"].append(
            s_phases["N3.2"]["start"] - s_phases["N2.1"]["end"]
        )
        data_sd["s10"].append(s_phases["N3.3"]["duration"])
        data_sd["d10"].append(
            s_phases["N3.3"]["start"] - s_phases["N2.2"]["end"]
        )
        data_sd["s11"].append(s_phases["N3.4"]["duration"])
        data_sd["d11"].append(
            s_phases["N3.4"]["start"] - s_phases["N2.2"]["end"]
        )

        data_sd["sd"].append(
            s_phases["N2.1"]["nuclear_start"] - s_phases["N1"]["end"]
        )
        data_sd["sd0"].append(
            s_phases["N3.1"]["nuclear_start"] - s_phases["N2.1"]["end"]
        )
        data_sd["sd1"].append(
            s_phases["N3.3"]["nuclear_start"] - s_phases["N2.2"]["end"]
        )

        data_sd["ds0"].append(
            s_phases["N2.1"]["start"] - s_phases["N2.1"]["nuclear_start"]
        )
        data_sd["ds1"].append(
            s_phases["N2.2"]["start"] - s_phases["N2.2"]["nuclear_start"]
        )
        data_sd["ds00"].append(
            s_phases["N3.1"]["start"] - s_phases["N3.1"]["nuclear_start"]
        )
        data_sd["ds01"].append(
            s_phases["N3.2"]["start"] - s_phases["N3.2"]["nuclear_start"]
        )
        data_sd["ds10"].append(
            s_phases["N3.3"]["start"] - s_phases["N3.3"]["nuclear_start"]
        )
        data_sd["ds11"].append(
            s_phases["N3.4"]["start"] - s_phases["N3.4"]["nuclear_start"]
        )

        data_sd["s_overlapp"].append(
            s_phases["N2.1"]["end"] - s_phases["N2.2"]["start"]
        )
    df = pd.DataFrame(data_sd)
    df.index = df["index"]
    df = df.drop(columns=["index"])
    df[df < 0] = np.nan
    return df.dropna(how="all")


def get_new_data():
    """Load and process Airyscan time-lapse data from the Excel workbook.

    Returns
    -------
    dict
        Keys ``'sc'`` (single-cell dicts), ``'bulk'`` (aggregated statistics),
        ``'sd'`` (S/D-phase durations DataFrame).
    """
    raw_data_unfiltered = pd.read_excel(
        DATAPATH + "airyscan_timelapse.xlsx", index_col=0, sheet_name=None
    )
    raw_data = {
        key: value
        for key, value in raw_data_unfiltered.items()
        if "F11" in key and "double" not in key
    }

    data = {}
    data["sc"] = {
        key: _airy_convert_raw_data(df) for key, df in raw_data.items()
    }
    data["bulk"] = _airy_add_bulk_data(data["sc"])
    data["sd"] = _airy_add_d_and_s_phases(raw_data)

    return data


def get_data(dataset, drop_data=DropData.NONE):
    """Load experimental data for the given dataset and apply optional filtering.

    Parameters
    ----------
    dataset : DataSet
        Must be ``DataSet.KLAUS`` or ``DataSet.AIRY``.
    drop_data : DropData
        ``NONE``: keep all raw data.  ``SUBTREE``: NaN-out incomplete sister
        pairs.  ``ALL``: drop rows with any missing phase.

    Returns
    -------
    dict
        Keys ``'sc'``, ``'bulk'``, ``'sd'``, plus dataset-specific entries.
    """
    # test if provided dataset is exp dataset
    assert dataset.value in [DataSet.KLAUS.value, DataSet.AIRY.value], (
        "Invalid exp dataset: Only DataSet.KLAUS and DataSet.AIRY are valid "
        "options"
    )
    match dataset.value:
        case DataSet.KLAUS.value:
            data = get_data_klaus()
        case DataSet.AIRY.value:
            data = get_new_data()

    if drop_data.equal(DropData.ALL):
        data["sd"] = data["sd"].dropna(subset=ALL_PHASES)
    elif drop_data.equal(DropData.SUBTREE):
        df = data["sd"]
        keys = np.unique(
            [
                col[:-1]
                for col in df.columns
                if (
                    len(col) > 1
                    and not col.startswith("ds")
                    and not col.startswith("sd")
                )
            ]
        )
        for key in keys:
            col_sisters = [f"{key}0", f"{key}1"]
            if all([sis in df.columns for sis in col_sisters]):
                df.loc[df[col_sisters].isna().any(axis=1), col_sisters] = np.nan
    return data


def convert_to_df(results):
    """Convert branching-process simulation results to a phase-duration DataFrame.

    Each row represents one simulated tree; columns are the S- and D-phase
    durations keyed by their binary-tree name (e.g. ``s``, ``s0``, ``d01``).

    Parameters
    ----------
    results : dict
        Mapping of tree ID to phase-interval dicts ``{name: (start, end)}``.

    Returns
    -------
    pd.DataFrame
        Phase durations with one row per tree.
    """
    keys = [
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
        "d000",
        "d001",
        "d010",
        "d011",
        "d100",
        "d101",
        "d110",
        "d111",
    ]

    sd = np.zeros((len(results), len(keys)))
    for i, tree in enumerate(results.values()):
        if np.all([key in tree.keys() for key in keys]):
            sd[i] = [tree[key][1] - tree[key][0] for key in keys]
        else:
            for j, key in enumerate(keys):
                if key in tree.keys():
                    sd[i, j] = tree[key][1] - tree[key][0]
                else:
                    sd[i, j] = np.nan

    sd_dict = {}
    for i, key in enumerate(keys):
        sd_dict[key] = sd[:, i]
    return pd.DataFrame(sd_dict, index=results.keys())
