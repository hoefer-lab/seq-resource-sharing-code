# Data files

## Experimental data — Klaus et al.  (spinning-disk confocal)

| File | Contents | Used by |
|------|----------|---------|
| `klaus_pcna1_mcherry_traces.xlsm` | PCNA1-GFP and NLS-mCherry fluorescence intensity traces, egress-aligned. Two sheets: *PCNA1-GFP Egress aligned*, *NLS-mCherry Egress aligned*. | `_klaus_get_total_pcna1_growth()` → Fig. 6 resource fitting |
| `klaus_schizogony_durations.csv` | Duration of schizogony: *first replication to egress* and *first to last replication* intervals. | `_klaus_read_data_schizo()` |
| `klaus_sd_phase_timing.csv` | S-phase and D-phase start/end frames for replication rounds 1–5, division timepoints, and initial fluorescence. Frame-based (×5 min). | `_klaus_read_data_sd()` → S/D-phase DataFrames |
| `klaus_sd_phase_timing_long.csv` | Extended S/D-phase timing for replication rounds 1–8 (longer recordings than `klaus_sd_phase_timing.csv`). Includes S-phase overlap and end-of-replication-to-division intervals. | `_klaus_read_data_sd2()` → S/D-phase DataFrames |
| `klaus_longitudinal_traces.csv` | Longitudinal single-cell traces: per-parasite columns (`r_`, `c_`, `g_` prefixes) encoding replication state, nuclei-in-S-phase count, and growth over time. | `_klaus_get_data_longitudinal()` → bulk statistics |

## Experimental data — Airyscan (Zeiss LSM900)

| File | Contents | Used by |
|------|----------|---------|
| `airyscan_timelapse.xlsx` | Airyscan 2 time-lapse data at 5 min intervals. Multi-sheet workbook (one sheet per parasite, filtered to "F11" sheets). Columns: time, nuclei-in-S-phase count, nuclei number estimation, per-nucleus replication state, egress. Nuclear cycles are ~33% slower than Klaus data due to increased phototoxicity. | `get_new_data()` → Fig. S7 |

## Pre-computed simulation results

| File | Contents | Used by |
|------|----------|---------|
| `param_scan_fig45.jsonl` | Branching-process parameter-scan results (newline-delimited JSON). Keys per line: `zeta`, `tau`, `k_b`, `n_gen`, S-phase durations (seq/par), growth rate λ, utilisation φ, efficiency η (each with mean/std). | `read_data()` → Figs. 4–5 |
| `param_scan_figs3.jsonl` | Same schema as `param_scan_fig45.jsonl` but for an alternative parameter regime. | `read_data(filename=FILENAME_SCAN_REGIME)` → Fig. S3 |
