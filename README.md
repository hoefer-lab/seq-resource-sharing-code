[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20698728.svg)](https://doi.org/10.5281/zenodo.20698728)

# Sequential Resource Sharing (SRS)

Companion code for reproducing all figures in:

> **Competitive resource allocation drives asynchronous and rapid nuclear
> multiplication in the malaria parasite**
>
> Patrick Binder, Aistė Kudulytė, Severina Klaus, Thomas Höfer,
> Ulrich S. Schwarz, Markus Ganter, Nils B. Becker
>
> *Nature Communications* **17**, 7413 (2026).
> [doi:10.1038/s41467-026-75378-x](https://doi.org/10.1038/s41467-026-75378-x)

The preprint is available on bioRxiv
([doi:10.1101/2025.08.07.669072](https://doi.org/10.1101/2025.08.07.669072)).

This package implements stochastic branching-process models for DNA
replication timing in *Plasmodium falciparum* schizogony.

## Installation

```bash
pip install -e .
```

This installs the package in editable mode together with all required
dependencies (typical install time: ~1 minute).  To include the test
dependencies, use:

```bash
pip install -e ".[test]"
```

> **Note:** If building `numbalsoda` fails (e.g. missing Fortran compiler),
> either install it via conda first (`conda install -c conda-forge numbalsoda`)
> or install a Fortran compiler (`brew install gcc` on macOS).

> **Tested with:** Python 3.10 on macOS 15 (Apple Silicon / arm64).
> A full conda environment specification is provided in
> [`tested_environment.yml`](tested_environment.yml).

After installation the subpackages `srs.lib`, `srs.model2` and
`srs.model3` are importable from anywhere in the environment.

## Notation mapping

| Code             | Manuscript                             | Description                           |
|------------------|----------------------------------------|---------------------------------------|
| `d`              | D*-phase                               | Delay / gap between replications      |
| `s`              | S*-phase                               | DNA replication                       |
| `std`            | σ<sub>D*</sub>                         | D-phase duration standard deviation   |
| `scaling`        | ⟨τ<sub>min</sub>⟩                      | Mean minimal S-phase duration         |
| `f_threshold`    | c<sub>i,min</sub>                      | Minimal activated fraction            |

## Experimental datasets

| `DataSet` enum | Microscope                                      | Used in       |
|----------------|-------------------------------------------------|---------------|
| `KLAUS`        | Spinning-disk PerkinElmer UltraVIEW VoX (Klaus *et al.* 2022)  | Figs. 2, 6 |
| `AIRY`         | Zeiss LSM900 with Airyscan 2 detector (higher resolution, ~33 % slower nuclear cycles due to increased phototoxicity) | Fig. S7 |

## Models overview

| Model   | `DataSet` enum    | Description                                                             |
|---------|-------------------|-------------------------------------------------------------------------|
| Model 1 | `MODEL1`         | Independent branching process without kinship inheritance.  Subset of model 2 with all off-diagonal correlations removed except the D-sister correlation (d0 ↔ d1). |
| Model 2 | `BARMODEL`       | Correlated branching process with the full empirical kinship correlation matrix. |
| Model 3 | `RESMODEL` / `RESPARMODEL` | Resource-sharing branching process with sequential or parallel sharing of a finite replication resource. |

## Reproducing the paper figures

All figure-generating functions live inside the package and can be called
from a Python session (or a short script).  Start by activating the
environment and launching Python from the repository root.

All output PDFs are saved to `src/srs/fig/`, which is created on first use.

> **Expected run times:** Plotting functions take a few
> minutes each.  Generating the full parameter-scan data for Fig. 4 & Fig. 5
> (`run_simulation_for_fig45` over the complete grid) takes days to weeks.

### Figure 2: Correlated branching process (model 2)

```python
from srs.model2.plotting.fig2 import plot_fig2
plot_fig2()
```

Output: main Fig. 2 and SI Figs. S1 and S2.

### Figures 4 & 5: Resource-sharing dynamics (model 3)

#### Step 0: Generate data

Generate the simulation data (parameter scans over scarcity ζ and delay τ).

```python
from srs.model3.generate.fig45 import run_simulation_for_fig45

# Example: one (zeta, tau) point
run_simulation_for_fig45(zeta=1.0, tau=0.2, n_sample=480, n_gen=15, std=0.1)
```

This appends the results to a newline-delimited JSON file in `src/srs/data/`.
You typically run many `(zeta, tau)` combinations (see
`model3/generate/fig45.py` for the full parameter grid).

#### Step 1: Plot

```python
from srs.model3.plotting.fig45 import plot_fig45

plot_fig45()
```

Output: main Figs. 4 and 5 and SI Fig. S3.

> **Note:**  Step 0 can also be skipped. Calling `plot_fig45()` without
> specifying a file uses the pre-computed scan files in `src/srs/data/`.
> To plot your own scan from Step 0, pass its path via `filename=...`.

### Figure 6: Simulated vs. experimental replication timing (model 3)

```python
from srs.model3.default_params import DEFAULT_PARAMS, F_THRESHOLD
from srs.model3.plotting.fig6 import plot_fig6

plot_fig6(params=DEFAULT_PARAMS, f_threshold=F_THRESHOLD, n=500)
```

Key parameters:

| Parameter     | Default              | Description                                  |
|---------------|----------------------|----------------------------------------------|
| `params`      | `DEFAULT_PARAMS`     | MAP parameter estimates from ABC-SMC         |
| `f_threshold` | `F_THRESHOLD` (0.15) | c<sub>i,min</sub>, activated-fraction threshold for S-phase |
| `n`           | `500`                | Number of simulated realisations             |
| `drop_data`   | `DropData.SUBTREE`   | How to filter incomplete sister pairs        |

Output: main Fig. 6, Airyscan SI Fig. S7 and fitting diagnostics
SI Fig. S4.

## Project structure

```
seq-resource-sharing-code/
├── pyproject.toml          # Package metadata (pip install -e .)
├── src/srs/
│   ├── lib/                    # Shared library
│   │   ├── enums.py            # EnhancedEnum + domain enumerations
│   │   ├── style.py            # Colour palettes, rcParams, figure sizing
│   │   ├── data_loader.py      # Unified data loader (Klaus + Airyscan)
│   │   ├── plotting.py         # Shared plotting functions
│   │   ├── correlation_matrix.py  # Kinship correlation computation
│   │   └── shared_resource.py  # Numba-accelerated resource ODE
│   ├── model2/                 # Correlated branching process
│   │   ├── _core/              # Simulation engine (internal)
│   │   └── plotting/           # Fig. 2 and SI panels
│   ├── model3/                 # Sequential resource-sharing model
│   │   ├── _core/              # ODE solvers + simulation loop (internal)
│   │   ├── plotting/           # Figs. 4–6 and tree visualisation
│   │   ├── generate/           # Data-generation scripts for Figs. 4 & 5
│   │   ├── _inference.py       # ABC-SMC parameter inference (internal)
│   │   └── default_params.py   # MAP estimates and F_THRESHOLD
│   ├── data/                   # Experimental & pre-computed data (see data/README.md)
│   └── fig/                    # Generated figures (PDF output)
└── tests/                      # Unit tests
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{binder2026competitive,
  author  = {Binder, Patrick and Kudulyt{\.e}, Aist{\.e} and Klaus, Severina and
             H{\"o}fer, Thomas and Schwarz, Ulrich S. and Ganter, Markus and
             Becker, Nils B.},
  title   = {Competitive resource allocation drives asynchronous and rapid
             nuclear multiplication in the malaria parasite},
  journal = {Nature Communications},
  volume  = {17},
  pages   = {7413},
  year    = {2026},
  doi     = {10.1038/s41467-026-75378-x}
}
```

The archived code itself is available on Zenodo
([doi:10.5281/zenodo.20698728](https://doi.org/10.5281/zenodo.20698728)).

## License

MIT, see [`LICENSE`](LICENSE).
