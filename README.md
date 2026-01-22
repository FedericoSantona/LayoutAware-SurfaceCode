# Layout-Aware Surface Code Builder for Lattice Surgery

Research code for building **layout-aware surface-code circuits** (standard + heavy-hex) with **lattice-surgery protocols** and running **phenomenological-noise** simulations in **Stim**, including **threshold sweeps** for both **memory** and **CNOT (lattice surgery)** experiments.

If you are reading this to reproduce results/figures: start from **Reproducibility (threshold sweeps)** below. For the underlying design/algorithms, see `documentation/TECHNICAL_SPECIFICATION.md`.

---

## What this repository provides

- **Surface-code model builders** for:
  - **Standard** planar surface code
  - **Heavy-hex** surface code
- A **layout manager** for multi-patch experiments (e.g., control/ancilla/target patches) with explicit seam ancillas.
- A **multi-phase Stim circuit builder** that handles stabilizer-set changes across phases (warmup rounds, detector continuity rules).
- A **lattice-surgery CNOT** scaffold following Horsman et al. (2013), with:
  - smooth merge/split (measuring \(Z_L^C Z_L^{INT}\))
  - rough merge/split (measuring \(X_L^{INT} X_L^{T}\))
- **Monte Carlo + decoding** utilities (Stim + PyMatching) and scripts to:
  - run single experiments (memory / surgery physics / surgery logical error rate)
  - run **threshold sweeps** and write results as CSV/JSON
  - compute **bootstrap confidence intervals** for threshold estimators (post-processing only; no resimulation)

---

## Installation

### Requirements

- **Python**: 3.10+ (this repo includes a local `myenv/` in the workspace; you can ignore it and use your own venv).
- **Core simulation stack**: `stim`, `pymatching`, `numpy`, `scipy`, `matplotlib` (and optionally `pandas`, `tqdm`).
- **Optional (needed for geometry/code builders)**: `qiskit-qec` (used by `src/surface_code/heavy_hex.py` and `src/surface_code/standard.py`).
  - The codebase contains an import guard: if `qiskit_qec` is missing, builders will raise a clear `ImportError`.

### Create an environment and install dependencies

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

To enable heavy-hex/standard geometry construction via Qiskit-QEC:

```bash
python -m pip install qiskit-qec
```

---

## Quickstart (single runs)

### Memory experiment (single distance)

Runs a phenomenological-noise memory experiment and prints logical error rate diagnostics:

```bash
python scripts/memory_experiment.py \
  --code-type standard \
  --distance 3 \
  --px 1e-3 \
  --pz 1e-3 \
  --shots 100000
```

This script also saves a tiling figure to `plots/` (e.g. `plots/surface_code_d3.png`).

### Lattice-surgery CNOT (physics-mode sanity check)

Samples Bell-type correlators (intended as a physics sanity check of the CNOT scaffold):

```bash
python scripts/surgery_experiment.py \
  --mode physics \
  --code-type standard \
  --distance 3 \
  --px 0 \
  --pz 0 \
  --shots 100000
```

---

## Reproducibility (threshold sweeps)

The main reproducibility entrypoint is:

- `scripts/threshold_experiment.py`: runs threshold sweeps and writes plots + data
- `scripts/threshold_bootstrap.py`: post-processes saved sweeps to produce bootstrap uncertainty (no resimulation)

### Run a threshold sweep

Example: **memory threshold**, heavy-hex layout, distances \(d \in \{3,5,7,9\}\):

```bash
python scripts/threshold_experiment.py \
  --layout heavy_hex \
  --experiment-type memory \
  --distances 3 5 7 9 \
  --shots 10000 \
  --p-min 5e-4 \
  --p-max 1e-1 \
  --num-points 20
```

Example: **CNOT (lattice-surgery) threshold**, standard layout:

```bash
python scripts/threshold_experiment.py \
  --layout standard \
  --experiment-type cnot \
  --distances 3 5 7 9 \
  --shots 10000 \
  --p-min 5e-4 \
  --p-max 1e-1 \
  --num-points 20
```

### Where results go (files)

By default, `scripts/threshold_experiment.py` writes:

- **Plots**: `plots/threshold/{layout}/{experiment_type}/*.png`
- **Per-scenario data**: `output/threshold/{layout}/{experiment_type}/*.csv` and `*.json`
- **Summary**: `output/threshold/{layout}/{experiment_type}/threshold_summary.json`

The summary JSON includes the run metadata (shots, distances, grid) and the paths to per-scenario files.

### Bootstrap uncertainty for threshold estimates

This is a post-processing step (reads the saved JSON; does not rerun Stim).

Example: bootstrap the heavy-hex memory summary:

```bash
python scripts/threshold_bootstrap.py \
  --preset heavy_hex_memory \
  --B 2000 \
  --seed 0 \
  --crossings all_pairs \
  --estimator best \
  --ci 0.95
```

You can also write a JSON report:

```bash
python scripts/threshold_bootstrap.py \
  --preset heavy_hex_memory \
  --json-out output/threshold/heavy_hex/memory/bootstrap_report.json
```

---

## Project structure

- `src/surface_code/`: geometry, stabilizers/logicals utilities, layout management, multi-phase Stim builder, lattice surgery protocols
- `src/simulation/`: Monte Carlo wrappers, logical error rate evaluation, and threshold utilities
- `scripts/`: reproducibility CLIs (memory, surgery, thresholds, bootstrap)
- `plots/` and `output/`: generated artefacts (figures, CSV/JSON summaries)
- `documentation/TECHNICAL_SPECIFICATION.md`: algorithmic/design documentation (recommended reading for method details)
- `test/`: unit tests for surgery geometry/masking and transpilation utilities

---

## How to cite

If you use this code in academic work, please cite the repository and/or the accompanying manuscript/thesis that reports the results.

Suggested citation (replace fields as appropriate):

```bibtex
@software{layoutaware_surfacecode,
  title        = {Layout-Aware Surface Code Builder for Lattice Surgery},
  author       = {QEC-Project Team},
  year         = {2024},
  note         = {Source code},
}
```



