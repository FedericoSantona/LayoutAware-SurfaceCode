# Layout-Aware Surface Code Builder for Lattice Surgery

Research code for building **layout-aware surface-code circuits** (standard + heavy-hex) with **lattice-surgery protocols** and running **device-aware noise simulations** in **Stim**, including **threshold sweeps** for both **memory** and **CNOT (lattice surgery)** experiments. Features **per-qubit T1/T2 decoherence modeling** from device calibration data with automatic layout selection based on device type.

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
- **Device-aware noise modeling** with:
  - Per-qubit T1/T2 decoherence parameters from device calibration
  - Support for IBM Quantum and IQM Crystal device calibration data
  - Automatic layout selection (heavy-hex for IBM, standard for IQM Crystal)
  - JSON-based calibration file format for reproducibility

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

Runs a memory experiment with device-aware noise and prints logical error rate diagnostics:

```bash
python scripts/memory_experiment.py \
  --code-type standard \
  --distance 3 \
  --px 1e-3 \
  --pz 1e-3 \
  --shots 100000
```

This script also saves a tiling figure to `plots/` (e.g. `plots/surface_code_d3.png`).

**Note**: For device-aware noise with calibration data, use `threshold_experiment.py` with `--noise-model device-aware`.

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

**Device-aware noise** (recommended for realistic hardware simulations):

See the "Device-aware noise model" section below for examples using calibration data.

**Uniform error rate model** (for theoretical/comparative studies):

You can also run threshold sweeps with uniform error rates across all qubits:

```bash
python scripts/threshold_experiment.py \
  --noise-model phenomenological \
  --layout heavy_hex \
  --experiment-type memory \
  --distances 3 5 7 9 \
  --shots 10000 \
  --p-min 5e-4 \
  --p-max 1e-1 \
  --num-points 20
```

This is useful for theoretical studies or comparing against device-specific results.

### Device-aware noise model

The code features **device-aware noise modeling** using real device calibration data (T1/T2, gate errors, readout errors). The layout is automatically selected based on the device type:

- **IBM devices** → `heavy_hex` layout
- **IQM Crystal** → `standard` layout

**Example: Device-aware threshold sweep with IQM Crystal calibration:**

```bash
python scripts/threshold_experiment.py \
  --noise-model device-aware \
  --calibration-file output/iqm_crystal_calibration.json \
  --round-duration 1.0 \
  --experiment-type cnot \
  --distances 3 5 7 9 \
  --shots 10000 \
  --p-min 5e-4 \
  --p-max 1e-1 \
  --num-points 20
```

The layout will be automatically set to `standard` for IQM Crystal. If you specify `--layout heavy_hex`, you'll get a warning and the layout will be auto-switched to the recommended value.

**Example: Device-aware threshold sweep with IBM device calibration:**

```bash
# First, load calibration from IBM backend (requires Qiskit IBM Runtime)
python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
from surface_code import DeviceCalibration
service = QiskitRuntimeService()
backend = service.backend('ibm_sherbrooke')
cal = DeviceCalibration.from_ibm_backend(backend)
cal.to_json('ibm_calibration.json')
"

# Then run threshold sweep
python scripts/threshold_experiment.py \
  --noise-model device-aware \
  --calibration-file ibm_calibration.json \
  --round-duration 1.0 \
  --experiment-type memory \
  --distances 3 5 7 9 \
  --shots 10000
```

The layout will be automatically set to `heavy_hex` for IBM devices.

### Calibration files

Example calibration files are provided in the `output/` directory:
- `output/iqm_crystal_calibration.json`: Synthetic IQM Crystal calibration (20 qubits)
- `output/example_calibration.json`: Generic example calibration file

You can create your own calibration files by:
1. Loading from IBM Quantum backends (requires Qiskit IBM Runtime)
2. Manually creating JSON files following the format shown in the examples
3. Using `DeviceCalibration.uniform()` for synthetic calibration for theoretical studies

### Noise model implementation

The **device-aware noise model** computes per-qubit error rates from T1/T2 decoherence times and gate errors from device calibration. It converts T1/T2 coherence times to Pauli error rates:
- Amplitude damping: `p_ad = 1 - exp(-t/T1)`
- Pure dephasing: `p_deph = 0.5 * (1 - exp(-t/T2_phi))` where `T2_phi = 1/(1/T2 - 1/(2*T1))`
- Pauli X rate: `p_x ≈ p_ad/4`
- Pauli Z rate: `p_z ≈ p_deph/2 + p_ad/4`

### Command-line options

When using `threshold_experiment.py`:

```bash
--noise-model {device-aware,phenomenological}
    Select noise model type (recommended: device-aware)

--calibration-file PATH
    Path to device calibration JSON file (required for device-aware mode)

--round-duration FLOAT
    Measurement round duration in microseconds (default: 1.0)
    Used to compute decoherence error rates from T1/T2

--layout {heavy_hex,standard}
    Surface code layout (auto-detected from calibration when using device-aware mode)
```

**Note**: The `--layout` option is automatically set when using device-aware noise to match the device type. You'll see a warning if you specify a conflicting layout, and it will be auto-switched.

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

- `src/surface_code/`: geometry, stabilizers/logicals utilities, layout management, multi-phase Stim builder, lattice surgery protocols, noise models, device calibration
- `src/simulation/`: Monte Carlo wrappers, logical error rate evaluation, and threshold utilities
- `scripts/`: reproducibility CLIs (memory, surgery, thresholds, bootstrap)
- `plots/` and `output/`: generated artefacts (figures, CSV/JSON summaries, calibration files)
- `documentation/TECHNICAL_SPECIFICATION.md`: algorithmic/design documentation (recommended reading for method details)
- `test/`: unit tests for surgery geometry/masking, transpilation utilities, and noise models

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



