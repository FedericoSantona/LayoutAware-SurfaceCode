# Technical Specification: Layout-Aware Surface Code Builder for Lattice Surgery

**Version:** 1.0  
**Date:** December 2024  
**Author:** QEC-Project Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Core Challenge: Making Lattice Surgery Work](#3-core-challenge-making-lattice-surgery-work)
4. [Technique 1: Multi-Patch Layout Management](#4-technique-1-multi-patch-layout-management)
5. [Technique 2: Geometry-Based Boundary Detection](#5-technique-2-geometry-based-boundary-detection)
6. [Technique 3: Commuting Boundary Masking](#6-technique-3-commuting-boundary-masking)
7. [Technique 4: Logical Operator Alignment](#7-technique-4-logical-operator-alignment)
8. [Technique 5: Multi-Phase Stim Circuit Building](#8-technique-5-multi-phase-stim-circuit-building)
9. [Technique 6: Seam Ancilla Management](#9-technique-6-seam-ancilla-management)
10. [Technique 7: Pauli Frame Tracking](#10-technique-7-pauli-frame-tracking)
11. [Technique 8: GF(2) Linear Algebra Utilities](#11-technique-8-gf2-linear-algebra-utilities)
12. [Technique 9: Device-Aware Noise Modeling](#12-technique-9-device-aware-noise-modeling)
13. [Module Reference](#13-module-reference)
14. [Known Limitations and Future Work](#14-known-limitations-and-future-work)

---

## 1. Executive Summary

This document describes the techniques developed to build a **Stim circuit builder** for surface codes that supports **lattice surgery** operations and **device-aware noise modeling**. The primary goal is to enable fault-tolerant CNOT gates via smooth and rough merges between logical qubit patches, with realistic noise modeling using device calibration data.

### Key Achievements

- Support for both **heavy-hex** and **standard** surface code layouts
- **Lattice surgery CNOT** implementation following Horsman et al. (2013)
- Correct **time-like detector** wiring across multi-phase protocols
- **Physics mode** for Bell correlator verification of CNOT action
- **Device-aware noise modeling** with per-qubit T1/T2 decoherence from device calibration
- **Automatic layout selection** based on device type (IBM → heavy-hex, IQM Crystal → standard)
- Modular architecture supporting arbitrary code distances

### Critical Problems Solved

| Problem | Solution |
|---------|----------|
| Stabilizer anti-commutation during merges | Commuting boundary mask algorithm |
| Logical operators breaking during rough merge | Logical-X alignment via X stabilizer multiplication |
| Incorrect boundary identification | Geometry-based boundary detection using coordinate analysis |
| Spurious logical dimensions from idle seams | Single-qubit Z pinning of seam ancillas |
| Detector discontinuity across phases | Stabilizer set change detection with warmup rounds |

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    surgery_experiment.py                         │
│         (High-level CNOT experiment orchestration)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LatticeSurgery                              │
│  surgery.py - Multi-patch merge/split protocol construction      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐
│    Layout     │  │  stabilizers  │  │        logicals           │
│  layout.py    │  │ stabilizers.py│  │       logicals.py         │
│               │  │               │  │                           │
│ • Patch offsets│ │ • CSS extract │  │ • find_logicals_*         │
│ • Seam alloc  │  │ • Boundary    │  │ • Logical alignment       │
│ • Boundary map│  │   masking     │  │ • Pauli multiplication    │
└───────┬───────┘  └───────────────┘  └───────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                   geometry_utils.py                            │
│         Coordinate-based boundary identification               │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│              PhenomenologicalStimBuilder                       │
│                    stim_builder.py                             │
│                                                                │
│  • MPP measurement emission                                    │
│  • Time-like detector wiring                                   │
│  • Multi-phase run_phases() API                                │
│  • Observable attachment                                       │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. Core Challenge: Making Lattice Surgery Work

### 3.1 The Lattice Surgery Protocol

Lattice surgery implements logical gates by temporarily merging and splitting surface code patches. For a CNOT gate between control (C) and target (T) patches using an ancilla (INT):

1. **Pre-merge phase**: Three independent patches undergo memory rounds
2. **Smooth merge (C + INT)**: Measure joint Z stabilizers along smooth boundary
3. **Smooth split**: Re-separate C and INT (now entangled)
4. **Rough merge (INT + T)**: Measure joint X stabilizers along rough boundary
5. **Rough split**: Re-separate INT and T
6. **Post-merge phase**: Final memory rounds on separated patches

### 3.2 The Fundamental Problem

When merging patches, we must:

1. **Identify correct boundary qubits** for smooth vs rough boundaries
2. **Add joint stabilizers** (ZZZ for smooth, XXX for rough) across seams
3. **Modify existing stabilizers** so they don't anti-commute with joint checks
4. **Adjust logical operators** to remain valid under modified stabilizers
5. **Wire time-like detectors** correctly when stabilizer sets change

Each step has subtle failure modes that produce invalid Stim circuits or incorrect physics.

---

## 4. Technique 1: Multi-Patch Layout Management

### 4.1 Problem Statement

Surface code patches must be combined into a single qubit index space while:
- Maintaining correct local stabilizer structure per patch
- Allocating seam ancillas between adjacent patches
- Tracking boundary qubits in global coordinates

### 4.2 Solution: The Layout Class

```python
# layout.py

class Layout:
    def __init__(
        self,
        distance: int,
        code_type: str,
        patch_order: List[str],        # e.g., ["C", "INT", "T"]
        seams: List[SeamSpec],         # Adjacency specification
        patch_metadata: Dict[str, str] | None = None,
    ):
        self.single_model = build_surface_code_model(distance, code_type)
        self._calculate_layout()
```

### 4.3 Qubit Index Allocation Algorithm

```
For patch_order = ["C", "INT", "T"] with d=3:

Offset 0:  [Patch C qubits: 0..n_single-1]
Offset n:  [Seam C-INT ancillas: n..n+d-1]
Offset n+d: [Patch INT qubits: n+d..2n+d-1]
Offset 2n+d: [Seam INT-T ancillas: 2n+d..2n+2d-1]
Offset 2n+2d: [Patch T qubits: 2n+2d..3n+2d-1]
```

### 4.4 Implementation Details

```python
def _calculate_layout(self) -> None:
    n_single = self.single_model.code.n
    offset = 0

    for i, name in enumerate(self.patch_order):
        patch_offsets[name] = offset
        offset += n_single

        if i < len(self.patch_order) - 1:
            next_name = self.patch_order[i + 1]
            seam = seams_by_pair.get(frozenset({name, next_name}))
            
            if seam is not None:
                # Seam size = number of boundary qubits
                local_boundary = (self._local_smooth if seam.boundary_type == "smooth" 
                                  else self._local_rough)
                ancilla_indices = list(range(offset, offset + len(local_boundary)))
                seam_qubits[(seam.left, seam.right)] = ancilla_indices
                offset += len(local_boundary)

    self._n_total = offset
```

### 4.5 Embedding Helper

To convert single-patch Pauli strings to global index space:

```python
def _embed_patch(self, pauli_str: str, patch_name: str) -> str:
    offset = self.layout.patch_offsets[patch_name]
    left = "I" * offset
    mid = pauli_str
    right = "I" * (self.n_total - offset - self.n_single)
    return left + mid + right
```

---

## 5. Technique 2: Geometry-Based Boundary Detection

### 5.1 Problem Statement

Surface codes have two types of boundaries:
- **Smooth boundaries**: Where Z stabilizers terminate (logical Z passes through)
- **Rough boundaries**: Where X stabilizers terminate (logical X passes through)

Naive approaches (e.g., using logical operator support) fail because interior logical strings can be mislabeled as boundaries.

### 5.2 Solution: Coordinate + Stabilizer Deficit Analysis

```python
# geometry_utils.py

def find_boundary_data_qubits(
    model: SurfaceCodeModel,
    boundary_type: Literal["smooth", "rough"] = "smooth"
) -> List[int]:
```

**Algorithm:**

1. **Extract qubit coordinates** from code geometry (`shell.vertices`)
2. **Identify geometric sides**: left, right, top, bottom based on coordinate bounds
3. **Count stabilizer participation** per qubit:
   - `num_x[q]` = number of X stabilizers touching qubit q
   - `num_z[q]` = number of Z stabilizers touching qubit q
4. **Compute deficit** for each side:
   - Smooth boundary: large Z stabilizer deficit
   - Rough boundary: large X stabilizer deficit
5. **Select best-matching side** for requested boundary type

### 5.3 Why This Works

True boundary qubits have **fewer stabilizer measurements** because stabilizers physically terminate at the code edge. Interior qubits (even those on logical strings) participate in the full complement of stabilizers.

```python
# Deficit calculation
side_deficits[name] = {
    "def_x": max_x - mean_x,  # X stabilizer deficit
    "def_z": max_z - mean_z,  # Z stabilizer deficit
}

# Smooth boundary = high Z deficit
# Rough boundary = high X deficit
```

### 5.4 Coordinate Extraction

```python
def _get_qubit_coordinates(model: SurfaceCodeModel) -> np.ndarray:
    coords = np.full((n, 2), np.nan, dtype=float)
    for vertex in code.shell.vertices:
        if vertex.id in code.qubit_data.index:
            qubit_idx = code.qubit_data.index[vertex.id]
            coords[qubit_idx] = np.array(vertex.pos, dtype=float)
    return coords
```

---

## 6. Technique 3: Commuting Boundary Masking

### 6.1 Problem Statement

During a **smooth merge**, we add joint Z stabilizers of the form:
```
Z_left ⊗ Z_seam ⊗ Z_right  (on boundary qubits)
```

These new checks may **anti-commute** with existing X stabilizers that overlap the boundary. Similarly, rough merges add joint X checks that may anti-commute with Z stabilizers.

Simply stripping the offending Paulis destroys the CSS structure, breaking commutation relations.

### 6.2 Solution: Tiered Boundary Masking

```python
# stabilizers.py

def _commuting_boundary_mask(
    z_stabilizers: Sequence[str],
    x_stabilizers: Sequence[str],
    boundary: Sequence[int],
    *,
    strip_pauli: str = "Z",  # "Z" for rough merge, "X" for smooth merge
) -> tuple[list[str], list[str]]:
```

**Three-tier strategy (least destructive first):**

#### Tier 1: Primary Strip
Strip the requested Pauli from boundary qubits:
```python
primary_masked = [strip_fn(s, boundary) for s in primary]
```

#### Tier 2: Additive Adjustment
If commutation breaks, solve a GF(2) system to find off-boundary adjustments:
```python
# Solve: which qubits outside boundary can we toggle to restore commutation?
delta = _solve_add_only(primary_matrix, b_vec, free_cols)
for col, val in zip(free_cols, delta):
    if val:
        sec_vec[col] ^= 1  # Toggle secondary Pauli off-boundary
```

#### Tier 3: Fallback Strip
As last resort, iteratively strip overlapping Paulis:
```python
while changed:
    for primary, secondary in pairs:
        if not commutes(primary, secondary):
            target = pick_overlap_qubit(prefer_off_boundary=True)
            strip(target)
```

### 6.3 Example: Smooth Merge

For a smooth merge, we strip X on the smooth boundary:

| Before | After |
|--------|-------|
| X stabilizer: `XXIII` | Masked: `IXIII` |
| Z stabilizer: `ZZIII` | Unchanged: `ZZIII` |
| Joint check: `ZZZ` (on boundary + seam) | Added |

The masking ensures `IXIII` commutes with the joint `ZZZ` check.

---

## 7. Technique 4: Logical Operator Alignment

### 7.1 Problem Statement

During a **rough merge**, we mask Z stabilizers at the rough boundary. This can cause the logical X operator to **anti-commute** with the modified Z checks, breaking the code structure.

Simply stripping the logical X at the boundary would change its logical class (equivalent to applying a logical Z).

### 7.2 Solution: Equivalence Class Adjustment

The key insight is that logical operators are defined **up to multiplication by stabilizers**. We can multiply the logical X by X stabilizers to find an equivalent representative that commutes with the masked Z checks.

```python
# logicals.py

def _align_logical_x_to_masked_z(
    logical_x: str,
    x_stabilizers: Sequence[str],
    masked_z: Sequence[str],
) -> str | None:
```

### 7.3 Algorithm

1. **Check if alignment needed**:
   ```python
   if all(_pauli_commutes(logical_x, z) for z in masked_z):
       return logical_x  # Already commutes
   ```

2. **Build GF(2) system**: Find coefficients `c_i` such that:
   ```
   logical_x * (∏ x_stab[i]^{c_i}) commutes with all masked_z
   ```

3. **Solve for stabilizer coefficients**:
   ```python
   # A[z][stab] = overlap(masked_z[z], x_stab[stab]) mod 2
   # b[z] = overlap(masked_z[z], logical_x) mod 2
   coeffs = _solve_gf2(A, b)
   ```

4. **Apply adjustment**:
   ```python
   for idx, (coeff, x_row) in enumerate(zip(coeffs, x_rows)):
       if coeff:
           delta = xor(delta, x_row)
   aligned = xor(logical_x, delta)
   ```

### 7.4 Correctness Guarantee

The aligned logical X:
- **Commutes** with all masked Z stabilizers (by construction)
- **Anti-commutes** with logical Z (preserved, since X stabilizers commute with Z_L)
- **Equivalent to original** in the code's logical Hilbert space

---

## 8. Technique 5: Multi-Phase Stim Circuit Building

### 8.1 Problem Statement

Lattice surgery requires multiple phases with **different stabilizer sets**:
- Pre-merge: 3 independent patches
- Smooth merge: C+INT joint stabilizers
- Split: back to independent
- Rough merge: INT+T joint stabilizers
- Post-merge: final independent patches

Stim's detector model requires correct **time-like detector wiring** — comparing consecutive measurements of the same stabilizer.

### 8.2 Solution: PhaseSpec + run_phases()

```python
# stim_builder.py

@dataclass
class PhaseSpec:
    name: str
    z_stabilizers: Sequence[str]
    x_stabilizers: Sequence[str]
    rounds: int
    measure_z: bool | None = None
    measure_x: bool | None = None
```

### 8.3 Detector Continuity Handling

```python
def run_phases(self, circuit, phases, config):
    sz_prev = None  # Previous Z measurement indices
    sx_prev = None  # Previous X measurement indices
    prev_z_set = None
    prev_x_set = None

    for phase in phases:
        # CRITICAL: Reset detectors when stabilizer sets change
        if prev_z_set is not None and z_stabs != list(prev_z_set):
            sz_prev = None  # Can't compare to different stabilizers
        if prev_x_set is not None and x_stabs != list(prev_x_set):
            sx_prev = None

        sz_prev, sx_prev = self._run_css_block(
            circuit,
            z_stabilizers=z_stabs,
            x_stabilizers=x_stabs,
            sz_prev=sz_prev,
            sx_prev=sx_prev,
            ...
        )
        
        prev_z_set = z_stabs
        prev_x_set = x_stabs
```

### 8.4 Warmup Rounds

Each phase with new stabilizers needs a **warmup round** — a measurement that serves as the reference for the first detector:

```python
def _run_css_block(...):
    # Warmup if this is a fresh stabilizer set
    if measure_Z and z_stabs and sz_prev is None:
        circuit.append_operation("TICK")
        sz_prev = self._measure_list(circuit, z_stabs, round_index=-1)
    
    # Main rounds with detectors
    for round_idx in range(rounds):
        sz_curr = self._measure_list(circuit, z_stabs, round_index=round_idx)
        if sz_prev is not None:
            self._add_detectors(circuit, sz_prev, sz_curr)
        sz_prev = sz_curr
```

### 8.5 Measurement Metadata

For debugging and Pauli frame tracking, we record metadata per measurement:

```python
self._meas_meta[idx] = {
    "family": "Z",
    "round": round_idx,
    "stab_index": stab_index,
    "pauli": pauli_string,
    "phase": phase_name,
}
```

---

## 9. Technique 6: Seam Ancilla Management

### 9.1 Problem Statement

Seam qubits between patches must be:
- **Active** during merges (participating in joint ZZZ or XXX checks)
- **Pinned** during idle phases (not contributing spurious logical dimension)

### 9.2 Solution: Single-Qubit Z Pinning

When a seam is idle, we add single-qubit Z stabilizers on each seam ancilla:

```python
# surgery.py

def _seam_idle_stabilizers(self, skip_seam=None):
    z_pins = []
    for seam, ancillas in self.layout.seam_qubits.items():
        if seam == skip_seam:
            continue  # This seam is active in merge
        for q in ancillas:
            chars = ["I"] * self.n_total
            chars[q] = "Z"
            z_pins.append("".join(chars))
    return z_pins, []
```

### 9.3 Usage in Phase Construction

```python
# Pre-merge: all seams pinned
base_z, base_x = self._base_stabilizers(patches)
all_seam_z, _ = self._seam_idle_stabilizers()
phase = PhaseSpec("pre-merge", base_z + all_seam_z, base_x, rounds_pre)

# Smooth merge C+INT: only INT-T seam pinned
smooth_merge_z, smooth_merge_x = self._smooth_merge_stabilizers(...)
smooth_merge_z += self._seam_idle_stabilizers(skip_seam=("C", "INT"))[0]
```

### 9.4 Code Parameter Verification

We verify correct logical dimension `k` per phase:

```python
def _phase_k(self, z_stabs, x_stabs):
    S = stabs_to_symplectic(z_stabs, x_stabs)
    r = rank_gf2(S)
    return self.n_total - r

# Expected: k=3 pre-merge, k=2 during merges, k=2 post-merge (after CNOT)
```

---

## 10. Technique 7: Pauli Frame Tracking

### 10.1 Problem Statement

For physics verification (Bell correlator measurements), we need to track how logical operators evolve through measurements. When a logical anti-commutes with a measured stabilizer, the measurement outcome **flips the logical eigenvalue**.

### 10.2 Solution: Propagation Through Measurements

```python
# surgery_experiment.py

def _propagate_logicals_through_measurements(
    logicals: dict[str, str],
    meas_meta: dict[int, dict],
    patch_logicals: dict | None = None,
) -> tuple[dict[str, str], dict[str, list[int]]]:
```

### 10.3 Algorithm

For each measurement (in circuit order):

1. **Check symplectic product** with each tracked logical:
   ```python
   if _symplectic_product(logical_vec, measured_pauli_vec):
       # Anti-commutation detected
       deps[logical_name].append(measurement_index)
       logical_vec = xor(logical_vec, measured_pauli_vec)
   ```

2. **Track stabilizer correlations**: When measuring `Z_C * Z_INT`, the product `X_C * X_INT` becomes a stabilizer

3. **Apply CNOT transformation**: When `X_INT * X_T` is measured:
   - `X_C → X_C * X_T`
   - `Z_T → Z_C * Z_T`

### 10.4 Bell Correlator Dressing

The final Bell measurement must be **dressed** with all frame bits:

```python
xx_indices = (
    [init_C, init_INT]     # Preparation bits
    + frame_xx             # Anti-commutation history
    + [xx_final_meas]      # Final Bell measurement
)

# Correlator = ∏ (-1)^{meas[i]} for i in xx_indices
```

---

## 11. Technique 8: GF(2) Linear Algebra Utilities

### 11.1 Core Operations

```python
# linalg.py

def rank_gf2(matrix: np.ndarray) -> int:
    """Gaussian elimination over GF(2) to compute matrix rank."""

def nullspace_gf2(matrix: np.ndarray) -> list[np.ndarray]:
    """Basis vectors for kernel of matrix over GF(2)."""

def row_in_span_gf2(matrix: np.ndarray, row: np.ndarray) -> bool:
    """Check if row is in row space of matrix."""

def _solve_gf2(A: list[list[int]], b: list[int]) -> list[int] | None:
    """Solve Ax = b over GF(2), return one solution or None."""
```

### 11.2 Symplectic Representation

Pauli strings are represented in symplectic form `[Z | X]`:

```python
def symplectic_to_pauli(vec: np.ndarray) -> str:
    """[Z|X] -> Pauli string: 00=I, 10=Z, 01=X, 11=Y"""
    n = len(vec) // 2
    for z, x in zip(vec[:n], vec[n:]):
        if z and x: letters.append("Y")
        elif z: letters.append("Z")
        elif x: letters.append("X")
        else: letters.append("I")
    return "".join(letters)
```

### 11.3 Commutation Check

Two Paulis commute iff their symplectic inner product is zero:

```python
def _pauli_commutes(a: str, b: str) -> bool:
    anti = 0
    for pa, pb in zip(a, b):
        if pa == "I" or pb == "I":
            continue
        if pa != pb:
            anti ^= 1
    return anti == 0
```

---

## 12. Technique 9: Device-Aware Noise Modeling

### 12.1 Overview

For **threshold studies**, noise is expressed in terms of **Pauli X and Z error rates** (`p_x`, `p_z`) per qubit per round (or per step): the decoder sees the same kind of syndrome statistics whether those rates come from a simple phenomenological model or from device calibration. The codebase supports both:

- **Device-aware noise:** Per-qubit `p_x` and `p_z` are *computed* from calibration (T1/T2, gate errors, readout errors). The simulator still applies `X_ERROR` and `Z_ERROR` with these effective rates, so threshold curves and logical error rates are directly comparable to phenomenological studies.
- **Phenomenological noise:** The user sets uniform `p_x` and `p_z` directly; useful for theory, scaling plots, or as a baseline.

So for threshold study one always uses **p_x and p_z errors** in the underlying Stim circuit; the device-aware model is a way to obtain those rates from real device parameters rather than from hand-picked values.

### 12.2 Noise Model Architecture

The noise model system uses an abstract base class `NoiseModel` with two implementations:

```python
class NoiseModel(ABC):
    @abstractmethod
    def apply_data_qubit_noise(circuit, qubits, duration) -> None
    @abstractmethod
    def apply_measurement_noise(circuit, qubits) -> None
    @abstractmethod
    def get_effective_error_rate(qubit, error_type) -> float
```

**DeviceAwareNoiseModel** (primary implementation): Uses per-qubit calibration data:
- Computes error rates from T1/T2 decoherence times
- Applies per-qubit readout errors
- Supports gate-dependent errors (optional)
- Enables realistic simulations with actual device parameters

**PhenomenologicalNoiseModel** (alternative experimental approach): Uniform error rate model:
- Applies uniform `p_x` and `p_z` errors to all qubits
- Useful for theoretical studies with idealized uniform noise or when comparing against device-specific results

### 12.3 Device-Aware Noise Model: Technical Specification

The **DeviceAwareNoiseModel** (`src/surface_code/noise_model.py`) implements per-qubit and per-coupler noise from device calibration. For threshold studies, it still injects noise as **p_x and p_z** (via Stim `X_ERROR` and `Z_ERROR`); those rates are derived from T1/T2 and gate errors instead of being set directly. The model supports four error channels: idle decoherence (→ p_x, p_z), gate errors (depolarizing), readout errors, and optional ZZ crosstalk.

#### 12.3.1 Parameter Structures

**Per-qubit parameters** (`QubitNoiseParams`):

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `t1` | float | µs | T1 relaxation time (amplitude damping). Must be > 0. |
| `t2` | float | µs | T2 dephasing time (includes T1). Must satisfy T2 ≤ 2·T1. |
| `readout_error_0to1` | float | — | Probability of measuring 1 given state \|0⟩. |
| `readout_error_1to0` | float | — | Probability of measuring 0 given state \|1⟩. |
| `single_qubit_gate_error` | float | — | Average single-qubit gate error rate (depolarizing). |
| `frequency` | float or None | GHz | Optional; reserved for crosstalk/frequency-dependent modeling. |

Derived quantities:
- **Pure dephasing time:** `T2_phi = 1 / (1/T2 - 1/(2*T1))`. If T2 = 2·T1, pure dephasing is infinite (no dephasing channel).
- **Average readout error:** `(readout_error_0to1 + readout_error_1to0) / 2`, used when applying readout noise.

**Per-coupler parameters** (`CouplerNoiseParams`), keyed by qubit pair `(q1, q2)` with q1 < q2:

| Field | Type | Description |
|-------|------|-------------|
| `two_qubit_gate_error` | float | Two-qubit gate error rate (e.g. CX/CZ), applied as two-qubit depolarizing noise. |
| `crosstalk_strength` | float | ZZ crosstalk rate; Z error probability on the idle qubit when the neighbor is active. |

**Model configuration:**
- `default_round_duration` (float, µs): Default idle duration for decoherence when `duration` is not provided (e.g. 1.0 µs).
- `gate_times` (dict): Gate name → duration in µs. Defaults include: `sx`/`x`/`h` 0.035, `s` 0 (virtual Z), `cx`/`cz` 0.3, `measure` 1.0.

#### 12.3.2 Idle / Data-Qubit Noise (Decoherence)

**Method:** `apply_data_qubit_noise(circuit, qubits, duration=None)`.

For each qubit in `qubits`, the model computes Pauli error rates from T1/T2 and optional gate-error contribution, then appends Stim noise operations.

1. **Amplitude damping (T1):**  
   `p_ad = 1 - exp(-t / T1)` with `t = duration` (µs) or `default_round_duration`.

2. **Pure dephasing (T2_phi):**  
   `p_deph = 0.5 * (1 - exp(-t / T2_phi))` if T2_phi is finite; else 0.

3. **Pauli rates from decoherence (small-error approximation):**  
   `p_x = p_ad/4`,  
   `p_z = p_deph/2 + p_ad/4`.

4. **Depolarizing contribution from single-qubit gate error:**  
   `p_x += single_qubit_gate_error/3`,  
   `p_z += single_qubit_gate_error/3`.

5. **Clamping:** `p_x` and `p_z` are clamped to [0, 0.5].

6. **Stim operations:** For each qubit, append `X_ERROR(qubit, p_x)` and `Z_ERROR(qubit, p_z)` when the rate is > 0. Qubits not present in `qubit_params` receive no idle noise.

#### 12.3.3 Gate Noise

**Method:** `apply_gate_noise(circuit, gate_name, qubits)`.

- **Single-qubit gates:** After the gate, append `DEPOLARIZE1(qubit, single_qubit_gate_error)` using that qubit’s `single_qubit_gate_error` from `QubitNoiseParams`.
- **Two-qubit gates:** After the gate, append `DEPOLARIZE2(q1, q2, two_qubit_gate_error)` using the coupler’s `two_qubit_gate_error` for the pair `(min(q1,q2), max(q1,q2))`. If the pair is not in `coupler_params`, no gate noise is applied.

Gate names are matched case-insensitively (e.g. `"CX"`, `"cx"`).

#### 12.3.4 Measurement / Readout Noise

**Method:** `apply_measurement_noise(circuit, qubits)`.

For each qubit in `qubits` that exists in `qubit_params`, the model uses the average readout error `p_readout = (readout_error_0to1 + readout_error_1to0) / 2` and appends `X_ERROR(qubit, p_readout)` before the measurement. This approximates asymmetric readout errors as a symmetric bit-flip channel. Qubits not in `qubit_params` are skipped.

#### 12.3.5 Crosstalk (Optional)

**Method:** `apply_crosstalk(circuit, active_qubits, all_qubits)`.

For each coupler `(q1, q2)` with `crosstalk_strength > 0`: if one of `q1`, `q2` is in `active_qubits` and the other is idle (in `all_qubits` but not in `active_qubits`), append `Z_ERROR(idle_qubit, crosstalk_strength)`. This models ZZ crosstalk as a Z error on the idle qubit. This method is only applied when explicitly invoked by the circuit construction path (e.g. circuit-level builder).

#### 12.3.6 Effective Error Rates and Unknown Qubits

**Method:** `get_effective_error_rate(qubit, error_type)` with `error_type` in `'x'`, `'z'`, `'readout'`, `'gate'`.

- For `'x'` and `'z'`: uses `_compute_pauli_rates(qubit, default_round_duration)` and returns the corresponding rate.
- For `'readout'`: returns the qubit’s average readout error.
- For `'gate'`: returns the qubit’s `single_qubit_gate_error`.
- If the qubit is not in `qubit_params`, returns 0.0.

#### 12.3.7 Integration Points

- **Phenomenological Stim builder:** Uses `apply_data_qubit_noise` once per measurement round (when applying “x” or “both”); does not call `apply_gate_noise` or `apply_measurement_noise` in the MPP flow.
- **Circuit-level Stim builder:** Calls `apply_gate_noise` after each single- and two-qubit gate, and `apply_data_qubit_noise` for idle qubits during each gate (with the gate duration as `duration`). Does not call `apply_measurement_noise` or `apply_crosstalk` unless explicitly wired elsewhere.

Calibration is supplied via `DeviceCalibration.to_noise_model(default_round_duration)` (see §12.5).

### 12.4 T1/T2 to Pauli Error Conversion (Formulas)

The device-aware model converts coherence times to Pauli error rates using physics-based formulas:

**Amplitude damping** (T1 relaxation):
```
p_ad = 1 - exp(-t/T1)
```

**Pure dephasing** (T2* contribution):
```
T2_phi = 1/(1/T2 - 1/(2*T1))
p_deph = 0.5 * (1 - exp(-t/T2_phi))
```

**Pauli error rates** (approximate for small errors):
```
p_x ≈ p_ad / 4
p_z ≈ p_deph / 2 + p_ad / 4
```

These approximations are valid when error rates are small (typical for surface code threshold studies).

### 12.5 Device Calibration System

The `DeviceCalibration` class provides a unified interface for loading device parameters:

**Sources:**
- IBM Quantum backends: `DeviceCalibration.from_ibm_backend(backend)` (primary method for real device data)
- JSON files: `DeviceCalibration.from_json(path)` (for saved/reproducible calibration)
- Synthetic: `DeviceCalibration.uniform(...)` (for theoretical studies with idealized parameters)

**Calibration data structure:**
```json
{
  "backend_name": "ibm_sherbrooke",
  "timestamp": "2026-01-22T12:00:00Z",
  "qubits": {
    "0": {
      "t1": 150.0,           // microseconds
      "t2": 100.0,           // microseconds
      "readout_error_0to1": 0.015,
      "readout_error_1to0": 0.020,
      "gate_error": 0.0005,
      "frequency": 5.05      // GHz (optional)
    }
  },
  "couplers": {
    "0-1": {
      "cx_error": 0.008,
      "crosstalk": 0.001
    }
  },
  "gate_times": {
    "sx": 0.035,    // microseconds
    "cx": 0.300,
    "measure": 1.0
  }
}
```

### 12.6 Automatic Layout Selection

The system automatically selects the appropriate surface code layout based on the device type:

- **IBM devices** → `heavy_hex` layout (native to IBM's heavy-hex architecture)
- **IQM Crystal** → `standard` layout (planar surface code)

This is implemented via `DeviceCalibration.get_recommended_layout()`, which detects the device type from the backend name and returns the appropriate layout string.

**Detection logic:**
```python
def get_recommended_layout(self) -> str:
    backend_lower = self.backend_name.lower()
    if "ibm" in backend_lower or known_ibm_device(backend_lower):
        return "heavy_hex"
    if "iqm" in backend_lower or "crystal" in backend_lower:
        return "standard"
    return "heavy_hex"  # default
```

When using device-aware noise in `threshold_experiment.py`, if the user-specified layout conflicts with the recommended layout, a warning is displayed and the layout is automatically switched.

#### 12.6.1 Qubit requirements for threshold experiments (heavy-hex)

For **heavy-hex** layout, the number of physical data qubits is \(n = (5d^2 - 2d - 1)/2\) (one logical qubit, distance \(d\)). Typical threshold sweeps use distances 3, 5, 7, and optionally 9:

| Distance \(d\) | Physical qubits \(n\) |
|----------------|------------------------|
| 3              | 19                     |
| 5              | 57                     |
| 7              | 115                    |
| 9              | 193                    |

The phenomenological Stim builder uses only these **data** qubits (no explicit ancilla wires in the circuit), so a device must have **at least \(n\) qubits** to run (or to supply calibration for) a threshold experiment at that distance.

**IBM devices that can support threshold experiments (calibration source):**

- **127-qubit systems** (e.g. **ibm_sherbrooke**, **ibm_kyoto**): \(n = 115\) for \(d = 7\), so they **can** support threshold experiments up to **distance 7** (e.g. distances 3, 5, 7). Calibration from these backends is suitable for device-aware threshold studies with `--distances 3 5 7`.
- **Larger systems** (e.g. 133+ qubits): Still cap at \(d = 7\) for the formula above; \(d = 9\) requires **193 qubits**, so no current IBM device has enough qubits for a full heavy-hex \(d = 9\) threshold run. For \(d = 9\) one would need either a future device with ≥193 qubits or a different code/layout.

**Recommendation:** Use **ibm_sherbrooke** or **ibm_kyoto** (127 qubits) to pull calibration via `DeviceCalibration.from_ibm_backend(backend)` for device-aware threshold experiments with default distances 3, 5, 7. Save calibration to JSON for reproducibility (see §12.5).

### 12.7 Integration with Stim Builder

The noise model is integrated into `PhenomenologicalStimConfig`:

```python
@dataclass
class PhenomenologicalStimConfig:
    rounds: int = 5
    p_x_error: float = 1e-3      # Used if noise_model is None
    p_z_error: float = 1e-3
    noise_model: Optional[NoiseModel] = None  # Device-aware noise (recommended)
```

The `PhenomenologicalStimBuilder._apply_noise()` method delegates to the noise model if provided, otherwise uses uniform error rates.

### 12.8 Current Limitations

The device-aware noise model applies errors **phenomenologically** (after each measurement round) rather than at the circuit level:
- Errors are applied after each measurement round, not during explicit gate execution
- Syndrome extraction uses MPP operations (not explicit ancilla + CNOT gates)
- Idle noise is approximated based on round duration

**Future extension**: Circuit-level noise with explicit gate decomposition would provide even more realistic modeling by applying noise after each individual gate operation.

---

## 13. Module Reference

| Module | Purpose | Key Functions/Classes |
|--------|---------|----------------------|
| `model.py` | Abstract surface code interface | `SurfaceCodeModel` |
| `heavy_hex.py` | Heavy-hex code builder | `build_heavy_hex_model()` |
| `standard.py` | Standard surface code builder | `build_standard_surface_code_model()` |
| `layout.py` | Multi-patch layout management | `Layout`, `SeamSpec` |
| `geometry_utils.py` | Boundary detection | `find_smooth_boundary_data_qubits()`, `find_rough_boundary_data_qubits()` |
| `stabilizers.py` | CSS stabilizer utilities | `extract_css_stabilizers()`, `_commuting_boundary_mask()` |
| `logicals.py` | Logical operator utilities | `find_logicals_general()`, `_align_logical_x_to_masked_z()` |
| `linalg.py` | GF(2) linear algebra | `rank_gf2()`, `nullspace_gf2()`, `_solve_gf2()` |
| `stim_builder.py` | Stim circuit construction | `PhenomenologicalStimBuilder`, `PhaseSpec` |
| `surgery.py` | Lattice surgery protocols | `LatticeSurgery`, `CNOTSpec` |
| `noise_model.py` | Noise model abstraction | `NoiseModel`, `PhenomenologicalNoiseModel`, `DeviceAwareNoiseModel` |
| `device_calibration.py` | Device calibration loader | `DeviceCalibration`, `QubitNoiseParams`, `CouplerNoiseParams` |

---

## 14. Known Limitations and Future Work

### 14.1 Current Limitations

1. **Phenomenological error application**: Device-aware noise uses T1/T2 but applies errors after measurement rounds (not circuit-level with explicit gates)
2. **Single logical qubit per patch**: No support for multi-qubit patches
3. **Linear patch arrangement**: Patches arranged in 1D order only
4. **Fixed merge sequences**: Hardcoded smooth→rough for CNOT

### 14.2 Future Extensions

1. **Circuit-level noise**: Decompose MPP operations into explicit gates (H, CNOT, measure) with noise applied after each gate
2. **Explicit ancilla qubits**: Allocate ancilla qubits for each stabilizer measurement
3. **Arbitrary patch topologies**: Support 2D patch arrangements
4. **Additional gates**: Hadamard, S gate via surgery/injection
5. **State injection**: Magic state preparation and distillation
6. **Real-time decoding**: Integration with streaming decoders
7. **More device support**: Additional calibration loaders for other quantum processors

### 13.3 Performance Considerations

- Stabilizer set comparison uses Python list equality (O(n·m) for n stabilizers of length m)
- GF(2) operations use pure Python; could benefit from Cython/NumPy optimization
- Memory metadata tracking grows linearly with circuit depth

---

## Appendix A: CNOT Phase Sequence Detail

```
Phase 1: pre-merge
  - 3 independent patches (C, INT, T)
  - All seam ancillas pinned with Z
  - k = 3 logical qubits

Phase 2: C+INT smooth merge
  - Joint ZZZ checks on C-INT boundary + seam
  - X stabilizers masked at smooth boundary
  - Logical Z_C * Z_INT measured (becomes stabilizer)
  - INT-T seam still pinned
  - k = 2

Phase 3: C|INT smooth split
  - Return to independent stabilizers
  - All seams pinned
  - k = 3 (but C and INT are now entangled)

Phase 4: INT+T rough merge
  - Joint XXX checks on INT-T boundary + seam
  - Z stabilizers masked at rough boundary
  - Logical X aligned to commute with masked Z
  - Logical X_INT * X_T measured
  - C-INT seam pinned
  - k = 2

Phase 5: INT|T rough split
  - Return to independent stabilizers
  - All seams pinned
  - k = 3

Phase 6: post-merge
  - Independent patches
  - CNOT has been applied: X_C → X_C X_T, Z_T → Z_C Z_T
  - k = 3
```

---

## Appendix B: Commutation Verification Example

For distance-3 standard surface code with rough boundary at qubits `[2, 4]`:

**Before masking:**
```
Z stabilizer: ZZIZI (acts on qubits 0,1,3)
X stabilizer: XIXXX (acts on qubits 0,2,3,4)
```

**After rough boundary mask (strip Z at boundary):**
```
Z stabilizer (masked): ZZIZI → ZZIZI (no change, no Z at boundary)
Z stabilizer 2 (masked): IIZZI → IIZII (stripped at qubit 4)
```

**Joint X check added:**
```
IIXIX (qubits 2, 4 on boundary + seam)
```

**Commutation check:**
- `IIZII` and `XIXXX`: overlap at qubit 3, both have one Pauli → anti-commute!
- Apply Tier 2: add X at qubit 0 to X stabilizer
- `IIZII` and `XXXXX`: overlap at 0,3, even count → commute ✓

---

*End of Technical Specification*
