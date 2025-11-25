"""Lattice-surgery CNOT experiment.

This script is intended to sit next to `memory_experiment.py` and reuse the same
Stim + PyMatching Monte Carlo infrastructure, but for a *gate* experiment
instead of a memory experiment.

Conceptually, we follow the CNOT construction of Horsman et al.,
"Surface code quantum computing by lattice surgery" (2013): two planar
logical patches (control C and target T) and an intermediate logical patch
(INT) prepared in |+>_L. The CNOT is realised by a smooth merge + split
between C and INT followed by a rough merge between INT and T.

This file does **not** yet implement the full geometry-specific stabilizers
for those merge/split phases. Instead, it provides a clear *template* showing
how to:

  * plug a multi-phase lattice-surgery circuit into the existing
    `PhenomenologicalStimConfig` + Monte Carlo pipeline;
  * where the per-phase Z/X stabilizers for the three-patch layout
    need to be supplied;
  * how and where to attach the logical CNOT observable.

Once the geometry-dependent stabilizer lists are implemented, the
`build_cnot_surgery_circuit` function will produce a Stim circuit that you
can feed directly to `run_circuit_logical_error_rate` to estimate a logical
CNOT error rate as a function of distance and noise parameters.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import matplotlib.pyplot as plt

import stim

# Make `src/` importable, mirroring memory_experiment.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from surface_code import (
    PhenomenologicalStimBuilder,
    PhenomenologicalStimConfig,
    Layout,
    SeamSpec,
    build_surface_code_model,
    find_smooth_boundary_data_qubits,
    find_rough_boundary_data_qubits,
)
from simulation import MonteCarloConfig, run_circuit_logical_error_rate


# ---------------------------------------------------------------------------
# Phase description
# ---------------------------------------------------------------------------


@dataclass
class PhaseSpec:
    """Describe one spacetime phase of the lattice-surgery protocol.

    Each phase uses a *single* CSS stabilizer family (Z and X) on a *fixed*
    layout of qubits (three patches plus any seam / intermediate qubits).
    The differences between phases come solely from which stabilizers are
    measured and for how many rounds.

    Attributes
    ----------
    name:
        Human-readable label, e.g. "pre-merge", "C+INT smooth merge".
    z_stabilizers / x_stabilizers:
        Lists of Pauli strings (on the *combined* code) describing the Z- and
        X-type checks to measure in this phase.
    rounds:
        Number of repeated stabilizer-measurement rounds in this phase.
    measure_z / measure_x:
        Optional flags to explicitly control whether Z or X stabilizers are
        measured in this phase. If None, uses the default behavior based on
        config.family and whether stabilizers are provided.
    """

    name: str
    z_stabilizers: Sequence[str]
    x_stabilizers: Sequence[str]
    rounds: int
    measure_z: bool | None = None
    measure_x: bool | None = None


# ---------------------------------------------------------------------------
# Core circuit builder
# ---------------------------------------------------------------------------


# Helper to strip Xs on specified qubits from a Pauli string
from typing import Sequence

def _strip_x_on_qubits(pauli_str: str, qubits: Sequence[int]) -> str:
    """Return a Pauli string with Xs removed on the given local qubits.

    This is used to modify X-type stabilizers near the C–INT smooth
    boundary so that, during the smooth-merge phase, the joint Z checks
    added along the seam do not anti-commute with any X stabilizers.

    The input `pauli_str` is in the single-patch index space; the caller
    is responsible for embedding it into the combined index space.
    """
    chars = list(pauli_str)
    n = len(chars)
    for q in qubits:
        if 0 <= q < n and chars[q] == "X":
            chars[q] = "I"
    return "".join(chars)

def _strip_z_on_qubits(pauli_str: str, qubits: Sequence[int]) -> str:
    """Return a Pauli string with Zs removed on the given local qubits.

    This is used to modify Z-type stabilizers near the INT–T rough
    boundary so that, during the rough-merge phase, the joint X checks
    added along the seam do not anti-commute with any Z stabilizers.

    The input `pauli_str` is in the single-patch index space; the caller
    is responsible for embedding it into the combined index space.
    """
    chars = list(pauli_str)
    n = len(chars)
    for q in qubits:
        if 0 <= q < n and chars[q] == "Z":
            chars[q] = "I"
    return "".join(chars)

def _pauli_commutes(a: str, b: str) -> bool:
    """Return True iff two Pauli strings commute (ignoring identities)."""
    anti = 0
    for pa, pb in zip(a, b):
        if pa == "I" or pb == "I":
            continue
        if pa != pb:
            anti ^= 1
    return anti == 0

def _commuting_boundary_mask(
    z_stabilizers: Sequence[str],
    x_stabilizers: Sequence[str],
    boundary: Sequence[int],
    *,
    strip_pauli: str = "Z",
    verbose: bool = False,
) -> tuple[list[str], list[str]]:
    """Strip only what is needed at a boundary while keeping CSS checks commuting.

    The routine is symmetric:
      * `strip_pauli="Z"`: strip Z on the boundary (for rough merge) and adjust X.
      * `strip_pauli="X"`: strip X on the boundary (for smooth merge) and adjust Z.

    Strategy (least destructive first):
      1) Strip boundary support of the requested Pauli.
      2) If commutation is broken, try toggling the *other* Pauli off-boundary
         to restore commutation without throwing away boundary information.
      3) As a last resort, strip overlapping Paulis (prefer off-boundary)
         until everything commutes.
    """
    strip_pauli = strip_pauli.upper()
    if strip_pauli not in {"X", "Z"}:
        raise ValueError("strip_pauli must be 'X' or 'Z'")

    primary_char = strip_pauli
    secondary_char = "X" if primary_char == "Z" else "Z"
    strip_fn = _strip_z_on_qubits if primary_char == "Z" else _strip_x_on_qubits
    primary = z_stabilizers if primary_char == "Z" else x_stabilizers
    secondary = x_stabilizers if primary_char == "Z" else z_stabilizers

    def _return_order(primary_list: list[str], secondary_list: list[str]) -> tuple[list[str], list[str]]:
        if primary_char == "Z":
            return primary_list, secondary_list
        return secondary_list, primary_list

    def _commutes(p: str, q: str) -> bool:
        anti = 0
        for a, b in zip(p, q):
            if a == "I" or b == "I":
                continue
            if a != b:
                anti ^= 1
        return anti == 0

    def _pauli_weight(chars: Sequence[str]) -> int:
        return sum(c != "I" for c in chars)

    def _solve_add_only(z_matrix: list[list[int]], b_vec: list[int], free_cols: list[int]) -> list[int] | None:
        """Solve z_matrix[:, free_cols] * delta = b_vec over GF(2)."""
        if not z_matrix:
            return [0] * len(free_cols)
        m = len(z_matrix)
        n = len(free_cols)
        aug = [[z_matrix[r][c] for c in free_cols] + [b_vec[r] % 2] for r in range(m)]
        row = 0
        pivots: list[int] = []
        for col in range(n):
            pivot = next((r for r in range(row, m) if aug[r][col]), None)
            if pivot is None:
                continue
            aug[row], aug[pivot] = aug[pivot], aug[row]
            pivots.append(col)
            for r in range(m):
                if r != row and aug[r][col]:
                    for c in range(col, n + 1):
                        aug[r][c] ^= aug[row][c]
            row += 1
        for r in range(row, m):
            if aug[r][n]:
                return None
        delta = [0] * n
        for r in range(row - 1, -1, -1):
            col = pivots[r]
            rhs = aug[r][n]
            for c in range(col + 1, n):
                rhs ^= aug[r][c] & delta[c]
            delta[col] = rhs
        return delta

    primary_masked = [strip_fn(s, boundary) for s in primary]
    secondary_preserved = list(secondary)

    boundary_removed = sum(
        1 for s, sm in zip(primary, primary_masked) for a, b in zip(s, sm) if a != b
    )

    # Fast path: if keeping X intact still commutes with masked Z, return early.
    if all(_commutes(p, s) for p in primary_masked for s in secondary_preserved):
        if verbose:
            print(
                f"[boundary-mask] stripped {primary_char}@boundary={boundary_removed}, "
                f"{secondary_char} preserved (already commuting)"
            )
        return _return_order(primary_masked, secondary_preserved)

    if verbose:
        print(
            f"[boundary-mask] stripped {primary_char}@boundary={boundary_removed}, "
            f"{secondary_char} needs adjustment to commute"
        )

    primary_matrix = [[1 if c == primary_char else 0 for c in row] for row in primary_masked]
    secondary_adjusted: list[str] = []
    additive_ok = True
    for stab in secondary_preserved:
        sec_vec = [1 if c == secondary_char else 0 for c in stab]
        b_vec = [sum(a * b for a, b in zip(row, sec_vec)) % 2 for row in primary_matrix]
        free_cols = [i for i in range(len(sec_vec)) if i not in boundary]
        delta = _solve_add_only(primary_matrix, b_vec, free_cols) if free_cols else None
        if delta is None:
            additive_ok = False
            break
        for col, val in zip(free_cols, delta):
            if val:
                sec_vec[col] ^= 1
        secondary_adjusted.append("".join(secondary_char if v else "I" for v in sec_vec))

    if additive_ok and all(_commutes(p, s) for p in primary_masked for s in secondary_adjusted):
        if verbose:
            added = sum(
                1 for s_old, s_new in zip(secondary_preserved, secondary_adjusted) for a, b in zip(s_old, s_new) if a != b
            )
            print(f"[boundary-mask] additive solution used, {secondary_char} added off-boundary={added}")
        return _return_order(primary_masked, secondary_adjusted)

    # Fallback: strip overlapping Paulis until everything commutes.
    primary_chars = [list(s) for s in primary_masked]
    secondary_chars = [list(s) for s in secondary_preserved]
    changed = True
    fallback_primary_strips = 0
    fallback_secondary_strips = 0
    while changed:
        changed = False
        for pi, p in enumerate(primary_chars):
            for si, s in enumerate(secondary_chars):
                if _commutes("".join(p), "".join(s)):
                    continue
                overlap = [
                    q
                    for q, (a, b) in enumerate(zip(p, s))
                    if a != "I" and b != "I" and a != b
                ]
                if not overlap:
                    continue
                target = next((q for q in overlap if q not in boundary), overlap[0])
                if _pauli_weight(p) <= 1 and _pauli_weight(s) > 1:
                    s[target] = "I"
                    secondary_chars[si] = s
                    fallback_secondary_strips += 1
                else:
                    p[target] = "I"
                    fallback_primary_strips += 1
                changed = True

    if verbose:
        print(
            "[boundary-mask] fallback used, "
            f"extra {primary_char} stripped={fallback_primary_strips}, "
            f"extra {secondary_char} stripped={fallback_secondary_strips}"
        )
    return _return_order(["".join(p) for p in primary_chars], ["".join(s) for s in secondary_chars])


def _solve_gf2(A: list[list[int]], b: list[int]) -> list[int] | None:
    """Solve A x = b over GF(2); return one solution or None if inconsistent."""
    if not A:
        return [0] * (len(A[0]) if b else 0)
    m, n = len(A), len(A[0])
    aug = [row[:] + [b[i] & 1] for i, row in enumerate(A)]
    r = 0
    pivots: list[int] = []
    for c in range(n):
        pivot = next((i for i in range(r, m) if aug[i][c]), None)
        if pivot is None:
            continue
        aug[r], aug[pivot] = aug[pivot], aug[r]
        pivots.append(c)
        for i in range(m):
            if i != r and aug[i][c]:
                for j in range(c, n + 1):
                    aug[i][j] ^= aug[r][j]
        r += 1
    for i in range(r, m):
        if aug[i][n]:
            return None
    x = [0] * n
    for i in range(r - 1, -1, -1):
        c = pivots[i]
        val = aug[i][n]
        for j in range(c + 1, n):
            val ^= aug[i][j] & x[j]
        x[c] = val
    return x


def _align_logical_x_to_masked_z(
    logical_x: str | None,
    x_stabilizers: Sequence[str],
    masked_z: Sequence[str],
    *,
    verbose: bool = False,
) -> str | None:
    """Pick an equivalent logical-X that commutes with masked Z checks.

    Rather than stripping Z stabilizers (which weakens error detection),
    adjust the representative of the logical X by multiplying X stabilizers
    so that it commutes with the boundary-masked Z set used during rough merge.
    """
    if logical_x is None or not masked_z or not x_stabilizers:
        print("[logical-align] No logical X or masked Z or X stabilizers provided")
        return logical_x

    n = len(logical_x)
    l_vec = [1 if c in {"X", "Y"} else 0 for c in logical_x]
    z_rows = [[1 if c == "Z" else 0 for c in stab] for stab in masked_z]
    x_rows = [[1 if c in {"X", "Y"} else 0 for c in stab] for stab in x_stabilizers]

    if all(_pauli_commutes(logical_x, z) for z in masked_z):
        print("[logical-align] Logical X already commutes with masked Z, no adjustment needed")
        return logical_x

    rhs = [sum(l * z for l, z in zip(l_vec, row)) % 2 for row in z_rows]
    A: list[list[int]] = []
    for row in z_rows:
        A.append([sum(x_row[i] & row[i] for i in range(n)) % 2 for x_row in x_rows])

    coeffs = _solve_gf2(A, rhs)
    if coeffs is None:
        print("[logical-align] No solution found for logical X alignment")
        return logical_x

    delta = [0] * n
    used_indices: list[int] = []
    for idx, (coeff, x_row) in enumerate(zip(coeffs, x_rows)):
        if coeff:
            used_indices.append(idx)
            delta = [(d ^ xr) for d, xr in zip(delta, x_row)]

    aligned_vec = [(l ^ d) for l, d in zip(l_vec, delta)]
    aligned = "".join("X" if v else "I" for v in aligned_vec)
    if not all(_pauli_commutes(aligned, z) for z in masked_z):
        print("[logical-align] Adjusted logical X does not commute with masked Z")
        return logical_x
    if verbose:
        delta_support = sum(1 for l, a in zip(logical_x, aligned) if l != a)
        print(
            "[logical-align] adjusted logical X: "
            f"used {sum(coeffs)} X stabilizers, "
            f"delta_support={delta_support}, "
            f"wX_before={logical_x.count('X')}, "
            f"wX_after={aligned.count('X')}"
        )
        if used_indices:
            print(f"[logical-align] stabilizer indices used (0-based): {[i for i,c in enumerate(coeffs) if c]}")
    return aligned


def _run_phase(
    circuit: stim.Circuit,
    builder: PhenomenologicalStimBuilder,
    phase: PhaseSpec,
    config: PhenomenologicalStimConfig,
    sz_prev: List[int] | None,
    sx_prev: List[int] | None,
) -> Tuple[List[int] | None, List[int] | None]:
    """Append one lattice-surgery phase to the circuit.

    This is a thin wrapper that reuses the existing helper methods on
    `PhenomenologicalStimBuilder` (noise model, `_measure_list`,
    `_add_detectors`) but allows different stabilizer sets per phase.

    Parameters
    ----------
    circuit:
        The Stim circuit being built.
    builder:
        Your existing `PhenomenologicalStimBuilder` instance. Only its helper
        methods and `code.n` are used here.
    phase:
        Which stabilizers to measure and for how many rounds.
    config:
        Noise parameters (p_x_error, p_z_error) and family selection.
    sz_prev / sx_prev:
        Absolute measurement indices from the previous phase / round, used to
        build time-like detectors that span *all* phases.

    Returns
    -------
    (sz_prev, sx_prev):
        The last round's measurement indices for Z and X checks, to be fed
        into the next phase.
    """

    n = builder.code.n

    # Which CSS family to measure in this circuit (same semantics as in
    # `PhenomenologicalStimBuilder.build`).
    fam = (config.family or "").upper()
    if fam not in {"", "Z", "X"}:
        raise ValueError("config.family must be one of None, 'Z', or 'X'")
    # Use per-phase flags if specified, otherwise fall back to config.family behavior
    measure_Z = phase.measure_z if phase.measure_z is not None else (fam in {"", "Z"})
    measure_X = phase.measure_x if phase.measure_x is not None else (fam in {"", "X"})

    def apply_x_noise() -> None:
        if config.p_x_error:
            circuit.append_operation("X_ERROR", list(range(n)), config.p_x_error)

    def apply_z_noise() -> None:
        if config.p_z_error:
            circuit.append_operation("Z_ERROR", list(range(n)), config.p_z_error)

    # noise-free warmup round for this phase if there is no
    # prior history for the corresponding stabilizer family. This mirrors the
    # behaviour of `PhenomenologicalStimBuilder.build` in the memory experiment.
    if measure_Z and phase.z_stabilizers and sz_prev is None:
        circuit.append_operation("TICK")
        sz_prev = builder._measure_list(
            circuit, phase.z_stabilizers, family="Z", round_index=-1
        )

    if measure_X and phase.x_stabilizers and sx_prev is None:
        circuit.append_operation("TICK")
        sx_prev = builder._measure_list(
            circuit, phase.x_stabilizers, family="X", round_index=-1
        )

    # Repeat this phase's stabilizers for the requested number of rounds.
    for round_idx in range(phase.rounds):
        # Z half
        if measure_Z and phase.z_stabilizers:
            circuit.append_operation("TICK")
            apply_x_noise()
            sz_curr = builder._measure_list(
                circuit, phase.z_stabilizers, family="Z", round_index=round_idx
            )
            if sz_prev is not None:
                builder._add_detectors(circuit, sz_prev, sz_curr)
            sz_prev = sz_curr

        # X half
        if measure_X and phase.x_stabilizers:
            circuit.append_operation("TICK")
            apply_z_noise()
            sx_curr = builder._measure_list(
                circuit, phase.x_stabilizers, family="X", round_index=round_idx
            )
            if sx_prev is not None:
                builder._add_detectors(circuit, sx_prev, sx_curr)
            sx_prev = sx_curr

    return sz_prev, sx_prev


def build_cnot_surgery_circuit(
    distance: int,
    code_type: str,
    p_x: float,
    p_z: float,
    rounds_pre: int,
    rounds_merge: int,
    rounds_post: int,
    verbose: bool = False,
) -> Tuple[stim.Circuit, List[Tuple[int, int]]]:
    """Return a Stim circuit implementing a *lattice-surgery* CNOT scaffold.

    Layout (following Horsman et al., arXiv:1111.4022): two planar patches
    (control C and target T) plus an intermediate ancilla patch INT prepared
    in |+>_L. The protocol is:

      1. (Optional) memory phase on three disjoint patches (pre-merge).
      2. Smooth merge C and INT (measures Z_L^C Z_L^INT).
      3. Smooth split to re-separate C and INT (they are now entangled).
      4. Rough merge INT and T (measures X_L^INT X_L^T), yielding a CNOT.
      5. (Optional) post-merge memory on the final two logical qubits.

    This function wires these phases together in one Stim circuit and returns
    it along with a list of logical observable pairs (for now a single logical
    CNOT observable at index 0).

    At this stage, only the pre-merge phase (three disjoint patches under
    noise) is implemented. Subsequent merge/split phases will be added on top
    of this.
    """

        # ------------------------------------------------------------------
    # Build 3-patch layout using the generic Layout class
    # ------------------------------------------------------------------
    
    layout = Layout(
        distance=distance,
        code_type=code_type,
        patch_order=["C", "INT", "T"],
        seams=[
            SeamSpec("C", "INT", "smooth"),  # C–INT smooth merge
            SeamSpec("INT", "T", "rough"),   # INT–T rough merge
        ],
        patch_metadata={"C": "control", "INT": "ancilla", "T": "target"},
    )
    

    if verbose:
        layout.print_layout()

    # Single-patch model (same object the layout used internally)
    single_model = layout.single_model
    n_single = single_model.code.n
    n_total = layout.n_total

    smooth_boundary_qubits = layout.local_boundary_qubits["smooth"]
    rough_boundary_qubits = layout.local_boundary_qubits["rough"]


    # Minimal code-like object exposing only `.n` for the builder.
    class CombinedCode:
        def __init__(self, n: int):
            self.n = n

    code = CombinedCode(n_total)

    builder = PhenomenologicalStimBuilder(
        code=code,
        z_stabilizers=[],
        x_stabilizers=[],
        logical_z=None,
        logical_x=None,
    )

    # Offsets for the three logical patches in the combined code.
    offset_C = layout.patch_offsets["C"]
    offset_INT = layout.patch_offsets["INT"]
    offset_T = layout.patch_offsets["T"]

    # Seam ancilla qubits between patches
    seam_C_INT = layout.get_seam_qubits("C", "INT")
    seam_INT_T = layout.get_seam_qubits("INT", "T")
    # Helper to embed a single-patch Pauli string into the combined three-patch
    # index space at a given offset.
    def embed_patch(pauli_str: str, offset: int) -> str:
        assert len(pauli_str) == n_single
        left = "I" * offset
        mid = pauli_str
        right = "I" * (n_total - offset - n_single)
        return left + mid + right


    # Build stabilizers for the disjoint three-patch configuration.
    z_single = list(single_model.z_stabilizers)
    x_single = list(single_model.x_stabilizers)

    # Base Z/X sets used in all phases except the merge windows (kept unmasked).
    base_z: List[str] = []
    base_x: List[str] = []

    for s in z_single:
        base_z.append(embed_patch(s, offset_C))
        base_z.append(embed_patch(s, offset_INT))
        base_z.append(embed_patch(s, offset_T))

    for s in x_single:
        base_x.append(embed_patch(s, offset_C))
        base_x.append(embed_patch(s, offset_INT))
        base_x.append(embed_patch(s, offset_T))

    # --- Smooth-merge phase stabilizers ---------------------------------
    #
    # In the C+INT smooth-merge window we:
    #   * keep all patch stabilizers for C, INT, and T;
    #   * drop the single-qubit X checks on the seam ancillas; and
    #   * add joint Z stabilizers that couple a smooth-boundary qubit of C,
    #     the corresponding seam ancilla, and the corresponding qubit of INT.
    #
    # This realises a smooth merge between C and INT: ancillas are prepared
    # in |+> (via the pre-merge X checks), then governed by Z-type joint
    # checks that effectively measure Z_L^C Z_L^INT over the merge window.
    smooth_merge_z: List[str] = []
    smooth_merge_x: List[str] = []

    smooth_z_masked, smooth_x_masked = _commuting_boundary_mask(
        z_stabilizers=z_single,
        x_stabilizers=x_single,
        boundary=smooth_boundary_qubits,
        strip_pauli="X",
        verbose=verbose,
    )

    # Mask only the merging patches (C and INT); leave T unmasked.
    for s, s_masked in zip(z_single, smooth_z_masked):
        smooth_merge_z.append(embed_patch(s_masked, offset_C))
        smooth_merge_z.append(embed_patch(s_masked, offset_INT))
        smooth_merge_z.append(embed_patch(s, offset_T))

    for s, s_masked in zip(x_single, smooth_x_masked):
        smooth_merge_x.append(embed_patch(s_masked, offset_C))
        smooth_merge_x.append(embed_patch(s_masked, offset_INT))
        smooth_merge_x.append(embed_patch(s, offset_T))

    #Use layout global indeces to add smooth merge stabilizers
    smooth_C   = layout.boundary_qubits["C"]["smooth"]
    smooth_INT = layout.boundary_qubits["INT"]["smooth"]
    seam_C_INT = layout.get_seam_qubits("C", "INT")

    for q_c, q_int, q_sea in zip(smooth_C, smooth_INT, seam_C_INT):
        chars = ["I"] * n_total
        chars[q_c]   = "Z"
        chars[q_sea] = "Z"
        chars[q_int] = "Z"
        smooth_merge_z.append("".join(chars))
        

    # --- Rough-merge phase stabilizers for INT+T ---------------------------
    #
    # In the INT+T rough-merge window we:
    #   * keep all (possibly stripped) Z stabilizers for C, INT, and T;
    #   * keep the X stabilizers for all patches; and
    #   * add joint X stabilizers that couple a rough-boundary qubit of INT,
    #     the corresponding seam ancilla, and the corresponding qubit of T.
    #
    # This realises a rough merge between INT and T: joint X checks effectively
    # measure X_L^INT X_L^T along the rough boundary where X stabilizers
    # terminate.
    # Rough-merge stabilizers: strip Z support on INT/T rough boundaries and
    # mask INT/T X stabilizers on that boundary (rather than dropping them),
    # then add joint INT–seam–T X checks. This keeps the rough-merge CSS set
    # commuting while preserving syndrome data near the merge.
    rough_merge_z: List[str] = []
    rough_merge_x: List[str] = []

    rough_z_masked, rough_x_masked = _commuting_boundary_mask(
        z_stabilizers=z_single,
        x_stabilizers=x_single,
        boundary=rough_boundary_qubits,
        verbose=verbose,
    )
    logical_x_aligned = _align_logical_x_to_masked_z(
        single_model.logical_x,
        x_single,
        rough_z_masked,
        verbose=verbose,
    )
    for s, s_masked in zip(z_single, rough_z_masked):
        rough_merge_z.append(embed_patch(s, offset_C))
        rough_merge_z.append(embed_patch(s_masked, offset_INT))
        rough_merge_z.append(embed_patch(s_masked, offset_T))


    for s, s_masked in zip(x_single, rough_x_masked):
        rough_merge_x.append(embed_patch(s, offset_C))
        rough_merge_x.append(embed_patch(s_masked, offset_INT))
        rough_merge_x.append(embed_patch(s_masked, offset_T))
    

    rough_INT = layout.boundary_qubits["INT"]["rough"]
    rough_T   = layout.boundary_qubits["T"]["rough"]
    seam_INT_T = layout.get_seam_qubits("INT", "T")

    for q_int, q_t, q_sea2 in zip(rough_INT, rough_T, seam_INT_T):
        chars = ["I"] * n_total
        chars[q_int]  = "X"
        chars[q_sea2] = "X"
        chars[q_t]    = "X"
        rough_merge_x.append("".join(chars))

    # We explicitly separate the spacetime into
    # phases: pre-merge memory, a smooth-merge window, a smooth-split window,
    # an INT+T rough-merge window, and a post-merge memory phase.
    phases = [
        PhaseSpec("pre-merge", base_z, base_x, rounds_pre),
        PhaseSpec("C+INT smooth merge", smooth_merge_z, smooth_merge_x, rounds_merge, measure_z=True, measure_x=True),
        PhaseSpec("C|INT smooth split", base_z, base_x, rounds_merge),
        PhaseSpec("INT+T rough merge", rough_merge_z, rough_merge_x, rounds_merge, measure_z=True, measure_x=True),
        PhaseSpec("INT+T rough split", base_z, base_x, rounds_merge),
        PhaseSpec("post-merge", base_z, base_x, rounds_post),
    ]

    circuit = stim.Circuit()

    # Attach simple 1D coordinates for all qubits in the combined layout.
    for q in range(code.n):
        circuit.append_operation("QUBIT_COORDS", [q], [q, 0])

    
    # Build a logical Z observable for the control patch by embedding the
    # single-patch logical_Z operator at the control offset. We measure it
    # once at the beginning and once at the end, and include their parity
    # as OBSERVABLE 0 (mirroring the memory experiment).
    logical_z_control: str | None = None
    if single_model.logical_z is not None:
        logical_z_control = embed_patch(single_model.logical_z, offset_C)

    # Build a logical X observable for the target patch by embedding the
    # single-patch logical_X operator at the target offset. We measure it
    # once at the beginning and once at the end, and include their parity
    # as OBSERVABLE 1.

    logical_x_target: str | None = None
    if logical_x_aligned is not None:
        logical_x_target = embed_patch(logical_x_aligned, offset_T)
    


    # Define the observable pairs for the logical Z and X measurements
    observable_pairs: List[Tuple[int, int]] = []

    #Measure logical Z on the control patch
    start_idx_control: int | None = None
    if logical_z_control is not None:
        circuit.append_operation("TICK")
        start_idx_control = builder._mpp_from_string(circuit, logical_z_control)


    #Measure logical X on the target patch
    start_idx_target: int | None = None
    if logical_x_target is not None:
        circuit.append_operation("TICK")
        start_idx_target = builder._mpp_from_string(circuit, logical_x_target)
    

    
    
    # Run the pre-merge phase using the same phenomenological noise model
    # used in the memory experiment.
    stim_config = PhenomenologicalStimConfig(
        rounds=1,           # per-phase rounds are taken from PhaseSpec
        p_x_error=p_x,
        p_z_error=p_z,
        init_label=None,
    )

    def _phase_measure_flags(phase: PhaseSpec) -> tuple[bool, bool]:
        fam = (stim_config.family or "").upper()
        if fam not in {"", "Z", "X"}:
            raise ValueError("config.family must be one of None, 'Z', or 'X'")
        measure_Z = phase.measure_z if phase.measure_z is not None else (fam in {"", "Z"})
        measure_X = phase.measure_x if phase.measure_x is not None else (fam in {"", "X"})
        return measure_Z, measure_X

    sz_prev: List[int] | None = None
    sx_prev: List[int] | None = None
    prev_z_set: Sequence[str] | None = None
    prev_x_set: Sequence[str] | None = None
    for phase in phases:
        # Reset time-like detectors when stabilizer sets change between phases.
        if prev_z_set is not None and phase.z_stabilizers != prev_z_set:
            sz_prev = None
        if prev_x_set is not None and phase.x_stabilizers != prev_x_set:
            sx_prev = None

        sz_prev, sx_prev = _run_phase(
            circuit=circuit,
            builder=builder,
            phase=phase,
            config=stim_config,
            sz_prev=sz_prev,
            sx_prev=sx_prev,
        )
        measure_Z, measure_X = _phase_measure_flags(phase)
        if not measure_Z:
            sz_prev = None
        if not measure_X:
            sx_prev = None
        prev_z_set = phase.z_stabilizers
        prev_x_set = phase.x_stabilizers

    
    # Final logical Z measurement on the control patch.
    end_idx_control: int | None = None
    if logical_z_control is not None:
        circuit.append_operation("TICK")
        end_idx_control = builder._mpp_from_string(circuit, logical_z_control)
        if start_idx_control is not None and end_idx_control is not None:
            circuit.append_operation(
                "OBSERVABLE_INCLUDE",
                [builder._rec_from_abs(circuit, start_idx_control),
                 builder._rec_from_abs(circuit, end_idx_control)],
                0,
            )
            observable_pairs.append((start_idx_control, end_idx_control))

    
    # Final logical X measurement on the target patch.
    end_idx_target: int | None = None
    if logical_x_target is not None:
        circuit.append_operation("TICK")
        end_idx_target = builder._mpp_from_string(circuit, logical_x_target)
        if start_idx_target is not None and end_idx_target is not None:
            circuit.append_operation(
                "OBSERVABLE_INCLUDE",
                [builder._rec_from_abs(circuit, start_idx_target),
                 builder._rec_from_abs(circuit, end_idx_target)],
                1,
            )
            observable_pairs.append((start_idx_target, end_idx_target))
    
    
    return circuit, observable_pairs

# ---------------------------------------------------------------------------
# CLI wrapper, mirroring memory_experiment.py
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a lattice-surgery CNOT experiment scaffold using the "
            "phenomenological surface-code model."
        )
    )

    parser.add_argument(
        "--code-type",
        type=str,
        default="heavy_hex",
        choices=["heavy_hex", "standard"],
        help="Type of surface code: 'heavy_hex' or 'standard'",
    )
    parser.add_argument(
        "--rounds-pre",
        type=int,
        default=None,
        help="Number of pre-surgery memory rounds (three disjoint patches, default: distance)",
    )
    parser.add_argument(
        "--rounds-merge",
        type=int,
        default=None,
        help="Number of rounds in each merge/split window (default: distance)",
    )
    parser.add_argument(
        "--rounds-post",
        type=int,
        default=None,
        help="Number of post-surgery memory rounds (default: distance)",
    )
    parser.add_argument("--distance", type=int, default=3, help="Code distance d")
    parser.add_argument("--px", type=float, default=1e-3, help="X error probability")
    parser.add_argument("--pz", type=float, default=1e-3, help="Z error probability")
    parser.add_argument("--shots", type=int, default=10**5, help="Monte Carlo shots")
    parser.add_argument("--seed", type=int, default=46, help="Stim / DEM seed")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output.",
    )
    return parser.parse_args()


def run_cnot_experiment(
    distance: int,
    code_type: str,
    rounds_pre: int | None,
    rounds_merge: int | None,
    rounds_post: int | None,
    p_x: float,
    p_z: float,
    shots: int,
    seed: int | None,
    verbose: bool = False,
):
    """Top-level driver for the CNOT experiment.

    Once `build_cnot_surgery_circuit` is fully implemented, this will:

      * build the lattice-surgery CNOT circuit,
      * derive its detector error model,
      * decode with PyMatching, and
      * report a logical CNOT error rate.
    """

    if rounds_pre is None:
        rounds_pre = distance
    if rounds_merge is None:
        rounds_merge = distance
    if rounds_post is None:
        rounds_post = distance

    circuit, observable_pairs = build_cnot_surgery_circuit(
        distance=distance,
        code_type=code_type,
        p_x=p_x,
        p_z=p_z,
        rounds_pre=rounds_pre,
        rounds_merge=rounds_merge,
        rounds_post=rounds_post,
        verbose=verbose,
    )


    # We reuse PhenomenologicalStimConfig purely for its noise parameters and
    # CSS-family selector; the `rounds` field is ignored by the multi-phase
    # builder.
    stim_config = PhenomenologicalStimConfig(
        rounds=1,
        p_x_error=p_x,
        p_z_error=p_z,
        init_label=None,
    )

    mc_config = MonteCarloConfig(shots=shots, seed=seed)
    result = run_circuit_logical_error_rate(circuit, observable_pairs, stim_config, mc_config )
    
    print(f"{code_type} code of distance ={distance}")
    print(f"shots={result.shots}")
    print(f"Physical error rates: p_x={p_x}, p_z={p_z}")
    if len(result.logical_error_rates) > 0:
        print(f"logical_error_rate (Control) = {result.logical_error_rates[0]:.3e}")
    else:
        print(f"logical_error_rate (Control) = {result.logical_error_rate:.3e}")
    if len(result.logical_error_rates) > 1:
        print(f"logical_error_rate (Target) = {result.logical_error_rates[1]:.3e}")
    print(f"avg_syndrome_weight = {result.avg_syndrome_weight:.3f}")
    print(f"click_rate(any_detector) = {result.click_rate:.3f}")
    print(f"num_detectors = {result.num_detectors}")

    return result


def main() -> None:
    args = parse_args()
    run_cnot_experiment(
        distance=args.distance,
        code_type=args.code_type,
        rounds_pre=args.rounds_pre,
        rounds_merge=args.rounds_merge,
        rounds_post=args.rounds_post,
        p_x=args.px,
        p_z=args.pz,
        shots=args.shots,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
