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

import stim

# Make `src/` importable, mirroring memory_experiment.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from surface_code import (
    PhenomenologicalStimBuilder,
    PhenomenologicalStimConfig,
    build_surface_code_model,
    find_smooth_boundary_data_qubits,
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
    """

    name: str
    z_stabilizers: Sequence[str]
    x_stabilizers: Sequence[str]
    rounds: int


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
    measure_Z = fam in {"", "Z"}
    measure_X = fam in {"", "X"}

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
        sz_prev = builder._measure_list(circuit, phase.z_stabilizers)

    if measure_X and phase.x_stabilizers and sx_prev is None:
        circuit.append_operation("TICK")
        sx_prev = builder._measure_list(circuit, phase.x_stabilizers)

    # Repeat this phase's stabilizers for the requested number of rounds.
    for _ in range(phase.rounds):
        # Z half
        if measure_Z and phase.z_stabilizers:
            circuit.append_operation("TICK")
            apply_x_noise()
            sz_curr = builder._measure_list(circuit, phase.z_stabilizers)
            if sz_prev is not None:
                builder._add_detectors(circuit, sz_prev, sz_curr)
            sz_prev = sz_curr

        # X half
        if measure_X and phase.x_stabilizers:
            circuit.append_operation("TICK")
            apply_z_noise()
            sx_curr = builder._measure_list(circuit, phase.x_stabilizers)
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

    # Use the single-patch model as a template to build a three-patch layout:
    # control (C), intermediate ancilla (INT), and target (T). For now we model
    # this purely at the level of indices: three disjoint copies of the same
    # patch, laid out in a single index space [0, ..., n_total-1].
    single_model = build_surface_code_model(distance, code_type)
    
    # Identify smooth boundary data qubits for correct seam placement
    # The seam between C and INT should be placed on the smooth boundary
    smooth_boundary_qubits = find_smooth_boundary_data_qubits(single_model)
    
    #-------Layout of the combined code-------
    n_single = single_model.code.n
    # Extra ancilla line between C and INT for smooth merge.
    # Use the actual number of smooth boundary qubits to determine seam size.
    # This ensures the seam aligns with the smooth boundary where Z stabilizers terminate.
    n_seam_C_INT = len(smooth_boundary_qubits)
    n_total = 3 * n_single + n_seam_C_INT

    class CombinedCode:
        """Minimal code-like object exposing only `.n` for the builder."""
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
    # Helper to embed a single-patch Pauli string into the combined three-patch
    # index space at a given offset.
    def embed_patch(pauli_str: str, offset: int) -> str:
        assert len(pauli_str) == n_single
        left = "I" * offset
        mid = pauli_str
        right = "I" * (n_total - offset - n_single)
        return left + mid + right

    # Offsets for the three logical patches in the combined code.
    offset_C = 0
    offset_INT = n_single
    offset_T = 2 * n_single
    # Seam ancilla qubits between C and INT.
    # These are placed on the smooth boundary (where Z stabilizers terminate)
    # to enable smooth merge operations between C and INT patches.
    offset_seam_C_INT = 3 * n_single
    seam_C_INT = list(range(offset_seam_C_INT, offset_seam_C_INT + n_seam_C_INT))


    # Build pre-merge stabilizers: three disjoint copies of the single patch.
    z_single = list(single_model.z_stabilizers)
    x_single = list(single_model.x_stabilizers)

    premerge_z: List[str] = []
    premerge_x: List[str] = []

    for s in z_single:
        premerge_z.append(embed_patch(s, offset_C))
        premerge_z.append(embed_patch(s, offset_INT))
        premerge_z.append(embed_patch(s, offset_T))

    for s in x_single:
        # Remove X support on the smooth boundary data qubits for C and INT.
        # This ensures that, when we later add joint Z checks along the seam
        # involving these boundary qubits, those joint Z stabilizers commute
        # with all X stabilizers in *every* phase of the protocol.
        s_stripped = _strip_x_on_qubits(s, smooth_boundary_qubits)
        premerge_x.append(embed_patch(s_stripped, offset_C))
        premerge_x.append(embed_patch(s_stripped, offset_INT))
        # The target patch keeps the full X stabilizers; it is unaffected by
        # the C–INT smooth merge along the seam.
        premerge_x.append(embed_patch(s, offset_T))

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
    merge_z: List[str] = list(premerge_z)
    merge_x: List[str] = []

    # Keep embedded X stabilizers for the three patches, but with X support
    # removed on the smooth boundary data qubits of C and INT. This ensures
    # that the joint C–seam–INT Z checks added below commute with all X
    # stabilizers measured during the smooth-merge phase.
    for s in x_single:
        s_stripped = _strip_x_on_qubits(s, smooth_boundary_qubits)
        merge_x.append(embed_patch(s_stripped, offset_C))
        merge_x.append(embed_patch(s_stripped, offset_INT))
        merge_x.append(embed_patch(s, offset_T))

    # Add distance-many joint Z checks tying C, seam, and INT along the smooth
    # boundary identified by `smooth_boundary_qubits`.
    for k, qb_local in enumerate(smooth_boundary_qubits):
        if k >= len(seam_C_INT):
            break
        q_c = offset_C + qb_local
        q_sea = seam_C_INT[k]
        q_int = offset_INT + qb_local

        chars = ["I"] * n_total
        chars[q_c] = "Z"
        chars[q_sea] = "Z"
        chars[q_int] = "Z"
        merge_z.append("".join(chars))
        

    # For this first implementation, we explicitly separate the spacetime into
    # phases: pre-merge memory, a smooth-merge window, a smooth-split window,
    # and a post-merge memory phase. The smooth-merge phase uses the modified
    # stabilizers (with joint C–seam–INT Z checks and no seam X checks), while
    # the other phases currently reuse the pre-merge stabilizers.
    phases = [
        PhaseSpec("pre-merge", premerge_z, premerge_x, rounds_pre),
        PhaseSpec("C+INT smooth merge", merge_z, merge_x, rounds_merge),
        PhaseSpec("C|INT smooth split", premerge_z, premerge_x, rounds_merge),
        PhaseSpec("post-merge", premerge_z, premerge_x, rounds_post),
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

    # Build a logical Z observable for the target patch by embedding the
    # single-patch logical_Z operator at the target offset. We measure it
    # once at the beginning and once at the end, and include their parity
    # as OBSERVABLE 1.
    logical_z_target: str | None = None
    if single_model.logical_z is not None:
        logical_z_target = embed_patch(single_model.logical_z, offset_T)

    observable_pairs: List[Tuple[int, int]] = []
    start_idx: int | None = None
    if logical_z_control is not None:
        circuit.append_operation("TICK")
        start_idx = builder._mpp_from_string(circuit, logical_z_control)

    start_idx_target: int | None = None
    if logical_z_target is not None:
        circuit.append_operation("TICK")
        start_idx_target = builder._mpp_from_string(circuit, logical_z_target)

    # Run the pre-merge phase using the same phenomenological noise model
    # used in the memory experiment.
    stim_config = PhenomenologicalStimConfig(
        rounds=1,           # per-phase rounds are taken from PhaseSpec
        p_x_error=p_x,
        p_z_error=p_z,
        init_label=None,
    )

    sz_prev: List[int] | None = None
    sx_prev: List[int] | None = None
    for phase in phases:
        sz_prev, sx_prev = _run_phase(
            circuit=circuit,
            builder=builder,
            phase=phase,
            config=stim_config,
            sz_prev=sz_prev,
            sx_prev=sx_prev,
        )

    # Final logical Z measurement on the control patch.
    end_idx: int | None = None
    if logical_z_control is not None:
        circuit.append_operation("TICK")
        end_idx = builder._mpp_from_string(circuit, logical_z_control)
        if start_idx is not None and end_idx is not None:
            circuit.append_operation(
                "OBSERVABLE_INCLUDE",
                [builder._rec_from_abs(circuit, start_idx),
                 builder._rec_from_abs(circuit, end_idx)],
                0,
            )
            observable_pairs.append((start_idx, end_idx))

    # Final logical Z measurement on the target patch.
    end_idx_target: int | None = None
    if logical_z_target is not None:
        circuit.append_operation("TICK")
        end_idx_target = builder._mpp_from_string(circuit, logical_z_target)
        if start_idx_target is not None and end_idx_target is not None:
            circuit.append_operation(
                "OBSERVABLE_INCLUDE",
                [builder._rec_from_abs(circuit, start_idx_target),
                 builder._rec_from_abs(circuit, end_idx_target)],
                1,
            )
            observable_pairs.append((start_idx_target, end_idx_target))

    return circuit, observable_pairs

    # Example sketch (once the TODO above is implemented) of how the rest of
    # this function should look:
    #
    # premerge_z, premerge_x = build_premerge_stabilizers(single_model, ...)
    # merge1_z, merge1_x   = build_merge_C_INT_stabilizers(single_model, ...)
    # split_z, split_x     = build_split_C_INT_stabilizers(single_model, ...)
    # merge2_z, merge2_x   = build_merge_INT_T_stabilizers(single_model, ...)
    # post_z, post_x       = build_postmerge_stabilizers(single_model, ...)
    #
    # phases = [
    #     PhaseSpec("pre-merge", premerge_z, premerge_x, rounds_pre),
    #     PhaseSpec("C+INT smooth merge", merge1_z, merge1_x, rounds_merge),
    #     PhaseSpec("C|INT smooth split", split_z, split_x, rounds_merge),
    #     PhaseSpec("INT+T rough merge", merge2_z, merge2_x, rounds_merge),
    #     PhaseSpec("post-merge", post_z, post_x, rounds_post),
    # ]
    #
    # circuit = stim.Circuit()
    #
    # # Attach coordinates (optional but nice for visualization / debugging).
    # for q in range(code.n):
    #     circuit.append_operation("QUBIT_COORDS", [q], [q, 0])
    #
    # stim_config = PhenomenologicalStimConfig(
    #     rounds=1,
    #     p_x_error=p_x,
    #     p_z_error=p_z,
    #     init_label=None,
    # )
    #
    # sz_prev: List[int] | None = None
    # sx_prev: List[int] | None = None
    # for phase in phases:
    #     sz_prev, sx_prev = _run_phase(
    #         circuit=circuit,
    #         builder=builder,
    #         phase=phase,
    #         config=stim_config,
    #         sz_prev=sz_prev,
    #         sx_prev=sx_prev,
    #     )
    #
    # # TODO: insert logical observables for the CNOT experiment here using
    # # e.g. `builder._mpp_from_string` + `OBSERVABLE_INCLUDE`.
    # observable_pairs: List[Tuple[int, int]] = []
    #
    # return circuit, observable_pairs


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
    result = run_circuit_logical_error_rate(circuit, observable_pairs, stim_config, mc_config)
    
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
    )


if __name__ == "__main__":
    main()