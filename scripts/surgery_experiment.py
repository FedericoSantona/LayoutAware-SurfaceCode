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
    PhaseSpec,
    _align_logical_x_to_masked_z,
    _commuting_boundary_mask,
)
from simulation import MonteCarloConfig, run_circuit_logical_error_rate


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

    # Run all phases (pre-merge, merges/splits, post-merge) using the
    # generalized multi-phase builder helper.
    builder.run_phases(
        circuit=circuit,
        phases=phases,
        config=stim_config,
    )

    
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
    parser.add_argument("--distance", type=int, default=7, help="Code distance d")
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
