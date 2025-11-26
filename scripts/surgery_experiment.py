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
    LatticeSurgery,
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

    n_total = layout.n_total
    single_model = layout.single_model

    surgery = LatticeSurgery(layout)
    cnot_spec = surgery.cnot(
        control="C",
        ancilla="INT",
        target="T",
        rounds_pre=rounds_pre,
        rounds_merge=rounds_merge,
        rounds_post=rounds_post,
        verbose=verbose,
    )

    phases = cnot_spec.phases
    logical_z_control = cnot_spec.logical_z_control
    logical_x_target  = cnot_spec.logical_x_target


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
