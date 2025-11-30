"""Lattice-surgery CNOT experiment.

Conceptually, we follow the CNOT construction of Horsman et al.,
"Surface code quantum computing by lattice surgery" (2013): two planar
logical patches (control C and target T) and an intermediate logical patch
(INT) prepared in |+>_L. The CNOT is realised by a smooth merge + split
between C and INT followed by a rough merge between INT and T.

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
from simulation import MonteCarloConfig, run_circuit_logical_error_rate, run_circuit_physics




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
    patch_logicals = cnot_spec.patch_logicals


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

   # --------------------------------------------------------------
    # Per-patch logical initialization (before any detectors)
    # --------------------------------------------------------------
    # Example choice:
    #   C   prepared in Z-basis (|0_L or |1_L)
    #   INT prepared in X-basis (|+_L ancilla)
    #   T   prepared in X-basis (or Z, depending on your protocol)
    patch_init_bases: dict[str, str] = {
        "C": "Z",
        "INT": "X",
        "T": "X",
    }

    #This is the initial logical measurement for the CNOT experiment, 
    # it initializes the logical qubits in the Z or X basis.
    init_indices: dict[str, int | None] = {}
    for patch, basis in patch_init_bases.items():
        logical_str = patch_logicals.get(patch, {}).get(basis)
        init_indices[patch] = builder.measure_logical_once(circuit, logical_str)


    #This is the start index for the logical measurements of the CNOT experiment.
    # For CNOT: track Z on control as observable 0, X on target as observable 1
    start_idx_control = init_indices["C"]  # Z_C
    start_idx_target  = init_indices["T"]  # basis depends on patch_init_bases["T"]

    
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

    
    # Final logical on control
    end_idx_control = builder.measure_logical_once(circuit, logical_z_control)
    builder.attach_observable_pair(
        circuit,
        start_idx=start_idx_control,
        end_idx=end_idx_control,
        observable_index=0,
        observable_pairs=observable_pairs,
    )

    # Final logical on target
    end_idx_target = builder.measure_logical_once(circuit, logical_x_target)
    builder.attach_observable_pair(
        circuit,
        start_idx=start_idx_target,
        end_idx=end_idx_target,
        observable_index=1,
        observable_pairs=observable_pairs,
    )

    
    return circuit, observable_pairs

# ---------------------------------------------------------------------------
# Build circuit for physics mode (Bell-state diagnostics)
# ---------------------------------------------------------------------------


def build_cnot_surgery_circuit_physics(
    distance: int,
    code_type: str,
    p_x: float,
    p_z: float,
    rounds_pre: int,
    rounds_merge: int,
    rounds_post: int,
    verbose: bool = False,
) -> Tuple[stim.Circuit, dict[str, list[int]]]:
    """Build a CNOT surgery circuit plus Bell-type logical measurements.

    This variant is for "physics mode": we *do not* attach DEM observables,
    but instead add explicit logical MPPs at the end of the protocol to probe
    Bell correlators such as X_C X_T and Z_C Z_T.
    """
    layout = Layout(
        distance=distance,
        code_type=code_type,
        patch_order=["C", "INT", "T"],
        seams=[
            SeamSpec("C", "INT", "smooth"),
            SeamSpec("INT", "T", "rough"),
        ],
        patch_metadata={"C": "control", "INT": "ancilla", "T": "target"},
    )

    if verbose:
        layout.print_layout()

    n_total = layout.n_total

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
    patch_logicals = cnot_spec.patch_logicals
    bell_obs = cnot_spec.bell_observables

    if verbose:
        print("[physics] Bell observables:")
        for name, obs in bell_obs.items():
            print(f"  {name}:")
            print(f"    pauli: {obs.pauli}")
            print(f"    frame_bits: {obs.frame_bits}")

    circuit = stim.Circuit()

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

    for q in range(code.n):
        circuit.append_operation("QUBIT_COORDS", [q], [q, 0])

    # Logical initialisation choice for Bell: C in X, INT in X, T in Z
    patch_init_bases: dict[str, str] = {
        "C": "X",
        "INT": "X",
        "T": "Z",
    }

    init_indices: dict[str, int | None] = {}
    for patch, basis in patch_init_bases.items():
        logical_str = patch_logicals.get(patch, {}).get(basis)
        init_indices[patch] = builder.measure_logical_once(circuit, logical_str)

    stim_config = PhenomenologicalStimConfig(
        rounds=1,
        p_x_error=p_x,
        p_z_error=p_z,
        init_label=None,
    )

    builder.run_phases(
        circuit=circuit,
        phases=phases,
        config=stim_config,
    )

    # Bell stabilizers 
    xx_logical = bell_obs["XX"].pauli
    zz_logical = bell_obs["ZZ"].pauli

    circuit.append_operation("TICK")
    xx_idx = builder.measure_logical_once(circuit, xx_logical)

    circuit.append_operation("TICK")
    zz_idx = builder.measure_logical_once(circuit, zz_logical)


    # --------------------------------------------------------------
    # Resolve symbolic frame_bits into measurement indices
    # --------------------------------------------------------------

    def _collect_phase_indices(
        pauli_strings: list[str],
        *,
        family: str,
        phase_name: str,
    ) -> list[int]:
        """For each Pauli in pauli_strings, find the last-round measurement index
        in builder._meas_meta with matching family and phase."""
        out: list[int] = []
        for p in pauli_strings:
            candidates: list[tuple[int, int]] = []
            for idx, meta in builder._meas_meta.items():
                if (
                    meta.get("pauli") == p
                    and meta.get("family") == family
                    and meta.get("phase") == phase_name
                ):
                    candidates.append((idx, meta["round"]))
            if not candidates:
                print(f"[physics] WARNING: no measurement found for {p!r} in {phase_name!r}")
                continue
            best_idx = max(candidates, key=lambda t: t[1])[0]
            out.append(best_idx)
        return out

    def resolve_frame_bits(labels: list[str]) -> list[int]:
        """Map symbolic frame bit labels from LogicalObservable.frame_bits
        to concrete measurement indices via layout + builder._meas_meta."""
        indices: list[int] = []

        for label in labels:
            kind, left, right = label.split(":")

            if kind == "SMOOTH_Z_SEAM":
                # Construct the joint Z checks across the smooth C–INT seam
                boundary_left = layout.boundary_qubits[left]["smooth"]
                boundary_right = layout.boundary_qubits[right]["smooth"]
                seam = layout.get_seam_qubits(left, right)

                paulis: list[str] = []
                for q_l, q_r, q_sea in zip(boundary_left, boundary_right, seam):
                    chars = ["I"] * n_total
                    chars[q_l] = "Z"
                    chars[q_sea] = "Z"
                    chars[q_r] = "Z"
                    paulis.append("".join(chars))

                phase_name = f"{left}+{right} smooth merge"
                indices.extend(
                    _collect_phase_indices(paulis, family="Z", phase_name=phase_name)
                )

            elif kind == "ROUGH_X_SEAM":
                # Construct the joint X checks across the rough INT–T seam
                boundary_left = layout.boundary_qubits[left]["rough"]
                boundary_right = layout.boundary_qubits[right]["rough"]
                seam = layout.get_seam_qubits(left, right)

                paulis: list[str] = []
                for q_l, q_r, q_sea in zip(boundary_left, boundary_right, seam):
                    chars = ["I"] * n_total
                    chars[q_l] = "X"
                    chars[q_sea] = "X"
                    chars[q_r] = "X"
                    paulis.append("".join(chars))

                phase_name = f"{left}+{right} rough merge"
                indices.extend(
                    _collect_phase_indices(paulis, family="X", phase_name=phase_name)
                )

            else:
                # Unknown label: ignore for now or raise
                if verbose:
                    print(f"[physics] WARNING: unknown frame bit label {label!r}")
                continue

        return indices

        # Resolve frame bits for XX and ZZ coming from the merge windows
    frame_xx = resolve_frame_bits(bell_obs["XX"].frame_bits)
    frame_zz = resolve_frame_bits(bell_obs["ZZ"].frame_bits)

    if verbose:
        print("[physics] frame_xx indices:", frame_xx)
        print("[physics] frame_zz indices:", frame_zz)

    # --------------------------------------------------------------
    # Include initial logical measurements in the Pauli frame
    # --------------------------------------------------------------
    # By construction:
    #   * We measured X_C at init_indices["C"] to "prepare" C in X-basis.
    #   * We measured Z_T at init_indices["T"] to "prepare" T in Z-basis.
    #
    # These are random ±1 but *known per shot*, so they must be part of the
    # dressed Bell stabilizers.
    xx_indices: list[int] = []
    if init_indices.get("C") is not None:
        xx_indices.append(init_indices["C"])   # X_C^{init} factor
    if init_indices.get("INT") is not None:
        xx_indices.append(init_indices["INT"]) # X_INT^{init} factor
    xx_indices.extend(frame_xx)                # rough seam X parities
    xx_indices.append(xx_idx)                  # final X_C X_T

    zz_indices: list[int] = []
    if init_indices.get("T") is not None:
        zz_indices.append(init_indices["T"])   # Z_T^{init} factor
    zz_indices.extend(frame_zz)                # smooth seam Z parities
    zz_indices.append(zz_idx)                  # final Z_C Z_T

    if verbose:
        print("[physics] XX indices (init + frame + final):", xx_indices)
        print("[physics] ZZ indices (init + frame + final):", zz_indices)

    correlator_map: dict[str, list[int]] = {
        "XX": xx_indices,
        "ZZ": zz_indices,
    }

    if verbose:
        print("[physics] Metadata for frame_xx:")
        for idx in frame_xx:
            print("  idx", idx, "->", builder._meas_meta.get(idx))

        print("[physics] Metadata for frame_zz:")
        for idx in frame_zz:
            print("  idx", idx, "->", builder._meas_meta.get(idx))

    return circuit, correlator_map

   

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
        default="standard",
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
    parser.add_argument("--px", type=float, default=1e-2, help="X error probability")
    parser.add_argument("--pz", type=float, default=1e-2, help="Z error probability")
    parser.add_argument("--shots", type=int, default=10**5, help="Monte Carlo shots")
    parser.add_argument("--seed", type=int, default=46, help="Stim / DEM seed")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ler", "physics", "both"],
        default="physics",
        help="Which experiment to run: logical error rate, physics correlators, or both.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Physics-mode experiment driver
# ---------------------------------------------------------------------------

def run_cnot_physics_experiment(
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
    """Top-level driver for Bell-state physics diagnostics.

    Builds the lattice-surgery CNOT circuit, then samples it directly to
    estimate logical Bell correlators such as X_C X_T and Z_C Z_T.
    """
    if rounds_pre is None:
        rounds_pre = distance
    if rounds_merge is None:
        rounds_merge = distance
    if rounds_post is None:
        rounds_post = distance

    circuit, correlator_map = build_cnot_surgery_circuit_physics(
        distance=distance,
        code_type=code_type,
        p_x=p_x,
        p_z=p_z,
        rounds_pre=rounds_pre,
        rounds_merge=rounds_merge,
        rounds_post=rounds_post,
        verbose=verbose,
    )

    mc_config = MonteCarloConfig(shots=shots, seed=seed)
    phys_result = run_circuit_physics(
        circuit=circuit,
        correlator_map=correlator_map,
        mc_config=mc_config,
        keep_samples=False,
        verbose=verbose,
    )

    print(f"{code_type} code of distance ={distance}")
    print(f"shots={phys_result.shots}")
    print(f"Physical error rates: p_x={p_x}, p_z={p_z}")
    for name, value in phys_result.correlators.items():
        print(f"⟨{name}⟩ = {value:.3f}")


    # Quick diagnostic: look at the first few shots by hand
    if verbose:
        small_mc = MonteCarloConfig(shots=8, seed=seed)
        import numpy as np
        sampler = circuit.compile_sampler(seed=seed)
        small_samples = np.asarray(sampler.sample(small_mc.shots), dtype=np.float64)
        pm1 = 1 - 2 * small_samples

        for name, idxs in correlator_map.items():
            idxs = list(idxs)
            vals = pm1[:, idxs].prod(axis=1)
            print(f"[physics] Sampled values for {name}:")
            for s, v in enumerate(vals):
                print(f"  shot {s}: {v}")
            print(f"  mean over {small_mc.shots} shots: {vals.mean():.3f}")

    return phys_result


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

    if args.mode in ("ler", "both"):
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

    if args.mode in ("physics", "both"):
        run_cnot_physics_experiment(
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
