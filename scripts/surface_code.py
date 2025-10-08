"""CLI entry point for running heavy-hex phenomenological simulations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
 
# Ensure the src/ directory is available for imports when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qiskit_qec.operators.pauli_list import PauliList

from surface_code import (
    PhenomenologicalStimBuilder,
    PhenomenologicalStimConfig,
    build_heavy_hex_model,
)
from simulation import MonteCarloConfig, run_circuit_logical_error_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=7, help="Code distance d of the heavy-hex code")
    parser.add_argument("--rounds", type=int, default=None, help="Number of measurement rounds (default: distance)")
    parser.add_argument("--px", type=float, default=1e-4, help="Phenomenological X error probability")
    parser.add_argument("--pz", type=float, default=1e-4, help="Phenomenological Z error probability")
    parser.add_argument("--init", type=str, default="0", help="Logical initialization: one of {0,1,+,-}")
    parser.add_argument("--shots", type=int, default=10**4, help="Number of Monte Carlo samples")
    parser.add_argument("--seed", type=int, default=46, help="Seed for Stim samplers")

    return parser.parse_args()


def run_experiment(
    distance: int,
    rounds: int | None,
    p_x: float,
    p_z: float,
    init_label: str,
    shots: int,
    seed: int | None,
    spatial: bool = True,
):
    model = build_heavy_hex_model(distance)


    # save the heavy-hex tiling
    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig = model.code.draw(
        face_colors=False,
        xcolor="lightcoral",
        zcolor="skyblue",
        figsize=(5, 5),
    )
    plt.savefig(plot_dir / f"heavy_hex_d{distance}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Heavy-hex tiling saved to {plot_dir}/heavy_hex_d{distance}.png")

    print(f"Heavy-hex code with d={distance} has {model.code.n} physical qubits.")
    print(f"Number of gauge generators: {len(model.generators)}")
    print("Gauge Generators:")
    for i, gen in enumerate(model.generators):
        print(f"  GG{i}: {gen}")

    stabilizers = PauliList(model.stabilizer_matrix)
    print("Stabilizers (basis):")
    for i, stab in enumerate(stabilizers):
        print(f"  SG{i}: {stab}")

    print(
        "Using CSS-projected stabilizers: "
        f"{len(model.z_stabilizers)} Z, {len(model.x_stabilizers)} X "
        f"(rank(S)={model.stabilizer_matrix.shape[0]}, total={len(model.z_stabilizers)+len(model.x_stabilizers)})"
    )

    print("Chosen Z_L (geometry):", model.logical_z)
    print("Chosen X_L (algebraic from Z_L):", model.logical_x)

    diagnostics = model.diagnostics()
    print("Logical operator checker:")
    print(f"  Z_L commutes with all stabilizers: {diagnostics['commute_Z']}")
    print(f"  X_L commutes with all stabilizers: {diagnostics['commute_X']}")
    print(f"  Z_L and X_L anticommute: {diagnostics['anticommute']}")
    print(f"  Z_L in stabilizer group: {diagnostics['Z_in_stabilizer']}")
    print(f"  X_L in stabilizer group: {diagnostics['X_in_stabilizer']}")
    print(f"  Z_L weight: {diagnostics['weight_Z']}")
    print(f"  X_L weight: {diagnostics['weight_X']}")

    print(f"n={diagnostics['n']}, s={diagnostics['s']}, r={diagnostics['r']}  =>  k={diagnostics['k']}")

    stim_rounds = rounds if rounds is not None else distance
    stim_config = PhenomenologicalStimConfig(
        rounds=stim_rounds,
        p_x_error=p_x,
        p_z_error=p_z,
        init_label=init_label,
    )
    builder = PhenomenologicalStimBuilder(
        code=model.code,
        z_stabilizers=model.z_stabilizers,
        x_stabilizers=model.x_stabilizers,
        logical_z=model.logical_z,
        logical_x=model.logical_x,
    )

    circuit, observable_pairs = builder.build(stim_config)
    dem = circuit.detector_error_model()
    print("detectors:", dem.num_detectors)

    mc_config = MonteCarloConfig(shots=shots, seed=seed)
    result = run_circuit_logical_error_rate(circuit, observable_pairs, stim_config, mc_config)

    print(f"shots={result.shots}")
    print(f"Physical error rates: p_x={p_x}, p_z={p_z}")
    print(f"logical_error_rate = {result.logical_error_rate:.3e}")
    if result.logical_error_rate > (p_x + p_z)/2:
        print("WARNING: Logical error rate is higher than physical error rates!")
    print(f"avg_syndrome_weight = {result.avg_syndrome_weight:.3f}")
    print(f"click_rate(any_detector) = {result.click_rate:.3f}")
    print("Decoding complete with logical initialization.")

    return result


def main() -> None:
    args = parse_args()
    run_experiment(
        distance=args.distance,
        rounds=args.rounds,
        p_x=args.px,
        p_z=args.pz,
        init_label=args.init,
        shots=args.shots,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
