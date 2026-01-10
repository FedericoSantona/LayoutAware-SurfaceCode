"""Minimal CSS sanity check under pure X/Z noise.

This script runs the four combinations suggested by the heavy-hex sanity check:
    1) pure Z noise with |0_L> and |+_L>
    2) pure X noise with |0_L> and |+_L>
Only one stabilizer family is measured (matching the threshold sweeps) so the
results directly expose any basis/observable bookkeeping errors.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when invoked as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from surface_code import (
    PhenomenologicalStimBuilder,
    PhenomenologicalStimConfig,
    build_surface_code_model,
)
from simulation import MonteCarloConfig, run_logical_error_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3, help="Code distance (default: 3)")
    parser.add_argument(
        "--code-type",
        type=str,
        default="heavy_hex",
        choices=["heavy_hex", "standard"],
        help="Surface code layout (default: heavy_hex)",
    )
    parser.add_argument("--p", type=float, default=0.02, help="Physical error rate for the active channel")
    parser.add_argument("--rounds", type=int, default=None, help="Syndrome rounds (default: distance)")
    parser.add_argument("--shots", type=int, default=2000, help="Monte Carlo shots per configuration")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for Stim samplers")
    return parser.parse_args()


def run_case(
    builder: PhenomenologicalStimBuilder,
    cfg: PhenomenologicalStimConfig,
    shots: int,
    seed: int | None,
):
    mc = MonteCarloConfig(shots=shots, seed=seed)
    return run_logical_error_rate(builder, cfg, mc, verbose=False)


def main() -> None:
    args = parse_args()
    rounds = args.rounds or args.distance
    noise_floor = 5 / args.shots if args.shots else 0.0

    model = build_surface_code_model(args.distance, args.code_type)
    builder = PhenomenologicalStimBuilder(
        code=model.code,
        z_stabilizers=model.z_stabilizers,
        x_stabilizers=model.x_stabilizers,
        logical_z=model.logical_z,
        logical_x=model.logical_x,
    )

    tests = [
        ("pure Z noise", "0", 0.0, args.p, "X"),
        ("pure Z noise", "+", 0.0, args.p, "X"),
        ("pure X noise", "0", args.p, 0.0, "Z"),
        ("pure X noise", "+", args.p, 0.0, "Z"),
    ]

    print(f"CSS sanity check (layout={args.code_type}, d={args.distance}, rounds={rounds}, p={args.p})")
    print(f"Noise floor ~{noise_floor:.4f} (5/{args.shots} shots)")

    for label, init_label, p_x, p_z, family in tests:
        cfg = PhenomenologicalStimConfig(
            rounds=rounds,
            p_x_error=p_x,
            p_z_error=p_z,
            init_label=init_label,
            family=family,
        )
        result = run_case(builder, cfg, args.shots, args.seed)

        sensitive = (init_label == "0" and p_x > 0) or (init_label == "+" and p_z > 0)
        tol_commuting = max(noise_floor, 0.05 * args.p)
        status = "OK"
        expectation = "commuting" if not sensitive else "sensitive"
        if not sensitive and result.logical_error_rate > tol_commuting:
            status = "MISMATCH"
        if sensitive and result.logical_error_rate <= noise_floor:
            status = "MISMATCH"

        print(
            f"{label:12s} | init={init_label:>2s} | "
            f"logical_error_rate={result.logical_error_rate:.4f} "
            f"({expectation}, {status})"
        )


if __name__ == "__main__":
    main()
