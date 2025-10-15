"""Quick playground for the heavy-hex transpilation pipeline.

This script stays intentionally thin: it wires together the high-level
components already defined in the project so you can iterate on new QEC
experiments without re-implementing logic. Extend it as you develop new
benchmarks, targets, or analysis steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Type
import importlib
import os
import sys
if __name__ == "__main__":
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

# Ensure the src/ directory is available for imports when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from benchmarks.BenchmarkCircuit import BenchmarkCircuit
from benchmarks.circuits.bell import BellStateBenchmark
from benchmarks.circuits.ghz import GHZ3Benchmark
from benchmarks.circuits.parity_check import ParityCheckBenchmark
from benchmarks.circuits.simple import Simple1QXZHBenchmark
from benchmarks.circuits.teleportation import TeleportationBenchmark
from surface_code.utils import (
    plot_heavy_hex_code,
    diagnostic_print,
    print_logical_results,
    compute_pauli_frame_stats,
)
from qiskit import QuantumCircuit
from qiskit.transpiler import Target

from transpile.config import TranspileConfig
from transpile.pipeline import HeavyHexTranspiler
from surface_code import (
    build_heavy_hex_model,
    PhenomenologicalStimBuilder,
    PhenomenologicalStimConfig,
)
from surface_code.logical_ops import (
    LogicalFrame,
    apply_sequence,
    end_basis_and_flip,
    circuit_to_gates,
    parse_init_label,
)
from simulation import MonteCarloConfig, run_circuit_logical_error_rate


# ---------------------------------------------------------------------------
# Benchmark registry (extend as you add new logical templates)
# ---------------------------------------------------------------------------

BENCHMARKS: Dict[str, Type[BenchmarkCircuit]] = {
    "bell": BellStateBenchmark,
    "ghz3": GHZ3Benchmark,
    "parity_check": ParityCheckBenchmark,
    "teleportation": TeleportationBenchmark,
    "simple_1q_xzh": Simple1QXZHBenchmark,
}


def _str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value for flag, got '{value}'.")


# ---------------------------------------------------------------------------
# Hardware target helpers
# ---------------------------------------------------------------------------

def _fake_backend_target(name: str) -> Target:
    """Load a fake heavy-hex backend from the runtime or legacy fake providers."""

    provider_modules = (
        "qiskit_ibm_runtime.fake_provider",
        "qiskit.providers.fake_provider",
    )

    candidates = {
        "fake_manila": ["FakeManilaV2", "FakeManila"],
        "fake_montreal": ["FakeMontrealV2", "FakeMontreal"],
        "fake_oslo": ["FakeOsloV2", "FakeOslo"],
        "fake_sherbrooke": ["FakeSherbrookeV2", "FakeSherbrooke"],
        "fake_ithaca": ["FakeIthacaV2", "FakeIthaca"],
    }

    if name not in candidates:
        raise ValueError(f"Unsupported fake backend '{name}'. Choices: {sorted(candidates)}")

    last_error: Exception | None = None
    for module_name in provider_modules:
        try:
            provider = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - import guard
            last_error = exc
            continue

        for attr in candidates[name]:
            backend_factory = getattr(provider, attr, None)
            if backend_factory is None:
                continue
            backend = backend_factory()
            target = getattr(backend, "target", None)
            if target is None:
                raise RuntimeError(
                    f"Backend {attr} does not expose a Target; update Qiskit or choose another backend."
                )
            return target

    if last_error is not None:
        raise ImportError(
            "Unable to import Qiskit's fake providers. Install 'qiskit-ibm-runtime' or base 'qiskit'."
        ) from last_error

    raise RuntimeError(
        f"None of the fake backend factories {candidates[name]} are available in the installed Qiskit packages."
    )


TARGET_LOADERS = {
    "fake_manila": _fake_backend_target,
    "fake_montreal": _fake_backend_target,
    "fake_oslo": _fake_backend_target,
    "fake_sherbrooke": _fake_backend_target,
    "fake_ithaca": _fake_backend_target,
}


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transpile logical benchmarks and optionally simulate logical 1Q sequences on the surface code.",
    )
    parser.add_argument(
        "--benchmark",
        default="simple_1q_xzh", # options are bell, ghz3, parity_check, teleportation, simple_1q_xzh
        choices=sorted(BENCHMARKS.keys()),
        help="Logical circuit template (used for transpile or simulation).",
    )
    parser.add_argument(
        "--logical-encoding",
        type=_str2bool,
        default=True,
        help="Run the logical encoding/surface-code simulation branch (true/false).",
    )
    parser.add_argument(
        "--transpilation",
        type=_str2bool,
        default=False,
        help="Run the heavy-hex transpilation branch (true/false).",
    )
    parser.add_argument(
        "--target",
        default="fake_oslo",
        choices=sorted(TARGET_LOADERS.keys()),
        help="Hardware target to load (extend via TARGET_LOADERS).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=8,
        help="How many layout/routing seeds to explore.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset applied to the seed stream for reproducibility tweaking.",
    )
    parser.add_argument(
        "--keep-top-k",
        type=int,
        default=3,
        help="How many candidates to keep in the leaderboard.",
    )
    parser.add_argument(
        "--schedule-mode",
        default="alap",
        choices=["alap", "asap"],
        help="Scheduling policy passed to the pipeline.",
    )
    parser.add_argument(
        "--dd-policy",
        default="none",
        choices=["none", "XIX", "XYXY"],
        help="Optional dynamical decoupling sequence.",
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        help="Optional path to dump a JSON snapshot of the results.",
    )
    # Simulation options (used when logical encoding branch is enabled)

    parser.add_argument("--distance", type=int, default=5, help="Code distance d for the heavy-hex code")
    parser.add_argument("--rounds", type=int, default=None, help="Number of measurement rounds (default: distance)")
    parser.add_argument("--px", type=float, default=5e-3, help="Phenomenological X error probability")
    parser.add_argument("--pz", type=float, default=5e-3, help="Phenomenological Z error probability")
    parser.add_argument("--init", type=str, default="0", help="Logical initialization: one of {0,1,+,-} , always 0, then apply gates")
    parser.add_argument(
        "--bracket-basis",
        type=str,
        choices=["auto", "Z", "X"],
        default="auto",
        help="Choose which logical basis to bracket in the DEM (start/end observable). 'auto' derives from --init (Z for 0/1, X for +/-).",
    )
    parser.add_argument(
        "--demo-basis",
        type=str,
        choices=["auto", "Z", "X", "none"],
        default="auto",
        help="End-only demo readout basis for reporting. 'auto' uses logical end basis when it differs from the bracket; 'none' disables the demo readout.",
    )
    parser.add_argument("--shots", type=int, default=10**6, help="Number of Monte Carlo samples")
    parser.add_argument("--seed", type=int, default=46, help="Seed for Stim samplers")
    
    return parser


def instantiate_benchmark(name: str) -> BenchmarkCircuit:
    cls = BENCHMARKS[name]
    return cls()


def load_target(name: str) -> Target:
    loader = TARGET_LOADERS[name]
    return loader(name)


def build_config(target: Target, args: argparse.Namespace) -> TranspileConfig:
    dd_policy = None if args.dd_policy.lower() == "none" else args.dd_policy
    return TranspileConfig(
        target=target,
        seeds=args.seeds,
        seed_offset=args.seed_offset,
        schedule_mode=args.schedule_mode,
        dd_policy=dd_policy,
        keep_top_k=args.keep_top_k,
    )


def format_leaderboard(
    leaderboard: Sequence[Tuple[QuantumCircuit, Dict[str, object]]]
) -> List[Dict[str, object]]:
    formatted: List[Dict[str, object]] = []
    for rank, (qc, metrics) in enumerate(leaderboard, start=1):
        formatted.append(
            {
                "rank": rank,
                "name": qc.name,
                "metrics": metrics,
            }
        )
    return formatted


def main() -> None:
    args = build_parser().parse_args()

    run_logical = bool(args.logical_encoding)
    run_transpile = bool(args.transpilation)

    if not run_logical and not run_transpile:
        print("No action requested: enable --logical-encoding true and/or --transpilation true.")
        return

    json_payloads: List[Dict[str, object]] = []

    if run_logical:
        benchmark = instantiate_benchmark(args.benchmark)
        qc = benchmark.get_circuit()

        gate_seq = circuit_to_gates(qc)
        frame = apply_sequence(LogicalFrame(), gate_seq)

        init_label = (args.init or "0").strip()
        # Parse basis and eigen-sign (+1/-1) from init label
        start_basis, init_sign = parse_init_label(init_label)
        end_basis, expected_flip = end_basis_and_flip(start_basis, frame)
        # Incorporate the initial eigen-sign into the expected correlation parity.
        # A '-1' eigenstate contributes a unit flip to the start-vs-end parity definition.
        expected_flip_total = int(expected_flip) ^ (0 if init_sign == +1 else 1)

        model = build_heavy_hex_model(args.distance)

        plot_heavy_hex_code(model, args.distance)

        diagnostic_print(model, args)

        stim_rounds = int(args.rounds) if args.rounds is not None else int(args.distance)
        # Resolve bracket basis: auto derives from init label
        if (args.bracket_basis or "auto").lower() == "auto":
            bracket_basis = start_basis
        else:
            bracket_basis = args.bracket_basis.strip().upper()
        # Resolve demo basis from CLI (auto/Z/X/none)
        db_mode = (args.demo_basis or "auto").lower()
        if db_mode == "none":
            demo_basis = None
        elif db_mode == "auto":
            # Physics-based reporting in the end basis: request an end-only demo
            # MPP if the end basis differs from the bracket basis.
            demo_basis = end_basis if end_basis != bracket_basis else None
        else:
            demo_basis = args.demo_basis.strip().upper()
        stim_cfg = PhenomenologicalStimConfig(
            rounds=stim_rounds,
            p_x_error=float(args.px),
            p_z_error=float(args.pz),
            init_label=init_label,
            logical_start=None,
            logical_end=None,
            bracket_basis=bracket_basis,
            demo_basis=demo_basis,
        )
        builder = PhenomenologicalStimBuilder(
            code=model.code,
            z_stabilizers=model.z_stabilizers,
            x_stabilizers=model.x_stabilizers,
            logical_z=model.logical_z,
            logical_x=model.logical_x,
        )
        circuit, observable_pairs, metadata = builder.build(stim_cfg)
        result = run_circuit_logical_error_rate(
            circuit,
            observable_pairs,
            stim_cfg,
            MonteCarloConfig(shots=int(args.shots), seed=int(args.seed) if args.seed is not None else None),
            metadata,
        )
        # If demo bits are present for physics-based end-basis reporting, compute
        # stats from them without applying decoder corrections (decoder doesn't act
        # on the demo readout). Otherwise, fall back to DEM observable.
        if getattr(result, "demo_bits", None) is not None and result.demo_bits is not None:
            frame_stats = compute_pauli_frame_stats(
                result,
                end_basis,
                expected_flip_total,
                override_raw=result.demo_bits,
                apply_decoder=False,
            )
        else:
            frame_stats = compute_pauli_frame_stats(result, end_basis, expected_flip_total)

        print_logical_results(
            args,
            gate_seq,
            start_basis,
            init_sign,
            end_basis,
            expected_flip_total,
            stim_rounds,
            result,
            frame_stats,
        )

        json_payloads.append(
            {
                "branch": "logical_encoding",
                "benchmark": args.benchmark,
                "sequence": gate_seq,
                "start_basis": start_basis,
                "end_basis": end_basis,
                "init_sign": int(init_sign),
                "expected_flip": int(expected_flip),
                "expected_flip_total": int(expected_flip_total),
                "decoder_frame_basis": end_basis,
                "distance": int(args.distance),
                "rounds": int(stim_rounds),
                "p_x": float(args.px),
                "p_z": float(args.pz),
                "shots": int(result.shots),
                "logical_error_rate": float(result.logical_error_rate),
                "avg_syndrome_weight": float(result.avg_syndrome_weight),
                "click_rate": float(result.click_rate),
                "frame_flip": int(expected_flip_total),
                "logical_mean_raw": float(frame_stats.logical_means["raw"]),
                "logical_mean_expected_frame": float(frame_stats.logical_means["expected"]),
                "logical_mean_decoder_frame": float(frame_stats.logical_means["decoder"]),
                "decoded_mean_raw": float(frame_stats.decoded_means["raw"]),
                "decoded_mean_expected_frame": float(frame_stats.decoded_means["expected"]),
                "decoded_mean_decoder_frame": float(frame_stats.decoded_means["decoder"]),
                "decoder_frame_flip_rate": float(frame_stats.decoder_flip_rate),
                "tracked_frame_flip_rate": float(frame_stats.tracked_frame_rate),
                "logical_prob_raw": {k: float(v) for k, v in frame_stats.logical_probs["raw"].items()},
                "logical_prob_expected_frame": {k: float(v) for k, v in frame_stats.logical_probs["expected_frame"].items()},
                "logical_prob_decoder_frame": {k: float(v) for k, v in frame_stats.logical_probs["decoder_frame"].items()},
                "decoded_prob_raw": {k: float(v) for k, v in frame_stats.decoded_probs["raw"].items()},
                "decoded_prob_expected_frame": {k: float(v) for k, v in frame_stats.decoded_probs["expected_frame"].items()},
                "decoded_prob_decoder_frame": {k: float(v) for k, v in frame_stats.decoded_probs["decoder_frame"].items()},
                "tracked_frame_prob": {k: float(v) for k, v in frame_stats.frame_prob.items()},
            }
        )

    if run_transpile:
        benchmark = instantiate_benchmark(args.benchmark)
        logical_metrics = benchmark.compute_logical_metrics()

        target = load_target(args.target)
        config = build_config(target, args)
        transpiler = HeavyHexTranspiler(config)

        logical_circuit = benchmark.get_circuit()
        best, best_metrics, leaderboard = transpiler.run_baseline(logical_circuit)

        leaderboard_payload = format_leaderboard(leaderboard)
        print(f"Benchmark: {args.benchmark}")
        print(f"Target: {args.target}")
        print(f"Logical metrics: {logical_metrics}")
        print(f"Best mapped metrics: {best_metrics}")
        print("Leaderboard (top candidates):")
        for entry in leaderboard_payload:
            metrics = entry["metrics"]
            print(
                "  #{rank}: depth={depth} twoq={twoq} swaps={swaps} duration_ns={dur}".format(
                    rank=entry["rank"],
                    depth=metrics.get("depth"),
                    twoq=metrics.get("twoq"),
                    swaps=metrics.get("swaps"),
                    dur=metrics.get("duration_ns"),
                )
            )

        json_payloads.append(
            {
                "branch": "transpilation",
                "benchmark": args.benchmark,
                "target": args.target,
                "logical_metrics": logical_metrics,
                "best_metrics": best_metrics,
                "leaderboard": leaderboard_payload,
            }
        )

    if args.dump_json and json_payloads:
        payload_out: List[Dict[str, object]] | Dict[str, object]
        if len(json_payloads) == 1:
            payload_out = json_payloads[0]
        else:
            payload_out = json_payloads
        args.dump_json.write_text(json.dumps(payload_out, indent=2, sort_keys=True))
        print(f"\nDumped JSON snapshot to {args.dump_json}")


if __name__ == "__main__":
    main()
