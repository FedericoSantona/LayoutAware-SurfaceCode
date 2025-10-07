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
from benchmarks.circuits.teleportation import TeleportationBenchmark
from qiskit import QuantumCircuit
from qiskit.transpiler import Target

from transpile.config import TranspileConfig
from transpile.pipeline import HeavyHexTranspiler


# ---------------------------------------------------------------------------
# Benchmark registry (extend as you add new logical templates)
# ---------------------------------------------------------------------------

BENCHMARKS: Dict[str, Type[BenchmarkCircuit]] = {
    "bell": BellStateBenchmark,
    "ghz3": GHZ3Benchmark,
    "parity_check": ParityCheckBenchmark,
    "teleportation": TeleportationBenchmark,
}


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
        description="Run the heavy-hex transpilation pipeline on a benchmark circuit.",
    )
    parser.add_argument(
        "--benchmark",
        default="parity_check", # options are bell, ghz3, parity_check, teleportation
        choices=sorted(BENCHMARKS.keys()),
        help="Logical circuit template to transpile.",
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

    benchmark = instantiate_benchmark(args.benchmark)
    logical_metrics = benchmark.compute_logical_metrics()

    target = load_target(args.target)
    config = build_config(target, args)
    transpiler = HeavyHexTranspiler(config)

    logical_circuit = benchmark.get_circuit()
    best, best_metrics, leaderboard = transpiler.run_baseline(logical_circuit)

    leaderboard_payload = format_leaderboard(leaderboard)
    payload = {
        "benchmark": args.benchmark,
        "target": args.target,
        "logical_metrics": logical_metrics,
        "best_metrics": best_metrics,
        "leaderboard": leaderboard_payload,
    }

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

    if args.dump_json:
        args.dump_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nDumped JSON snapshot to {args.dump_json}")


if __name__ == "__main__":
    main()
