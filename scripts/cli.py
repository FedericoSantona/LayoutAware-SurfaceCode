"""Command-line interface for playground.py - clean separation of CLI logic."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Type

from qiskit import QuantumCircuit
from qiskit.transpiler import Target

# Ensure src/ directory is available for imports
import sys
CLI_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = CLI_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from benchmarks.BenchmarkCircuit import BenchmarkCircuit
from benchmarks.circuits.bell import BellStateBenchmark
from benchmarks.circuits.ghz import GHZ3Benchmark
from benchmarks.circuits.parity_check import ParityCheckBenchmark
from benchmarks.circuits.simple import Simple1QXZHBenchmark
from transpile.config import TranspileConfig


# ---------------------------------------------------------------------------
# Benchmark registry (extend as you add new logical templates)
# ---------------------------------------------------------------------------

BENCHMARKS: Dict[str, Type[BenchmarkCircuit]] = {
    "bell": BellStateBenchmark,
    "ghz3": GHZ3Benchmark,
    "parity_check": ParityCheckBenchmark,
    "simple_1q_xzh": Simple1QXZHBenchmark,
}


def _str2bool(value) -> bool:
    """Convert string to boolean for argparse."""
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
    """Build and return the argument parser with all CLI options."""
    parser = argparse.ArgumentParser(
        description="Transpile logical benchmarks and optionally simulate logical 1Q sequences on the surface code.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run logical encoding simulation with default settings
  python scripts/playground.py --logical-encoding true

  # Run transpilation benchmark
  python scripts/playground.py --transpilation true --benchmark bell

  # Run simulation with custom parameters
  python scripts/playground.py --benchmark ghz3 --distance 5 --shots 50000 --px 1e-4

  # Run both branches
  python scripts/playground.py --logical-encoding true --transpilation true
        """
    )
    
    # Main execution flags
    main_group = parser.add_argument_group("Main execution flags")
    main_group.add_argument(
        "--benchmark",
        default="simple_1q_xzh", #simple_1q_xzh, bell, parity_check, ghz3, simple_1q_xzh, simulate, simulate_correlated
        choices=sorted(BENCHMARKS.keys()),
        help="Logical circuit template (used for transpile or simulation).",
    )
    main_group.add_argument(
        "--logical-encoding",
        type=_str2bool,
        default=True,
        help="Run the logical encoding/surface-code simulation branch (true/false).",
    )
    main_group.add_argument(
        "--transpilation",
        type=_str2bool,
        default=False,
        help="Run the heavy-hex transpilation branch (true/false).",
    )
    
    # Transpilation options
    transpile_group = parser.add_argument_group("Transpilation options")
    transpile_group.add_argument(
        "--target",
        default="fake_oslo",
        choices=sorted(TARGET_LOADERS.keys()),
        help="Hardware target to load (extend via TARGET_LOADERS).",
    )
    transpile_group.add_argument(
        "--seeds",
        type=int,
        default=8,
        help="How many layout/routing seeds to explore.",
    )
    transpile_group.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset applied to the seed stream for reproducibility tweaking.",
    )
    transpile_group.add_argument(
        "--keep-top-k",
        type=int,
        default=3,
        help="How many candidates to keep in the leaderboard.",
    )
    transpile_group.add_argument(
        "--schedule-mode",
        default="alap",
        choices=["alap", "asap"],
        help="Scheduling policy passed to the pipeline.",
    )
    transpile_group.add_argument(
        "--dd-policy",
        default="none",
        choices=["none", "XIX", "XYXY"],
        help="Optional dynamical decoupling sequence.",
    )
    
    # Simulation options
    sim_group = parser.add_argument_group("Simulation options")
    sim_group.add_argument(
        "--distance",
        type=int,
        default=7,
        help="Code distance d for the surface code"
    )
    sim_group.add_argument(
        "--stim-memory",
        type=bool,
        default=False,
        help="Use Stim's built-in rotated-memory circuit for single-patch benchmarks.",
    )
    sim_group.add_argument(
        "--code-type",
        choices=["heavy_hex", "standard"],
        default="standard",
        help="Type of surface code to use (default: standard)",
    )
    sim_group.add_argument(
        "--rounds",
        type=int,
        default=7,
        help="Number of measurement rounds (default: distance)"
    )
    sim_group.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
        help="Number of warmup rounds (default: 1)"
    )
    sim_group.add_argument(
        "--ancilla-buffer",
        type=float,
        default=1.0,
        help="Buffer spacing between ancilla and template patch (default: 1.0)"
    )
    sim_group.add_argument(
        "--px",
        type=float,
        default=1e-3,
        help="Phenomenological X error probability"
    )
    sim_group.add_argument(
        "--pz",
        type=float,
        default=1e-3,
        help="Phenomenological Z error probability"
    )
    sim_group.add_argument(
        "--p-meas",
        type=float,
        default=1e-12,
        help="Measurement error probability (phenomenological noise on measurement results). Default 0.0 for code-capacity mode."
    )
    sim_group.add_argument(
        "--init",
        type=str,
        default="0",
        help="Logical initialization: one of {0,1,+,-}, always 0, then apply gates"
    )
    sim_group.add_argument(
        "--bracket-basis",
        type=str,
        choices=["auto", "Z", "X"],
        default="auto",
        help="Choose which logical basis to bracket in the DEM (start/end observable). 'auto' derives from --init (Z for 0/1, X for +/-).",
    )
    sim_group.add_argument(
        "--demo-basis",
        type=str,
        choices=["auto", "Z", "X", "none"],
        default="auto",
        help="End-only demo readout basis for reporting. 'auto' uses logical end basis when it differs from the bracket; 'none' disables the demo readout.",
    )
    sim_group.add_argument(
        "--shots",
        type=int,
        default=10**5,
        help="Number of Monte Carlo samples"
    )
    sim_group.add_argument(
        "--seed",
        type=int,
        default=46,
        help="Seed for Stim samplers"
    )
    sim_group.add_argument(
        "--seam-json",
        type=Path,
        default=None,
        help="Optional JSON path specifying explicit seam pairs per CNOT ((kind,a,b)->pairs).",
    )
    sim_group.add_argument(
        "--cnot-ancilla-strategy",
        type=str,
        choices=["serialize", "parallelize"],
        default="serialize",
        help="CNOT ancilla allocation strategy: 'serialize' reuses one ancilla (default), 'parallelize' uses multiple ancillas for concurrent CNOTs.",
    )
    sim_group.add_argument(
        "--corr-pairs",
        type=str,
        default=None,
        help="Custom correlation pairs in format 'q0,q1;q2,q3' for two-qubit correlation analysis.",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--dump-json",
        type=Path,
        help="Optional path to dump a JSON snapshot of the results.",
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output including raw/expected/decoded distributions and detailed diagnostics.",
    )
    
    return parser


def parse_args(args=None):
    """Parse command-line arguments and return the namespace."""
    return build_parser().parse_args(args)


def instantiate_benchmark(name: str) -> BenchmarkCircuit:
    """Instantiate a benchmark by name."""
    cls = BENCHMARKS[name]
    return cls()


def load_target(name: str) -> Target:
    """Load a hardware target by name."""
    loader = TARGET_LOADERS[name]
    return loader(name)


def build_config(target: Target, args: argparse.Namespace) -> TranspileConfig:
    """Build TranspileConfig from parsed arguments."""
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
    """Format leaderboard results for JSON output."""
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
