"""Quick playground for the heavy-hex transpilation pipeline.

This script stays intentionally thin: it wires together the high-level
components already defined in the project so you can iterate on new QEC
experiments without re-implementing logic. Extend it as you develop new
benchmarks, targets, or analysis steps.
"""

from __future__ import annotations

import importlib
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Type

# Add the project root to Python path to enable src imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.surface_code.builder import augment_dem_with_boundary_anchors

import numpy as np
import pymatching as pm
import stim

# --------------------------
# MWPM / DEM debug utilities
# --------------------------
from collections import defaultdict
import re

# Parse 'error(...) D# D# L#' lines from a stim.DetectorErrorModel
_D_TOKEN_RE = re.compile(r"D(\d+)")
_L_TOKEN_RE = re.compile(r"L(\d+)")

def _parse_dem_errors(dem):
    """
    Return a list of elementary faults as dicts:
        {"detectors":[int,...], "observables":[int,...], "raw": line}
    """
    out = []
    for line in str(dem).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not s.lower().startswith("error"):
            continue
        det_ids = [int(m.group(1)) for m in _D_TOKEN_RE.finditer(s)]
        obs_ids = [int(m.group(1)) for m in _L_TOKEN_RE.finditer(s)]
        out.append({"detectors": det_ids, "observables": obs_ids, "raw": s})
    return out

def _pm_find_offending_shot(matcher, det_samp, *, max_scan=2048):
    """Inspect the actual PyMatching graph for boundaryless odd-parity components.

    Prints the first offending shot+component if found. Returns True if an
    offender was found, else False. Also reports isolated detector nodes.
    """
    try:
        import numpy as _np
        import networkx as _nx
    except Exception as _exc:
        print(f"[PM-CHECK] skipped (imports failed): {_exc}")
        return False

    try:
        G = matcher.to_networkx()
    except Exception as _exc2:
        print(f"[PM-CHECK] to_networkx failed: {_exc2}")
        return False

    num_det = getattr(matcher, 'num_detectors', 0)
    comps = list(_nx.connected_components(G))
    boundary_nodes = {n for n, d in G.nodes(data=True) if d.get('is_boundary', False)}

    comp_info = []
    for comp_id, comp in enumerate(comps):
        det_nodes = sorted([n for n in comp if isinstance(n, int) and n < num_det])
        has_boundary = any(n in boundary_nodes for n in comp)
        comp_info.append((comp_id, det_nodes, has_boundary))

    nscan = min(det_samp.shape[0], int(max_scan)) if det_samp is not None else 0
    for s in range(nscan):
        syn = det_samp[s].astype(_np.uint8)
        for comp_id, det_nodes, has_boundary in comp_info:
            if not det_nodes:
                continue
            parity = int(_np.bitwise_xor.reduce(syn[det_nodes])) if det_nodes else 0
            if (parity & 1) and (not has_boundary):
                print(f"[PM-CHECK] offending shot={s}, COMP#{comp_id}, has_boundary={has_boundary}, size={len(det_nodes)}")
                print(f"  first few det nodes: {det_nodes[:16]}")
                return True

    print("[PM-CHECK] no boundaryless odd-parity component found in scanned shots.")
    # Check for isolated detector nodes (degree 0) and whether any shot hits them
    deg = dict(G.degree())
    isolates = [n for n in range(num_det) if deg.get(n, 0) == 0]
    if isolates:
        print(f"[PM-CHECK] isolated detector nodes (deg 0): {isolates[:16]}")
        for s in range(nscan):
            syn = det_samp[s].astype(_np.uint8)
            if any(int(syn[n]) & 1 for n in isolates if n < len(syn)):
                print(f"[PM-CHECK] shot {s} has 1s on isolated detectors (impossible without boundary).")
                break
    return False


def _anchor_pm_isolates(dem, matcher, *, epsilon=1e-12):
    """Ensure every detector in the PyMatching graph has non-zero degree by adding boundary hooks.

    Returns (new_dem, new_matcher, num_added, isolate_ids).
    """
    try:
        import networkx as _nx
    except Exception:
        return dem, matcher, 0, []

    try:
        G = matcher.to_networkx()
    except Exception:
        return dem, matcher, 0, []

    num_det = getattr(matcher, "num_detectors", 0)
    deg = dict(G.degree())
    isolates = [n for n in range(num_det) if deg.get(n, 0) == 0]
    if not isolates:
        return dem, matcher, 0, []

    dem_text = str(dem)
    if dem_text and not dem_text.endswith("\n"):
        dem_text += "\n"
    p_str = f"{float(epsilon):.12g}"
    for idx in isolates:
        dem_text += f"error({p_str}) D{idx}\n"
    new_dem = stim.DetectorErrorModel(dem_text)
    new_matcher = pm.Matching.from_detector_error_model(new_dem)
    return new_dem, new_matcher, len(isolates), isolates

def _build_components_from_dem(dem):
    """
    Build detector connected components from the DEM.
    Link detectors that co-appear in any elementary fault.
    Also mark if a component 'has boundary' (∃ fault with an odd number of
    detectors from that component, usually a single D# flip).
    """
    n = dem.num_detectors
    errors = _parse_dem_errors(dem)

    # Union–Find
    parent = list(range(n))
    rank = [0] * n
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # Link detectors that appear together in an error
    for err in errors:
        ds = err["detectors"]
        for i in range(len(ds) - 1):
            union(ds[i], ds[i + 1])

    # Group by root
    comps_dict = defaultdict(list)
    for d in range(n):
        root = find(d)
        comps_dict[root].append(d)
    comps = [sorted(vs) for vs in comps_dict.values()]

    # Boundary flag per component: odd intersection with some fault
    idx_to_comp = {}
    for ci, comp in enumerate(comps):
        for d in comp:
            idx_to_comp[d] = ci

    comp_has_boundary = [False] * len(comps)
    for err in errors:
        touched = defaultdict(int)
        for d in err["detectors"]:
            ci = idx_to_comp.get(d)
            if ci is not None:
                touched[ci] ^= 1
        for ci, parity in touched.items():
            if parity & 1:
                comp_has_boundary[ci] = True

    return comps, comp_has_boundary, errors

def _report_boundaryless_components(dem):
    comps, comp_has_boundary, _ = _build_components_from_dem(dem)
    n = len(comps)
    nob = sum(1 for b in comp_has_boundary if not b)
    print(f"[DEM-CHECK] components={n}, boundaryless={nob}")
    if nob:
        for i, comp in enumerate(comps):
            if comp_has_boundary[i]:
                continue
            head = comp[:24]
            tail = comp[-8:] if len(comp) > 32 else []
            print(f"  [COMP#{i}] size={len(comp)} head={head}{' tail='+str(tail) if tail else ''}")

def _scan_boundaryless_odd_shot(dem, det_samp, *, max_scan=512):
    if det_samp is None or dem.num_detectors == 0:
        return
    comps, comp_has_boundary, _ = _build_components_from_dem(dem)
    comp_sets = [set(c) for c in comps]
    nscan = min(det_samp.shape[0], int(max_scan))
    for s in range(nscan):
        syn = det_samp[s].astype(np.uint8)
        for ci, comp in enumerate(comps):
            if not comp or comp_has_boundary[ci]:
                continue
            # odd parity in a boundaryless component → infeasible
            parity = int(np.bitwise_xor.reduce(syn[comp]))
            if parity & 1:
                print(f"[DEM-CHECK] offending shot={s} boundaryless COMP#{ci} size={len(comp)} sample={comp[:16]}")
                print("  detslice filter:", "[" + ", ".join(f"'D{d}'" for d in comp[:12]) + "]")
                # quick arity histogram for faults touching this comp
                errs = _parse_dem_errors(dem)
                hist = {}
                for e in errs:
                    k = sum(1 for d in e['detectors'] if d in comp_sets[ci])
                    if k:
                        hist[k] = hist.get(k, 0) + 1
                print("  error-arity histogram:", dict(sorted(hist.items())))
                # show a few example error lines touching the component (odd and even)
                odd_examples = []
                even_examples = []
                for e in errs:
                    k = sum(1 for d in e['detectors'] if d in comp_sets[ci])
                    if k == 0:
                        continue
                    (odd_examples if (k % 2 == 1) else even_examples).append(e["raw"])
                    if len(odd_examples) >= 4 and len(even_examples) >= 4:
                        break
                if odd_examples:
                    print("  example odd-intersection errors:")
                    for ln in odd_examples[:4]:
                        print("    ", ln)
                if even_examples:
                    print("  example even-intersection errors:")
                    for ln in even_examples[:4]:
                        print("    ", ln)
                return

def harden_dem_add_boundaries(dem, *, epsilon=1e-12):
     """
     For each detector connected component that currently has no boundary
     (no fault with an odd intersection), append an infinitesimal-probability
     single-detector error to create a boundary hook. This guarantees MWPM
     feasibility without materially changing statistics.
     """
     comps, comp_has_boundary, _ = _build_components_from_dem(dem)
     added = 0
     for ci, comp in enumerate(comps):
         if comp and not comp_has_boundary[ci]:
             # Hook the first detector in this component
             d0 = comp[0]
             dem.append("error", epsilon, [stim.DemTarget.detector(d0)])
             added += 1
     return added

# --------------------------------------------------------
# Harden DEM for pairwise matching: ensure every component has a singleton
# --------------------------------------------------------
def harden_dem_for_pairwise_matching(dem, *, epsilon=1e-12):
    """
    Ensure every connected component has at least one *single-detector*
    error (a real boundary edge for pairwise MWPM).
    Returns a new DEM and the number of hooks added.
    """
    comps, _, errors = _build_components_from_dem(dem)

    # Detectors that already have a 1-detector error
    singleton_det_ids = set()
    for e in errors:
        ds = e["detectors"]
        if len(ds) == 1:
            singleton_det_ids.add(ds[0])

    hook_ids = []
    for comp in comps:
        # Skip components that already contain a singleton detector error
        if any(d in singleton_det_ids for d in comp):
            continue
        # Need to add a hook for this component
        d0 = comp[0]
        hook_ids.append(d0)
    added = len(hook_ids)

    dem_text = str(dem)
    if not dem_text.endswith('\n'):
        dem_text += '\n'
    p_str = f"{float(epsilon):.12g}"
    for d0 in hook_ids:
        dem_text += f"error({p_str}) D{d0}\n"
    new_dem = stim.DetectorErrorModel(dem_text)
    return new_dem, added
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
from surface_code.utils import (
    plot_heavy_hex_code,
    diagnostic_print,
    wilson_rate_ci,
    compute_two_qubit_correlations,
    compute_joint_correlations,
)
from surface_code.reporting import (
    print_header,
    print_per_qubit_results,
    print_physics_demo,
    print_pauli_frame_audit,
    print_debug_details,
    _print_demo_preamble,
    generate_detailed_json,
    save_detailed_json,
    print_final_state_distribution,
    print_decoded_logical_distribution,
    print_schrodinger_snapshot_distribution,
)
from qiskit import QuantumCircuit
from qiskit.transpiler import Target

from transpile.config import TranspileConfig
from transpile.pipeline import HeavyHexTranspiler
from surface_code import (
    build_heavy_hex_model,
    PhenomenologicalStimConfig,
)
from surface_code.layout import PatchObject
from surface_code.surgery_compile import compile_circuit_to_surgery
from surface_code.builder import GlobalStimBuilder
from surface_code.joint_parity import decode_joint_parity
from surface_code.pauli import PauliTracker, parse_init_label, sequence_from_qc


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
        default="bell", # options are bell, ghz3, parity_check, simple_1q_xzh
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

    parser.add_argument("--distance", type=int, default=3, help="Code distance d for the heavy-hex code")
    parser.add_argument("--rounds", type=int, default=None, help="Number of measurement rounds (default: distance)")
    parser.add_argument("--warmup-rounds", type=int, default=1, help="Number of warmup rounds (default: 1)")
    parser.add_argument("--ancilla-buffer", type=float, default=1.0, help="Buffer spacing between ancilla and template patch (default: 1.0)")
    parser.add_argument("--px", type=float, default=1e-3, help="Phenomenological X error probability")
    parser.add_argument("--pz", type=float, default=1e-3, help="Phenomenological Z error probability")
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
    parser.add_argument("--shots", type=int, default=10**4, help="Number of Monte Carlo samples")
    parser.add_argument("--seed", type=int, default=46, help="Seed for Stim samplers")
    parser.add_argument(
        "--seam-json",
        type=Path,
        default=None,
        help="Optional JSON path specifying explicit seam pairs per CNOT ((kind,a,b)->pairs).",
    )
    parser.add_argument(
        "--cnot-ancilla-strategy",
        type=str,
        choices=["serialize", "parallelize"],
        default="serialize",
        help="CNOT ancilla allocation strategy: 'serialize' reuses one ancilla (default), 'parallelize' uses multiple ancillas for concurrent CNOTs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output including raw/expected/decoded distributions and detailed diagnostics.",
    )
    parser.add_argument(
        "--corr-pairs",
        type=str,
        default=None,
        help="Custom correlation pairs in format 'q0,q1;q2,q3' for two-qubit correlation analysis.",
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

    run_logical = bool(args.logical_encoding)
    run_transpile = bool(args.transpilation)

    if not run_logical and not run_transpile:
        print("No action requested: enable --logical-encoding true and/or --transpilation true.")
        return

    json_payloads: List[Dict[str, object]] = []

    if run_logical:
        benchmark = instantiate_benchmark(args.benchmark)
        qc = benchmark.get_circuit()

        # Multi-qubit unified path: gate summary and start/end basis
        init_label = (args.init or "0").strip()
        start_basis, _ = parse_init_label(init_label)
        gate_seq = [ci.operation.name.upper() for ci in qc.data]
        # Global end-basis heuristic for demo auto: toggle basis if any H is present
        any_h = any(ci.operation.name.lower() == "h" for ci in qc.data)
        end_basis = ("X" if start_basis == "Z" else "Z") if any_h else start_basis

        # Unified surgery-based path (works for any number of qubits)
        model = build_heavy_hex_model(args.distance)

        plot_heavy_hex_code(model, args.distance)
        if args.verbose:
            diagnostic_print(model, args)
    

        d = int(args.distance)
        stim_rounds = int(args.rounds) if args.rounds is not None else d
        warmup_rounds = int(args.warmup_rounds)

        # Resolve bracket basis for all patches
        if (args.bracket_basis or "auto").lower() == "auto":
            bracket_basis = start_basis
        else:
            bracket_basis = args.bracket_basis.strip().upper()
        db_mode = (args.demo_basis or "auto").lower()
        if db_mode == "none":
            demo_basis = None
        elif db_mode == "auto":
            demo_basis = end_basis if end_basis != bracket_basis else None
        else:
            # Handle single basis (Z or X) or comma-separated formats like "Z,X"
            demo_str = args.demo_basis.strip().upper()
            demo_basis = demo_str

        # Build one PatchObject per qubit
        def build_patch() -> PatchObject:
            return PatchObject.from_code_model(model)

        # Create patches with spatial offsets to avoid overlap
        patches = {}
        for i in range(qc.num_qubits):
            patch = build_patch()
            # Offset each patch horizontally to prevent overlap
            patches[f"q{i}"] = patch.with_offset(0, i * 3.0, 0.0)

        # Start with empty seams - let _auto_generate_seams() handle everything
        seams = {}

        # Optional seam JSON override
        if args.seam_json is not None:
            try:
                seam_cfg = json.loads(args.seam_json.read_text())
                # Expect keys like "rough:q0:q1" mapping to list of [i,j]
                for key, pairs in seam_cfg.items():
                    try:
                        kind, a, b = key.split(":", 2)
                    except Exception:
                        continue
                    seams[(kind, a, b)] = [tuple(map(int, p)) for p in pairs]
            except Exception as _exc:
                pass

        bracket_map = {f"q{i}": bracket_basis for i in range(qc.num_qubits)}
        layout, ops = compile_circuit_to_surgery(
            qc, patches, seams, distance=d, bracket_map=bracket_map, 
            warmup_rounds=warmup_rounds, ancilla_strategy=args.cnot_ancilla_strategy, ancilla_buffer=args.ancilla_buffer
        )

        layout.plot(
            annotate=False,         # set True to label global qubit indices
            seams=True,             # draw rough/smooth seam edges
            title=f"Layout for {args.benchmark} (d={d})",
            save_path=PROJECT_ROOT / f"plots/layout_{args.benchmark}_d{d}.png",
        )

        if args.verbose:
            print("ops")
            print(ops)

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

        gb = GlobalStimBuilder(layout)
        circuit, observable_pairs, metadata = gb.build(ops, stim_cfg, bracket_map, qc)

        if args.verbose:
            try:
                print("\n[DEBUG] Stim diagram (detslice-with-ops):")
                print(circuit.diagram('detslice-with-ops', tick=range(0, circuit.num_ticks)))
            except Exception:
                pass

        # Sample DEM and decode
        dem = circuit.detector_error_model()

        # --- NEW: add boundary anchors from builder metadata (if any) ---
        try:
            ba = (metadata.get("boundary_anchors", {}) or {})
            anchor_ids = list(ba.get("detector_ids", []) or [])
            anchor_eps = float(ba.get("epsilon", 1e-12))
            if args.verbose:
                print(f"[DEM-CHECK] attempting anchor augmentation: ids={len(anchor_ids)}, eps={anchor_eps:g}")
            dem = augment_dem_with_boundary_anchors(dem, anchor_ids, anchor_eps)
        except Exception as _exc:
            if args.verbose:
                print(f"[DEM-CHECK] anchor augmentation skipped due to error: {_exc}")

        # Ensure every connected component has a singleton for pairwise MWPM
        dem, added_pairwise_hooks = harden_dem_for_pairwise_matching(dem, epsilon=1e-12)
        if args.verbose:
            print(f"[DEM-CHECK] added {added_pairwise_hooks} pairwise boundary hooks and rebuilt DEM via text")

        dem_sampler = dem.compile_sampler(seed=int(args.seed))
        det_samp, obs_samp, _ = dem_sampler.sample(int(args.shots))
        obs_u8 = np.asarray(obs_samp, dtype=np.uint8) if obs_samp is not None and obs_samp.size > 0 else np.zeros((int(args.shots), len(observable_pairs)), dtype=np.uint8)

        try:
            # Pairwise matching (no correlations; requires true single-detector boundaries)
            matcher = pm.Matching.from_detector_error_model(dem)
            dem, matcher, iso_added, iso_ids = _anchor_pm_isolates(dem, matcher, epsilon=1e-12)
            if iso_added and args.verbose:
                print(f"[DECODE] anchored {iso_added} isolated detectors: {iso_ids[:12]}")
            preds = matcher.decode_batch(det_samp.astype(bool))
        except Exception as exc_mwpm:
            if args.verbose:
                print("[DECODE] Pairwise MWPM failed; hardening DEM and retrying once:", repr(exc_mwpm))
            try:
                # Harden again (in case the sampler revealed an odd-parity component)
                dem, added_retry = harden_dem_for_pairwise_matching(dem, epsilon=1e-12)
                if args.verbose:
                    print(f"[DEM-CHECK] added {added_retry} additional pairwise boundary hooks on retry and rebuilt DEM via text")
                matcher = pm.Matching.from_detector_error_model(dem)
                dem, matcher, iso_added_retry, iso_ids_retry = _anchor_pm_isolates(dem, matcher, epsilon=1e-12)
                if iso_added_retry and args.verbose:
                    print(f"[DECODE] anchored {iso_added_retry} isolated detectors on retry: {iso_ids_retry[:12]}")
                preds = matcher.decode_batch(det_samp.astype(bool))
            except Exception as exc_retry:
                # Keep existing diagnostics only if decoding still fails after retry
                print("\n[ERROR] MWPM decode failed; printing mwpm_debug summary:")
                dbg = metadata.get("mwpm_debug", {}) or {}
                seam_wraps = dbg.get("seam_wrap_counts", {})
                row_wraps = dbg.get("row_wraps", {})
                deg_viol = dbg.get("degree_violations", [])
                odd_details = dbg.get("odd_degree_details", {}) or {}
                edge_records_count = dbg.get("edge_records_count", 0)
                print("[DEBUG] seam wraps:")
                for k, v in seam_wraps.items():
                    print(f"  {k} -> {v}")
                print("[DEBUG] Z row wraps:")
                for q, rows in (row_wraps.get("Z", {}) or {}).items():
                    print(f"  {q}: {rows}")
                print("[DEBUG] X row wraps:")
                for q, rows in (row_wraps.get("X", {}) or {}).items():
                    print(f"  {q}: {rows}")
                print(f"[DEBUG] degree violations (abs meas idx with degree!=2): {deg_viol[:200]}")
                if deg_viol:
                    from collections import Counter
                    tag_counter = Counter()
                    for idx in deg_viol[:200]:
                        for rec in odd_details.get(idx, [])[:8]:
                            tag_counter.update([str(rec.get('tag'))])
                    print(f"[DEBUG] odd-degree provenance tags (top): {tag_counter.most_common(12)}")
                    shown = 0
                    for idx in deg_viol[:50]:
                        recs = odd_details.get(idx, [])
                        if not recs:
                            continue
                        print(f"  [ODD] idx={idx}, examples:")
                        for rec in recs[:3]:
                            print(f"    tag={rec.get('tag')}, neighbor={rec.get('neighbor')}, ctx={rec.get('context')}")
                        shown += 1
                        if shown >= 6:
                            break
                    print(f"[DEBUG] total edge records captured: {edge_records_count}")

                # DEM component-level diagnostics and odd-parity boundaryless scan
                try:
                    print("\n[DEM-CHECK] analyzing DEM components & boundaries...")
                    _report_boundaryless_components(dem)
                    _scan_boundaryless_odd_shot(dem, det_samp, max_scan=min(2048, int(args.shots)))
                except Exception as _exc:
                    print(f"[DEM-CHECK] diagnostics failed: {_exc}")
                # PyMatching graph-level parity check on the actual matching graph
                try:
                    if 'matcher' in locals():
                        print("\n[PM-CHECK] analyzing PyMatching graph components & boundaries...")
                        _pm_find_offending_shot(matcher, det_samp, max_scan=min(2048, int(args.shots)))
                    else:
                        print("[PM-CHECK] matcher unavailable; skipping")
                except Exception as _exc3:
                    print(f"[PM-CHECK] diagnostics failed: {_exc3}")
                raise
        preds = np.asarray(preds, dtype=np.uint8)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        # Sample raw measurements for merge byproduct extraction
        circ_sampler = circuit.compile_sampler(seed=int(args.seed))
        m_samples = circ_sampler.sample(shots=int(args.shots))

        # Extract CNOT parity bits directly from single-shot MPPs and update Pauli frame
        pfm = PauliTracker(qc.num_qubits)
        # Initialize frame bits with correct shots dimension
        shots_count = int(args.shots)
        for i in range(qc.num_qubits):
            qname = f"q{i}"
            pfm.frame[qname]["fx"] = np.zeros(shots_count, dtype=np.uint8)
            pfm.frame[qname]["fz"] = np.zeros(shots_count, dtype=np.uint8)
        cnot_metadata = []
        
        # Determine which byproducts to enable for this run
        run_bases = set()
        snap_basis = (metadata.get("final_snapshot", {}) or {}).get("basis", None)
        if isinstance(snap_basis, str):
            run_bases.add(snap_basis.upper())
        arg_demo = (args.demo_basis or "Z").strip().upper()
        if arg_demo in ("Z", "X"):
            run_bases.add(arg_demo)
        # Enable m_ZZ only if Z is present; enable m_XX only if X is present
        enable_mzz = ("Z" in run_bases)
        enable_mxx = ("X" in run_bases)

        for cnot_op in metadata.get("cnot_operations", []):
            control = cnot_op["control"]
            target = cnot_op["target"]
            ancilla = cnot_op["ancilla"]
            
            # Extract m_ZZ and m_XX directly from single-shot MPP indices
            m_zz_mpp_idx = cnot_op.get("m_zz_mpp_idx")
            m_xx_mpp_idx = cnot_op.get("m_xx_mpp_idx")
            
            m_zz = m_samples[:, m_zz_mpp_idx] if (enable_mzz and m_zz_mpp_idx is not None) else np.zeros(int(args.shots), dtype=np.uint8)
            m_xx = m_samples[:, m_xx_mpp_idx] if (enable_mxx and m_xx_mpp_idx is not None) else np.zeros(int(args.shots), dtype=np.uint8)
            
            # Update Pauli frame: fz[target] ^= m_ZZ, fx[control] ^= m_XX
            pfm.update_cnot(control, target, m_zz, m_xx)
            
            # Store CNOT metadata for reporting
            cnot_metadata.append({
                "control": control,
                "target": target,
                "ancilla": ancilla,
                "m_zz_mean": float(m_zz.mean()) if enable_mzz else 0.0,
                "m_xx_mean": float(m_xx.mean()) if enable_mxx else 0.0,
            })
        
        # Apply Pauli frame and decoder corrections to logical observables
        corrected_obs = obs_u8.copy()
        
        # Map patch names to their observable indices
        patch_to_obs_idx = {}
        for i, patch_name in enumerate(sorted(bracket_map.keys())):
            patch_to_obs_idx[patch_name] = i
        
        # Apply Pauli frame corrections
        for patch_name in sorted(bracket_map.keys()):
            if patch_name in patch_to_obs_idx:
                obs_idx = patch_to_obs_idx[patch_name]
                # Check if obs_idx is within bounds
                if obs_idx < corrected_obs.shape[1]:
                    basis = bracket_map[patch_name]
                    if basis == "Z":
                        corrected_obs[:, obs_idx] ^= pfm.frame[patch_name]["fz"]
                    else:
                        corrected_obs[:, obs_idx] ^= pfm.frame[patch_name]["fx"]
        
        # Apply decoder predictions to flip outcomes when decoder detects errors
        # The decoder predictions indicate when the logical outcome should be flipped
        corrected_obs ^= preds

        # Get basis labels
        basis_labels = tuple(metadata.get("observable_basis", tuple()))
        if not basis_labels or len(basis_labels) != obs_u8.shape[1]:
            basis_labels = tuple(bracket_map[q] for q in sorted(bracket_map))
        
        # Ensure basis_labels matches the actual number of observable columns
        if len(basis_labels) > obs_u8.shape[1]:
            basis_labels = basis_labels[:obs_u8.shape[1]]

        # Track virtual single-qubit gates centrally
        for qname, gates in sequence_from_qc(qc).items():
            qidx = int(qname[1:])
            if gates:
                pfm.set_sequence(qidx, gates)
        
        # Derive per-qubit expected flips for debug/verbose via centralized helper
        expected_flips = []
        for i in range(qc.num_qubits):
            seq = pfm.virtual_gates[f"q{i}"]
            _, phase = PauliTracker.conjugate_axis_by_sequence(bracket_basis, seq)
            expected_flips.append(1 if phase < 0 else 0)

        # Extract demo readouts for physics analysis
        demo_z_bits = {}
        demo_x_bits = {}
        demo_meta = metadata.get("demo", {})

        # DEBUG: print tail of circuit operations to ensure joint MPPs are last
        if args.verbose:
            try:
                tail_ops = str(circuit).strip().splitlines()[-80:]
                print("\n[DEBUG] Tail of Stim circuit (last ~80 ops):")
                for ln in tail_ops:
                    print("  ", ln)
            except Exception:
                pass

        # Compile sampler once for both singles and joint demos
        # (circ_sampler and m_samples already created above)

        # Singles (if present)
        if demo_meta:
            for name in sorted(demo_meta.keys()):
                info = demo_meta[name]
                idx = int(info.get("index")) if info.get("index") is not None else None
                b = info.get("basis")
                patch_name = info.get("patch")
                if idx is None or patch_name is None:
                    continue
                col = np.asarray(m_samples[:, idx], dtype=np.uint8)
                if b == "Z":
                    demo_z_bits[patch_name] = col
                elif b == "X":
                    demo_x_bits[patch_name] = col

        # Extract joint demo readouts for correlations
        joint_demo_bits = {}
        joint_demo_meta = metadata.get("joint_demos", {})

        if joint_demo_meta:
            for joint_key in sorted(joint_demo_meta.keys()):
                joint_info = joint_demo_meta[joint_key]
                idx = int(joint_info.get("index")) if joint_info.get("index") is not None else None
                basis = joint_info.get("basis")
                pair = joint_info.get("pair")
                if idx is None or basis is None or pair is None:
                    continue
                col = np.asarray(m_samples[:, idx], dtype=np.uint8)
                # DEBUG: print direct raw mean for this column
                print(f"[DEBUG] Joint {joint_key}: idx={idx}, raw_mean={float(col.mean()):.4f}, phys={joint_info.get('physical_realization','')}")
                joint_demo_bits[joint_key] = {
                    "bits": col,
                    "basis": basis,
                    "pair": pair,
                    "logical_operator": joint_info.get("logical_operator", "unknown"),
                    "physical_realization": joint_info.get("physical_realization", ""),
                    "final_bases": joint_info.get("final_bases", {}),
                    "axes": joint_info.get("axes", {}),
                }

        # Apply Pauli-frame corrections to joint demo bits (flip parity by fx/fz as needed)
        if joint_demo_bits and pfm.frame:
            corrected_joint_demo_bits: Dict[str, Dict[str, object]] = {}
            for joint_key, demo_data in joint_demo_bits.items():
                bits = demo_data["bits"]
                basis = demo_data["basis"]
                pair = demo_data["pair"]
                axes_map = demo_data.get("axes", {})
                final_bases = demo_data.get("final_bases", {})

                # Prefer axes_map from conjugated operator; fallback to final_bases
                flips = np.zeros_like(bits, dtype=np.uint8)
                for qubit_name in pair:
                    frame = pfm.frame.get(qubit_name)
                    if frame is None:
                        continue
                    axes = axes_map.get(qubit_name)
                    if not axes:
                        fb = final_bases.get(qubit_name, basis)
                        axes = [fb]
                    partial = np.zeros_like(bits, dtype=np.uint8)
                    for ax in axes:
                        axis_key = "fz" if ax == "Z" else "fx"
                        raw_val = frame.get(axis_key, 0)
                        if isinstance(raw_val, np.ndarray):
                            partial ^= raw_val.astype(np.uint8)
                        else:
                            if int(raw_val) & 1:
                                partial ^= np.ones_like(bits, dtype=np.uint8)
                    flips ^= partial

                corrected_bits = np.bitwise_xor(bits, flips)
                new_entry = dict(demo_data)
                new_entry["raw_bits"] = bits
                new_entry["bits"] = corrected_bits
                new_entry["frame_flip"] = flips
                corrected_joint_demo_bits[joint_key] = new_entry
            joint_demo_bits = corrected_joint_demo_bits
        
        # Extract final snapshot bits (if present)
        snapshot_bits = {}
        snapshot_meta = metadata.get("final_snapshot", {})
        if snapshot_meta.get("enabled"):
            order = snapshot_meta["order"]
            indices = snapshot_meta["indices"]
            for qubit_name, idx in zip(order, indices):
                snapshot_bits[qubit_name] = np.asarray(m_samples[:, idx], dtype=np.uint8)
        
        correlation_pairs = []
        for cnot_op in metadata.get("cnot_operations", []):
            control = cnot_op["control"]
            target = cnot_op["target"]
            correlation_pairs.append((control, target))
        
        # Add custom correlation pairs if specified
        if args.corr_pairs:
            try:
                custom_pairs = args.corr_pairs.split(';')
                for pair_str in custom_pairs:
                    if ',' in pair_str:
                        q1, q2 = pair_str.strip().split(',', 1)
                        correlation_pairs.append((q1.strip(), q2.strip()))
            except Exception:
                pass  # Ignore malformed correlation pairs

        # Compute per-qubit LER with Wilson CI
        per_qubit_ler = []
        per_qubit_ler_ci = []
        for i in range(min(len(basis_labels), obs_u8.shape[1])):
            errors = np.bitwise_xor(obs_u8[:, i], preds[:, i])
            error_count = int(np.sum(errors))
            ler = error_count / int(args.shots)
            ler_ci = wilson_rate_ci(error_count, int(args.shots))
            per_qubit_ler.append(ler)
            per_qubit_ler_ci.append(ler_ci)

        # Compute two-qubit correlations using joint demos
        # Compute joint correlations with byproduct corrections
        correlations = {}
        if joint_demo_bits:
            # Apply byproduct corrections to joint demo bits before computing correlations
            corrected_joint_demo_bits = {}
            for joint_key, demo_data in joint_demo_bits.items():
                corrected_data = demo_data.copy()
                pair = demo_data["pair"]
                basis = demo_data["basis"]
                
                # Find corresponding CNOT operation for this pair
                byproduct_correction = np.zeros(int(args.shots), dtype=np.uint8)
                for cnot_op in metadata.get("cnot_operations", []):
                    control = cnot_op["control"]
                    target = cnot_op["target"]
                    
                    # Check if this joint measurement involves the CNOT control-target pair
                    if ((pair[0] == control and pair[1] == target) or 
                        (pair[0] == target and pair[1] == control)):
                        
                        if basis == "Z":
                            # For ZZ correlations, flip by m_ZZ byproduct
                            m_zz_mpp_idx = cnot_op.get("m_zz_mpp_idx")
                            if m_zz_mpp_idx is not None:
                                byproduct_correction = m_samples[:, m_zz_mpp_idx]
                                print(f"[DEBUG] Joint {joint_key}: applying m_ZZ correction from CNOT({control}->{target}), mean={byproduct_correction.mean():.3f}")
                        elif basis == "X":
                            # For XX correlations, flip by m_XX byproduct
                            m_xx_mpp_idx = cnot_op.get("m_xx_mpp_idx")
                            if m_xx_mpp_idx is not None:
                                byproduct_correction = m_samples[:, m_xx_mpp_idx]
                                print(f"[DEBUG] Joint {joint_key}: applying m_XX correction from CNOT({control}->{target}), mean={byproduct_correction.mean():.3f}")
                        break
                
                # Apply byproduct correction: flip bits where byproduct is 1
                corrected_bits = demo_data["bits"] ^ byproduct_correction
                corrected_data["bits"] = corrected_bits
                corrected_data["raw_bits"] = demo_data["bits"]  # Keep original for comparison
                corrected_joint_demo_bits[joint_key] = corrected_data
            
            correlations = compute_joint_correlations(corrected_joint_demo_bits, int(args.shots))
        elif correlation_pairs and demo_z_bits and demo_x_bits:
            # Fallback to old method if no joint demos available
            correlations = compute_two_qubit_correlations(demo_z_bits, demo_x_bits, correlation_pairs, int(args.shots))

        # Print structured report
        print_header(args, model, dem, metadata, {}, cnot_metadata, stim_rounds, int(args.shots))
        if args.verbose:
            dbg = metadata.get("mwpm_debug", {}) or {}
            seam_wraps = dbg.get("seam_wrap_counts", {})
            row_wraps = dbg.get("row_wraps", {})
            if seam_wraps:
                print("[DEBUG] seam wraps:")
                for k, v in seam_wraps.items():
                    print(f"  {k} -> {v}")
            if row_wraps:
                print("[DEBUG] Z row wraps:")
                for q, rows in (row_wraps.get("Z", {}) or {}).items():
                    print(f"  {q}: {rows}")
                print("[DEBUG] X row wraps:")
                for q, rows in (row_wraps.get("X", {}) or {}).items():
                    print(f"  {q}: {rows}")
        # ---- Clarifying preamble (requested vs emitted; operator semantics) ----
        _print_demo_preamble(metadata, stim_cfg)
    
        print_per_qubit_results(args, bracket_map, corrected_obs, obs_u8, preds, int(args.shots))
        print_physics_demo(
            demo_meta,
            demo_z_bits,
            demo_x_bits,
            correlation_pairs,
            int(args.shots),
            pfm.frame,
            joint_demo_bits,
            virtual_gates_per_qubit=pfm.virtual_gates,
        )
        try:
            decoded_order = [name for name in sorted(bracket_map.keys()) if name in bracket_map]
            print_decoded_logical_distribution(decoded_order, corrected_obs, int(args.shots))
        except Exception as _exc_dist:
            if args.verbose:
                print(f"[DEBUG] decoded logical distribution unavailable: {_exc_dist}")
        decoder_flip_map = {}
        for patch_name, obs_idx in patch_to_obs_idx.items():
            if obs_idx < preds.shape[1]:
                decoder_flip_map[patch_name] = preds[:, obs_idx]

        print_final_state_distribution(
            metadata.get("final_snapshot", {}),
            snapshot_bits,
            pfm.frame,
            int(args.shots),
            apply_frame_correction=True,
            decoder_flips=decoder_flip_map,
        )
        try:
            print_schrodinger_snapshot_distribution(
                metadata.get("final_snapshot", {}),
                snapshot_bits,
            )
        except Exception as _exc_sz:
            if args.verbose:
                print(f"[DEBUG] Schrödinger Z snapshot derivation failed: {_exc_sz}")
        print_pauli_frame_audit(pfm.virtual_gates, pfm.frame, cnot_metadata)
        
        if args.verbose:
            print_debug_details(args, basis_labels, obs_u8, preds, corrected_obs, expected_flips)

        # Generate and save detailed JSON report
        # Add snapshot metadata to args for JSON generation
        args.snapshot_meta = metadata.get("final_snapshot", {})

        if args.dump_json:
            detailed_json = generate_detailed_json(
                args, model, metadata, cnot_metadata,
                bracket_map, corrected_obs, obs_u8, preds,
                demo_z_bits, demo_x_bits, correlations,
                pfm.virtual_gates, pfm.frame, int(args.shots), stim_rounds,
                snapshot_bits
            )
            json_filepath = save_detailed_json(detailed_json, args)
            print(f"\nDetailed report saved to: {json_filepath}")


    if run_transpile:
        benchmark = instantiate_benchmark(args.benchmark)
        logical_metrics = benchmark.compute_logical_metrics()

        target = load_target(args.target)
        config = build_config(target, args)
        transpiler = HeavyHexTranspiler(config)

        logical_circuit = benchmark.get_circuit()
        _best, best_metrics, leaderboard = transpiler.run_baseline(logical_circuit)

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
