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
import numpy as np
import pymatching as pm
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
    generate_detailed_json,
    save_detailed_json,
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
        default="bell", # options are bell, ghz3, parity_check, teleportation, simple_1q_xzh
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
    parser.add_argument("--px", type=float, default=0, help="Phenomenological X error probability")
    parser.add_argument("--pz", type=float, default=0, help="Phenomenological Z error probability")
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
        choices=["auto", "Z", "X", "both", "none"],
        default="Z",
        help="End-only demo readout basis for reporting. 'auto' uses logical end basis when it differs from the bracket; 'both' enables both Z and X measurements; 'none' disables the demo readout.",
    )
    parser.add_argument("--shots", type=int, default=10**6, help="Number of Monte Carlo samples")
    parser.add_argument("--seed", type=int, default=46, help="Seed for Stim samplers")
    parser.add_argument(
        "--joint-only",
        action="store_true",
        help="Emit only joint product demos at end of circuit (skip singles)",
    )
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
        start_basis, init_sign = parse_init_label(init_label)
        gate_seq = [ci.operation.name.upper() for ci in qc.data]
        # Global end-basis heuristic for demo auto: flip if any H is present
        any_h = any(ci.operation.name.lower() == "h" for ci in qc.data)
        end_basis = ("X" if start_basis == "Z" else "Z") if any_h else start_basis
        expected_flip = 0
        # Incorporate the initial eigen-sign into the expected correlation parity.
        # A '-1' eigenstate contributes a unit flip to the start-vs-end parity definition.
        expected_flip_total = int(expected_flip) ^ (0 if init_sign == +1 else 1)

        # Unified surgery-based path (works for any number of qubits)
        model = build_heavy_hex_model(args.distance)

        plot_heavy_hex_code(model, args.distance)
        diagnostic_print(model, args)

        d = int(args.distance)
        stim_rounds = int(args.rounds) if args.rounds is not None else d

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
        elif db_mode == "both":
            demo_basis = ["Z", "X"]
        else:
            # Handle single basis (Z or X) or comma-separated formats like "Z,X"
            demo_str = args.demo_basis.strip().upper()
            if "," in demo_str:
                # Parse comma-separated list and normalize to ["Z", "X"]
                bases = [b.strip() for b in demo_str.split(",")]
                if set(bases) == {"Z", "X"}:
                    demo_basis = ["Z", "X"]
                else:
                    demo_basis = demo_str  # Fallback to single string
            else:
                demo_basis = demo_str

        # Build one PatchObject per qubit
        def build_patch() -> PatchObject:
            return PatchObject(
                n=model.code.n,
                z_stabs=model.z_stabilizers,
                x_stabs=model.x_stabilizers,
                logical_z=model.logical_z,
                logical_x=model.logical_x,
                coords={i: (float(i), 0.0) for i in range(model.code.n)},
            )

        patches = {f"q{i}": build_patch() for i in range(qc.num_qubits)}

        # Default seams: for each CNOT pair, use (i,i) for i in [0..d-1]
        default_pairs = [(i, i) for i in range(d)]
        seams = {}
        for ci in qc.data:
            name = ci.operation.name.lower()
            if name in {"cx", "cz", "cnot"}:
                qa, qb = ci.qubits[0], ci.qubits[1]
                a = f"q{qc.find_bit(qa).index}"
                b = f"q{qc.find_bit(qb).index}"
                seams[("rough", a, b)] = list(default_pairs)
                seams[("smooth", a, b)] = list(default_pairs)

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
            warmup_rounds=1, ancilla_strategy=args.cnot_ancilla_strategy
        )

        stim_cfg = PhenomenologicalStimConfig(
            rounds=stim_rounds,
            p_x_error=float(args.px),
            p_z_error=float(args.pz),
            init_label=init_label,
            logical_start=None,
            logical_end=None,
            bracket_basis=bracket_basis,
            demo_basis=demo_basis,
            demo_joint_only=bool(args.joint_only),
        )

        gb = GlobalStimBuilder(layout)
        circuit, observable_pairs, metadata = gb.build(ops, stim_cfg, bracket_map, qc)

        # Sample DEM and decode
        dem = circuit.detector_error_model()
        matcher = pm.Matching.from_detector_error_model(dem)
        dem_sampler = dem.compile_sampler(seed=int(args.seed))
        det_samp, obs_samp, _ = dem_sampler.sample(int(args.shots))
        det_samp_u8 = np.asarray(det_samp, dtype=np.uint8)
        obs_u8 = np.asarray(obs_samp, dtype=np.uint8) if obs_samp is not None and obs_samp.size > 0 else np.zeros((int(args.shots), len(observable_pairs)), dtype=np.uint8)
        preds = matcher.decode_batch(det_samp.astype(bool))
        preds = np.asarray(preds, dtype=np.uint8)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        # Per-window parity bits via temporal MWPM (1D chain XOR)
        merge_bits = {}
        for window in metadata.get("merge_windows", []):
            bit = decode_joint_parity(det_samp_u8, window)
            key = (window.get("parity_type"), window.get("a"), window.get("b"), window.get("id"))
            merge_bits[key] = bit

        # Extract CNOT parity bits and update Pauli frame
        pauli_frame = {f"q{i}": {"fx": 0, "fz": 0} for i in range(qc.num_qubits)}
        cnot_metadata = []
        
        for cnot_op in metadata.get("cnot_operations", []):
            control = cnot_op["control"]
            target = cnot_op["target"]
            ancilla = cnot_op["ancilla"]
            rough_window_id = cnot_op["rough_window_id"]
            smooth_window_id = cnot_op["smooth_window_id"]
            
            # Extract m_ZZ from rough merge window (Control-Ancilla)
            rough_key = ("ZZ", control, ancilla, rough_window_id)
            m_zz = merge_bits.get(rough_key, np.zeros(int(args.shots), dtype=np.uint8))
            
            # Extract m_XX from smooth merge window (Ancilla-Target)  
            smooth_key = ("XX", ancilla, target, smooth_window_id)
            m_xx = merge_bits.get(smooth_key, np.zeros(int(args.shots), dtype=np.uint8))
            
            # Update Pauli frame: fz[target] ^= m_ZZ, fx[control] ^= m_XX
            pauli_frame[target]["fz"] ^= m_zz
            pauli_frame[control]["fx"] ^= m_xx
            
            # Store CNOT metadata for reporting
            cnot_metadata.append({
                "control": control,
                "target": target,
                "ancilla": ancilla,
                "m_zz_mean": float(m_zz.mean()),
                "m_xx_mean": float(m_xx.mean()),
            })
        
        # Apply Pauli frame and decoder corrections to logical observables
        corrected_obs = obs_u8.copy()
        
        # Map patch names to their observable indices
        patch_to_obs_idx = {}
        for i, patch_name in enumerate(sorted(bracket_map.keys())):
            patch_to_obs_idx[patch_name] = i
        
        # Apply Pauli frame corrections
        for patch_name, frame in pauli_frame.items():
            if patch_name in patch_to_obs_idx:
                obs_idx = patch_to_obs_idx[patch_name]
                # Check if obs_idx is within bounds
                if obs_idx < corrected_obs.shape[1]:
                    basis = bracket_map[patch_name]
                    if basis == "Z":
                        # For Z-basis bracketing, apply fz frame
                        corrected_obs[:, obs_idx] ^= frame["fz"]
                    else:  # X-basis bracketing
                        # For X-basis bracketing, apply fx frame  
                        corrected_obs[:, obs_idx] ^= frame["fx"]
        
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

        # Track virtual single-qubit gates (X,Z,H) to compute expected flips per qubit
        gate_map = {f"q{i}": [] for i in range(qc.num_qubits)}
        for ci in qc.data:
            name = ci.operation.name.lower()
            if name in {"x", "z", "h"}:
                for qb in ci.qubits:
                    qidx = qc.find_bit(qb).index
                    gate_map[f"q{qidx}"].append(name.upper())
        
        # Derive per-qubit expected flips in the chosen bracket basis
        import surface_code.logical_ops as lo
        expected_flips = []
        for i in range(qc.num_qubits):
            seq = gate_map[f"q{i}"]
            frame = lo.apply_sequence(lo.LogicalFrame(), seq)
            end_b, flip = lo.end_basis_and_flip(bracket_basis, frame)
            expected_flips.append(int(flip))

        # Extract demo readouts for physics analysis
        demo_z_bits = {}
        demo_x_bits = {}
        demo_meta = metadata.get("demo", {})

        # DEBUG: print tail of circuit operations to ensure joint MPPs are last
        try:
            tail_ops = str(circuit).strip().splitlines()[-80:]
            print("\n[DEBUG] Tail of Stim circuit (last ~80 ops):")
            for ln in tail_ops:
                print("  ", ln)
        except Exception:
            pass

        # Compile sampler once for both singles and joint demos
        circ_sampler = circuit.compile_sampler(seed=int(args.seed))
        m_samples = circ_sampler.sample(shots=int(args.shots))

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
        if joint_demo_bits and pauli_frame:
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
                    frame = pauli_frame.get(qubit_name)
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
        correlations = {}
        if joint_demo_bits:
            correlations = compute_joint_correlations(joint_demo_bits, int(args.shots))
        elif correlation_pairs and demo_z_bits and demo_x_bits:
            # Fallback to old method if no joint demos available
            correlations = compute_two_qubit_correlations(demo_z_bits, demo_x_bits, correlation_pairs, int(args.shots))

        # Print structured report
        print_header(args, model, dem, metadata, merge_bits, cnot_metadata, stim_rounds, int(args.shots))
        print_per_qubit_results(args, bracket_map, corrected_obs, obs_u8, preds, int(args.shots))
        print_physics_demo(demo_meta, demo_z_bits, demo_x_bits, correlation_pairs, int(args.shots), pauli_frame, joint_demo_bits)
        print_pauli_frame_audit(gate_map, pauli_frame, cnot_metadata)
        
        if args.verbose:
            print_debug_details(args, basis_labels, obs_u8, preds, corrected_obs, expected_flips)

        # Generate and save detailed JSON report
        detailed_json = generate_detailed_json(
            args, model, metadata, merge_bits, cnot_metadata,
            bracket_map, corrected_obs, obs_u8, preds,
            demo_z_bits, demo_x_bits, correlations,
            gate_map, pauli_frame, int(args.shots), stim_rounds
        )
        
        json_filepath = save_detailed_json(detailed_json, args)
        print(f"\nDetailed report saved to: {json_filepath}")

        # Keep existing JSON dump for backward compatibility
        json_payloads.append(
            {
                "branch": "surgery",
                "benchmark": args.benchmark,
                "sequence": gate_seq,
                "num_qubits": int(qc.num_qubits),
                "distance": int(d),
                "rounds": int(stim_rounds),
                "p_x": float(args.px),
                "p_z": float(args.pz),
                "shots": int(args.shots),
                "detectors": int(dem.num_detectors),
                "observable_basis": list(basis_labels),
                "per_qubit_raw_mean": [float(obs_u8[:, i].mean()) for i in range(obs_u8.shape[1])],
                "per_qubit_decoder_mean": [float(preds[:, i].mean()) for i in range(preds.shape[1])],
                "per_qubit_corrected_mean": [float(corrected_obs[:, i].mean()) for i in range(corrected_obs.shape[1])],
                "cnot_operations": cnot_metadata,
                "merge_windows": [
                    {
                        "id": int(w.get("id")),
                        "type": w.get("type"),
                        "parity_type": w.get("parity_type"),
                        "a": w.get("a"),
                        "b": w.get("b"),
                        "rounds": int(w.get("rounds", 0)),
                        "mean": float(merge_bits[(w.get("parity_type"), w.get("a"), w.get("b"), w.get("id"))].mean()) if (w.get("parity_type"), w.get("a"), w.get("b"), w.get("id")) in merge_bits else 0.0,
                    }
                    for w in metadata.get("merge_windows", [])
                ],
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
