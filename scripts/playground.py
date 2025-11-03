"""Quick playground for the heavy-hex transpilation pipeline.

This script stays intentionally thin: it wires together the high-level
components already defined in the project so you can iterate on new QEC
experiments without re-implementing logic. Extend it as you develop new
benchmarks, targets, or analysis steps.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Dict, List

# Add the project root to Python path to enable src imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure the src/ directory is available for imports when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import CLI utilities
from cli import parse_args, instantiate_benchmark, load_target, build_config, format_leaderboard

# Import simulation module
from surface_code.simulation import run_logical_simulation

from benchmarks.BenchmarkCircuit import BenchmarkCircuit
from surface_code.utils import (
    plot_heavy_hex_code,
    diagnostic_print,
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

from transpile.pipeline import HeavyHexTranspiler
from surface_code import (
    build_surface_code_model,
    PhenomenologicalStimConfig,
)
from surface_code.layout import PatchObject
from surface_code.surgery_compile import compile_circuit_to_surgery
from surface_code.builder import GlobalStimBuilder

from surface_code.pauli import PauliTracker, parse_init_label, sequence_from_qc


def main() -> None:
    args = parse_args()

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

        # Global end-basis heuristic for demo auto: toggle basis if any H is present
        any_h = any(ci.operation.name.lower() == "h" for ci in qc.data)
        end_basis = ("X" if start_basis == "Z" else "Z") if any_h else start_basis

        # Unified surgery-based path (works for any number of qubits)
        model = build_surface_code_model(args.distance, code_type=args.code_type)

        plot_heavy_hex_code(model, args.distance, code_type=args.code_type)
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

        # Build a template PatchObject to infer geometry/spacing
        template_patch: PatchObject = PatchObject.from_code_model(model)

        # Compute horizontal spacing based on the actual patch width.
        # Using a fixed offset (e.g., 3.0) can cause overlap at larger d.
        if template_patch.coords:
            xs = [x for (x, _y) in template_patch.coords.values()]
            patch_width = (max(xs) - min(xs)) if xs else 0.0
        else:
            patch_width = 0.0

        # Add a small margin so patches don't touch; reuse ancilla_buffer as a sensible margin
        h_margin = float(args.ancilla_buffer) if hasattr(args, "ancilla_buffer") else 1.0
        h_spacing = patch_width + max(0.5, h_margin)

        # Create patches with geometry-aware horizontal offsets to prevent overlap
        patches = {}
        for i in range(qc.num_qubits):
            # Fresh patch derived from the same template geometry
            patch = template_patch.with_offset(0, i * h_spacing, 0.0)
            patches[f"q{i}"] = patch

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
            p_meas=float(args.p_meas),
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

        # Run simulation
        results = run_logical_simulation(
            circuit=circuit,
            dem=dem,
            metadata=metadata,
            observable_pairs=observable_pairs,
            bracket_map=bracket_map,
            qc=qc,
            shots=int(args.shots),
            seed=int(args.seed),
            demo_basis=demo_basis,
            bracket_basis=bracket_basis,
            corr_pairs=args.corr_pairs,
            verbose=args.verbose,
        )

        # Print structured report
        print_header(args, model, results.dem, metadata, {}, results.cnot_metadata, stim_rounds, int(args.shots))
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
    
        print_per_qubit_results(args, bracket_map, results.corrected_obs, results.obs_u8, results.preds, int(args.shots))
        print_physics_demo(
            results.demo_meta,
            results.demo_z_bits,
            results.demo_x_bits,
            results.correlation_pairs,
            int(args.shots),
            results.pauli_tracker.frame,
            results.joint_demo_bits,
            virtual_gates_per_qubit=results.pauli_tracker.virtual_gates,
        )
        try:
            decoded_order = [name for name in sorted(bracket_map.keys()) if name in bracket_map]
            print_decoded_logical_distribution(decoded_order, results.corrected_obs, int(args.shots))
        except Exception as _exc_dist:
            if args.verbose:
                print(f"[DEBUG] decoded logical distribution unavailable: {_exc_dist}")
        decoder_flip_map = {}
        for patch_name, obs_idx in results.patch_to_obs_idx.items():
            if obs_idx < results.preds.shape[1]:
                decoder_flip_map[patch_name] = results.preds[:, obs_idx]

        print_final_state_distribution(
            metadata.get("final_snapshot", {}),
            results.snapshot_bits,
            results.pauli_tracker.frame,
            int(args.shots),
            apply_frame_correction=True,
            decoder_flips=decoder_flip_map,
        )
        try:
            print_schrodinger_snapshot_distribution(
                metadata.get("final_snapshot", {}),
                results.snapshot_bits,
            )
        except Exception as _exc_sz:
            if args.verbose:
                print(f"[DEBUG] Schrödinger Z snapshot derivation failed: {_exc_sz}")
        print_pauli_frame_audit(results.pauli_tracker.virtual_gates, results.pauli_tracker.frame, results.cnot_metadata)
        
        if args.verbose:
            print_debug_details(args, results.basis_labels, results.obs_u8, results.preds, results.corrected_obs, results.expected_flips)

        # Generate and save detailed JSON report
        # Add snapshot metadata to args for JSON generation
        args.snapshot_meta = metadata.get("final_snapshot", {})

        if args.dump_json:
            detailed_json = generate_detailed_json(
                args, model, metadata, results.cnot_metadata,
                bracket_map, results.corrected_obs, results.obs_u8, results.preds,
                results.demo_z_bits, results.demo_x_bits, results.correlations,
                results.pauli_tracker.virtual_gates, results.pauli_tracker.frame, int(args.shots), stim_rounds,
                results.snapshot_bits
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
