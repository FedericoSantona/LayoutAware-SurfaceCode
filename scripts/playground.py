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
from surface_code.memory_layout import build_memory_layout
from surface_code.dem_utils import circuit_to_graphlike_dem

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

        use_memory_layout = (qc.num_qubits == 1)
        if use_memory_layout:
            layout, ops, bracket_map = build_memory_layout(
                model,
                distance=d,
                rounds=stim_rounds,
                bracket_basis=bracket_basis,
                family=None,
            )
        else:
            bracket_map = {f"q{i}": bracket_basis for i in range(qc.num_qubits)}
            layout, ops = compile_circuit_to_surgery(
                qc, patches, seams, distance=d, bracket_map=bracket_map,
                warmup_rounds=warmup_rounds, ancilla_strategy=args.cnot_ancilla_strategy, ancilla_buffer=args.ancilla_buffer
            )

        layout.plot(
            annotate=False,         # set True to label global qubit indices
            seams=True,             # draw rough/smooth seam edges
            title=f"Layout for {args.benchmark} (d={d})",
            save_path=PROJECT_ROOT / "plots" / "layout" / args.code_type / f"layout_{args.benchmark}_d{d}.png",
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
            """
            try:
                print("\n[DEBUG] Stim diagram (detslice-with-ops):")
                print(circuit.diagram('detslice-with-ops', tick=range(0, circuit.num_ticks)))
            except Exception:
                pass
            """

        # Sample DEM and decode
        dem = circuit_to_graphlike_dem(circuit)


        if args.verbose:
            import re
            from collections import Counter

            # 1) Dump the DEM text for manual inspection (optional but handy)
            dem_text = str(dem)
            (PROJECT_ROOT / "plots").mkdir(exist_ok=True)
            (PROJECT_ROOT / "plots" / f"dem_{args.benchmark}.dem.txt").write_text(dem_text)

            # 2) Map detector id -> (tag, patch, row)
            #    This comes from your DetectorManager diagnostics.
            dbg = (metadata.get("mwpm_debug", {}) or {})
            det_ctx = dbg.get("detector_context", {})  # {det_id: {"tag": "...", "context": {...}}}


            missing_ctx = {"x_temporal": 0, "z_temporal": 0}
            wrong_keys = []
            rows_seen = {"x_temporal": set(), "z_temporal": set()}
            for did, meta in det_ctx.items():
                tag = (meta or {}).get("tag")
                if tag in ("x_temporal", "z_temporal"):
                    ctx = (meta or {}).get("context", {}) or {}
                    if "patch" not in ctx or "row" not in ctx:
                        missing_ctx[tag] += 1
                        wrong_keys.append((did, tag, sorted(ctx.keys())))
                    else:
                        rows_seen[tag].add((ctx.get("patch"), ctx.get("row")))
            print("[DEM-CHECK] missing (patch,row):", missing_ctx)
            print("[DEM-CHECK] unique rows per tag: sizes",
                {t: len(s) for t, s in rows_seen.items()})
            if wrong_keys[:5]:
                print("[DEM-CHECK] sample missing/renamed ctx keys:", wrong_keys[:5])

         
            tag_stats = dbg.get("tag_stats", {})
            if tag_stats:
                print("[DEM-CHECK] Detector tag statistics:")
                for tag, stats in tag_stats.items():
                    if "temporal" in tag:
                        print(f"  {tag}: emitted={stats.get('emitted', 0)}, kept={stats.get('kept', 0)}, dropped={stats.get('dropped', 0)}")

            def _info(did: int):
                meta = det_ctx.get(int(did), {}) or {}
                tag = meta.get("tag")
                ctx = meta.get("context", {}) or {}
                patch = ctx.get("patch")
                row = ctx.get("row")
                return tag, patch, row

            # 3) Parse ERROR(...) lines; collect 1-detector and 2-detector terms
            err_line_re = re.compile(r"^\s*error\(([^)]+)\)\s+(.*)$")
            D_re = re.compile(r"\bD(\d+)\b")

            one_det = []     # [(p, d0)]
            error_terms: List[Tuple[float, List[int]]] = []  # [(p, [d0,d1,...])]
            other = 0

            for line in dem_text.splitlines():
                m = err_line_re.match(line)
                if not m:
                    continue
                p_str, rhs = m.groups()
                try:
                    p_val = float(p_str)
                except:
                    # could be a named probability; keep the string
                    p_val = p_str

                Ds = [int(x) for x in D_re.findall(rhs)]
                if len(Ds) == 1:
                    one_det.append((p_val, Ds[0]))
                elif len(Ds) >= 2:
                    error_terms.append((p_val, Ds))
                else:
                    other += 1

            # 4) Summarize two-detector terms that look like cross-row DATA faults:
            #    both ends are temporal detectors of the same basis and same patch,
            #    but on *different* stabilizer rows.
            #    (That’s the signature of a single data X fault for Z-basis rows,
            #     or a data Z fault for X-basis rows.)
            candidates = []
            for p_val, ids in error_terms:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        d0, d1 = ids[i], ids[j]
                        tag0, patch0, row0 = _info(d0)
                        tag1, patch1, row1 = _info(d1)
                        # same patch, same temporal-basis tag, different rows
                        if tag0 in ("z_temporal", "x_temporal") and tag0 == tag1 \
                        and patch0 is not None and patch0 == patch1 \
                        and isinstance(row0, int) and isinstance(row1, int) and row0 != row1:
                            # normalize pair order
                            r_pair = tuple(sorted((row0, row1)))
                            candidates.append((tag0, patch0, r_pair, p_val, d0, d1))


            x_temporal_candidates = [(tag, patch, rpair, p_val, d0, d1) 
                                    for tag, patch, rpair, p_val, d0, d1 in candidates 
                                    if tag == "x_temporal"]
            if x_temporal_candidates:
                print(f"[DEM-CHECK] Found {len(x_temporal_candidates)} x_temporal cross-row candidates")
                for tag, patch, rpair, p_val, d0, d1 in x_temporal_candidates[:5]:
                    print(f"  {tag} {patch} rows={rpair} error({p_val}) D{d0} D{d1}")
            else:
                print("[DEM-CHECK] No x_temporal cross-row candidates found")
                # Check if x_temporal detectors appear in any error terms
                x_temporal_in_errors = []
                for p_val, d0, d1 in two_det:
                    tag0, _, _ = _info(d0)
                    tag1, _, _ = _info(d1)
                    if tag0 == "x_temporal" or tag1 == "x_temporal":
                        x_temporal_in_errors.append((p_val, d0, d1, tag0, tag1))
                print(f"[DEM-CHECK] x_temporal detectors appear in {len(x_temporal_in_errors)} two-detector error terms")
                if x_temporal_in_errors:
                    print("[DEM-CHECK] Sample x_temporal error terms:")
                    for p_val, d0, d1, tag0, tag1 in x_temporal_in_errors[:5]:
                        print(f"  error({p_val}) D{d0}({tag0}) D{d1}({tag1})")

            # 5) Print a compact summary
            print(f"[DEM-CHECK] total ERROR terms: 1-det={len(one_det)}, >=2-det={len(error_terms)}, other={other}")

            # Group by (basis, patch, (row_i,row_j)) and show counts
            bucket = Counter((tag, patch, rpair) for tag, patch, rpair, _, _, _ in candidates)
            print("[DEM-CHECK] cross-row 2-detector candidates (same patch & basis, different rows):")
            for (tag, patch, rpair), cnt in bucket.most_common(15):
                print(f"  {tag}  {patch} rows={rpair}  count={cnt}")

            # Show a few concrete examples with probabilities
            print("[DEM-CHECK] sample cross-row edges (up to 10):")
            for tag, patch, rpair, p_val, d0, d1 in candidates[:10]:
                print(f"  error({p_val}) D{d0} D{d1}   -> {tag}  {patch} rows={rpair}")

            # 6) Also sanity-check the single-detector measurement-fault edges for weights
            #    (useful to see p_meas clustered here)
            meas_ones = []
            for p_val, d0 in one_det:
                tag0, patch0, row0 = _info(d0)
                if tag0 in ("z_temporal", "x_temporal"):
                    meas_ones.append(p_val)
            if meas_ones:
                # crude: just print a few and a rough min/max
                vals = [v for v in meas_ones if isinstance(v, float)]
                print(f"[DEM-CHECK] single-detector temporal terms (measurement-like) samples: {meas_ones[:5]}")
                if vals:
                    print(f"[DEM-CHECK] single-detector (temporal) prob range: {min(vals):.3g}..{max(vals):.3g}")

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

        # Post-augmentation DEM sanity (augmented graph)
        if args.verbose:
            try:
                dem_text_post = str(results.dem)
                import re
                err_re = re.compile(r"^\s*error\(([^)]+)\)\s+(.*)$")
                D_re = re.compile(r"\bD(\d+)\b")

                dbg = (metadata.get("mwpm_debug", {}) or {})
                det_ctx = dbg.get("detector_context", {}) or {}

                def _info_post(did: int):
                    meta = det_ctx.get(int(did), {}) or {}
                    tag = meta.get("tag")
                    ctx = meta.get("context", {}) or {}
                    patch = ctx.get("patch")
                    row = ctx.get("row")
                    return tag, patch, row

                two_det_post = []
                for line in dem_text_post.splitlines():
                    m = err_re.match(line)
                    if not m:
                        continue
                    p_str, rhs = m.groups()
                    try:
                        p_val = float(p_str)
                    except Exception:
                        p_val = p_str
                    Ds = [int(x) for x in D_re.findall(rhs)]
                    if len(Ds) == 2:
                        two_det_post.append((p_val, Ds[0], Ds[1]))

                # Count post-aug cross-row candidates
                x_post = []
                z_post = []
                for p_val, d0, d1 in two_det_post:
                    tag0, patch0, row0 = _info_post(d0)
                    tag1, patch1, row1 = _info_post(d1)
                    if not (patch0 == patch1 and isinstance(row0, int) and isinstance(row1, int) and row0 != row1):
                        continue
                    if tag0 == tag1 == "x_temporal":
                        x_post.append((p_val, d0, d1, patch0, tuple(sorted((row0, row1)))))
                    if tag0 == tag1 == "z_temporal":
                        z_post.append((p_val, d0, d1, patch0, tuple(sorted((row0, row1)))))

                from collections import Counter
                bx = Counter((patch, rpair) for _p, _d0, _d1, patch, rpair in x_post)
                bz = Counter((patch, rpair) for _p, _d0, _d1, patch, rpair in z_post)
                print("[DEM-CHECK:POST] cross-row 2-detector candidates (augmented):")
                if bx:
                    print("  x_temporal:")
                    for (patch, rpair), cnt in list(bx.most_common(10)):
                        print(f"    {patch} rows={rpair}  count={cnt}")
                else:
                    print("  x_temporal: none")
                if bz:
                    print("  z_temporal:")
                    for (patch, rpair), cnt in list(bz.most_common(10)):
                        print(f"    {patch} rows={rpair}  count={cnt}")
                else:
                    print("  z_temporal: none")
            except Exception as _exc_post:
                print(f"[DEM-CHECK:POST] scan failed: {_exc_post}")

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
            print_decoded_logical_distribution(
                decoded_order,
                results.corrected_obs,
                int(args.shots),
                pauli_frame=results.pauli_tracker.frame,
                basis_labels=results.basis_labels,
            )
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
