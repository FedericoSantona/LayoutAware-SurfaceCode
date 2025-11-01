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

import numpy as np
import pymatching as pm
import stim

# Import DEM utilities
from surface_code.dem_utils import (
    augment_dem_with_boundary_anchors,
    harden_dem_for_pairwise_matching,
    anchor_pm_isolates,
    report_boundaryless_components,
    scan_boundaryless_odd_shot,
    pm_find_offending_shot,
)

# Import CLI utilities
from cli import parse_args, instantiate_benchmark, load_target, build_config, format_leaderboard

from benchmarks.BenchmarkCircuit import BenchmarkCircuit
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
            dem, matcher, iso_added, iso_ids = anchor_pm_isolates(dem, matcher, epsilon=1e-12)
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
                dem, matcher, iso_added_retry, iso_ids_retry = anchor_pm_isolates(dem, matcher, epsilon=1e-12)
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
                    report_boundaryless_components(dem)
                    scan_boundaryless_odd_shot(dem, det_samp, max_scan=min(2048, int(args.shots)))
                except Exception as _exc:
                    print(f"[DEM-CHECK] diagnostics failed: {_exc}")
                # PyMatching graph-level parity check on the actual matching graph
                try:
                    if 'matcher' in locals():
                        print("\n[PM-CHECK] analyzing PyMatching graph components & boundaries...")
                        pm_find_offending_shot(matcher, det_samp, max_scan=min(2048, int(args.shots)))
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
