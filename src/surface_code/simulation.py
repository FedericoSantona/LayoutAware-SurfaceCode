"""Simulation execution for surface code logical circuits.

This module handles the full simulation pipeline:
- DEM hardening and MWPM decoding
- Pauli frame tracking and CNOT byproduct extraction
- Observable correction
- Measurement extraction (demos, snapshots)
- Correlation computation
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pymatching as pm
import stim
from qiskit import QuantumCircuit

from .dem_utils import (
    augment_dem_with_boundary_anchors,
    harden_dem_for_pairwise_matching,
    anchor_pm_isolates,
    report_boundaryless_components,
    scan_boundaryless_odd_shot,
    pm_find_offending_shot,
)
from .pauli import PauliTracker, sequence_from_qc
from .utils import compute_two_qubit_correlations, compute_joint_correlations, wilson_rate_ci


@dataclass
class SimulationResults:
    """Structured results from logical simulation."""
    # Core simulation results
    dem: stim.DetectorErrorModel
    det_samp: np.ndarray
    obs_u8: np.ndarray  # Raw observable samples
    preds: np.ndarray   # Decoder predictions
    corrected_obs: np.ndarray  # Corrected observables
    
    # Pauli frame tracking
    pauli_tracker: PauliTracker
    cnot_metadata: List[Dict[str, Any]]
    
    # Measurement extraction
    demo_z_bits: Dict[str, np.ndarray]
    demo_x_bits: Dict[str, np.ndarray]
    demo_meta: Dict[str, Any]
    joint_demo_bits: Dict[str, Dict[str, Any]]
    snapshot_bits: Dict[str, np.ndarray]
    
    # Post-processing
    basis_labels: Tuple[str, ...]
    patch_to_obs_idx: Dict[str, int]
    expected_flips: List[int]
    correlation_pairs: List[Tuple[str, str]]
    correlations: Dict[str, Dict[str, float]]
    per_qubit_ler: List[float]
    per_qubit_ler_ci: List[Tuple[float, float]]


def _harden_and_decode_dem(
    dem: stim.DetectorErrorModel,
    metadata: Dict[str, Any],
    observable_pairs: Sequence[Tuple[int, int]],
    shots: int,
    seed: int,
    verbose: bool = False,
) -> Tuple[stim.DetectorErrorModel, np.ndarray, np.ndarray, np.ndarray]:
    """Harden DEM and decode using MWPM. Returns (dem, det_samp, obs_u8, preds)."""
    # Add boundary anchors from builder metadata (if any)
    try:
        ba = (metadata.get("boundary_anchors", {}) or {})
        anchor_ids = list(ba.get("detector_ids", []) or [])
        anchor_eps = float(ba.get("epsilon", 1e-12))
        if verbose:
            print(f"[DEM-CHECK] attempting anchor augmentation: ids={len(anchor_ids)}, eps={anchor_eps:g}")
        dem = augment_dem_with_boundary_anchors(dem, anchor_ids, anchor_eps)
    except Exception as _exc:
        if verbose:
            print(f"[DEM-CHECK] anchor augmentation skipped due to error: {_exc}")

    # Ensure every connected component has a singleton for pairwise MWPM
    dem, added_pairwise_hooks = harden_dem_for_pairwise_matching(dem, epsilon=1e-12)
    if verbose:
        print(f"[DEM-CHECK] added {added_pairwise_hooks} pairwise boundary hooks and rebuilt DEM via text")

    dem_sampler = dem.compile_sampler(seed=seed)
    det_samp, obs_samp, _ = dem_sampler.sample(shots)
    obs_u8 = np.asarray(obs_samp, dtype=np.uint8) if obs_samp is not None and obs_samp.size > 0 else np.zeros((shots, len(observable_pairs)), dtype=np.uint8)

    try:
        # Pairwise matching (no correlations; requires true single-detector boundaries)
        matcher = pm.Matching.from_detector_error_model(dem)
        dem, matcher, iso_added, iso_ids = anchor_pm_isolates(dem, matcher, epsilon=1e-12)
        if iso_added and verbose:
            print(f"[DECODE] anchored {iso_added} isolated detectors: {iso_ids[:12]}")
        preds = matcher.decode_batch(det_samp.astype(bool))
    except Exception as exc_mwpm:
        if verbose:
            print("[DECODE] Pairwise MWPM failed; hardening DEM and retrying once:", repr(exc_mwpm))
        try:
            # Harden again (in case the sampler revealed an odd-parity component)
            dem, added_retry = harden_dem_for_pairwise_matching(dem, epsilon=1e-12)
            if verbose:
                print(f"[DEM-CHECK] added {added_retry} additional pairwise boundary hooks on retry and rebuilt DEM via text")
            matcher = pm.Matching.from_detector_error_model(dem)
            dem, matcher, iso_added_retry, iso_ids_retry = anchor_pm_isolates(dem, matcher, epsilon=1e-12)
            if iso_added_retry and verbose:
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
                scan_boundaryless_odd_shot(dem, det_samp, max_scan=min(2048, shots))
            except Exception as _exc:
                print(f"[DEM-CHECK] diagnostics failed: {_exc}")
            # PyMatching graph-level parity check on the actual matching graph
            try:
                if 'matcher' in locals():
                    print("\n[PM-CHECK] analyzing PyMatching graph components & boundaries...")
                    pm_find_offending_shot(matcher, det_samp, max_scan=min(2048, shots))
                else:
                    print("[PM-CHECK] matcher unavailable; skipping")
            except Exception as _exc3:
                print(f"[PM-CHECK] diagnostics failed: {_exc3}")
            raise
    
    preds = np.asarray(preds, dtype=np.uint8)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    
    return dem, det_samp, obs_u8, preds


def _track_pauli_frame(
    circuit: stim.Circuit,
    metadata: Dict[str, Any],
    qc: QuantumCircuit,
    shots: int,
    seed: int,
    demo_basis: Optional[str],
) -> Tuple[PauliTracker, List[Dict[str, Any]], np.ndarray]:
    """Extract CNOT byproducts and track Pauli frame. Returns (pauli_tracker, cnot_metadata, m_samples)."""
    # Sample raw measurements for merge byproduct extraction
    circ_sampler = circuit.compile_sampler(seed=seed)
    m_samples = circ_sampler.sample(shots=shots)

    # Extract CNOT parity bits directly from single-shot MPPs and update Pauli frame
    pfm = PauliTracker(qc.num_qubits)
    # Initialize frame bits with correct shots dimension
    shots_count = shots
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
    arg_demo = (demo_basis or "Z").strip().upper()
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
        
        m_zz = m_samples[:, m_zz_mpp_idx] if (enable_mzz and m_zz_mpp_idx is not None) else np.zeros(shots, dtype=np.uint8)
        m_xx = m_samples[:, m_xx_mpp_idx] if (enable_mxx and m_xx_mpp_idx is not None) else np.zeros(shots, dtype=np.uint8)
        
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
    
    return pfm, cnot_metadata, m_samples


def _apply_corrections(
    obs_u8: np.ndarray,
    preds: np.ndarray,
    bracket_map: Dict[str, str],
    pauli_tracker: PauliTracker,
) -> Tuple[np.ndarray, Dict[str, int], Tuple[str, ...]]:
    """Apply Pauli frame and decoder corrections to logical observables. Returns (corrected_obs, patch_to_obs_idx, basis_labels)."""
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
                    corrected_obs[:, obs_idx] ^= pauli_tracker.frame[patch_name]["fz"]
                else:
                    corrected_obs[:, obs_idx] ^= pauli_tracker.frame[patch_name]["fx"]
    
    # Apply decoder predictions to flip outcomes when decoder detects errors
    # The decoder predictions indicate when the logical outcome should be flipped
    corrected_obs ^= preds
    
    return corrected_obs, patch_to_obs_idx, tuple(bracket_map[q] for q in sorted(bracket_map))


def _extract_measurements(
    circuit: stim.Circuit,
    metadata: Dict[str, Any],
    m_samples: np.ndarray,
    verbose: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    """Extract demo readouts and snapshots. Returns (demo_z_bits, demo_x_bits, demo_meta, joint_demo_bits, snapshot_bits)."""
    # Extract demo readouts for physics analysis
    demo_z_bits = {}
    demo_x_bits = {}
    demo_meta = metadata.get("demo", {})

    # DEBUG: print tail of circuit operations to ensure joint MPPs are last
    if verbose:
        try:
            tail_ops = str(circuit).strip().splitlines()[-80:]
            print("\n[DEBUG] Tail of Stim circuit (last ~80 ops):")
            for ln in tail_ops:
                print("  ", ln)
        except Exception:
            pass

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

    # Extract final snapshot bits (if present)
    snapshot_bits = {}
    snapshot_meta = metadata.get("final_snapshot", {})
    if snapshot_meta.get("enabled"):
        order = snapshot_meta["order"]
        indices = snapshot_meta["indices"]
        for qubit_name, idx in zip(order, indices):
            snapshot_bits[qubit_name] = np.asarray(m_samples[:, idx], dtype=np.uint8)
    
    return demo_z_bits, demo_x_bits, demo_meta, joint_demo_bits, snapshot_bits


def _apply_joint_demo_corrections(
    joint_demo_bits: Dict[str, Dict[str, Any]],
    pauli_tracker: PauliTracker,
) -> Dict[str, Dict[str, Any]]:
    """Apply Pauli-frame corrections to joint demo bits."""
    if not joint_demo_bits or not pauli_tracker.frame:
        return joint_demo_bits
    
    corrected_joint_demo_bits: Dict[str, Dict[str, Any]] = {}
    for joint_key, demo_data in joint_demo_bits.items():
        bits = demo_data["bits"]
        basis = demo_data["basis"]
        pair = demo_data["pair"]
        axes_map = demo_data.get("axes", {})
        final_bases = demo_data.get("final_bases", {})

        # Prefer axes_map from conjugated operator; fallback to final_bases
        flips = np.zeros_like(bits, dtype=np.uint8)
        for qubit_name in pair:
            frame = pauli_tracker.frame.get(qubit_name)
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
    
    return corrected_joint_demo_bits


def _compute_correlations(
    joint_demo_bits: Dict[str, Dict[str, Any]],
    demo_z_bits: Dict[str, np.ndarray],
    demo_x_bits: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    m_samples: np.ndarray,
    correlation_pairs: List[Tuple[str, str]],
    shots: int,
) -> Dict[str, Dict[str, float]]:
    """Compute correlations with byproduct corrections."""
    correlations = {}
    if joint_demo_bits:
        # Apply byproduct corrections to joint demo bits before computing correlations
        corrected_joint_demo_bits = {}
        for joint_key, demo_data in joint_demo_bits.items():
            corrected_data = demo_data.copy()
            pair = demo_data["pair"]
            basis = demo_data["basis"]
            
            # Find corresponding CNOT operation for this pair
            byproduct_correction = np.zeros(shots, dtype=np.uint8)
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
        
        correlations = compute_joint_correlations(corrected_joint_demo_bits, shots)
    elif correlation_pairs and demo_z_bits and demo_x_bits:
        # Fallback to old method if no joint demos available
        correlations = compute_two_qubit_correlations(demo_z_bits, demo_x_bits, correlation_pairs, shots)
    
    return correlations


def run_logical_simulation(
    circuit: stim.Circuit,
    dem: stim.DetectorErrorModel,
    metadata: Dict[str, Any],
    observable_pairs: Sequence[Tuple[int, int]],
    bracket_map: Dict[str, str],
    qc: QuantumCircuit,
    shots: int,
    seed: int,
    demo_basis: Optional[str],
    bracket_basis: str,
    corr_pairs: Optional[str] = None,
    verbose: bool = False,
) -> SimulationResults:
    """Run full logical simulation with DEM decoding, Pauli frame tracking, and measurement extraction.
    
    Args:
        circuit: Compiled Stim circuit
        dem: Initial detector error model
        metadata: Builder metadata containing CNOT operations, demos, etc.
        observable_pairs: Observable pairs from builder
        bracket_map: Map of patch names to bracket basis ('Z' or 'X')
        qc: Original Qiskit circuit
        shots: Number of Monte Carlo samples
        seed: Random seed for samplers
        demo_basis: Demo basis ('Z', 'X', 'auto', or None)
        bracket_basis: Bracket basis used for initialization
        corr_pairs: Optional custom correlation pairs string ('q0,q1;q2,q3')
        verbose: Enable verbose debug output
    
    Returns:
        SimulationResults containing all simulation outputs
    """
    # Step 1: Harden DEM and decode
    dem, det_samp, obs_u8, preds = _harden_and_decode_dem(
        dem, metadata, observable_pairs, shots, seed, verbose
    )
    
    # Step 2: Track Pauli frame and extract CNOT byproducts
    pauli_tracker, cnot_metadata, m_samples = _track_pauli_frame(
        circuit, metadata, qc, shots, seed, demo_basis
    )
    
    # Step 3: Apply corrections to observables
    corrected_obs, patch_to_obs_idx, basis_labels_from_bracket = _apply_corrections(
        obs_u8, preds, bracket_map, pauli_tracker
    )
    
    # Get basis labels from metadata or fallback to bracket_map
    basis_labels = tuple(metadata.get("observable_basis", tuple()))
    if not basis_labels or len(basis_labels) != obs_u8.shape[1]:
        basis_labels = basis_labels_from_bracket
    
    # Ensure basis_labels matches the actual number of observable columns
    if len(basis_labels) > obs_u8.shape[1]:
        basis_labels = basis_labels[:obs_u8.shape[1]]
    
    # Step 4: Track virtual gates
    for qname, gates in sequence_from_qc(qc).items():
        qidx = int(qname[1:])
        if gates:
            pauli_tracker.set_sequence(qidx, gates)
    
    # Derive per-qubit expected flips for debug/verbose
    expected_flips = []
    for i in range(qc.num_qubits):
        seq = pauli_tracker.virtual_gates[f"q{i}"]
        _, phase = PauliTracker.conjugate_axis_by_sequence(bracket_basis, seq)
        expected_flips.append(1 if phase < 0 else 0)
    
    # Step 5: Extract measurements
    demo_z_bits, demo_x_bits, demo_meta, joint_demo_bits, snapshot_bits = _extract_measurements(
        circuit, metadata, m_samples, verbose
    )
    
    # Step 6: Apply Pauli frame corrections to joint demos
    joint_demo_bits = _apply_joint_demo_corrections(joint_demo_bits, pauli_tracker)
    
    # Step 7: Build correlation pairs
    correlation_pairs = []
    for cnot_op in metadata.get("cnot_operations", []):
        control = cnot_op["control"]
        target = cnot_op["target"]
        correlation_pairs.append((control, target))
    
    # Add custom correlation pairs if specified
    if corr_pairs:
        try:
            custom_pairs = corr_pairs.split(';')
            for pair_str in custom_pairs:
                if ',' in pair_str:
                    q1, q2 = pair_str.strip().split(',', 1)
                    correlation_pairs.append((q1.strip(), q2.strip()))
        except Exception:
            pass  # Ignore malformed correlation pairs
    
    # Step 8: Compute per-qubit LER with Wilson CI
    per_qubit_ler = []
    per_qubit_ler_ci = []
    for i in range(min(len(basis_labels), obs_u8.shape[1])):
        errors = np.bitwise_xor(obs_u8[:, i], preds[:, i])
        error_count = int(np.sum(errors))
        ler = error_count / shots
        ler_ci = wilson_rate_ci(error_count, shots)
        per_qubit_ler.append(ler)
        per_qubit_ler_ci.append(ler_ci)
    
    # Step 9: Compute correlations
    correlations = _compute_correlations(
        joint_demo_bits, demo_z_bits, demo_x_bits, metadata, m_samples, correlation_pairs, shots
    )
    
    return SimulationResults(
        dem=dem,
        det_samp=det_samp,
        obs_u8=obs_u8,
        preds=preds,
        corrected_obs=corrected_obs,
        pauli_tracker=pauli_tracker,
        cnot_metadata=cnot_metadata,
        demo_z_bits=demo_z_bits,
        demo_x_bits=demo_x_bits,
        demo_meta=demo_meta,
        joint_demo_bits=joint_demo_bits,
        snapshot_bits=snapshot_bits,
        basis_labels=basis_labels,
        patch_to_obs_idx=patch_to_obs_idx,
        expected_flips=expected_flips,
        correlation_pairs=correlation_pairs,
        correlations=correlations,
        per_qubit_ler=per_qubit_ler,
        per_qubit_ler_ci=per_qubit_ler_ci,
    )

