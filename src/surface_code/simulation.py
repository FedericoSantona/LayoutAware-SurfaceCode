"""Simulation execution for surface code logical circuits.

This module handles the full simulation pipeline:
- MWPM decoding
- Pauli frame tracking and CNOT byproduct extraction
- Observable correction
- Measurement extraction (demos, snapshots)
- Correlation computation
"""

from __future__ import annotations

from collections import Counter
import numbers
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pymatching as pm
import stim
import networkx as nx
from qiskit import QuantumCircuit

from .dem_diagnostics import log_dem_diagnostics, env_dem_debug_enabled
from .dem_utils import (
    report_boundaryless_components,
    scan_boundaryless_odd_shot,
    pm_find_offending_shot,
    single_detector_hook_ids,
    log_hook_summary,
    scan_infeasible_shot,
    collect_detectors_in_errors,
    update_tag_stats_with_presence,
)
from .pauli import PauliTracker, sequence_from_qc
from .utils import compute_two_qubit_correlations, compute_joint_correlations, wilson_rate_ci


def _parity_from_indices(
    m_samples: np.ndarray,
    indices: Optional[Sequence[int]],
) -> np.ndarray:
    """Return XOR parity across the specified measurement columns."""
    if indices:
        cols = m_samples[:, list(indices)]
        cols = np.asarray(cols, dtype=np.uint8)
        if cols.ndim == 1:
            return cols.copy()
        return np.bitwise_xor.reduce(cols, axis=1).astype(np.uint8)
    # No indices ⇒ deterministic zero parity.
    return np.zeros(m_samples.shape[0], dtype=np.uint8)


# Helper: Count observable support in ERROR terms
def _debug_obs_support(dem: stim.DetectorErrorModel) -> List[int]:
    """Return a list where entry k is the number of ERROR terms that reference observable k."""
    try:
        try:
            n_obs = int(dem.num_observables)
        except Exception:
            n_obs = int(dem.num_observables())
        counts = [0] * max(n_obs, 0)
        for inst in dem:
            # Robust "is this an ERROR instruction?" check
            is_error = False
            try:
                # Preferred: enum comparison
                if hasattr(stim, "DemInstructionType"):
                    is_error = (getattr(inst, "type", None) == stim.DemInstructionType.ERROR)
                # Fallbacks: string name or enum-name
                if not is_error:
                    name = getattr(inst, "name", None)
                    if name is not None:
                        is_error = (str(name).lower() == "error")
                if not is_error:
                    t = getattr(inst, "type", None)
                    if t is not None:
                        is_error = (str(getattr(t, "name", t)).lower() == "error")
                # Last-resort: textual representation
                if not is_error:
                    s = str(inst).lstrip().lower()
                    if s.startswith("error"):
                        is_error = True
            except Exception:
                is_error = False
            if not is_error:
                continue

            # Access targets in a Stim-version-proof way
            targets = []
            try:
                if hasattr(inst, "targets_copy"):
                    targets = list(inst.targets_copy())
                else:
                    raw = getattr(inst, "targets", ())
                    try:
                        targets = list(raw)
                    except Exception:
                        targets = []
            except Exception:
                targets = []

            # Count observable ids; fallback to text parse if needed
            any_obs = False
            for t in targets:
                try:
                    if hasattr(t, "is_observable_id") and t.is_observable_id():
                        k = int(getattr(t, "value", getattr(t, "val", 0)))
                        if 0 <= k < len(counts):
                            counts[k] += 1
                            any_obs = True
                except Exception:
                    pass
            if not any_obs:
                # Fallback: parse textual tokens like "L0", "L1"
                try:
                    toks = str(inst).replace(",", " ").split()
                    for tok in toks:
                        if tok.startswith("L") and tok[1:].isdigit():
                            k = int(tok[1:])
                            if 0 <= k < len(counts):
                                counts[k] += 1
                                any_obs = True
                except Exception:
                    pass
        return counts
    except Exception:
        return []


def _debug_iter_observable_error_terms(dem: stim.DetectorErrorModel, limit: int = 12) -> List[str]:
    """Return formatted summaries of the first few ERROR terms that include any observable ids."""
    out: List[str] = []
    try:
        for inst in dem:
            # Robust "is this an ERROR instruction?" check
            is_error = False
            try:
                if hasattr(stim, "DemInstructionType"):
                    is_error = (getattr(inst, "type", None) == stim.DemInstructionType.ERROR)
                if not is_error:
                    name = getattr(inst, "name", None)
                    if name is not None:
                        is_error = (str(name).lower() == "error")
                if not is_error:
                    t = getattr(inst, "type", None)
                    if t is not None:
                        is_error = (str(getattr(t, "name", t)).lower() == "error")
                if not is_error:
                    s0 = str(inst).lstrip().lower()
                    if s0.startswith("error"):
                        is_error = True
            except Exception:
                is_error = False
            if not is_error:
                continue

            # Stim-version-safe target access
            targets = []
            try:
                if hasattr(inst, "targets_copy"):
                    targets = list(inst.targets_copy())
                else:
                    raw = getattr(inst, "targets", ())
                    try:
                        targets = list(raw)
                    except Exception:
                        targets = []
            except Exception:
                targets = []

            obs_ids: List[int] = []
            det_ids: List[int] = []
            for t in targets:
                try:
                    if hasattr(t, "is_observable_id") and t.is_observable_id():
                        obs_ids.append(int(getattr(t, "value", getattr(t, "val", 0))))
                    elif hasattr(t, "is_detector_id") and t.is_detector_id():
                        det_ids.append(int(getattr(t, "value", getattr(t, "val", 0))))
                except Exception:
                    pass

            # Try to read the probability/weight if available
            prob = None
            try:
                if hasattr(inst, "args_copy"):
                    args = inst.args_copy()
                else:
                    args = getattr(inst, "args", None) or getattr(inst, "arguments", None)
                if args:
                    try:
                        prob = float(args[0])
                    except Exception:
                        pass
            except Exception:
                pass
            if prob is None:
                # Parse from textual form like "error(0.001) ..."
                try:
                    s1 = str(inst)
                    if "error(" in s1:
                        s2 = s1.split("error(", 1)[1]
                        s3 = s2.split(")", 1)[0]
                        prob = float(s3.strip().split()[0])
                except Exception:
                    pass

            if obs_ids:
                prob_str = f"p={prob:.3g}" if isinstance(prob, float) else "p=?"
                out.append(f"ERROR {prob_str}  dets={det_ids[:6]}  obs={obs_ids}")
                if len(out) >= limit:
                    break
    except Exception:
        pass
    return out


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


def _decode_dem(
    dem: stim.DetectorErrorModel,
    metadata: Dict[str, Any],
    observable_pairs: Sequence[Tuple[int, int]],
    shots: int,
    seed: int,
    verbose: bool = False,
) -> Tuple[stim.DetectorErrorModel, np.ndarray, np.ndarray, np.ndarray]:
    """Decode DEM using MWPM. Returns (dem, det_samp, obs_u8, preds)."""
    dem_debug = verbose or env_dem_debug_enabled()
    log_dem_diagnostics("pre", dem, metadata, enabled=dem_debug)

    # Snapshot original DEM for diagnostics
    if verbose:
        try:
            obs_support = _debug_obs_support(dem)
            print(f"[OBS-SUPPORT] per-observable error terms: {obs_support}")
            if sum(obs_support) > 0:
                samples = _debug_iter_observable_error_terms(dem, limit=6)
                if samples:
                    print("[OBS-SUPPORT] sample ERROR lines carrying observables:")
                    for s in samples:
                        print("  ", s)
            try:
                txt = str(dem)
                c0 = txt.count(" L0")
                c1 = txt.count(" L1")
                print(f"[OBS-TEXT] raw DEM text counts: L0={c0} L1={c1}")
            except Exception:
                pass
        except Exception as _exc:
            print(f"[OBS-SUPPORT] scan failed: {_exc}")

    dem_orig = dem
    if verbose:
        try:
            base_hooks = single_detector_hook_ids(dem_orig)
            print(f"[DEM-HOOKS] initial single-detector hooks: {len(base_hooks)} (sample {base_hooks[:12]})")
        except Exception as _exc:
            print(f"[DEM-HOOKS] initial hook scan failed: {_exc}")

    pm_nodes = collect_detectors_in_errors(dem)
    update_tag_stats_with_presence(metadata, pm_nodes)

    if verbose:
        try:
            log_hook_summary(dem_orig, dem, prefix="[DEM-HOOKS]")
        except Exception as _exc:
            print(f"[DEM-HOOKS] summary failed: {_exc}")

    # Build matcher before sampling so we can ensure PM components are bounded.
    if verbose:
        print("[DECODE] PyMatching mode: correlated (enable_correlations=False)")
    matcher = pm.Matching.from_detector_error_model(dem, enable_correlations=False)
    graph = matcher.to_networkx()
    mg = matcher._matching_graph
    det_nodes = {int(n) for n in graph.nodes() if isinstance(n, int) and 0 <= int(n) < matcher.num_detectors}
    boundary_detectors = {
        int(a) for a, b, _data in mg.get_edges()
        if b is None and isinstance(a, numbers.Integral)
    }

    components: List[List[int]] = []
    boundaryless_components: List[List[int]] = []
    for comp in nx.connected_components(graph):
        dets = [int(n) for n in comp if isinstance(n, int) and 0 <= int(n) < matcher.num_detectors]
        if not dets:
            continue
        components.append(dets)
        if not any(d in boundary_detectors for d in dets):
            boundaryless_components.append(dets)

    size_counter = Counter(len(comp) for comp in components if comp)
    hist_preview = {k: size_counter[k] for k in list(size_counter.keys())[:8]}
    ctx_map = (metadata.get("mwpm_debug", {}) or {}).get("detector_context", {}) or {}
    if verbose:
        print(f"[DEM-CHECK] component histogram: {hist_preview}")
        if boundaryless_components:
            sample = []
            for comp in boundaryless_components[:16]:
                rep = int(comp[0])
                info = ctx_map.get(rep, {}) or {}
                sample.append((rep, info.get("tag"), info.get("context")))
            print(f"[DEM-CHECK] boundaryless components (sample): {sample}")

    # Observable support snapshot (for debugging)
    if verbose:
        try:
            obs_support_post = _debug_obs_support(dem)
            print(f"[OBS-SUPPORT] per-observable error terms: {obs_support_post}")
            if sum(obs_support_post) > 0:
                samples = _debug_iter_observable_error_terms(dem, limit=6)
                if samples:
                    print("[OBS-SUPPORT] sample ERROR lines carrying observables:")
                    for s in samples:
                        print("  ", s)
            else:
                print("[OBS-SUPPORT] no ERROR terms carry any observables.")
            try:
                txt = str(dem)
                c0 = txt.count(" L0")
                c1 = txt.count(" L1")
                print(f"[OBS-TEXT] raw DEM text counts: L0={c0} L1={c1}")
            except Exception:
                pass
        except Exception as _exc:
            print(f"[OBS-SUPPORT] scan failed: {_exc}")

    actual_hooks = single_detector_hook_ids(dem)
    metadata.setdefault("boundary_anchors", {})["detector_ids"] = list(actual_hooks)
    log_dem_diagnostics("post", dem, metadata, enabled=dem_debug)

    pm_nodes = collect_detectors_in_errors(dem)

    dem_sampler = dem.compile_sampler(seed=seed)
    det_samp, obs_samp, _ = dem_sampler.sample(shots)
    obs_u8 = np.asarray(obs_samp, dtype=np.uint8) if obs_samp is not None and obs_samp.size > 0 else np.zeros((shots, len(observable_pairs)), dtype=np.uint8)
    
    # Debug: Check if observables are connected to errors in DEM
    if verbose or (obs_u8.size > 0 and len(observable_pairs) > 0):
        try:
            dem_text = str(dem)
            obs_errors = []
            for i in range(len(observable_pairs)):
                obs_label = f"L{i}"
                # Count error lines that include this observable
                import re
                error_lines = [line for line in dem_text.splitlines() if "error(" in line.lower() and obs_label in line]
                obs_errors.append(len(error_lines))
            if verbose:
                print(f"[OBS-ERRORS] Observable error term counts: {obs_errors}")
            # If no observables are connected to errors, that's a problem!
            if all(count == 0 for count in obs_errors) and len(obs_errors) > 0:
                print(f"[CRITICAL] No observables are connected to ERROR terms in DEM!")
                print(f"          This means observables are deterministic and won't track errors!")
                print(f"          Observable pairs: {observable_pairs}")
        except Exception as _exc:
            if verbose:
                print(f"[OBS-ERRORS] Check failed: {_exc}")

    try:
        if verbose:
            try:
                graph = matcher.to_networkx()
                num_det = matcher.num_detectors
                det_nodes = {int(n) for n in graph.nodes() if isinstance(n, int) and 0 <= int(n) < num_det}
                mg = matcher._matching_graph
                boundary_nodes = {
                    int(a) for a, b, _data in mg.get_edges()
                    if b is None and isinstance(a, numbers.Integral)
                }
                degree_zero = [
                    int(n) for n in det_nodes
                    if graph.degree(n) == 0
                ]
                missing_nodes = sorted(set(range(num_det)) - det_nodes)
                comp_sizes: Counter[int] = Counter()
                boundaryless_reps = []
                for comp in nx.connected_components(graph):
                    dets = [int(n) for n in comp if isinstance(n, numbers.Integral) and 0 <= int(n) < num_det]
                    if not dets:
                        continue
                    comp_sizes[len(dets)] += 1
                    if not any(d in boundary_nodes for d in dets):
                        boundaryless_reps.append(dets[0])
                print(
                    f"[PM-GRAPH] det={num_det} nodes={len(det_nodes)} "
                    f"boundary_nodes={len(boundary_nodes)} zero_degree={len(degree_zero)} missing={len(missing_nodes)}"
                )
                if comp_sizes:
                    sample_hist = dict(list(comp_sizes.items())[:8])
                    print(f"[PM-GRAPH] component size histogram (size->count): {sample_hist}")
                if boundaryless_reps:
                    ctx_map = (metadata.get("mwpm_debug", {}) or {}).get("detector_context", {}) or {}
                    sample_ctx = {det: ctx_map.get(int(det)) for det in boundaryless_reps[:16]}
                    print(f"[PM-GRAPH] boundaryless component samples: {sample_ctx}")
                if degree_zero:
                    print(f"[PM-GRAPH] zero-degree sample: {degree_zero[:16]}")
                if missing_nodes:
                    ctx_map = (metadata.get("mwpm_debug", {}) or {}).get("detector_context", {}) or {}
                    sample_ctx = {det: ctx_map.get(int(det)) for det in missing_nodes[:16]}
                    print(f"[PM-GRAPH] missing node context: {sample_ctx}")
            except Exception as _exc:
                print(f"[PM-GRAPH] diagnostics failed: {_exc}")
        preds = matcher.decode_batch(det_samp.astype(bool), enable_correlations=False)
        
        # Debug: Check if preds and obs_u8 are suspiciously identical
        if obs_u8.size > 0 and len(observable_pairs) > 0:
            preds_check = np.asarray(preds, dtype=np.uint8)
            if preds_check.ndim == 1:
                preds_check = preds_check.reshape(-1, 1)
            for i in range(min(len(observable_pairs), obs_u8.shape[1], preds_check.shape[1])):
                if np.array_equal(obs_u8[:, i], preds_check[:, i]):
                    print(f"[CRITICAL] Observable {i}: preds and obs_u8 are IDENTICAL!")
                    print(f"          obs_u8 mean: {obs_u8[:, i].mean():.6f}, preds mean: {preds_check[:, i].mean():.6f}")
                    print(f"          This suggests the decoder is not actually decoding from detectors!")
                    print(f"          Observable pair: {observable_pairs[i] if i < len(observable_pairs) else 'N/A'}")
                    # Check if detectors are actually firing
                    det_firing_rate = det_samp.astype(bool).sum(axis=1).mean()
                    print(f"          Average detectors firing per shot: {det_firing_rate:.2f}")
    except Exception as exc_mwpm:
        if verbose:
            print("[DECODE] Pairwise MWPM failed:", repr(exc_mwpm))
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
        raise exc_mwpm
    
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
        m_zz_indices = cnot_op.get("m_zz_indices")
        m_xx_indices = cnot_op.get("m_xx_indices")
        m_xx_logical_idx = cnot_op.get("m_xx_logical_idx")
        
        if enable_mzz:
            m_zz = _parity_from_indices(m_samples, m_zz_indices)
        else:
            m_zz = np.zeros(shots, dtype=np.uint8)
        if enable_mxx:
            m_xx = _parity_from_indices(m_samples, m_xx_indices or ([m_xx_logical_idx] if m_xx_logical_idx is not None else None))
        else:
            m_xx = np.zeros(shots, dtype=np.uint8)
        
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
    metadata: Dict[str, Any],
    bracket_map: Dict[str, str],
    pauli_tracker: PauliTracker,
) -> Tuple[np.ndarray, Dict[str, int], Tuple[str, ...]]:
    """Apply Pauli frame and decoder corrections to logical observables. Returns (corrected_obs, patch_to_obs_idx, basis_labels)."""
    # Apply Pauli frame and decoder corrections to logical observables
    corrected_obs = obs_u8.copy()
    
    # Map patch names to their observable indices
    patch_order = list(metadata.get("observable_patches", ()))
    if not patch_order:
        patch_order = sorted(bracket_map.keys())

    patch_to_obs_idx = {patch: idx for idx, patch in enumerate(patch_order)}
    
    # Apply Pauli frame corrections
    basis_labels = list(metadata.get("observable_basis", ()))
    for patch_name in patch_order:
        if patch_name in patch_to_obs_idx:
            obs_idx = patch_to_obs_idx[patch_name]
            # Check if obs_idx is within bounds
            if obs_idx < corrected_obs.shape[1]:
                if obs_idx < len(basis_labels):
                    basis = basis_labels[obs_idx]
                else:
                    basis = bracket_map.get(patch_name, "Z")
                frame_entry = pauli_tracker.frame.get(patch_name, {})
                if basis == "Z":
                    val = frame_entry.get("fz", 0)
                else:
                    val = frame_entry.get("fx", 0)
                if isinstance(val, np.ndarray):
                    val = int(val[-1]) if val.size else 0
                else:
                    val = int(val)
                corrected_obs[:, obs_idx] ^= val & 1
    
    # Apply decoder predictions to flip outcomes when decoder detects errors
    # The decoder predictions indicate when the logical outcome should be flipped
    corrected_obs ^= preds
    
    if not basis_labels or len(basis_labels) < len(patch_order):
        basis_labels = [bracket_map.get(p, "Z") for p in patch_order]

    if metadata.get("_debug_verbose"):
        print(f"[OBS-MAP] patch_order={patch_order} basis={basis_labels}")

    return corrected_obs, patch_to_obs_idx, tuple(basis_labels)


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

    """
    # DEBUG: print tail of circuit operations to ensure joint MPPs are last
    if verbose:
        
        try:
            tail_ops = str(circuit).strip().splitlines()[-80:]
            print("\n[DEBUG] Tail of Stim circuit (last ~80 ops):")
            for ln in tail_ops:
                print("  ", ln)
        except Exception:
            pass
    """

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
                        corr = _parity_from_indices(
                            m_samples,
                            cnot_op.get("m_zz_indices"),
                        )
                        byproduct_correction = corr
                        print(f"[DEBUG] Joint {joint_key}: applying m_ZZ correction from CNOT({control}->{target}), mean={byproduct_correction.mean():.3f}")
                    elif basis == "X":
                        # For XX correlations, flip by m_XX byproduct
                        corr = _parity_from_indices(
                            m_samples,
                            cnot_op.get("m_xx_indices") or ([cnot_op.get("m_xx_logical_idx")] if cnot_op.get("m_xx_logical_idx") is not None else None),
                        )
                        byproduct_correction = corr
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
    # Step 1: Decode DEM
    if verbose:
        metadata["_debug_verbose"] = True

    dem, det_samp, obs_u8, preds = _decode_dem(
        dem, metadata, observable_pairs, shots, seed, verbose
    )
    
    
    # Step 2: Track Pauli frame and extract CNOT byproducts
    pauli_tracker, cnot_metadata, m_samples = _track_pauli_frame(
        circuit, metadata, qc, shots, seed, demo_basis
    )
    
    # Step 3: Apply corrections to observables
    corrected_obs, patch_to_obs_idx, basis_labels_from_bracket = _apply_corrections(
        obs_u8, preds, metadata, bracket_map, pauli_tracker
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

    if metadata.get("_debug_verbose"):
        print(f"[OBS-DEBUG] preds_mean={np.mean(preds, axis=0) if preds.size else []}")
    
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
