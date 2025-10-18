"""Structured reporting module for quantum error correction simulations.

This module provides clean, organized reporting functions that follow the detailed
report structure specified in the plan. It includes sections for header information,
per-qubit logical outcomes, physics demo readouts, Pauli-frame audit, and optional
debug details.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .utils import wilson_rate_ci, compute_two_qubit_correlations


def print_header(
    args: Any,
    model: Any,
    dem: Any,
    metadata: Dict[str, Any],
    merge_bits: Dict[Tuple[str, str, str, int], np.ndarray],
    cnot_metadata: List[Dict[str, Any]],
    stim_rounds: int,
    shots: int,
) -> None:
    """Print Section A: Header with context, geometry, noise, DEM summary, and surgery timeline."""
    
    print("=" * 80)
    print("QUANTUM ERROR CORRECTION SIMULATION REPORT")
    print("=" * 80)
    
    # Scenario information
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Scenario: {args.benchmark} benchmark")
    print(f"Timestamp: {timestamp}")
    print(f"Seed: {args.seed}")
    
    # Code geometry
    print(f"\nCode Geometry:")
    print(f"  Distance: d={args.distance}")
    print(f"  Physical qubits: n={model.code.n}")
    print(f"  Measurement rounds: {stim_rounds}")
    
    # Stabilizer information
    print(f"  Z stabilizers: {len(model.z_stabilizers)}")
    print(f"  X stabilizers: {len(model.x_stabilizers)}")
    print(f"  Logical operators: Z_L weight={len([x for x in str(model.logical_z) if x != 'I'])}, X_L weight={len([x for x in str(model.logical_x) if x != 'I'])}")
    
    # Noise configuration
    print(f"\nNoise Configuration:")
    print(f"  X error probability: p_x = {args.px}")
    print(f"  Z error probability: p_z = {args.pz}")
    print(f"  Shots: {shots:,}")
    
    # DEM summary
    print(f"\nDetector Error Model Summary:")
    print(f"  Detectors: {dem.num_detectors}")
    print(f"  Observables: {dem.num_observables}")
    
    # Surgery timeline summary
    print(f"\nSurgery Timeline:")
    merge_windows = metadata.get("merge_windows", [])
    for window in merge_windows:
        window_id = window.get("id")
        parity_type = window.get("parity_type", "unknown")
        a = window.get("a", "unknown")
        b = window.get("b", "unknown")
        rounds = window.get("rounds", 0)
        
        # Get mean parity from merge_bits
        key = (parity_type, a, b, window_id)
        mean_parity = float(merge_bits[key].mean()) if key in merge_bits else 0.0
        
        print(f"  merge ({parity_type}, {a}, {b}, rounds={rounds}): mean={mean_parity:.3f}")
    
    # CNOT operations
    if cnot_metadata:
        print(f"\nCNOT Operations:")
        for cnot in cnot_metadata:
            control = cnot["control"]
            target = cnot["target"]
            m_zz = cnot["m_zz_mean"]
            m_xx = cnot["m_xx_mean"]
            print(f"  CNOT({control}->{target}): m_ZZ={m_zz:.3f}, m_XX={m_xx:.3f}")
            print(f"    Applied Pauli-frame updates:")
            print(f"      fz[{target}] ^= m_ZZ")
            print(f"      fx[{control}] ^= m_XX")


def print_per_qubit_results(
    args: Any,
    bracket_map: Dict[str, str],
    corrected_obs: np.ndarray,
    obs_u8: np.ndarray,
    preds: np.ndarray,
    shots: int,
) -> None:
    """Print Section B: Per-qubit logical outcomes with raw and post-correction distributions and LER with CI."""
    
    print("\n" + "=" * 80)
    print("PER-QUBIT LOGICAL OUTCOMES")
    print("=" * 80)
    
    # Get sorted qubit names from bracket_map
    qubit_names = sorted(bracket_map.keys())
    
    # Compute per-qubit logical error rates and distributions
    for i, qubit_name in enumerate(qubit_names):
        if i >= obs_u8.shape[1]:
            break
            
        basis = bracket_map[qubit_name]
        
        # LER = mismatch between observables and predictions
        errors = np.bitwise_xor(obs_u8[:, i], preds[:, i])
        error_count = int(np.sum(errors))
        ler = error_count / shots
        ler_ci = wilson_rate_ci(error_count, shots)
        
        # Raw distribution (pre-correction)
        raw_mean = float(obs_u8[:, i].mean())
        raw_p0 = (1.0 - raw_mean) * 100.0
        raw_p1 = raw_mean * 100.0
        
        # Post-correction distribution (after applying decoder's Pauli frame)
        corrected_mean = float(corrected_obs[:, i].mean())
        corrected_p0 = (1.0 - corrected_mean) * 100.0
        corrected_p1 = corrected_mean * 100.0
        
        print(f"\n{qubit_name} (basis {basis}):")
        print(f"  Raw distribution:           |0⟩ = {raw_p0:6.2f}% |1⟩ = {raw_p1:6.2f}%")
        print(f"  Post-correction distribution:|0⟩ = {corrected_p0:6.2f}% |1⟩ = {corrected_p1:6.2f}%")
        print(f"  Logical error rate: {ler:.3e} (95% CI: [{ler_ci[0]:.3e}, {ler_ci[1]:.3e}])")


def print_physics_demo(
    demo_meta: Dict[str, Any],
    demo_z_bits: Dict[str, np.ndarray],
    demo_x_bits: Dict[str, np.ndarray],
    correlation_pairs: List[Tuple[str, str]],
    shots: int,
) -> None:
    """Print Section C: Physics demo readouts with single-qubit marginals and two-qubit correlations."""
    
    print("\n" + "=" * 80)
    print("PHYSICS DEMO READOUTS (END-ONLY MPPs)")
    print("=" * 80)
    
    # Single-qubit marginals
    print("\nSingle-qubit marginals:")
    if demo_z_bits or demo_x_bits:
        for qubit_name in sorted(set(list(demo_z_bits.keys()) + list(demo_x_bits.keys()))):
            z_prob = float(demo_z_bits[qubit_name].mean()) if qubit_name in demo_z_bits else None
            x_prob = float(demo_x_bits[qubit_name].mean()) if qubit_name in demo_x_bits else None
            
            if z_prob is not None and x_prob is not None:
                print(f"  {qubit_name}: P_Z(|1⟩) = {z_prob:.3f}  P_X(|1⟩) = {x_prob:.3f}")
            elif x_prob is not None:
                print(f"  {qubit_name}: P_X(|1⟩) = {x_prob:.3f}  (Z-basis measurement not available)")
            elif z_prob is not None:
                print(f"  {qubit_name}: P_Z(|1⟩) = {z_prob:.3f}  (X-basis measurement not available)")
    else:
        print("  No demo readouts available")
    
    # Two-qubit correlations
    if correlation_pairs and (demo_z_bits or demo_x_bits):
        print("\nTwo-qubit correlations:")
        
        # Check if we have both Z and X measurements for proper Bell state verification
        has_z_measurements = bool(demo_z_bits)
        has_x_measurements = bool(demo_x_bits)
        
        if has_z_measurements and has_x_measurements:
            # Full Bell state verification possible
            correlations = compute_two_qubit_correlations(demo_z_bits, demo_x_bits, correlation_pairs, shots)
            
            for pair_key, corr_data in correlations.items():
                q1, q2 = pair_key.split(',')
                zz_corr = corr_data["zz_correlator"]
                xx_corr = corr_data["xx_correlator"]
                zz_ci = corr_data["zz_ci"]
                xx_ci = corr_data["xx_ci"]
                fidelity_bound = corr_data["fidelity_bound"]
                
                print(f"\n  ({q1},{q2}): ⟨Z⊗Z⟩ = {zz_corr:+.3f}  (CI on parity-0: [{zz_ci[0]:.3f}, {zz_ci[1]:.3f}])")
                print(f"            ⟨X⊗X⟩ = {xx_corr:+.3f}  (CI on parity-0: [{xx_ci[0]:.3f}, {xx_ci[1]:.3f}])")
                print(f"            Bell fidelity bound F ≥ 0.5(⟨ZZ⟩+⟨XX⟩) = {fidelity_bound:.3f}")
        else:
            # Partial verification - only X⊗X correlations available
            print("  Note: Only X-basis measurements available. Full Bell state verification requires both Z⊗Z and X⊗X correlations.")
            if has_x_measurements:
                # Compute X⊗X correlations only
                correlations = compute_two_qubit_correlations({}, demo_x_bits, correlation_pairs, shots)
                
                for pair_key, corr_data in correlations.items():
                    q1, q2 = pair_key.split(',')
                    xx_corr = corr_data["xx_correlator"]
                    xx_ci = corr_data["xx_ci"]
                    
                    print(f"\n  ({q1},{q2}): ⟨X⊗X⟩ = {xx_corr:+.3f}  (CI on parity-0: [{xx_ci[0]:.3f}, {xx_ci[1]:.3f}])")
                    print(f"            Note: Z⊗Z correlation not available - cannot compute full Bell fidelity bound")
    else:
        print("\nTwo-qubit correlations:")
        print("  No correlation pairs or demo readouts available for entanglement verification")


def print_pauli_frame_audit(
    gate_map: Dict[str, List[str]],
    pauli_frame: Dict[str, Dict[str, np.ndarray]],
    cnot_metadata: List[Dict[str, Any]],
) -> None:
    """Print Section D: Pauli-frame audit showing virtual gates applied and final frame state."""
    
    print("\n" + "=" * 80)
    print("PAULI-FRAME AUDIT")
    print("=" * 80)
    
    # Virtual gates per qubit
    print("\nApplied virtual gates per qubit:")
    for qubit_name in sorted(gate_map.keys()):
        gates = gate_map[qubit_name]
        gate_str = ' '.join(gates) if gates else '(none)'
        print(f"  {qubit_name}: {gate_str}")
    
    # Final Pauli frame state
    print("\nFinal Pauli frame state:")
    for qubit_name in sorted(pauli_frame.keys()):
        frame = pauli_frame[qubit_name]
        fx_mean = float(frame["fx"].mean()) if isinstance(frame["fx"], np.ndarray) else frame["fx"]
        fz_mean = float(frame["fz"].mean()) if isinstance(frame["fz"], np.ndarray) else frame["fz"]
        print(f"  {qubit_name}: fx={fx_mean:.3f}, fz={fz_mean:.3f}")
    
    # CNOT parity bits used for frame updates
    if cnot_metadata:
        print("\nCNOT parity bits used for frame updates:")
        for cnot in cnot_metadata:
            control = cnot["control"]
            target = cnot["target"]
            m_zz = cnot["m_zz_mean"]
            m_xx = cnot["m_xx_mean"]
            print(f"  CNOT({control}->{target}): m_ZZ={m_zz:.3f}, m_XX={m_xx:.3f}")


def print_debug_details(
    args: Any,
    basis_labels: Tuple[str, ...],
    obs_u8: np.ndarray,
    preds: np.ndarray,
    corrected_obs: np.ndarray,
    expected_flips: List[int],
) -> None:
    """Print Section E: Debug details including raw/expected/decoded distributions (verbose only)."""
    
    print("\n" + "=" * 80)
    print("DEBUG DETAILS (VERBOSE)")
    print("=" * 80)
    
    print("\nRaw vs expected vs decoded distributions per qubit:")
    for i, basis in enumerate(basis_labels):
        if i >= obs_u8.shape[1]:
            break
            
        qubit_name = f"Q{i+1}"
        expected_flip = expected_flips[i] if i < len(expected_flips) else 0
        
        # Raw distributions
        raw_mean = float(obs_u8[:, i].mean())
        raw_p0 = (1.0 - raw_mean) * 100.0
        raw_p1 = raw_mean * 100.0
        
        # Expected distributions (with expected flip)
        expected_obs = np.bitwise_xor(obs_u8[:, i], expected_flip)
        expected_mean = float(expected_obs.mean())
        expected_p0 = (1.0 - expected_mean) * 100.0
        expected_p1 = expected_mean * 100.0
        
        # Decoded distributions
        decoded_mean = float(preds[:, i].mean())
        decoded_p0 = (1.0 - decoded_mean) * 100.0
        decoded_p1 = decoded_mean * 100.0
        
        print(f"\n{qubit_name} (basis {basis}):")
        print(f"  Raw: |0⟩ = {raw_p0:6.2f}%, |1⟩ = {raw_p1:6.2f}%")
        print(f"  Expected (flip={expected_flip}): |0⟩ = {expected_p0:6.2f}%, |1⟩ = {expected_p1:6.2f}%")
        print(f"  Decoded: |0⟩ = {decoded_p0:6.2f}%, |1⟩ = {decoded_p1:6.2f}%")


def generate_detailed_json(
    args: Any,
    model: Any,
    metadata: Dict[str, Any],
    merge_bits: Dict[Tuple[str, str, str, int], np.ndarray],
    cnot_metadata: List[Dict[str, Any]],
    bracket_map: Dict[str, str],
    corrected_obs: np.ndarray,
    obs_u8: np.ndarray,
    preds: np.ndarray,
    demo_z_bits: Dict[str, np.ndarray],
    demo_x_bits: Dict[str, np.ndarray],
    correlations: Dict[str, Dict[str, float]],
    gate_map: Dict[str, List[str]],
    pauli_frame: Dict[str, Dict[str, np.ndarray]],
    shots: int,
    stim_rounds: int,
) -> Dict[str, Any]:
    """Generate JSON artifact mirroring all printed sections."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Header section
    header = {
        "scenario": args.benchmark,
        "timestamp": timestamp,
        "seed": args.seed,
        "geometry": {
            "distance": args.distance,
            "physical_qubits": model.code.n,
            "measurement_rounds": stim_rounds,
            "z_stabilizers": len(model.z_stabilizers),
            "x_stabilizers": len(model.x_stabilizers),
            "logical_z_weight": len([x for x in str(model.logical_z) if x != 'I']),
            "logical_x_weight": len([x for x in str(model.logical_x) if x != 'I']),
        },
        "noise_config": {
            "p_x": args.px,
            "p_z": args.pz,
            "shots": shots,
        },
        "dem_summary": {
            "detectors": metadata.get("detector_count", "unknown"),
            "observables": metadata.get("observable_count", "unknown"),
        },
        "surgery_timeline": [
            {
                "id": window.get("id"),
                "type": window.get("parity_type"),
                "a": window.get("a"),
                "b": window.get("b"),
                "rounds": window.get("rounds", 0),
                "mean_parity": float(merge_bits[(window.get("parity_type"), window.get("a"), window.get("b"), window.get("id"))].mean()) if (window.get("parity_type"), window.get("a"), window.get("b"), window.get("id")) in merge_bits else 0.0,
            }
            for window in metadata.get("merge_windows", [])
        ],
        "cnot_operations": cnot_metadata,
    }
    
    # Per-qubit results section
    per_qubit_results = []
    qubit_names = sorted(bracket_map.keys())
    
    for i, qubit_name in enumerate(qubit_names):
        if i >= obs_u8.shape[1]:
            break
            
        basis = bracket_map[qubit_name]
        
        # LER = mismatch between observables and predictions
        errors = np.bitwise_xor(obs_u8[:, i], preds[:, i])
        error_count = int(np.sum(errors))
        ler = error_count / shots
        ler_ci = wilson_rate_ci(error_count, shots)
        
        # Raw distribution (pre-correction)
        raw_mean = float(obs_u8[:, i].mean())
        
        # Post-correction distribution (after applying decoder's Pauli frame)
        corrected_mean = float(corrected_obs[:, i].mean())
        
        per_qubit_results.append({
            "qubit": qubit_name,
            "basis": basis,
            "raw_distribution": {
                "p0": 1.0 - raw_mean,
                "p1": raw_mean,
            },
            "post_correction_distribution": {
                "p0": 1.0 - corrected_mean,
                "p1": corrected_mean,
            },
            "logical_error_rate": ler,
            "logical_error_rate_ci": {
                "lower": ler_ci[0],
                "upper": ler_ci[1],
            },
        })
    
    # Physics demo section
    physics_demo = {
        "single_qubit_marginals": {},
        "two_qubit_correlations": {},
    }
    
    # Single-qubit marginals
    for qubit_name in sorted(demo_z_bits.keys()):
        if qubit_name in demo_z_bits:
            z_mean = float(demo_z_bits[qubit_name].mean())
            physics_demo["single_qubit_marginals"][f"{qubit_name}_z"] = {
                "p0": 1.0 - z_mean,
                "p1": z_mean,
            }
        
        if qubit_name in demo_x_bits:
            x_mean = float(demo_x_bits[qubit_name].mean())
            physics_demo["single_qubit_marginals"][f"{qubit_name}_x"] = {
                "p0": 1.0 - x_mean,
                "p1": x_mean,
            }
    
    # Two-qubit correlations
    for pair_key, corr_data in correlations.items():
        physics_demo["two_qubit_correlations"][pair_key] = {
            "zz_correlator": corr_data["zz_correlator"],
            "xx_correlator": corr_data["xx_correlator"],
            "zz_ci": {"lower": corr_data["zz_ci"][0], "upper": corr_data["zz_ci"][1]},
            "xx_ci": {"lower": corr_data["xx_ci"][0], "upper": corr_data["xx_ci"][1]},
            "fidelity_bound": corr_data["fidelity_bound"],
        }
    
    # Pauli-frame audit section
    pauli_frame_audit = {
        "virtual_gates": {qubit: gates for qubit, gates in gate_map.items()},
        "final_frame_state": {},
        "cnot_parity_bits": cnot_metadata,
    }
    
    for qubit_name in sorted(pauli_frame.keys()):
        frame = pauli_frame[qubit_name]
        fx_mean = float(frame["fx"].mean()) if isinstance(frame["fx"], np.ndarray) else frame["fx"]
        fz_mean = float(frame["fz"].mean()) if isinstance(frame["fz"], np.ndarray) else frame["fz"]
        pauli_frame_audit["final_frame_state"][qubit_name] = {
            "fx": fx_mean,
            "fz": fz_mean,
        }
    
    return {
        "header": header,
        "per_qubit_results": per_qubit_results,
        "physics_demo": physics_demo,
        "pauli_frame_audit": pauli_frame_audit,
    }


def save_detailed_json(
    json_data: Dict[str, Any],
    args: Any,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save detailed JSON report to file."""
    
    if output_dir is None:
        output_dir = Path("output")
    
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.benchmark}_{timestamp}_detailed_report.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=2, sort_keys=True)
    
    return filepath
