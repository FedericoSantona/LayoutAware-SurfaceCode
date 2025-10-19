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



def _fmt_bool(b: bool) -> str:
    return "ON" if b else "OFF"

def _fmt_list(xs) -> str:
    return ", ".join(xs) if xs else "none"

def _extract_requested_bases(cfg) -> list[str]:
    db = getattr(cfg, "demo_basis", None)
    if db is None:
        return []
    if isinstance(db, str):
        return [db.upper()]
    try:
        return [str(x).upper() for x in db]
    except Exception:
        return [str(db).upper()]

def _demo_mode_summary(metadata: dict, cfg) -> list[str]:
    requested = _extract_requested_bases(cfg)
    requested_clean = [b for b in requested if b in ("Z", "X")]
    both_requested = set(requested_clean) == {"Z","X"}
    force_noncomm = bool(getattr(cfg, "demo_force_noncommuting_singles", False))
    joint_only = bool(getattr(cfg, "demo_joint_only", False))

    joint = (metadata or {}).get("joint_demos", {}) or {}
    singles = (metadata or {}).get("demo", {}) or {}

    emitted_joints = sorted({v.get("basis","?") for v in joint.values()})
    emitted_singles = sorted({v.get("basis","?") for v in singles.values()})

    lines = []
    lines.append(f"  Requested bases: {_fmt_list(requested_clean or ['none'])}")
    lines.append(f"  Emitted: {_fmt_list(['joint '+b for b in emitted_joints]) or 'none'}")
    lines.append(f"  Singles: {_fmt_list(emitted_singles)}")
    lines.append(f"  Non-commuting singles override: {_fmt_bool(force_noncomm)}")
    
    # Check for snapshot and warn about non-commuting demos
    snapshot_meta = (metadata or {}).get("final_snapshot", {})
    if snapshot_meta.get("enabled"):
        snap_basis = snapshot_meta["basis"]
        lines.append(f"  Final snapshot: enabled (basis {snap_basis})")
        
        # Warn if non-commuting singles were emitted
        if snap_basis == "Z" and "X" in emitted_singles:
            lines.append("  ⚠ WARNING: X-basis singles were emitted with Z-basis snapshot; X-singles may be randomized.")
        elif snap_basis == "X" and "Z" in emitted_singles:
            lines.append("  ⚠ WARNING: Z-basis singles were emitted with X-basis snapshot; Z-singles may be randomized.")
    
    if both_requested and emitted_singles and not force_noncomm:
        lines.append("  WARNING: Z and X singles were emitted in the same run; these do not commute and the second basis can be randomized by the first.")
    if joint_only and emitted_singles:
        lines.append("  NOTE: cfg.demo_joint_only=True but singles are present (check builder settings).")
    return lines

def _explain_operator_semantics() -> list[str]:
    return [
        "  Final-frame (Heisenberg) semantics:",
        "    • Requested basis determines the requested logical operator σ(q) with σ∈{Z,X}.",
        "    • We actually measure the conjugated logical U† σ(q) U at the end of the circuit.",
        "    • Joint ZZ/XX are one MPP on the conjugated product U†[σ_a(qi)⊗σ_b(qj)]U.",
        "  Stim MPP bit convention:",
        "    • bit=0 ↔ eigenvalue +1, bit=1 ↔ eigenvalue −1;  ⟨O⟩ = 1 − 2·P(bit=1).",
    ]

# --- Helper for sign-aware Heisenberg conjugation of Pauli axis by virtual gates ---
def _conjugate_axis_and_phase(axis: str, virtual_gates: List[str]) -> tuple[str, int]:
    """Heisenberg-conjugate a single-qubit Pauli axis ('Z' or 'X') by a list of virtual gates.
    Walk gates in reverse order (right-to-left): returns (final_axis, phase) with phase in {+1,-1}.
    Rules used:
      H: swaps X<->Z (no sign)
      X: XZX = -Z,  XXX = +X
      Z: ZXZ = -X,  ZZZ = +Z
    """
    axis = axis.upper()
    if axis not in ("Z", "X"):
        raise ValueError(f"axis must be 'Z' or 'X', got {axis}")
    x = (axis == "X")
    z = (axis == "Z")
    phase = +1
    for g in reversed(virtual_gates or []):
        g = str(g).upper()
        if g == "H":
            x, z = z, x  # swap, no sign
        elif g == "X":
            if z:
                phase *= -1  # XZX = -Z
            # XXX = +X (no sign)
        elif g == "Z":
            if x:
                phase *= -1  # ZXZ = -X
            # ZZZ = +Z (no sign)
        else:
            # ignore gates we don't model; extend here if needed
            continue
    return ("X" if x else "Z"), phase

def _get_virtual_gates_for_qubit(qubit: str, virtual_gates_per_qubit: Optional[Dict[str, List[str]]], demo_meta: Optional[Dict[str, Any]]) -> List[str]:
    """Try several places to find the per-qubit virtual gate list for conjugation.
    Priority: explicit arg -> demo_meta['virtual_gates_per_qubit'] -> demo_meta['gate_map'] -> demo_meta['virtual_gates'].
    Returns [] if not found.
    """
    if isinstance(virtual_gates_per_qubit, dict) and qubit in virtual_gates_per_qubit:
        return list(virtual_gates_per_qubit.get(qubit, []))
    if isinstance(demo_meta, dict):
        for key in ("virtual_gates_per_qubit", "gate_map", "virtual_gates"):
            m = demo_meta.get(key, None)
            if isinstance(m, dict) and qubit in m:
                return list(m.get(qubit, []))
    return []


# --- Helper functions for demo readout phase/frame handling ---
def _get_frame_bit(pauli_frame: Optional[Dict[str, Dict[str, Any]]], qubit: str, basis_axis: str) -> int:
    """Return the frame flip bit for a given qubit and basis.
    For singles: Z-basis ← fx; X-basis ← fz.
    basis_axis is 'Z' or 'X'.
    """
    try:
        if not pauli_frame or qubit not in pauli_frame:
            return 0
        key = "fx" if basis_axis == "Z" else "fz"
        v = pauli_frame[qubit].get(key, 0)
        if isinstance(v, np.ndarray):
            # Reduce to a single bit: majority/mean -> {0,1}
            return int(round(float(v.mean()))) & 1
        return int(v) & 1
    except Exception:
        return 0

def _apply_phase_and_frame(bits: np.ndarray, qubit: str, basis_axis: str, entry_meta: Dict[str, Any], pauli_frame: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Compute raw and frame-corrected P1 and expectations, including conjugation phase.
    Returns dict with keys: p1_raw, p1_frame, phase, frame_flip, expect_final.
    """
    p1_raw = float(bits.mean())
    frame_flip = _get_frame_bit(pauli_frame, qubit, basis_axis)
    if frame_flip:
        bits = np.bitwise_xor(bits, 1)
    p1_frame = float(bits.mean())
    phase = int(entry_meta.get("phase", +1))
    phase = +1 if phase >= 0 else -1
    expect_final = phase * (1.0 - 2.0 * p1_frame)
    return {
        "p1_raw": p1_raw,
        "p1_frame": p1_frame,
        "phase": phase,
        "frame_flip": frame_flip,
        "expect_final": expect_final,
    }

def _apply_phase_and_frame_joint(bits: np.ndarray, qa: str, qb: str, basis_axis: str, entry_meta: Dict[str, Any], pauli_frame: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Joint case: frame flip is XOR of per-qubit flips for the basis."""
    p1_raw = float(bits.mean())
    fa = _get_frame_bit(pauli_frame, qa, basis_axis)
    fb = _get_frame_bit(pauli_frame, qb, basis_axis)
    frame_flip = fa ^ fb
    if frame_flip:
        bits = np.bitwise_xor(bits, 1)
    p1_frame = float(bits.mean())
    phase = int(entry_meta.get("phase", +1))
    phase = +1 if phase >= 0 else -1
    expect_final = phase * (1.0 - 2.0 * p1_frame)
    return {
        "p1_raw": p1_raw,
        "p1_frame": p1_frame,
        "phase": phase,
        "frame_flip": frame_flip,
        "expect_final": expect_final,
    }

def _print_demo_preamble(metadata: dict, cfg) -> None:
    print("Demo Mode Summary:")
    for line in _demo_mode_summary(metadata, cfg):
        print(line)
    print()
    print("Operator semantics:")
    for line in _explain_operator_semantics():
        print(line)
    print()


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
        
        print(f"  merge ({parity_type}, {a}, {b}, rounds={rounds}): mean={mean_parity:.5f}")
    
    # CNOT operations
    if cnot_metadata:
        print(f"\nCNOT Operations:")
        for cnot in cnot_metadata:
            control = cnot["control"]
            target = cnot["target"]
            m_zz = cnot["m_zz_mean"]
            m_xx = cnot["m_xx_mean"]
            print(f"  CNOT({control}->{target}): m_ZZ={m_zz:.5f}, m_XX={m_xx:.5f}")
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

    # Warn if observables columns do not match number of logical qubits
    if obs_u8.shape[1] != len(qubit_names):
        print(f"  [WARN] Observables columns ({obs_u8.shape[1]}) != logical qubits in bracket_map ({len(qubit_names)}). Using first {obs_u8.shape[1]} in sorted order.")

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
    pauli_frame: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    joint_demo_bits: Optional[Dict[str, Dict]] = None,
    virtual_gates_per_qubit: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Print Section C: Physics demo readouts with single-qubit marginals and two-qubit correlations."""
    
    print("\n" + "=" * 80)
    print("PHYSICS DEMO READOUTS (FRAME-CONJUGATED OPERATORS)")
    print("=" * 80)

    # --- What is present in this run (singles vs joint, by basis) ---
    try:
        singles_present = sorted({ (demo_meta[k] or {}).get("basis","?")
                                   for k in (demo_meta or {}).keys()
                                   if isinstance(demo_meta.get(k,{}), dict) and (demo_meta[k].get("basis") in ("Z","X")) })
    except Exception:
        singles_present = []
    try:
        joint_present = sorted({ v.get("basis","?")
                                 for v in (joint_demo_bits or {}).values()
                                 if isinstance(v, dict) and (v.get("basis") in ("Z","X")) })
    except Exception:
        joint_present = []
    if singles_present or joint_present:
        print("Present demo bases:")
        print(f"  Singles: {_fmt_list(singles_present)}")
        print(f"  Joint:   {_fmt_list(joint_present)}")
        print()
    
    # Helper to gate post-frame correction for singles to truly single-qubit operators only
    def _can_apply_post_frame(op_str: Optional[str], qubit: str, axis: str) -> bool:
        """Return True iff the reported operator is exactly the single-qubit axis on this qubit (e.g., 'Z(q0)' or 'X(q1)')."""
        if not isinstance(op_str, str):
            return False
        op_str = op_str.strip()
        target = f"{axis}({qubit})"
        return op_str == target

    print("\nSingle-qubit marginals (final-frame):")

    # Warn if both bases are present in the same run (non-commuting singles)
    try:
        _singles_meta = (demo_meta or {})
        _bases_present = sorted({v.get("basis","?") for v in _singles_meta.values() if isinstance(v, dict)})
        if set(_bases_present) == {"Z","X"}:
            print("  WARNING: Z and X singles are present in the same run; these do not commute and whichever comes later can be randomized by the earlier one.")
    except Exception:
        pass

    if demo_z_bits or demo_x_bits:
        # Try to obtain per-qubit virtual gate lists for sign-aware conjugation.
        # Prefer explicit arg; otherwise look inside demo_meta under common keys.
        vg_map_explicit = virtual_gates_per_qubit if isinstance(virtual_gates_per_qubit, dict) else {}
        vg_map_meta = {}
        if isinstance(demo_meta, dict):
            for key in ("virtual_gates_per_qubit", "gate_map", "virtual_gates"):
                v = demo_meta.get(key, None)
                if isinstance(v, dict):
                    vg_map_meta = v
                    break
        all_qs = sorted(set(list(demo_z_bits.keys()) + list(demo_x_bits.keys())))
        for qubit_name in all_qs:
            print(f"  {qubit_name}:")
            # Z-basis single (if present)
            if qubit_name in demo_z_bits:
                z_key = f"{qubit_name}_Z"
                z_entry = dict((demo_meta or {}).get(z_key, {})) if isinstance(demo_meta, dict) else {}
                z_requested = f"Z({qubit_name})"
                # If builder didn't provide a logical_operator/phase, derive via virtual gates.
                if "logical_operator" not in z_entry or "phase" not in z_entry:
                    vg_list = vg_map_explicit.get(qubit_name, []) or vg_map_meta.get(qubit_name, []) or _get_virtual_gates_for_qubit(qubit_name, virtual_gates_per_qubit, demo_meta)
                    final_axis, phase_c = _conjugate_axis_and_phase("Z", vg_list)
                    z_entry.setdefault("logical_operator", f"{final_axis}({qubit_name})")
                    z_entry["phase"] = int(z_entry.get("phase", +1)) * int(phase_c)
                z_logical = z_entry.get("logical_operator", f"Z({qubit_name})")
                # Choose frame bit based on final operator axis if single-qubit; else skip frame correction
                final_axis = None
                if isinstance(z_logical, str) and z_logical.strip() in (f"Z({qubit_name})", f"X({qubit_name})"):
                    final_axis = z_logical[0]
                stats = _apply_phase_and_frame(demo_z_bits[qubit_name], qubit_name, (final_axis or "Z"), z_entry, pauli_frame) if final_axis else {
                    **_apply_phase_and_frame(demo_z_bits[qubit_name], qubit_name, "Z", z_entry, pauli_frame)
                }
                print(f"    Basis Z: requested {z_requested}; measured (final-frame) {z_logical}")
                print(f"      bit=1 ↔ eigenvalue −1")
                axis_tag = "fx" if (final_axis or "Z") == "Z" else "fz"
                print(f"      P(bit=1): raw={stats['p1_raw']:.3f}, frame={stats['p1_frame']:.3f}  |  phase={stats['phase']:+d}, frame_flip({axis_tag})={stats['frame_flip']}")
                print(f"      ⟨O⟩ (phase×frame) = {stats['expect_final']:+.3f}")
            else:
                print(f"    Z-basis: not emitted")

            # X-basis single (if present)
            if qubit_name in demo_x_bits:
                x_key = f"{qubit_name}_X"
                x_entry = dict((demo_meta or {}).get(x_key, {})) if isinstance(demo_meta, dict) else {}
                x_requested = f"X({qubit_name})"
                if "logical_operator" not in x_entry or "phase" not in x_entry:
                    vg_list = vg_map_explicit.get(qubit_name, []) or vg_map_meta.get(qubit_name, []) or _get_virtual_gates_for_qubit(qubit_name, virtual_gates_per_qubit, demo_meta)
                    final_axis, phase_c = _conjugate_axis_and_phase("X", vg_list)
                    x_entry.setdefault("logical_operator", f"{final_axis}({qubit_name})")
                    x_entry["phase"] = int(x_entry.get("phase", +1)) * int(phase_c)
                x_logical = x_entry.get("logical_operator", f"X({qubit_name})")
                final_axis = None
                if isinstance(x_logical, str) and x_logical.strip() in (f"Z({qubit_name})", f"X({qubit_name})"):
                    final_axis = x_logical[0]
                stats = _apply_phase_and_frame(demo_x_bits[qubit_name], qubit_name, (final_axis or "X"), x_entry, pauli_frame) if final_axis else {
                    **_apply_phase_and_frame(demo_x_bits[qubit_name], qubit_name, "X", x_entry, pauli_frame)
                }
                print(f"    Basis X: requested {x_requested}; measured (final-frame) {x_logical}")
                print(f"      bit=1 ↔ eigenvalue −1")
                axis_tag = "fx" if (final_axis or "X") == "Z" else "fz"
                print(f"      P(bit=1): raw={stats['p1_raw']:.3f}, frame={stats['p1_frame']:.3f}  |  phase={stats['phase']:+d}, frame_flip({axis_tag})={stats['frame_flip']}")
                print(f"      ⟨O⟩ (phase×frame) = {stats['expect_final']:+.3f}")
            else:
                print(f"    X-basis: not emitted")
    else:
        print("  No demo readouts available")
    
    # Two-qubit correlations
    if joint_demo_bits:
        print("\nTwo-qubit correlations (joint MPPs; final-frame operators):")

        # Group joint demos by pair
        pair_demos = {}
        for joint_key, demo_data in joint_demo_bits.items():
            pair = demo_data["pair"]
            basis = demo_data["basis"]
            pair_key = f"{pair[0]},{pair[1]}"
            if pair_key not in pair_demos:
                pair_demos[pair_key] = {}
            pair_demos[pair_key][basis] = demo_data

        for pair_key, demos in pair_demos.items():
            zz_data = demos.get("Z")
            xx_data = demos.get("X")
            qa, qb = pair_key.split(',', 1)
            print(f"\n  ({qa},{qb}):  bases present → {_fmt_list(sorted(demos.keys()))}")

            if zz_data is not None:
                zz_bits = zz_data["bits"]
                qa, qb = zz_data.get("pair", ("?","?"))
                # If corrected bits provided (raw_bits+frame_flip), avoid re-applying frame here
                if "raw_bits" in zz_data and "frame_flip" in zz_data:
                    p1_raw = float(np.asarray(zz_data["raw_bits"], dtype=np.uint8).mean())
                    p1_frame = float(np.asarray(zz_bits, dtype=np.uint8).mean())
                    phase = int(zz_data.get("phase", +1))
                    phase = +1 if phase >= 0 else -1
                    ff_val = zz_data.get("frame_flip", 0)
                    if isinstance(ff_val, np.ndarray):
                        frame_flip = int(round(float(ff_val.mean()))) & 1
                    else:
                        frame_flip = int(ff_val) & 1
                    stats = {
                        "p1_raw": p1_raw,
                        "p1_frame": p1_frame,
                        "phase": phase,
                        "frame_flip": frame_flip,
                        "expect_final": phase * (1.0 - 2.0 * p1_frame),
                    }
                else:
                    stats = _apply_phase_and_frame_joint(zz_bits, qa, qb, "Z", zz_data, pauli_frame)
                zz_operator = zz_data.get("logical_operator", "Z⊗Z")
                zz_physical = zz_data.get("physical_realization") or "unknown"
                zz_requested = f"Z({qa})⊗Z({qb})"
                print(f"    [Basis Z] ⟨Z⊗Z⟩ = {stats['expect_final']:+.3f}")
                print(f"       requested: {zz_requested}")
                print(f"       measured (final-frame): {zz_operator}")
                print(f"       physical targets: {zz_physical}")
                print(f"       bit=1 ↔ eigenvalue −1")
                print(f"       P(bit=1): raw={stats['p1_raw']:.3f}, frame={stats['p1_frame']:.3f}  |  phase={stats['phase']:+d}, frame_flip(fx⊕fx)={stats['frame_flip']}")
                # CI on expectation derived from frame-corrected p1
                p1 = stats["p1_frame"]
                p1_count = int(round(p1 * shots))
                p1_ci = wilson_rate_ci(p1_count, shots)
                zz_ci = (1.0 - 2.0 * p1_ci[1], 1.0 - 2.0 * p1_ci[0])
                print(f"       CI on ⟨O⟩: [{zz_ci[0]:.3f}, {zz_ci[1]:.3f}]")

            if xx_data is not None:
                xx_bits = xx_data["bits"]
                qa, qb = xx_data.get("pair", ("?","?"))
                if "raw_bits" in xx_data and "frame_flip" in xx_data:
                    p1_raw = float(np.asarray(xx_data["raw_bits"], dtype=np.uint8).mean())
                    p1_frame = float(np.asarray(xx_bits, dtype=np.uint8).mean())
                    phase = int(xx_data.get("phase", +1))
                    phase = +1 if phase >= 0 else -1
                    ff_val = xx_data.get("frame_flip", 0)
                    if isinstance(ff_val, np.ndarray):
                        frame_flip = int(round(float(ff_val.mean()))) & 1
                    else:
                        frame_flip = int(ff_val) & 1
                    stats = {
                        "p1_raw": p1_raw,
                        "p1_frame": p1_frame,
                        "phase": phase,
                        "frame_flip": frame_flip,
                        "expect_final": phase * (1.0 - 2.0 * p1_frame),
                    }
                else:
                    stats = _apply_phase_and_frame_joint(xx_bits, qa, qb, "X", xx_data, pauli_frame)
                xx_operator = xx_data.get("logical_operator", "X⊗X")
                xx_physical = xx_data.get("physical_realization") or "unknown"
                xx_requested = f"X({qa})⊗X({qb})"
                print(f"    [Basis X] ⟨X⊗X⟩ = {stats['expect_final']:+.3f}")
                print(f"       requested: {xx_requested}")
                print(f"       measured (final-frame): {xx_operator}")
                print(f"       physical targets: {xx_physical}")
                print(f"       bit=1 ↔ eigenvalue −1")
                print(f"       P(bit=1): raw={stats['p1_raw']:.3f}, frame={stats['p1_frame']:.3f}  |  phase={stats['phase']:+d}, frame_flip(fz⊕fz)={stats['frame_flip']}")
                p1 = stats["p1_frame"]
                p1_count = int(round(p1 * shots))
                p1_ci = wilson_rate_ci(p1_count, shots)
                xx_ci = (1.0 - 2.0 * p1_ci[1], 1.0 - 2.0 * p1_ci[0])
                print(f"       CI on ⟨O⟩: [{xx_ci[0]:.3f}, {xx_ci[1]:.3f}]")

            if zz_data is not None and xx_data is not None:
                fidelity_bound = 0.5 * (stats['expect_final'] + stats['expect_final'])
                # Actually, use the respective expectations:
                zz_stats = _apply_phase_and_frame_joint(zz_data["bits"], qa, qb, "Z", zz_data, pauli_frame)
                xx_stats = _apply_phase_and_frame_joint(xx_data["bits"], qa, qb, "X", xx_data, pauli_frame)
                fidelity_bound = 0.5 * (zz_stats['expect_final'] + xx_stats['expect_final'])
                print(f"    Bell fidelity bound F ≥ 0.5(⟨ZZ⟩+⟨XX⟩) = {fidelity_bound:.3f}")
    elif correlation_pairs and (demo_z_bits or demo_x_bits):
        print("Two-qubit correlations (joint MPPs, measured first):")
        
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
        elif has_x_measurements:
            # Only X⊗X correlations available
            correlations = compute_two_qubit_correlations({}, demo_x_bits, correlation_pairs, shots)
            
            for pair_key, corr_data in correlations.items():
                q1, q2 = pair_key.split(',')
                xx_corr = corr_data["xx_correlator"]
                xx_ci = corr_data["xx_ci"]
                
                print(f"\n  ({q1},{q2}): ⟨X⊗X⟩ = {xx_corr:+.3f}  (CI on parity-0: [{xx_ci[0]:.3f}, {xx_ci[1]:.3f}])")
        elif has_z_measurements:
            # Only Z⊗Z correlations available
            correlations = compute_two_qubit_correlations(demo_z_bits, {}, correlation_pairs, shots)
            
            for pair_key, corr_data in correlations.items():
                q1, q2 = pair_key.split(',')
                zz_corr = corr_data["zz_correlator"]
                zz_ci = corr_data["zz_ci"]
                
                print(f"\n  ({q1},{q2}): ⟨Z⊗Z⟩ = {zz_corr:+.3f}  (CI on parity-0: [{zz_ci[0]:.3f}, {zz_ci[1]:.3f}])")
    else:
        print("\nTwo-qubit correlations:")
        print("  No correlation pairs or demo readouts available for entanglement verification")

    print("="*80)
    print("PAULI-FRAME NOTES")
    print("="*80)
    post_frame_applied = True
    print("  Post-frame flips applied in reporting: YES")
    print("  Policy:")
    print("    • Singles: flip Z by fx[q], flip X by fz[q].")
    print("    • Joint:   flip Z⊗Z by fx[qa]⊕fx[qb], X⊗X by fz[qa]⊕fz[qb].")
    if pauli_frame:
        print("  Frame bits used:")
        for q in sorted(pauli_frame.keys()):
            fx = pauli_frame[q].get("fx", 0)
            fz = pauli_frame[q].get("fz", 0)
            if isinstance(fx, np.ndarray): fx = int(round(float(fx.mean()))) & 1
            else: fx = int(fx) & 1
            if isinstance(fz, np.ndarray): fz = int(round(float(fz.mean()))) & 1
            else: fz = int(fz) & 1
            print(f"    {q}: fx={fx}, fz={fz}")
    # Summaries of what was emitted
    print(f"  Singles emitted: {_fmt_list(singles_present) if singles_present else 'none'}")
    print(f"  Joint emitted:   {_fmt_list(joint_present) if joint_present else 'none'}")
    if set(singles_present) == {"Z", "X"}:
        print("  NOTE: Z and X singles were emitted together in one run; they do not commute and whichever is measured second can randomize the first.")
    print()


def print_final_state_distribution(
    snapshot_meta: Dict[str, Any],
    snapshot_bits: Dict[str, np.ndarray],
    pauli_frame: Optional[Dict[str, Dict[str, Any]]],
    shots: int,
    apply_frame_correction: bool = True,
) -> None:
    """Print Section C2: Final logical state distribution (computational basis histogram)."""
    
    if not snapshot_meta.get("enabled") or not snapshot_bits:
        return
    
    print("\n" + "=" * 80)
    print("FINAL LOGICAL STATE DISTRIBUTION")
    print("=" * 80)
    
    basis = snapshot_meta["basis"]
    order = snapshot_meta["order"]
    k = len(order)
    
    print(f"Order (MSB→LSB): {' '.join(order)}")
    print(f"Frame correction: {'ON' if apply_frame_correction else 'OFF'} (using f{'x' if basis=='Z' else 'z'} per qubit)")
    print(f"Measured operators (final-frame): {snapshot_meta.get('logical_ops', [])}")
    print()
    
    # Build bitstrings from snapshot measurements
    bits_array = np.zeros((shots, k), dtype=np.uint8)
    for i, qname in enumerate(order):
        col = snapshot_bits.get(qname)
        if col is None:
            continue
        bits_array[:, i] = col
        
        # Apply frame correction
        if apply_frame_correction and pauli_frame and qname in pauli_frame:
            frame_key = "fx" if basis == "Z" else "fz"
            flip = pauli_frame[qname].get(frame_key, 0)
            if isinstance(flip, np.ndarray):
                flip = int(round(float(flip.mean()))) & 1
            else:
                flip = int(flip) & 1
            if flip:
                bits_array[:, i] ^= 1
    
    # Assemble bitstrings and count
    from collections import Counter
    bitstrings = [''.join(str(b) for b in row) for row in bits_array]
    counts = Counter(bitstrings)
    
    # Generate all possible states and their probabilities
    total = sum(counts.values())
    all_possible_states = []
    for i in range(2**k):
        state = format(i, f'0{k}b')
        count = counts.get(state, 0)
        prob = count / total
        all_possible_states.append((state, count, prob))
    
    # Sort by probability (descending)
    sorted_states = sorted(all_possible_states, key=lambda x: x[2], reverse=True)
    
    # Convert bitstrings to proper basis labels
    if basis == "X":
        # For X-basis: 0 → |+⟩, 1 → |-⟩
        state_labels = {state: ''.join('+' if b == '0' else '-' for b in state) for state, count, prob in sorted_states}
        print(f"{'State':<{k+2}} {'Shots':<12} Prob")
        print("-" * 40)
        for state, count, prob in sorted_states:
            label = state_labels[state]
            print(f"{label:<{k+2}} {count:<12,} {prob:.4f}")
    else:
        # For Z-basis: 0 → |0⟩, 1 → |1⟩
        print(f"{'State':<{k+2}} {'Shots':<12} Prob")
        print("-" * 40)
        for state, count, prob in sorted_states:
            print(f"{state:<{k+2}} {count:<12,} {prob:.4f}")
    
    print()


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
    if not gate_map:
        print("  (none reported)")
    
    # Final Pauli frame state
    if pauli_frame:
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
            print(f"  CNOT({control}->{target}): m_ZZ={m_zz:.5f}, m_XX={m_xx:.5f}")
    else:
        print("\nCNOT parity bits used for frame updates:")
        print("  (none)")


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
        corrected_mean = float(corrected_obs[:, i].mean())
        corrected_p0 = (1.0 - corrected_mean) * 100.0
        corrected_p1 = corrected_mean * 100.0
        
        print(f"\n{qubit_name} (basis {basis}):")
        print(f"  Raw: |0⟩ = {raw_p0:6.2f}%, |1⟩ = {raw_p1:6.2f}%")
        print(f"  Expected (flip={expected_flip}): |0⟩ = {expected_p0:6.2f}%, |1⟩ = {expected_p1:6.2f}%")
        print(f"  Decoded: |0⟩ = {corrected_p0:6.2f}%, |1⟩ = {corrected_p1:6.2f}%")


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
    snapshot_bits: Optional[Dict[str, np.ndarray]] = None,
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
    all_demo_names = sorted(set(demo_z_bits.keys()) | set(demo_x_bits.keys()))
    for qubit_name in all_demo_names:
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
    
    # Final state distribution section (if present)
    final_state_dist = {}
    if snapshot_bits:
        # Get snapshot metadata from args (we'll need to pass it through)
        snapshot_meta = getattr(args, 'snapshot_meta', {})
        if snapshot_meta.get("enabled"):
            basis = snapshot_meta["basis"]
            order = snapshot_meta["order"]
            k = len(order)
            
            # Build bitstrings from snapshot measurements (similar to print function)
            bits_array = np.zeros((shots, k), dtype=np.uint8)
            for i, qname in enumerate(order):
                col = snapshot_bits.get(qname)
                if col is None:
                    continue
                bits_array[:, i] = col
                
                # Apply frame correction
                if pauli_frame and qname in pauli_frame:
                    frame_key = "fx" if basis == "Z" else "fz"
                    flip = pauli_frame[qname].get(frame_key, 0)
                    if isinstance(flip, np.ndarray):
                        flip = int(round(float(flip.mean()))) & 1
                    else:
                        flip = int(flip) & 1
                    if flip:
                        bits_array[:, i] ^= 1
            
            # Assemble bitstrings and count
            from collections import Counter
            bitstrings = [''.join(str(b) for b in row) for row in bits_array]
            counts = Counter(bitstrings)
            
            total = sum(counts.values())
            
            # Generate all possible states with proper labels
            all_states = {}
            for i in range(2**k):
                state = format(i, f'0{k}b')
                count = counts.get(state, 0)
                prob = count / total
                
                if basis == "X":
                    # For X-basis: 0 → |+⟩, 1 → |-⟩
                    label = ''.join('+' if b == '0' else '-' for b in state)
                else:
                    # For Z-basis: keep as 0/1
                    label = state
                
                all_states[label] = float(prob)
            
            final_state_dist = {
                "enabled": True,
                "basis": basis,
                "order": order,
                "distribution": all_states,
            }
    
    return {
        "header": header,
        "per_qubit_results": per_qubit_results,
        "physics_demo": physics_demo,
        "final_state_distribution": final_state_dist,
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
