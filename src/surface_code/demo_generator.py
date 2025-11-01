"""Demo measurement generation.

This module handles end-of-circuit demo measurements, including joint correlators,
single-qubit demos, and final computational-basis snapshots.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import stim

from .builder_state import BuilderState
from .builder_utils import _mpp_targets_from_pauli
from .layout import Layout
from .pauli import Pauli, conjugate_through_circuit, PauliTracker


class DemoGenerator:
    """Handles end-of-circuit demo measurement generation."""
    
    def __init__(self, layout: Layout, builder):
        """Initialize demo generator.
        
        Args:
            layout: Layout instance
            builder: GlobalStimBuilder instance
        """
        self.layout = layout
        self.builder = builder
    
    def generate_demos(
        self,
        circuit: stim.Circuit,
        cfg,
        state: BuilderState,
        bracket_map: Dict[str, str],
        qiskit_circuit: Optional[object],
    ) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
        """Generate demo measurements.
        
        Args:
            circuit: stim.Circuit to append operations to
            cfg: PhenomenologicalStimConfig instance
            state: BuilderState instance
            bracket_map: Map from patch names to basis
            qiskit_circuit: Optional Qiskit circuit for conjugation
            
        Returns:
            Tuple of (demo_info, joint_demo_info, snapshot_info)
        """
        demo_info: Dict[str, object] = {}
        joint_demo_info: Dict[str, object] = {}
        snapshot_info = {"enabled": False}
        
        # Try to read demo basis from cfg; treat any invalid as disabled
        demo_basis = None
        try:
            db = getattr(cfg, "demo_basis", None)
            if db is not None:
                demo_basis = db
        except Exception:
            demo_basis = None
        
        # Normalize demo_basis to list format
        demo_bases = []
        if demo_basis is not None:
            if isinstance(demo_basis, list):
                demo_bases = demo_basis
            else:
                demo_bases = [demo_basis]
        
        if not demo_bases or qiskit_circuit is None:
            return demo_info, joint_demo_info, snapshot_info
        
        # Prepare mapping between logical names and qiskit indices
        name_to_idx: Dict[str, int] = {}
        idx_to_name: Dict[int, str] = {}
        n_logical = qiskit_circuit.num_qubits
        for qi in range(n_logical):
            name = f"q{qi}"
            name_to_idx[name] = qi
            idx_to_name[qi] = name
        
        # Correlation pairs from compiled CNOT operations (fallback to first two logicals)
        correlation_pairs: List[Tuple[str, str]] = []
        for cnot_op in state.cnot_operations:
            control = cnot_op["control"]
            target = cnot_op["target"]
            if control in self.layout.patches and target in self.layout.patches:
                correlation_pairs.append((control, target))
        if not correlation_pairs:
            logical_names = [nm for nm in bracket_map.keys() if nm in self.layout.patches]
            if len(logical_names) >= 2:
                correlation_pairs.append((logical_names[0], logical_names[1]))
        
        # Determine if we should use the combined path
        requested = {b.upper() for b in demo_bases if isinstance(b, str)}
        use_combined = requested == {"Z", "X"}
        
        # Helper to emit a joint correlator (ZZ or XX) for a logical pair
        def _emit_joint_for_pair(basis: str, q0_name: str, q1_name: str):
            # Heisenberg-frame: measure U†(ZZ/XX)U at the end
            if basis == "X":
                op = Pauli.two_xx(n_logical, name_to_idx[q0_name], name_to_idx[q1_name])
            else:
                op = Pauli.two_zz(n_logical, name_to_idx[q0_name], name_to_idx[q1_name])
            conj = conjugate_through_circuit(op, qiskit_circuit)
            mpp_targets, axes_map = _mpp_targets_from_pauli(conj, self.layout, idx_to_name)
            if not mpp_targets:
                return None, None, None
            circuit.append_operation("MPP", mpp_targets)
            joint_idx = circuit.num_measurements - 1
            return joint_idx, axes_map, conj
        
        if use_combined and correlation_pairs:
            # ---------- Combined final layer: joint ZZ and joint XX within the SAME TICK ----------
            circuit.append_operation("TICK")
            
            for (q0_name, q1_name) in correlation_pairs:
                # Joint ZZ
                idx_zz, axes_map_zz, conj_zz = _emit_joint_for_pair("Z", q0_name, q1_name)
                if idx_zz is not None:
                    joint_demo_info[f"{q0_name}_{q1_name}_Z"] = {
                        "pair": [q0_name, q1_name],
                        "logical_operator": f"Z_L({q0_name})⊗Z_L({q1_name})",
                        "physical_realization": conj_zz.to_string(),
                        "basis": "Z",
                        "axes": axes_map_zz,
                        "index": idx_zz,
                    }
                
                # Joint XX
                idx_xx, axes_map_xx, conj_xx = _emit_joint_for_pair("X", q0_name, q1_name)
                if idx_xx is not None:
                    joint_demo_info[f"{q0_name}_{q1_name}_X"] = {
                        "pair": [q0_name, q1_name],
                        "logical_operator": f"X_L({q0_name})⊗X_L({q1_name})",
                        "physical_realization": conj_xx.to_string(),
                        "basis": "X",
                        "axes": axes_map_xx,
                        "index": idx_xx,
                    }
            
            # No singles or snapshot in combined mode
        else:
            # ---------- Single basis mode: per-basis emission with singles and snapshot ----------
            for basis in demo_bases:
                # ----- Joint product first for this basis -----
                circuit.append_operation("TICK")
                for (q0_name, q1_name) in correlation_pairs:
                    if q0_name not in self.layout.patches or q1_name not in self.layout.patches:
                        continue
                    idx_joint, axes_map, conj = _emit_joint_for_pair(basis, q0_name, q1_name)
                    if idx_joint is None:
                        continue
                    joint_key = f"{q0_name}_{q1_name}_{basis}"
                    joint_demo_info[joint_key] = {
                        "pair": [q0_name, q1_name],
                        "logical_operator": f"{basis}_L({q0_name})⊗{basis}_L({q1_name})",
                        "physical_realization": conj.to_string(),
                        "basis": basis,
                        "axes": axes_map,
                        "index": idx_joint,
                    }
                circuit.append_operation("TICK")
                
                # ----- Then single-qubit demos for this basis -----
                logical_names: List[str] = [nm for nm in bracket_map.keys() if nm in self.layout.patches]
                for patch_name in logical_names:
                    if basis == "Z":
                        initial_pauli = Pauli.single_z(n_logical, name_to_idx.get(patch_name, 0))
                    else:
                        initial_pauli = Pauli.single_x(n_logical, name_to_idx.get(patch_name, 0))
                    conjugated_pauli = conjugate_through_circuit(initial_pauli, qiskit_circuit)
                    singles_targets, _ = _mpp_targets_from_pauli(conjugated_pauli, self.layout, idx_to_name)
                    if not singles_targets:
                        continue
                    circuit.append_operation("MPP", singles_targets)
                    demo_idx = circuit.num_measurements - 1
                    key = f"{patch_name}_{basis}"
                    demo_info[key] = {
                        "basis": basis,
                        "index": demo_idx,
                        "patch": patch_name,
                        "logical_operator": conjugated_pauli.to_string(),
                        "phase": conjugated_pauli.phase_sign(),
                    }
                circuit.append_operation("TICK")
            
            # ----- Final computational-basis snapshot (single basis mode only) -----
            if qiskit_circuit is not None and demo_bases:
                snapshot_basis = demo_bases[0].upper()
                circuit.append_operation("TICK")
                logical_names = [nm for nm in sorted(bracket_map.keys()) if nm in self.layout.patches]
                snapshot_indices = []
                snapshot_ops = []
                snapshot_axes = []
                snapshot_phases = []
                order_out: List[str] = []
                
                for patch_name in logical_names:
                    # Build final-frame operator for this qubit
                    qi = name_to_idx.get(patch_name)
                    if qi is None:
                        continue
                    if snapshot_basis == "Z":
                        init_op = Pauli.single_z(n_logical, qi)
                    else:
                        init_op = Pauli.single_x(n_logical, qi)
                    conj_op = conjugate_through_circuit(init_op, qiskit_circuit)
                    targets, _ = _mpp_targets_from_pauli(conj_op, self.layout, idx_to_name)
                    if targets:
                        circuit.append_operation("MPP", targets)
                        idx = circuit.num_measurements - 1
                        snapshot_indices.append(idx)
                        # Use unified tracker helper to derive axis and phase
                        tracker = PauliTracker(n_logical)
                        info = tracker.final_operator_info(qi, snapshot_basis, qiskit_circuit)
                        snapshot_ops.append(info["operator_string"])
                        snapshot_axes.append(info["axis"])
                        snapshot_phases.append(int(info["phase"]))
                        order_out.append(patch_name)
                
                snapshot_info = {
                    "enabled": True,
                    "basis": snapshot_basis,
                    "order": order_out,
                    "indices": snapshot_indices,
                    "logical_ops": snapshot_ops,
                    "axes": snapshot_axes,
                    "phases": snapshot_phases,
                }
        
        return demo_info, joint_demo_info, snapshot_info

