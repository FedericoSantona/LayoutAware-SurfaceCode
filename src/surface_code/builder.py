"""Global Stim builder orchestrating multiple patches and surgery timeline.

The builder emits a deterministic DEM by:
  - Performing Z/X halves with one data-noise placement per half.
  - Adding temporal detectors between consecutive rounds for all checks.
  - Keeping per-patch logical bracketing fixed (start/end) via OBSERVABLE_INCLUDE.
  - During merges, injecting joint 2-body checks across explicit seams and only
    adding their time-difference detectors (not the raw joint parity) to DEM.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import stim

from .layout import Layout
from .surgery_ops import MeasureRound, Merge, Split, ParityReadout
from .configs import PhenomenologicalStimConfig
from .builder_utils import mpp_from_positions, rec_from_abs, add_temporal_detectors_with_index, _mpp_targets_from_pauli
from .pauli import Pauli, conjugate_through_circuit, PauliTracker


GateTarget = stim.GateTarget


@dataclass
class _PrevState:
    z_prev: Dict[str, List[Optional[int]]]
    x_prev: Dict[str, List[Optional[int]]]
    joint_prev: Dict[Tuple[str, str, str], List[int]]  # key=(kind,a,b)


class GlobalStimBuilder:
    """Build a multi-patch circuit from a layout and a sequence of ops."""

    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self._detector_count: int = 0

    def _emit_qubit_coords(self, circuit: stim.Circuit) -> None:
        coords = self.layout.global_coords()
        for q, (x, y) in coords.items():
            circuit.append_operation("QUBIT_COORDS", [q], [x, y])

    def _measure_patch_stabilizers(
        self,
        circuit: stim.Circuit,
        patch_names: Iterable[str],
        basis: str,
        skip_indices: Optional[Dict[str, Set[int]]] = None,
        prev_map: Optional[Dict[str, List[Optional[int]]]] = None,
    ) -> Dict[str, List[Optional[int]]]:
        offs = self.layout.offsets()
        measured: Dict[str, List[Optional[int]]] = {}
        for name in patch_names:
            patch = self.layout.patches[name]
            stabs = patch.z_stabs if basis == "Z" else patch.x_stabs
            #During a merge, stabilizers that conflict with seam checks are temporarily suppressed
            skip = set() if skip_indices is None else skip_indices.get(name, set())
            prev_list = []
            if prev_map is not None:
                prev_list = list(prev_map.get(name, []))
            indices: List[Optional[int]] = []
            for idx, s in enumerate(stabs):
                if skip and any(i in skip for i, c in enumerate(s) if c == basis):
                    prev_idx = prev_list[idx] if idx < len(prev_list) else None
                    indices.append(prev_idx)
                    continue
                # Build a global MPP by mapping local characters to global positions
                positions: List[int] = []
                for i, c in enumerate(s):
                    if c == basis:
                        positions.append(self.layout.globalize_local_index(name, i))
                idx = mpp_from_positions(circuit, positions, basis)
                indices.append(idx)
            measured[name] = indices
        return measured

    def _mask_prev_stabilizers(
        self,
        prev_dict: Dict[str, List[Optional[int]]],
        patch_name: str,
        basis: str,
        local_indices: Iterable[int],
    ) -> Set[int]:
        if patch_name not in prev_dict:
            return set()
        arr = list(prev_dict.get(patch_name, []))
        if not arr:
            return set()
        patch = self.layout.patches.get(patch_name)
        if patch is None:
            return set()
        stabs = patch.x_stabs if basis == "X" else patch.z_stabs
        mask_idxs: Set[int] = set()
        local_set = set(local_indices)
        for stab_idx, stab in enumerate(stabs):
            if stab_idx >= len(arr):
                break
            for li in local_set:
                if li < len(stab) and stab[li] == basis:
                    mask_idxs.add(stab_idx)
                    break
        if not mask_idxs:
            return set()
        for idx in mask_idxs:
            if 0 <= idx < len(arr):
                arr[idx] = None
        prev_dict[patch_name] = arr
        return mask_idxs

    def _measure_joint_checks(
        self,
        circuit: stim.Circuit,
        kind: str,
        a: str,
        b: str,
    ) -> List[int]:
        """Measure simple 2-body joint checks across the seam.

        kind='rough' uses Z⊗Z across pairs; kind='smooth' uses X⊗X.
        Returns the list of absolute measurement indices (one per pair).
        """
        key = (kind, a, b)
        pairs = self.layout.seams.get(key, [])
        if not pairs:
            return []
        pauli = "Z" if kind == "rough" else "X"
        indices: List[int] = []
        for ia, ib in pairs:
            global_a = self.layout.globalize_local_index(a, ia)
            global_b = self.layout.globalize_local_index(b, ib)
            idx = mpp_from_positions(circuit, [global_a, global_b], pauli)
            if idx is not None:
                indices.append(idx)
        return indices

    def build(
        self,
        ops: Sequence[object],
        cfg: PhenomenologicalStimConfig,
        bracket_map: Dict[str, str],  # patch_name -> 'Z'|'X'
        qiskit_circuit: Optional[object] = None,  # Qiskit circuit for demo conjugation
    ) -> Tuple[stim.Circuit, List[Tuple[int, int]], Dict[str, object]]:
        layout = self.layout
        circuit = stim.Circuit()

        # Coordinates
        self._emit_qubit_coords(circuit)

        # Detector index appender (tracks DEM column indices)
        def append_detector(targets: List[GateTarget]) -> int:
            idx = self._detector_count
            circuit.append_operation("DETECTOR", targets)
            self._detector_count += 1
            return idx

        # Resolve patch order and selection helpers
        all_patches: List[str] = list(layout.patches.keys())

        def select_patches(spec: Optional[List[str]]) -> List[str]:
            return all_patches if spec is None else list(spec)

        # Determine merge participation per patch for bracket adjustments
        rough_merge_patches: Set[str] = set()
        smooth_merge_patches: Set[str] = set()
        for op in ops:
            if isinstance(op, Merge):
                k = op.type.strip().lower()
                if k == "rough":
                    rough_merge_patches.update({op.a, op.b})
                elif k == "smooth":
                    smooth_merge_patches.update({op.a, op.b})

        # Effective bracket basis per patch (ensure commutes with merge windows)
        effective_basis_map: Dict[str, str] = {}
        for name, req_basis in bracket_map.items():
            basis = req_basis.upper()
            if basis not in {"Z", "X"}:
                raise ValueError("bracket_map values must be 'Z' or 'X'")
            effective = basis
            if basis == "Z" and name in smooth_merge_patches:
                effective = "X"
            elif basis == "X" and name in rough_merge_patches:
                effective = "Z"
            effective_basis_map[name] = effective

        # Bracketing: start logicals per patch (skip if they would anti-commute with merges)
        start_indices: Dict[str, Optional[int]] = {name: None for name in all_patches}
        end_indices: Dict[str, Optional[int]] = {name: None for name in all_patches}

        circuit.append_operation("TICK")
        # Only bracket patches that are explicitly in bracket_map (excludes ancillas)
        for name in bracket_map.keys():
            if name not in layout.patches:
                continue  # Skip if patch doesn't exist in layout
            basis = bracket_map[name].upper()
            effective = effective_basis_map.get(name, basis)
            # Skip immediate start if upcoming merges would anti-commute; we'll capture later.
            if (effective == "Z" and name in smooth_merge_patches) or (
                effective == "X" and name in rough_merge_patches
            ):
                continue
            patch = layout.patches[name]
            s = patch.logical_z if effective == "Z" else patch.logical_x
            # Convert local string to global positions with same basis
            offs = layout.offsets()[name]
            positions = [offs + i for i, c in enumerate(s) if c == effective]
            start_indices[name] = mpp_from_positions(circuit, positions, effective)

        # Pending starts to be captured later once compatible measurements occur
        pending_start: Dict[str, str] = {
            name: effective_basis_map[name]
            for name in effective_basis_map.keys()
            if start_indices.get(name) is None
        }


        # Track remaining merge windows that conflict with seam stabilizer bases
        conflict_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for op in ops:
            if isinstance(op, Merge):
                k = op.type.strip().lower()
                if k == "rough":
                    conflict_counts[(op.a, "X")] += 1
                    conflict_counts[(op.b, "X")] += 1
                elif k == "smooth":
                    conflict_counts[(op.a, "Z")] += 1
                    conflict_counts[(op.b, "Z")] += 1


        # Establish initial references: Z then X for active patches
        prev = _PrevState(z_prev={}, x_prev={}, joint_prev={})
        # Z refs
        circuit.append_operation("TICK")
        z_meas = self._measure_patch_stabilizers(circuit, all_patches, "Z")
        prev.z_prev = z_meas
        # X refs
        circuit.append_operation("TICK")
        x_meas = self._measure_patch_stabilizers(circuit, all_patches, "X")
        prev.x_prev = x_meas

        # Active merge trackers and metadata for joint windows
        active_rough: Optional[Tuple[str, str, int]] = None  # (a,b,remaining)
        active_smooth: Optional[Tuple[str, str, int]] = None
        merge_windows: List[Dict[str, object]] = []
        current_window: Optional[Dict[str, object]] = None
        window_id = 0
        
        # CNOT operation tracking for Pauli frame updates
        cnot_operations: List[Dict[str, object]] = []
        current_cnot: Optional[Dict[str, object]] = None

        def _begin_window(kind: str, a: str, b: str, rounds: int) -> None:
            nonlocal current_window, window_id
            current_window = {
                "id": window_id,
                "type": kind,
                "parity_type": "ZZ" if kind == "rough" else "XX",
                "a": a,
                "b": b,
                "rounds": int(rounds),
            }
            window_id += 1

        def _end_window() -> None:
            nonlocal current_window
            if current_window is not None:
                merge_windows.append(current_window)
                current_window = None

        # Iterate timeline
        for op in ops:
            if isinstance(op, MeasureRound):
                # One full ZX cycle
                names = select_patches(op.patch_ids)

                measure_z = getattr(op, "measure_z", True)
                measure_x = getattr(op, "measure_x", True)

                # Z half
                circuit.append_operation("TICK")
                if cfg.p_x_error:
                    circuit.append_operation("X_ERROR", list(range(layout.global_n())), cfg.p_x_error)
                z_curr = {name: (list(vals) if vals is not None else []) for name, vals in prev.z_prev.items()}
                if measure_z:
                    meas_z_curr = self._measure_patch_stabilizers(
                        circuit,
                        names,
                        "Z",
                    )
                    for name in names:
                        add_temporal_detectors_with_index(
                            circuit,
                            prev.z_prev.get(name, []),
                            meas_z_curr.get(name, []),
                            append_detector,
                        )
                        z_curr[name] = list(meas_z_curr.get(name, []))
                        if (
                            name in pending_start
                            and pending_start[name] == "Z"
                            and not (
                                (active_smooth is not None and name in {active_smooth[0], active_smooth[1]})
                                or (active_rough is not None and name in {active_rough[0], active_rough[1]})
                            )
                            and conflict_counts.get((name, "Z"), 0) == 0
                        ):
                            indices = meas_z_curr.get(name, [])
                            first_idx = next((idx for idx in indices if idx is not None), None)
                            if first_idx is not None:
                                start_indices[name] = first_idx
                                pending_start.pop(name, None)
                else:
                    for name in names:
                        if name not in z_curr:
                            z_curr[name] = list(prev.z_prev.get(name, []))

                if measure_z and active_rough is not None:
                    # Measure rough seam ZZ checks when the merge is active.
                    seam_a, seam_b, _ = active_rough
                    key = ("rough", seam_a, seam_b)
                    prev_joint = prev.joint_prev.get(key, [])
                    joint_curr = self._measure_joint_checks(circuit, "rough", seam_a, seam_b)
                    add_temporal_detectors_with_index(
                        circuit,
                        prev_joint,
                        joint_curr,
                        append_detector,
                    )
                    prev.joint_prev[key] = list(joint_curr)

                prev.z_prev = z_curr

                # X half
                circuit.append_operation("TICK")
                if cfg.p_z_error:
                    circuit.append_operation("Z_ERROR", list(range(layout.global_n())), cfg.p_z_error)
                x_curr = {name: (list(vals) if vals is not None else []) for name, vals in prev.x_prev.items()}
                if measure_x:
                    meas_x_curr = self._measure_patch_stabilizers(
                        circuit,
                        names,
                        "X",
                    )
                    for name in names:
                        add_temporal_detectors_with_index(
                            circuit,
                            prev.x_prev.get(name, []),
                            meas_x_curr.get(name, []),
                            append_detector,
                        )
                        x_curr[name] = list(meas_x_curr.get(name, []))
                        if (
                            name in pending_start
                            and pending_start[name] == "X"
                            and not (
                                (active_smooth is not None and name in {active_smooth[0], active_smooth[1]})
                                or (active_rough is not None and name in {active_rough[0], active_rough[1]})
                            )
                            and conflict_counts.get((name, "X"), 0) == 0
                        ):
                            indices = meas_x_curr.get(name, [])
                            first_idx = next((idx for idx in indices if idx is not None), None)
                            if first_idx is not None:
                                start_indices[name] = first_idx
                                pending_start.pop(name, None)
                else:
                    for name in names:
                        if name not in x_curr:
                            x_curr[name] = list(prev.x_prev.get(name, []))

                if measure_x and active_smooth is not None:
                    # Measure smooth seam XX checks when the merge is active.
                    seam_a, seam_b, _ = active_smooth
                    key = ("smooth", seam_a, seam_b)
                    prev_joint = prev.joint_prev.get(key, [])
                    joint_curr = self._measure_joint_checks(circuit, "smooth", seam_a, seam_b)
                    add_temporal_detectors_with_index(
                        circuit,
                        prev_joint,
                        joint_curr,
                        append_detector,
                    )
                    prev.joint_prev[key] = list(joint_curr)

                prev.x_prev = x_curr

                # Update merge countdowns and close windows when done
                if active_rough is not None:
                    a, b, rem = active_rough
                    rem -= 1
                    if rem <= 0:
                        active_rough = None
                    else:
                        active_rough = (a, b, rem)
                if active_smooth is not None:
                    a, b, rem = active_smooth
                    rem -= 1
                    if rem <= 0:
                        active_smooth = None
                    else:
                        active_smooth = (a, b, rem)

            elif isinstance(op, Merge):
                k = op.type.strip().lower()
                if k not in {"rough", "smooth"}:
                    raise ValueError("Merge.type must be 'rough' or 'smooth'")
                if k == "rough":
                    if active_rough is not None:
                        raise RuntimeError("A rough merge is already active")
                    seam_pairs = layout.seams.get(("rough", op.a, op.b), [])
                    indices_a = {ia for ia, _ in seam_pairs}
                    indices_b = {ib for _, ib in seam_pairs}
                    self._mask_prev_stabilizers(prev.x_prev, op.a, "X", indices_a)
                    self._mask_prev_stabilizers(prev.x_prev, op.b, "X", indices_b)
                    active_rough = (op.a, op.b, int(op.rounds))
                else:
                    if active_smooth is not None:
                        raise RuntimeError("A smooth merge is already active")
                    seam_pairs = layout.seams.get(("smooth", op.a, op.b), [])
                    indices_a = {ia for ia, _ in seam_pairs}
                    indices_b = {ib for _, ib in seam_pairs}
                    self._mask_prev_stabilizers(prev.z_prev, op.a, "Z", indices_a)
                    self._mask_prev_stabilizers(prev.z_prev, op.b, "Z", indices_b)
                    active_smooth = (op.a, op.b, int(op.rounds))
                # Clear any lingering joint history for this seam
                prev.joint_prev[(k, op.a, op.b)] = []
                _begin_window(k, op.a, op.b, int(op.rounds))

            elif isinstance(op, Split):
                k = op.type.strip().lower()
                if k not in {"rough", "smooth"}:
                    raise ValueError("Split.type must be 'rough' or 'smooth'")
                if k == "rough":
                    active_rough = None
                else:
                    active_smooth = None
                _end_window()

                # Decrement remaining conflicting merges for involved patches
                for patch_name in (op.a, op.b):
                    basis = "X" if k == "rough" else "Z"
                    key = (patch_name, basis)
                    if conflict_counts.get(key, 0) > 0:
                        conflict_counts[key] -= 1
                prev.joint_prev[(k, op.a, op.b)] = []

            elif isinstance(op, ParityReadout):
                # Emit single-shot MPP for merge byproduct extraction
                if op.type == "ZZ":
                    # Emit Z_L(a) ⊗ Z_L(b) MPP
                    patch_a = layout.patches[op.a]
                    patch_b = layout.patches[op.b]
                    base_a = layout.offsets()[op.a]
                    base_b = layout.offsets()[op.b]
                    
                    zz_targets: List[stim.GateTarget] = []
                    first = True
                    # Add Z targets from patch a
                    for i, ch in enumerate(patch_a.logical_z):
                        if ch == "Z":
                            if not first:
                                zz_targets.append(stim.target_combiner())
                            zz_targets.append(stim.target_z(base_a + i))
                            first = False
                    # Add Z targets from patch b
                    for i, ch in enumerate(patch_b.logical_z):
                        if ch == "Z":
                            if not first:
                                zz_targets.append(stim.target_combiner())
                            zz_targets.append(stim.target_z(base_b + i))
                            first = False
                    
                    circuit.append_operation("MPP", zz_targets)
                    m_zz_mpp_idx = circuit.num_measurements - 1
                    
                    # Track CNOT operations by grouping ZZ and XX parity readouts
                    if current_cnot is not None:
                        current_cnot["m_zz_mpp_idx"] = m_zz_mpp_idx
                    else:
                        # Start of a CNOT operation (rough merge completed)
                        current_cnot = {
                            "control": op.a,
                            "target": op.b,  # This will be updated to actual target when XX comes
                            "ancilla": op.b,  # For ZZ, b is the ancilla
                            "rough_window_id": window_id - 1 if current_window is None else window_id,
                            "smooth_window_id": None,
                            "m_zz_mpp_idx": m_zz_mpp_idx,
                            "m_xx_mpp_idx": None,
                        }
                        
                elif op.type == "XX":
                    # Emit X_L(a) ⊗ X_L(b) MPP using Pauli conjugation
                    if qiskit_circuit is not None:
                        # Use existing Pauli→targets helper for final-frame XX
                        n_logical = qiskit_circuit.num_qubits
                        name_to_idx = {f"q{i}": i for i in range(n_logical)}
                        idx_to_name = {i: f"q{i}" for i in range(n_logical)}
                        
                        # Find logical indices for patches a and b
                        idx_a = name_to_idx.get(op.a, 0)
                        idx_b = name_to_idx.get(op.b, 0)
                        
                        # Create XX Pauli and conjugate through circuit
                        op_xx = Pauli.two_xx(n_logical, idx_a, idx_b)
                        conj_xx = conjugate_through_circuit(op_xx, qiskit_circuit)
                        xx_targets, _ = _mpp_targets_from_pauli(conj_xx, layout, idx_to_name)
                        
                        circuit.append_operation("MPP", xx_targets)
                        m_xx_mpp_idx = circuit.num_measurements - 1
                    else:
                        # Fallback: direct XX measurement (shouldn't happen in normal usage)
                        patch_a = layout.patches[op.a]
                        patch_b = layout.patches[op.b]
                        base_a = layout.offsets()[op.a]
                        base_b = layout.offsets()[op.b]
                        
                        xx_targets: List[stim.GateTarget] = []
                        first = True
                        # Add X targets from patch a
                        for i, ch in enumerate(patch_a.logical_x):
                            if ch == "X":
                                if not first:
                                    xx_targets.append(stim.target_combiner())
                                xx_targets.append(stim.target_x(base_a + i))
                                first = False
                        # Add X targets from patch b
                        for i, ch in enumerate(patch_b.logical_x):
                            if ch == "X":
                                if not first:
                                    xx_targets.append(stim.target_combiner())
                                xx_targets.append(stim.target_x(base_b + i))
                                first = False
                        
                        circuit.append_operation("MPP", xx_targets)
                        m_xx_mpp_idx = circuit.num_measurements - 1
                    
                    # Complete the CNOT operation (smooth merge completed)
                    if current_cnot is not None:
                        current_cnot["target"] = op.b  # Update target (for XX, b is the target)
                        current_cnot["smooth_window_id"] = window_id - 1 if current_window is None else window_id
                        current_cnot["m_xx_mpp_idx"] = m_xx_mpp_idx
                        cnot_operations.append(current_cnot)
                        current_cnot = None

            else:
                raise TypeError(f"Unsupported op type: {type(op)!r}")

        # Emit boundary anchors after all merges complete
        # Bracketing: end logicals per patch and observable includes
        circuit.append_operation("TICK")
        observable_pairs: List[Tuple[int, int]] = []
        basis_labels: List[str] = []  
        observable_index = 0
        # Only bracket patches that are explicitly in bracket_map (excludes ancillas)
        for name in bracket_map.keys():
            if name not in layout.patches:
                continue  # Skip if patch doesn't exist in layout
            requested_basis = bracket_map[name].upper()
            effective_basis = effective_basis_map.get(name, requested_basis)
            patch = layout.patches[name]
            s = patch.logical_z if effective_basis == "Z" else patch.logical_x
            positions, _ = layout.globalize_local_pauli_string(name, s)
            end_idx = mpp_from_positions(circuit, positions, effective_basis)
            end_indices[name] = end_idx
            start_idx = start_indices[name]
            targets: List[GateTarget] = []
            if start_idx is not None:
                targets.append(rec_from_abs(circuit, start_idx))
            if end_idx is not None:
                targets.append(rec_from_abs(circuit, end_idx))
            if targets:
                circuit.append_operation("OBSERVABLE_INCLUDE", targets, observable_index)
                observable_pairs.append((start_idx, end_idx))
                basis_labels.append(effective_basis)
                observable_index += 1

        # Optional end-only demo readout with conjugated operators
        demo_info: Dict[str, object] = {}
        
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

        # Emit end-of-circuit demos.
        # If both Z and X demo bases are requested, emit a single *combined* final layer
        # where the joint ZZ and joint XX correlators are measured back-to-back *within
        # the same TICK*. Then (optionally) emit any single-qubit demos in a later TICK.
        joint_demo_info: Dict[str, object] = {}
        if demo_bases and qiskit_circuit is not None:
            # Prepare mapping between logical names and qiskit indices using the
            # explicit QuantumCircuit qubit order. This implicitly excludes any
            # ancilla patches that are not part of the original circuit.
            name_to_idx: Dict[str, int] = {}
            idx_to_name: Dict[int, str] = {}
            n_logical = qiskit_circuit.num_qubits
            for qi in range(n_logical):
                name = f"q{qi}"
                name_to_idx[name] = qi
                idx_to_name[qi] = name

            # Correlation pairs from compiled CNOT operations (fallback to first two logicals).
            correlation_pairs: List[Tuple[str, str]] = []
            for cnot_op in cnot_operations:
                control = cnot_op["control"]
                target = cnot_op["target"]
                if control in layout.patches and target in layout.patches:
                    correlation_pairs.append((control, target))
            if not correlation_pairs:
                logical_names = [nm for nm in bracket_map.keys() if nm in layout.patches]
                if len(logical_names) >= 2:
                    correlation_pairs.append((logical_names[0], logical_names[1]))

            # Determine if we should use the combined path.
            requested = {b.upper() for b in demo_bases if isinstance(b, str)}
            use_combined = requested == {"Z", "X"}

            if use_combined and correlation_pairs:
                # ---------- Combined final layer: joint ZZ and joint XX within the SAME TICK ----------
                circuit.append_operation("TICK")

                for (q0_name, q1_name) in correlation_pairs:
                    # Joint ZZ: build Pauli ZZ on (q0,q1), Heisenberg-conjugate through qc,
                    # then map to physical targets.
                    op_zz = Pauli.two_zz(n_logical, name_to_idx[q0_name], name_to_idx[q1_name])
                    conj_zz = conjugate_through_circuit(op_zz, qiskit_circuit)
                    zz_targets, axes_map_zz = _mpp_targets_from_pauli(conj_zz, layout, idx_to_name)
                    circuit.append_operation("MPP", zz_targets)
                    idx_zz = circuit.num_measurements - 1
                    joint_demo_info[f"{q0_name}_{q1_name}_Z"] = {
                        "pair": [q0_name, q1_name],
                        "logical_operator": f"Z_L({q0_name})⊗Z_L({q1_name})",
                        "physical_realization": conj_zz.to_string(),
                        "basis": "Z",
                        "axes": axes_map_zz,
                        "index": idx_zz,
                    }

                    # Joint XX: build Pauli XX on (q0,q1), Heisenberg-conjugate through qc,
                    # then map to physical targets.
                    op_xx = Pauli.two_xx(n_logical, name_to_idx[q0_name], name_to_idx[q1_name])
                    conj_xx = conjugate_through_circuit(op_xx, qiskit_circuit)
                    xx_targets, axes_map_xx = _mpp_targets_from_pauli(conj_xx, layout, idx_to_name)
                    circuit.append_operation("MPP", xx_targets)
                    idx_xx = circuit.num_measurements - 1
                    joint_demo_info[f"{q0_name}_{q1_name}_X"] = {
                        "pair": [q0_name, q1_name],
                        "logical_operator": f"X_L({q0_name})⊗X_L({q1_name})",
                        "physical_realization": conj_xx.to_string(),
                        "basis": "X",
                        "axes": axes_map_xx,
                        "index": idx_xx,
                    }

                # New contract: if both Z and X are requested, emit only joint correlators (ZZ and XX)
                # If only one basis is requested, emit singles for that basis only
                emit_singles = len(requested) == 1
                
                if emit_singles:
                    # Optional singles in a *later* TICK (won't disturb joint bits already measured).
                    circuit.append_operation("TICK")
                    logical_names: List[str] = [nm for nm in bracket_map.keys() if nm in layout.patches]
                    # Emit singles only for the single requested basis
                    single_basis = next(iter(requested))
                    for patch_name in logical_names:
                        if single_basis == "Z":
                            init = Pauli.single_z(n_logical, name_to_idx.get(patch_name, 0))
                        else:
                            init = Pauli.single_x(n_logical, name_to_idx.get(patch_name, 0))
                        conj = conjugate_through_circuit(init, qiskit_circuit)
                        singles_targets, axes_map = _mpp_targets_from_pauli(conj, layout, idx_to_name)
                        if not singles_targets:
                            continue
                        circuit.append_operation("MPP", singles_targets)
                        demo_idx = circuit.num_measurements - 1
                        key = f"{patch_name}_{single_basis}"
                        demo_info[key] = {
                            "basis": single_basis,
                            "index": demo_idx,
                            "patch": patch_name,
                            "logical_operator": conj.to_string(),
                            "phase": conj.phase_sign(),
                        }
                    circuit.append_operation("TICK")
                    # Do not append any further detectors or observables after this point.
            else:
                # ---------- Fallback: per-basis emission (legacy path) ----------
                for basis in demo_bases:
                    # ----- Joint product first for this basis -----
                    circuit.append_operation("TICK")
                    for (q0_name, q1_name) in correlation_pairs:
                        if q0_name not in layout.patches or q1_name not in layout.patches:
                            continue
                        offs = layout.offsets()
                        if basis == "X":
                            op = Pauli.two_xx(n_logical, name_to_idx[q0_name], name_to_idx[q1_name])
                            conj = conjugate_through_circuit(op, qiskit_circuit)
                            mpp_targets, axes_map = _mpp_targets_from_pauli(conj, layout, idx_to_name)
                            if not mpp_targets:
                                continue
                            circuit.append_operation("MPP", mpp_targets)
                            joint_idx = circuit.num_measurements - 1
                            joint_key = f"{q0_name}_{q1_name}_{basis}"
                            joint_demo_info[joint_key] = {
                                "pair": [q0_name, q1_name],
                                "logical_operator": f"{basis}_L({q0_name})⊗{basis}_L({q1_name})",
                                "physical_realization": conj.to_string(),
                                "basis": basis,
                                "axes": axes_map,
                                "index": joint_idx,
                            }
                        else:
                            # ZZ: build Pauli ZZ, Heisenberg-conjugate through qc, then map to physical targets
                            op = Pauli.two_zz(n_logical, name_to_idx[q0_name], name_to_idx[q1_name])
                            conj = conjugate_through_circuit(op, qiskit_circuit)
                            mpp_targets, axes_map = _mpp_targets_from_pauli(conj, layout, idx_to_name)
                            if not mpp_targets:
                                continue
                            circuit.append_operation("MPP", mpp_targets)
                            joint_idx = circuit.num_measurements - 1
                            joint_key = f"{q0_name}_{q1_name}_{basis}"
                            joint_demo_info[joint_key] = {
                                "pair": [q0_name, q1_name],
                                "logical_operator": f"{basis}_L({q0_name})⊗{basis}_L({q1_name})",
                                "physical_realization": conj.to_string(),
                                "basis": basis,
                                "axes": axes_map,
                                "index": joint_idx,
                            }
                    circuit.append_operation("TICK")

                    # ----- Then single-qubit demos for this basis -----
                    # New contract: emit singles only for single-basis requests
                    # Skip singles if both Z and X are requested (joint-only mode)
                    emit_singles_for_basis = len(demo_bases) == 1
                    
                    if emit_singles_for_basis:
                        logical_names: List[str] = [nm for nm in bracket_map.keys() if nm in layout.patches]
                        for patch_name in logical_names:
                            if basis == "Z":
                                initial_pauli = Pauli.single_z(n_logical, name_to_idx.get(patch_name, 0))
                            else:
                                initial_pauli = Pauli.single_x(n_logical, name_to_idx.get(patch_name, 0))
                            conjugated_pauli = conjugate_through_circuit(initial_pauli, qiskit_circuit)
                            singles_targets, _ = _mpp_targets_from_pauli(conjugated_pauli, layout, idx_to_name)
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

        # Final computational-basis snapshot (only if single basis requested)
        snapshot_info = {"enabled": False}
        if len(demo_bases) == 1 and qiskit_circuit is not None:
            snapshot_basis = demo_bases[0].upper()
            if snapshot_basis in ("Z", "X"):
                circuit.append_operation("TICK")
                logical_names = [nm for nm in sorted(bracket_map.keys()) if nm in layout.patches]
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
                    targets, _ = _mpp_targets_from_pauli(conj_op, layout, idx_to_name)
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

        metadata: Dict[str, object] = {
            "merge_windows": merge_windows,
            "observable_basis": tuple(basis_labels),
            "demo": demo_info,
            "joint_demos": joint_demo_info,
            "cnot_operations": cnot_operations,
            "final_snapshot": snapshot_info,
        }

        return circuit, observable_pairs, metadata
