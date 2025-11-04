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
from .surgery_ops import MeasureRound, Merge, Split, ParityReadout, TerminatePatch
from .configs import PhenomenologicalStimConfig
from .builder_utils import mpp_from_positions
from .builder_state import BuilderState, _PrevState
from .detector_manager import DetectorManager
from .segment_tracker import SegmentTracker
from .merge_manager import MergeManager
from .measurement_half import MeasurementHalf
from .observable_manager import ObservableManager
from .demo_generator import DemoGenerator


GateTarget = stim.GateTarget


class GlobalStimBuilder:
    """Build a multi-patch circuit from a layout and a sequence of ops."""

    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self._detector_count: int = 0
        self._boundary_rows: Dict[Tuple[str, str], Set[int]] = self._compute_boundary_rows()

    def _compute_boundary_rows(self) -> Dict[Tuple[str, str], Set[int]]:
        """Pre-compute which stabilizer rows sit on patch boundaries.

        We classify a stabilizer row as a boundary when the participating data
        qubits live near the geometric edge of the patch (within a tolerance).
        For models without coordinate metadata we fall back to a degree-based
        heuristic (qubits appearing in only one stabilizer are treated as edges).
        """
        boundary_rows: Dict[Tuple[str, str], Set[int]] = {}
        for name, patch in self.layout.patches.items():
            coords = {
                q: (float(x), float(y))
                for q, (x, y) in patch.coords.items()
            }
            if coords:
                xs = [x for x, _ in coords.values()]
                ys = [y for _, y in coords.values()]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                tolerance = 0.6

                def classify(stabs: List[str], pauli: str) -> Set[int]:
                    rows: Set[int] = set()
                    for si, stab in enumerate(stabs):
                        points = [(coords[idx][0], coords[idx][1]) for idx, char in enumerate(stab) if char == pauli and idx in coords]
                        if not points:
                            continue
                        avg_x = sum(px for px, _ in points) / len(points)
                        avg_y = sum(py for _, py in points) / len(points)
                        dist = min(
                            abs(avg_x - xmin),
                            abs(avg_x - xmax),
                            abs(avg_y - ymin),
                            abs(avg_y - ymax),
                        )
                        if dist <= tolerance:
                            rows.add(si)
                    return rows
            else:
                n = patch.n

                def classify(stabs: List[str], pauli: str) -> Set[int]:
                    counts = [0] * n
                    for stab in stabs:
                        for qi, char in enumerate(stab):
                            if char == pauli:
                                counts[qi] += 1
                    rows: Set[int] = set()
                    for si, stab in enumerate(stabs):
                        boundary = False
                        for qi, char in enumerate(stab):
                            if char == pauli and counts[qi] <= 1:
                                boundary = True
                                break
                        if boundary:
                            rows.add(si)
                    return rows

            boundary_rows[(name, "Z")] = classify(patch.z_stabs, "Z")
            boundary_rows[(name, "X")] = classify(patch.x_stabs, "X")

        return boundary_rows

    def is_boundary_row(self, patch: str, basis: str, row_idx: int) -> bool:
        """Return True when the stabilizer row is treated as a physical boundary."""
        key = (patch, basis.upper())
        rows = self._boundary_rows.get(key)
        if not rows:
            return False
        return row_idx in rows

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
        p_meas: float = 0.0,
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
                idx = mpp_from_positions(circuit, positions, basis, p_meas=p_meas)
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
        
        DEPRECATED: Use MergeManager.measure_joint_checks instead.
        This method is kept for backward compatibility but delegates to merge_manager.
        """
        # This method is kept for backward compatibility but is no longer used internally
        # Internal code now uses merge_manager.measure_joint_checks directly
        temp_manager = MergeManager(self.layout)
        return temp_manager.measure_joint_checks(circuit, kind, a, b)

    def _emit_logical_mpp(
        self,
        circuit: stim.Circuit,
        patch_name: str,
        basis: str,
    ) -> Optional[int]:
        """Emit an MPP for the logical operator of ``patch_name`` in ``basis``.

        Returns the absolute measurement index or ``None`` when the logical has
        no support (should not occur for surface-code patches).
        """
        patch = self.layout.patches.get(patch_name)
        if patch is None:
            return None

        logical_string = patch.logical_z if basis == "Z" else patch.logical_x
        positions, chars = self.layout.globalize_local_pauli_string(patch_name, logical_string)
        if not positions:
            return None

        # Defensive check: ensure the stored logical matches the requested basis.
        if any(c != basis for c in chars):
            raise ValueError(
                f"Logical operator for patch '{patch_name}' contains non-{basis} axes: {chars}"
            )

        # Logical bracket MPPs must remain noiseless (p_meas=0.0) to keep observables deterministic anchors
        return mpp_from_positions(circuit, positions, basis, p_meas=0.0)

    def build(
        self,
        ops: Sequence[object],
        cfg: PhenomenologicalStimConfig,
        bracket_map: Dict[str, str],  # patch_name -> 'Z'|'X'
        qiskit_circuit: Optional[object] = None,  # Qiskit circuit for demo conjugation
        *,
        explicit_logical_start: bool = True,
    ) -> Tuple[stim.Circuit, List[Tuple[int, int]], Dict[str, object]]:
        layout = self.layout
        circuit = stim.Circuit()

        # Coordinates
        self._emit_qubit_coords(circuit)

        # Initialize qubits based on init_label
        # Stim assumes qubits start in |0> by default, so we only need to apply gates for other states
        if cfg.init_label is not None:
            from .pauli import parse_init_label
            init_basis, init_phase = parse_init_label(cfg.init_label)
            all_qubits = list(range(layout.global_n()))
            
            if init_basis == "X":
                # Initialize to |+> or |->
                # |+> = H|0>, |-> = H|1> = HX|0>
                if init_phase == -1:
                    # |-> state: apply X then H
                    circuit.append_operation("X", all_qubits)
                circuit.append_operation("H", all_qubits)
            elif init_basis == "Z" and init_phase == -1:
                # |1> state: apply X
                circuit.append_operation("X", all_qubits)
            # If init_basis=="Z" and init_phase==+1, qubits are already in |0>, no initialization needed

        # Initialize state and managers
        state = BuilderState()
        force_boundaries = getattr(cfg, "force_boundaries", True)
        boundary_error_prob = getattr(cfg, "boundary_error_prob", 1e-12)
        detector_manager = DetectorManager(force_boundaries=force_boundaries, boundary_error_prob=boundary_error_prob)
        segment_tracker = SegmentTracker(boundary_checker=self.is_boundary_row)
        merge_manager = MergeManager(self.layout)

        # Resolve patch order and selection helpers
        all_patches: List[str] = list(layout.patches.keys())
        
        # Track patches terminated by mid-circuit measurements (now in state)

        def select_patches(spec: Optional[List[str]]) -> List[str]:
            # Filter out terminated patches
            active = [p for p in all_patches if p not in state.terminated_patches]
            return active if spec is None else [p for p in spec if p not in state.terminated_patches]


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

        # Effective bracket basis per patch: DO NOT flip due to merges.
        # Observables must commute with initial collapse (|0> by default),
        # so honor the requested bracket basis (e.g., Z) and capture anchors
        # only at commuting times via the pending-start logic.
        effective_basis_map: Dict[str, str] = {}
        for name, req_basis in bracket_map.items():
            basis = req_basis.upper()
            if basis not in {"Z", "X"}:
                raise ValueError("bracket_map values must be 'Z' or 'X'")
            effective_basis_map[name] = basis

        # Initialize observable and demo managers
        observable_manager = ObservableManager(self.layout, bracket_map)
        demo_generator = DemoGenerator(self.layout, self)
        
        # Sync effective_basis_map to observable_manager
        observable_manager.effective_basis_map = effective_basis_map

        # Track observable bracketing indices per patch
        start_indices: Dict[str, Optional[int]] = {name: None for name in all_patches}
        end_indices: Dict[str, Optional[int]] = {name: None for name in all_patches}

        # Emit explicit logical brackets only when no merges are present
        emit_explicit_logicals = not rough_merge_patches and not smooth_merge_patches

        pending_start: Dict[str, str] = {}
        if emit_explicit_logicals and explicit_logical_start:
            for name, basis in effective_basis_map.items():
                idx = self._emit_logical_mpp(circuit, name, basis)
                if idx is not None:
                    start_indices[name] = idx
                    observable_manager.capture_start(name, idx)
                else:
                    pending_start[name] = basis
        else:
            # Defer start capture to the first compatible stabilizer half
            pending_start = {name: effective_basis_map[name] for name in effective_basis_map.keys()}

        circuit.append_operation("TICK")



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

        # Helper: last non-None index in a list
        def _last_non_none(idxs: List[Optional[int]]) -> Optional[int]:
            for k in range(len(idxs) - 1, -1, -1):
                if idxs[k] is not None:
                    return idxs[k]
            return None

        # Map local data indices to stabilizer row indices that include them
        def _rows_touching_local_indices(patch_name: str, basis: str, local_indices: Iterable[int]) -> Set[int]:
            patch = layout.patches.get(patch_name)
            if patch is None:
                return set()
            stabs = patch.z_stabs if basis == "Z" else patch.x_stabs
            local_set = set(local_indices)
            rows: Set[int] = set()
            for si, stab in enumerate(stabs):
                for li in local_set:
                    if li < len(stab) and stab[li] == basis:
                        rows.add(si)
                        break
            return rows

        # (Segment tracking now handled by segment_tracker)
        
        # (Merge tracking now handled by merge_manager)

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
                
                z_half = MeasurementHalf(self, "Z")
                z_curr = z_half.measure_round(
                    circuit,
                    names,
                    cfg,
                    state,
                    detector_manager,
                    segment_tracker,
                    merge_manager,
                    measure_z,
                    pending_start,
                    conflict_counts,
                    start_indices,
                    _rows_touching_local_indices,
                    observable_manager,
                )
                state.prev.z_prev = z_curr

                # X half
                circuit.append_operation("TICK")
                if cfg.p_z_error:
                    circuit.append_operation("Z_ERROR", list(range(layout.global_n())), cfg.p_z_error)
                
                x_half = MeasurementHalf(self, "X")
                x_curr = x_half.measure_round(
                    circuit,
                    names,
                    cfg,
                    state,
                    detector_manager,
                    segment_tracker,
                    merge_manager,
                    measure_x,
                    pending_start,
                    conflict_counts,
                    start_indices,
                    _rows_touching_local_indices,
                    observable_manager,
                )
                state.prev.x_prev = x_curr

                # Update merge countdowns and close windows when done
                if merge_manager.active_rough is not None:
                    a, b, rem = merge_manager.active_rough
                    rem -= 1
                    if rem <= 0:
                        merge_manager.active_rough = None
                    else:
                        merge_manager.active_rough = (a, b, rem)
                if merge_manager.active_smooth is not None:
                    a, b, rem = merge_manager.active_smooth
                    rem -= 1
                    if rem <= 0:
                        merge_manager.active_smooth = None
                    else:
                        merge_manager.active_smooth = (a, b, rem)

            elif isinstance(op, Merge):
                k = op.type.strip().lower()
                if k not in {"rough", "smooth"}:
                    raise ValueError("Merge.type must be 'rough' or 'smooth'")
                if k == "rough":
                    if merge_manager.active_rough is not None:
                        raise RuntimeError("A rough merge is already active")
                    seam_pairs = layout.seams.get(("rough", op.a, op.b), [])
                    if not seam_pairs:
                        state.prev.joint_prev[(k, op.a, op.b)] = [None] * 0
                        continue
                    indices_a = {ia for ia, _ in seam_pairs}
                    indices_b = {ib for _, ib in seam_pairs}
                    # Seal X-basis observables on involved patches if still unset
                    for pname in (op.a, op.b):
                        if effective_basis_map.get(pname) == "X" and end_indices.get(pname) is None:
                            end_idx = _last_non_none(list(state.prev.x_prev.get(pname, [])))
                            observable_manager.seal_end(pname, "X", end_idx)
                            end_indices[pname] = end_idx
                    mask_a = self._mask_prev_stabilizers(state.prev.x_prev, op.a, "X", indices_a)
                    mask_b = self._mask_prev_stabilizers(state.prev.x_prev, op.b, "X", indices_b)
                    # Close current X segments touching the seam before the gap
                    for row_idx in mask_a:
                        detector_manager.mark_row_dynamic(op.a, "X", int(row_idx))
                    for row_idx in mask_b:
                        detector_manager.mark_row_dynamic(op.b, "X", int(row_idx))
                    segment_tracker.wrap_close_segment(op.a, "X", detector_manager, mask_a, skip_boundary_rows=True)
                    segment_tracker.wrap_close_segment(op.b, "X", detector_manager, mask_b, skip_boundary_rows=True)
                    merge_manager.active_rough = (op.a, op.b, int(op.rounds))
                else:
                    if merge_manager.active_smooth is not None:
                        raise RuntimeError("A smooth merge is already active")
                    seam_pairs = layout.seams.get(("smooth", op.a, op.b), [])
                    if not seam_pairs:
                        state.prev.joint_prev[(k, op.a, op.b)] = [None] * 0
                        continue
                    indices_a = {ia for ia, _ in seam_pairs}
                    indices_b = {ib for _, ib in seam_pairs}
                    # Seal Z-basis observables on involved patches if still unset
                    for pname in (op.a, op.b):
                        if effective_basis_map.get(pname) == "Z" and end_indices.get(pname) is None:
                            end_idx = _last_non_none(list(state.prev.z_prev.get(pname, [])))
                            observable_manager.seal_end(pname, "Z", end_idx)
                            end_indices[pname] = end_idx
                    mask_a = self._mask_prev_stabilizers(state.prev.z_prev, op.a, "Z", indices_a)
                    mask_b = self._mask_prev_stabilizers(state.prev.z_prev, op.b, "Z", indices_b)
                    for row_idx in mask_a:
                        detector_manager.mark_row_dynamic(op.a, "Z", int(row_idx))
                    for row_idx in mask_b:
                        detector_manager.mark_row_dynamic(op.b, "Z", int(row_idx))
                    segment_tracker.wrap_close_segment(op.a, "Z", detector_manager, mask_a, skip_boundary_rows=True)
                    segment_tracker.wrap_close_segment(op.b, "Z", detector_manager, mask_b, skip_boundary_rows=True)
                    merge_manager.active_smooth = (op.a, op.b, int(op.rounds))
                # Clear any lingering joint history for this seam
                state.prev.joint_prev[(k, op.a, op.b)] = [None] * len(seam_pairs)
                merge_manager.begin_window(k, op.a, op.b, int(op.rounds), state)

            elif isinstance(op, Split):
                k = op.type.strip().lower()
                if k not in {"rough", "smooth"}:
                    raise ValueError("Split.type must be 'rough' or 'smooth'")
                if k == "rough":
                    merge_manager.active_rough = None
                else:
                    merge_manager.active_smooth = None
                merge_manager.end_window()

                # Decrement remaining conflicting merges for involved patches
                for patch_name in (op.a, op.b):
                    basis = "X" if k == "rough" else "Z"
                    key2 = (patch_name, basis)
                    if conflict_counts.get(key2, 0) > 0:
                        conflict_counts[key2] -= 1
                # Snapshot the last measured joint indices for this window before clearing
                merge_manager.last_window_joint[(k, op.a, op.b)] = list(state.prev.joint_prev.get((k, op.a, op.b), []))
                # Wrap-close the seam chain by adding a detector between the first and last joint measurements for each pair
                key = (k, op.a, op.b)
                first_list = list(merge_manager.first_window_joint.get(key, []))
                last_list = list(merge_manager.last_window_joint.get(key, []))
                # Only emit seam wrap if we had at least 2 measured rounds
                if merge_manager.seam_round_counts.get(key, 0) >= 2:
                    from itertools import zip_longest as _ziplg
                    wrap_added = 0
                    for pair_i, (a_idx, b_idx) in enumerate(_ziplg(first_list, last_list, fillvalue=None)):
                        if a_idx is None or b_idx is None or a_idx == b_idx:
                            continue
                        det_id = detector_manager.defer_detector([a_idx, b_idx], f"{k}_wrap", {"seam": key, "pair_idx": pair_i})
                        if detector_manager.force_boundaries:
                            anchor_key = (k, op.a, op.b, pair_i)
                            if anchor_key not in detector_manager.seam_wrap_anchor_emitted:
                                detector_manager.anchor_detector_ids.append(det_id)
                                detector_manager.seam_wrap_anchor_emitted.add(anchor_key)
                                detector_manager.seam_boundary_counts[anchor_key] = detector_manager.seam_boundary_counts.get(anchor_key, 0) + 1
                        wrap_added += 1
                    if wrap_added:
                        merge_manager.seam_wrap_counts[key] = merge_manager.seam_wrap_counts.get(key, 0) + wrap_added
                # Clear stored endpoints for this window
                merge_manager.first_window_joint.pop(key, None)
                state.prev.joint_prev[(k, op.a, op.b)] = []
                merge_manager.seam_round_counts.pop(key, None)

            elif isinstance(op, ParityReadout):
                # Deterministic DEM handling for byproduct extraction.
                # Instead of emitting a fresh logical MPP (which can anti-commute with
                # temporal detectors), derive the byproduct from the *last* round
                # of joint seam checks measured during the preceding merge window.
                if op.type == "ZZ":
                    seam_kind = "rough"
                elif op.type == "XX":
                    seam_kind = "smooth"
                else:
                    raise ValueError("ParityReadout.type must be 'ZZ' or 'XX'")

                key = (seam_kind, op.a, op.b)
                indices = list(merge_manager.last_window_joint.get(key, []))
                
                # Validate that we have joint measurements
                if not indices:
                    # Handle case where no joint measurements were made
                    # This could happen if the merge window had 0 rounds
                    print(f"Warning: No joint measurements found for {key}")

                # Record byproduct info for post-processing (Pauli frame updates, etc.).
                byproduct_info = {
                    "type": op.type,
                    "a": op.a,
                    "b": op.b,
                    "seam_key": key,
                    "indices": indices,          # absolute measurement indices of the last seam round
                    "source": "seam_last_round"  # documentation tag
                }
                state.byproducts.append(byproduct_info)

                # Convenience: primary index (first seam pair) used for Pauli-frame sampling
                primary_idx = indices[0] if indices else None
                
                # Track CNOT operations by grouping ZZ and XX parity readouts
                if state.current_cnot is not None:
                    if op.type == "ZZ":
                        state.current_cnot["m_zz_byproduct"] = byproduct_info
                        state.current_cnot["m_zz_mpp_idx"] = primary_idx
                    elif op.type == "XX":
                        state.current_cnot["m_xx_byproduct"] = byproduct_info
                        state.current_cnot["m_xx_mpp_idx"] = primary_idx
                        state.current_cnot["target"] = op.b  # Update target (for XX, b is the target)
                        state.current_cnot["smooth_window_id"] = merge_manager.get_current_window_id()
                        state.cnot_operations.append(state.current_cnot)
                        state.current_cnot = None
                else:
                    # Start of a CNOT operation (rough merge completed)
                    state.current_cnot = {
                        "control": op.a,
                        "target": op.b,  # This will be updated to actual target when XX comes
                        "ancilla": op.b,  # For ZZ, b is the ancilla
                        "rough_window_id": merge_manager.get_current_window_id(),
                        "smooth_window_id": None,
                        "m_zz_byproduct": byproduct_info if op.type == "ZZ" else None,
                        "m_zz_mpp_idx": primary_idx if op.type == "ZZ" else None,
                        "m_xx_byproduct": None,
                        "m_xx_mpp_idx": None,
                    }
                
                # NOTE: No circuit operations are emitted here to keep the DEM deterministic.

            elif isinstance(op, TerminatePatch):
                # Handle mid-circuit measurement: close segments and mark as terminated
                patch_name = op.patch_id
                if patch_name not in layout.patches or patch_name in state.terminated_patches:
                    continue
                
                # Track detector count before closing segments
                num_detectors_before = len(detector_manager.deferred_detectors)

                # Mark all rows in both bases as having experienced a dynamic event
                patch = layout.patches.get(patch_name)
                if patch is not None:
                    for row_idx in range(len(patch.z_stabs)):
                        detector_manager.mark_row_dynamic(patch_name, "Z", row_idx)
                    for row_idx in range(len(patch.x_stabs)):
                        detector_manager.mark_row_dynamic(patch_name, "X", row_idx)
                
                # Close stabilizer segments for this patch - this creates wrap detectors
                segment_tracker.wrap_close_segment(
                    patch_name,
                    "Z",
                    detector_manager,
                    None,
                    skip_boundary_rows=True,
                )
                segment_tracker.wrap_close_segment(
                    patch_name,
                    "X",
                    detector_manager,
                    None,
                    skip_boundary_rows=True,
                )
                
                # Add boundary anchors for the wrap detectors that were just created
                if detector_manager.force_boundaries:
                    num_detectors_after = len(detector_manager.deferred_detectors)
                    # All detectors added between before and after are wrap detectors from this termination
                    for det_idx in range(num_detectors_before, num_detectors_after):
                        detector_manager.anchor_detector_ids.append(det_idx)
                        detector_manager.boundary_counts_z[patch_name] = detector_manager.boundary_counts_z.get(patch_name, 0) + 1
                
                # Seal the end observable for this patch
                if patch_name in bracket_map:
                    basis = effective_basis_map.get(patch_name, bracket_map[patch_name]).upper()
                    if basis == "Z":
                        end_idx = _last_non_none(list(state.prev.z_prev.get(patch_name, [])))
                        observable_manager.seal_end(patch_name, "Z", end_idx)
                        end_indices[patch_name] = end_idx
                    else:
                        end_idx = _last_non_none(list(state.prev.x_prev.get(patch_name, [])))
                        observable_manager.seal_end(patch_name, "X", end_idx)
                        end_indices[patch_name] = end_idx
                
                # Mark as terminated - it won't be measured in future rounds
                state.terminated_patches.add(patch_name)

            else:
                raise TypeError(f"Unsupported op type: {type(op)!r}")

        # Close any still-open stabilizer segments (no later conflicting gaps)
        # Also close any still-open seam windows by wrapping first↔last if ≥2 rounds measured
        if merge_manager.first_window_joint:
            for key, first_list in list(merge_manager.first_window_joint.items()):
                last_list = list(state.prev.joint_prev.get(key, []))
                if merge_manager.seam_round_counts.get(key, 0) and merge_manager.seam_round_counts.get(key, 0) >= 2:
                    from itertools import zip_longest as _ziplg
                    wrap_added = 0
                    for pair_i, (a_idx, b_idx) in enumerate(_ziplg(first_list, last_list, fillvalue=None)):
                        if a_idx is None or b_idx is None or a_idx == b_idx:
                            continue
                        det_id = detector_manager.defer_detector([a_idx, b_idx], "seam_wrap_finalize", {"seam": key, "pair_idx": pair_i})
                        if detector_manager.force_boundaries:
                            anchor_key = (key[0], key[1], key[2], pair_i)
                            if anchor_key not in detector_manager.seam_wrap_anchor_emitted:
                                detector_manager.anchor_detector_ids.append(det_id)
                                detector_manager.seam_wrap_anchor_emitted.add(anchor_key)
                                detector_manager.seam_boundary_counts[anchor_key] = detector_manager.seam_boundary_counts.get(anchor_key, 0) + 1
                        wrap_added += 1
                    if wrap_added:
                        merge_manager.seam_wrap_counts[key] = merge_manager.seam_wrap_counts.get(key, 0) + wrap_added
            merge_manager.first_window_joint.clear()
        for pname in layout.patches.keys():
            segment_tracker.wrap_close_segment(
                pname,
                "Z",
                detector_manager,
                None,
                skip_boundary_rows=True,
            )
            segment_tracker.wrap_close_segment(
                pname,
                "X",
                detector_manager,
                None,
                skip_boundary_rows=True,
            )

        if emit_explicit_logicals:
            for name, basis in effective_basis_map.items():
                if end_indices.get(name) is not None:
                    continue
                end_idx = self._emit_logical_mpp(circuit, name, basis)
                end_indices[name] = end_idx
                observable_manager.seal_end(name, basis, end_idx)

        # Emit boundary anchors after all merges complete
        # Bracketing: end logicals per patch and observable includes
        # IMPORTANT: do not emit new MPPs here; reuse the last compatible
        # stabilizer measurements from the final round to keep DEM deterministic.
        observable_pairs, basis_labels, deferred_observables = observable_manager.finalize_observables(
            circuit, state, _last_non_none
        )

        # Generate demo measurements
        demo_info, joint_demo_info, snapshot_info = demo_generator.generate_demos(
            circuit, cfg, state, bracket_map, qiskit_circuit
        )

        # Append all deferred detectors at the very end using final measurement count
        detector_manager.emit_all_detectors(
            circuit,
            noise_model={
                "p_x_error": float(getattr(cfg, "p_x_error", 0.0) or 0.0),
                "p_z_error": float(getattr(cfg, "p_z_error", 0.0) or 0.0),
                "p_meas": float(getattr(cfg, "p_meas", 0.0) or 0.0),
            },
        )

        # Diagnostics: compute detector degree per absolute measurement index
        # Only count 2-target detectors (graph edges). Single-target anchors are ignored here.
        degree_violations, odd_degree_details = detector_manager.compute_diagnostics()

        # NOTE: We intentionally do not auto-close per-row temporal chains here.
        # Each stabilizer row must form a single cycle via:
        #  - temporal edges emitted per round (z_temporal/x_temporal),
        #  - wrap edges produced by `segment_tracker.wrap_close_segment` at merge boundaries and at end,
        #  - seam wrap edges produced at Split/finalization for joint checks.
        # If degree_violations remains non-empty, the schedule left a dangling endpoint
        # and must be fixed at the source (segment anchoring or seam suppression), not patched.
        # Append all deferred observables at the very end using final measurement count
        observable_manager.emit_observables(circuit, deferred_observables)

        metadata: Dict[str, object] = {
            "merge_windows": merge_manager.get_windows(),
            "observable_basis": tuple(basis_labels),
            "demo": demo_info,
            "joint_demos": joint_demo_info,
            "cnot_operations": state.cnot_operations,
            "final_snapshot": snapshot_info,
            "byproducts": state.byproducts,
            "boundary_anchors": detector_manager.get_boundary_anchors_metadata(),
            "mwpm_debug": detector_manager.get_diagnostics_metadata(
                merge_manager.get_seam_wrap_counts(),
                *segment_tracker.get_row_wraps(),
            ),
            "explicit_logical_brackets": emit_explicit_logicals,
            "noise_model": {
                "p_x_error": float(getattr(cfg, "p_x_error", 0.0) or 0.0),
                "p_z_error": float(getattr(cfg, "p_z_error", 0.0) or 0.0),
                "p_meas": float(getattr(cfg, "p_meas", 0.0) or 0.0),
            },
        }

        return circuit, observable_pairs, metadata
