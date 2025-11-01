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
from .builder_utils import mpp_from_positions, rec_from_abs, add_temporal_detectors_with_index, _mpp_targets_from_pauli
from .pauli import Pauli, conjugate_through_circuit, PauliTracker
from .builder_state import BuilderState, _PrevState
from .detector_manager import DetectorManager
from .segment_tracker import SegmentTracker
from .merge_manager import MergeManager
from .measurement_half import MeasurementHalf


GateTarget = stim.GateTarget


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
        
        DEPRECATED: Use MergeManager.measure_joint_checks instead.
        This method is kept for backward compatibility but delegates to merge_manager.
        """
        # This method is kept for backward compatibility but is no longer used internally
        # Internal code now uses merge_manager.measure_joint_checks directly
        from .merge_manager import MergeManager
        temp_manager = MergeManager(self.layout)
        return temp_manager.measure_joint_checks(circuit, kind, a, b)

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

        # Initialize state and managers
        state = BuilderState()
        force_boundaries = getattr(cfg, "force_boundaries", True)
        boundary_error_prob = getattr(cfg, "boundary_error_prob", 1e-12)
        detector_manager = DetectorManager(force_boundaries=force_boundaries, boundary_error_prob=boundary_error_prob)
        segment_tracker = SegmentTracker()
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

        # Bracketing: start logicals per patch (skip if they would anti-commute with merges)
        start_indices: Dict[str, Optional[int]] = {name: None for name in all_patches}
        end_indices: Dict[str, Optional[int]] = {name: None for name in all_patches}

        circuit.append_operation("TICK")
        # Do NOT emit a start logical MPP. Capture starts from the first
        # compatible stabilizer layer measured later in the schedule.
        pending_start: Dict[str, str] = {
            name: effective_basis_map[name]
            for name in effective_basis_map.keys()
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
                            end_indices[pname] = _last_non_none(list(state.prev.x_prev.get(pname, [])))
                    mask_a = self._mask_prev_stabilizers(state.prev.x_prev, op.a, "X", indices_a)
                    mask_b = self._mask_prev_stabilizers(state.prev.x_prev, op.b, "X", indices_b)
                    # Close current X segments touching the seam before the gap
                    segment_tracker.wrap_close_segment(op.a, "X", detector_manager, mask_a)
                    segment_tracker.wrap_close_segment(op.b, "X", detector_manager, mask_b)
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
                            end_indices[pname] = _last_non_none(list(state.prev.z_prev.get(pname, [])))
                    mask_a = self._mask_prev_stabilizers(state.prev.z_prev, op.a, "Z", indices_a)
                    mask_b = self._mask_prev_stabilizers(state.prev.z_prev, op.b, "Z", indices_b)
                    segment_tracker.wrap_close_segment(op.a, "Z", detector_manager, mask_a)
                    segment_tracker.wrap_close_segment(op.b, "Z", detector_manager, mask_b)
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
                
                # Close stabilizer segments for this patch - this creates wrap detectors
                segment_tracker.wrap_close_segment(patch_name, "Z", detector_manager, None)
                segment_tracker.wrap_close_segment(patch_name, "X", detector_manager, None)
                
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
                        end_indices[patch_name] = _last_non_none(list(state.prev.z_prev.get(patch_name, [])))
                    else:
                        end_indices[patch_name] = _last_non_none(list(state.prev.x_prev.get(patch_name, [])))
                
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
            segment_tracker.wrap_close_segment(pname, "Z", detector_manager, None)
            segment_tracker.wrap_close_segment(pname, "X", detector_manager, None)

        # Emit boundary anchors after all merges complete
        # Bracketing: end logicals per patch and observable includes
        # IMPORTANT: do not emit new MPPs here; reuse the last compatible
        # stabilizer measurements from the final round to keep DEM deterministic.
        observable_pairs: List[Tuple[int, int]] = []
        basis_labels: List[str] = []
        observable_index = 0
        deferred_observables: List[Tuple[Optional[int], Optional[int], int]] = []

        # At the very end, fallback-seal any observables that didn't conflict
        for pname, basis in effective_basis_map.items():
            if pname not in end_indices or end_indices[pname] is not None:
                continue
            if basis == "Z":
                end_indices[pname] = _last_non_none(list(state.prev.z_prev.get(pname, [])))
            else:
                end_indices[pname] = _last_non_none(list(state.prev.x_prev.get(pname, [])))

        # Only bracket patches that are explicitly in bracket_map (excludes ancillas and terminated patches)
        for name in bracket_map.keys():
            if name not in layout.patches or name in state.terminated_patches:
                continue  # Skip if patch doesn't exist or was terminated
            requested_basis = bracket_map[name].upper()
            effective_basis = effective_basis_map.get(name, requested_basis)

            # Prefer a pre-sealed end (set when a conflicting window began)
            end_idx = end_indices.get(name)
            if end_idx is None:
                if effective_basis == "Z":
                    end_idx = _last_non_none(list(state.prev.z_prev.get(name, [])))
                else:
                    end_idx = _last_non_none(list(state.prev.x_prev.get(name, [])))
                end_indices[name] = end_idx
            start_idx = start_indices[name]

            targets: List[GateTarget] = []
            if start_idx is not None:
                targets.append(rec_from_abs(circuit, start_idx))
            if end_idx is not None:
                targets.append(rec_from_abs(circuit, end_idx))

            if targets:
                # Defer OBSERVABLE_INCLUDE to the very end to avoid later anti-commuting MPPs
                deferred_observables.append((start_idx, end_idx, observable_index))
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
        
        # Prepare mapping between logical names and qiskit indices using the
        # explicit QuantumCircuit qubit order. This implicitly excludes any
        # ancilla patches that are not part of the original circuit.
        name_to_idx: Dict[str, int] = {}
        idx_to_name: Dict[int, str] = {}
        n_logical = 0
        if qiskit_circuit is not None:
            n_logical = qiskit_circuit.num_qubits
            for qi in range(n_logical):
                name = f"q{qi}"
                name_to_idx[name] = qi
                idx_to_name[qi] = name
        
        # Initialize snapshot_info outside conditional blocks
        snapshot_info = {"enabled": False}
        
        if demo_bases and qiskit_circuit is not None:

            # Correlation pairs from compiled CNOT operations (fallback to first two logicals).
            correlation_pairs: List[Tuple[str, str]] = []
            for cnot_op in state.cnot_operations:
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

            # Helper to emit a joint correlator (ZZ or XX) for a logical pair
            def _emit_joint_for_pair(basis: str, q0_name: str, q1_name: str):
                # Heisenberg-frame: measure U†(ZZ/XX)U at the end
                if basis == "X":
                    op = Pauli.two_xx(n_logical, name_to_idx[q0_name], name_to_idx[q1_name])
                else:
                    op = Pauli.two_zz(n_logical, name_to_idx[q0_name], name_to_idx[q1_name])
                conj = conjugate_through_circuit(op, qiskit_circuit)
                mpp_targets, axes_map = _mpp_targets_from_pauli(conj, layout, idx_to_name)
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
                        if q0_name not in layout.patches or q1_name not in layout.patches:
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

                # ----- Final computational-basis snapshot (single basis mode only) -----
                if qiskit_circuit is not None and demo_bases:
                    snapshot_basis = demo_bases[0].upper()
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

        # Append all deferred detectors at the very end using final measurement count
        detector_manager.emit_all_detectors(circuit)

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
        if deferred_observables:
            final_m2 = circuit.num_measurements
            for s_idx, e_idx, obs_k in deferred_observables:
                obs_targets: List[GateTarget] = []
                if s_idx is not None:
                    obs_targets.append(stim.target_rec(s_idx - final_m2))
                if e_idx is not None:
                    obs_targets.append(stim.target_rec(e_idx - final_m2))
                if obs_targets:
                    circuit.append_operation("OBSERVABLE_INCLUDE", obs_targets, obs_k)

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
        }

        return circuit, observable_pairs, metadata


def augment_dem_with_boundary_anchors(
    dem: stim.DetectorErrorModel,
    anchor_detector_ids: List[int],
    error_probability: float,
) -> stim.DetectorErrorModel:
    """Return a new DEM with boundary edges injected at given detector ids.

    Each anchor id k receives a tiny-probability single-detector error line:
        error p Dk
    which creates a boundary edge for MWPM without altering physical noise.
    """
    if not anchor_detector_ids or not isinstance(error_probability, (int, float)):
        return dem
    if error_probability <= 0:
        return dem
    # Deduplicate and keep stable order
    seen: Set[int] = set()
    ordered_ids: List[int] = []
    for k in anchor_detector_ids:
        if isinstance(k, int) and k >= 0 and k not in seen:
            seen.add(k)
            ordered_ids.append(k)
    if not ordered_ids:
        return dem
    # Append lines to DEM text
    dem_text = str(dem)
    if dem_text and not dem_text.endswith("\n"):
        dem_text += "\n"
    p_str = f"{float(error_probability):.12g}"
    for k in ordered_ids:
        dem_text += f"error({p_str}) D{k}\n"
    return stim.DetectorErrorModel(dem_text)
