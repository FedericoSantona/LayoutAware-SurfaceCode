"""Unified measurement half logic for Z and X bases.

This module eliminates code duplication between Z and X measurement halves
by providing a unified implementation that handles both bases.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import stim

from .builder_state import BuilderState
from .detector_manager import DetectorManager
from .layout import Layout
from .merge_manager import MergeManager
from .segment_tracker import SegmentTracker


class MeasurementHalf:
    """Unified logic for Z/X measurement halves."""
    
    def __init__(self, builder, basis: str):
        """Initialize measurement half.
        
        Args:
            builder: GlobalStimBuilder instance
            basis: Basis ("Z" or "X")
        """
        self.builder = builder
        self.basis = basis.upper()
        self.opposite_basis = "X" if basis.upper() == "Z" else "Z"
        
        if self.basis not in {"Z", "X"}:
            raise ValueError(f"Basis must be 'Z' or 'X', got {basis}")
    
    def measure_round(
        self,
        circuit: stim.Circuit,
        names: List[str],
        cfg,
        state: BuilderState,
        detector_manager: DetectorManager,
        segment_tracker: SegmentTracker,
        merge_manager: MergeManager,
        measure_basis: bool,
        pending_start: Dict[str, str],
        conflict_counts: Dict[Tuple[str, str], int],
        start_indices: Dict[str, Optional[int]],
        _rows_touching_local_indices,
        observable_manager=None,  # Optional ObservableManager
        round_index: int = 0,
    ) -> Dict[str, List[Optional[int]]]:
        """Measure one half (Z or X) of a round.
        
        Args:
            circuit: stim.Circuit to append operations to
            names: List of patch names to measure
            cfg: PhenomenologicalStimConfig
            state: BuilderState instance
            detector_manager: DetectorManager instance
            segment_tracker: SegmentTracker instance
            merge_manager: MergeManager instance
            measure_basis: Whether to actually measure (can be False for skip)
            pending_start: Dict mapping patch names to basis waiting for start
            conflict_counts: Dict tracking merge conflicts
            start_indices: Dict to update with start indices
            _rows_touching_local_indices: Helper function from builder
            observable_manager: Optional ObservableManager instance
            round_index: Temporal round counter for this basis
            
        Returns:
            Dictionary mapping patch names to current measurement indices
        """
        # Initialize current measurements from previous
        prev_dict = state.prev.z_prev if self.basis == "Z" else state.prev.x_prev
        curr_measurements = {name: (list(vals) if vals is not None else []) for name, vals in prev_dict.items()}
        
        # Pending seam detector emission (emit after patch MPPs)
        pending_seam_key = None
        pending_seam_prev = None
        pending_seam_curr = None
        
        if measure_basis:
            # Determine which merge type conflicts with this basis
            # Z conflicts with SMOOTH (XX), X conflicts with ROUGH (ZZ)
            conflicting_merge_type = "smooth" if self.basis == "Z" else "rough"
            active_conflicting_merge = merge_manager.active_smooth if self.basis == "Z" else merge_manager.active_rough
            
            # Compute skip indices for stabilizers that touch conflicting seams
            skip_indices: Dict[str, Set[int]] = {}
            if active_conflicting_merge is not None:
                seam_a, seam_b, _ = active_conflicting_merge
                pairs = self.builder.layout.seams.get((conflicting_merge_type, seam_a, seam_b), [])
                if pairs:
                    skip_indices[seam_a] = {ia for ia, _ in pairs}
                    skip_indices[seam_b] = {ib for _, ib in pairs}
            
            # Measure patch stabilizers
            meas_curr = self.builder._measure_patch_stabilizers(
                circuit,
                names,
                self.basis,
                skip_indices=skip_indices if skip_indices else None,
                prev_map=prev_dict,
                p_meas=cfg.p_meas,
            )

            # (No spatial detectors are emitted here; spatial connectivity is
            # captured via error hyperedges in the DEM and temporal chaining.)
            # Process each patch
            for name in names:
                # Segment tracking: update last seen; first is set on first temporal edge
                meas_list = list(meas_curr.get(name, []))
                segment_tracker.ensure_seg_lists(name, self.basis, len(meas_list))
                for si, idx_abs in enumerate(meas_list):
                    if idx_abs is not None:
                        segment_tracker.update_segment(name, self.basis, si, idx_abs)
                
                # Suppress temporal detectors while this patch is in an active conflicting merge window
                # Per-row suppression: only skip rows that touch the active seam
                row_skip: Set[int] = set()
                if active_conflicting_merge is not None and name in {active_conflicting_merge[0], active_conflicting_merge[1]}:
                    rows = self.builder.layout.seams.get((conflicting_merge_type, active_conflicting_merge[0], active_conflicting_merge[1]), [])
                    if rows:
                        if name == active_conflicting_merge[0]:
                            local_set = {ia for ia, _ in rows}
                        else:
                            local_set = {ib for _, ib in rows}
                        row_skip = _rows_touching_local_indices(name, self.basis, local_set)
                        for skipped in row_skip:
                            detector_manager.mark_row_dynamic(name, self.basis, int(skipped))
                
                # Record temporal detector edges (absolute measurement indices) for non-skipped rows
                # CRITICAL: Get prev_dict fresh from state to ensure we have the latest values
                # After warmup round, state.prev.z_prev/x_prev gets reassigned, so we need to refresh the reference
                # Refresh prev_dict reference to ensure we have the latest state after potential reassignment
                prev_dict_refresh = state.prev.z_prev if self.basis == "Z" else state.prev.x_prev
                p_list = list(prev_dict_refresh.get(name, []))
                c_list = list(meas_curr.get(name, []))
                
                # Pad p_list to match c_list length if needed (for cases where stabilizer count changed)
                # This ensures zip_longest works correctly
                if len(p_list) < len(c_list):
                    p_list.extend([None] * (len(c_list) - len(p_list)))
                from itertools import zip_longest as _ziplg
                has_history = any(val is not None for val in p_list)
                for si, (a, b) in enumerate(_ziplg(p_list, c_list, fillvalue=None)):
                    if si in row_skip:
                        continue
                    if a is None or b is None or a == b:
                        continue
                    
                    # On first edge within a fresh segment, set first=a so wrap uses [a,last]
                    segment_tracker.ensure_seg_lists(name, self.basis, max(len(p_list), len(c_list)))
                    first_edge = segment_tracker.set_first_if_none(name, self.basis, si, a)
                    segment_tracker.update_segment(name, self.basis, si, b)
                    
                    # Emit temporal edge
                    temporal_type = f"{self.basis.lower()}_temporal"
                    _det_id = detector_manager.defer_detector(
                        [a, b],
                        temporal_type,
                        {"patch": name, "row": si, "round": int(round_index)},
                    )
                    segment_tracker.record_temporal_detector(name, self.basis, si, _det_id)
                    is_boundary_row = self.builder.is_boundary_row(name, self.basis, si)
                    if (
                        is_boundary_row
                        and first_edge
                        and not segment_tracker.has_start_anchor(name, self.basis, si)
                        and not detector_manager.is_row_dynamic(name, self.basis, si)
                    ):
                        detector_manager.anchor_detector_ids.append(_det_id)
                        segment_tracker.mark_start_anchor(name, self.basis, si)
                    if detector_manager.force_boundaries:
                        segment_tracker.ensure_seg_lists(name, self.basis, si + 1)
                        key = (name, self.basis)
                        emitted_flags = segment_tracker.seg_boundary_emitted.get(key, [])
                        if is_boundary_row:
                            if si < len(emitted_flags):
                                emitted_flags[si] = True
                            detector_manager.anchor_detector_ids.append(_det_id)
                            if self.basis == "Z":
                                detector_manager.boundary_counts_z[name] = detector_manager.boundary_counts_z.get(name, 0) + 1
                            else:
                                detector_manager.boundary_counts_x[name] = detector_manager.boundary_counts_x.get(name, 0) + 1
                    segment_tracker.mark_had_edge(name, self.basis, si)

                curr_measurements[name] = list(meas_curr.get(name, []))

                # Capture start index if conditions are met
                if (
                    name in pending_start
                    and pending_start[name] == self.basis
                    and not (
                        (merge_manager.active_smooth is not None and name in {merge_manager.active_smooth[0], merge_manager.active_smooth[1]})
                        or (merge_manager.active_rough is not None and name in {merge_manager.active_rough[0], merge_manager.active_rough[1]})
                    )
                    and conflict_counts.get((name, self.basis), 0) == 0
                ):
                    indices = meas_curr.get(name, [])
                    first_idx = next((idx for idx in indices if idx is not None), None)
                    if first_idx is not None:
                        start_indices[name] = first_idx
                        if observable_manager is not None:
                            observable_manager.capture_start(name, first_idx)
                        pending_start.pop(name, None)
        else:
            # If not measuring, preserve previous measurements
            for name in names:
                if name not in curr_measurements:
                    curr_measurements[name] = list(prev_dict.get(name, []))
        
        # Handle seam operations for this basis
        # Z uses rough seams, X uses smooth seams
        seam_type = "rough" if self.basis == "Z" else "smooth"
        active_seam_merge = merge_manager.active_rough if self.basis == "Z" else merge_manager.active_smooth
        
        if measure_basis and active_seam_merge is not None:
            seam_a, seam_b, _ = active_seam_merge
            pending_seam_key = (seam_type, seam_a, seam_b)
            pending_seam_prev = list(state.prev.joint_prev.get(pending_seam_key, []))
            pending_seam_curr = merge_manager.measure_joint_checks(circuit, seam_type, seam_a, seam_b, p_meas=cfg.p_meas)
            if pending_seam_curr:
                merge_manager.seam_round_counts[pending_seam_key] = merge_manager.seam_round_counts.get(pending_seam_key, 0) + 1
            # Capture the first joint round indices for wrap-around closure
            if pending_seam_prev is None or not pending_seam_prev or all(x is None for x in pending_seam_prev):
                if pending_seam_curr:
                    merge_manager.first_window_joint[pending_seam_key] = list(pending_seam_curr)
        
        # Emit seam temporal detectors *after* all MPPs of this half
        if pending_seam_key is not None and pending_seam_curr is not None:
            if merge_manager.seam_round_counts.get(pending_seam_key, 0) >= 2:
                p_list = list(pending_seam_prev if pending_seam_prev is not None else [])
                c_list = list(pending_seam_curr)
                from itertools import zip_longest as _ziplg
                for pair_i, (a, b) in enumerate(_ziplg(p_list, c_list, fillvalue=None)):
                    if a is None or b is None or a == b:
                        continue
                    key_emit = (pending_seam_key[0], pending_seam_key[1], pending_seam_key[2], pair_i, merge_manager.get_current_window_id())
                    temporal_type = f"{seam_type}_temporal"
                    if detector_manager.force_boundaries and key_emit not in detector_manager.seam_pair_boundary_emitted:
                        _det_id = detector_manager.defer_detector([a, b], temporal_type, {"seam": pending_seam_key, "pair_idx": pair_i})
                        detector_manager.anchor_detector_ids.append(_det_id)
                        detector_manager.seam_pair_boundary_emitted.add(key_emit)
                    else:
                        detector_manager.defer_detector([a, b], temporal_type, {"seam": pending_seam_key, "pair_idx": pair_i})
                state.prev.joint_prev[pending_seam_key] = list(pending_seam_curr)
        
        return curr_measurements
