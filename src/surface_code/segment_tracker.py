"""Segment tracking for stabilizer wrap-around closure.

This module handles tracking of stabilizer segments for creating wrap detectors
that connect the first and last measurements of each segment.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple


class SegmentTracker:
    """Tracks stabilizer segments for wrap-around closure."""
    
    def __init__(self, boundary_checker: Optional[Callable[[str, str, int], bool]] = None):
        """Initialize segment tracker with empty state."""
        # Track first/last measured indices per stabilizer (segment tracking)
        self.seg_first: Dict[Tuple[str, str], List[Optional[int]]] = {}  # key=(patch,basis)
        self.seg_last: Dict[Tuple[str, str], List[Optional[int]]] = {}
        self.seg_first_det: Dict[Tuple[str, str], List[Optional[int]]] = {}
        self.seg_last_det: Dict[Tuple[str, str], List[Optional[int]]] = {}
        # Track whether each stabilizer row had any temporal edge within the current open segment
        self.seg_had_edge: Dict[Tuple[str, str], List[bool]] = {}
        # Track whether a boundary anchor was emitted for the current open segment per row
        self.seg_boundary_emitted: Dict[Tuple[str, str], List[bool]] = {}
        # Row wrap summaries for diagnostics
        self.z_row_wraps: Dict[str, List[int]] = {}
        self.x_row_wraps: Dict[str, List[int]] = {}
        self._boundary_checker = boundary_checker
    
    def ensure_seg_lists(self, patch_name: str, basis: str, length: int) -> None:
        """Ensure segment lists exist and are long enough.
        
        Args:
            patch_name: Name of the patch
            basis: Basis ("Z" or "X")
            length: Required length
        """
        key = (patch_name, basis)
        if key not in self.seg_first:
            self.seg_first[key] = [None] * length
            self.seg_last[key] = [None] * length
            self.seg_first_det[key] = [None] * length
            self.seg_last_det[key] = [None] * length
            self.seg_had_edge[key] = [False] * length
            self.seg_boundary_emitted[key] = [False] * length
        else:
            # grow if needed
            if len(self.seg_first[key]) < length:
                self.seg_first[key].extend([None] * (length - len(self.seg_first[key])))
                self.seg_last[key].extend([None] * (length - len(self.seg_last[key])))
                self.seg_first_det[key].extend([None] * (length - len(self.seg_first_det[key])))
                self.seg_last_det[key].extend([None] * (length - len(self.seg_last_det[key])))
                self.seg_had_edge[key].extend([False] * (length - len(self.seg_had_edge[key])))
                self.seg_boundary_emitted[key].extend([False] * (length - len(self.seg_boundary_emitted[key])))
    
    def update_segment(self, patch_name: str, basis: str, stab_idx: int, measurement_idx: int) -> None:
        """Update segment tracking for a measurement.
        
        Args:
            patch_name: Name of the patch
            basis: Basis ("Z" or "X")
            stab_idx: Stabilizer row index
            measurement_idx: Absolute measurement index
        """
        key = (patch_name, basis)
        self.ensure_seg_lists(patch_name, basis, stab_idx + 1)
        if measurement_idx is not None:
            self.seg_last[key][stab_idx] = measurement_idx
    
    def set_first_if_none(self, patch_name: str, basis: str, stab_idx: int, measurement_idx: int) -> bool:
        """Set first measurement index if not already set.
        
        Args:
            patch_name: Name of the patch
            basis: Basis ("Z" or "X")
            stab_idx: Stabilizer row index
            measurement_idx: Absolute measurement index
            
        Returns:
            True if this was the first edge (first was None), False otherwise
        """
        key = (patch_name, basis)
        self.ensure_seg_lists(patch_name, basis, stab_idx + 1)
        if self.seg_first[key][stab_idx] is None:
            self.seg_first[key][stab_idx] = measurement_idx
            return True
        return False

    def record_temporal_detector(self, patch_name: str, basis: str, stab_idx: int, detector_id: int) -> None:
        """Track the first and most recent temporal detector id for a stabilizer row."""
        key = (patch_name, basis)
        self.ensure_seg_lists(patch_name, basis, stab_idx + 1)
        first_list = self.seg_first_det[key]
        last_list = self.seg_last_det[key]
        if first_list[stab_idx] is None:
            first_list[stab_idx] = detector_id
        last_list[stab_idx] = detector_id
    
    def mark_had_edge(self, patch_name: str, basis: str, stab_idx: int) -> None:
        """Mark that this row had a temporal edge.
        
        Args:
            patch_name: Name of the patch
            basis: Basis ("Z" or "X")
            stab_idx: Stabilizer row index
        """
        key = (patch_name, basis)
        self.ensure_seg_lists(patch_name, basis, stab_idx + 1)
        if stab_idx < len(self.seg_had_edge[key]):
            self.seg_had_edge[key][stab_idx] = True
    
    def wrap_close_segment(
        self,
        patch_name: str,
        basis: str,
        detector_manager,
        stab_indices: Optional[Set[int]] = None,
        *,
        skip_boundary_rows: bool = False,
    ) -> int:
        """Close a segment and return number of wrap detectors added.
        
        Args:
            patch_name: Name of the patch
            basis: Basis ("Z" or "X")
            detector_manager: DetectorManager instance for emitting detectors
            stab_indices: Optional set of stabilizer indices to close. If None, close all.
            
        Returns:
            Number of wrap detectors added
        """
        key = (patch_name, basis)
        first_list = self.seg_first.get(key, [])
        last_list = self.seg_last.get(key, [])
        had_edge_list = self.seg_had_edge.get(key, [])
        if not first_list:
            return 0
        
        wrap_count = 0
        rng = range(len(first_list)) if stab_indices is None else stab_indices

        def reset_row(row_idx: int) -> None:
            if row_idx < len(first_list):
                self.seg_first[key][row_idx] = None
            if row_idx < len(last_list):
                self.seg_last[key][row_idx] = None
            if key not in self.seg_first_det:
                self.seg_first_det[key] = [None] * len(first_list)
            if key not in self.seg_last_det:
                self.seg_last_det[key] = [None] * len(first_list)
            first_det_list = self.seg_first_det.get(key, [])
            last_det_list = self.seg_last_det.get(key, [])
            if row_idx < len(first_det_list):
                first_det_list[row_idx] = None
            if row_idx < len(last_det_list):
                last_det_list[row_idx] = None
            if row_idx < len(had_edge_list):
                self.seg_had_edge[key][row_idx] = False
            if key not in self.seg_boundary_emitted:
                self.seg_boundary_emitted[key] = [False] * len(first_list)
            b_list = self.seg_boundary_emitted.get(key, [])
            if row_idx < len(b_list):
                b_list[row_idx] = False

        for si in rng:
            if si is None or si >= len(first_list):
                continue
            a = first_list[si]
            b = last_list[si] if si < len(last_list) else None
            is_boundary_row = (
                skip_boundary_rows
                and self._boundary_checker is not None
                and self._boundary_checker(patch_name, basis, si)
            )
            if is_boundary_row:
                anchors_to_add: List[int] = []
                first_det_list = self.seg_first_det.get(key, [])
                last_det_list = self.seg_last_det.get(key, [])
                first_det = first_det_list[si] if si < len(first_det_list) else None
                last_det = last_det_list[si] if si < len(last_det_list) else None
                if isinstance(first_det, int) and first_det >= 0:
                    anchors_to_add.append(first_det)
                if isinstance(last_det, int) and last_det >= 0 and last_det != first_det:
                    anchors_to_add.append(last_det)
                for det_id in anchors_to_add:
                    detector_manager.anchor_detector_ids.append(det_id)
                    if basis == "Z":
                        detector_manager.boundary_counts_z[patch_name] = detector_manager.boundary_counts_z.get(patch_name, 0) + 1
                    else:
                        detector_manager.boundary_counts_x[patch_name] = detector_manager.boundary_counts_x.get(patch_name, 0) + 1
                reset_row(si)
                continue
            # Wrap-close any open segment where endpoints differ
            if a is not None and b is not None and a != b:
                det_id = detector_manager.defer_detector([a, b], f"{basis.lower()}_wrap", {"patch": patch_name, "row": si})
                if (
                    detector_manager.force_boundaries
                    and self._boundary_checker is not None
                    and self._boundary_checker(patch_name, basis, si)
                ):
                    detector_manager.anchor_detector_ids.append(det_id)
                    if basis == "Z":
                        detector_manager.boundary_counts_z[patch_name] = detector_manager.boundary_counts_z.get(patch_name, 0) + 1
                    else:
                        detector_manager.boundary_counts_x[patch_name] = detector_manager.boundary_counts_x.get(patch_name, 0) + 1
                    if basis == "Z":
                        self.z_row_wraps.setdefault(patch_name, []).append(si)
                    else:
                        self.x_row_wraps.setdefault(patch_name, []).append(si)
                wrap_count += 1
            reset_row(si)

        return wrap_count
    
    def get_row_wraps(self) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Get row wrap diagnostics.
        
        Returns:
            Tuple of (z_row_wraps, x_row_wraps) dictionaries
        """
        return self.z_row_wraps, self.x_row_wraps
