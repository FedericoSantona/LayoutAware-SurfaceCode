"""Segment tracking for stabilizer wrap-around closure.

This module handles tracking of stabilizer segments for creating wrap detectors
that connect the first and last measurements of each segment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple


class SegmentTracker:
    """Tracks stabilizer segments for wrap-around closure."""
    
    def __init__(self):
        """Initialize segment tracker with empty state."""
        # Track first/last measured indices per stabilizer (segment tracking)
        self.seg_first: Dict[Tuple[str, str], List[Optional[int]]] = {}  # key=(patch,basis)
        self.seg_last: Dict[Tuple[str, str], List[Optional[int]]] = {}
        # Track whether each stabilizer row had any temporal edge within the current open segment
        self.seg_had_edge: Dict[Tuple[str, str], List[bool]] = {}
        # Track whether a boundary anchor was emitted for the current open segment per row
        self.seg_boundary_emitted: Dict[Tuple[str, str], List[bool]] = {}
        # Row wrap summaries for diagnostics
        self.z_row_wraps: Dict[str, List[int]] = {}
        self.x_row_wraps: Dict[str, List[int]] = {}
    
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
            self.seg_had_edge[key] = [False] * length
            self.seg_boundary_emitted[key] = [False] * length
        else:
            # grow if needed
            if len(self.seg_first[key]) < length:
                self.seg_first[key].extend([None] * (length - len(self.seg_first[key])))
                self.seg_last[key].extend([None] * (length - len(self.seg_last[key])))
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
        for si in rng:
            if si is None or si >= len(first_list):
                continue
            a = first_list[si]
            b = last_list[si] if si < len(last_list) else None
            # Wrap-close any open segment where endpoints differ
            if a is not None and b is not None and a != b:
                detector_manager.defer_detector([a, b], f"{basis.lower()}_wrap", {"patch": patch_name, "row": si})
                if basis == "Z":
                    self.z_row_wraps.setdefault(patch_name, []).append(si)
                else:
                    self.x_row_wraps.setdefault(patch_name, []).append(si)
                wrap_count += 1
            
            # Reset segment for these stabilizers so a new segment can start after the gap
            if si < len(first_list):
                self.seg_first[key][si] = None
            if si < len(last_list):
                self.seg_last[key][si] = None
            if si < len(had_edge_list):
                self.seg_had_edge[key][si] = False
            if key not in self.seg_boundary_emitted:
                self.seg_boundary_emitted[key] = [False] * len(first_list)
            b_list = self.seg_boundary_emitted[key]
            if si < len(b_list):
                b_list[si] = False
        
        return wrap_count
    
    def get_row_wraps(self) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Get row wrap diagnostics.
        
        Returns:
            Tuple of (z_row_wraps, x_row_wraps) dictionaries
        """
        return self.z_row_wraps, self.x_row_wraps

