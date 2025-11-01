"""Detector management for deferred detector emission and tracking.

This module handles all detector-related operations including deferred emission,
boundary anchor tracking, and diagnostic computation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import stim


GateTarget = stim.GateTarget


class DetectorManager:
    """Manages deferred detector emission and tracking."""
    
    def __init__(self, force_boundaries: bool = True, boundary_error_prob: float = 1e-12):
        """Initialize detector manager.
        
        Args:
            force_boundaries: Whether to force boundary anchors
            boundary_error_prob: Error probability for boundary anchors
        """
        self.force_boundaries = force_boundaries
        self.boundary_error_prob = boundary_error_prob
        
        # Deferred detector storage
        self.deferred_detectors: List[List[int]] = []
        self.edge_records: List[Dict[str, object]] = []
        
        # Boundary anchor tracking
        self.anchor_detector_ids: List[int] = []
        self.seam_pair_boundary_emitted: Set[Tuple[str, str, str, int, int]] = set()
        self.seam_wrap_anchor_emitted: Set[Tuple[str, str, str, int]] = set()
        
        # Diagnostics tracking
        self.row_temporal_degree: Dict[Tuple[str, str, int], Dict[int, int]] = {}
        self.boundary_counts_z: Dict[str, int] = {}
        self.boundary_counts_x: Dict[str, int] = {}
        self.seam_boundary_counts: Dict[Tuple[str, str, str, int], int] = {}
    
    def defer_detector(
        self,
        abs_indices: List[int],
        tag: str,
        context: Optional[Dict[str, object]] = None
    ) -> int:
        """Add a deferred detector and return its index.
        
        Args:
            abs_indices: Absolute measurement indices for the detector
            tag: Tag identifying the detector type (e.g., "z_temporal", "x_temporal")
            context: Optional context dictionary with metadata
            
        Returns:
            Detector index (position in deferred_detectors list)
        """
        self.deferred_detectors.append(list(abs_indices))
        self.edge_records.append({
            "indices": list(abs_indices),
            "tag": tag,
            "context": dict(context or {}),
        })
        
        # Track per-row temporal degrees for x_temporal/z_temporal
        if tag in ("x_temporal", "z_temporal"):
            patch_name = (context or {}).get("patch")
            row = (context or {}).get("row")
            if isinstance(patch_name, str) and isinstance(row, int):
                basis = "X" if tag.startswith("x_") else "Z"
                key_rt = (basis, patch_name, int(row))
                deg_map = self.row_temporal_degree.setdefault(key_rt, {})
                for idx in abs_indices:
                    deg_map[idx] = deg_map.get(idx, 0) + 1
        
        # Return a stable index even though it's not used downstream.
        return len(self.deferred_detectors) - 1
    
    def emit_all_detectors(self, circuit: stim.Circuit) -> None:
        """Emit all deferred detectors to the circuit.
        
        Args:
            circuit: The stim circuit to append detectors to
        """
        if not self.deferred_detectors:
            return
        
        final_m = circuit.num_measurements
        for abs_list in self.deferred_detectors:
            targets: List[GateTarget] = []
            for k, abs_idx in enumerate(abs_list):
                # rec offsets are relative to current #measurements
                targets.append(stim.target_rec(abs_idx - final_m))
            circuit.append_operation("DETECTOR", targets)
    
    def compute_diagnostics(self) -> Tuple[List[int], Dict[int, List[Dict[str, object]]]]:
        """Compute detector degree diagnostics.
        
        Returns:
            Tuple of (degree_violations, odd_degree_details)
        """
        # Diagnostics: compute detector degree per absolute measurement index
        # Only count 2-target detectors (graph edges). Single-target anchors are ignored here.
        det_degree: Dict[int, int] = {}
        for abs_list in self.deferred_detectors:
            if len(abs_list) != 2:
                continue
            for abs_idx in abs_list:
                det_degree[abs_idx] = det_degree.get(abs_idx, 0) + 1
        
        degree_violations = [idx for idx, deg in det_degree.items() if deg not in (0, 2)]
        
        # Build per-index provenance for odd-degree indices
        odd_degree_details: Dict[int, List[Dict[str, object]]] = {}
        if degree_violations:
            for rec in self.edge_records:
                indices = rec.get("indices", [])
                if not isinstance(indices, list) or len(indices) != 2:
                    continue
                a_i, b_i = indices[0], indices[1]
                if a_i in degree_violations:
                    odd_degree_details.setdefault(a_i, []).append({
                        "tag": rec.get("tag"),
                        "neighbor": b_i,
                        "context": rec.get("context", {}),
                    })
                if b_i in degree_violations:
                    odd_degree_details.setdefault(b_i, []).append({
                        "tag": rec.get("tag"),
                        "neighbor": a_i,
                        "context": rec.get("context", {}),
                    })
        
        return degree_violations, odd_degree_details
    
    def get_boundary_anchors_metadata(self) -> Dict[str, object]:
        """Get boundary anchors metadata in the expected format.
        
        Returns:
            Dictionary with "detector_ids" and "epsilon" keys
        """
        return {
            "detector_ids": list(self.anchor_detector_ids),
            "epsilon": float(self.boundary_error_prob),
        }
    
    def get_diagnostics_metadata(
        self,
        seam_wrap_counts: Dict[Tuple[str, str, str], int],
        z_row_wraps: Dict[str, List[int]],
        x_row_wraps: Dict[str, List[int]],
    ) -> Dict[str, object]:
        """Get MWPM debug diagnostics metadata.
        
        Args:
            seam_wrap_counts: Seam wrap counts from merge manager
            z_row_wraps: Z row wraps from segment tracker
            x_row_wraps: X row wraps from segment tracker
            
        Returns:
            Dictionary with mwpm_debug structure
        """
        degree_violations, odd_degree_details = self.compute_diagnostics()
        
        return {
            "seam_wrap_counts": {str(k): v for k, v in seam_wrap_counts.items()},
            "row_wraps": {
                "Z": {k: list(vs) for k, vs in z_row_wraps.items()},
                "X": {k: list(vs) for k, vs in x_row_wraps.items()},
            },
            "degree_violations": degree_violations,
            "odd_degree_details": odd_degree_details,
            "edge_records_count": len(self.edge_records),
            "boundary_counts": {
                "Z": {k: int(v) for k, v in self.boundary_counts_z.items()},
                "X": {k: int(v) for k, v in self.boundary_counts_x.items()},
                "seam": {str(k): int(v) for k, v in self.seam_boundary_counts.items()},
            },
        }

