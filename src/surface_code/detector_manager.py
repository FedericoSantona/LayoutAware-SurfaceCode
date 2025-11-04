"""Detector management for deferred detector emission and tracking.

This module handles all detector-related operations including deferred emission,
boundary anchor tracking, and diagnostic computation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import stim


GateTarget = stim.GateTarget


@dataclass
class _TagStats:
    emitted: int = 0
    kept: int = 0
    dropped: int = 0
    present_in_error: int = 0


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
        self.detector_context: Dict[int, Dict[str, object]] = {}
        
        # Boundary anchor tracking
        self.anchor_detector_ids: List[int] = []
        self.seam_pair_boundary_emitted: Set[Tuple[str, str, str, int, int]] = set()
        self.seam_wrap_anchor_emitted: Set[Tuple[str, str, str, int]] = set()
        
        # Detector pruning / liveness tracking
        self._detector_live_mask: List[bool] = []
        self._old_to_new_index: Dict[int, int] = {}
        self._noise_model: Dict[str, float] = {"p_x_error": 0.0, "p_z_error": 0.0, "p_meas": 0.0}
        self._tag_stats: Dict[str, _TagStats] = {}
        self._row_dynamic: Set[Tuple[str, str, int]] = set()
        self._seam_dynamic: Set[Tuple[str, str, str]] = set()

        # Diagnostics tracking
        self.row_temporal_degree: Dict[Tuple[str, str, int], Dict[int, int]] = {}
        self.boundary_counts_z: Dict[str, int] = {}
        self.boundary_counts_x: Dict[str, int] = {}
        self.seam_boundary_counts: Dict[Tuple[str, str, str, int], int] = {}

    def mark_row_dynamic(self, patch: str, basis: str, row: int) -> None:
        """Mark that a stabilizer row participates in a dynamic check event."""
        if not isinstance(patch, str):
            return
        if not isinstance(row, int):
            return
        basis_u = str(basis).upper()
        if basis_u not in {"Z", "X"}:
            return
        self._row_dynamic.add((patch, basis_u, int(row)))

    def is_row_dynamic(self, patch: str, basis: str, row: int) -> bool:
        """Return True if the given stabilizer row has been marked dynamic."""
        if not isinstance(patch, str) or not isinstance(row, int):
            return False
        return (patch, str(basis).upper(), int(row)) in self._row_dynamic

    def mark_seam_dynamic(self, seam_key: Tuple[str, str, str]) -> None:
        """Mark that a seam component participates in dynamic events."""
        if not isinstance(seam_key, tuple) or len(seam_key) < 3:
            return
        kind, a, b = seam_key[:3]
        self._seam_dynamic.add((str(kind), str(a), str(b)))

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
        tag = str(tag)
        ctx = dict(context or {})
        self.deferred_detectors.append(list(abs_indices))
        self.edge_records.append({
            "indices": list(abs_indices),
            "tag": tag,
            "context": ctx,
        })
        det_idx = len(self.deferred_detectors) - 1
        self.detector_context[det_idx] = {
            "tag": tag,
            "context": ctx,
        }

        stats = self._tag_stats.setdefault(tag, _TagStats())
        stats.emitted += 1
        
        # Track per-row temporal degrees for x_temporal/z_temporal
        if tag in ("x_temporal", "z_temporal"):
            patch_name = ctx.get("patch")
            row = ctx.get("row")
            if isinstance(patch_name, str) and isinstance(row, int):
                basis = "X" if tag.startswith("x_") else "Z"
                key_rt = (basis, patch_name, int(row))
                deg_map = self.row_temporal_degree.setdefault(key_rt, {})
                for idx in abs_indices:
                    deg_map[idx] = deg_map.get(idx, 0) + 1

        # Mark dynamic structures for wrap and seam detectors
        if tag in ("rough_temporal", "smooth_temporal", "rough_wrap", "smooth_wrap", "seam_wrap_finalize"):
            seam_ctx = ctx.get("seam")
            if isinstance(seam_ctx, tuple):
                self.mark_seam_dynamic(tuple(seam_ctx[:3]))
        
        # Return a stable index even though it's not used downstream.
        return len(self.deferred_detectors) - 1
    
    def emit_all_detectors(
        self,
        circuit: stim.Circuit,
        noise_model: Optional[Dict[str, float]] = None,
    ) -> None:
        """Emit all deferred detectors to the circuit.
        
        Args:
            circuit: The stim circuit to append detectors to
            noise_model: Optional map with noise parameters (p_x_error, p_z_error, p_meas)
        """
        if not self.deferred_detectors:
            return

        if noise_model is not None:
            self._noise_model = {
                "p_x_error": float(noise_model.get("p_x_error", 0.0) or 0.0),
                "p_z_error": float(noise_model.get("p_z_error", 0.0) or 0.0),
                "p_meas": float(noise_model.get("p_meas", 0.0) or 0.0),
            }

        live_mask = self._compute_live_mask()
        self._detector_live_mask = list(live_mask)

        for stats in self._tag_stats.values():
            stats.kept = 0
            stats.dropped = 0

        new_detectors: List[List[int]] = []
        new_edge_records: List[Dict[str, object]] = []
        new_context: Dict[int, Dict[str, object]] = {}
        old_to_new: Dict[int, int] = {}

        for idx, keep in enumerate(live_mask):
            rec = self.edge_records[idx]
            tag = str(rec.get("tag", ""))
            stats = self._tag_stats.setdefault(tag, _TagStats())
            if keep:
                stats.kept += 1
                new_idx = len(new_detectors)
                new_detectors.append(self.deferred_detectors[idx])
                new_edge_records.append(rec)
                new_context[new_idx] = dict(self.detector_context.get(idx, {}))
                old_to_new[idx] = new_idx
            else:
                stats.dropped += 1

        self._old_to_new_index = old_to_new
        self.deferred_detectors = new_detectors
        self.edge_records = new_edge_records
        self.detector_context = new_context

        # Remap anchors to filtered detectors and deduplicate.
        remapped: List[int] = []
        seen: Set[int] = set()
        for det_id in self.anchor_detector_ids:
            new_id = old_to_new.get(det_id)
            if new_id is None or new_id in seen:
                continue
            seen.add(new_id)
            remapped.append(new_id)
        self.anchor_detector_ids = remapped
        final_m = circuit.num_measurements
        for abs_list in self.deferred_detectors:
            targets: List[GateTarget] = []
            for k, abs_idx in enumerate(abs_list):
                # rec offsets are relative to current #measurements
                targets.append(stim.target_rec(abs_idx - final_m))
            circuit.append_operation("DETECTOR", targets)
    
    def _compute_live_mask(self) -> List[bool]:
        """Compute which deferred detectors should be kept."""
        if not self.deferred_detectors:
            return []
        mask: List[bool] = []
        for idx, rec in enumerate(self.edge_records):
            tag = str(rec.get("tag", ""))
            context = rec.get("context", {}) or {}
            keep = self._tag_is_live(tag, context)
            mask.append(bool(keep))
        return mask

    def _tag_is_live(self, tag: str, context: Dict[str, object]) -> bool:
        """Determine if a detector tag should be retained under the noise model."""
        tag = str(tag)
        p_meas = self._noise_model.get("p_meas", 0.0) or 0.0
        p_x = self._noise_model.get("p_x_error", 0.0) or 0.0
        p_z = self._noise_model.get("p_z_error", 0.0) or 0.0

        if tag in ("z_temporal", "z_wrap"):
            return (p_meas > 0.0) or (p_x > 0.0) or self._row_is_dynamic(context, "Z")
        if tag in ("x_temporal", "x_wrap"):
            return (p_meas > 0.0) or (p_z > 0.0) or self._row_is_dynamic(context, "X")
        if tag in ("rough_temporal", "smooth_temporal"):
            seam_key = self._context_seam_key(context)
            return seam_key is not None and self._seam_is_dynamic(seam_key)
        if tag in ("rough_wrap", "smooth_wrap", "seam_wrap_finalize"):
            seam_key = self._context_seam_key(context)
            return seam_key is not None and self._seam_is_dynamic(seam_key)
        if tag == "z_spatial":
            # Z stabilizers flip when X-type data noise is present and the row couples to data.
            return p_x > 0.0 or self._row_is_dynamic(context, "Z")
        if tag == "x_spatial":
            return p_z > 0.0 or self._row_is_dynamic(context, "X")
        # Conservatively keep unknown tags.
        return True

    def _row_is_dynamic(self, context: Dict[str, object], basis_hint: str) -> bool:
        patch = context.get("patch")
        row = context.get("row")
        if not isinstance(patch, str) or not isinstance(row, int):
            return False
        return (patch, basis_hint.upper(), int(row)) in self._row_dynamic

    def _context_seam_key(self, context: Dict[str, object]) -> Optional[Tuple[str, str, str]]:
        seam = context.get("seam")
        if isinstance(seam, tuple) and len(seam) >= 3:
            kind, a, b = seam[:3]
            return (str(kind), str(a), str(b))
        return None

    def _seam_is_dynamic(self, seam_key: Tuple[str, str, str]) -> bool:
        return seam_key in self._seam_dynamic

    def record_present_in_error(self, detector_ids: Iterable[int]) -> None:
        """Record detectors that participate in ERROR lines for diagnostics."""
        seen: Set[int] = set(int(d) for d in detector_ids if isinstance(d, int))
        for det_id in seen:
            ctx = self.detector_context.get(det_id)
            if not ctx:
                continue
            tag = str(ctx.get("tag", ""))
            stats = self._tag_stats.setdefault(tag, _TagStats())
            stats.present_in_error += 1

    def get_tag_stats(self) -> Dict[str, Dict[str, int]]:
        """Return per-tag emission statistics."""
        return {tag: asdict(stats) for tag, stats in self._tag_stats.items()}
    
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
        unique_ids: List[int] = []
        seen: Set[int] = set()
        for det_id in self.anchor_detector_ids:
            if det_id in seen:
                continue
            seen.add(int(det_id))
            unique_ids.append(int(det_id))
        return {
            "detector_ids": unique_ids,
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
            "detector_context": {int(k): dict(v) for k, v in self.detector_context.items()},
            "tag_stats": self.get_tag_stats(),
        }
