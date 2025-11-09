"""Helpers for printing Detector Error Model diagnostics during verbose runs."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import os

import stim  # type: ignore

from .dem_utils import _is_error_instruction, build_components_from_dem


_TRUTHY = {"1", "true", "t", "yes", "on"}


def env_dem_debug_enabled() -> bool:
    """Return True when SURFACE_CODE_DEM_DEBUG enables diagnostics."""
    value = os.getenv("SURFACE_CODE_DEM_DEBUG")
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


class _UnionFind:
    """Lightweight union-find for component analysis."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while item != root:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _extract_targets(inst: stim.DemInstruction) -> Tuple[List[int], List[int]]:  # type: ignore[name-defined]
    """Return detector and observable ids referenced by an ERROR instruction."""
    det_ids: List[int] = []
    obs_ids: List[int] = []
    try:
        targets = list(inst.targets_copy()) if hasattr(inst, "targets_copy") else list(getattr(inst, "targets", ()))
    except Exception:
        targets = []
    for target in targets:
        try:
            if hasattr(target, "is_detector_id") and target.is_detector_id():
                det_ids.append(int(getattr(target, "value", getattr(target, "val", 0))))
            elif hasattr(target, "is_relative_detector_id") and target.is_relative_detector_id():
                rel = getattr(target, "relative_detector_id", None)
                det_ids.append(int(rel() if callable(rel) else rel))
            elif hasattr(target, "is_observable_id") and target.is_observable_id():
                obs_ids.append(int(getattr(target, "value", getattr(target, "val", 0))))
            elif hasattr(target, "is_logical_observable_id") and target.is_logical_observable_id():
                rel = getattr(target, "logical_observable_id", None)
                obs_ids.append(int(rel() if callable(rel) else rel))
        except Exception:
            continue
    return det_ids, obs_ids


def _normalize_detector_context(metadata: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    ctx_raw = (metadata.get("mwpm_debug", {}) or {}).get("detector_context", {}) or {}
    ctx_map: Dict[int, Dict[str, Any]] = {}
    for key, info in ctx_raw.items():
        try:
            det_id = int(key)
        except Exception:
            continue
        ctx_map[det_id] = dict(info or {})
    return ctx_map


def _row_wrap_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    wraps = (metadata.get("mwpm_debug", {}) or {}).get("row_wraps", {}) or {}
    z_wraps = wraps.get("Z", {}) or {}
    x_wraps = wraps.get("X", {}) or {}
    return {
        "Z_total": sum(len(rows or []) for rows in z_wraps.values()),
        "X_total": sum(len(rows or []) for rows in x_wraps.values()),
        "Z_by_patch": {patch: list(rows) for patch, rows in z_wraps.items() if rows},
        "X_by_patch": {patch: list(rows) for patch, rows in x_wraps.items() if rows},
    }


def _boundary_row_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    rows = metadata.get("boundary_rows")
    if not isinstance(rows, dict):
        return {}
    summary: Dict[str, Any] = {}
    for patch, info in rows.items():
        try:
            z_info = info.get("Z", {})
            x_info = info.get("X", {})
            summary[patch] = {
                "Z": f"{len(z_info.get('rows', []))}/{z_info.get('total', 0)}",
                "X": f"{len(x_info.get('rows', []))}/{x_info.get('total', 0)}",
            }
        except Exception:
            continue
    return summary


def analyze_dem_structure(
    dem: stim.DetectorErrorModel,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Return structural summary of the DEM."""
    try:
        num_det = int(dem.num_detectors)
    except Exception:
        num_det = int(dem.num_detectors())
    try:
        num_obs = int(dem.num_observables)
    except Exception:
        num_obs = int(dem.num_observables())

    anchor_ids: set[int] = set()
    boundary_meta = metadata.get("boundary_anchors") or {}
    for det_id in boundary_meta.get("detector_ids", []) or []:
        try:
            anchor_ids.add(int(det_id))
        except Exception:
            continue

    comps, comp_has_boundary, _ = build_components_from_dem(dem)
    component_hist = Counter(len(comp) for comp in comps)
    comp_has_anchor = [
        any(det in anchor_ids for det in comp) for comp in comps
    ]
    anchorless_indices = [i for i, has_anchor in enumerate(comp_has_anchor) if not has_anchor]
    anchorless_samples = []
    for idx in anchorless_indices[:6]:
        comp = comps[idx]
        if comp:
            anchorless_samples.append(comp[0])
    ctx_map = _normalize_detector_context(metadata)
    anchorless_ctx = {det: ctx_map.get(det, {}) for det in anchorless_samples}

    row_wraps = _row_wrap_summary(metadata)
    boundary_counts = (metadata.get("mwpm_debug", {}) or {}).get("boundary_counts", {})

    boundary_row_summary = _boundary_row_summary(metadata)

    return {
        "num_detectors": num_det,
        "num_observables": num_obs,
        "anchor_count": len(anchor_ids),
        "component_hist": component_hist,
        "component_count": len(comps),
        "anchorless_count": len(anchorless_indices),
        "largest_component": max(component_hist, default=0),
        "largest_anchorless": max((len(comps[idx]) for idx in anchorless_indices), default=0),
        "anchorless_samples": anchorless_samples,
        "anchorless_context": anchorless_ctx,
        "row_wraps": row_wraps,
        "boundary_counts": boundary_counts,
        "boundary_rows": boundary_row_summary,
        "observable_components": sum(1 for flag in comp_has_boundary if flag),
    }


def log_dem_diagnostics(
    label: str,
    dem: stim.DetectorErrorModel,
    metadata: Dict[str, Any],
    *,
    enabled: bool = False,
    max_hist_entries: int = 8,
) -> None:
    """Pretty-print DEM summary when enabled."""
    if not enabled:
        return
    try:
        summary = analyze_dem_structure(dem, metadata)
    except Exception as exc:
        print(f"[DEM-STATS] {label}: failed to analyze DEM ({exc})")
        return

    num_det = summary["num_detectors"]
    anchors = summary["anchor_count"]
    anchor_ratio = (anchors / num_det * 100.0) if num_det else 0.0
    print(
        f"[DEM-STATS] {label}: det={num_det} obs={summary['num_observables']} "
        f"anchors={anchors} ({anchor_ratio:.1f}%) comps={summary['component_count']} "
        f"anchorless={summary['anchorless_count']} largest={summary['largest_component']} "
        f"largest_anchorless={summary['largest_anchorless']} "
        f"observable_components={summary['observable_components']}"
    )

    hist_items = sorted(summary["component_hist"].items())
    if hist_items:
        sample = dict(hist_items[:max_hist_entries])
        print(f"[DEM-STATS] {label}: component histogram (size->count): {sample}")

    row_wraps = summary["row_wraps"]
    if row_wraps:
        print(
            f"[DEM-STATS] {label}: row_wrap totals Z={row_wraps.get('Z_total', 0)} "
            f"X={row_wraps.get('X_total', 0)}"
        )
        if row_wraps.get("Z_by_patch"):
            print(f"[DEM-STATS]    Z row wraps by patch: {row_wraps['Z_by_patch']}")
        if row_wraps.get("X_by_patch"):
            print(f"[DEM-STATS]    X row wraps by patch: {row_wraps['X_by_patch']}")

    boundary_counts = summary["boundary_counts"]
    if boundary_counts:
        print(f"[DEM-STATS] {label}: boundary_counts={boundary_counts}")

    boundary_rows = summary["boundary_rows"]
    if boundary_rows:
        print(f"[DEM-STATS] {label}: boundary_row fractions={boundary_rows}")

    if summary["anchorless_context"]:
        print(f"[DEM-STATS] {label}: anchorless detector samples={summary['anchorless_context']}")
