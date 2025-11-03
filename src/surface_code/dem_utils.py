"""Utilities for DEM (Detector Error Model) manipulation and MWPM (Minimum Weight Perfect Matching) debugging.

This module provides tools for:
- Parsing and analyzing DEM structures
- Debugging MWPM decoder issues
- Hardening DEMs for pairwise matching compatibility
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional, Set, Dict

import numpy as np
import pymatching as pm
import stim


# Parse 'error(...) D# D# L#' lines from a stim.DetectorErrorModel
_D_TOKEN_RE = re.compile(r"D(\d+)")
_L_TOKEN_RE = re.compile(r"L(\d+)")


def parse_dem_errors(dem):
    """
    Return a list of elementary faults as dicts:
        {"detectors":[int,...], "observables":[int,...], "raw": line}
    """
    out = []
    for line in str(dem).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not s.lower().startswith("error"):
            continue
        det_ids = [int(m.group(1)) for m in _D_TOKEN_RE.finditer(s)]
        obs_ids = [int(m.group(1)) for m in _L_TOKEN_RE.finditer(s)]
        out.append({"detectors": det_ids, "observables": obs_ids, "raw": s})
    return out


def build_components_from_dem(dem):
    """
    Build detector connected components from the DEM.
    Link detectors that co-appear in any elementary fault.
    Also mark if a component 'has boundary' (∃ fault with an odd number of
    detectors from that component, usually a single D# flip).
    """
    n = dem.num_detectors
    errors = parse_dem_errors(dem)

    # Union–Find
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # Link detectors that appear together in an error
    for err in errors:
        ds = err["detectors"]
        for i in range(len(ds) - 1):
            union(ds[i], ds[i + 1])

    # Group by root
    comps_dict = defaultdict(list)
    for d in range(n):
        root = find(d)
        comps_dict[root].append(d)
    comps = [sorted(vs) for vs in comps_dict.values()]

    # Boundary flag per component: odd intersection with some fault
    idx_to_comp = {}
    for ci, comp in enumerate(comps):
        for d in comp:
            idx_to_comp[d] = ci

    comp_has_boundary = [False] * len(comps)
    for err in errors:
        touched = defaultdict(int)
        for d in err["detectors"]:
            ci = idx_to_comp.get(d)
            if ci is not None:
                touched[ci] ^= 1
        for ci, parity in touched.items():
            if parity & 1:
                comp_has_boundary[ci] = True

    return comps, comp_has_boundary, errors


def pm_find_offending_shot(matcher, det_samp, *, max_scan=2048):
    """Inspect the actual PyMatching graph for boundaryless odd-parity components.

    Prints the first offending shot+component if found. Returns True if an
    offender was found, else False. Also reports isolated detector nodes.
    """
    try:
        import numpy as _np
        import networkx as _nx
    except Exception as _exc:
        print(f"[PM-CHECK] skipped (imports failed): {_exc}")
        return False

    try:
        G = matcher.to_networkx()
    except Exception as _exc2:
        print(f"[PM-CHECK] to_networkx failed: {_exc2}")
        return False

    num_det = getattr(matcher, 'num_detectors', 0)
    comps = list(_nx.connected_components(G))
    boundary_nodes = {n for n, d in G.nodes(data=True) if d.get('is_boundary', False)}

    comp_info = []
    for comp_id, comp in enumerate(comps):
        det_nodes = sorted([n for n in comp if isinstance(n, int) and n < num_det])
        has_boundary = any(n in boundary_nodes for n in comp)
        comp_info.append((comp_id, det_nodes, has_boundary))

    nscan = min(det_samp.shape[0], int(max_scan)) if det_samp is not None else 0
    for s in range(nscan):
        syn = det_samp[s].astype(_np.uint8)
        for comp_id, det_nodes, has_boundary in comp_info:
            if not det_nodes:
                continue
            parity = int(_np.bitwise_xor.reduce(syn[det_nodes])) if det_nodes else 0
            if (parity & 1) and (not has_boundary):
                print(f"[PM-CHECK] offending shot={s}, COMP#{comp_id}, has_boundary={has_boundary}, size={len(det_nodes)}")
                print(f"  first few det nodes: {det_nodes[:16]}")
                return True

    print("[PM-CHECK] no boundaryless odd-parity component found in scanned shots.")
    # Check for isolated detector nodes (degree 0) and whether any shot hits them
    deg = dict(G.degree())
    isolates = [n for n in range(num_det) if deg.get(n, 0) == 0]
    if isolates:
        print(f"[PM-CHECK] isolated detector nodes (deg 0): {isolates[:16]}")
        for s in range(nscan):
            syn = det_samp[s].astype(_np.uint8)
            if any(int(syn[n]) & 1 for n in isolates if n < len(syn)):
                print(f"[PM-CHECK] shot {s} has 1s on isolated detectors (impossible without boundary).")
                break
    return False


def anchor_pm_isolates(dem, matcher, *, epsilon=1e-12):
    """Ensure every detector in the PyMatching graph has non-zero degree by adding boundary hooks.
    Also ensures every connected component has at least one boundary node.

    Returns (new_dem, new_matcher, num_added, isolate_ids).
    """
    try:
        import networkx as _nx
    except Exception:
        return dem, matcher, 0, []

    try:
        G = matcher.to_networkx()
    except Exception:
        return dem, matcher, 0, []

    num_det = getattr(matcher, "num_detectors", 0)
    deg = dict(G.degree())
    isolates = [n for n in range(num_det) if deg.get(n, 0) == 0]
    
    # Find components without boundaries
    boundary_nodes = {n for n, d in G.nodes(data=True) if d.get('is_boundary', False)}
    comps = list(_nx.connected_components(G))
    components_needing_boundaries = []
    
    for comp in comps:
        det_nodes = [n for n in comp if isinstance(n, int) and n < num_det]
        has_boundary = any(n in boundary_nodes for n in comp)
        if det_nodes and not has_boundary:
            # Component has detector nodes but no boundary
            components_needing_boundaries.append(det_nodes[0])  # Use first detector as anchor
    
    # Combine isolated detectors and components needing boundaries
    all_hooks = list(set(isolates + components_needing_boundaries))
    
    if not all_hooks:
        return dem, matcher, 0, []

    dem_text = str(dem)
    if dem_text and not dem_text.endswith("\n"):
        dem_text += "\n"
    p_str = f"{float(epsilon):.12g}"
    for idx in all_hooks:
        dem_text += f"error({p_str}) D{idx}\n"
    new_dem = stim.DetectorErrorModel(dem_text)
    new_matcher = pm.Matching.from_detector_error_model(new_dem)
    return new_dem, new_matcher, len(all_hooks), all_hooks


def report_boundaryless_components(dem):
    """Report boundaryless components in the DEM."""
    comps, comp_has_boundary, _ = build_components_from_dem(dem)
    n = len(comps)
    nob = sum(1 for b in comp_has_boundary if not b)
    print(f"[DEM-CHECK] components={n}, boundaryless={nob}")
    if nob:
        for i, comp in enumerate(comps):
            if comp_has_boundary[i]:
                continue
            head = comp[:24]
            tail = comp[-8:] if len(comp) > 32 else []
            print(f"  [COMP#{i}] size={len(comp)} head={head}{' tail='+str(tail) if tail else ''}")


def scan_boundaryless_odd_shot(dem, det_samp, *, max_scan=512):
    """Scan for boundaryless components with odd parity in detector samples."""
    if det_samp is None or dem.num_detectors == 0:
        return
    comps, comp_has_boundary, _ = build_components_from_dem(dem)
    comp_sets = [set(c) for c in comps]
    nscan = min(det_samp.shape[0], int(max_scan))
    for s in range(nscan):
        syn = det_samp[s].astype(np.uint8)
        for ci, comp in enumerate(comps):
            if not comp or comp_has_boundary[ci]:
                continue
            # odd parity in a boundaryless component → infeasible
            parity = int(np.bitwise_xor.reduce(syn[comp]))
            if parity & 1:
                print(f"[DEM-CHECK] offending shot={s} boundaryless COMP#{ci} size={len(comp)} sample={comp[:16]}")
                print("  detslice filter:", "[" + ", ".join(f"'D{d}'" for d in comp[:12]) + "]")
                # quick arity histogram for faults touching this comp
                errs = parse_dem_errors(dem)
                hist = {}
                for e in errs:
                    k = sum(1 for d in e['detectors'] if d in comp_sets[ci])
                    if k:
                        hist[k] = hist.get(k, 0) + 1
                print("  error-arity histogram:", dict(sorted(hist.items())))
                # show a few example error lines touching the component (odd and even)
                odd_examples = []
                even_examples = []
                for e in errs:
                    k = sum(1 for d in e['detectors'] if d in comp_sets[ci])
                    if k == 0:
                        continue
                    (odd_examples if (k % 2 == 1) else even_examples).append(e["raw"])
                    if len(odd_examples) >= 4 and len(even_examples) >= 4:
                        break
                if odd_examples:
                    print("  example odd-intersection errors:")
                    for ln in odd_examples[:4]:
                        print("    ", ln)
                if even_examples:
                    print("  example even-intersection errors:")
                    for ln in even_examples[:4]:
                        print("    ", ln)
                return


def harden_dem_add_boundaries(dem, *, epsilon=1e-12):
    """
    For each detector connected component that currently has no boundary
    (no fault with an odd intersection), append an infinitesimal-probability
    single-detector error to create a boundary hook. This guarantees MWPM
    feasibility without materially changing statistics.
    """
    comps, comp_has_boundary, _ = build_components_from_dem(dem)
    added = 0
    for ci, comp in enumerate(comps):
        if comp and not comp_has_boundary[ci]:
            # Hook the first detector in this component
            d0 = comp[0]
            dem.append("error", epsilon, [stim.DemTarget.detector(d0)])
            added += 1
    return added


def harden_dem_for_pairwise_matching(dem, *, epsilon=1e-12):
    """
    Ensure every connected component has at least one *single-detector*
    error (a real boundary edge for pairwise MWPM).
    Returns a new DEM and the number of hooks added.
    
    This function iterates until all components have boundaries, as rebuilding
    the DEM from text can change component structure.
    """
    max_iterations = 5  # Safety limit to prevent infinite loops
    current_dem = dem
    total_added = 0
    
    for iteration in range(max_iterations):
        comps, comp_has_boundary, errors = build_components_from_dem(current_dem)
        
        # Check if all components have boundaries
        components_needing_boundaries = [i for i, comp in enumerate(comps) 
                                       if comp and not comp_has_boundary[i]]
        
        if not components_needing_boundaries:
            # All components have boundaries, we're done
            return current_dem, total_added
        
        # Add boundaries to components that need them
        hook_ids = []
        for comp_idx in components_needing_boundaries:
            comp = comps[comp_idx]
            if comp:
                # Use the first detector in the component as the boundary hook
                d0 = comp[0]
                hook_ids.append(d0)
        
        if not hook_ids:
            # No hooks to add, break
            break
        
        added_this_iteration = len(hook_ids)
        total_added += added_this_iteration
        
        # Append boundary errors to DEM text
        dem_text = str(current_dem)
        if not dem_text.endswith('\n'):
            dem_text += '\n'
        p_str = f"{float(epsilon):.12g}"
        for d0 in hook_ids:
            dem_text += f"error({p_str}) D{d0}\n"
        
        # Rebuild DEM and continue iteration if needed
        current_dem = stim.DetectorErrorModel(dem_text)
    
    # Final check: verify all components have boundaries
    final_comps, final_has_boundary, _ = build_components_from_dem(current_dem)
    still_needing = [i for i, comp in enumerate(final_comps) 
                     if comp and not final_has_boundary[i]]
    
    if still_needing:
        # If some components still don't have boundaries, add them aggressively
        hook_ids = []
        for comp_idx in still_needing:
            comp = final_comps[comp_idx]
            if comp:
                hook_ids.append(comp[0])
        
        if hook_ids:
            dem_text = str(current_dem)
            if not dem_text.endswith('\n'):
                dem_text += '\n'
            p_str = f"{float(epsilon):.12g}"
            for d0 in hook_ids:
                dem_text += f"error({p_str}) D{d0}\n"
            current_dem = stim.DetectorErrorModel(dem_text)
            total_added += len(hook_ids)
    
    return current_dem, total_added


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
    seen: set[int] = set()
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


# --- Hook/anchor analysis helpers ---

def single_detector_hook_ids(dem: stim.DetectorErrorModel) -> List[int]:
    """
    Return the sorted unique detector ids that appear in *single-detector* error lines
    of the DEM. These are lines of the form 'error(p) Dk' (possibly with comments).
    This function does not try to distinguish between physically modeled boundaries
    and synthetic anchors; use diffs (before vs. after) to see what was added.
    """
    ids: set[int] = set()
    for line in str(dem).splitlines():
        s = line.strip()
        if not s or s.startswith("#") or not s.lower().startswith("error"):
            continue
        dets = [int(m.group(1)) for m in _D_TOKEN_RE.finditer(s)]
        # Count only single-detector errors
        if len(dets) == 1:
            ids.add(dets[0])
    return sorted(ids)


def summarize_hook_diffs(
    before: stim.DetectorErrorModel, after: stim.DetectorErrorModel
) -> dict:
    """
    Compare single-detector hooks between two DEMs and return a summary dict with:
        - 'before_ids': sorted list of hook detector ids in 'before'
        - 'after_ids' : sorted list of hook detector ids in 'after'
        - 'added_ids' : sorted list of ids present only in 'after'
        - 'removed_ids': sorted list of ids present only in 'before'
    """
    b = set(single_detector_hook_ids(before))
    a = set(single_detector_hook_ids(after))
    return {
        "before_ids": sorted(b),
        "after_ids": sorted(a),
        "added_ids": sorted(a - b),
        "removed_ids": sorted(b - a),
    }


def log_hook_summary(
    before: stim.DetectorErrorModel,
    after: stim.DetectorErrorModel,
    *,
    prefix: str = "[DEM-HOOKS]"
) -> dict:
    """
    Print a short summary of boundary hooks before vs after and return the summary dict.
    """
    info = summarize_hook_diffs(before, after)
    print(
        f"{prefix} before={len(info['before_ids'])} "
        f"after={len(info['after_ids'])} "
        f"added={len(info['added_ids'])} "
        f"removed={len(info['removed_ids'])}"
    )
    if info["added_ids"]:
        print(f"{prefix} sample added: {info['added_ids'][:16]}")
    if info["removed_ids"]:
        print(f"{prefix} sample removed: {info['removed_ids'][:16]}")
    return info


# ---- DEM feasibility diagnostics (pairwise connectivity view) ----
def _dem_iter_error_lines(dem: stim.DetectorErrorModel):
    """Yield the detector-id list for each ERROR line in the DEM text."""
    for line in str(dem).splitlines():
        s = line.strip()
        if not s or s.startswith("#") or not s.lower().startswith("error"):
            continue
        dets = [int(m.group(1)) for m in _D_TOKEN_RE.finditer(s)]
        if dets:
            yield dets


def compute_dem_components(
    dem: stim.DetectorErrorModel,
) -> Tuple[List[Set[int]], Set[int]]:
    """
    Construct a simple graph from the DEM:
      - Nodes: all detectors that appear in any ERROR line.
      - For each ERROR:
          * if it touches exactly 1 detector -> that detector is a *boundary node*.
          * if it touches >=2 detectors -> add undirected edges between all pairs (clique)
            to preserve connectivity.
    Returns (components, boundary_nodes) where components is a list of node sets.
    """
    nodes: Set[int] = set()
    boundaries: Set[int] = set()
    adj: Dict[int, Set[int]] = {}

    for dets in _dem_iter_error_lines(dem):
        for d in dets:
            nodes.add(d)
            adj.setdefault(d, set())
        if len(dets) == 1:
            boundaries.add(dets[0])
        else:
            # connect all pairs (clique) to ensure connectivity within the error
            for i in range(len(dets)):
                for j in range(i + 1, len(dets)):
                    a, b = dets[i], dets[j]
                    adj[a].add(b)
                    adj[b].add(a)

    # Flood-fill components
    unvisited = set(nodes)
    comps: List[Set[int]] = []
    while unvisited:
        root = unvisited.pop()
        stack = [root]
        comp = {root}
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):
                if v in unvisited:
                    unvisited.remove(v)
                    comp.add(v)
                    stack.append(v)
        comps.append(comp)

    return comps, boundaries


def component_anchor_coverage(
    components: Sequence[Set[int]],
    boundaries: Set[int],
    explicit_anchor_ids: Sequence[int] | None,
) -> Dict[str, object]:
    """
    For each component, report whether it contains a boundary node
    (either from a single-detector ERROR or from the explicit anchor list).
    """
    exp = set(explicit_anchor_ids or [])
    covered_flags: List[bool] = [len((boundaries | exp) & comp) > 0 for comp in components]
    return {
        "components": len(components),
        "covered": sum(1 for x in covered_flags if x),
        "uncovered": sum(1 for x in covered_flags if not x),
        "uncovered_indices": [i for i, x in enumerate(covered_flags) if not x],
    }


def scan_infeasible_shot(
    dem: stim.DetectorErrorModel,
    det_samp: np.ndarray,
    max_scan: int = 2048,
) -> Optional[Dict[str, object]]:
    """
    Find the first shot s.t. some component has odd syndrome parity and no boundary.
    Returns a dict with details or None if none found within the scan window.
    """
    if det_samp is None or det_samp.size == 0:
        return None
    components, boundaries = compute_dem_components(dem)
    comp_nodes = [sorted(list(comp)) for comp in components]

    shots = min(max_scan, det_samp.shape[0])
    for s in range(shots):
        syn = det_samp[s].astype(np.uint8)
        for i, nodes in enumerate(comp_nodes):
            if not nodes:
                continue
            parity = int(np.bitwise_xor.reduce(syn[nodes]))
            has_boundary = len((boundaries & set(nodes))) > 0
            if (parity & 1) and not has_boundary:
                firing = [n for n in nodes if int(syn[n]) & 1]
                return {
                    "shot": s,
                    "component_index": i,
                    "component_size": len(nodes),
                    "boundary_count": len(boundaries & set(nodes)),
                    "firing_nodes": firing[:64],
                    "total_firing_in_comp": len(firing),
                }
    return None


def subdem_for_component(
    dem: stim.DetectorErrorModel,
    component_nodes: Sequence[int],
) -> stim.DetectorErrorModel:
    """
    Extract a minimal sub-DEM containing only ERROR lines supported entirely on
    the given component nodes (plus single-detector hooks that lie in the set).
    Non-ERROR lines are dropped for compactness.
    """
    keep = set(int(x) for x in component_nodes)
    out_lines: List[str] = []
    for line in str(dem).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not s.lower().startswith("error"):
            continue
        dets = [int(m.group(1)) for m in _D_TOKEN_RE.finditer(s)]
        if not dets:
            continue
        if (len(dets) == 1 and dets[0] in keep) or all(d in keep for d in dets):
            out_lines.append(s)
    return stim.DetectorErrorModel("\n".join(out_lines))
