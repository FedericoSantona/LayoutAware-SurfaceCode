"""Utilities for DEM (Detector Error Model) manipulation and MWPM (Minimum Weight Perfect Matching) debugging.

This module provides tools for:
- Parsing and analyzing DEM structures
- Debugging MWPM decoder issues
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, List, Tuple, Sequence, Optional, Set, Dict
import os
import numbers

import numpy as np
import pymatching as pm
import stim


# Parse 'error(...) D# D# L#' lines from a stim.DetectorErrorModel
_D_TOKEN_RE = re.compile(r"D(\d+)")
_L_TOKEN_RE = re.compile(r"L(\d+)")


def _is_error_instruction(inst: stim.DemInstruction) -> bool:  # type: ignore[name-defined]
    """Stim-version-safe check for ERROR instructions."""
    try:
        if hasattr(stim, "DemInstructionType"):
            return getattr(inst, "type", None) == stim.DemInstructionType.ERROR
        name = getattr(inst, "name", None)
        if name is not None:
            return str(name).lower() == "error"
        inst_type = getattr(inst, "type", None)
        if inst_type is not None:
            return str(getattr(inst_type, "name", inst_type)).lower() == "error"
        return str(inst).lstrip().lower().startswith("error")
    except Exception:
        return False


def _env_graphlike_preference() -> Optional[bool]:
    """Return env-controlled preference for graphlike DEMs, if set."""
    for name in ("SC_USE_GRAPHLIKE_DEM", "SC_USE_GRAPHIKE_DEM"):
        flag = os.getenv(name)
        if flag is not None:
            text = str(flag).strip().lower()
            return text in {"1", "true", "yes", "on"}
    return None


def circuit_to_graphlike_dem(
    circuit: stim.Circuit,
    *,
    boundary_epsilon: float = 1e-12,
    force_graphlike: Optional[bool] = None,
) -> stim.DetectorErrorModel:
    """Return Stim's detector error model, optionally graphified for legacy runs."""
    try:
        dem = circuit.detector_error_model(decompose_errors=True)
    except ValueError:
        dem = circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True,
        )
    env_pref = _env_graphlike_preference()
    if force_graphlike is not None:
        use_graphlike = force_graphlike
    elif env_pref is not None:
        use_graphlike = env_pref
    else:
        use_graphlike = True
    if use_graphlike:
        dem = convert_dem_to_graphlike(dem)
    return dem


def sample_circuit_dem_data(
    circuit: stim.Circuit,
    shots: int,
    *,
    seed: Optional[int] = None,
    return_measurements: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Sample measurement data once and derive detector/observable arrays.

    Returns
    -------
    det: np.ndarray
        Detection event bits with shape (shots, circuit.num_detectors).
    obs: np.ndarray
        Observable bits with shape (shots, circuit.num_observables).
    measurements: Optional[np.ndarray]
        Raw measurement record (shots, circuit.num_measurements) when
        ``return_measurements`` is True, else ``None``.
    """

    # Stim's measurement sampler includes all noise present in the circuit.
    circ_sampler = circuit.compile_sampler(seed=seed)
    meas = circ_sampler.sample(shots=int(shots))
    meas = np.asarray(meas, dtype=np.bool_)

    converter = circuit.compile_m2d_converter()
    det_bits, obs_bits = converter.convert(
        measurements=meas,
        separate_observables=True,
    )
    det_bits = np.asarray(det_bits, dtype=np.bool_)
    if obs_bits is None:
        obs_bits = np.zeros((meas.shape[0], 0), dtype=np.bool_)
    else:
        obs_bits = np.asarray(obs_bits, dtype=np.bool_)

    if return_measurements:
        return det_bits, obs_bits, meas
    return det_bits, obs_bits, None


def add_boundary_hooks_to_dem(
    dem: stim.DetectorErrorModel,
    metadata: Dict[str, object],
) -> stim.DetectorErrorModel:
    """Append 1-detector ERROR hooks for true boundaries described in metadata.

    Hooks inherit a probability comparable to the physical error rates specified
    in ``metadata['noise_model']`` (fallback to the boundary ``epsilon`` field or
    1e-4 when no noise data is present).
    """
    if not isinstance(metadata, dict):
        return dem
    anchors_conf = metadata.get("boundary_anchors", {}) or {}
    anchors = anchors_conf.get("detector_ids")
    if not anchors:
        return dem
    noise = metadata.get("noise_model", {}) or {}
    prob_candidates: List[float] = []
    for key in ("p_x_error", "p_z_error", "p_meas"):
        val = noise.get(key)
        try:
            if val is not None:
                prob_candidates.append(float(val))
        except Exception:
            continue
    prob = max(prob_candidates) if prob_candidates else 0.0
    if prob <= 0.0:
        try:
            prob = float(anchors_conf.get("epsilon", 1e-4))
        except Exception:
            prob = 1e-4
    prob = max(prob, 1e-6)
    new_dem = dem.copy()
    seen: Set[int] = set()
    for det_id in anchors:
        try:
            idx = int(det_id)
        except Exception:
            continue
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        new_dem.append("error", prob, [stim.target_relative_detector_id(idx)])
    return new_dem


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

    def _is_det_node(node) -> bool:
        return isinstance(node, numbers.Integral) and 0 <= int(node) < num_det

    comps = list(_nx.connected_components(G))
    boundary_nodes = {
        n for n, d in G.nodes(data=True) if d.get('is_boundary', False)
    }

    comp_info = []
    for comp_id, comp in enumerate(comps):
        det_nodes = sorted(int(n) for n in comp if _is_det_node(n))
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
    present_det_nodes = {int(n) for n in G.nodes() if _is_det_node(n)}
    deg = dict(G.degree())
    present_det_nodes = {int(n) for n in G.nodes() if _is_det_node(n)}
    isolates = [
        int(n) for n, degree in deg.items()
        if _is_det_node(n) and degree == 0
    ]
    missing_det_nodes = [k for k in range(num_det) if k not in present_det_nodes]
    isolates.extend(missing_det_nodes)
    isolates = sorted(set(isolates))
    isolates.extend(sorted(set(range(num_det)) - present_det_nodes))
    isolates = sorted(set(isolates))
    if isolates:
        print(f"[PM-CHECK] isolated detector nodes (deg 0): {isolates[:16]}")
        for s in range(nscan):
            syn = det_samp[s].astype(_np.uint8)
            if any(int(syn[n]) & 1 for n in isolates if n < len(syn)):
                print(f"[PM-CHECK] shot {s} has 1s on isolated detectors (impossible without boundary).")
                break
    return False


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


# --- Hook/anchor analysis helpers ---

def single_detector_hook_ids(dem: stim.DetectorErrorModel) -> List[int]:
    """
    Return the sorted unique detector ids that appear in *single-detector* error lines
    of the DEM. These are lines of the form 'error(p) Dk' (possibly with comments).
    This function does not try to distinguish between physically modeled boundaries
    and synthetic anchors; use diffs (before vs. after) to see what was added.
    """
    ids: set[int] = set()
    for dets in _dem_iter_error_blocks(dem):
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
def _detector_id_from_target(target) -> Optional[int]:
    """Extract an absolute detector id from a stim target."""
    try:
        if hasattr(target, "is_detector_id") and target.is_detector_id():
            val = getattr(target, "value", None)
            if val is None:
                val = getattr(target, "val", None)
            if val is not None:
                return int(val)
        if hasattr(target, "is_relative_detector_id") and target.is_relative_detector_id():
            # Relative ids expose `.val` with the absolute detector index
            val = getattr(target, "val", None)
            if val is not None:
                return int(val)
    except Exception:
        return None
    return None


def _dem_iter_error_blocks(dem: stim.DetectorErrorModel):
    """
    Yield the detector-id list for each ERROR component (between separators)
    in the DEM. This respects stim.target_separator() so decomposed errors
    are treated as separate graphlike terms.
    """
    try:
        iterator = dem.flattened()
    except Exception:
        iterator = dem
    for inst in iterator:
        try:
            if not _is_error_instruction(inst):
                continue
        except Exception:
            continue
        try:
            targets = list(inst.targets_copy()) if hasattr(inst, "targets_copy") else list(getattr(inst, "targets", ()))
        except Exception:
            targets = []
        block: List[int] = []
        for tgt in targets:
            try:
                if hasattr(tgt, "is_separator") and tgt.is_separator():
                    if block:
                        yield block
                        block = []
                    continue
            except Exception:
                pass
            det_id = _detector_id_from_target(tgt)
            if det_id is not None:
                block.append(det_id)
        if block:
            yield block


def _dem_instruction_probability(inst) -> Optional[float]:
    """Extract the probability argument from an ERROR instruction."""
    try:
        if hasattr(inst, "args_copy"):
            args = inst.args_copy()
        else:
            args = getattr(inst, "args", None) or getattr(inst, "arguments", None)
        if args:
            try:
                return float(args[0])
            except Exception:
                return None
    except Exception:
        return None
    return None


def iter_error_blocks_with_prob(
    dem: stim.DetectorErrorModel,
) -> Iterable[Tuple[Optional[float], List[int]]]:
    """Yield (probability, detector_ids) for each ERROR component."""
    try:
        iterator = dem.flattened()
    except Exception:
        iterator = dem
    for inst in iterator:
        try:
            if not _is_error_instruction(inst):
                continue
        except Exception:
            continue
        prob = _dem_instruction_probability(inst)
        try:
            targets = list(inst.targets_copy()) if hasattr(inst, "targets_copy") else list(getattr(inst, "targets", ()))
        except Exception:
            targets = []
        block: List[int] = []
        for tgt in targets:
            try:
                if hasattr(tgt, "is_separator") and tgt.is_separator():
                    if block:
                        yield prob, block
                        block = []
                    continue
            except Exception:
                pass
            det_id = _detector_id_from_target(tgt)
            if det_id is not None:
                block.append(det_id)
        if block:
            yield prob, block


def _graphify_error_targets(inst) -> Optional[Tuple[List[stim.DemTarget], object, str]]:
    """Return rewritten targets/args/tag if the ERROR instruction needs graphification."""
    try:
        targets = list(inst.targets_copy()) if hasattr(inst, "targets_copy") else list(getattr(inst, "targets", ()))
    except Exception:
        targets = []
    blocks: List[List[stim.DemTarget]] = []
    others: List[stim.DemTarget] = []
    current: List[stim.DemTarget] = []
    for tgt in targets:
        try:
            if hasattr(tgt, "is_separator") and tgt.is_separator():
                if current:
                    blocks.append(current)
                    current = []
                continue
        except Exception:
            pass
        if _detector_id_from_target(tgt) is not None:
            current.append(tgt)
        else:
            others.append(tgt)
    if current:
        blocks.append(current)
    if not blocks or all(len(block) <= 2 for block in blocks):
        return None
    new_targets: List[stim.DemTarget] = []
    for block in blocks:
        if len(block) <= 2:
            new_targets.extend(block)
            new_targets.append(stim.target_separator())
        else:
            for i in range(len(block) - 1):
                new_targets.append(block[i])
                new_targets.append(block[i + 1])
                new_targets.append(stim.target_separator())
    if new_targets and getattr(new_targets[-1], "is_separator", lambda: False)():
        new_targets.pop()
    new_targets.extend(others)
    try:
        args = inst.args_copy() if hasattr(inst, "args_copy") else getattr(inst, "args", None)
    except Exception:
        args = None
    if args is None:
        paren_args = ()
    elif isinstance(args, (list, tuple)):
        paren_args = list(args)
        if len(paren_args) == 1:
            paren_args = paren_args[0]
    else:
        paren_args = args
    tag = getattr(inst, "tag", "")
    return new_targets, paren_args, str(tag) if tag else ""


def convert_dem_to_graphlike(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Return a DEM where every ERROR component has at most two detectors."""
    new_dem = stim.DetectorErrorModel()
    for inst in dem:
        try:
            if not _is_error_instruction(inst):
                new_dem.append(inst)
                continue
        except Exception:
            new_dem.append(inst)
            continue
        rewrite = _graphify_error_targets(inst)
        if rewrite is None:
            new_dem.append(inst)
            continue
        new_targets, paren_args, tag = rewrite
        if tag:
            new_dem.append("error", paren_args, new_targets, tag=tag)
        else:
            new_dem.append("error", paren_args, new_targets)
    return new_dem


def add_spatial_correlations_to_dem(
    dem: stim.DetectorErrorModel,
    metadata: Dict[str, object],
) -> stim.DetectorErrorModel:
    """Augment the DEM with spatial edges inferred from builder metadata."""
    dbg = (metadata.get("mwpm_debug", {}) or {})
    spatial = dbg.get("spatial_pairs") or {}
    if not spatial:
        return dem
    det_ctx = dbg.get("detector_context") or {}
    # New builders emit explicit butterfly detectors; if they are present the
    # DEM already has the correct spatial structure and no augmentation is
    # necessary.
    for info in det_ctx.values():
        if not isinstance(info, dict):
            continue
        tag = info.get("tag")
        if tag in {"z_butterfly", "x_butterfly"}:
            return dem
    if not det_ctx:
        return dem
    noise = metadata.get("noise_model", {}) or {}
    basis_prob = {
        "Z": float(noise.get("p_x_error", 0.0) or 0.0),
        "X": float(noise.get("p_z_error", 0.0) or 0.0),
    }

    detectors_by_round: Dict[Tuple[str, str], Dict[int, Dict[int, int]]] = {}
    neighbor_rows: Dict[Tuple[str, str], Dict[int, Set[int]]] = {}

    for basis, per_patch in spatial.items():
        for patch, entries in (per_patch or {}).items():
            key = (basis, patch)
            nbr = neighbor_rows.setdefault(key, {})
            for entry in entries or []:
                rows = entry.get("rows")
                if not rows or len(rows) != 2:
                    continue
                ra, rb = int(rows[0]), int(rows[1])
                nbr.setdefault(ra, set()).add(rb)
                nbr.setdefault(rb, set()).add(ra)

    for det_id, info in det_ctx.items():
        try:
            det_idx = int(det_id)
        except Exception:
            continue
        tag = (info or {}).get("tag")
        if tag not in ("z_temporal", "x_temporal"):
            continue
        ctx = (info or {}).get("context", {}) or {}
        patch = ctx.get("patch")
        row = ctx.get("row")
        round_idx = ctx.get("round")
        if patch is None or row is None or round_idx is None:
            continue
        basis = "Z" if tag.startswith("z_") else "X"
        key = (basis, str(patch))
        round_map = detectors_by_round.setdefault(key, {})
        row_map = round_map.setdefault(int(round_idx), {})
        row_map[int(row)] = det_idx

    new_dem = dem.copy()
    for (basis, patch), rounds_map in detectors_by_round.items():
        prob = float(basis_prob.get(basis, 0.0) or 0.0)
        if prob <= 0.0:
            continue
        neighbors = neighbor_rows.get((basis, patch))
        if not neighbors:
            continue
        for round_idx, row_to_det in rounds_map.items():
            for row, det_a in row_to_det.items():
                for neigh in neighbors.get(row, ()):
                    if row >= neigh:
                        continue
                    det_b = row_to_det.get(neigh)
                    if det_a is None or det_b is None or det_a == det_b:
                        continue
                    new_dem.append(
                        "error",
                        prob,
                        [
                            stim.target_relative_detector_id(int(det_a)),
                            stim.target_relative_detector_id(int(det_b)),
                        ],
                    )
    return new_dem


def dem_error_block_histogram(
    dem: stim.DetectorErrorModel,
) -> Dict[int, int]:
    """Return a histogram (block_size -> count) over ERROR components."""
    hist: Dict[int, int] = defaultdict(int)
    for _prob, block in iter_error_blocks_with_prob(dem):
        hist[len(block)] += 1
    return dict(sorted(hist.items()))


def collect_detectors_in_errors(dem: stim.DetectorErrorModel) -> Set[int]:
    """Return the set of detector ids that participate in any ERROR line."""
    nodes: Set[int] = set()
    for dets in _dem_iter_error_blocks(dem):
        for d in dets:
            nodes.add(int(d))
    return nodes


def collect_detectors_in_multi_errors(dem: stim.DetectorErrorModel) -> Set[int]:
    """Return detectors that participate in ERROR lines touching ≥2 detectors."""
    nodes: Set[int] = set()
    for dets in _dem_iter_error_blocks(dem):
        if len(dets) < 2:
            continue
        for d in dets:
            nodes.add(int(d))
    return nodes


def remap_metadata_detectors(metadata: Dict[str, object], mapping: Dict[int, int]) -> None:
    """Apply detector id remapping to metadata structures in-place."""
    if not mapping:
        return
    if not isinstance(metadata, dict):
        return

    def _remap_list(ids: Iterable[int]) -> List[int]:
        out: List[int] = []
        for k in ids:
            mk = mapping.get(int(k))
            if mk is not None and mk not in out:
                out.append(mk)
        return out

    # Boundary anchors
    ba = metadata.get("boundary_anchors")
    if isinstance(ba, dict):
        ids = ba.get("detector_ids") or []
        if isinstance(ids, (list, tuple)):
            ba["detector_ids"] = _remap_list(int(x) for x in ids)

    # MWPM debug detector context
    mwpm_dbg = metadata.get("mwpm_debug")
    if isinstance(mwpm_dbg, dict):
        ctx = mwpm_dbg.get("detector_context")
        if isinstance(ctx, dict):
            remapped_ctx: Dict[object, Dict[str, object]] = {}
            measurements = ctx.get("__measurements__")
            if isinstance(measurements, dict):
                remapped_ctx["__measurements__"] = dict(measurements)
            for k, v in ctx.items():
                if k == "__measurements__":
                    continue
                try:
                    old_idx = int(k)
                except Exception:
                    continue
                mk = mapping.get(old_idx)
                if mk is not None:
                    remapped_ctx[int(mk)] = dict(v)
            mwpm_dbg["detector_context"] = remapped_ctx
        # Update degree violation listings if present
        deg = mwpm_dbg.get("degree_violations")
        if isinstance(deg, list):
            mwpm_dbg["degree_violations"] = [mapping.get(int(x), int(x)) for x in deg if mapping.get(int(x)) is not None]
        odd = mwpm_dbg.get("odd_degree_details")
        if isinstance(odd, dict):
            remapped_odds: Dict[int, List[Dict[str, object]]] = {}
            for k, entries in odd.items():
                mk = mapping.get(int(k))
                if mk is None:
                    continue
                remapped_odds[int(mk)] = list(entries)
            mwpm_dbg["odd_degree_details"] = remapped_odds


def update_tag_stats_with_presence(
    metadata: Dict[str, object],
    detectors_in_error: Iterable[int],
) -> None:
    """Augment tag_stats in metadata with present_in_error counts."""
    if not isinstance(metadata, dict):
        return
    mwpm = metadata.get("mwpm_debug")
    if not isinstance(mwpm, dict):
        return
    tag_stats = mwpm.get("tag_stats")
    detector_context = mwpm.get("detector_context")
    if not isinstance(tag_stats, dict) or not isinstance(detector_context, dict):
        return
    seen: Set[int] = {int(d) for d in detectors_in_error if isinstance(d, numbers.Integral)}
    for det_id in seen:
        ctx = detector_context.get(int(det_id))
        if not isinstance(ctx, dict):
            continue
        tag = str(ctx.get("tag", ""))
        entry = tag_stats.setdefault(tag, {"emitted": 0, "kept": 0, "dropped": 0, "present_in_error": 0})
        entry["present_in_error"] = int(entry.get("present_in_error", 0)) + 1


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

    for dets in _dem_iter_error_blocks(dem):
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


def enforce_component_boundaries(
    dem: stim.DetectorErrorModel,
    *,
    epsilon: float = 1e-12,
    explicit_anchor_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Assert that every connected component already has a boundary node.
    Raises ValueError if a boundaryless component is detected.
    """
    components, boundaries = compute_dem_components(dem)
    boundary_set: Set[int] = set(boundaries)
    if explicit_anchor_ids:
        for idx in explicit_anchor_ids:
            try:
                boundary_set.add(int(idx))
            except Exception:
                continue
    uncovered = [comp for comp in components if comp and not (boundary_set & comp)]
    if uncovered:
        example = sorted(list(uncovered[0]))[:8]
        raise ValueError(
            f"Detector error model still has boundaryless components; "
            f"example nodes={example}"
        )
    return []


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
