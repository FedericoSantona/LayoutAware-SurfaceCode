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
from .surgery_ops import MeasureRound, Merge, Split, ParityReadout
from .configs import PhenomenologicalStimConfig
from .builder_utils import mpp_from_positions, rec_from_abs, add_temporal_detectors_with_index, _mpp_targets_from_pauli
from .pauli import Pauli, conjugate_through_circuit, PauliTracker


GateTarget = stim.GateTarget


@dataclass
class _PrevState:
    z_prev: Dict[str, List[Optional[int]]]
    x_prev: Dict[str, List[Optional[int]]]
    joint_prev: Dict[Tuple[str, str, str], List[int]]  # key=(kind,a,b)


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

        kind='rough' uses Z⊗Z across pairs; kind='smooth' uses X⊗X.
        Returns the list of absolute measurement indices (one per pair).
        """
        key = (kind, a, b)
        pairs = self.layout.seams.get(key, [])
        if not pairs:
            return []
        pauli = "Z" if kind == "rough" else "X"
        indices: List[int] = []
        for ia, ib in pairs:
            global_a = self.layout.globalize_local_index(a, ia)
            global_b = self.layout.globalize_local_index(b, ib)
            idx = mpp_from_positions(circuit, [global_a, global_b], pauli)
            if idx is not None:
                indices.append(idx)
        return indices

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


        # Defer DETECTOR appends to the end for determinism.
        # Collect edges as absolute measurement index lists (len 1 or 2).
        deferred_detectors: List[List[int]] = []

        def _defer_detector_from_abs(abs_indices: List[int]) -> int:
            deferred_detectors.append(list(abs_indices))
            # Return a stable index even though it's not used downstream.
            return len(deferred_detectors) - 1

        # Resolve patch order and selection helpers
        all_patches: List[str] = list(layout.patches.keys())

        def select_patches(spec: Optional[List[str]]) -> List[str]:
            return all_patches if spec is None else list(spec)


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

        # Track first/last measured indices per stabilizer (segment tracking)
        seg_first: Dict[Tuple[str, str], List[Optional[int]]] = {}  # key=(patch,basis)
        seg_last: Dict[Tuple[str, str], List[Optional[int]]] = {}
        # Track whether each stabilizer row had any temporal edge within the current open segment
        seg_had_edge: Dict[Tuple[str, str], List[bool]] = {}
        # Row wrap summaries for diagnostics
        z_row_wraps: Dict[str, List[int]] = {}
        x_row_wraps: Dict[str, List[int]] = {}

        def _ensure_seg_lists(patch_name: str, basis: str, length: int):
            key = (patch_name, basis)
            if key not in seg_first:
                seg_first[key] = [None] * length
                seg_last[key] = [None] * length
                seg_had_edge[key] = [False] * length
            else:
                # grow if needed
                if len(seg_first[key]) < length:
                    seg_first[key].extend([None] * (length - len(seg_first[key])))
                    seg_last[key].extend([None] * (length - len(seg_last[key])))
                    seg_had_edge[key].extend([False] * (length - len(seg_had_edge[key])))

        def _wrap_close_segment(patch_name: str, basis: str, stab_indices: Optional[Set[int]] = None):
            """Add wrap detectors first⊕last for current segment(s), then reset them.
            If stab_indices is None, apply to all stabilizers of the basis for the patch.
            """
            key = (patch_name, basis)
            first_list = seg_first.get(key, [])
            last_list = seg_last.get(key, [])
            had_edge_list = seg_had_edge.get(key, [])
            if not first_list:
                return
            rng = range(len(first_list)) if stab_indices is None else stab_indices
            for si in rng:
                if si is None or si >= len(first_list):
                    continue
                a = first_list[si]
                b = last_list[si] if si < len(last_list) else None
                # Only enforce a wrap if this row actually had any temporal edge within the segment
                had_edge = False
                if si < len(had_edge_list):
                    had_edge = bool(had_edge_list[si])
                if had_edge:
                    if a is not None and b is not None and a != b:
                        _defer_detector_from_abs([a, b])
                        if basis == "Z":
                            z_row_wraps.setdefault(patch_name, []).append(si)
                        else:
                            x_row_wraps.setdefault(patch_name, []).append(si)
                # reset segment for these stabilizers so a new segment can start after the gap
                if si < len(first_list):
                    first_list[si] = None
                if si < len(last_list):
                    last_list[si] = None
                if si < len(had_edge_list):
                    had_edge_list[si] = False


        # Establish initial references: Z then X for active patches
        prev = _PrevState(z_prev={}, x_prev={}, joint_prev={})
        # Cache the last joint indices per merge window and accumulate byproduct metadata
        last_window_joint: Dict[Tuple[str, str, str], List[int]] = {}
        # Track the first joint indices per merge window for wrap-around closure
        first_window_joint: Dict[Tuple[str, str, str], List[int]] = {}
        byproducts: List[Dict[str, object]] = []

        # Active merge trackers and metadata for joint windows
        active_rough: Optional[Tuple[str, str, int]] = None  # (a,b,remaining)
        active_smooth: Optional[Tuple[str, str, int]] = None
        merge_windows: List[Dict[str, object]] = []
        current_window: Optional[Dict[str, object]] = None
        window_id = 0
        
        # CNOT operation tracking for Pauli frame updates
        cnot_operations: List[Dict[str, object]] = []
        current_cnot: Optional[Dict[str, object]] = None

        # Track whether we've added a boundary anchor at the start of a window
        seam_boundary_started: Set[Tuple[str, str, str]] = set()
        # Track seam round counts and wraps for diagnostics and gating
        seam_round_counts: Dict[Tuple[str, str, str], int] = {}
        seam_wrap_counts: Dict[Tuple[str, str, str], int] = {}

        def _begin_window(kind: str, a: str, b: str, rounds: int) -> None:
            nonlocal current_window, window_id
            current_window = {
                "id": window_id,
                "type": kind,
                "parity_type": "ZZ" if kind == "rough" else "XX",
                "a": a,
                "b": b,
                "rounds": int(rounds),
            }
            window_id += 1
            seam_round_counts[(kind, a, b)] = 0

        def _end_window() -> None:
            nonlocal current_window
            if current_window is not None:
                merge_windows.append(current_window)
                current_window = None

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
                z_curr = {name: (list(vals) if vals is not None else []) for name, vals in prev.z_prev.items()}

                # Pending seam detector emission (emit after patch Z MPPs)
                pending_rough_key = None
                pending_rough_prev = None
                pending_rough_curr = None

                if measure_z:
                    # During a SMOOTH (XX) window, Z stabilizers that touch the seam anti-commute
                    # with the joint XX checks. Skip them (reuse previous indices) to keep DEM deterministic.
                    skip_z: Dict[str, Set[int]] = {}
                    if active_smooth is not None:
                        seam_a, seam_b, _ = active_smooth
                        pairs = self.layout.seams.get(("smooth", seam_a, seam_b), [])
                        if pairs:
                            skip_z[seam_a] = {ia for ia, _ in pairs}
                            skip_z[seam_b] = {ib for _, ib in pairs}
                    meas_z_curr = self._measure_patch_stabilizers(
                        circuit,
                        names,
                        "Z",
                        skip_indices=skip_z if skip_z else None,
                        prev_map=prev.z_prev,
                    )
                    for name in names:
                        # Segment tracking: remember first and last measured indices per stabilizer
                        z_list = list(meas_z_curr.get(name, []))
                        _ensure_seg_lists(name, "Z", len(z_list))
                        keyZ = (name, "Z")
                        for si, idx_abs in enumerate(z_list):
                            if idx_abs is not None:
                                if seg_first[keyZ][si] is None:
                                    seg_first[keyZ][si] = idx_abs
                                seg_last[keyZ][si] = idx_abs
                                seg_had_edge[keyZ][si] = False # Reset edge flag for new segment
                        # Suppress Z temporal detectors while this patch is in an active SMOOTH (XX) merge window
                        emit_z_dets = not (active_smooth is not None and name in {active_smooth[0], active_smooth[1]})
                        if emit_z_dets:
                            # Record temporal detector edges (absolute measurement indices)
                            p_list = list(prev.z_prev.get(name, []))
                            c_list = list(meas_z_curr.get(name, []))
                            from itertools import zip_longest as _ziplg
                            for si, (a, b) in enumerate(_ziplg(p_list, c_list, fillvalue=None)):
                                if a is None or b is None or a == b:
                                    continue
                                _defer_detector_from_abs([a, b])
                                keyZ_h = (name, "Z")
                                _ensure_seg_lists(name, "Z", max(len(p_list), len(c_list)))
                                if si < len(seg_had_edge[keyZ_h]):
                                    seg_had_edge[keyZ_h][si] = True
                        z_curr[name] = list(meas_z_curr.get(name, []))
                        if (
                            name in pending_start
                            and pending_start[name] == "Z"
                            and not (
                                (active_smooth is not None and name in {active_smooth[0], active_smooth[1]})
                                or (active_rough is not None and name in {active_rough[0], active_rough[1]})
                            )
                            and conflict_counts.get((name, "Z"), 0) == 0
                        ):
                            indices = meas_z_curr.get(name, [])
                            first_idx = next((idx for idx in indices if idx is not None), None)
                            if first_idx is not None:
                                start_indices[name] = first_idx
                                pending_start.pop(name, None)
                else:
                    for name in names:
                        if name not in z_curr:
                            z_curr[name] = list(prev.z_prev.get(name, []))

                # Rough seam: only record for later emission (after patch Z MPPs)
                if measure_z and active_rough is not None:
                    seam_a, seam_b, _ = active_rough
                    pending_rough_key = ("rough", seam_a, seam_b)
                    pending_rough_prev = list(prev.joint_prev.get(pending_rough_key, []))
                    pending_rough_curr = self._measure_joint_checks(circuit, "rough", seam_a, seam_b)
                    if pending_rough_curr:
                        seam_round_counts[pending_rough_key] = seam_round_counts.get(pending_rough_key, 0) + 1
                    # Capture the first joint round indices for wrap-around closure
                    if pending_rough_prev is None or not pending_rough_prev or all(x is None for x in pending_rough_prev):
                        if pending_rough_curr:
                            first_window_joint[pending_rough_key] = list(pending_rough_curr)

                # Emit seam temporal detectors *after* all Z MPPs of this half
                if pending_rough_key is not None and pending_rough_curr is not None:
                    if seam_round_counts.get(pending_rough_key, 0) >= 2:
                        p_list = list(pending_rough_prev if pending_rough_prev is not None else [])
                        c_list = list(pending_rough_curr)
                        from itertools import zip_longest as _ziplg
                        for a, b in _ziplg(p_list, c_list, fillvalue=None):
                            if a is None or b is None or a == b:
                                continue
                            _defer_detector_from_abs([a, b])
                    prev.joint_prev[pending_rough_key] = list(pending_rough_curr)

                prev.z_prev = z_curr

                # X half
                circuit.append_operation("TICK")
                if cfg.p_z_error:
                    circuit.append_operation("Z_ERROR", list(range(layout.global_n())), cfg.p_z_error)
                x_curr = {name: (list(vals) if vals is not None else []) for name, vals in prev.x_prev.items()}

                # Pending seam detector emission (emit after patch X MPPs)
                pending_smooth_key = None
                pending_smooth_prev = None
                pending_smooth_curr = None

                if measure_x:
                    # During a ROUGH (ZZ) window, X stabilizers that touch the seam anti-commute
                    # with the joint ZZ checks. Skip them (reuse previous indices) to keep DEM deterministic.
                    skip_x: Dict[str, Set[int]] = {}
                    if active_rough is not None:
                        seam_a, seam_b, _ = active_rough
                        pairs = self.layout.seams.get(("rough", seam_a, seam_b), [])
                        if pairs:
                            skip_x[seam_a] = {ia for ia, _ in pairs}
                            skip_x[seam_b] = {ib for _, ib in pairs}
                    meas_x_curr = self._measure_patch_stabilizers(
                        circuit,
                        names,
                        "X",
                        skip_indices=skip_x if skip_x else None,
                        prev_map=prev.x_prev,
                    )
                    for name in names:
                        # Segment tracking for X basis
                        x_list = list(meas_x_curr.get(name, []))
                        _ensure_seg_lists(name, "X", len(x_list))
                        keyX = (name, "X")
                        for si, idx_abs in enumerate(x_list):
                            if idx_abs is not None:
                                if seg_first[keyX][si] is None:
                                    seg_first[keyX][si] = idx_abs
                                seg_last[keyX][si] = idx_abs
                                seg_had_edge[keyX][si] = False # Reset edge flag for new segment
                        # Suppress X temporal detectors while this patch is in an active ROUGH (ZZ) merge window
                        emit_x_dets = not (active_rough is not None and name in {active_rough[0], active_rough[1]})
                        if emit_x_dets:
                            p_list = list(prev.x_prev.get(name, []))
                            c_list = list(meas_x_curr.get(name, []))
                            from itertools import zip_longest as _ziplg
                            for si, (a, b) in enumerate(_ziplg(p_list, c_list, fillvalue=None)):
                                if a is None or b is None or a == b:
                                    continue
                                _defer_detector_from_abs([a, b])
                                seg_had_edge[keyX][si] = True # Mark edge if temporal detector emitted
                        x_curr[name] = list(meas_x_curr.get(name, []))
                        if (
                            name in pending_start
                            and pending_start[name] == "X"
                            and not (
                                (active_smooth is not None and name in {active_smooth[0], active_smooth[1]})
                                or (active_rough is not None and name in {active_rough[0], active_rough[1]})
                            )
                            and conflict_counts.get((name, "X"), 0) == 0
                        ):
                            indices = meas_x_curr.get(name, [])
                            first_idx = next((idx for idx in indices if idx is not None), None)
                            if first_idx is not None:
                                start_indices[name] = first_idx
                                pending_start.pop(name, None)
                else:
                    for name in names:
                        if name not in x_curr:
                            x_curr[name] = list(prev.x_prev.get(name, []))

                # Smooth seam: only record for later emission (after patch X MPPs)
                if measure_x and active_smooth is not None:
                    seam_a, seam_b, _ = active_smooth
                    pending_smooth_key = ("smooth", seam_a, seam_b)
                    pending_smooth_prev = list(prev.joint_prev.get(pending_smooth_key, []))
                    pending_smooth_curr = self._measure_joint_checks(circuit, "smooth", seam_a, seam_b)
                    if pending_smooth_curr:
                        seam_round_counts[pending_smooth_key] = seam_round_counts.get(pending_smooth_key, 0) + 1
                    # Capture the first joint round indices for wrap-around closure
                    if pending_smooth_prev is None or not pending_smooth_prev or all(x is None for x in pending_smooth_prev):
                        if pending_smooth_curr:
                            first_window_joint[pending_smooth_key] = list(pending_smooth_curr)

                # Emit seam temporal detectors *after* all X MPPs of this half
                if pending_smooth_key is not None and pending_smooth_curr is not None:
                    if seam_round_counts.get(pending_smooth_key, 0) >= 2:
                        p_list = list(pending_smooth_prev if pending_smooth_prev is not None else [])
                        c_list = list(pending_smooth_curr)
                        from itertools import zip_longest as _ziplg
                        for a, b in _ziplg(p_list, c_list, fillvalue=None):
                            if a is None or b is None or a == b:
                                continue
                            _defer_detector_from_abs([a, b])
                    prev.joint_prev[pending_smooth_key] = list(pending_smooth_curr)

                prev.x_prev = x_curr

                # Update merge countdowns and close windows when done
                if active_rough is not None:
                    a, b, rem = active_rough
                    rem -= 1
                    if rem <= 0:
                        active_rough = None
                    else:
                        active_rough = (a, b, rem)
                if active_smooth is not None:
                    a, b, rem = active_smooth
                    rem -= 1
                    if rem <= 0:
                        active_smooth = None
                    else:
                        active_smooth = (a, b, rem)

            elif isinstance(op, Merge):
                k = op.type.strip().lower()
                if k not in {"rough", "smooth"}:
                    raise ValueError("Merge.type must be 'rough' or 'smooth'")
                if k == "rough":
                    if active_rough is not None:
                        raise RuntimeError("A rough merge is already active")
                    seam_pairs = layout.seams.get(("rough", op.a, op.b), [])
                    indices_a = {ia for ia, _ in seam_pairs}
                    indices_b = {ib for _, ib in seam_pairs}
                    # Seal X-basis observables on involved patches if still unset
                    for pname in (op.a, op.b):
                        if effective_basis_map.get(pname) == "X" and end_indices.get(pname) is None:
                            end_indices[pname] = _last_non_none(list(prev.x_prev.get(pname, [])))
                    mask_a = self._mask_prev_stabilizers(prev.x_prev, op.a, "X", indices_a)
                    mask_b = self._mask_prev_stabilizers(prev.x_prev, op.b, "X", indices_b)
                    # Close current X segments touching the seam before the gap
                    _wrap_close_segment(op.a, "X", mask_a)
                    _wrap_close_segment(op.b, "X", mask_b)
                    active_rough = (op.a, op.b, int(op.rounds))
                else:
                    if active_smooth is not None:
                        raise RuntimeError("A smooth merge is already active")
                    seam_pairs = layout.seams.get(("smooth", op.a, op.b), [])
                    indices_a = {ia for ia, _ in seam_pairs}
                    indices_b = {ib for _, ib in seam_pairs}
                    # Seal Z-basis observables on involved patches if still unset
                    for pname in (op.a, op.b):
                        if effective_basis_map.get(pname) == "Z" and end_indices.get(pname) is None:
                            end_indices[pname] = _last_non_none(list(prev.z_prev.get(pname, [])))
                    mask_a = self._mask_prev_stabilizers(prev.z_prev, op.a, "Z", indices_a)
                    mask_b = self._mask_prev_stabilizers(prev.z_prev, op.b, "Z", indices_b)
                    _wrap_close_segment(op.a, "Z", mask_a)
                    _wrap_close_segment(op.b, "Z", mask_b)
                    active_smooth = (op.a, op.b, int(op.rounds))
                # Clear any lingering joint history for this seam
                prev.joint_prev[(k, op.a, op.b)] = []
                _begin_window(k, op.a, op.b, int(op.rounds))

            elif isinstance(op, Split):
                k = op.type.strip().lower()
                if k not in {"rough", "smooth"}:
                    raise ValueError("Split.type must be 'rough' or 'smooth'")
                if k == "rough":
                    active_rough = None
                else:
                    active_smooth = None
                _end_window()

                # Decrement remaining conflicting merges for involved patches
                for patch_name in (op.a, op.b):
                    basis = "X" if k == "rough" else "Z"
                    key2 = (patch_name, basis)
                    if conflict_counts.get(key2, 0) > 0:
                        conflict_counts[key2] -= 1
                # Snapshot the last measured joint indices for this window before clearing
                last_window_joint[(k, op.a, op.b)] = list(prev.joint_prev.get((k, op.a, op.b), []))
                # Wrap-close the seam chain by adding a detector between the first and last joint measurements for each pair
                key = (k, op.a, op.b)
                first_list = list(first_window_joint.get(key, []))
                last_list = list(last_window_joint.get(key, []))
                # Only emit seam wrap if we had at least 2 measured rounds
                if seam_round_counts.get(key, 0) >= 2:
                    from itertools import zip_longest as _ziplg
                    wrap_added = 0
                    for a_idx, b_idx in _ziplg(first_list, last_list, fillvalue=None):
                        if a_idx is None or b_idx is None or a_idx == b_idx:
                            continue
                        _defer_detector_from_abs([a_idx, b_idx])
                        wrap_added += 1
                    if wrap_added:
                        seam_wrap_counts[key] = seam_wrap_counts.get(key, 0) + wrap_added
                # Clear stored endpoints for this window
                first_window_joint.pop(key, None)
                prev.joint_prev[(k, op.a, op.b)] = []
                seam_round_counts.pop(key, None)

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
                indices = list(last_window_joint.get(key, []))
                
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
                byproducts.append(byproduct_info)
                
                # Track CNOT operations by grouping ZZ and XX parity readouts
                if current_cnot is not None:
                    if op.type == "ZZ":
                        current_cnot["m_zz_byproduct"] = byproduct_info
                    elif op.type == "XX":
                        current_cnot["m_xx_byproduct"] = byproduct_info
                        current_cnot["target"] = op.b  # Update target (for XX, b is the target)
                        current_cnot["smooth_window_id"] = window_id - 1 if current_window is None else window_id
                        cnot_operations.append(current_cnot)
                        current_cnot = None
                else:
                    # Start of a CNOT operation (rough merge completed)
                    current_cnot = {
                        "control": op.a,
                        "target": op.b,  # This will be updated to actual target when XX comes
                        "ancilla": op.b,  # For ZZ, b is the ancilla
                        "rough_window_id": window_id - 1 if current_window is None else window_id,
                        "smooth_window_id": None,
                        "m_zz_byproduct": byproduct_info if op.type == "ZZ" else None,
                        "m_xx_byproduct": None,
                    }
                
                # NOTE: No circuit operations are emitted here to keep the DEM deterministic.

            else:
                raise TypeError(f"Unsupported op type: {type(op)!r}")

        # Close any still-open stabilizer segments (no later conflicting gaps)
        for pname in layout.patches.keys():
            _wrap_close_segment(pname, "Z", None)
            _wrap_close_segment(pname, "X", None)

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
                end_indices[pname] = _last_non_none(list(prev.z_prev.get(pname, [])))
            else:
                end_indices[pname] = _last_non_none(list(prev.x_prev.get(pname, [])))

        # Only bracket patches that are explicitly in bracket_map (excludes ancillas)
        for name in bracket_map.keys():
            if name not in layout.patches:
                continue  # Skip if patch doesn't exist in layout
            requested_basis = bracket_map[name].upper()
            effective_basis = effective_basis_map.get(name, requested_basis)

            # Prefer a pre-sealed end (set when a conflicting window began)
            end_idx = end_indices.get(name)
            if end_idx is None:
                if effective_basis == "Z":
                    end_idx = _last_non_none(list(prev.z_prev.get(name, [])))
                else:
                    end_idx = _last_non_none(list(prev.x_prev.get(name, [])))
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
            for cnot_op in cnot_operations:
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
        if deferred_detectors:
            final_m = circuit.num_measurements
            for abs_list in deferred_detectors:
                targets: List[GateTarget] = []
                for k, abs_idx in enumerate(abs_list):
                    # rec offsets are relative to current #measurements
                    targets.append(stim.target_rec(abs_idx - final_m))
                circuit.append_operation("DETECTOR", targets)

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
            "merge_windows": merge_windows,
            "observable_basis": tuple(basis_labels),
            "demo": demo_info,
            "joint_demos": joint_demo_info,
            "cnot_operations": cnot_operations,
            "final_snapshot": snapshot_info,
            "byproducts": byproducts,
            "mwpm_debug": {
                "seam_wrap_counts": {str(k): v for k, v in seam_wrap_counts.items()},
                "row_wraps": {
                    "Z": {k: list(vs) for k, vs in z_row_wraps.items()},
                    "X": {k: list(vs) for k, vs in x_row_wraps.items()},
                },
            },
        }

        return circuit, observable_pairs, metadata
