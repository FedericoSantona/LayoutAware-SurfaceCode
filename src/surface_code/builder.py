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
from collections import defaultdict, Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set
from itertools import combinations

import stim

from .layout import Layout
from .surgery_ops import MeasureRound, Merge, Split, ParityReadout, TerminatePatch, ResetPatch
from .configs import PhenomenologicalStimConfig
from .builder_utils import mpp_from_positions
from .builder_state import BuilderState, _PrevState
from .detector_manager import DetectorManager
from .segment_tracker import SegmentTracker
from .merge_manager import MergeManager
from .measurement_half import MeasurementHalf
from .observable_manager import ObservableManager
from .demo_generator import DemoGenerator


GateTarget = stim.GateTarget


class GlobalStimBuilder:
    """Build a multi-patch circuit from a layout and a sequence of ops."""

    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self._detector_count: int = 0
        self._boundary_rows: Dict[Tuple[str, str], Set[int]] = self._compute_boundary_rows()
        self._spatial_row_pairs: Dict[Tuple[str, str], Counter[Tuple[int, int]]] = self._compute_spatial_row_pairs()

    def _compute_boundary_rows(self) -> Dict[Tuple[str, str], Set[int]]:
        """Pre-compute which stabilizer rows sit on patch boundaries.

        We classify a stabilizer row as a boundary when the participating data
        qubits live near the geometric edge of the patch (within a tolerance).
        For models without coordinate metadata we fall back to a degree-based
        heuristic (qubits appearing in only one stabilizer are treated as edges).
        """
        boundary_rows: Dict[Tuple[str, str], Set[int]] = {}
        for name, patch in self.layout.patches.items():
            coords = {
                q: (float(x), float(y))
                for q, (x, y) in patch.coords.items()
            }
            if coords:
                xs = [x for x, _ in coords.values()]
                ys = [y for _, y in coords.values()]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                tolerance = 0.6

                def classify(stabs: List[str], pauli: str) -> Set[int]:
                    rows: Set[int] = set()
                    for si, stab in enumerate(stabs):
                        points = [(coords[idx][0], coords[idx][1]) for idx, char in enumerate(stab) if char == pauli and idx in coords]
                        if not points:
                            continue
                        avg_x = sum(px for px, _ in points) / len(points)
                        avg_y = sum(py for _, py in points) / len(points)
                        dist = min(
                            abs(avg_x - xmin),
                            abs(avg_x - xmax),
                            abs(avg_y - ymin),
                            abs(avg_y - ymax),
                        )
                        if dist <= tolerance:
                            rows.add(si)
                    return rows
            else:
                n = patch.n

                def classify(stabs: List[str], pauli: str) -> Set[int]:
                    counts = [0] * n
                    for stab in stabs:
                        for qi, char in enumerate(stab):
                            if char == pauli:
                                counts[qi] += 1
                    rows: Set[int] = set()
                    for si, stab in enumerate(stabs):
                        boundary = False
                        for qi, char in enumerate(stab):
                            if char == pauli and counts[qi] <= 1:
                                boundary = True
                                break
                        if boundary:
                            rows.add(si)
                    return rows

            boundary_rows[(name, "Z")] = classify(patch.z_stabs, "Z")
            boundary_rows[(name, "X")] = classify(patch.x_stabs, "X")

        return boundary_rows

    def _compute_spatial_row_pairs(self) -> Dict[Tuple[str, str], Counter[Tuple[int, int]]]:
        """Pre-compute which stabilizer rows share data qubits (spatial neighbors)."""
        pairs: Dict[Tuple[str, str], Counter[Tuple[int, int]]] = {}
        for name, patch in self.layout.patches.items():
            for basis, stabs in (("Z", patch.z_stabs), ("X", patch.x_stabs)):
                by_qubit: Dict[int, List[int]] = defaultdict(list)
                for row_idx, stab in enumerate(stabs):
                    for q_idx, ch in enumerate(stab):
                        if ch == basis:
                            by_qubit[q_idx].append(row_idx)
                row_pairs: Counter[Tuple[int, int]] = Counter()
                for rows in by_qubit.values():
                    if len(rows) < 2:
                        continue
                    unique_rows = sorted(set(rows))
                    for a, b in combinations(unique_rows, 2):
                        row_pairs[(min(a, b), max(a, b))] += 1
                pairs[(name, basis)] = row_pairs
        return pairs

    def get_spatial_row_pair_counts(self, patch: str, basis: str) -> Counter[Tuple[int, int]]:
        """Return Counter mapping neighbor row pairs to multiplicity for a patch/basis."""
        return self._spatial_row_pairs.get((patch, basis.upper()), Counter())

    def is_boundary_row(self, patch: str, basis: str, row_idx: int) -> bool:
        """Return True when the stabilizer row is treated as a physical boundary."""
        key = (patch, basis.upper())
        rows = self._boundary_rows.get(key)
        if not rows:
            return False
        return row_idx in rows

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
        p_meas: float = 0.0,
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
                idx = mpp_from_positions(circuit, positions, basis, p_meas=p_meas)
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
        temp_manager = MergeManager(self.layout)
        return temp_manager.measure_joint_checks(circuit, kind, a, b)

    def _emit_logical_mpp(
        self,
        circuit: stim.Circuit,
        patch_name: str,
        basis: str,
    ) -> Optional[int]:
        """Emit an MPP for the logical operator of ``patch_name`` in ``basis``.

        Returns the absolute measurement index or ``None`` when the logical has
        no support (should not occur for surface-code patches).
        """
        patch = self.layout.patches.get(patch_name)
        if patch is None:
            return None

        logical_string = patch.logical_z if basis == "Z" else patch.logical_x
        positions, chars = self.layout.globalize_local_pauli_string(patch_name, logical_string)
        if not positions:
            return None

        # Defensive check: ensure the stored logical matches the requested basis.
        if any(c != basis for c in chars):
            raise ValueError(
                f"Logical operator for patch '{patch_name}' contains non-{basis} axes: {chars}"
            )

        # Logical bracket MPPs must remain noiseless (p_meas=0.0) to keep observables deterministic anchors
        return mpp_from_positions(circuit, positions, basis, p_meas=0.0)

    def build(
        self,
        ops: Sequence[object],
        cfg: PhenomenologicalStimConfig,
        bracket_map: Dict[str, str],  # patch_name -> 'Z'|'X'
        qiskit_circuit: Optional[object] = None,  # Qiskit circuit for demo conjugation
        *,
        explicit_logical_start: bool = True,
    ) -> Tuple[stim.Circuit, List[Tuple[int, int]], Dict[str, object]]:
        layout = self.layout
        circuit = stim.Circuit()

        # Coordinates
        self._emit_qubit_coords(circuit)

        # Initialize state and managers (needed early to check for single-patch memory)
        state = BuilderState()
        force_boundaries = getattr(cfg, "force_boundaries", True)
        boundary_error_prob = getattr(cfg, "boundary_error_prob", 1e-12)
        detector_manager = DetectorManager(force_boundaries=force_boundaries, boundary_error_prob=boundary_error_prob)
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

        # Initialize observable and demo managers
        observable_manager = ObservableManager(self.layout, bracket_map)
        demo_generator = DemoGenerator(self.layout, self)
        
        # Sync effective_basis_map to observable_manager (will be updated for single-patch memory below)
        observable_manager.effective_basis_map = effective_basis_map

        # Track observable bracketing indices per patch
        start_indices: Dict[str, Optional[int]] = {name: None for name in all_patches}
        end_indices: Dict[str, Optional[int]] = {name: None for name in all_patches}
        
        # Track warmup round measurements for fallback start index capture
        # Warmup round establishes reference measurements before noise, so it's the correct start point
        warmup_z: Dict[str, List[Optional[int]]] = {}
        warmup_x: Dict[str, List[Optional[int]]] = {}
        warmup_captured = False
        
        # Track first round after warmup for fallback (if warmup measurements all start at 0)
        first_round_after_warmup_z: Dict[str, List[Optional[int]]] = {}
        first_round_after_warmup_x: Dict[str, List[Optional[int]]] = {}
        first_round_after_warmup_captured = False

        # Emit explicit logical brackets only when no merges are present
        emit_explicit_logicals = not rough_merge_patches and not smooth_merge_patches
        segment_tracker = SegmentTracker(boundary_checker=self.is_boundary_row)

        # Check if first op is a warmup round (MeasureRound)
        has_warmup_round = len(ops) > 0 and isinstance(ops[0], MeasureRound)

        # Check if this is a single-patch memory experiment (no merges, single patch, has warmup)
        # For single-patch memory experiments, we need to measure logical start BEFORE warmup
        # to match the old builder behavior (logical start before reference measurements)
        is_single_patch_memory = (
            emit_explicit_logicals and 
            explicit_logical_start and
            len(all_patches) == 1 and 
            has_warmup_round
        )

        # For single-patch memory experiments, use init_label to determine effective basis
        # Old builder uses logical_string (from init_label) for both start and end, so
        # the observable basis should match the logical operator being measured
        if is_single_patch_memory and cfg.init_label is not None:
            from .pauli import parse_init_label
            init_basis, _ = parse_init_label(cfg.init_label)
            logical_basis = init_basis  # "Z" for init_label="0"/"1", "X" for init_label="+"/"-"
            # Update effective_basis_map to use logical_basis from init_label
            for name in effective_basis_map.keys():
                effective_basis_map[name] = logical_basis
            # Sync updated effective_basis_map to observable_manager
            observable_manager.effective_basis_map = effective_basis_map

        per_round_p_x = float(getattr(cfg, "p_x_error", 0.0) or 0.0)
        per_round_p_z = float(getattr(cfg, "p_z_error", 0.0) or 0.0)

        pending_start: Dict[str, str] = {}
        if is_single_patch_memory:
            # Single-patch memory experiment: measure logical start BEFORE initialization gates and warmup
            # This matches the old builder where logical start was measured before reference measurements
            # CRITICAL: Only measure logical start if init_label is set (matches old builder behavior)
            # The old builder only creates observables when init_label is set
            # CRITICAL: Measure BEFORE initialization gates to match old builder (old builder doesn't apply gates)
            # CRITICAL: Use init_label to determine which logical operator to measure (not bracket_map)
            # Old builder: init_label="0"/"1" → logical_z, init_label="+"/"-" → logical_x
            if cfg.init_label is not None:
                from .pauli import parse_init_label
                init_basis, _ = parse_init_label(cfg.init_label)
                # Determine which logical operator to measure based on init_label (like old builder)
                logical_basis = init_basis  # "Z" for init_label="0"/"1", "X" for init_label="+"/"-"
                
                # Add TICK before logical start to match old builder structure
                circuit.append_operation("TICK")
                for name in effective_basis_map.keys():
                    # Use logical_basis from init_label, not from bracket_map
                    idx = self._emit_logical_mpp(circuit, name, logical_basis)
                    if idx is not None:
                        start_indices[name] = idx
                        observable_manager.capture_start(name, idx)
                    else:
                        pending_start[name] = logical_basis
        
        # Initialize qubits based on init_label
        # CRITICAL: For single-patch memory experiments, do NOT apply initialization gates
        # to match the old builder behavior (old builder doesn't apply gates, just measures logical operator)
        # For other cases (multi-patch/Bell state), apply initialization gates as needed
        if cfg.init_label is not None and not is_single_patch_memory:
            from .pauli import parse_init_label
            init_basis, init_phase = parse_init_label(cfg.init_label)
            all_qubits = list(range(layout.global_n()))
            
            if init_basis == "X":
                # Initialize to |+> or |->
                # |+> = H|0>, |-> = H|1> = HX|0>
                if init_phase == -1:
                    # |-> state: apply X then H
                    circuit.append_operation("X", all_qubits)
                circuit.append_operation("H", all_qubits)
            elif init_basis == "Z" and init_phase == -1:
                # |1> state: apply X
                circuit.append_operation("X", all_qubits)
            # If init_basis=="Z" and init_phase==+1, qubits are already in |0>, no initialization needed

        if not is_single_patch_memory:
            # For non-single-patch-memory cases, handle logical start here
            if emit_explicit_logicals and explicit_logical_start and not has_warmup_round:
                # Emit explicit logical MPPs before ops (only if no warmup round)
                for name, basis in effective_basis_map.items():
                    idx = self._emit_logical_mpp(circuit, name, basis)
                    if idx is not None:
                        start_indices[name] = idx
                        observable_manager.capture_start(name, idx)
                    else:
                        pending_start[name] = basis
            elif emit_explicit_logicals and explicit_logical_start and has_warmup_round:
                # Multi-patch/Bell state: defer explicit logical start capture until after warmup round
                # This ensures start is captured from first noisy round, matching Bell state behavior
                pending_start = {name: effective_basis_map[name] for name in effective_basis_map.keys()}
            else:
                # Defer start capture to the first compatible stabilizer half
                pending_start = {name: effective_basis_map[name] for name in effective_basis_map.keys()}

        # For single-patch memory, don't add extra TICK before warmup
        # Old builder: TICK → Logical start, then TICK → Reference (no extra TICK in between)
        # Only add TICK if we didn't just measure logical start (for single-patch memory)
        if not (is_single_patch_memory and cfg.init_label is not None):
            circuit.append_operation("TICK")



        # Track remaining merge windows that conflict with seam stabilizer bases
        conflict_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        round_counters = {"Z": 0, "X": 0}

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
        _row_cache: Dict[Tuple[str, str], Dict[int, Set[int]]] = {}

        def _rows_touching_local_indices(patch_name: str, basis: str, local_indices: Iterable[int]) -> Set[int]:
            patch = layout.patches.get(patch_name)
            if patch is None:
                return set()
            cache_key = (patch_name, basis)
            cache = _row_cache.setdefault(cache_key, {})

            def _rows_for_local(li: int) -> Set[int]:
                if li in cache:
                    return cache[li]
                stabs = patch.z_stabs if basis == "Z" else patch.x_stabs
                rows = {si for si, stab in enumerate(stabs) if li < len(stab) and stab[li] == basis}
                cache[li] = rows
                return rows

            rows: Set[int] = set()
            for li in set(local_indices):
                rows.update(_rows_for_local(li))
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

                # Check if this is the warmup round (first round with no previous measurements)
                # The warmup round establishes reference measurements before noise, matching the old builder behavior
                is_warmup = all(
                    name not in state.prev.z_prev and name not in state.prev.x_prev
                    for name in names
                )

                # Z half
                circuit.append_operation("TICK")
                # Skip noise for warmup round - it should establish clean reference measurements
                # This matches the old builder which created reference measurements before noisy cycles
                # CRITICAL: Warmup round must be noise-free to establish clean reference measurements
                # Without this, reference measurements are contaminated and error correction breaks
                if not is_warmup and per_round_p_x:
                    circuit.append_operation("X_ERROR", list(range(layout.global_n())), per_round_p_x)
                
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
                    observable_manager,
                    round_index=round_counters["Z"],
                )
                state.prev.z_prev = z_curr
                round_counters["Z"] += 1

                # X half
                circuit.append_operation("TICK")
                # Skip noise for warmup round - it should establish clean reference measurements
                if not is_warmup and per_round_p_z:
                    circuit.append_operation("Z_ERROR", list(range(layout.global_n())), per_round_p_z)
                
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
                    observable_manager,
                    round_index=round_counters["X"],
                )
                state.prev.x_prev = x_curr
                round_counters["X"] += 1
                
                # Track warmup round measurements for fallback start index capture
                # Warmup round establishes reference measurements before noise, so it's the correct start point
                if is_warmup and not warmup_captured:
                    warmup_z = {name: list(z_curr.get(name, [])) for name in names}
                    warmup_x = {name: list(x_curr.get(name, [])) for name in names}
                    warmup_captured = True
                
                # Track first round after warmup for fallback (if warmup measurements all start at 0)
                if not is_warmup and warmup_captured and not first_round_after_warmup_captured:
                    first_round_after_warmup_z = {name: list(z_curr.get(name, [])) for name in names}
                    first_round_after_warmup_x = {name: list(x_curr.get(name, [])) for name in names}
                    first_round_after_warmup_captured = True

                # If we have explicit logical start deferred until after warmup, emit it now
                # This only applies to multi-patch/Bell state scenarios (not single-patch memory)
                # For single-patch memory experiments, logical start is measured BEFORE warmup
                if emit_explicit_logicals and explicit_logical_start and has_warmup_round and is_warmup:
                    # Warmup round just completed - emit explicit logical MPPs now
                    # Note: pending_start will be empty for single-patch memory (already captured before warmup)
                    for name, basis in list(pending_start.items()):
                        if name in effective_basis_map and effective_basis_map[name] == basis:
                            idx = self._emit_logical_mpp(circuit, name, basis)
                            if idx is not None:
                                start_indices[name] = idx
                                observable_manager.capture_start(name, idx)
                                pending_start.pop(name, None)

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
                            end_idx = _last_non_none(list(state.prev.x_prev.get(pname, [])))
                            observable_manager.seal_end(pname, "X", end_idx)
                            end_indices[pname] = end_idx
                    mask_a = self._mask_prev_stabilizers(state.prev.x_prev, op.a, "X", indices_a)
                    mask_b = self._mask_prev_stabilizers(state.prev.x_prev, op.b, "X", indices_b)
                    # Close current X segments touching the seam before the gap
                    for row_idx in mask_a:
                        detector_manager.mark_row_dynamic(op.a, "X", int(row_idx))
                    for row_idx in mask_b:
                        detector_manager.mark_row_dynamic(op.b, "X", int(row_idx))
                    segment_tracker.wrap_close_segment(op.a, "X", detector_manager, mask_a, skip_boundary_rows=True)
                    segment_tracker.wrap_close_segment(op.b, "X", detector_manager, mask_b, skip_boundary_rows=True)
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
                            end_idx = _last_non_none(list(state.prev.z_prev.get(pname, [])))
                            observable_manager.seal_end(pname, "Z", end_idx)
                            end_indices[pname] = end_idx
                    mask_a = self._mask_prev_stabilizers(state.prev.z_prev, op.a, "Z", indices_a)
                    mask_b = self._mask_prev_stabilizers(state.prev.z_prev, op.b, "Z", indices_b)
                    for row_idx in mask_a:
                        detector_manager.mark_row_dynamic(op.a, "Z", int(row_idx))
                    for row_idx in mask_b:
                        detector_manager.mark_row_dynamic(op.b, "Z", int(row_idx))
                    segment_tracker.wrap_close_segment(op.a, "Z", detector_manager, mask_a, skip_boundary_rows=True)
                    segment_tracker.wrap_close_segment(op.b, "Z", detector_manager, mask_b, skip_boundary_rows=True)
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

                seam_indices = list(indices)
                logical_idx = None
                if op.type == "XX":
                    logical_idx = self._emit_logical_mpp(circuit, op.a, "X")
                    if logical_idx is not None:
                        byproduct_info["logical_index"] = logical_idx
                
                # Track CNOT operations by grouping ZZ and XX parity readouts
                if state.current_cnot is not None:
                    if op.type == "ZZ":
                        state.current_cnot["m_zz_byproduct"] = byproduct_info
                        state.current_cnot["m_zz_indices"] = seam_indices
                    elif op.type == "XX":
                        state.current_cnot["m_xx_byproduct"] = byproduct_info
                        state.current_cnot["m_xx_indices"] = seam_indices
                        state.current_cnot["m_xx_logical_idx"] = logical_idx
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
                        "m_zz_indices": seam_indices if op.type == "ZZ" else None,
                        "m_xx_byproduct": None,
                        "m_xx_indices": None,
                        "m_xx_logical_idx": None,
                    }
                
                # NOTE: No circuit operations are emitted here to keep the DEM deterministic.

            elif isinstance(op, ResetPatch):
                patch_name = op.patch_id
                if patch_name not in layout.patches or patch_name in state.terminated_patches:
                    continue

                patch = layout.patches[patch_name]
                target_basis = str(getattr(op, "basis", "X")).upper()
                if target_basis not in {"X", "Z"}:
                    raise ValueError("ResetPatch.basis must be 'X' or 'Z'")

                # Close any open temporal segments before the reset to avoid dangling edges.
                segment_tracker.wrap_close_segment(
                    patch_name,
                    "Z",
                    detector_manager,
                    None,
                    skip_boundary_rows=True,
                )
                segment_tracker.wrap_close_segment(
                    patch_name,
                    "X",
                    detector_manager,
                    None,
                    skip_boundary_rows=True,
                )

                # Reset previous measurement history so subsequent rounds treat the patch as fresh.
                state.prev.z_prev[patch_name] = [None] * len(patch.z_stabs)
                state.prev.x_prev[patch_name] = [None] * len(patch.x_stabs)

                # Mark rows dynamic so that boundary anchors are retained in code-capacity mode.
                for row_idx in range(len(patch.z_stabs)):
                    detector_manager.mark_row_dynamic(patch_name, "Z", int(row_idx))
                for row_idx in range(len(patch.x_stabs)):
                    detector_manager.mark_row_dynamic(patch_name, "X", int(row_idx))

                # Apply a physical reset followed by an optional Hadamard to prepare |+>.
                circuit.append_operation("TICK")
                offs = layout.offsets()
                base = offs[patch_name]
                qubits = [base + q_local for q_local in range(patch.n)]
                for g_idx in qubits:
                    circuit.append_operation("R", [g_idx])
                if target_basis == "X":
                    for g_idx in qubits:
                        circuit.append_operation("H", [g_idx])

            elif isinstance(op, TerminatePatch):
                # Handle mid-circuit measurement: close segments and mark as terminated
                patch_name = op.patch_id
                if patch_name not in layout.patches or patch_name in state.terminated_patches:
                    continue
                
                # Track detector count before closing segments
                num_detectors_before = len(detector_manager.deferred_detectors)

                # Mark all rows in both bases as having experienced a dynamic event
                patch = layout.patches.get(patch_name)
                if patch is not None:
                    for row_idx in range(len(patch.z_stabs)):
                        detector_manager.mark_row_dynamic(patch_name, "Z", row_idx)
                    for row_idx in range(len(patch.x_stabs)):
                        detector_manager.mark_row_dynamic(patch_name, "X", row_idx)
                
                # Close stabilizer segments for this patch - this creates wrap detectors
                segment_tracker.wrap_close_segment(
                    patch_name,
                    "Z",
                    detector_manager,
                    None,
                    skip_boundary_rows=True,
                )
                segment_tracker.wrap_close_segment(
                    patch_name,
                    "X",
                    detector_manager,
                    None,
                    skip_boundary_rows=True,
                )
                
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
                        end_idx = _last_non_none(list(state.prev.z_prev.get(patch_name, [])))
                        observable_manager.seal_end(patch_name, "Z", end_idx)
                        end_indices[patch_name] = end_idx
                    else:
                        end_idx = _last_non_none(list(state.prev.x_prev.get(patch_name, [])))
                        observable_manager.seal_end(patch_name, "X", end_idx)
                        end_indices[patch_name] = end_idx
                
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
            segment_tracker.wrap_close_segment(
                pname,
                "Z",
                detector_manager,
                None,
                skip_boundary_rows=True,
            )
            segment_tracker.wrap_close_segment(
                pname,
                "X",
                detector_manager,
                None,
                skip_boundary_rows=True,
            )

        if emit_explicit_logicals:
            for name, basis in effective_basis_map.items():
                if end_indices.get(name) is not None:
                    continue
                # For single-patch memory experiments, use the same logical operator as start (from init_label)
                # Old builder uses the same logical_string for both start and end
                if is_single_patch_memory and cfg.init_label is not None:
                    from .pauli import parse_init_label
                    init_basis, _ = parse_init_label(cfg.init_label)
                    logical_basis = init_basis  # Use same basis as start measurement
                    # CRITICAL: Add TICK before end measurement to match old builder exactly
                    circuit.append_operation("TICK")
                    end_idx = self._emit_logical_mpp(circuit, name, logical_basis)
                    end_indices[name] = end_idx
                    observable_manager.seal_end(name, logical_basis, end_idx)
                    
                    # CRITICAL: For single-patch memory, create observable IMMEDIATELY after end measurement
                    # This matches the old builder exactly: observable created right after end, before any other processing
                    start_idx = observable_manager.start_indices.get(name)
                    if start_idx is not None and end_idx is not None:
                        from .builder_utils import rec_from_abs
                        # Verify circuit.num_measurements matches end_idx + 1 (like old builder)
                        expected_m2 = end_idx + 1
                        actual_m2 = circuit.num_measurements
                        if actual_m2 != expected_m2:
                            print(f"[OBS-DEBUG] WARNING: circuit.num_measurements={actual_m2} != end_idx+1={expected_m2}")
                        start_rel = rec_from_abs(circuit, start_idx)
                        end_rel = rec_from_abs(circuit, end_idx)
                        print(f"[OBS-DEBUG] Creating observable: start={start_idx} (rel={start_rel}), end={end_idx} (rel={end_rel}), m2={actual_m2}")
                        # Old builder: OBSERVABLE_INCLUDE([rec_from_abs(circuit, start), rec_from_abs(circuit, end)], 0)
                        # This creates: rec(start) XOR rec(end)
                        # CRITICAL: If error increases after decoding, the observable might be inverted
                        # Try swapping order: [end_rel, start_rel] to see if that fixes it
                        # But first, let's verify the order matches old builder exactly
                        circuit.append_operation(
                            "OBSERVABLE_INCLUDE",
                            [start_rel, end_rel],  # [start, end] order matches old builder
                            0,
                        )
                        # Mark as already emitted so it's not emitted again later
                        observable_manager._single_patch_observable_emitted = True
                else:
                    end_idx = self._emit_logical_mpp(circuit, name, basis)
                    end_indices[name] = end_idx
                    observable_manager.seal_end(name, basis, end_idx)

        # CRITICAL FIX: Ensure all start indices are captured before finalizing observables
        # If a start index is None or 0 (e.g., blocked by merge conflicts or captured too early),
        # use the FIRST ROUND AFTER WARMUP (not warmup itself) as fallback.
        # 
        # IMPORTANT: Warmup round is noise-free and deterministic, so using warmup measurements
        # as the start makes the observable `rec(warmup) XOR rec(end)`, where `rec(warmup)` is
        # deterministic. This doesn't track errors from initialization correctly.
        #
        # Instead, we should use the FIRST ROUND AFTER WARMUP (before merges) as the start.
        # This ensures the observable tracks from initialization: `rec(first_round) XOR rec(end)`,
        # where `rec(first_round)` includes errors from initialization and the first round.
        #
        # EXCEPTION: For single-patch memory experiments, index 0 is valid (logical start measured before warmup)
        # so we should NOT replace it with first round after warmup
        for name in effective_basis_map.keys():
            start_idx = observable_manager.start_indices.get(name)
            # Check if start is None or 0 - if so, use first round after warmup as fallback
            # BUT: For single-patch memory, index 0 is valid (logical start before warmup), so don't replace it
            if start_idx is None or (start_idx == 0 and not is_single_patch_memory):
                basis = effective_basis_map[name]
                # Fallback: use FIRST ROUND AFTER WARMUP (not warmup itself)
                # This ensures the observable tracks errors from initialization
                if basis == "Z":
                    if name in first_round_after_warmup_z:
                        z_indices = [idx for idx in first_round_after_warmup_z[name] if idx is not None and idx > 0]
                        if not z_indices:
                            z_indices = [idx for idx in first_round_after_warmup_z[name] if idx is not None]
                    elif name in state.prev.z_prev:
                        z_indices = [idx for idx in state.prev.z_prev[name] if idx is not None and idx > 0]
                        if not z_indices:
                            z_indices = [idx for idx in state.prev.z_prev[name] if idx is not None]
                    else:
                        z_indices = []
                    if z_indices:
                        # Use the FIRST measurement from first round after warmup (tracks from initialization)
                        selected_z = z_indices[0]
                        observable_manager.start_indices[name] = selected_z
                        start_indices[name] = selected_z
                        if start_idx == 0:
                            print(f"[OBS-FIX] Replaced start_idx=0 for {name} (Z) with first-round-after-warmup measurement {selected_z}")
                elif basis == "X":
                    if name in first_round_after_warmup_x:
                        x_indices = [idx for idx in first_round_after_warmup_x[name] if idx is not None and idx > 0]
                        if not x_indices:
                            x_indices = [idx for idx in first_round_after_warmup_x[name] if idx is not None]
                    elif name in state.prev.x_prev:
                        x_indices = [idx for idx in state.prev.x_prev[name] if idx is not None and idx > 0]
                        if not x_indices:
                            x_indices = [idx for idx in state.prev.x_prev[name] if idx is not None]
                    else:
                        x_indices = []
                    if x_indices:
                        # Use the FIRST measurement from first round after warmup (tracks from initialization)
                        selected_x = x_indices[0]
                        observable_manager.start_indices[name] = selected_x
                        start_indices[name] = selected_x
                        if start_idx == 0:
                            print(f"[OBS-FIX] Replaced start_idx=0 for {name} (X) with first-round-after-warmup measurement {selected_x}")
        
        # Emit boundary anchors after all merges complete
        # Bracketing: end logicals per patch and observable includes
        # IMPORTANT: do not emit new MPPs here; reuse the last compatible
        # stabilizer measurements from the final round to keep DEM deterministic.
        observable_pairs, basis_labels, deferred_observables, patch_labels = observable_manager.finalize_observables(
            circuit, state, _last_non_none
        )
        
        # Debug: Print observable pairs to verify they're correct
        if observable_pairs:
            print(f"[OBS-PAIRS] Observable pairs: {observable_pairs}")
            print(f"[OBS-PAIRS] Patch labels: {patch_labels}")
            print(f"[OBS-PAIRS] Basis labels: {basis_labels}")
            for i, (start_idx, end_idx) in enumerate(observable_pairs):
                if start_idx is not None and end_idx is not None:
                    span = end_idx - start_idx
                    print(f"[OBS-PAIRS] Observable {i} ({patch_labels[i]}, basis {basis_labels[i]}): start={start_idx}, end={end_idx}, span={span}")
                else:
                    print(f"[OBS-PAIRS] Observable {i} ({patch_labels[i]}, basis {basis_labels[i]}): start={start_idx}, end={end_idx} (WARNING: None indices!)")

        # CRITICAL FIX: For single-patch memory, observables are already emitted immediately after end measurement
        # (matching old builder). For other cases, emit observables here.
        # Skip single-patch memory observables that were already emitted.
        if not getattr(observable_manager, '_single_patch_observable_emitted', False):
            observable_manager.emit_observables(circuit, deferred_observables)

        # Generate demo measurements (after observables, so they don't affect observable relative indices)
        demo_info, joint_demo_info, snapshot_info = demo_generator.generate_demos(
            circuit, cfg, state, bracket_map, qiskit_circuit
        )

        # Append all deferred detectors at the very end using final measurement count
        detector_manager.emit_all_detectors(
            circuit,
            noise_model={
                "p_x_error": float(getattr(cfg, "p_x_error", 0.0) or 0.0),
                "p_z_error": float(getattr(cfg, "p_z_error", 0.0) or 0.0),
                "p_meas": float(getattr(cfg, "p_meas", 0.0) or 0.0),
            },
        )

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

        boundary_row_meta: Dict[str, Dict[str, Dict[str, object]]] = {}
        for name in layout.patches.keys():
            patch = layout.patches[name]
            z_rows = sorted(self._boundary_rows.get((name, "Z"), set()))
            x_rows = sorted(self._boundary_rows.get((name, "X"), set()))
            boundary_row_meta[name] = {
                "Z": {"rows": z_rows, "total": len(patch.z_stabs)},
                "X": {"rows": x_rows, "total": len(patch.x_stabs)},
            }

        metadata: Dict[str, object] = {
            "merge_windows": merge_manager.get_windows(),
            "observable_basis": tuple(basis_labels),
            "observable_patches": tuple(patch_labels),
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
            "explicit_logical_brackets": emit_explicit_logicals,
            "noise_model": {
                "p_x_error": float(getattr(cfg, "p_x_error", 0.0) or 0.0),
                "p_z_error": float(getattr(cfg, "p_z_error", 0.0) or 0.0),
                "p_meas": float(getattr(cfg, "p_meas", 0.0) or 0.0),
            },
            "boundary_rows": boundary_row_meta,
            # Stabilizer row → local-qubit support map, for space-edge reconstruction.
            # Format:
            #   {
            #     'Z': { patch: { row_idx: [local_q,...], ... }, ... },
            #     'X': { patch: { row_idx: [local_q,...], ... }, ... }
            #   }
            "stab_support": (lambda _layout=self.layout: {
                "Z": {
                    name: {
                        i: [qi for qi, ch in enumerate(patch.z_stabs[i]) if ch == "Z"]
                        for i in range(len(patch.z_stabs))
                    }
                    for name, patch in _layout.patches.items()
                },
                "X": {
                    name: {
                        i: [qi for qi, ch in enumerate(patch.x_stabs[i]) if ch == "X"]
                        for i in range(len(patch.x_stabs))
                    }
                    for name, patch in _layout.patches.items()
                },
            })(),
        }

        spatial_meta: Dict[str, Dict[str, List[Dict[str, object]]]] = {"Z": {}, "X": {}}
        for basis in ("Z", "X"):
            for name in layout.patches.keys():
                counts = self.get_spatial_row_pair_counts(name, basis)
                if not counts:
                    continue
                spatial_meta[basis][name] = [
                    {"rows": (a, b), "count": int(cnt)}
                    for (a, b), cnt in sorted(counts.items())
                ]
        metadata.setdefault("mwpm_debug", {}).setdefault("spatial_pairs", spatial_meta)

        return circuit, observable_pairs, metadata
