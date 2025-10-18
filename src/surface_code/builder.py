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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import stim

from .layout import Layout
from .surgery_ops import MeasureRound, Merge, Split, ParityReadout
from .configs import PhenomenologicalStimConfig
from .builder_utils import mpp_from_positions, rec_from_abs, add_temporal_detectors_with_index


GateTarget = stim.GateTarget


def _mpp_from_positions(circuit: stim.Circuit, positions: Sequence[int], pauli: str) -> Optional[int]:
    """Append an MPP for a tensor product of identical Paulis at given positions.

    Returns the absolute measurement index, or None if positions is empty.
    """
    return mpp_from_positions(circuit, positions, pauli)


def _rec_from_abs(circuit: stim.Circuit, idx: int) -> GateTarget:
    return rec_from_abs(circuit, idx)


def _add_temporal_detectors_with_index(
    circuit: stim.Circuit,
    prev: Sequence[int],
    curr: Sequence[int],
    append_detector_cb,
) -> List[int]:
    return add_temporal_detectors_with_index(circuit, prev, curr, append_detector_cb)


@dataclass
class _PrevState:
    z_prev: Dict[str, List[int]]
    x_prev: Dict[str, List[int]]
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
    ) -> Dict[str, List[int]]:
        offs = self.layout.offsets()
        measured: Dict[str, List[int]] = {}
        for name in patch_names:
            patch = self.layout.patches[name]
            base = offs[name]
            stabs = patch.z_stabs if basis == "Z" else patch.x_stabs
            indices: List[int] = []
            for s in stabs:
                # Build a global MPP by mapping local characters to global positions
                positions: List[int] = []
                for i, c in enumerate(s):
                    if c == basis:
                        positions.append(base + i)
                idx = _mpp_from_positions(circuit, positions, basis)
                if idx is not None:
                    indices.append(idx)
            measured[name] = indices
        return measured

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
        offs = self.layout.offsets()
        base_a = offs[a]
        base_b = offs[b]
        pauli = "Z" if kind == "rough" else "X"
        indices: List[int] = []
        for ia, ib in pairs:
            idx = _mpp_from_positions(circuit, [base_a + ia, base_b + ib], pauli)
            if idx is not None:
                indices.append(idx)
        return indices

    def build(
        self,
        ops: Sequence[object],
        cfg: PhenomenologicalStimConfig,
        bracket_map: Dict[str, str],  # patch_name -> 'Z'|'X'
    ) -> Tuple[stim.Circuit, List[Tuple[int, int]], Dict[str, object]]:
        layout = self.layout
        circuit = stim.Circuit()

        # Coordinates
        self._emit_qubit_coords(circuit)

        # Detector index appender (tracks DEM column indices)
        def append_detector(targets: List[GateTarget]) -> int:
            idx = self._detector_count
            circuit.append_operation("DETECTOR", targets)
            self._detector_count += 1
            return idx

        # Resolve patch order and selection helpers
        all_patches: List[str] = list(layout.patches.keys())

        def select_patches(spec: Optional[List[str]]) -> List[str]:
            return all_patches if spec is None else list(spec)

        # Bracketing: start logicals per patch
        start_indices: Dict[str, Optional[int]] = {name: None for name in all_patches}
        end_indices: Dict[str, Optional[int]] = {name: None for name in all_patches}

        circuit.append_operation("TICK")
        for name in all_patches:
            basis = bracket_map.get(name, "Z").upper()
            if basis not in {"Z", "X"}:
                raise ValueError("bracket_map values must be 'Z' or 'X'")
            patch = layout.patches[name]
            s = patch.logical_z if basis == "Z" else patch.logical_x
            # Convert local string to global positions with same basis
            offs = layout.offsets()[name]
            positions = [offs + i for i, c in enumerate(s) if c == basis]
            start_indices[name] = _mpp_from_positions(circuit, positions, basis)

        # Establish initial references: Z then X for active patches
        prev = _PrevState(z_prev={}, x_prev={}, joint_prev={})
        # Z refs
        circuit.append_operation("TICK")
        z_meas = self._measure_patch_stabilizers(circuit, all_patches, "Z")
        prev.z_prev = z_meas
        # X refs
        circuit.append_operation("TICK")
        x_meas = self._measure_patch_stabilizers(circuit, all_patches, "X")
        prev.x_prev = x_meas

        # Active merge trackers and metadata for joint windows
        active_rough: Optional[Tuple[str, str, int]] = None  # (a,b,remaining)
        active_smooth: Optional[Tuple[str, str, int]] = None
        merge_windows: List[Dict[str, object]] = []
        current_window: Optional[Dict[str, object]] = None
        window_id = 0
        
        # CNOT operation tracking for Pauli frame updates
        cnot_operations: List[Dict[str, object]] = []
        current_cnot: Optional[Dict[str, object]] = None

        def _begin_window(kind: str, a: str, b: str, rounds: int) -> None:
            nonlocal current_window, window_id
            current_window = {
                "id": window_id,
                "type": kind,
                "parity_type": "ZZ" if kind == "rough" else "XX",
                "a": a,
                "b": b,
                "rounds": int(rounds),
                "seam_pairs_global": [
                    (layout.offsets()[a] + ia, layout.offsets()[b] + ib)
                    for ia, ib in layout.seams.get((kind, a, b), [])
                ],
                "joint_meas_indices": [],
                "joint_detector_indices": [],
            }
            window_id += 1

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

                # Z half
                circuit.append_operation("TICK")
                if cfg.p_x_error:
                    circuit.append_operation("X_ERROR", list(range(layout.global_n())), cfg.p_x_error)
                z_curr = self._measure_patch_stabilizers(circuit, names, "Z")
                for name in names:
                    _add_temporal_detectors_with_index(
                        circuit,
                        prev.z_prev.get(name, []),
                        z_curr.get(name, []),
                        append_detector,
                    )
                # Joint checks for rough merge (if active)
                if active_rough is not None:
                    a, b, rem = active_rough
                    joint_curr = self._measure_joint_checks(circuit, "rough", a, b)
                    joint_det_idxs = _add_temporal_detectors_with_index(
                        circuit,
                        prev.joint_prev.get(("rough", a, b), []),
                        joint_curr,
                        append_detector,
                    )
                    prev.joint_prev[("rough", a, b)] = joint_curr
                    if current_window is not None and current_window.get("type") == "rough":
                        current_window["joint_meas_indices"].append(joint_curr)
                        current_window["joint_detector_indices"].append(joint_det_idxs)
                    # countdown managed outside the half

                prev.z_prev = z_curr

                # X half
                circuit.append_operation("TICK")
                if cfg.p_z_error:
                    circuit.append_operation("Z_ERROR", list(range(layout.global_n())), cfg.p_z_error)
                x_curr = self._measure_patch_stabilizers(circuit, names, "X")
                for name in names:
                    _add_temporal_detectors_with_index(
                        circuit,
                        prev.x_prev.get(name, []),
                        x_curr.get(name, []),
                        append_detector,
                    )
                # Joint checks for smooth merge (if active)
                if active_smooth is not None:
                    a, b, rem = active_smooth
                    joint_curr = self._measure_joint_checks(circuit, "smooth", a, b)
                    joint_det_idxs = _add_temporal_detectors_with_index(
                        circuit,
                        prev.joint_prev.get(("smooth", a, b), []),
                        joint_curr,
                        append_detector,
                    )
                    prev.joint_prev[("smooth", a, b)] = joint_curr
                    if current_window is not None and current_window.get("type") == "smooth":
                        current_window["joint_meas_indices"].append(joint_curr)
                        current_window["joint_detector_indices"].append(joint_det_idxs)

                prev.x_prev = x_curr

                # Update merge countdowns and close windows when done
                if active_rough is not None:
                    a, b, rem = active_rough
                    rem -= 1
                    if rem <= 0:
                        active_rough = None
                        _end_window()
                    else:
                        active_rough = (a, b, rem)
                if active_smooth is not None:
                    a, b, rem = active_smooth
                    rem -= 1
                    if rem <= 0:
                        active_smooth = None
                        _end_window()
                    else:
                        active_smooth = (a, b, rem)

            elif isinstance(op, Merge):
                k = op.type.strip().lower()
                if k not in {"rough", "smooth"}:
                    raise ValueError("Merge.type must be 'rough' or 'smooth'")
                if k == "rough":
                    if active_rough is not None:
                        raise RuntimeError("A rough merge is already active")
                    active_rough = (op.a, op.b, int(op.rounds))
                else:
                    if active_smooth is not None:
                        raise RuntimeError("A smooth merge is already active")
                    active_smooth = (op.a, op.b, int(op.rounds))
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

            elif isinstance(op, ParityReadout):
                # Track CNOT operations by grouping ZZ and XX parity readouts
                if op.type == "ZZ":
                    # Start of a CNOT operation (rough merge completed)
                    current_cnot = {
                        "control": op.a,
                        "target": op.b,  # This will be updated to actual target when XX comes
                        "ancilla": op.b,  # For ZZ, b is the ancilla
                        "rough_window_id": window_id - 1 if current_window is None else window_id,
                        "smooth_window_id": None,
                        "m_zz": None,  # Will be filled in post-processing
                        "m_xx": None,  # Will be filled in post-processing
                    }
                elif op.type == "XX" and current_cnot is not None:
                    # Complete the CNOT operation (smooth merge completed)
                    current_cnot["target"] = op.b  # Update target (for XX, b is the target)
                    current_cnot["smooth_window_id"] = window_id - 1 if current_window is None else window_id
                    cnot_operations.append(current_cnot)
                    current_cnot = None

            else:
                raise TypeError(f"Unsupported op type: {type(op)!r}")

        # Bracketing: end logicals per patch and observable includes
        circuit.append_operation("TICK")
        observable_pairs: List[Tuple[int, int]] = []
        basis_labels: List[str] = []
        for name in all_patches:
            basis = bracket_map.get(name, "Z").upper()
            patch = layout.patches[name]
            s = patch.logical_z if basis == "Z" else patch.logical_x
            offs = layout.offsets()[name]
            positions = [offs + i for i, c in enumerate(s) if c == basis]
            end_idx = _mpp_from_positions(circuit, positions, basis)
            end_indices[name] = end_idx
            start_idx = start_indices[name]
            targets: List[GateTarget] = []
            if start_idx is not None:
                targets.append(_rec_from_abs(circuit, start_idx))
            if end_idx is not None:
                targets.append(_rec_from_abs(circuit, end_idx))
            if targets:
                circuit.append_operation("OBSERVABLE_INCLUDE", targets, 0)
                observable_pairs.append((start_idx, end_idx))
                basis_labels.append(basis)

        # Optional end-only demo readout per patch (same basis for all patches, if requested)
        demo_info: Dict[str, object] = {}
        # Try to read a bracket/demo basis from cfg; treat any invalid as disabled
        demo_basis = None
        try:
            db = getattr(cfg, "demo_basis", None)
            if db is not None:
                b = str(db).strip().upper()
                if b in {"X", "Z"}:
                    demo_basis = b
        except Exception:
            demo_basis = None

        if demo_basis is not None:
            circuit.append_operation("TICK")
            for name in all_patches:
                patch = layout.patches[name]
                s = patch.logical_x if demo_basis == "X" else patch.logical_z
                offs = layout.offsets()[name]
                positions = [offs + i for i, c in enumerate(s) if c == demo_basis]
                demo_idx = _mpp_from_positions(circuit, positions, demo_basis)
                demo_info[name] = {"basis": demo_basis, "index": demo_idx}

        metadata: Dict[str, object] = {
            "merge_windows": merge_windows,
            "observable_basis": tuple(basis_labels),
            "demo": demo_info,
            "cnot_operations": cnot_operations,
        }

        return circuit, observable_pairs, metadata


