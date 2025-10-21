"""Utility functions for Stim circuit builders."""
from __future__ import annotations

from itertools import zip_longest
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import stim

if TYPE_CHECKING:  # Import only for typing to avoid circular deps at runtime
    from .layout import Layout
    from .pauli import Pauli as PauliOperator


GateTarget = stim.GateTarget


def mpp_from_positions(circuit: stim.Circuit, positions: Sequence[int], pauli: str) -> Optional[int]:
    """Append an MPP for a tensor product of identical Paulis at given positions.

    Returns the absolute measurement index, or None if positions is empty.
    """
    if not positions:
        return None
    targets: List[GateTarget] = []
    first = True
    for q in positions:
        if not first:
            targets.append(stim.target_combiner())
        if pauli == "X":
            targets.append(stim.target_x(q))
        elif pauli == "Z":
            targets.append(stim.target_z(q))
        elif pauli == "Y":
            targets.append(stim.target_y(q))
        else:
            raise ValueError("Unsupported Pauli for MPP")
        first = False
    circuit.append_operation("MPP", targets)
    return circuit.num_measurements - 1


def rec_from_abs(circuit: stim.Circuit, idx: int) -> GateTarget:
    """Convert absolute measurement index to relative target."""
    return stim.target_rec(idx - circuit.num_measurements)


def add_temporal_detectors_with_index(
    circuit: stim.Circuit,
    prev: Sequence[Optional[int]],
    curr: Sequence[Optional[int]],
    append_detector_cb,
) -> List[Optional[int]]:
    """Add temporal detectors between consecutive rounds and return detector indices."""
    indices: List[Optional[int]] = []
    for a, b in zip_longest(prev, curr, fillvalue=None):
        if a is None or b is None or a == b:
            indices.append(None)
            continue
        idx = append_detector_cb([rec_from_abs(circuit, a), rec_from_abs(circuit, b)])
        indices.append(idx)
    return indices


# Helper: build physical MPP targets from a PauliOperator (aggregates X/Z->Y collisions).
def _mpp_targets_from_pauli(
    op: "PauliOperator",
    layout: "Layout",
    idx_to_name: Dict[int, str],
) -> Tuple[List[stim.GateTarget], Dict[str, List[str]]]:
    offs = layout.offsets()
    n_logical = op.n_qubits
    physical_targets: List[Tuple[int, str]] = []
    axes_map: Dict[str, List[str]] = {}
    for qi in range(n_logical):
        name_i = idx_to_name.get(qi)
        if name_i is None or name_i not in layout.patches:
            continue
        pch = op.get_axis(qi)
        if pch == "I":
            continue
        patch = layout.patches[name_i]
        if pch == "X":
            axes_map[name_i] = ["X"]
            positions, _ = layout.globalize_local_pauli_string(name_i, patch.logical_x)
            for pos in positions:
                physical_targets.append((pos, "X"))
        elif pch == "Z":
            axes_map[name_i] = ["Z"]
            positions, _ = layout.globalize_local_pauli_string(name_i, patch.logical_z)
            for pos in positions:
                physical_targets.append((pos, "Z"))
        else:  # Y → emit both and resolve below
            axes_map[name_i] = ["X", "Z"]
            x_positions, _ = layout.globalize_local_pauli_string(name_i, patch.logical_x)
            z_positions, _ = layout.globalize_local_pauli_string(name_i, patch.logical_z)
            # Handle Y case: if both X and Z are present at same position
            for pos in x_positions:
                if pos in z_positions:
                    physical_targets.append((pos, "Y"))
                else:
                    physical_targets.append((pos, "X"))
            for pos in z_positions:
                if pos not in x_positions:
                    physical_targets.append((pos, "Z"))
    # Resolve collisions X+Z -> Y
    qops: Dict[int, set] = {}
    for gidx, pch in physical_targets:
        qops.setdefault(gidx, set()).add(pch)
    final_targets: List[Tuple[int, str]] = []
    for gidx, opset in qops.items():
        if opset == {"X", "Z"}:
            final_targets.append((gidx, "Y"))
        else:
            final_targets.append((gidx, next(iter(opset))))

    mpp_targets: List[stim.GateTarget] = []
    for k, (gidx, pch) in enumerate(final_targets):
        if k > 0:
            mpp_targets.append(stim.target_combiner())
        if pch == "X":
            mpp_targets.append(stim.target_x(gidx))
        elif pch == "Z":
            mpp_targets.append(stim.target_z(gidx))
        else:
            mpp_targets.append(stim.target_y(gidx))
    return mpp_targets, axes_map
