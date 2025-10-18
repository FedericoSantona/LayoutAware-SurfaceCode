"""Utility functions for Stim circuit builders."""
from __future__ import annotations

from typing import List, Optional, Sequence

import stim


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
    prev: Sequence[int],
    curr: Sequence[int],
    append_detector_cb,
) -> List[int]:
    """Add temporal detectors between consecutive rounds and return detector indices."""
    indices: List[int] = []
    for a, b in zip(prev, curr):
        idx = append_detector_cb([rec_from_abs(circuit, a), rec_from_abs(circuit, b)])
        indices.append(idx)
    return indices
