"""Helpers to build single-patch memory layouts for logical benchmarks."""
from __future__ import annotations

from typing import Dict, List, Tuple

from .layout import Layout, PatchObject
from .surgery_ops import MeasureRound


def _measure_flags(family: str | None) -> Tuple[bool, bool]:
    family_u = (family or "").strip().upper()
    if family_u == "Z":
        return True, False
    if family_u == "X":
        return False, True
    return True, True


def build_memory_layout(
    model,
    distance: int,
    *,
    rounds: int,
    bracket_basis: str,
    family: str | None = None,
) -> Tuple[Layout, List[object], Dict[str, str]]:
    """Create a single-patch layout for memory experiments.

    Args:
        model: Surface code model for the logical qubit.
        distance: Code distance (currently unused but kept for symmetry).
        rounds: Number of memory rounds to simulate.
        bracket_basis: Logical basis ('Z' or 'X') for decoding.
        family: Optional CSS family flag controlling which stabilizers to measure.
    """
    measure_z, measure_x = _measure_flags(family)

    data_patch = PatchObject.from_code_model(model, name="q0")
    layout = Layout()
    layout.add_patch("q0", data_patch)

    ops: List[object] = []
    # Warm-up round to seed detector references
    ops.append(MeasureRound(measure_z=measure_z, measure_x=measure_x))
    for _ in range(max(1, int(rounds))):
        ops.append(MeasureRound(measure_z=measure_z, measure_x=measure_x))

    basis = (bracket_basis or "Z").strip().upper()
    if basis not in {"X", "Z"}:
        raise ValueError(f"Unsupported bracket basis '{bracket_basis}'")
    bracket_map = {"q0": basis}

    return layout, ops, bracket_map
