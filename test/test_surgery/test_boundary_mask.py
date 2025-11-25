"""Unit tests for the commuting boundary mask helper."""
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.surgery_experiment import _commuting_boundary_mask, _align_logical_x_to_masked_z
from surface_code import build_surface_code_model, find_rough_boundary_data_qubits


def _pairwise_commute(z_stabilizers: list[str], x_stabilizers: list[str]) -> bool:
    """Return True if all Z/X pairs commute (ignoring identities)."""
    for z in z_stabilizers:
        for x in x_stabilizers:
            overlap = sum(
                1 for a, b in zip(z, x) if a != "I" and b != "I" and a != b
            )
            if overlap % 2:
                return False
    return True


def test_commuting_boundary_mask_recovers_commutation_rough_boundary():
    """Stripping Z on a rough boundary should stay commuting after adjustment."""
    z_stabilizers = ["ZZII", "IIZZ"]
    x_stabilizers = ["XXII", "IIXX"]
    boundary = [1]

    assert _pairwise_commute(z_stabilizers, x_stabilizers)

    # Simple strip would break commutation (overlap becomes odd)
    z_simple_strip = ["IZII", "IIZZ"]
    assert not _pairwise_commute(z_simple_strip, x_stabilizers)

    z_masked, x_masked = _commuting_boundary_mask(
        z_stabilizers=z_stabilizers,
        x_stabilizers=x_stabilizers,
        boundary=boundary,
    )

    assert _pairwise_commute(z_masked, x_masked)
    assert all(stab[boundary[0]] == "I" for stab in z_masked)


def test_commuting_boundary_mask_recovers_commutation_smooth_boundary():
    """Stripping X on a smooth boundary should stay commuting after adjustment."""
    z_stabilizers = ["ZZII", "IIZZ"]
    x_stabilizers = ["XXII", "IIXX"]
    boundary = [0]

    assert _pairwise_commute(z_stabilizers, x_stabilizers)

    x_simple_strip = ["IXII", "IIXX"]
    assert not _pairwise_commute(z_stabilizers, x_simple_strip)

    smooth_z, smooth_x = _commuting_boundary_mask(
        z_stabilizers=z_stabilizers,
        x_stabilizers=x_stabilizers,
        boundary=boundary,
        strip_pauli="X",
    )

    assert _pairwise_commute(smooth_z, smooth_x)
    assert smooth_x[0][boundary[0]] == "I"
    # Adjustments should not touch the boundary for the opposite Pauli
    assert all(stab[boundary[0]] == original[boundary[0]] for stab, original in zip(smooth_z, z_stabilizers))


def _commutes(pauli: str, logical: str) -> bool:
    anti = 0
    for a, b in zip(pauli, logical):
        if a == "I" or b == "I":
            continue
        if a != b:
            anti ^= 1
    return anti == 0


@pytest.mark.parametrize("distance", [5, 7])
def test_rough_mask_commutes_with_logical_x_standard(distance: int):
    """Align logical-X via X stabilizers so masked Z checks commute (no Z stripping)."""
    model = build_surface_code_model(distance, "standard")
    rough_boundary = find_rough_boundary_data_qubits(model)

    rough_z_masked, _ = _commuting_boundary_mask(
        z_stabilizers=model.z_stabilizers,
        x_stabilizers=model.x_stabilizers,
        boundary=rough_boundary,
    )
    rough_z_before = list(rough_z_masked)
    aligned_logical_x = _align_logical_x_to_masked_z(
        model.logical_x, model.x_stabilizers, rough_z_masked
    )

    assert aligned_logical_x is not None
    assert all(_commutes(stab, aligned_logical_x) for stab in rough_z_masked)
    # Z stabilizers themselves should remain unchanged by the alignment step
    assert rough_z_masked == rough_z_before
