"""Unit tests for the commuting boundary mask helper."""

from scripts.surgery_experiment import _commuting_boundary_mask


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
