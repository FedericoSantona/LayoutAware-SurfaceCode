"""Validate the logical Clifford map of the lattice-surgery CNOT."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import stim
import pytest_check as check

from surface_code import (
    Layout,
    LatticeSurgery,
    PhenomenologicalStimBuilder,
    PhenomenologicalStimConfig,
    SeamSpec,
    stabs_to_symplectic,
)
from scripts.surgery_experiment import (
    _build_stab_basis_from_symplectic,
    _canonicalize_logical,
    _propagate_logicals_through_measurements,
    _multiply_paulis_disjoint,
)


def _post_stabilizer_basis(phases):
    post_phase = next(
        ph
        for ph in phases
        if ph.name.endswith("post-merge") or ph.name == "post-merge"
    )
    S = stabs_to_symplectic(post_phase.z_stabilizers, post_phase.x_stabilizers)
    return _build_stab_basis_from_symplectic(S)


def test_cnot_logical_clifford_map_distance_3_standard():
    """Test all four CNOT logical operator transformations:
    X_C -> X_C X_T, Z_C -> Z_C, X_T -> X_T, Z_T -> Z_C Z_T
    """
    layout = Layout(
        distance=3,
        code_type="standard",
        patch_order=["C", "INT", "T"],
        seams=[
            SeamSpec("C", "INT", "smooth"),
            SeamSpec("INT", "T", "rough"),
        ],
        patch_metadata={"C": "control", "INT": "ancilla", "T": "target"},
    )

    surgery = LatticeSurgery(layout)
    spec = surgery.cnot(
        control="C",
        ancilla="INT",
        target="T",
        rounds_pre=1,
        rounds_merge=1,
        rounds_post=1,
    )

    phases = spec.phases
    patch_logicals = spec.patch_logicals
    stab_basis = _post_stabilizer_basis(phases)

    class CombinedCode:
        def __init__(self, n: int):
            self.n = n

    builder = PhenomenologicalStimBuilder(
        code=CombinedCode(layout.n_total),
        z_stabilizers=[],
        x_stabilizers=[],
        logical_z=None,
        logical_x=None,
    )

    circuit = stim.Circuit()
    builder.run_phases(
        circuit=circuit,
        phases=phases,
        config=PhenomenologicalStimConfig(
            rounds=1, p_x_error=0.0, p_z_error=0.0, init_label=None
        ),
    )

    # Track all four logical operators
    tracked = {
        "X_C": patch_logicals["C"]["X"],
        "Z_C": patch_logicals["C"]["Z"],
        "X_T": patch_logicals["T"]["X"],
        "Z_T": patch_logicals["T"]["Z"],
    }
    final_ops, _ = _propagate_logicals_through_measurements(
        logicals=tracked, meas_meta=builder._meas_meta
    )

    # Expected transformations for CNOT
    expected_xc = _multiply_paulis_disjoint(
        patch_logicals["C"]["X"], patch_logicals["T"]["X"]
    )
    expected_zc = patch_logicals["C"]["Z"]  # Should remain unchanged
    expected_xt = patch_logicals["T"]["X"]   # Should remain unchanged
    expected_zt = _multiply_paulis_disjoint(
        patch_logicals["C"]["Z"], patch_logicals["T"]["Z"]
    )

    # Canonicalize all operators for comparison
    final_xc = _canonicalize_logical(final_ops["X_C"], stab_basis)
    final_zc = _canonicalize_logical(final_ops["Z_C"], stab_basis)
    final_xt = _canonicalize_logical(final_ops["X_T"], stab_basis)
    final_zt = _canonicalize_logical(final_ops["Z_T"], stab_basis)
    
    exp_xc = _canonicalize_logical(expected_xc, stab_basis)
    exp_zc = _canonicalize_logical(expected_zc, stab_basis)
    exp_xt = _canonicalize_logical(expected_xt, stab_basis)
    exp_zt = _canonicalize_logical(expected_zt, stab_basis)

    # Assert all four transformations (using pytest-check to check all even if some fail)
    # Each check will be evaluated and reported, even if others fail
    print(f"\nChecking X_C transformation: {final_xc} == {exp_xc}")
    check.equal(
        final_xc, exp_xc,
        f"X_C transformation failed: expected {exp_xc}, got {final_xc}"
    )
    
    print(f"Checking Z_C unchanged: {final_zc} == {exp_zc}")
    check.equal(
        final_zc, exp_zc,
        f"Z_C should be unchanged: expected {exp_zc}, got {final_zc}"
    )
    
    print(f"Checking X_T unchanged: {final_xt} == {exp_xt}")
    check.equal(
        final_xt, exp_xt,
        f"X_T should be unchanged: expected {exp_xt}, got {final_xt}"
    )
    
    print(f"Checking Z_T transformation: {final_zt} == {exp_zt}")
    check.equal(
        final_zt, exp_zt,
        f"Z_T transformation failed: expected {exp_zt}, got {final_zt}"
    )
