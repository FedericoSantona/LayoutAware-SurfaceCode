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
import pytest
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
    build_cnot_surgery_circuit_physics,
)
from simulation import MonteCarloConfig, run_circuit_physics


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

    # Track all four logical operators PLUS X_INT and Z_INT to see the full picture
    tracked = {
        "X_C": patch_logicals["C"]["X"],
        "Z_C": patch_logicals["C"]["Z"],
        "X_INT": patch_logicals["INT"]["X"],
        "Z_INT": patch_logicals["INT"]["Z"],
        "X_T": patch_logicals["T"]["X"],
        "Z_T": patch_logicals["T"]["Z"],
    }
    # Debug: Print initial logicals
    print("\n=== Initial Logical Operators ===")
    for name, op in tracked.items():
        print(f"{name}: {op}")
    
    # Debug: Print key measurements (smooth and rough merge phases)
    print("\n=== Key Measurements (smooth and rough merge phases) ===")
    smooth_merge_measurements = []
    rough_merge_measurements = []
    for idx in sorted(builder._meas_meta.keys()):
        meta = builder._meas_meta[idx]
        phase = meta.get('phase', 'unknown')
        pauli = meta.get("pauli")
        if pauli and ('smooth merge' in phase or 'rough merge' in phase):
            if 'smooth merge' in phase:
                smooth_merge_measurements.append((idx, pauli))
            if 'rough merge' in phase:
                rough_merge_measurements.append((idx, pauli))
    
    print(f"\nSmooth merge measurements (first 10):")
    for idx, pauli in smooth_merge_measurements[:10]:
        print(f"  {idx}: {pauli[:60]}...")
    
    print(f"\nRough merge measurements (first 10):")
    for idx, pauli in rough_merge_measurements[:10]:
        print(f"  {idx}: {pauli[:60]}...")
    
    # Check if we can find the logical parity stabilizers
    print("\n=== Looking for Logical Parity Stabilizers ===")
    z_c = patch_logicals["C"]["Z"]
    z_int = patch_logicals["INT"]["Z"]
    z_t = patch_logicals["T"]["Z"]
    x_int = patch_logicals["INT"]["X"]
    x_t = patch_logicals["T"]["X"]
    
    z_c_z_int = _multiply_paulis_disjoint(z_c, z_int)
    x_int_x_t = _multiply_paulis_disjoint(x_int, x_t)
    
    print(f"Expected Z_C * Z_INT: {z_c_z_int[:60]}...")
    print(f"Expected X_INT * X_T: {x_int_x_t[:60]}...")
    
    # Check if these are in the measurements
    found_z_parity = False
    found_x_parity = False
    for idx in sorted(builder._meas_meta.keys()):
        meta = builder._meas_meta[idx]
        pauli = meta.get("pauli")
        if pauli:
            if pauli == z_c_z_int:
                print(f"  FOUND Z_C * Z_INT at measurement {idx} (phase: {meta.get('phase', 'unknown')})")
                found_z_parity = True
            if pauli == x_int_x_t:
                print(f"  FOUND X_INT * X_T at measurement {idx} (phase: {meta.get('phase', 'unknown')})")
                found_x_parity = True
    
    if not found_z_parity:
        print("  WARNING: Z_C * Z_INT NOT FOUND in measurements!")
    if not found_x_parity:
        print("  WARNING: X_INT * X_T NOT FOUND in measurements!")
    
    final_ops, deps = _propagate_logicals_through_measurements(
        logicals=tracked, meas_meta=builder._meas_meta
    )

    # Debug: Print dependencies
    print("\n=== Measurement Dependencies ===")
    for name, dep_list in deps.items():
        print(f"{name} depends on measurements: {dep_list}")
        # Print the actual Pauli strings of these measurements
        for idx in dep_list[:5]:  # Show first 5
            meta = builder._meas_meta.get(idx)
            if meta:
                print(f"  -> Measurement {idx}: {meta.get('pauli', 'N/A')[:50]}... (phase: {meta.get('phase', 'unknown')})")

    # Debug: Print final operators before canonicalization
    print("\n=== Final Operators (after propagation, before canonicalization) ===")
    for name, op in final_ops.items():
        print(f"{name}: {op}")

    # Expected transformations for CNOT
    expected_xc = _multiply_paulis_disjoint(
        patch_logicals["C"]["X"], patch_logicals["T"]["X"]
    )
    expected_zc = patch_logicals["C"]["Z"]  # Should remain unchanged
    expected_xt = patch_logicals["T"]["X"]   # Should remain unchanged
    expected_zt = _multiply_paulis_disjoint(
        patch_logicals["C"]["Z"], patch_logicals["T"]["Z"]
    )

    # Debug: Print intermediate relationships (note: may have Y operators due to correlations)
    print("\n=== Checking Intermediate Relationships ===")
    x_c_final = final_ops["X_C"]
    x_int_final = final_ops["X_INT"]
    x_t_final = final_ops["X_T"]
    print(f"X_C final: {x_c_final[:60]}...")
    print(f"X_INT final: {x_int_final[:60]}...")
    print(f"X_T final: {x_t_final[:60]}...")
    print(f"Expected X_C * X_T: {expected_xc[:60]}...")
    # Note: X_C and X_INT may have Y operators now due to correlation tracking,
    # so we can't use _multiply_paulis_disjoint which requires disjoint support

    # Debug: Print expected operators before canonicalization
    print("\n=== Expected Operators (before canonicalization) ===")
    print(f"X_C expected: {expected_xc}")
    print(f"Z_C expected: {expected_zc}")
    print(f"X_T expected: {expected_xt}")
    print(f"Z_T expected: {expected_zt}")
    
    # Debug: Print stabilizer basis info
    print(f"\n=== Stabilizer Basis Info ===")
    print(f"Number of stabilizers in basis: {len(stab_basis)}")
    print(f"Total qubits: {layout.n_total}")
    print(f"Expected logical qubits (k): {layout.n_total - len(stab_basis)}")
    
    # Canonicalize all operators for comparison
    final_xc = _canonicalize_logical(final_ops["X_C"], stab_basis)
    final_zc = _canonicalize_logical(final_ops["Z_C"], stab_basis)
    final_xt = _canonicalize_logical(final_ops["X_T"], stab_basis)
    final_zt = _canonicalize_logical(final_ops["Z_T"], stab_basis)
    
    exp_xc = _canonicalize_logical(expected_xc, stab_basis)
    exp_zc = _canonicalize_logical(expected_zc, stab_basis)
    exp_xt = _canonicalize_logical(expected_xt, stab_basis)
    exp_zt = _canonicalize_logical(expected_zt, stab_basis)

    # Debug: Print canonicalized operators
    print("\n=== Canonicalized Operators ===")
    print(f"Final X_C: {final_xc}")
    print(f"Expected X_C: {exp_xc}")
    print(f"Final Z_C: {final_zc}")
    print(f"Expected Z_C: {exp_zc}")
    print(f"Final X_T: {final_xt}")
    print(f"Expected X_T: {exp_xt}")
    print(f"Final Z_T: {final_zt}")
    print(f"Expected Z_T: {exp_zt}")

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


@pytest.mark.xfail(reason="Bell-frame tracking still under investigation", strict=False)
def test_cnot_bell_correlators_noise_free():
    """In physics mode with zero noise, Bell correlators should be near ±1."""
    circuit, correlator_map = build_cnot_surgery_circuit_physics(
        distance=3,
        code_type="standard",
        p_x=0,
        p_z=0,
        rounds_pre=1,
        rounds_merge=1,
        rounds_post=1,
        verbose=False,
    )

    result = run_circuit_physics(
        circuit=circuit,
        correlator_map=correlator_map,
        mc_config=MonteCarloConfig(shots=512, seed=123),
        keep_samples=False,
        verbose=False,
    )

    check.greater(
        abs(result.correlators.get("XX", 0.0)),
        0.9,
        f"Expected |<XX>| ~ 1, got {result.correlators.get('XX')}",
    )
    check.greater(
        abs(result.correlators.get("ZZ", 0.0)),
        0.9,
        f"Expected |<ZZ>| ~ 1, got {result.correlators.get('ZZ')}",
    )
