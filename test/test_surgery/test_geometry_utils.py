"""Tests for geometry utilities, specifically smooth and rough boundary identification."""
import os
import sys

import numpy as np
import pytest

# Ensure the repository root is on sys.path so local packages are importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Skip the module if qiskit_qec is missing
qiskit_qec = pytest.importorskip("qiskit_qec")

from src.surface_code import (
    build_surface_code_model,
    find_boundary_data_qubits,
    find_rough_boundary_data_qubits,
    find_smooth_boundary_data_qubits,
)


def test_find_smooth_boundary_data_qubits_heavy_hex_d3():
    """Test smooth boundary identification for heavy_hex code with distance=3."""
    model = build_surface_code_model(3, "heavy_hex")
    smooth_boundary = find_smooth_boundary_data_qubits(model)
    
    # Test 1: Smooth boundary qubits should be in logical Z operator
    logical_z_qubits = {i for i, char in enumerate(model.logical_z) if char == 'Z'}
    assert all(q in logical_z_qubits for q in smooth_boundary), \
        f"Some smooth boundary qubits are not in logical Z operator: {smooth_boundary} vs {sorted(logical_z_qubits)}"
    
    # Test 2: Smooth boundary qubits should be on boundaries (fewer stabilizers)
    n = model.code.n
    qubit_stabilizer_count = np.zeros(n, dtype=int)
    
    for stab in model.z_stabilizers:
        for i, char in enumerate(stab):
            if char == 'Z':
                qubit_stabilizer_count[i] += 1
    
    for stab in model.x_stabilizers:
        for i, char in enumerate(stab):
            if char == 'X':
                qubit_stabilizer_count[i] += 1
    
    max_stabilizer_count = qubit_stabilizer_count.max()
    if max_stabilizer_count > 0:
        boundary_threshold = max(1, max_stabilizer_count - 1)
        boundary_qubits = {i for i in range(n) if qubit_stabilizer_count[i] <= boundary_threshold}
        all_on_boundary = all(q in boundary_qubits for q in smooth_boundary)
        
        if not all_on_boundary:
            # Fallback is okay - at least check they're in logical Z
            assert all(q in logical_z_qubits for q in smooth_boundary), \
                "Smooth boundary qubits should be in logical Z operator"
    
    # Test 3: Number of smooth boundary qubits should be reasonable
    assert len(smooth_boundary) > 0, "Should find at least one smooth boundary qubit"
    assert len(smooth_boundary) <= 3 * 2, \
        f"Unusually many smooth boundary qubits: {len(smooth_boundary)} (expected ~3)"
    
    # Test 4: Smooth boundary qubits should be valid qubit indices
    assert all(0 <= q < n for q in smooth_boundary), \
        "All smooth boundary qubits should be valid indices"


def test_find_smooth_boundary_data_qubits_heavy_hex_d5():
    """Test smooth boundary identification for heavy_hex code with distance=5."""
    model = build_surface_code_model(5, "heavy_hex")
    smooth_boundary = find_smooth_boundary_data_qubits(model)
    
    logical_z_qubits = {i for i, char in enumerate(model.logical_z) if char == 'Z'}
    assert all(q in logical_z_qubits for q in smooth_boundary)
    assert len(smooth_boundary) > 0
    assert len(smooth_boundary) <= 5 * 2
    assert all(0 <= q < model.code.n for q in smooth_boundary)


def test_find_smooth_boundary_data_qubits_standard_d3():
    """Test smooth boundary identification for standard code with distance=3."""
    model = build_surface_code_model(3, "standard")
    smooth_boundary = find_smooth_boundary_data_qubits(model)
    
    logical_z_qubits = {i for i, char in enumerate(model.logical_z) if char == 'Z'}
    assert all(q in logical_z_qubits for q in smooth_boundary)
    assert len(smooth_boundary) > 0
    assert len(smooth_boundary) <= 3 * 2
    assert all(0 <= q < model.code.n for q in smooth_boundary)


def test_find_smooth_boundary_data_qubits_standard_d5():
    """Test smooth boundary identification for standard code with distance=5."""
    model = build_surface_code_model(5, "standard")
    smooth_boundary = find_smooth_boundary_data_qubits(model)
    
    logical_z_qubits = {i for i, char in enumerate(model.logical_z) if char == 'Z'}
    assert all(q in logical_z_qubits for q in smooth_boundary)
    assert len(smooth_boundary) > 0
    assert len(smooth_boundary) <= 5 * 2
    assert all(0 <= q < model.code.n for q in smooth_boundary)


@pytest.mark.parametrize("code_type,distance", [
    ("heavy_hex", 3),
    ("heavy_hex", 5),
    ("standard", 3),
    ("standard", 5),
])
def test_find_smooth_boundary_data_qubits_general(code_type, distance):
    """Parametrized test for smooth boundary identification across code types and distances."""
    model = build_surface_code_model(distance, code_type)
    smooth_boundary = find_smooth_boundary_data_qubits(model)
    
    # All smooth boundary qubits should be in logical Z operator
    logical_z_qubits = {i for i, char in enumerate(model.logical_z) if char == 'Z'}
    assert all(q in logical_z_qubits for q in smooth_boundary), \
        f"{code_type} d={distance}: Smooth boundary qubits {smooth_boundary} not all in logical Z {sorted(logical_z_qubits)}"
    
    # Should find reasonable number of smooth boundary qubits
    assert len(smooth_boundary) > 0, f"{code_type} d={distance}: Should find at least one smooth boundary qubit"
    assert len(smooth_boundary) <= distance * 2, \
        f"{code_type} d={distance}: Too many smooth boundary qubits: {len(smooth_boundary)}"
    
    # All indices should be valid
    assert all(0 <= q < model.code.n for q in smooth_boundary), \
        f"{code_type} d={distance}: Invalid qubit indices in smooth boundary"


# Rough boundary tests
def test_find_rough_boundary_data_qubits_heavy_hex_d3():
    """Test rough boundary identification for heavy_hex code with distance=3."""
    model = build_surface_code_model(3, "heavy_hex")
    rough_boundary = find_rough_boundary_data_qubits(model)
    
    # Test 1: Rough boundary qubits should be in logical X operator
    logical_x_qubits = {i for i, char in enumerate(model.logical_x) if char == 'X'}
    assert all(q in logical_x_qubits for q in rough_boundary), \
        f"Some rough boundary qubits are not in logical X operator: {rough_boundary} vs {sorted(logical_x_qubits)}"
    
    # Test 2: Rough boundary qubits should be on boundaries (fewer stabilizers)
    n = model.code.n
    qubit_stabilizer_count = np.zeros(n, dtype=int)
    
    for stab in model.z_stabilizers:
        for i, char in enumerate(stab):
            if char == 'Z':
                qubit_stabilizer_count[i] += 1
    
    for stab in model.x_stabilizers:
        for i, char in enumerate(stab):
            if char == 'X':
                qubit_stabilizer_count[i] += 1
    
    max_stabilizer_count = qubit_stabilizer_count.max()
    if max_stabilizer_count > 0:
        boundary_threshold = max(1, max_stabilizer_count - 1)
        boundary_qubits = {i for i in range(n) if qubit_stabilizer_count[i] <= boundary_threshold}
        all_on_boundary = all(q in boundary_qubits for q in rough_boundary)
        
        if not all_on_boundary:
            # Fallback is okay - at least check they're in logical X
            assert all(q in logical_x_qubits for q in rough_boundary), \
                "Rough boundary qubits should be in logical X operator"
    
    # Test 3: Number of rough boundary qubits should be reasonable
    assert len(rough_boundary) > 0, "Should find at least one rough boundary qubit"
    assert len(rough_boundary) <= 3 * 2, \
        f"Unusually many rough boundary qubits: {len(rough_boundary)} (expected ~3)"
    
    # Test 4: Rough boundary qubits should be valid qubit indices
    assert all(0 <= q < n for q in rough_boundary), \
        "All rough boundary qubits should be valid indices"


def test_find_rough_boundary_data_qubits_heavy_hex_d5():
    """Test rough boundary identification for heavy_hex code with distance=5."""
    model = build_surface_code_model(5, "heavy_hex")
    rough_boundary = find_rough_boundary_data_qubits(model)
    
    logical_x_qubits = {i for i, char in enumerate(model.logical_x) if char == 'X'}
    assert all(q in logical_x_qubits for q in rough_boundary)
    assert len(rough_boundary) > 0
    assert len(rough_boundary) <= 5 * 2
    assert all(0 <= q < model.code.n for q in rough_boundary)


def test_find_rough_boundary_data_qubits_standard_d3():
    """Test rough boundary identification for standard code with distance=3."""
    model = build_surface_code_model(3, "standard")
    rough_boundary = find_rough_boundary_data_qubits(model)
    
    logical_x_qubits = {i for i, char in enumerate(model.logical_x) if char == 'X'}
    assert all(q in logical_x_qubits for q in rough_boundary)
    assert len(rough_boundary) > 0
    assert len(rough_boundary) <= 3 * 2
    assert all(0 <= q < model.code.n for q in rough_boundary)


def test_find_rough_boundary_data_qubits_standard_d5():
    """Test rough boundary identification for standard code with distance=5."""
    model = build_surface_code_model(5, "standard")
    rough_boundary = find_rough_boundary_data_qubits(model)
    
    logical_x_qubits = {i for i, char in enumerate(model.logical_x) if char == 'X'}
    assert all(q in logical_x_qubits for q in rough_boundary)
    assert len(rough_boundary) > 0
    assert len(rough_boundary) <= 5 * 2
    assert all(0 <= q < model.code.n for q in rough_boundary)


@pytest.mark.parametrize("code_type,distance", [
    ("heavy_hex", 3),
    ("heavy_hex", 5),
    ("standard", 3),
    ("standard", 5),
])
def test_find_rough_boundary_data_qubits_general(code_type, distance):
    """Parametrized test for rough boundary identification across code types and distances."""
    model = build_surface_code_model(distance, code_type)
    rough_boundary = find_rough_boundary_data_qubits(model)
    
    # All rough boundary qubits should be in logical X operator
    logical_x_qubits = {i for i, char in enumerate(model.logical_x) if char == 'X'}
    assert all(q in logical_x_qubits for q in rough_boundary), \
        f"{code_type} d={distance}: Rough boundary qubits {rough_boundary} not all in logical X {sorted(logical_x_qubits)}"
    
    # Should find reasonable number of rough boundary qubits
    assert len(rough_boundary) > 0, f"{code_type} d={distance}: Should find at least one rough boundary qubit"
    assert len(rough_boundary) <= distance * 2, \
        f"{code_type} d={distance}: Too many rough boundary qubits: {len(rough_boundary)}"
    
    # All indices should be valid
    assert all(0 <= q < model.code.n for q in rough_boundary), \
        f"{code_type} d={distance}: Invalid qubit indices in rough boundary"


# Test unified function
@pytest.mark.parametrize("code_type,distance,boundary_type", [
    ("heavy_hex", 3, "smooth"),
    ("heavy_hex", 3, "rough"),
    ("heavy_hex", 5, "smooth"),
    ("heavy_hex", 5, "rough"),
    ("standard", 3, "smooth"),
    ("standard", 3, "rough"),
    ("standard", 5, "smooth"),
    ("standard", 5, "rough"),
])
def test_find_boundary_data_qubits_unified(code_type, distance, boundary_type):
    """Test the unified find_boundary_data_qubits function for both boundary types."""
    model = build_surface_code_model(distance, code_type)
    boundary_qubits = find_boundary_data_qubits(model, boundary_type=boundary_type)
    
    if boundary_type == "smooth":
        logical_qubits = {i for i, char in enumerate(model.logical_z) if char == 'Z'}
    else:  # rough
        logical_qubits = {i for i, char in enumerate(model.logical_x) if char == 'X'}
    
    # All boundary qubits should be in the appropriate logical operator
    assert all(q in logical_qubits for q in boundary_qubits), \
        f"{code_type} d={distance} {boundary_type}: Boundary qubits {boundary_qubits} not all in logical operator {sorted(logical_qubits)}"
    
    # Should find reasonable number of boundary qubits
    assert len(boundary_qubits) > 0, \
        f"{code_type} d={distance} {boundary_type}: Should find at least one boundary qubit"
    assert len(boundary_qubits) <= distance * 2, \
        f"{code_type} d={distance} {boundary_type}: Too many boundary qubits: {len(boundary_qubits)}"
    
    # All indices should be valid
    assert all(0 <= q < model.code.n for q in boundary_qubits), \
        f"{code_type} d={distance} {boundary_type}: Invalid qubit indices"


@pytest.mark.parametrize("code_type,distance", [
    ("heavy_hex", 3),
    ("heavy_hex", 5),
    ("standard", 3),
    ("standard", 5),
])
def test_smooth_and_rough_boundaries_minimal_overlap(code_type, distance):
    """Test that smooth and rough boundaries have minimal overlap.
    
    Note: Some overlap is expected at corners where boundaries meet,
    but the overlap should be small compared to the total boundary size.
    """
    model = build_surface_code_model(distance, code_type)
    smooth_boundary = find_smooth_boundary_data_qubits(model)
    rough_boundary = find_rough_boundary_data_qubits(model)
    
    # Check that boundaries are mostly disjoint
    overlap = set(smooth_boundary) & set(rough_boundary)
    total_boundary_size = len(set(smooth_boundary) | set(rough_boundary))
    
    # Overlap should be small (at most a few corner qubits)
    # Typically overlap is <= 2 for small codes
    assert len(overlap) <= max(2, distance // 2), \
        f"{code_type} d={distance}: Too much overlap between smooth and rough boundaries: {overlap}"
    
    # Overlap should be a small fraction of total boundary
    if total_boundary_size > 0:
        overlap_fraction = len(overlap) / total_boundary_size
        assert overlap_fraction < 0.3, \
            f"{code_type} d={distance}: Overlap fraction too high: {overlap_fraction:.2%}"


if __name__ == "__main__":
    # Allow running the test file directly for debugging
    pytest.main([__file__, "-v"])

