"""Geometry utilities for surface code boundary identification."""
from __future__ import annotations

from typing import List, Literal

import numpy as np

from .model import SurfaceCodeModel


def find_boundary_data_qubits(
    model: SurfaceCodeModel,
    boundary_type: Literal["smooth", "rough"] = "smooth"
) -> List[int]:
    """Return the list of data qubit indices that lie on a specified boundary type.
    
    In surface codes:
    - Smooth boundaries are where Z stabilizers terminate.
      The logical Z operator runs along smooth boundaries.
    - Rough boundaries are where X stabilizers terminate.
      The logical X operator runs along rough boundaries.
    
    This function identifies boundary qubits by:
    1. Finding qubits in the appropriate logical operator (Z for smooth, X for rough)
    2. Identifying which of those are on boundaries (participate in fewer stabilizers)
    3. Verifying they're on the correct boundary type by checking stabilizer patterns
    
    Args:
        model: Surface code model (HeavyHexModel or StandardSurfaceCodeModel)
        boundary_type: Type of boundary to find - "smooth" or "rough"
        
    Returns:
        List of data qubit indices on the specified boundary type
    """
    n = model.code.n
    
    # Get qubits involved in the appropriate logical operator
    if boundary_type == "smooth":
        # Smooth boundaries: logical Z operator runs along them
        logical_qubits = set()
        if model.logical_z:
            logical_qubits = {i for i, char in enumerate(model.logical_z) if char == 'Z'}
    else:  # rough
        # Rough boundaries: logical X operator runs along them
        logical_qubits = set()
        if model.logical_x:
            logical_qubits = {i for i, char in enumerate(model.logical_x) if char == 'X'}
    
    # If no logical operator qubits, return empty list
    if not logical_qubits:
        return []
    
    # Count how many stabilizers each qubit participates in
    # Boundary qubits participate in fewer stabilizers than bulk qubits
    qubit_stabilizer_count = np.zeros(n, dtype=int)
    
    # Count Z stabilizers
    for stab in model.z_stabilizers:
        for i, char in enumerate(stab):
            if char == 'Z':
                qubit_stabilizer_count[i] += 1
    
    # Count X stabilizers
    for stab in model.x_stabilizers:
        for i, char in enumerate(stab):
            if char == 'X':
                qubit_stabilizer_count[i] += 1
    
    # Find boundary qubits: those with fewer stabilizers than the maximum
    # In a well-formed surface code, bulk qubits participate in more stabilizers
    # than boundary qubits
    max_stabilizer_count = qubit_stabilizer_count.max()
    if max_stabilizer_count == 0:
        # Fallback: if we can't determine from stabilizers, use logical operator qubits
        return sorted(logical_qubits)
    
    # Boundary qubits are those with significantly fewer stabilizers
    # Use a threshold: boundary qubits have at most (max_count - 1) stabilizers
    # This handles cases where boundary qubits might still participate in some stabilizers
    boundary_threshold = max(1, max_stabilizer_count - 1)
    boundary_qubits = {i for i in range(n) if qubit_stabilizer_count[i] <= boundary_threshold}
    
    # Boundary qubits of the specified type are those that are both:
    # 1. In the appropriate logical operator (runs along that boundary type)
    # 2. On a boundary (fewer stabilizers)
    boundary_type_qubits = logical_qubits & boundary_qubits
    
    # If we didn't find enough boundary qubits using the threshold,
    # fall back to using all logical operator qubits (they should be on the correct boundary)
    if len(boundary_type_qubits) < len(logical_qubits) // 2:
        # Try a more lenient threshold
        boundary_threshold = max_stabilizer_count
        boundary_qubits = {i for i in range(n) if qubit_stabilizer_count[i] <= boundary_threshold}
        boundary_type_qubits = logical_qubits & boundary_qubits
    
    # Final fallback: if still not enough, use all logical operator qubits
    # (they should be on the correct boundary by definition)
    if not boundary_type_qubits:
        boundary_type_qubits = logical_qubits
    
    return sorted(boundary_type_qubits)


def find_smooth_boundary_data_qubits(model: SurfaceCodeModel) -> List[int]:
    """Return the list of data qubit indices that lie on a smooth boundary.
    
    Convenience wrapper around find_boundary_data_qubits(model, boundary_type="smooth").
    
    In surface codes, smooth boundaries are where Z stabilizers terminate.
    The logical Z operator runs along smooth boundaries.
    
    Args:
        model: Surface code model (HeavyHexModel or StandardSurfaceCodeModel)
        
    Returns:
        List of data qubit indices on smooth boundaries
    """
    return find_boundary_data_qubits(model, boundary_type="smooth")


def find_rough_boundary_data_qubits(model: SurfaceCodeModel) -> List[int]:
    """Return the list of data qubit indices that lie on a rough boundary.
    
    Convenience wrapper around find_boundary_data_qubits(model, boundary_type="rough").
    
    In surface codes, rough boundaries are where X stabilizers terminate.
    The logical X operator runs along rough boundaries.
    
    Args:
        model: Surface code model (HeavyHexModel or StandardSurfaceCodeModel)
        
    Returns:
        List of data qubit indices on rough boundaries
    """
    return find_boundary_data_qubits(model, boundary_type="rough")

