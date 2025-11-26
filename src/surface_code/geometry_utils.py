"""Geometry utilities for surface code boundary identification."""
from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np

from .model import SurfaceCodeModel


def _get_qubit_coordinates(model: SurfaceCodeModel) -> Optional[np.ndarray]:
    """Extract qubit coordinates from the code object.
    
    Accesses coordinates via code.shell.vertices, where each vertex has a pos
    attribute containing [x, y] coordinates. Maps vertices to qubit indices
    using code.qubit_data.index[vertex.id] (PauliList index).
    
    Args:
        model: Surface code model (HeavyHexModel or StandardSurfaceCodeModel)
        
    Returns:
        Array of shape (n, 2) with [x, y] coordinates for each qubit, or None
        if coordinates cannot be extracted (e.g., code doesn't have shell/qubit_data)
    """
    code = model.code
    
    # Check if code has shell and qubit_data attributes
    if not hasattr(code, 'shell') or code.shell is None:
        raise ValueError("Code object does not have a shell attribute")
    if not hasattr(code, 'qubit_data') or code.qubit_data is None:
        raise ValueError("Code object does not have a qubit_data attribute")
    
    n = code.n
    coords = np.full((n, 2), np.nan, dtype=float)
    
    # Iterate through vertices and map to qubit indices
    # Use qubit_data.index which maps to PauliList index (qubit index in our code)
    vertices_mapped = 0
    for vertex in code.shell.vertices:
        if vertex.id in code.qubit_data.index:
            qubit_idx = code.qubit_data.index[vertex.id]
            if 0 <= qubit_idx < n:
                coords[qubit_idx] = np.array(vertex.pos, dtype=float)
                vertices_mapped += 1
    
    # Check if we successfully extracted coordinates for at least some qubits
    # Not all qubits may be represented as vertices (e.g., ancilla qubits)
    if vertices_mapped == 0:
        raise ValueError("No qubit coordinates could be mapped from vertices")
    
    # Check if we have coordinates for a reasonable fraction of qubits
    # For surface codes, most data qubits should be in the shell
    coords_found = (~np.isnan(coords)).sum()
    if coords_found < n // 2:
        raise ValueError(
            f"Too few qubit coordinates found: {coords_found}/{n}. "
            "Expected at least half of qubits to have coordinates."
        )
    
    return coords



def find_boundary_data_qubits(
    model: SurfaceCodeModel,
    boundary_type: Literal["smooth", "rough"] = "smooth"
) -> List[int]:
    """Return the list of data qubit indices that lie on a specified boundary type.
    
    Uses geometry-based boundary detection by:
    1. Extracting qubit coordinates from the code object
    2. Identifying the four outermost sides (left, right, top, bottom) based on coordinate bounds
    3. Using stabilizer deficits (missing X or Z stabilizers) to determine which sides are smooth vs rough
    4. Returning only qubits on the true outermost boundaries
    
    In surface codes:
    - Smooth boundaries are where Z stabilizers terminate (sides with Z stabilizer deficit)
    - Rough boundaries are where X stabilizers terminate (sides with X stabilizer deficit)
    
    This geometry-based approach avoids mislabeling interior logical strings as boundaries,
    which can occur when using logical operators alone.
    
    Args:
        model: Surface code model (HeavyHexModel or StandardSurfaceCodeModel)
        boundary_type: Type of boundary to find - "smooth" or "rough"
        
    Returns:
        List of data qubit indices on the specified boundary type
    """
    n = model.code.n
    
    # --- 1) Get coordinates for each data qubit ---
    coords = _get_qubit_coordinates(model)
    if coords is None:
        # Fallback: if coordinates cannot be extracted, raise an error
        raise ValueError("Qubit coordinates cannot be extracted from the code object")
    
    # Filter to only qubits with valid coordinates
    valid_mask = ~np.isnan(coords).any(axis=1)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        # Fallback: if no qubits with valid coordinates found, raise an error
        raise ValueError("No qubits with valid coordinates found")
    
    xs = coords[valid_mask, 0]
    ys = coords[valid_mask, 1]
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    # --- 2) Count X and Z stabilizers per qubit separately ---
    num_x = np.zeros(n, dtype=int)
    num_z = np.zeros(n, dtype=int)
    
    for stab in model.x_stabilizers:
        for i, char in enumerate(stab):
            if char == "X":
                num_x[i] += 1
    
    for stab in model.z_stabilizers:
        for i, char in enumerate(stab):
            if char == "Z":
                num_z[i] += 1
    
    max_x = num_x.max() if num_x.size > 0 else 0
    max_z = num_z.max() if num_z.size > 0 else 0
    
    # --- 3) Define the four geometric sides of the patch ---
    # Use a small tolerance in case coords are floats
    tol = 1e-6
    
    # Map side indices back to global qubit indices
    sides: dict[str, np.ndarray] = {
        "left":   valid_indices[np.where(np.abs(xs - x_min) < tol)[0]],
        "right":  valid_indices[np.where(np.abs(xs - x_max) < tol)[0]],
        "bottom": valid_indices[np.where(np.abs(ys - y_min) < tol)[0]],
        "top":    valid_indices[np.where(np.abs(ys - y_max) < tol)[0]],
    }
    
    # --- 4) For each side, measure how many X/Z stabilizers are "missing" ---
    side_deficits = {}
    for name, idxs in sides.items():
        if len(idxs) == 0:
            continue
        mean_x = num_x[idxs].mean() if len(idxs) > 0 else 0
        mean_z = num_z[idxs].mean() if len(idxs) > 0 else 0
        side_deficits[name] = {
            "def_x": max_x - mean_x,
            "def_z": max_z - mean_z,
        }
    
    # --- 5) Decide which side is smooth/rough ---
    if boundary_type == "smooth":
        # Smooth boundaries: Z stabilizers terminate -> big deficit in Z
        sorted_sides = sorted(
            side_deficits.items(),
            key=lambda kv: kv[1]["def_z"],
            reverse=True,
        )
    else:  # rough
        # Rough boundaries: X stabilizers terminate -> big deficit in X
        sorted_sides = sorted(
            side_deficits.items(),
            key=lambda kv: kv[1]["def_x"],
            reverse=True,
        )
    
    # Pick only the single best-matching side (one boundary = one side)
    if len(sorted_sides) < 1:
        # Fallback: if we can't find any sides, raise an error
        raise ValueError("Cannot find any sides for boundary detection")
    
    chosen_side_name = sorted_sides[0][0]
    
    # --- 6) Collect all qubits on that side ---
    boundary_qubits = sides[chosen_side_name].tolist()
    boundary_qubits = sorted(set(boundary_qubits))
    
    return boundary_qubits


def find_smooth_boundary_data_qubits(model: SurfaceCodeModel) -> List[int]:
    """Return the list of data qubit indices that lie on a smooth boundary.
    
    Convenience wrapper around find_boundary_data_qubits(model, boundary_type="smooth").
    
    Uses geometry-based boundary detection to identify qubits on the outermost
    edges where Z stabilizers terminate (sides with Z stabilizer deficit).
    
    Args:
        model: Surface code model (HeavyHexModel or StandardSurfaceCodeModel)
        
    Returns:
        List of data qubit indices on smooth boundaries
    """
    return find_boundary_data_qubits(model, boundary_type="smooth")


def find_rough_boundary_data_qubits(model: SurfaceCodeModel) -> List[int]:
    """Return the list of data qubit indices that lie on a rough boundary.
    
    Convenience wrapper around find_boundary_data_qubits(model, boundary_type="rough").
    
    Uses geometry-based boundary detection to identify qubits on the outermost
    edges where X stabilizers terminate (sides with X stabilizer deficit).
    
    Args:
        model: Surface code model (HeavyHexModel or StandardSurfaceCodeModel)
        
    Returns:
        List of data qubit indices on rough boundaries
    """
    return find_boundary_data_qubits(model, boundary_type="rough")

