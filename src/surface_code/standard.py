"""Build and package standard surface-code data for simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qiskit_qec.codes.codebuilders.surface_code_builder import SurfaceCodeBuilder
from qiskit_qec.linear.symplectic import normalizer

from .linalg import rank_gf2
from .logicals import check_logicals, find_logicals_standard
from .model import SurfaceCodeModel
from .stabilizers import extract_css_stabilizers


def _native_plaquette_stabilizers(code) -> tuple[list[str], list[str]]:
    """Return the local plaquette stabilizers emitted by SurfaceCodeBuilder.

    SurfaceCodeBuilder's gauge generators already contain pure-Z and pure-X
    plaquette checks (weight 3–4) that keep each data qubit in at most two
    Z and two X measurements. Using these directly avoids the high-degree,
    high-weight CSS basis that ``extract_css_stabilizers`` can produce for
    the standard layout.
    """
    M = (code.generators.matrix.astype(np.uint8) & 1)
    n = code.n
    z_stabs: list[str] = []
    x_stabs: list[str] = []

    for row in M:
        z_block = row[:n]
        x_block = row[n:]
        if z_block.any() and not x_block.any():
            z_stabs.append(''.join('Z' if val else 'I' for val in z_block.tolist()))
        elif x_block.any() and not z_block.any():
            x_stabs.append(''.join('X' if val else 'I' for val in x_block.tolist()))

    return z_stabs, x_stabs


@dataclass
class StandardSurfaceCodeModel(SurfaceCodeModel):
    """Standard surface code model implementation."""
    
    def diagnostics(self) -> dict[str, int | bool]:
        diag = check_logicals(self.logical_z_vec, self.logical_x_vec, self.stabilizer_matrix)
        rank_m = rank_gf2(self.generators.matrix.astype(np.uint8))
        n = self.code.n
        s = self.stabilizer_matrix.shape[0]
        r = (rank_m - s) // 2
        diag.update({
            "n": n,
            "s": s,
            "r": r,
            "k": n - s - r,
        })
        return diag


def build_standard_surface_code_model(distance: int) -> StandardSurfaceCodeModel:
    """Build a standard surface code model for the given distance.
    
    Args:
        distance: Code distance (must be odd, typically 3, 5, 7, 9, ...)
        
    Returns:
        StandardSurfaceCodeModel instance with all code properties initialized
    """
    code = SurfaceCodeBuilder(d=distance).build()
    generators = code.generators
    M = generators.matrix.astype(np.uint8) & 1
    S_mat, _, _ = normalizer(M)
    z_stabs, x_stabs = _native_plaquette_stabilizers(code)
    if not z_stabs or not x_stabs:
        raise RuntimeError("No native plaquette stabilizers found for standard surface code.")
        #z_stabs, x_stabs = extract_css_stabilizers(S_mat)
    logical_z, logical_x, logical_z_vec, logical_x_vec = find_logicals_standard(code, S_mat, distance)
    return StandardSurfaceCodeModel(
        distance=distance,
        code=code,
        generators=generators,
        stabilizer_matrix=S_mat,
        z_stabilizers=z_stabs,
        x_stabilizers=x_stabs,
        logical_z=logical_z,
        logical_x=logical_x,
        logical_z_vec=logical_z_vec,
        logical_x_vec=logical_x_vec,
    )
