"""Build and package heavy-hex surface-code data for simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qiskit_qec.codes.codebuilders.heavyhex_code_builder import HeavyHexCodeBuilder
from qiskit_qec.linear.symplectic import normalizer

from .linalg import rank_gf2
from .logicals import check_logicals, find_logicals_heavyhex
from .stabilizers import extract_css_stabilizers


@dataclass
class HeavyHexModel:
    distance: int
    code: Any
    generators: Any
    stabilizer_matrix: np.ndarray
    z_stabilizers: list[str]
    x_stabilizers: list[str]
    logical_z: str
    logical_x: str
    logical_z_vec: np.ndarray
    logical_x_vec: np.ndarray

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


def build_heavy_hex_model(distance: int) -> HeavyHexModel:
    code = HeavyHexCodeBuilder(d=distance).build()
    generators = code.generators
    M = generators.matrix.astype(np.uint8) & 1
    S_mat, _, _ = normalizer(M)
    z_stabs, x_stabs = extract_css_stabilizers(S_mat)
    logical_z, logical_x, logical_z_vec, logical_x_vec = find_logicals_heavyhex(code, S_mat, distance)
    return HeavyHexModel(
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
