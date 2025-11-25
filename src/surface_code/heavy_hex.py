"""Build and package heavy-hex surface-code data for simulations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from qiskit_qec.codes.codebuilders.heavyhex_code_builder import HeavyHexCodeBuilder
from qiskit_qec.linear.symplectic import normalizer

from .linalg import rank_gf2
from .logicals import check_logicals, find_logicals_heavyhex
from .model import SurfaceCodeModel
from .stabilizers import extract_css_stabilizers

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class HeavyHexModel(SurfaceCodeModel):
    """Heavy-hex surface code model implementation."""
    
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
    plot_code(code, distance)
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


def plot_code(code, distance: int) -> None:
    """Plot the code."""
    # save the code tiling
    plot_dir = PROJECT_ROOT / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig = code.draw(
        face_colors=False,
        xcolor="lightcoral",
        zcolor="skyblue",
        figsize=(5, 5),
        show_index=True,

    )
    plt.savefig(plot_dir / f"heavy_hex_d{distance}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"heavy hex code tiling saved to {plot_dir}/heavy_hex_d{distance}.png")
