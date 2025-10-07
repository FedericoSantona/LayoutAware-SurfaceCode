"""Helpers for extracting CSS stabilizers from gauge generator data."""
from __future__ import annotations

import numpy as np

from .linalg import nullspace_gf2, rank_gf2


def rank_increases(matrix: np.ndarray, row: np.ndarray) -> bool:
    """Return ``True`` iff appending ``row`` increases GF(2) rank of ``matrix``."""
    if matrix.size == 0:
        return bool(np.any(row))
    current_rank = rank_gf2(matrix)
    candidate_rank = rank_gf2(np.vstack([matrix, row & 1]))
    return candidate_rank > current_rank


def extract_css_stabilizers(stabilizer_mat: np.ndarray) -> tuple[list[str], list[str]]:
    """Project an arbitrary stabilizer basis onto CSS Z- and X-type generators."""
    S_mat = (stabilizer_mat & 1).astype(np.uint8)
    _, two_n = S_mat.shape
    n = two_n // 2
    S_z = S_mat[:, :n]
    S_x = S_mat[:, n:]

    chosen_full = np.zeros((0, 2 * n), dtype=np.uint8)
    z_rows: list[np.ndarray] = []
    x_rows: list[np.ndarray] = []

    for row in S_mat:
        z_block, x_block = row[:n], row[n:]
        if not x_block.any():
            if rank_increases(chosen_full, row):
                chosen_full = np.vstack([chosen_full, row])
                z_rows.append(z_block.copy())
        elif not z_block.any():
            if rank_increases(chosen_full, row):
                chosen_full = np.vstack([chosen_full, row])
                x_rows.append(x_block.copy())

    for coeff in nullspace_gf2(S_x.T):
        if not coeff.any():
            continue
        z = (coeff @ S_z) & 1
        if not z.any():
            continue
        candidate = np.zeros(2 * n, dtype=np.uint8)
        candidate[:n] = z
        if rank_increases(chosen_full, candidate):
            chosen_full = np.vstack([chosen_full, candidate])
            z_rows.append(z)

    for coeff in nullspace_gf2(S_z.T):
        if not coeff.any():
            continue
        x = (coeff @ S_x) & 1
        if not x.any():
            continue
        candidate = np.zeros(2 * n, dtype=np.uint8)
        candidate[n:] = x
        if rank_increases(chosen_full, candidate):
            chosen_full = np.vstack([chosen_full, candidate])
            x_rows.append(x)

    def z_string(z_vec: np.ndarray) -> str:
        return ''.join('Z' if val else 'I' for val in z_vec.tolist())

    def x_string(x_vec: np.ndarray) -> str:
        return ''.join('X' if val else 'I' for val in x_vec.tolist())

    z_stabs = [z_string(z) for z in z_rows if np.any(z)]
    x_stabs = [x_string(x) for x in x_rows if np.any(x)]
    return z_stabs, x_stabs
