"""Linear algebra helpers over GF(2) used by the surface-code tooling."""
from __future__ import annotations

import numpy as np


def rank_gf2(matrix: np.ndarray) -> int:
    """Return the rank of ``matrix`` over GF(2)."""
    mat = (matrix.copy() & 1).astype(np.uint8)
    rows, cols = mat.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if mat[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != rank:
            mat[[rank, pivot]] = mat[[pivot, rank]]
        for r in range(rows):
            if r != rank and mat[r, col]:
                mat[r, :] ^= mat[rank, :]
        rank += 1
    return rank


def nullspace_gf2(matrix: np.ndarray) -> list[np.ndarray]:
    """Return a basis for the nullspace of ``matrix`` over GF(2)."""
    mat = (matrix.copy() & 1).astype(np.uint8)
    rows, cols = mat.shape
    rref = mat.copy()
    pivot_cols = [-1] * rows
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if rref[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != rank:
            rref[[rank, pivot]] = rref[[pivot, rank]]
        pivot_cols[rank] = col
        for r in range(rows):
            if r != rank and rref[r, col]:
                rref[r, :] ^= rref[rank, :]
        rank += 1
    free_cols = [c for c in range(cols) if c not in pivot_cols[:rank]]
    basis = []
    for free in free_cols:
        vec = np.zeros(cols, dtype=np.uint8)
        vec[free] = 1
        for pivot_row in range(rank - 1, -1, -1):
            pivot_col = pivot_cols[pivot_row]
            if pivot_col == -1:
                continue
            if rref[pivot_row, free]:
                vec[pivot_col] ^= 1
        basis.append(vec)
    return basis


def symplectic_to_pauli(symplectic_vec: np.ndarray) -> str:
    """Convert a binary symplectic vector ``[z | x]`` to a Pauli string."""
    n = symplectic_vec.size // 2
    z_block = symplectic_vec[:n]
    x_block = symplectic_vec[n:]
    letters = []
    for z_val, x_val in zip(z_block, x_block):
        if z_val and x_val:
            letters.append("Y")
        elif z_val:
            letters.append("Z")
        elif x_val:
            letters.append("X")
        else:
            letters.append("I")
    return "".join(letters)


def row_in_span_gf2(matrix: np.ndarray, row: np.ndarray) -> bool:
    """Return ``True`` iff ``row`` is in the GF(2) row span of ``matrix``."""
    if matrix.size == 0:
        return bool(np.any(row))
    current_rank = rank_gf2(matrix)
    stacked_rank = rank_gf2(np.vstack([matrix, row & 1]))
    return stacked_rank == current_rank
