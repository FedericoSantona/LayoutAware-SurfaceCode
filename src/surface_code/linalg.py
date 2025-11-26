"""Linear algebra helpers over GF(2) used by the surface-code tooling."""
from __future__ import annotations

import numpy as np

def _pauli_commutes(a: str, b: str) -> bool:
    """Return True iff two Pauli strings commute (ignoring identities)."""
    anti = 0
    for pa, pb in zip(a, b):
        if pa == "I" or pb == "I":
            continue
        if pa != pb:
            anti ^= 1
    return anti == 0


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

def _solve_gf2(A: list[list[int]], b: list[int]) -> list[int] | None:
    """Solve A x = b over GF(2); return one solution or None if inconsistent."""
    if not A:
        return [0] * (len(A[0]) if b else 0)
    m, n = len(A), len(A[0])
    aug = [row[:] + [b[i] & 1] for i, row in enumerate(A)]
    r = 0
    pivots: list[int] = []
    for c in range(n):
        pivot = next((i for i in range(r, m) if aug[i][c]), None)
        if pivot is None:
            continue
        aug[r], aug[pivot] = aug[pivot], aug[r]
        pivots.append(c)
        for i in range(m):
            if i != r and aug[i][c]:
                for j in range(c, n + 1):
                    aug[i][j] ^= aug[r][j]
        r += 1
    for i in range(r, m):
        if aug[i][n]:
            return None
    x = [0] * n
    for i in range(r - 1, -1, -1):
        c = pivots[i]
        val = aug[i][n]
        for j in range(c + 1, n):
            val ^= aug[i][j] & x[j]
        x[c] = val
    return x

def _solve_add_only(z_matrix: list[list[int]], b_vec: list[int], free_cols: list[int]) -> list[int] | None:
        """Solve z_matrix[:, free_cols] * delta = b_vec over GF(2)."""
        if not z_matrix:
            return [0] * len(free_cols)
        m = len(z_matrix)
        n = len(free_cols)
        aug = [[z_matrix[r][c] for c in free_cols] + [b_vec[r] % 2] for r in range(m)]
        row = 0
        pivots: list[int] = []
        for col in range(n):
            pivot = next((r for r in range(row, m) if aug[r][col]), None)
            if pivot is None:
                continue
            aug[row], aug[pivot] = aug[pivot], aug[row]
            pivots.append(col)
            for r in range(m):
                if r != row and aug[r][col]:
                    for c in range(col, n + 1):
                        aug[r][c] ^= aug[row][c]
            row += 1
        for r in range(row, m):
            if aug[r][n]:
                return None
        delta = [0] * n
        for r in range(row - 1, -1, -1):
            col = pivots[r]
            rhs = aug[r][n]
            for c in range(col + 1, n):
                rhs ^= aug[r][c] & delta[c]
            delta[col] = rhs
        return delta

