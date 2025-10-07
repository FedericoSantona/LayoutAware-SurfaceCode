"""Logical-operator utilities for heavy-hex surface codes."""
from __future__ import annotations

import numpy as np

from .linalg import nullspace_gf2, row_in_span_gf2, symplectic_to_pauli


def find_logicals_heavyhex(code, stabilizer_mat: np.ndarray, distance: int):
    """Return low-weight logical operators for a heavy-hex code of given distance."""
    n = code.n
    M = (code.generators.matrix.astype(np.uint8) & 1)
    Z_M = M[:, :n]
    X_M = M[:, n:]

    S_mat = (stabilizer_mat & 1).astype(np.uint8)
    S_Z = S_mat[:, :n]
    S_X = S_mat[:, n:]

    def minimize_weight_z(z_vec: np.ndarray) -> np.ndarray:
        candidate = (z_vec & 1).copy()
        improved = True
        while improved:
            improved = False
            for row in S_Z:
                trial = candidate ^ row
                if (X_M.dot(trial) % 2).any():
                    continue
                if trial.sum() < candidate.sum():
                    candidate = trial
                    improved = True
        return candidate

    def minimize_weight_x(x_vec: np.ndarray) -> np.ndarray:
        candidate = (x_vec & 1).copy()
        improved = True
        while improved:
            improved = False
            for row in S_X:
                trial = candidate ^ row
                if (Z_M.dot(trial) % 2).any():
                    continue
                if trial.sum() < candidate.sum():
                    candidate = trial
                    improved = True
        return candidate

    z_candidates = [vec for vec in nullspace_gf2(X_M) if vec.any()]
    if not z_candidates:
        raise RuntimeError("No nontrivial Z logical candidate in nullspace of X_M.")

    best_z = None
    best_weight = 10 ** 9
    for candidate in z_candidates:
        reduced = minimize_weight_z(candidate)
        full = np.zeros(2 * n, dtype=np.uint8)
        full[:n] = reduced
        if row_in_span_gf2(M, full) or row_in_span_gf2(S_mat, full):
            continue
        weight = int(reduced.sum())
        if weight < best_weight:
            best_weight = weight
            best_z = reduced
    if best_z is None:
        raise RuntimeError("Failed to find a Z logical outside the stabilizer/gauge span.")

    x_candidates = [vec for vec in nullspace_gf2(Z_M) if vec.any()]
    if not x_candidates:
        raise RuntimeError("No nontrivial X logical candidate in nullspace of Z_M.")

    best_x = None
    best_x_weight = 10 ** 9
    for candidate in x_candidates:
        if (int(np.dot(best_z, candidate)) & 1) == 0:
            continue
        reduced = minimize_weight_x(candidate)
        full = np.zeros(2 * n, dtype=np.uint8)
        full[n:] = reduced
        if row_in_span_gf2(M, full) or row_in_span_gf2(S_mat, full):
            continue
        weight = int(reduced.sum())
        if weight < best_x_weight:
            best_x_weight = weight
            best_x = reduced
    if best_x is None:
        raise RuntimeError("Failed to find an X logical with odd overlap outside stabilizer span.")

    ZL_vec = np.zeros(2 * n, dtype=np.uint8)
    XL_vec = np.zeros(2 * n, dtype=np.uint8)
    ZL_vec[:n] = best_z
    XL_vec[n:] = best_x

    if not (best_z.sum() == distance and best_x.sum() == distance):
        weight_z = int(best_z.sum())
        weight_x = int(best_x.sum())
        raise RuntimeError(
            f"Logical weights not equal to expected distance d={distance}: "
            f"wZ={weight_z}, wX={weight_x}"
        )

    return symplectic_to_pauli(ZL_vec), symplectic_to_pauli(XL_vec), ZL_vec, XL_vec


def check_logicals(ZL_vec: np.ndarray, XL_vec: np.ndarray, stabilizer_mat: np.ndarray) -> dict[str, bool | int]:
    """Return diagnostics ensuring chosen logicals behave as expected."""
    S_mat = (stabilizer_mat & 1).astype(np.uint8)
    n = ZL_vec.size // 2

    def symplectic_commute(a: np.ndarray, b: np.ndarray) -> bool:
        z1, x1 = a[:n], a[n:]
        z2, x2 = b[:n], b[n:]
        return ((np.dot(z1, x2) + np.dot(x1, z2)) % 2) == 0

    commute_Z = [symplectic_commute(ZL_vec, row) for row in S_mat]
    commute_X = [symplectic_commute(XL_vec, row) for row in S_mat]
    anticommute = not symplectic_commute(ZL_vec, XL_vec)

    def in_stabilizer(vec: np.ndarray) -> bool:
        return row_in_span_gf2(S_mat, vec & 1)

    diagnostics = {
        "commute_Z": all(commute_Z),
        "commute_X": all(commute_X),
        "anticommute": anticommute,
        "Z_in_stabilizer": in_stabilizer(ZL_vec),
        "X_in_stabilizer": in_stabilizer(XL_vec),
        "weight_Z": int(ZL_vec[:n].sum() + ZL_vec[n:].sum()),
        "weight_X": int(XL_vec[:n].sum() + XL_vec[n:].sum()),
    }
    return diagnostics
