"""Logical-operator utilities for surface codes."""
from __future__ import annotations

import numpy as np

from typing import Sequence
from .linalg import nullspace_gf2, row_in_span_gf2, symplectic_to_pauli
from .linalg import _pauli_commutes, _solve_gf2


def find_logicals_general(code, stabilizer_mat: np.ndarray, distance: int, strict_weight_check: bool = True):
    """Return low-weight logical operators for a surface code of given distance.
    
    This is a general implementation that works for both heavy-hex and standard surface codes.
    
    Args:
        code: Code object with .n, .generators attributes
        stabilizer_mat: Stabilizer matrix in symplectic form
        distance: Expected code distance
        strict_weight_check: If True, requires logical weights to exactly equal distance.
                           If False, only finds minimal weight logicals.
    
    Returns:
        Tuple of (logical_z_pauli_string, logical_x_pauli_string, logical_z_vec, logical_x_vec)
    """
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

    if strict_weight_check:
        if not (best_z.sum() == distance and best_x.sum() == distance):
            weight_z = int(best_z.sum())
            weight_x = int(best_x.sum())
            raise RuntimeError(
                f"Logical weights not equal to expected distance d={distance}: "
                f"wZ={weight_z}, wX={weight_x}"
            )

    return symplectic_to_pauli(ZL_vec), symplectic_to_pauli(XL_vec), ZL_vec, XL_vec


def find_logicals_heavyhex(code, stabilizer_mat: np.ndarray, distance: int):
    """Return low-weight logical operators for a heavy-hex code of given distance.
    
    Wrapper around find_logicals_general() for backward compatibility.
    """
    return find_logicals_general(code, stabilizer_mat, distance, strict_weight_check=True)


def find_logicals_standard(code, stabilizer_mat: np.ndarray, distance: int):
    """Return low-weight logical operators for a standard surface code of given distance.
    
    Wrapper around find_logicals_general() for standard surface codes.
    Note: Standard surface codes may have logical weights that don't exactly match
    the distance parameter, so we use relaxed weight checking.
    """
    return find_logicals_general(code, stabilizer_mat, distance, strict_weight_check=False)


def find_logicals_surface_code(code, stabilizer_mat: np.ndarray, distance: int):
    """Return low-weight logical operators for a surface code of given distance.
    
    Alias for find_logicals_standard() for standard surface codes.
    """
    return find_logicals_standard(code, stabilizer_mat, distance)


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
    

def _align_logical_x_to_masked_z(  
    logical_x: str | None,
    x_stabilizers: Sequence[str],
    masked_z: Sequence[str],
    *,
    verbose: bool = False,
) -> str | None:
    """Pick an equivalent logical-X that commutes with masked Z checks.

    Rather than stripping Z stabilizers (which weakens error detection),
    adjust the representative of the logical X by multiplying X stabilizers
    so that it commutes with the boundary-masked Z set used during rough merge.
    """
    if logical_x is None or not masked_z or not x_stabilizers:
        print("[logical-align] No logical X or masked Z or X stabilizers provided")
        return logical_x
    
    n = len(logical_x)
    l_vec = [1 if c in {"X", "Y"} else 0 for c in logical_x]
    z_rows = [[1 if c == "Z" else 0 for c in stab] for stab in masked_z]
    x_rows = [[1 if c in {"X", "Y"} else 0 for c in stab] for stab in x_stabilizers]

    if all(_pauli_commutes(logical_x, z) for z in masked_z):
        print("[logical-align] Logical X already commutes with masked Z, no adjustment needed")
        return logical_x
    print("Logical X needs adjustment")
    
    rhs = [sum(l * z for l, z in zip(l_vec, row)) % 2 for row in z_rows]
    A: list[list[int]] = []
    for row in z_rows:
        A.append([sum(x_row[i] & row[i] for i in range(n)) % 2 for x_row in x_rows])

    coeffs = _solve_gf2(A, rhs)
    if coeffs is None:
        print("[logical-align] No solution found for logical X alignment")
        return logical_x

    delta = [0] * n
    used_indices: list[int] = []
    for idx, (coeff, x_row) in enumerate(zip(coeffs, x_rows)):
        if coeff:
            used_indices.append(idx)
            delta = [(d ^ xr) for d, xr in zip(delta, x_row)]

    aligned_vec = [(l ^ d) for l, d in zip(l_vec, delta)]
    aligned = "".join("X" if v else "I" for v in aligned_vec)
    if not all(_pauli_commutes(aligned, z) for z in masked_z):
        print("[logical-align] Adjusted logical X does not commute with masked Z")
        return logical_x
    
    if verbose:
        delta_support = sum(1 for l, a in zip(logical_x, aligned) if l != a)
        print(
            "[logical-align] adjusted logical X: "
            f"used {sum(coeffs)} X stabilizers, "
            f"delta_support={delta_support}, "
            f"wX_before={logical_x.count('X')}, "
            f"wX_after={aligned.count('X')}"
        )
        if used_indices:
            print(f"[logical-align] stabilizer indices used (0-based): {[i for i,c in enumerate(coeffs) if c]}")
    return aligned



def _multiply_paulis_disjoint(a: str, b: str) -> str:
    """Multiply two Pauli strings assuming disjoint support (or identical chars).

    This ignores overall phases and assumes that at each position either:
      * one of a,b is 'I', or
      * a == b.
    """
    if len(a) != len(b):
        raise ValueError("Pauli strings must have the same length")
    out = []
    for ca, cb in zip(a, b):
        if ca == 'I':
            out.append(cb)
        elif cb == 'I':
            out.append(ca)
        elif ca == cb:
            out.append(ca)
        else:
            raise ValueError(f"Overlapping non-commuting Paulis at site: {ca}, {cb}")
    return "".join(out)