from __future__ import annotations

import numpy as np

from .linalg import nullspace_gf2, rank_gf2, _solve_add_only, _pauli_commutes
from typing import Sequence

"""Helpers for extracting CSS stabilizers from gauge generator data."""

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


"Helper functions for stripping stabilizers at boundaries, so as to satisfy STIM circuit requirements."


def _strip_x_on_qubits(pauli_str: str, qubits: Sequence[int]) -> str:
    """Return a Pauli string with Xs removed on the given local qubits.

    This is used to modify X-type stabilizers near the C–INT smooth
    boundary so that, during the smooth-merge phase, the joint Z checks
    added along the seam do not anti-commute with any X stabilizers.

    The input `pauli_str` is in the single-patch index space; the caller
    is responsible for embedding it into the combined index space.
    """
    chars = list(pauli_str)
    n = len(chars)
    for q in qubits:
        if 0 <= q < n and chars[q] == "X":
            chars[q] = "I"
    return "".join(chars)

def _strip_z_on_qubits(pauli_str: str, qubits: Sequence[int]) -> str:
    """Return a Pauli string with Zs removed on the given local qubits.

    This is used to modify Z-type stabilizers near the INT–T rough
    boundary so that, during the rough-merge phase, the joint X checks
    added along the seam do not anti-commute with any Z stabilizers.

    The input `pauli_str` is in the single-patch index space; the caller
    is responsible for embedding it into the combined index space.
    """
    chars = list(pauli_str)
    n = len(chars)
    for q in qubits:
        if 0 <= q < n and chars[q] == "Z":
            chars[q] = "I"
    return "".join(chars)


def _commuting_boundary_mask(
    z_stabilizers: Sequence[str],
    x_stabilizers: Sequence[str],
    boundary: Sequence[int],
    *,
    strip_pauli: str = "Z",
    verbose: bool = False,
) -> tuple[list[str], list[str]]:
    """Strip only what is needed at a boundary while keeping CSS checks commuting.

    The routine is symmetric:
      * `strip_pauli="Z"`: strip Z on the boundary (for rough merge) and adjust X.
      * `strip_pauli="X"`: strip X on the boundary (for smooth merge) and adjust Z.

    Strategy (least destructive first):
      1) Strip boundary support of the requested Pauli.
      2) If commutation is broken, try toggling the *other* Pauli off-boundary
         to restore commutation without throwing away boundary information.
      3) As a last resort, strip overlapping Paulis (prefer off-boundary)
         until everything commutes.
    """
    strip_pauli = strip_pauli.upper()
    if strip_pauli not in {"X", "Z"}:
        raise ValueError("strip_pauli must be 'X' or 'Z'")

    primary_char = strip_pauli
    secondary_char = "X" if primary_char == "Z" else "Z"
    strip_fn = _strip_z_on_qubits if primary_char == "Z" else _strip_x_on_qubits
    primary = z_stabilizers if primary_char == "Z" else x_stabilizers
    secondary = x_stabilizers if primary_char == "Z" else z_stabilizers

    def _return_order(primary_list: list[str], secondary_list: list[str]) -> tuple[list[str], list[str]]:
        if primary_char == "Z":
            return primary_list, secondary_list
        return secondary_list, primary_list

    def _pauli_weight(chars: Sequence[str]) -> int:
        return sum(c != "I" for c in chars)

    
    primary_masked = [strip_fn(s, boundary) for s in primary]
    secondary_preserved = list(secondary)

    boundary_removed = sum(
        1 for s, sm in zip(primary, primary_masked) for a, b in zip(s, sm) if a != b
    )

    # Fast path: if keeping X intact still commutes with masked Z, return early.
    if all(_pauli_commutes(p, s) for p in primary_masked for s in secondary_preserved):
        if verbose:
            print(
                f"[boundary-mask] stripped {primary_char}@boundary={boundary_removed}, "
                f"{secondary_char} preserved (already commuting)"
            )
        return _return_order(primary_masked, secondary_preserved)

    if verbose:
        print(
            f"[boundary-mask] stripped {primary_char}@boundary={boundary_removed}, "
            f"{secondary_char} needs adjustment to commute"
        )

    primary_matrix = [[1 if c == primary_char else 0 for c in row] for row in primary_masked]
    secondary_adjusted: list[str] = []
    additive_ok = True
    for stab in secondary_preserved:
        sec_vec = [1 if c == secondary_char else 0 for c in stab]
        b_vec = [sum(a * b for a, b in zip(row, sec_vec)) % 2 for row in primary_matrix]
        free_cols = [i for i in range(len(sec_vec)) if i not in boundary]
        delta = _solve_add_only(primary_matrix, b_vec, free_cols) if free_cols else None
        if delta is None:
            additive_ok = False
            break
        for col, val in zip(free_cols, delta):
            if val:
                sec_vec[col] ^= 1
        secondary_adjusted.append("".join(secondary_char if v else "I" for v in sec_vec))

    if additive_ok and all(_pauli_commutes(p, s) for p in primary_masked for s in secondary_adjusted):
        if verbose:
            added = sum(
                1 for s_old, s_new in zip(secondary_preserved, secondary_adjusted) for a, b in zip(s_old, s_new) if a != b
            )
            print(f"[boundary-mask] additive solution used, {secondary_char} added off-boundary={added}")
        return _return_order(primary_masked, secondary_adjusted)

    # Fallback: strip overlapping Paulis until everything commutes.
    primary_chars = [list(s) for s in primary_masked]
    secondary_chars = [list(s) for s in secondary_preserved]
    changed = True
    fallback_primary_strips = 0
    fallback_secondary_strips = 0
    while changed:
        changed = False
        for pi, p in enumerate(primary_chars):
            for si, s in enumerate(secondary_chars):
                if _pauli_commutes("".join(p), "".join(s)):
                    continue
                overlap = [
                    q
                    for q, (a, b) in enumerate(zip(p, s))
                    if a != "I" and b != "I" and a != b
                ]
                if not overlap:
                    continue
                target = next((q for q in overlap if q not in boundary), overlap[0])
                if _pauli_weight(p) <= 1 and _pauli_weight(s) > 1:
                    s[target] = "I"
                    secondary_chars[si] = s
                    fallback_secondary_strips += 1
                else:
                    p[target] = "I"
                    fallback_primary_strips += 1
                changed = True

    if verbose:
        print(
            "[boundary-mask] fallback used, "
            f"extra {primary_char} stripped={fallback_primary_strips}, "
            f"extra {secondary_char} stripped={fallback_secondary_strips}"
        )
    return _return_order(["".join(p) for p in primary_chars], ["".join(s) for s in secondary_chars])


def stabs_to_symplectic(z_stabs: list[str], x_stabs: list[str]) -> np.ndarray:
    """Convert CSS Z/X stabilizers into a full symplectic matrix [Z | X]."""
    if not z_stabs and not x_stabs:
        return np.zeros((0, 0), dtype=np.uint8)

    n = len(z_stabs[0] if z_stabs else x_stabs[0])
    rows: list[np.ndarray] = []

    # Z-type generators
    for s in z_stabs:
        assert len(s) == n
        vec = np.zeros(2 * n, dtype=np.uint8)
        for q, c in enumerate(s):
            if c == "Z":
                vec[q] = 1
            elif c == "Y":
                vec[q] = 1
                vec[n + q] = 1
            elif c == "X":
                vec[n + q] = 1
        rows.append(vec)

    # X-type generators
    for s in x_stabs:
        assert len(s) == n
        vec = np.zeros(2 * n, dtype=np.uint8)
        for q, c in enumerate(s):
            if c == "X":
                vec[n + q] = 1
            elif c == "Y":
                vec[q] = 1
                vec[n + q] = 1
            elif c == "Z":
                vec[q] = 1
        rows.append(vec)

    return np.vstack(rows) if rows else np.zeros((0, 2 * n), dtype=np.uint8)

