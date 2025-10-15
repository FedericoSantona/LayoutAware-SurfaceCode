"""Utilities to decode joint parity bits from merge windows.

Given metadata emitted by GlobalStimBuilder (per-window joint measurement
indices), this module extracts a single parity bit per merge window from a set
of detector samples. The simplest method assumes detectors track time
differences (prev→curr) and uses 1D majority to summarize the chain.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def decode_joint_parity(
    detector_samples: np.ndarray,
    window_meta: Dict[str, object],
) -> np.ndarray:
    """Return one parity bit per shot for a merge window via 1D MWPM.

    We treat the per-round temporal detectors for each joint check as nodes in a
    1D chain over time. For each joint check, we decode the sequence of clicks
    to infer the start/end parity. With independent measurement noise and
    phenomenological data errors folded into time-differences, the optimal
    decoder along time is MWPM on a path graph, equivalent to parity of clicks.

    Implementation shortcut: for a 1D chain with endpoints free, MWPM reduces to
    XOR of all detector clicks along the chain. We therefore compute the XOR of
    all temporal detector bits per joint-check chain and then XOR across all
    chains in the window to obtain a single parity bit (m_ZZ or m_XX).
    """
    shots = detector_samples.shape[0]
    det_sets: List[List[int]] = window_meta.get("joint_detector_indices") or []
    if not det_sets:
        return np.zeros((shots,), dtype=np.uint8)

    # Each entry of det_sets corresponds to one round (list of detector columns
    # for each joint check measured that round). Flatten by joint-check chain:
    # we assume consistent ordering across rounds so position k follows the same
    # joint pair across time.
    # Build per-check list of detector columns across rounds
    max_checks = max(len(round_cols) for round_cols in det_sets if round_cols)
    per_check_cols: List[List[int]] = [[] for _ in range(max_checks)]
    for round_cols in det_sets:
        for k, col in enumerate(round_cols):
            per_check_cols[k].append(col)

    # For each chain, XOR all clicks along time → chain parity bit per shot
    chain_bits: List[np.ndarray] = []
    for cols in per_check_cols:
        if not cols:
            continue
        sub = detector_samples[:, cols]
        # XOR across columns (time) per shot
        chain_parity = np.bitwise_xor.reduce(sub, axis=1)
        chain_bits.append(chain_parity)

    if not chain_bits:
        return np.zeros((shots,), dtype=np.uint8)

    # Final parity bit: XOR across chains in the merge window
    window_parity = np.bitwise_xor.reduce(np.stack(chain_bits, axis=1), axis=1)
    return window_parity.astype(np.uint8)


