"""Configuration classes for Stim circuit builders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PhenomenologicalStimConfig:
    """Configuration values for phenomenological stabilizer sampling.

    family:
        None  -> interleave Z and X halves (measure both each round)
        "Z"   -> Z-only family (CSS split, measure only Z stabilizers)
        "X"   -> X-only family (CSS split, measure only X stabilizers)
    """
    rounds: int = 5
    p_x_error: float = 1e-3
    p_z_error: float = 1e-3
    family: Optional[str] = None
    # Optional labels to track logical state preparation/termination
    init_label: Optional[str] = None
    logical_start: Optional[str] = None
    logical_end: Optional[str] = None
    # Per-run logical bracketing basis ('X' or 'Z') for observables/decoding
    bracket_basis: Optional[str] = None
    # Optional: append a final end-only demo readout MPP in the requested basis
    # ("X" or "Z"). This extra measurement is NOT part of OBSERVABLE_INCLUDE and
    # NOT used by detectors; it is for physics-based reporting in the end basis.
    demo_basis: Optional[str] = None
