"""Surgery operation dataclasses for timeline-driven global building.

Supported ops:
  - MeasureRound: perform one ZX cycle on specified patches (or all if None).
  - Merge: activate joint checks across a seam for a fixed number of rounds.
  - Split: deactivate joint checks across a seam.
  - ParityReadout: mark a merge window for parity extraction downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class MeasureRound:
    patch_ids: List[str] | None = None  # None ⇒ all patches


@dataclass
class Merge:
    type: str  # 'rough' | 'smooth'
    a: str
    b: str
    rounds: int  # d


@dataclass
class Split:
    type: str  # 'rough' | 'smooth'
    a: str
    b: str


@dataclass
class ParityReadout:
    name: str   # label, e.g. 'ZZ' or 'XX'
    type: str   # 'ZZ' | 'XX'
    a: str
    b: str


@dataclass
class CNOTOp:
    """High-level CNOT operation using ancilla-mediated surgery.
    
    This operation will be expanded by the compiler into:
    1. Rough ZZ merge (control-ancilla) for d rounds
    2. Split rough seam
    3. Smooth XX merge (ancilla-target) for d rounds  
    4. Split smooth seam
    5. Parity readouts for both merge windows
    
    The ancilla is virtually initialized in |+⟩_L (X-basis, +1 eigenstate).
    """
    control: str
    target: str
    ancilla: str
    rounds: int  # d


