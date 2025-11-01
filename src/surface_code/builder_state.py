"""State management classes for circuit building.

This module encapsulates all state dictionaries used during circuit building
to provide centralized state management and clearer initialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class _PrevState:
    """Previous measurement state for Z, X, and joint checks."""
    z_prev: Dict[str, List[Optional[int]]] = field(default_factory=dict)
    x_prev: Dict[str, List[Optional[int]]] = field(default_factory=dict)
    joint_prev: Dict[Tuple[str, str, str], List[int]] = field(default_factory=dict)  # key=(kind,a,b)


@dataclass
class BuilderState:
    """Container for all state dictionaries used during circuit building."""
    
    prev: _PrevState = field(default_factory=_PrevState)
    terminated_patches: Set[str] = field(default_factory=set)
    byproducts: List[Dict[str, object]] = field(default_factory=list)
    cnot_operations: List[Dict[str, object]] = field(default_factory=list)
    current_cnot: Optional[Dict[str, object]] = None
    
    def __post_init__(self) -> None:
        """Ensure prev is properly initialized."""
        if self.prev is None:
            self.prev = _PrevState()

