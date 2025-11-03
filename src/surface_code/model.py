"""Abstract base class for surface code models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SurfaceCodeModel(ABC):
    """Abstract base class for surface code models.
    
    This class defines the common interface that all surface code implementations
    must follow, allowing seamless switching between different code types.
    """
    distance: int
    code: Any
    generators: Any
    stabilizer_matrix: np.ndarray
    z_stabilizers: list[str]
    x_stabilizers: list[str]
    logical_z: str
    logical_x: str
    logical_z_vec: np.ndarray
    logical_x_vec: np.ndarray

    @abstractmethod
    def diagnostics(self) -> dict[str, int | bool]:
        """Return diagnostic information about the code model.
        
        Returns:
            Dictionary containing code properties like n, s, r, k, and logical checks.
        """
        pass

