"""Surface code utilities and models for logical qubit simulations."""

from .model import SurfaceCodeModel
from .heavy_hex import HeavyHexModel, build_heavy_hex_model
from .surface_code import StandardSurfaceCodeModel, build_standard_surface_code_model
from .builder import GlobalStimBuilder
from .configs import PhenomenologicalStimConfig
from .layout import Layout, PatchObject, create_single_patch_layout
from .surgery_ops import MeasureRound, Merge, Split, ParityReadout
from .pauli import PauliTracker


def build_surface_code_model(distance: int, code_type: str = "heavy_hex") -> SurfaceCodeModel:
    """Factory function to build surface code models of different types.
    
    Args:
        distance: Code distance
        code_type: Type of surface code to build. Must be one of:
            - "heavy_hex": Heavy-hex surface code (default)
            - "standard": Standard surface code
            
    Returns:
        SurfaceCodeModel instance (either HeavyHexModel or StandardSurfaceCodeModel)
        
    Raises:
        ValueError: If code_type is not supported
    """
    if code_type == "heavy_hex":
        return build_heavy_hex_model(distance)
    elif code_type == "standard":
        return build_standard_surface_code_model(distance)
    else:
        raise ValueError(f"Unsupported code_type: {code_type}. Must be 'heavy_hex' or 'standard'.")


__all__ = [
    "SurfaceCodeModel",
    "HeavyHexModel",
    "StandardSurfaceCodeModel",
    "build_heavy_hex_model",
    "build_standard_surface_code_model",
    "build_surface_code_model",
    "GlobalStimBuilder",
    "PhenomenologicalStimConfig",
    "Layout",
    "PatchObject",
    "create_single_patch_layout",
    "MeasureRound",
    "Merge",
    "Split",
    "ParityReadout",
    "PauliTracker",
]
