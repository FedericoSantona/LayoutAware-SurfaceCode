"""Surface code utilities and models for logical qubit simulations."""

from .model import SurfaceCodeModel
from .layout import Layout, SeamSpec
from .stim_builder import PhenomenologicalStimBuilder, PhenomenologicalStimConfig, PhaseSpec
from .stabilizers import stabs_to_symplectic
from .logicals import _multiply_paulis_disjoint
from .surgery import LatticeSurgery
from .geometry_utils import (
    find_boundary_data_qubits,
    find_rough_boundary_data_qubits,
    find_smooth_boundary_data_qubits,
)
from .noise_model import (
    NoiseModel,
    QubitNoiseParams,
    CouplerNoiseParams,
    PhenomenologicalNoiseModel,
    DeviceAwareNoiseModel,
)
from .device_calibration import DeviceCalibration
_heavy_hex_import_error = None
try:
    from .heavy_hex import HeavyHexModel, build_heavy_hex_model
except ImportError as e:  # pragma: no cover - optional dependency
    _heavy_hex_import_error = e
    HeavyHexModel = None

    def build_heavy_hex_model(*args, **kwargs):
        """Lazy import guard for optional heavy-hex dependency."""
        raise ImportError(
            "Heavy-hex codes require the 'qiskit_qec' package. "
            "Install qiskit-qec to enable heavy-hex builders."
        ) from _heavy_hex_import_error

_standard_import_error = None
try:
    from .standard import StandardSurfaceCodeModel, build_standard_surface_code_model
except ImportError as e:  # pragma: no cover - optional dependency
    _standard_import_error = e
    StandardSurfaceCodeModel = None

    def build_standard_surface_code_model(*args, **kwargs):
        """Lazy import guard for optional standard surface-code dependency."""
        raise ImportError(
            "Standard surface-code builders require the 'qiskit_qec' package. "
            "Install qiskit-qec to enable these helpers."
        ) from _standard_import_error


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
    # Surface code models
    "SurfaceCodeModel",
    "HeavyHexModel",
    "StandardSurfaceCodeModel",
    "build_heavy_hex_model",
    "build_standard_surface_code_model",
    "build_surface_code_model",
    # Stim circuit builders
    "PhenomenologicalStimBuilder",
    "PhenomenologicalStimConfig",
    "PhaseSpec",
    # Noise models
    "NoiseModel",
    "QubitNoiseParams",
    "CouplerNoiseParams",
    "PhenomenologicalNoiseModel",
    "DeviceAwareNoiseModel",
    "DeviceCalibration",
    # Layout and surgery
    "Layout",
    "SeamSpec",
    "LatticeSurgery",
    # Geometry utilities
    "find_boundary_data_qubits",
    "find_rough_boundary_data_qubits",
    "find_smooth_boundary_data_qubits",
    # Internal utilities (exported for advanced use)
    "_pauli_commutes",
    "_solve_gf2",
    "_align_logical_x_to_masked_z",
    "_commuting_boundary_mask",
    "stabs_to_symplectic",
    "_multiply_paulis_disjoint",
]
