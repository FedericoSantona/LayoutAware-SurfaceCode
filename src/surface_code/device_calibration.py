"""Device calibration data loader for realistic noise modeling.

This module provides utilities to load device calibration data from
various sources (JSON files, IBM Quantum backends) and convert them
into a format suitable for DeviceAwareNoiseModel.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .noise_model import QubitNoiseParams, CouplerNoiseParams, DeviceAwareNoiseModel


@dataclass
class DeviceCalibration:
    """Container for device calibration data.
    
    This class holds all calibration data needed to construct a
    DeviceAwareNoiseModel, including per-qubit coherence times,
    gate errors, readout errors, and coupling information.
    
    Attributes:
        backend_name: Name of the quantum backend/device.
        timestamp: ISO-format timestamp of when calibration was taken.
        qubit_params: Per-qubit noise parameters.
        coupler_params: Per-coupler noise parameters.
        gate_times: Gate durations in microseconds.
        metadata: Additional metadata (optional).
    """
    backend_name: str
    timestamp: str
    qubit_params: Dict[int, QubitNoiseParams]
    coupler_params: Dict[Tuple[int, int], CouplerNoiseParams] = field(default_factory=dict)
    gate_times: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default gate times if not provided."""
        default_gate_times = {
            "sx": 0.035,
            "x": 0.035,
            "h": 0.035,
            "s": 0.0,
            "rz": 0.0,
            "cx": 0.3,
            "cz": 0.3,
            "ecr": 0.5,
            "measure": 1.0,
        }
        for gate, time in default_gate_times.items():
            if gate not in self.gate_times:
                self.gate_times[gate] = time
    
    @property
    def num_qubits(self) -> int:
        """Number of qubits with calibration data."""
        return len(self.qubit_params)
    
    @property
    def qubit_indices(self) -> List[int]:
        """Sorted list of qubit indices."""
        return sorted(self.qubit_params.keys())
    
    def get_recommended_layout(self) -> str:
        """Get the recommended surface code layout type for this device.
        
        Returns:
            "heavy_hex" for IBM devices, "standard" for IQM Crystal,
            or "heavy_hex" as default.
        """
        backend_lower = self.backend_name.lower()
        
        # Check for IBM devices
        if "ibm" in backend_lower or any(
            name in backend_lower 
            for name in ["sherbrooke", "kyoto", "osaka", "kolkata", "mumbai", "perth"]
        ):
            return "heavy_hex"
        
        # Check for IQM Crystal
        if "iqm" in backend_lower or "crystal" in backend_lower:
            return "standard"
        
        # Default to heavy_hex for unknown devices
        return "heavy_hex"
    
    def to_noise_model(
        self,
        default_round_duration: float = 1.0,
    ) -> DeviceAwareNoiseModel:
        """Convert calibration data to a DeviceAwareNoiseModel.
        
        Args:
            default_round_duration: Default measurement round duration in µs.
            
        Returns:
            DeviceAwareNoiseModel instance configured with this calibration.
        """
        return DeviceAwareNoiseModel(
            qubit_params=self.qubit_params,
            coupler_params=self.coupler_params,
            default_round_duration=default_round_duration,
            gate_times=self.gate_times,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "backend_name": self.backend_name,
            "timestamp": self.timestamp,
            "qubits": {
                str(q): {
                    "t1": params.t1,
                    "t2": params.t2,
                    "readout_error_0to1": params.readout_error_0to1,
                    "readout_error_1to0": params.readout_error_1to0,
                    "gate_error": params.single_qubit_gate_error,
                    "frequency": params.frequency,
                }
                for q, params in self.qubit_params.items()
            },
            "couplers": {
                f"{q1}-{q2}": {
                    "cx_error": params.two_qubit_gate_error,
                    "crosstalk": params.crosstalk_strength,
                }
                for (q1, q2), params in self.coupler_params.items()
            },
            "gate_times": self.gate_times,
            "metadata": self.metadata,
        }
    
    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        """Save calibration data to a JSON file.
        
        Args:
            path: Path to output JSON file.
            indent: JSON indentation level.
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceCalibration":
        """Create DeviceCalibration from a dictionary.
        
        Args:
            data: Dictionary with calibration data.
            
        Returns:
            DeviceCalibration instance.
        """
        qubit_params: Dict[int, QubitNoiseParams] = {}
        for q_str, qdata in data.get("qubits", {}).items():
            q = int(q_str)
            qubit_params[q] = QubitNoiseParams(
                t1=qdata["t1"],
                t2=qdata["t2"],
                readout_error_0to1=qdata.get("readout_error_0to1", qdata.get("readout_error", 0.0)),
                readout_error_1to0=qdata.get("readout_error_1to0", qdata.get("readout_error", 0.0)),
                single_qubit_gate_error=qdata.get("gate_error", 0.0),
                frequency=qdata.get("frequency"),
            )
        
        coupler_params: Dict[Tuple[int, int], CouplerNoiseParams] = {}
        for edge_str, edata in data.get("couplers", {}).items():
            # Parse edge string like "0-1" or "0_1"
            parts = edge_str.replace("_", "-").split("-")
            q1, q2 = int(parts[0]), int(parts[1])
            # Normalize to (min, max) order
            key = (min(q1, q2), max(q1, q2))
            coupler_params[key] = CouplerNoiseParams(
                two_qubit_gate_error=edata.get("cx_error", edata.get("two_qubit_gate_error", 0.0)),
                crosstalk_strength=edata.get("crosstalk", 0.0),
            )
        
        return cls(
            backend_name=data.get("backend_name", "unknown"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            qubit_params=qubit_params,
            coupler_params=coupler_params,
            gate_times=data.get("gate_times", {}),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DeviceCalibration":
        """Load calibration data from a JSON file.
        
        Args:
            path: Path to JSON calibration file.
            
        Returns:
            DeviceCalibration instance.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_ibm_backend(cls, backend: Any) -> "DeviceCalibration":
        """Load calibration from an IBM Quantum backend.
        
        This method extracts T1, T2, readout errors, and gate errors
        from an IBM Quantum backend's properties.
        
        Args:
            backend: A Qiskit IBMBackend object (from qiskit_ibm_runtime).
            
        Returns:
            DeviceCalibration instance.
            
        Raises:
            ImportError: If qiskit_ibm_runtime is not available.
            ValueError: If backend properties cannot be extracted.
        """
        try:
            # Try qiskit_ibm_runtime first (newer API)
            backend_name = backend.name
        except AttributeError:
            backend_name = str(backend)
        
        # Get backend properties
        try:
            properties = backend.properties()
        except AttributeError:
            # Newer Qiskit IBMBackend API
            try:
                properties = backend.target
                return cls._from_ibm_target(backend_name, properties)
            except AttributeError:
                raise ValueError(
                    "Cannot extract properties from backend. "
                    "Ensure you're using a Qiskit IBMBackend object."
                )
        
        if properties is None:
            raise ValueError("Backend properties not available.")
        
        return cls._from_ibm_properties(backend_name, properties)
    
    @classmethod
    def _from_ibm_properties(
        cls,
        backend_name: str,
        properties: Any,
    ) -> "DeviceCalibration":
        """Extract calibration from IBM backend properties (legacy API)."""
        qubit_params: Dict[int, QubitNoiseParams] = {}
        coupler_params: Dict[Tuple[int, int], CouplerNoiseParams] = {}
        gate_times: Dict[str, float] = {}
        
        # Extract qubit properties
        for q in range(len(properties.qubits)):
            try:
                t1 = properties.t1(q) * 1e6  # Convert seconds to µs
                t2 = properties.t2(q) * 1e6
                
                # Readout errors
                readout_error = properties.readout_error(q)
                
                # Single-qubit gate error (use SX gate if available)
                try:
                    gate_error = properties.gate_error("sx", q)
                except Exception:
                    try:
                        gate_error = properties.gate_error("x", q)
                    except Exception:
                        gate_error = 0.001  # Default fallback
                
                # Frequency
                try:
                    frequency = properties.frequency(q) / 1e9  # Convert Hz to GHz
                except Exception:
                    frequency = None
                
                qubit_params[q] = QubitNoiseParams(
                    t1=t1,
                    t2=t2,
                    readout_error_0to1=readout_error,
                    readout_error_1to0=readout_error,
                    single_qubit_gate_error=gate_error,
                    frequency=frequency,
                )
            except Exception:
                # Skip qubits with missing data
                continue
        
        # Extract coupling map and two-qubit gate errors
        try:
            coupling_map = properties.coupling_map
            for edge in coupling_map:
                q1, q2 = edge
                key = (min(q1, q2), max(q1, q2))
                if key in coupler_params:
                    continue  # Already processed this edge
                
                try:
                    cx_error = properties.gate_error("cx", [q1, q2])
                except Exception:
                    try:
                        cx_error = properties.gate_error("ecr", [q1, q2])
                    except Exception:
                        cx_error = 0.01  # Default fallback
                
                coupler_params[key] = CouplerNoiseParams(
                    two_qubit_gate_error=cx_error,
                    crosstalk_strength=0.0,
                )
        except Exception:
            pass  # No coupling map available
        
        # Extract gate times
        try:
            for gate in ["sx", "x", "cx", "ecr"]:
                try:
                    # Get gate length for first available qubit
                    for q in range(len(properties.qubits)):
                        try:
                            length = properties.gate_length(gate, q) * 1e6  # µs
                            gate_times[gate] = length
                            break
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            pass
        
        return cls(
            backend_name=backend_name,
            timestamp=datetime.now().isoformat(),
            qubit_params=qubit_params,
            coupler_params=coupler_params,
            gate_times=gate_times,
            metadata={"source": "ibm_properties_api"},
        )
    
    @classmethod
    def _from_ibm_target(
        cls,
        backend_name: str,
        target: Any,
    ) -> "DeviceCalibration":
        """Extract calibration from IBM backend Target (newer API)."""
        qubit_params: Dict[int, QubitNoiseParams] = {}
        coupler_params: Dict[Tuple[int, int], CouplerNoiseParams] = {}
        gate_times: Dict[str, float] = {}
        
        # Get number of qubits
        num_qubits = target.num_qubits
        
        for q in range(num_qubits):
            try:
                # Extract from target qubit properties
                qubit_props = target.qubit_properties
                if qubit_props is not None and q < len(qubit_props):
                    props = qubit_props[q]
                    t1 = getattr(props, 't1', 100e-6) * 1e6  # µs
                    t2 = getattr(props, 't2', 50e-6) * 1e6
                    frequency = getattr(props, 'frequency', None)
                    if frequency:
                        frequency = frequency / 1e9  # GHz
                else:
                    t1, t2, frequency = 100.0, 50.0, None
                
                # Get gate error from instruction properties
                gate_error = 0.001
                for gate_name in ["sx", "x"]:
                    try:
                        inst_props = target[gate_name][(q,)]
                        if inst_props and inst_props.error is not None:
                            gate_error = inst_props.error
                            break
                    except (KeyError, TypeError):
                        continue
                
                # Get readout error
                readout_error = 0.01
                try:
                    meas_props = target["measure"][(q,)]
                    if meas_props and meas_props.error is not None:
                        readout_error = meas_props.error
                except (KeyError, TypeError):
                    pass
                
                qubit_params[q] = QubitNoiseParams(
                    t1=t1,
                    t2=min(t2, 2 * t1 - 0.001),  # Ensure T2 <= 2*T1
                    readout_error_0to1=readout_error,
                    readout_error_1to0=readout_error,
                    single_qubit_gate_error=gate_error,
                    frequency=frequency,
                )
            except Exception:
                continue
        
        # Extract coupling information
        try:
            coupling_map = target.build_coupling_map()
            if coupling_map:
                for edge in coupling_map.get_edges():
                    q1, q2 = edge
                    key = (min(q1, q2), max(q1, q2))
                    if key in coupler_params:
                        continue
                    
                    cx_error = 0.01
                    for gate_name in ["cx", "ecr", "cz"]:
                        try:
                            inst_props = target[gate_name][(q1, q2)]
                            if inst_props and inst_props.error is not None:
                                cx_error = inst_props.error
                                break
                        except (KeyError, TypeError):
                            continue
                    
                    coupler_params[key] = CouplerNoiseParams(
                        two_qubit_gate_error=cx_error,
                        crosstalk_strength=0.0,
                    )
        except Exception:
            pass
        
        # Extract gate times from target
        for gate_name in ["sx", "x", "cx", "ecr", "cz"]:
            try:
                for qargs in target.qargs_for_operation_name(gate_name):
                    inst_props = target[gate_name][qargs]
                    if inst_props and inst_props.duration is not None:
                        gate_times[gate_name] = inst_props.duration * 1e6  # µs
                        break
            except (KeyError, TypeError):
                continue
        
        return cls(
            backend_name=backend_name,
            timestamp=datetime.now().isoformat(),
            qubit_params=qubit_params,
            coupler_params=coupler_params,
            gate_times=gate_times,
            metadata={"source": "ibm_target_api"},
        )
    
    @classmethod
    def uniform(
        cls,
        num_qubits: int,
        t1: float = 100.0,
        t2: float = 80.0,
        readout_error: float = 0.01,
        gate_error_1q: float = 0.001,
        gate_error_2q: float = 0.01,
        connectivity: Optional[List[Tuple[int, int]]] = None,
    ) -> "DeviceCalibration":
        """Create uniform calibration with same parameters for all qubits.
        
        Useful for testing or when no real calibration data is available.
        
        Args:
            num_qubits: Number of qubits.
            t1: T1 time in µs (same for all).
            t2: T2 time in µs (same for all).
            readout_error: Readout error rate (same for all).
            gate_error_1q: Single-qubit gate error (same for all).
            gate_error_2q: Two-qubit gate error (same for all).
            connectivity: List of (q1, q2) tuples for coupling map.
                         If None, assumes all-to-all connectivity.
            
        Returns:
            DeviceCalibration with uniform parameters.
        """
        qubit_params = {
            q: QubitNoiseParams(
                t1=t1,
                t2=t2,
                readout_error_0to1=readout_error,
                readout_error_1to0=readout_error,
                single_qubit_gate_error=gate_error_1q,
            )
            for q in range(num_qubits)
        }
        
        if connectivity is None:
            # All-to-all connectivity
            connectivity = [
                (i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)
            ]
        
        coupler_params = {
            (min(q1, q2), max(q1, q2)): CouplerNoiseParams(
                two_qubit_gate_error=gate_error_2q,
            )
            for q1, q2 in connectivity
        }
        
        return cls(
            backend_name="uniform_synthetic",
            timestamp=datetime.now().isoformat(),
            qubit_params=qubit_params,
            coupler_params=coupler_params,
            metadata={"synthetic": True},
        )
    
    def summary(self) -> str:
        """Return a human-readable summary of the calibration."""
        lines = [
            f"Device: {self.backend_name}",
            f"Timestamp: {self.timestamp}",
            f"Qubits: {self.num_qubits}",
            f"Couplers: {len(self.coupler_params)}",
        ]
        
        if self.qubit_params:
            t1_vals = [p.t1 for p in self.qubit_params.values()]
            t2_vals = [p.t2 for p in self.qubit_params.values()]
            ro_vals = [p.readout_error for p in self.qubit_params.values()]
            
            lines.extend([
                f"T1 range: {min(t1_vals):.1f} - {max(t1_vals):.1f} µs",
                f"T2 range: {min(t2_vals):.1f} - {max(t2_vals):.1f} µs",
                f"Readout error range: {min(ro_vals):.4f} - {max(ro_vals):.4f}",
            ])
        
        if self.coupler_params:
            cx_vals = [p.two_qubit_gate_error for p in self.coupler_params.values()]
            lines.append(f"2Q gate error range: {min(cx_vals):.4f} - {max(cx_vals):.4f}")
        
        return "\n".join(lines)
