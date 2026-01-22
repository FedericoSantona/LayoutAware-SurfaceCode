"""Noise model abstraction for device-aware surface code simulations.

This module provides an abstraction layer for noise models, allowing
both simple phenomenological noise (uniform error rates) and realistic
device-aware noise (per-qubit T1/T2, gate errors, readout errors, crosstalk).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import stim


@dataclass
class QubitNoiseParams:
    """Per-qubit noise parameters from device calibration.
    
    Attributes:
        t1: T1 relaxation time in microseconds (amplitude damping).
        t2: T2 dephasing time in microseconds (includes T1 contribution).
        readout_error_0to1: Probability of measuring 1 given state |0⟩.
        readout_error_1to0: Probability of measuring 0 given state |1⟩.
        single_qubit_gate_error: Average single-qubit gate error rate.
        frequency: Optional qubit frequency in GHz (for crosstalk modeling).
    """
    t1: float  # microseconds
    t2: float  # microseconds
    readout_error_0to1: float = 0.0
    readout_error_1to0: float = 0.0
    single_qubit_gate_error: float = 0.0
    frequency: Optional[float] = None
    
    def __post_init__(self):
        """Validate physical constraints."""
        if self.t1 <= 0:
            raise ValueError(f"T1 must be positive, got {self.t1}")
        if self.t2 <= 0:
            raise ValueError(f"T2 must be positive, got {self.t2}")
        if self.t2 > 2 * self.t1:
            # T2 ≤ 2*T1 is a physical constraint
            raise ValueError(f"T2 ({self.t2}) cannot exceed 2*T1 ({2*self.t1})")
    
    @property
    def t2_phi(self) -> float:
        """Pure dephasing time: 1/T2_phi = 1/T2 - 1/(2*T1)."""
        # Avoid division by zero or negative values
        inv_t2_phi = 1.0 / self.t2 - 1.0 / (2.0 * self.t1)
        if inv_t2_phi <= 0:
            # T2 = 2*T1 limit: pure dephasing vanishes
            return float('inf')
        return 1.0 / inv_t2_phi
    
    @property
    def readout_error(self) -> float:
        """Average readout error rate."""
        return (self.readout_error_0to1 + self.readout_error_1to0) / 2.0


@dataclass
class CouplerNoiseParams:
    """Per-coupler (edge) noise parameters for two-qubit gates.
    
    Attributes:
        two_qubit_gate_error: Average two-qubit gate error rate (e.g., CX/CZ).
        crosstalk_strength: Crosstalk coupling strength (ZZ interaction rate).
    """
    two_qubit_gate_error: float = 0.0
    crosstalk_strength: float = 0.0


class NoiseModel(ABC):
    """Abstract base class for noise models.
    
    A noise model defines how errors are injected into a Stim circuit
    during stabilizer measurement rounds. Implementations can range from
    simple uniform error rates to realistic device-calibrated noise.
    """
    
    @abstractmethod
    def apply_data_qubit_noise(
        self,
        circuit: stim.Circuit,
        qubits: Sequence[int],
        duration: Optional[float] = None,
    ) -> None:
        """Apply noise to data qubits (idle/decoherence errors).
        
        This is called during each stabilizer measurement round to inject
        errors on data qubits. For phenomenological noise, this applies
        uniform X/Z errors. For device-aware noise, this computes error
        rates from T1/T2 decoherence.
        
        Args:
            circuit: Stim circuit to append noise operations to.
            qubits: List of qubit indices to apply noise to.
            duration: Optional time duration in microseconds for decoherence
                     calculation. If None, uses a default round duration.
        """
        pass
    
    @abstractmethod
    def apply_measurement_noise(
        self,
        circuit: stim.Circuit,
        qubits: Sequence[int],
    ) -> None:
        """Apply measurement (readout) errors.
        
        This is called before/after measurement operations to inject
        readout errors. For phenomenological noise, this may be a no-op.
        For device-aware noise, this applies per-qubit readout error rates.
        
        Args:
            circuit: Stim circuit to append noise operations to.
            qubits: List of qubit indices being measured.
        """
        pass
    
    @abstractmethod
    def get_effective_error_rate(
        self,
        qubit: int,
        error_type: str,
    ) -> float:
        """Get the effective error rate for a specific qubit and error type.
        
        Useful for inspection and debugging of noise parameters.
        
        Args:
            qubit: Qubit index.
            error_type: One of 'x', 'z', 'readout', 'gate'.
            
        Returns:
            The effective error rate for the specified error type.
        """
        pass
    
    def apply_gate_noise(
        self,
        circuit: stim.Circuit,
        gate_name: str,
        qubits: Sequence[int],
    ) -> None:
        """Apply noise after a gate operation (optional override).
        
        Default implementation is a no-op. Override in subclasses for
        circuit-level noise modeling.
        
        Args:
            circuit: Stim circuit to append noise operations to.
            gate_name: Name of the gate (e.g., 'CX', 'H', 'S').
            qubits: Qubits the gate acts on.
        """
        pass


class PhenomenologicalNoiseModel(NoiseModel):
    """Simple phenomenological noise model with uniform error rates.
    
    This wraps the existing behavior where all qubits have the same
    X and Z error probabilities, applied uniformly each round.
    
    Attributes:
        p_x: Probability of X error per qubit per round.
        p_z: Probability of Z error per qubit per round.
        p_readout: Optional uniform readout error rate.
    """
    
    def __init__(
        self,
        p_x: float = 0.0,
        p_z: float = 0.0,
        p_readout: float = 0.0,
    ):
        """Initialize phenomenological noise model.
        
        Args:
            p_x: X error probability per qubit per round.
            p_z: Z error probability per qubit per round.
            p_readout: Readout error probability (symmetric).
        """
        if not (0 <= p_x <= 1):
            raise ValueError(f"p_x must be in [0, 1], got {p_x}")
        if not (0 <= p_z <= 1):
            raise ValueError(f"p_z must be in [0, 1], got {p_z}")
        if not (0 <= p_readout <= 1):
            raise ValueError(f"p_readout must be in [0, 1], got {p_readout}")
        
        self.p_x = p_x
        self.p_z = p_z
        self.p_readout = p_readout
    
    def apply_data_qubit_noise(
        self,
        circuit: stim.Circuit,
        qubits: Sequence[int],
        duration: Optional[float] = None,
    ) -> None:
        """Apply uniform X and Z errors to all specified qubits."""
        qubit_list = list(qubits)
        if not qubit_list:
            return
        
        if self.p_x > 0:
            circuit.append_operation("X_ERROR", qubit_list, self.p_x)
        if self.p_z > 0:
            circuit.append_operation("Z_ERROR", qubit_list, self.p_z)
    
    def apply_measurement_noise(
        self,
        circuit: stim.Circuit,
        qubits: Sequence[int],
    ) -> None:
        """Apply uniform readout errors before measurement."""
        if self.p_readout <= 0:
            return
        qubit_list = list(qubits)
        if not qubit_list:
            return
        # X_ERROR before measurement flips the measurement outcome
        circuit.append_operation("X_ERROR", qubit_list, self.p_readout)
    
    def get_effective_error_rate(
        self,
        qubit: int,
        error_type: str,
    ) -> float:
        """Return uniform error rate (same for all qubits)."""
        error_type = error_type.lower()
        if error_type == 'x':
            return self.p_x
        elif error_type == 'z':
            return self.p_z
        elif error_type == 'readout':
            return self.p_readout
        elif error_type == 'gate':
            return 0.0  # No gate errors in phenomenological model
        else:
            raise ValueError(f"Unknown error type: {error_type}")


class DeviceAwareNoiseModel(NoiseModel):
    """Device-aware noise model with per-qubit calibration data.
    
    This model uses T1/T2 coherence times to compute X and Z error rates
    for each qubit, along with per-qubit readout errors and optional
    crosstalk effects.
    
    The conversion from T1/T2 to Pauli error rates follows:
    - Amplitude damping: p_ad = 1 - exp(-t/T1)
    - Pure dephasing: p_deph = 0.5 * (1 - exp(-t/T2_phi))
    - Pauli X rate: p_x ≈ p_ad / 4
    - Pauli Z rate: p_z ≈ p_deph / 2 + p_ad / 4
    
    Attributes:
        qubit_params: Dictionary mapping qubit index to QubitNoiseParams.
        coupler_params: Dictionary mapping (q1, q2) tuple to CouplerNoiseParams.
        default_round_duration: Default duration of a measurement round in µs.
        gate_times: Dictionary mapping gate names to durations in µs.
    """
    
    def __init__(
        self,
        qubit_params: Dict[int, QubitNoiseParams],
        coupler_params: Optional[Dict[Tuple[int, int], CouplerNoiseParams]] = None,
        default_round_duration: float = 1.0,  # µs
        gate_times: Optional[Dict[str, float]] = None,
    ):
        """Initialize device-aware noise model.
        
        Args:
            qubit_params: Per-qubit noise parameters.
            coupler_params: Per-coupler noise parameters (optional).
            default_round_duration: Default measurement round duration in µs.
            gate_times: Gate durations in µs (optional).
        """
        self.qubit_params = qubit_params
        self.coupler_params = coupler_params or {}
        self.default_round_duration = default_round_duration
        self.gate_times = gate_times or {
            "sx": 0.035,
            "x": 0.035,
            "h": 0.035,
            "s": 0.0,  # Virtual Z gates are instantaneous
            "cx": 0.3,
            "cz": 0.3,
            "measure": 1.0,
        }
    
    def _compute_pauli_rates(
        self,
        qubit: int,
        duration: float,
    ) -> Tuple[float, float]:
        """Compute X and Z error rates from T1/T2 decoherence.
        
        Args:
            qubit: Qubit index.
            duration: Time duration in microseconds.
            
        Returns:
            Tuple of (p_x, p_z) error probabilities.
        """
        if qubit not in self.qubit_params:
            # Unknown qubit: return zero errors
            return 0.0, 0.0
        
        params = self.qubit_params[qubit]
        t1 = params.t1
        t2 = params.t2
        
        # Amplitude damping probability
        p_ad = 1.0 - math.exp(-duration / t1)
        
        # Pure dephasing probability
        t2_phi = params.t2_phi
        if math.isinf(t2_phi):
            p_deph = 0.0
        else:
            p_deph = 0.5 * (1.0 - math.exp(-duration / t2_phi))
        
        # Convert to Pauli error rates
        # These approximations are valid for small error rates
        p_x = p_ad / 4.0
        p_z = p_deph / 2.0 + p_ad / 4.0
        
        # Add single-qubit gate error contribution (depolarizing)
        gate_err = params.single_qubit_gate_error
        p_x += gate_err / 3.0
        p_z += gate_err / 3.0
        
        # Clamp to valid probability range
        p_x = min(max(p_x, 0.0), 0.5)
        p_z = min(max(p_z, 0.0), 0.5)
        
        return p_x, p_z
    
    def apply_data_qubit_noise(
        self,
        circuit: stim.Circuit,
        qubits: Sequence[int],
        duration: Optional[float] = None,
    ) -> None:
        """Apply per-qubit decoherence noise based on T1/T2."""
        if duration is None:
            duration = self.default_round_duration
        
        for qubit in qubits:
            p_x, p_z = self._compute_pauli_rates(qubit, duration)
            
            if p_x > 0:
                circuit.append_operation("X_ERROR", [qubit], p_x)
            if p_z > 0:
                circuit.append_operation("Z_ERROR", [qubit], p_z)
    
    def apply_measurement_noise(
        self,
        circuit: stim.Circuit,
        qubits: Sequence[int],
    ) -> None:
        """Apply per-qubit readout errors."""
        for qubit in qubits:
            if qubit not in self.qubit_params:
                continue
            
            params = self.qubit_params[qubit]
            # Use average readout error for simplicity
            # (asymmetric errors would need more complex modeling)
            p_readout = params.readout_error
            
            if p_readout > 0:
                circuit.append_operation("X_ERROR", [qubit], p_readout)
    
    def get_effective_error_rate(
        self,
        qubit: int,
        error_type: str,
    ) -> float:
        """Get effective error rate for a specific qubit."""
        error_type = error_type.lower()
        
        if qubit not in self.qubit_params:
            return 0.0
        
        params = self.qubit_params[qubit]
        
        if error_type == 'x':
            p_x, _ = self._compute_pauli_rates(qubit, self.default_round_duration)
            return p_x
        elif error_type == 'z':
            _, p_z = self._compute_pauli_rates(qubit, self.default_round_duration)
            return p_z
        elif error_type == 'readout':
            return params.readout_error
        elif error_type == 'gate':
            return params.single_qubit_gate_error
        else:
            raise ValueError(f"Unknown error type: {error_type}")
    
    def apply_gate_noise(
        self,
        circuit: stim.Circuit,
        gate_name: str,
        qubits: Sequence[int],
    ) -> None:
        """Apply depolarizing noise after a gate."""
        gate_name = gate_name.lower()
        qubit_list = list(qubits)
        
        if len(qubit_list) == 1:
            # Single-qubit gate
            qubit = qubit_list[0]
            if qubit in self.qubit_params:
                p_err = self.qubit_params[qubit].single_qubit_gate_error
                if p_err > 0:
                    circuit.append_operation("DEPOLARIZE1", [qubit], p_err)
        
        elif len(qubit_list) == 2:
            # Two-qubit gate
            q1, q2 = qubit_list
            key = (min(q1, q2), max(q1, q2))
            if key in self.coupler_params:
                p_err = self.coupler_params[key].two_qubit_gate_error
                if p_err > 0:
                    circuit.append_operation("DEPOLARIZE2", qubit_list, p_err)
    
    def apply_crosstalk(
        self,
        circuit: stim.Circuit,
        active_qubits: Sequence[int],
        all_qubits: Sequence[int],
    ) -> None:
        """Apply crosstalk effects from active qubits to idle neighbors.
        
        This models ZZ crosstalk between coupled qubits when one is
        undergoing a gate and the other is idle.
        
        Args:
            circuit: Stim circuit to append noise to.
            active_qubits: Qubits currently undergoing gates.
            all_qubits: All qubits in the system.
        """
        active_set = set(active_qubits)
        idle_qubits = [q for q in all_qubits if q not in active_set]
        
        for (q1, q2), params in self.coupler_params.items():
            if params.crosstalk_strength <= 0:
                continue
            
            # Check if one qubit is active and one is idle
            if q1 in active_set and q2 in idle_qubits:
                # ZZ crosstalk appears as Z error on idle qubit
                circuit.append_operation("Z_ERROR", [q2], params.crosstalk_strength)
            elif q2 in active_set and q1 in idle_qubits:
                circuit.append_operation("Z_ERROR", [q1], params.crosstalk_strength)
