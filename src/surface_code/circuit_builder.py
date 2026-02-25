"""Circuit-level Stim builder with explicit ancilla qubits and gate-by-gate noise.

This module provides a circuit-level alternative to the phenomenological builder,
using explicit syndrome extraction circuits with ancilla qubits and applying noise
after each gate operation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any, TYPE_CHECKING

import stim

if TYPE_CHECKING:
    from .noise_model import NoiseModel


@dataclass
class AncillaAllocator:
    """Allocates ancilla qubit indices for stabilizer measurements.
    
    Given the number of data qubits and stabilizer counts, this class
    assigns unique qubit indices for ancillas:
    
    Layout:
        Data qubits:     [0, 1, 2, ..., n_data-1]
        Z-ancillas:      [n_data, n_data+1, ..., n_data+n_z_stabs-1]
        X-ancillas:      [n_data+n_z_stabs, ..., n_data+n_z_stabs+n_x_stabs-1]
    
    Attributes:
        n_data: Number of data qubits.
        n_z_stabs: Number of Z-type stabilizers.
        n_x_stabs: Number of X-type stabilizers.
    """
    n_data: int
    n_z_stabs: int
    n_x_stabs: int
    
    @property
    def n_total(self) -> int:
        """Total number of qubits (data + ancillas)."""
        return self.n_data + self.n_z_stabs + self.n_x_stabs
    
    @property
    def z_ancilla_start(self) -> int:
        """Starting index for Z-type ancillas."""
        return self.n_data
    
    @property
    def x_ancilla_start(self) -> int:
        """Starting index for X-type ancillas."""
        return self.n_data + self.n_z_stabs
    
    def z_ancilla(self, stab_index: int) -> int:
        """Get the ancilla qubit index for the i-th Z stabilizer."""
        if not (0 <= stab_index < self.n_z_stabs):
            raise IndexError(f"Z stabilizer index {stab_index} out of range [0, {self.n_z_stabs})")
        return self.z_ancilla_start + stab_index
    
    def x_ancilla(self, stab_index: int) -> int:
        """Get the ancilla qubit index for the i-th X stabilizer."""
        if not (0 <= stab_index < self.n_x_stabs):
            raise IndexError(f"X stabilizer index {stab_index} out of range [0, {self.n_x_stabs})")
        return self.x_ancilla_start + stab_index
    
    def data_qubits(self) -> List[int]:
        """Return list of all data qubit indices."""
        return list(range(self.n_data))
    
    def z_ancillas(self) -> List[int]:
        """Return list of all Z-ancilla indices."""
        return list(range(self.z_ancilla_start, self.z_ancilla_start + self.n_z_stabs))
    
    def x_ancillas(self) -> List[int]:
        """Return list of all X-ancilla indices."""
        return list(range(self.x_ancilla_start, self.x_ancilla_start + self.n_x_stabs))
    
    def all_ancillas(self) -> List[int]:
        """Return list of all ancilla indices."""
        return self.z_ancillas() + self.x_ancillas()


@dataclass
class CircuitLevelStimConfig:
    """Configuration for circuit-level stabilizer measurement.
    
    Attributes:
        rounds: Number of syndrome measurement rounds.
        noise_model: NoiseModel instance for gate-level noise injection.
        two_qubit_gate: Type of two-qubit gate to use ('CX' or 'CZ').
        gate_times: Dictionary of gate durations in microseconds.
        apply_idle_noise: Whether to apply decoherence to idle qubits during gates.
        init_label: Logical state initialization ('0', '1', '+', '-').
        family: CSS family to measure (None=both, 'Z'=Z-only, 'X'=X-only).
    """
    rounds: int = 5
    noise_model: Optional["NoiseModel"] = None
    two_qubit_gate: str = "CX"  # 'CX' or 'CZ'
    gate_times: Dict[str, float] = field(default_factory=lambda: {
        "R": 0.0,      # Reset (instantaneous in simulation)
        "H": 0.035,    # Hadamard gate
        "CX": 0.3,     # CNOT gate
        "CZ": 0.3,     # CZ gate
        "M": 1.0,      # Measurement
    })
    apply_idle_noise: bool = True
    init_label: Optional[str] = None
    family: Optional[str] = None  # None, 'Z', or 'X'
    
    def __post_init__(self):
        if self.two_qubit_gate.upper() not in ("CX", "CZ"):
            raise ValueError(f"two_qubit_gate must be 'CX' or 'CZ', got {self.two_qubit_gate}")
        self.two_qubit_gate = self.two_qubit_gate.upper()


def _parse_stabilizer_support(pauli_str: str, pauli_type: str) -> List[int]:
    """Extract data qubit indices where the stabilizer has non-identity support.
    
    Args:
        pauli_str: Pauli string like "ZZII" or "IXXI".
        pauli_type: Expected Pauli type ('Z' or 'X').
        
    Returns:
        List of qubit indices where the stabilizer acts non-trivially.
    """
    support = []
    for i, char in enumerate(pauli_str):
        if char == pauli_type:
            support.append(i)
    return support


class CircuitLevelStimBuilder:
    """Build Stim circuits with explicit syndrome extraction circuits.
    
    This builder creates circuits where each stabilizer measurement uses:
    - A dedicated ancilla qubit
    - Explicit gate sequences (H, CX/CZ, M)
    - Gate-level noise injection
    
    This is more realistic than the phenomenological approach which uses
    MPP operations and applies noise per round rather than per gate.
    """
    
    def __init__(
        self,
        code,
        z_stabilizers: Sequence[str],
        x_stabilizers: Sequence[str],
        logical_z: Optional[str] = None,
        logical_x: Optional[str] = None,
    ) -> None:
        """Initialize the circuit-level builder.
        
        Args:
            code: Surface code object with attribute `n` (number of data qubits).
            z_stabilizers: List of Z-type stabilizer Pauli strings.
            x_stabilizers: List of X-type stabilizer Pauli strings.
            logical_z: Logical Z operator as Pauli string.
            logical_x: Logical X operator as Pauli string.
        """
        self.code = code
        self.z_stabilizers = list(z_stabilizers)
        self.x_stabilizers = list(x_stabilizers)
        self.logical_z = logical_z
        self.logical_x = logical_x
        
        # Pre-parse stabilizer supports
        self._z_supports = [_parse_stabilizer_support(s, 'Z') for s in self.z_stabilizers]
        self._x_supports = [_parse_stabilizer_support(s, 'X') for s in self.x_stabilizers]
        
        # Create ancilla allocator
        self.allocator = AncillaAllocator(
            n_data=code.n,
            n_z_stabs=len(self.z_stabilizers),
            n_x_stabs=len(self.x_stabilizers),
        )
        
        # Debug metadata for measurements
        self._meas_meta: Dict[int, Dict[str, Any]] = {}
    
    # ----- Helpers -----
    
    @staticmethod
    def _init_intent(label: str) -> Tuple[str, int]:
        """Parse initialization label to (basis, eigenvalue)."""
        if label == "0":
            return "Z", +1
        if label == "1":
            return "Z", -1
        if label == "+":
            return "X", +1
        if label == "-":
            return "X", -1
        raise ValueError("init label must be one of '0','1','+','-'")
    
    @staticmethod
    def _rec_from_abs(circuit: stim.Circuit, index: int) -> stim.GateTarget:
        """Convert absolute measurement index to relative record target."""
        return stim.target_rec(index - circuit.num_measurements)
    
    def _apply_gate_noise(
        self,
        circuit: stim.Circuit,
        config: CircuitLevelStimConfig,
        gate_name: str,
        qubits: List[int],
    ) -> None:
        """Apply noise after a gate operation."""
        if config.noise_model is not None:
            config.noise_model.apply_gate_noise(circuit, gate_name, qubits)
    
    def _apply_idle_noise(
        self,
        circuit: stim.Circuit,
        config: CircuitLevelStimConfig,
        idle_qubits: List[int],
        duration: float,
    ) -> None:
        """Apply decoherence noise to idle qubits during a gate."""
        if not config.apply_idle_noise or not idle_qubits:
            return
        if config.noise_model is not None:
            # Use apply_data_qubit_noise with the gate duration
            config.noise_model.apply_data_qubit_noise(circuit, idle_qubits, duration)
    
    # ----- Syndrome Extraction Circuits -----
    
    def _measure_z_stabilizer(
        self,
        circuit: stim.Circuit,
        config: CircuitLevelStimConfig,
        stab_index: int,
        all_qubits: List[int],
    ) -> int:
        """Build syndrome extraction circuit for a single Z stabilizer.
        
        For a Z stabilizer on qubits {q0, q1, q2, q3}:
            ancilla: R -- CX -- CX -- CX -- CX -- M
                          |     |     |     |
            data:    -----●-----●-----●-----●-----
        
        Returns the absolute measurement index.
        """
        ancilla = self.allocator.z_ancilla(stab_index)
        support = self._z_supports[stab_index]
        gate_time = config.gate_times.get(config.two_qubit_gate, 0.3)
        
        # Reset ancilla to |0⟩
        circuit.append_operation("R", [ancilla])
        
        # Apply CNOTs from data qubits to ancilla
        for data_q in support:
            if config.two_qubit_gate == "CX":
                # CX: data is control, ancilla is target (for Z stabilizer)
                circuit.append_operation("CX", [data_q, ancilla])
            else:  # CZ
                # For CZ, we need H on ancilla before and after
                circuit.append_operation("H", [ancilla])
                self._apply_gate_noise(circuit, config, "H", [ancilla])
                circuit.append_operation("CZ", [data_q, ancilla])
                circuit.append_operation("H", [ancilla])
                self._apply_gate_noise(circuit, config, "H", [ancilla])
            
            # Apply two-qubit gate noise
            self._apply_gate_noise(circuit, config, config.two_qubit_gate, [data_q, ancilla])
            
            # Apply idle noise to other qubits
            idle = [q for q in all_qubits if q not in (data_q, ancilla)]
            self._apply_idle_noise(circuit, config, idle, gate_time)
        
        # Measure ancilla
        circuit.append_operation("M", [ancilla])
        return circuit.num_measurements - 1
    
    def _measure_x_stabilizer(
        self,
        circuit: stim.Circuit,
        config: CircuitLevelStimConfig,
        stab_index: int,
        all_qubits: List[int],
    ) -> int:
        """Build syndrome extraction circuit for a single X stabilizer.
        
        For an X stabilizer on qubits {q0, q1, q2, q3}:
            ancilla: R -- H -- CX -- CX -- CX -- CX -- H -- M
                               |     |     |     |
            data:    ----------●-----●-----●-----●---------
        
        Returns the absolute measurement index.
        """
        ancilla = self.allocator.x_ancilla(stab_index)
        support = self._x_supports[stab_index]
        gate_time = config.gate_times.get(config.two_qubit_gate, 0.3)
        h_time = config.gate_times.get("H", 0.035)
        
        # Reset ancilla to |0⟩
        circuit.append_operation("R", [ancilla])
        
        # Hadamard to prepare |+⟩
        circuit.append_operation("H", [ancilla])
        self._apply_gate_noise(circuit, config, "H", [ancilla])
        idle = [q for q in all_qubits if q != ancilla]
        self._apply_idle_noise(circuit, config, idle, h_time)
        
        # Apply CNOTs from ancilla to data qubits
        for data_q in support:
            if config.two_qubit_gate == "CX":
                # CX: ancilla is control, data is target (for X stabilizer)
                circuit.append_operation("CX", [ancilla, data_q])
            else:  # CZ
                # CZ is symmetric, but for X stabilizer we add H on data
                circuit.append_operation("H", [data_q])
                self._apply_gate_noise(circuit, config, "H", [data_q])
                circuit.append_operation("CZ", [ancilla, data_q])
                circuit.append_operation("H", [data_q])
                self._apply_gate_noise(circuit, config, "H", [data_q])
            
            # Apply two-qubit gate noise
            self._apply_gate_noise(circuit, config, config.two_qubit_gate, [ancilla, data_q])
            
            # Apply idle noise to other qubits
            idle = [q for q in all_qubits if q not in (data_q, ancilla)]
            self._apply_idle_noise(circuit, config, idle, gate_time)
        
        # Hadamard before measurement
        circuit.append_operation("H", [ancilla])
        self._apply_gate_noise(circuit, config, "H", [ancilla])
        idle = [q for q in all_qubits if q != ancilla]
        self._apply_idle_noise(circuit, config, idle, h_time)
        
        # Measure ancilla
        circuit.append_operation("M", [ancilla])
        return circuit.num_measurements - 1
    
    def _measure_all_z_stabilizers(
        self,
        circuit: stim.Circuit,
        config: CircuitLevelStimConfig,
        round_index: int,
        phase_name: str = "unknown",
    ) -> List[int]:
        """Measure all Z stabilizers and return their measurement indices."""
        all_qubits = list(range(self.allocator.n_total))
        indices = []
        
        for stab_index in range(len(self.z_stabilizers)):
            idx = self._measure_z_stabilizer(circuit, config, stab_index, all_qubits)
            indices.append(idx)
            
            # Record metadata
            self._meas_meta[idx] = {
                "family": "Z",
                "round": round_index,
                "stab_index": stab_index,
                "pauli": self.z_stabilizers[stab_index],
                "phase": phase_name,
            }
        
        return indices
    
    def _measure_all_x_stabilizers(
        self,
        circuit: stim.Circuit,
        config: CircuitLevelStimConfig,
        round_index: int,
        phase_name: str = "unknown",
    ) -> List[int]:
        """Measure all X stabilizers and return their measurement indices."""
        all_qubits = list(range(self.allocator.n_total))
        indices = []
        
        for stab_index in range(len(self.x_stabilizers)):
            idx = self._measure_x_stabilizer(circuit, config, stab_index, all_qubits)
            indices.append(idx)
            
            # Record metadata
            self._meas_meta[idx] = {
                "family": "X",
                "round": round_index,
                "stab_index": stab_index,
                "pauli": self.x_stabilizers[stab_index],
                "phase": phase_name,
            }
        
        return indices
    
    # ----- Detectors -----
    
    def _add_detectors(
        self,
        circuit: stim.Circuit,
        prev: Sequence[int],
        curr: Sequence[int],
    ) -> None:
        """Add time-like detectors comparing consecutive measurement rounds."""
        for curr_idx, prev_idx in zip(curr, prev):
            circuit.append_operation(
                "DETECTOR",
                [self._rec_from_abs(circuit, prev_idx), self._rec_from_abs(circuit, curr_idx)],
            )
    
    # ----- Logical Measurement -----
    
    def _mpp_from_string(self, circuit: stim.Circuit, pauli_str: str) -> Optional[int]:
        """Measure a Pauli string using MPP (for logical operators)."""
        targets: List[stim.GateTarget] = []
        first = True
        for qubit, char in enumerate(pauli_str):
            if char == "I":
                continue
            if not first:
                targets.append(stim.target_combiner())
            if char == "X":
                targets.append(stim.target_x(qubit))
            elif char == "Z":
                targets.append(stim.target_z(qubit))
            elif char == "Y":
                targets.append(stim.target_y(qubit))
            first = False
        if not targets:
            return None
        circuit.append_operation("MPP", targets)
        return circuit.num_measurements - 1
    
    def measure_logical_once(
        self,
        circuit: stim.Circuit,
        logical_str: Optional[str],
    ) -> Optional[int]:
        """Measure a logical operator once and return its absolute index."""
        if logical_str is None:
            return None
        circuit.append_operation("TICK")
        return self._mpp_from_string(circuit, logical_str)
    
    def attach_observable_pair(
        self,
        circuit: stim.Circuit,
        start_idx: Optional[int],
        end_idx: Optional[int],
        observable_index: int,
        observable_pairs: List[Tuple[int, int]],
    ) -> None:
        """Wire two measurements into an OBSERVABLE and record the pair."""
        if start_idx is None or end_idx is None:
            return
        circuit.append_operation(
            "OBSERVABLE_INCLUDE",
            [
                self._rec_from_abs(circuit, start_idx),
                self._rec_from_abs(circuit, end_idx),
            ],
            observable_index,
        )
        observable_pairs.append((start_idx, end_idx))
    
    # ----- Main Build Method -----
    
    def build(
        self,
        config: CircuitLevelStimConfig,
    ) -> Tuple[stim.Circuit, List[Tuple[int, int]]]:
        """Build a complete circuit-level Stim circuit.
        
        Args:
            config: Circuit-level configuration.
            
        Returns:
            Tuple of (stim.Circuit, observable_pairs).
        """
        circuit = stim.Circuit()
        
        # Validate family configuration
        fam = (config.family or "").upper()
        if fam not in {"", "Z", "X"}:
            raise ValueError("config.family must be one of None, 'Z', or 'X'")
        measure_Z = fam in {"", "Z"}
        measure_X = fam in {"", "X"}
        
        # Determine logical operator to track
        logical_string = None
        if config.init_label is not None:
            basis, _ = self._init_intent(config.init_label.strip())
            if basis == "Z":
                if self.logical_z is None:
                    raise ValueError("Z logical operator required for Z-basis initialization")
                logical_string = self.logical_z
            else:
                if self.logical_x is None:
                    raise ValueError("X logical operator required for X-basis initialization")
                logical_string = self.logical_x
        
        # Set up qubit coordinates
        for q in range(self.allocator.n_total):
            if q < self.allocator.n_data:
                # Data qubit
                circuit.append_operation("QUBIT_COORDS", [q], [q, 0])
            elif q < self.allocator.x_ancilla_start:
                # Z ancilla
                circuit.append_operation("QUBIT_COORDS", [q], [q, 1])
            else:
                # X ancilla
                circuit.append_operation("QUBIT_COORDS", [q], [q, 2])
        
        observable_pairs: List[Tuple[int, int]] = []
        
        # Initial logical measurement
        start: Optional[int] = self.measure_logical_once(circuit, logical_string)
        
        # Initialize previous measurement indices
        sz_prev: Optional[List[int]] = None
        sx_prev: Optional[List[int]] = None
        
        # Warmup round (reference measurements without detectors)
        if measure_Z and self.z_stabilizers:
            circuit.append_operation("TICK")
            sz_prev = self._measure_all_z_stabilizers(circuit, config, round_index=-1)
        
        if measure_X and self.x_stabilizers:
            circuit.append_operation("TICK")
            sx_prev = self._measure_all_x_stabilizers(circuit, config, round_index=-1)
        
        # Main measurement rounds
        for round_idx in range(config.rounds):
            if measure_Z and self.z_stabilizers:
                circuit.append_operation("TICK")
                sz_curr = self._measure_all_z_stabilizers(circuit, config, round_index=round_idx)
                if sz_prev is not None:
                    self._add_detectors(circuit, sz_prev, sz_curr)
                sz_prev = sz_curr
            
            if measure_X and self.x_stabilizers:
                circuit.append_operation("TICK")
                sx_curr = self._measure_all_x_stabilizers(circuit, config, round_index=round_idx)
                if sx_prev is not None:
                    self._add_detectors(circuit, sx_prev, sx_curr)
                sx_prev = sx_curr
        
        # Final logical measurement
        end: Optional[int] = self.measure_logical_once(circuit, logical_string)
        self.attach_observable_pair(
            circuit,
            start_idx=start,
            end_idx=end,
            observable_index=0,
            observable_pairs=observable_pairs,
        )
        
        return circuit, observable_pairs
