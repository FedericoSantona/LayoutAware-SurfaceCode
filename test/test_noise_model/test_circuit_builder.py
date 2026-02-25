"""Tests for the circuit-level Stim builder with explicit ancilla qubits."""
import pytest
import stim
import numpy as np

from surface_code import (
    CircuitLevelStimBuilder,
    CircuitLevelStimConfig,
    AncillaAllocator,
    PhenomenologicalNoiseModel,
    DeviceAwareNoiseModel,
    QubitNoiseParams,
    CouplerNoiseParams,
)


class TestAncillaAllocator:
    """Test the AncillaAllocator class."""
    
    def test_basic_allocation(self):
        """Test basic ancilla allocation with 9 data qubits, 4 Z stabs, 4 X stabs."""
        allocator = AncillaAllocator(n_data=9, n_z_stabs=4, n_x_stabs=4)
        
        assert allocator.n_total == 17  # 9 + 4 + 4
        assert allocator.z_ancilla_start == 9
        assert allocator.x_ancilla_start == 13
        
        # Check data qubits
        assert allocator.data_qubits() == list(range(9))
        
        # Check Z ancillas
        assert allocator.z_ancillas() == [9, 10, 11, 12]
        assert allocator.z_ancilla(0) == 9
        assert allocator.z_ancilla(3) == 12
        
        # Check X ancillas
        assert allocator.x_ancillas() == [13, 14, 15, 16]
        assert allocator.x_ancilla(0) == 13
        assert allocator.x_ancilla(3) == 16
    
    def test_index_bounds(self):
        """Test that out-of-bounds indices raise errors."""
        allocator = AncillaAllocator(n_data=9, n_z_stabs=4, n_x_stabs=4)
        
        with pytest.raises(IndexError):
            allocator.z_ancilla(4)  # Only 0-3 valid
        
        with pytest.raises(IndexError):
            allocator.x_ancilla(-1)
        
        with pytest.raises(IndexError):
            allocator.x_ancilla(4)
    
    def test_all_ancillas(self):
        """Test that all_ancillas returns both Z and X ancillas."""
        allocator = AncillaAllocator(n_data=5, n_z_stabs=2, n_x_stabs=3)
        
        all_anc = allocator.all_ancillas()
        assert len(all_anc) == 5  # 2 + 3
        assert all_anc == [5, 6, 7, 8, 9]


class MockCode:
    """Mock surface code object for testing."""
    
    def __init__(self, n: int):
        self.n = n


class TestCircuitLevelStimBuilder:
    """Test the CircuitLevelStimBuilder class."""
    
    @pytest.fixture
    def simple_code(self):
        """Create a simple mock code with 4 data qubits (distance 2)."""
        return MockCode(n=4)
    
    @pytest.fixture
    def simple_stabilizers(self):
        """Simple stabilizers for a 4-qubit code."""
        # Single Z stabilizer on qubits 0,1,2,3
        z_stabs = ["ZZZZ"]
        # Single X stabilizer on qubits 0,1,2,3
        x_stabs = ["XXXX"]
        return z_stabs, x_stabs
    
    def test_builder_initialization(self, simple_code, simple_stabilizers):
        """Test builder initializes correctly."""
        z_stabs, x_stabs = simple_stabilizers
        builder = CircuitLevelStimBuilder(
            code=simple_code,
            z_stabilizers=z_stabs,
            x_stabilizers=x_stabs,
            logical_z="ZZZZ",
            logical_x="XXXX",
        )
        
        assert builder.allocator.n_data == 4
        assert builder.allocator.n_z_stabs == 1
        assert builder.allocator.n_x_stabs == 1
        assert builder.allocator.n_total == 6  # 4 + 1 + 1
    
    def test_build_without_noise(self, simple_code, simple_stabilizers):
        """Test building a circuit without noise."""
        z_stabs, x_stabs = simple_stabilizers
        builder = CircuitLevelStimBuilder(
            code=simple_code,
            z_stabilizers=z_stabs,
            x_stabilizers=x_stabs,
            logical_z="ZZZZ",
            logical_x="XXXX",
        )
        
        config = CircuitLevelStimConfig(
            rounds=3,
            noise_model=None,
            init_label="0",
        )
        
        circuit, observable_pairs = builder.build(config)
        
        # Check circuit is valid
        assert isinstance(circuit, stim.Circuit)
        assert circuit.num_qubits == 6  # 4 data + 2 ancillas
        
        # Should have measurements (warmup + rounds)
        assert circuit.num_measurements > 0
        
        # Should have detectors
        assert circuit.num_detectors > 0
    
    def test_build_with_phenomenological_noise(self, simple_code, simple_stabilizers):
        """Test building with phenomenological noise model."""
        z_stabs, x_stabs = simple_stabilizers
        builder = CircuitLevelStimBuilder(
            code=simple_code,
            z_stabilizers=z_stabs,
            x_stabilizers=x_stabs,
            logical_z="ZZZZ",
            logical_x="XXXX",
        )
        
        noise_model = PhenomenologicalNoiseModel(
            p_x=0.001,
            p_z=0.001,
            p_readout=0.01,
            p_gate_1q=0.001,
            p_gate_2q=0.01,
        )
        
        config = CircuitLevelStimConfig(
            rounds=2,
            noise_model=noise_model,
            init_label="0",
        )
        
        circuit, observable_pairs = builder.build(config)
        
        # Check that circuit contains noise operations
        circuit_str = str(circuit)
        assert "DEPOLARIZE1" in circuit_str or "DEPOLARIZE2" in circuit_str
    
    def test_build_with_device_aware_noise(self, simple_code, simple_stabilizers):
        """Test building with device-aware noise model."""
        z_stabs, x_stabs = simple_stabilizers
        builder = CircuitLevelStimBuilder(
            code=simple_code,
            z_stabilizers=z_stabs,
            x_stabilizers=x_stabs,
            logical_z="ZZZZ",
            logical_x="XXXX",
        )
        
        # Create per-qubit noise params (only for data qubits, ancillas use defaults)
        qubit_params = {
            i: QubitNoiseParams(
                t1=100.0,
                t2=80.0,
                readout_error_0to1=0.01,
                readout_error_1to0=0.01,
                single_qubit_gate_error=0.001,
            )
            for i in range(4)
        }
        
        # Default params for ancillas
        default_qubit = QubitNoiseParams(
            t1=100.0,
            t2=80.0,
            readout_error_0to1=0.01,
            readout_error_1to0=0.01,
            single_qubit_gate_error=0.001,
        )
        
        noise_model = DeviceAwareNoiseModel(
            qubit_params=qubit_params,
            coupler_params={},
            default_qubit_params=default_qubit,
            default_coupler_params=CouplerNoiseParams(two_qubit_gate_error=0.01),
        )
        
        config = CircuitLevelStimConfig(
            rounds=2,
            noise_model=noise_model,
            init_label="0",
        )
        
        circuit, observable_pairs = builder.build(config)
        
        # Check circuit is valid
        assert isinstance(circuit, stim.Circuit)
        assert circuit.num_qubits == 6
        
        # Check that noise operations are present
        circuit_str = str(circuit)
        assert "X_ERROR" in circuit_str or "Z_ERROR" in circuit_str or "DEPOLARIZE" in circuit_str
    
    def test_cz_gate_option(self, simple_code, simple_stabilizers):
        """Test building with CZ gates instead of CX."""
        z_stabs, x_stabs = simple_stabilizers
        builder = CircuitLevelStimBuilder(
            code=simple_code,
            z_stabilizers=z_stabs,
            x_stabilizers=x_stabs,
            logical_z="ZZZZ",
            logical_x="XXXX",
        )
        
        config = CircuitLevelStimConfig(
            rounds=2,
            noise_model=None,
            two_qubit_gate="CZ",
            init_label="0",
        )
        
        circuit, _ = builder.build(config)
        circuit_str = str(circuit)
        
        # CZ should be present, not CX
        assert "CZ" in circuit_str
        # Note: For CZ-based syndrome extraction, we still use H gates
        assert "H" in circuit_str
    
    def test_z_only_family(self, simple_code, simple_stabilizers):
        """Test measuring only Z stabilizers."""
        z_stabs, x_stabs = simple_stabilizers
        builder = CircuitLevelStimBuilder(
            code=simple_code,
            z_stabilizers=z_stabs,
            x_stabilizers=x_stabs,
            logical_z="ZZZZ",
            logical_x="XXXX",
        )
        
        config = CircuitLevelStimConfig(
            rounds=2,
            noise_model=None,
            family="Z",
            init_label="0",
        )
        
        circuit, _ = builder.build(config)
        
        # Should still build successfully
        assert circuit.num_measurements > 0
    
    def test_x_only_family(self, simple_code, simple_stabilizers):
        """Test measuring only X stabilizers."""
        z_stabs, x_stabs = simple_stabilizers
        builder = CircuitLevelStimBuilder(
            code=simple_code,
            z_stabilizers=z_stabs,
            x_stabilizers=x_stabs,
            logical_z="ZZZZ",
            logical_x="XXXX",
        )
        
        config = CircuitLevelStimConfig(
            rounds=2,
            noise_model=None,
            family="X",
            init_label="+",
        )
        
        circuit, _ = builder.build(config)
        
        # Should still build successfully
        assert circuit.num_measurements > 0


class TestCircuitLevelConfig:
    """Test CircuitLevelStimConfig validation."""
    
    def test_valid_config(self):
        """Test creating a valid config."""
        config = CircuitLevelStimConfig(
            rounds=5,
            noise_model=None,
            two_qubit_gate="CX",
        )
        assert config.rounds == 5
        assert config.two_qubit_gate == "CX"
    
    def test_cz_option(self):
        """Test CZ gate option."""
        config = CircuitLevelStimConfig(two_qubit_gate="cz")
        assert config.two_qubit_gate == "CZ"  # Should be uppercased
    
    def test_invalid_gate(self):
        """Test that invalid gate type raises error."""
        with pytest.raises(ValueError):
            CircuitLevelStimConfig(two_qubit_gate="CNOT")


class TestNoiseModelGateNoise:
    """Test gate-level noise application in noise models."""
    
    def test_phenomenological_gate_noise(self):
        """Test phenomenological model applies gate noise correctly."""
        noise_model = PhenomenologicalNoiseModel(
            p_x=0.0,
            p_z=0.0,
            p_gate_1q=0.01,
            p_gate_2q=0.02,
        )
        
        circuit = stim.Circuit()
        
        # Apply single-qubit gate noise
        circuit.append_operation("H", [0])
        noise_model.apply_gate_noise(circuit, "H", [0])
        
        circuit_str = str(circuit)
        assert "DEPOLARIZE1(0.01)" in circuit_str
        
        # Apply two-qubit gate noise
        circuit2 = stim.Circuit()
        circuit2.append_operation("CX", [0, 1])
        noise_model.apply_gate_noise(circuit2, "CX", [0, 1])
        
        circuit2_str = str(circuit2)
        assert "DEPOLARIZE2(0.02)" in circuit2_str
    
    def test_device_aware_gate_noise(self):
        """Test device-aware model applies gate noise correctly."""
        qubit_params = {
            0: QubitNoiseParams(t1=100.0, t2=80.0, single_qubit_gate_error=0.001),
            1: QubitNoiseParams(t1=100.0, t2=80.0, single_qubit_gate_error=0.002),
        }
        coupler_params = {
            (0, 1): CouplerNoiseParams(two_qubit_gate_error=0.01),
        }
        
        noise_model = DeviceAwareNoiseModel(
            qubit_params=qubit_params,
            coupler_params=coupler_params,
        )
        
        # Apply single-qubit gate noise
        circuit = stim.Circuit()
        circuit.append_operation("H", [0])
        noise_model.apply_gate_noise(circuit, "H", [0])
        
        circuit_str = str(circuit)
        assert "DEPOLARIZE1(0.001)" in circuit_str
        
        # Apply two-qubit gate noise
        circuit2 = stim.Circuit()
        circuit2.append_operation("CX", [0, 1])
        noise_model.apply_gate_noise(circuit2, "CX", [0, 1])
        
        circuit2_str = str(circuit2)
        assert "DEPOLARIZE2(0.01)" in circuit2_str
    
    def test_device_aware_default_params(self):
        """Test device-aware model uses default params for unknown qubits."""
        qubit_params = {
            0: QubitNoiseParams(t1=100.0, t2=80.0, single_qubit_gate_error=0.001),
        }
        
        # Default params for unknown qubits (like ancillas)
        default_params = QubitNoiseParams(
            t1=100.0,
            t2=80.0,
            single_qubit_gate_error=0.005,
        )
        
        noise_model = DeviceAwareNoiseModel(
            qubit_params=qubit_params,
            default_qubit_params=default_params,
        )
        
        # Apply gate noise to unknown qubit (should use default)
        circuit = stim.Circuit()
        circuit.append_operation("H", [5])  # Qubit 5 not in qubit_params
        noise_model.apply_gate_noise(circuit, "H", [5])
        
        circuit_str = str(circuit)
        assert "DEPOLARIZE1(0.005)" in circuit_str


class TestIdleNoise:
    """Test idle noise application."""
    
    def test_idle_noise_delegates(self):
        """Test that apply_idle_noise delegates to apply_data_qubit_noise."""
        noise_model = PhenomenologicalNoiseModel(
            p_x=0.01,
            p_z=0.02,
        )
        
        circuit = stim.Circuit()
        noise_model.apply_idle_noise(circuit, [0, 1], duration=0.3)
        
        circuit_str = str(circuit)
        assert "X_ERROR(0.01)" in circuit_str
        assert "Z_ERROR(0.02)" in circuit_str
