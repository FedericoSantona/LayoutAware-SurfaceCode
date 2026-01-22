"""Unit tests for noise model classes."""
from __future__ import annotations

import math
import pytest
import stim

import sys
from pathlib import Path

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from surface_code.noise_model import (
    QubitNoiseParams,
    CouplerNoiseParams,
    NoiseModel,
    PhenomenologicalNoiseModel,
    DeviceAwareNoiseModel,
)


class TestQubitNoiseParams:
    """Tests for QubitNoiseParams dataclass."""
    
    def test_valid_params(self):
        """Test creating valid qubit parameters."""
        params = QubitNoiseParams(
            t1=100.0,
            t2=80.0,
            readout_error_0to1=0.02,
            readout_error_1to0=0.03,
            single_qubit_gate_error=0.001,
            frequency=5.0,
        )
        assert params.t1 == 100.0
        assert params.t2 == 80.0
        assert params.readout_error_0to1 == 0.02
        assert params.readout_error_1to0 == 0.03
        assert params.single_qubit_gate_error == 0.001
        assert params.frequency == 5.0
    
    def test_average_readout_error(self):
        """Test readout_error property computes average."""
        params = QubitNoiseParams(t1=100.0, t2=80.0, readout_error_0to1=0.02, readout_error_1to0=0.04)
        assert params.readout_error == 0.03
    
    def test_t2_phi_calculation(self):
        """Test pure dephasing time calculation."""
        # T2 = 80, T1 = 100 -> 1/T2_phi = 1/80 - 1/200 = 0.0125 - 0.005 = 0.0075
        # T2_phi = 1/0.0075 = 133.33...
        params = QubitNoiseParams(t1=100.0, t2=80.0)
        expected = 1.0 / (1.0/80.0 - 1.0/(2*100.0))
        assert abs(params.t2_phi - expected) < 1e-6
    
    def test_t2_phi_at_limit(self):
        """Test T2_phi when T2 = 2*T1 (pure dephasing vanishes)."""
        params = QubitNoiseParams(t1=100.0, t2=200.0)
        assert math.isinf(params.t2_phi)
    
    def test_invalid_t1_raises(self):
        """Test that non-positive T1 raises ValueError."""
        with pytest.raises(ValueError, match="T1 must be positive"):
            QubitNoiseParams(t1=0.0, t2=50.0)
        
        with pytest.raises(ValueError, match="T1 must be positive"):
            QubitNoiseParams(t1=-10.0, t2=50.0)
    
    def test_invalid_t2_raises(self):
        """Test that non-positive T2 raises ValueError."""
        with pytest.raises(ValueError, match="T2 must be positive"):
            QubitNoiseParams(t1=100.0, t2=0.0)
    
    def test_t2_greater_than_2t1_raises(self):
        """Test that T2 > 2*T1 raises ValueError (physical constraint)."""
        with pytest.raises(ValueError, match="T2.*cannot exceed 2\\*T1"):
            QubitNoiseParams(t1=100.0, t2=201.0)


class TestCouplerNoiseParams:
    """Tests for CouplerNoiseParams dataclass."""
    
    def test_default_values(self):
        """Test default coupler parameters."""
        params = CouplerNoiseParams()
        assert params.two_qubit_gate_error == 0.0
        assert params.crosstalk_strength == 0.0
    
    def test_custom_values(self):
        """Test custom coupler parameters."""
        params = CouplerNoiseParams(two_qubit_gate_error=0.01, crosstalk_strength=0.001)
        assert params.two_qubit_gate_error == 0.01
        assert params.crosstalk_strength == 0.001


class TestPhenomenologicalNoiseModel:
    """Tests for PhenomenologicalNoiseModel."""
    
    def test_creation(self):
        """Test creating a phenomenological noise model."""
        model = PhenomenologicalNoiseModel(p_x=0.01, p_z=0.02, p_readout=0.03)
        assert model.p_x == 0.01
        assert model.p_z == 0.02
        assert model.p_readout == 0.03
    
    def test_default_values(self):
        """Test default values are zero."""
        model = PhenomenologicalNoiseModel()
        assert model.p_x == 0.0
        assert model.p_z == 0.0
        assert model.p_readout == 0.0
    
    def test_invalid_probability_raises(self):
        """Test that invalid probabilities raise ValueError."""
        with pytest.raises(ValueError):
            PhenomenologicalNoiseModel(p_x=-0.1)
        
        with pytest.raises(ValueError):
            PhenomenologicalNoiseModel(p_z=1.5)
        
        with pytest.raises(ValueError):
            PhenomenologicalNoiseModel(p_readout=-0.01)
    
    def test_apply_data_qubit_noise(self):
        """Test that noise is applied to circuit."""
        model = PhenomenologicalNoiseModel(p_x=0.01, p_z=0.02)
        circuit = stim.Circuit()
        
        model.apply_data_qubit_noise(circuit, [0, 1, 2])
        
        # Should have X_ERROR and Z_ERROR operations
        circuit_str = str(circuit)
        assert "X_ERROR(0.01)" in circuit_str
        assert "Z_ERROR(0.02)" in circuit_str
    
    def test_apply_no_noise_when_zero(self):
        """Test that no noise is applied when rates are zero."""
        model = PhenomenologicalNoiseModel(p_x=0.0, p_z=0.0)
        circuit = stim.Circuit()
        
        model.apply_data_qubit_noise(circuit, [0, 1, 2])
        
        # Circuit should be empty
        assert len(circuit) == 0
    
    def test_apply_measurement_noise(self):
        """Test measurement noise application."""
        model = PhenomenologicalNoiseModel(p_readout=0.05)
        circuit = stim.Circuit()
        
        model.apply_measurement_noise(circuit, [0, 1])
        
        circuit_str = str(circuit)
        assert "X_ERROR(0.05)" in circuit_str
    
    def test_get_effective_error_rate(self):
        """Test retrieving effective error rates."""
        model = PhenomenologicalNoiseModel(p_x=0.01, p_z=0.02, p_readout=0.03)
        
        assert model.get_effective_error_rate(0, "x") == 0.01
        assert model.get_effective_error_rate(0, "z") == 0.02
        assert model.get_effective_error_rate(0, "readout") == 0.03
        assert model.get_effective_error_rate(0, "gate") == 0.0
        
        # Should work for any qubit index (uniform rates)
        assert model.get_effective_error_rate(99, "x") == 0.01
    
    def test_get_effective_error_rate_invalid_type(self):
        """Test that invalid error type raises ValueError."""
        model = PhenomenologicalNoiseModel()
        with pytest.raises(ValueError, match="Unknown error type"):
            model.get_effective_error_rate(0, "invalid")


class TestDeviceAwareNoiseModel:
    """Tests for DeviceAwareNoiseModel."""
    
    @pytest.fixture
    def sample_qubit_params(self):
        """Create sample qubit parameters for testing."""
        return {
            0: QubitNoiseParams(t1=100.0, t2=80.0, readout_error_0to1=0.01, readout_error_1to0=0.02, single_qubit_gate_error=0.001),
            1: QubitNoiseParams(t1=150.0, t2=120.0, readout_error_0to1=0.015, readout_error_1to0=0.015, single_qubit_gate_error=0.0008),
            2: QubitNoiseParams(t1=80.0, t2=60.0, readout_error_0to1=0.025, readout_error_1to0=0.025, single_qubit_gate_error=0.002),
        }
    
    @pytest.fixture
    def sample_coupler_params(self):
        """Create sample coupler parameters for testing."""
        return {
            (0, 1): CouplerNoiseParams(two_qubit_gate_error=0.01, crosstalk_strength=0.001),
            (1, 2): CouplerNoiseParams(two_qubit_gate_error=0.012),
        }
    
    def test_creation(self, sample_qubit_params, sample_coupler_params):
        """Test creating a device-aware noise model."""
        model = DeviceAwareNoiseModel(
            qubit_params=sample_qubit_params,
            coupler_params=sample_coupler_params,
            default_round_duration=1.5,
        )
        assert len(model.qubit_params) == 3
        assert len(model.coupler_params) == 2
        assert model.default_round_duration == 1.5
    
    def test_compute_pauli_rates(self, sample_qubit_params):
        """Test Pauli rate computation from T1/T2."""
        model = DeviceAwareNoiseModel(qubit_params=sample_qubit_params)
        
        # Compute rates for qubit 0 with duration 1.0 µs
        p_x, p_z = model._compute_pauli_rates(0, 1.0)
        
        # Should be small positive values
        assert 0 < p_x < 0.1
        assert 0 < p_z < 0.1
        
        # Z error should generally be >= X error due to dephasing contribution
        # (This depends on exact T1/T2 values, but typically true)
        assert p_z >= p_x
    
    def test_compute_pauli_rates_unknown_qubit(self, sample_qubit_params):
        """Test that unknown qubit returns zero error rates."""
        model = DeviceAwareNoiseModel(qubit_params=sample_qubit_params)
        
        p_x, p_z = model._compute_pauli_rates(99, 1.0)
        
        assert p_x == 0.0
        assert p_z == 0.0
    
    def test_longer_duration_higher_error(self, sample_qubit_params):
        """Test that longer duration gives higher error rates."""
        model = DeviceAwareNoiseModel(qubit_params=sample_qubit_params)
        
        p_x_short, p_z_short = model._compute_pauli_rates(0, 0.1)
        p_x_long, p_z_long = model._compute_pauli_rates(0, 10.0)
        
        assert p_x_long > p_x_short
        assert p_z_long > p_z_short
    
    def test_apply_data_qubit_noise(self, sample_qubit_params):
        """Test applying noise with per-qubit rates."""
        model = DeviceAwareNoiseModel(qubit_params=sample_qubit_params)
        circuit = stim.Circuit()
        
        model.apply_data_qubit_noise(circuit, [0, 1, 2])
        
        # Should have individual X_ERROR and Z_ERROR for each qubit
        circuit_str = str(circuit)
        assert "X_ERROR" in circuit_str
        assert "Z_ERROR" in circuit_str
    
    def test_apply_measurement_noise(self, sample_qubit_params):
        """Test applying per-qubit readout errors."""
        model = DeviceAwareNoiseModel(qubit_params=sample_qubit_params)
        circuit = stim.Circuit()
        
        model.apply_measurement_noise(circuit, [0, 1])
        
        circuit_str = str(circuit)
        assert "X_ERROR" in circuit_str
    
    def test_get_effective_error_rate(self, sample_qubit_params):
        """Test retrieving effective error rates for specific qubits."""
        model = DeviceAwareNoiseModel(qubit_params=sample_qubit_params)
        
        # Different qubits should have different rates
        rate_q0 = model.get_effective_error_rate(0, "x")
        rate_q1 = model.get_effective_error_rate(1, "x")
        
        # Qubit 1 has longer T1, should have lower error rate
        assert rate_q1 < rate_q0
        
        # Readout error should match the params
        assert model.get_effective_error_rate(0, "readout") == 0.015  # (0.01 + 0.02) / 2
    
    def test_apply_gate_noise_single_qubit(self, sample_qubit_params):
        """Test single-qubit gate noise application."""
        model = DeviceAwareNoiseModel(qubit_params=sample_qubit_params)
        circuit = stim.Circuit()
        
        model.apply_gate_noise(circuit, "x", [0])
        
        circuit_str = str(circuit)
        assert "DEPOLARIZE1" in circuit_str
    
    def test_apply_gate_noise_two_qubit(self, sample_qubit_params, sample_coupler_params):
        """Test two-qubit gate noise application."""
        model = DeviceAwareNoiseModel(
            qubit_params=sample_qubit_params,
            coupler_params=sample_coupler_params,
        )
        circuit = stim.Circuit()
        
        model.apply_gate_noise(circuit, "cx", [0, 1])
        
        circuit_str = str(circuit)
        assert "DEPOLARIZE2" in circuit_str
    
    def test_apply_crosstalk(self, sample_qubit_params, sample_coupler_params):
        """Test crosstalk application."""
        model = DeviceAwareNoiseModel(
            qubit_params=sample_qubit_params,
            coupler_params=sample_coupler_params,
        )
        circuit = stim.Circuit()
        
        # Qubit 0 is active, qubits 1 and 2 are idle
        # Should apply crosstalk Z error on qubit 1 (coupled to 0)
        model.apply_crosstalk(circuit, active_qubits=[0], all_qubits=[0, 1, 2])
        
        circuit_str = str(circuit)
        # Crosstalk on qubit 1 (neighbor of active qubit 0)
        assert "Z_ERROR" in circuit_str


class TestNoiseModelInterface:
    """Tests for NoiseModel ABC interface compliance."""
    
    def test_phenomenological_is_noise_model(self):
        """Test that PhenomenologicalNoiseModel is a NoiseModel."""
        model = PhenomenologicalNoiseModel()
        assert isinstance(model, NoiseModel)
    
    def test_device_aware_is_noise_model(self):
        """Test that DeviceAwareNoiseModel is a NoiseModel."""
        model = DeviceAwareNoiseModel(qubit_params={})
        assert isinstance(model, NoiseModel)
