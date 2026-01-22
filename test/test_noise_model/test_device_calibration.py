"""Unit tests for device calibration loader."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import sys

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from surface_code.device_calibration import DeviceCalibration
from surface_code.noise_model import (
    QubitNoiseParams,
    CouplerNoiseParams,
    DeviceAwareNoiseModel,
)


class TestDeviceCalibration:
    """Tests for DeviceCalibration class."""
    
    @pytest.fixture
    def sample_calibration(self):
        """Create a sample DeviceCalibration for testing."""
        qubit_params = {
            0: QubitNoiseParams(t1=100.0, t2=80.0, readout_error_0to1=0.02, readout_error_1to0=0.02, single_qubit_gate_error=0.001),
            1: QubitNoiseParams(t1=120.0, t2=90.0, readout_error_0to1=0.015, readout_error_1to0=0.015, single_qubit_gate_error=0.0008),
            2: QubitNoiseParams(t1=90.0, t2=70.0, readout_error_0to1=0.025, readout_error_1to0=0.025, single_qubit_gate_error=0.0012),
        }
        coupler_params = {
            (0, 1): CouplerNoiseParams(two_qubit_gate_error=0.01, crosstalk_strength=0.001),
            (1, 2): CouplerNoiseParams(two_qubit_gate_error=0.012),
        }
        return DeviceCalibration(
            backend_name="test_backend",
            timestamp="2026-01-22T10:00:00Z",
            qubit_params=qubit_params,
            coupler_params=coupler_params,
            gate_times={"sx": 0.04, "cx": 0.35},
            metadata={"test": True},
        )
    
    def test_creation(self, sample_calibration):
        """Test creating a DeviceCalibration."""
        assert sample_calibration.backend_name == "test_backend"
        assert sample_calibration.num_qubits == 3
        assert len(sample_calibration.coupler_params) == 2
    
    def test_num_qubits(self, sample_calibration):
        """Test num_qubits property."""
        assert sample_calibration.num_qubits == 3
    
    def test_qubit_indices(self, sample_calibration):
        """Test qubit_indices property returns sorted list."""
        assert sample_calibration.qubit_indices == [0, 1, 2]
    
    def test_default_gate_times(self):
        """Test that default gate times are set."""
        cal = DeviceCalibration(
            backend_name="test",
            timestamp="2026-01-22",
            qubit_params={},
        )
        # Should have default gate times
        assert "sx" in cal.gate_times
        assert "cx" in cal.gate_times
        assert "measure" in cal.gate_times
    
    def test_to_noise_model(self, sample_calibration):
        """Test converting to DeviceAwareNoiseModel."""
        noise_model = sample_calibration.to_noise_model(default_round_duration=1.5)
        
        assert isinstance(noise_model, DeviceAwareNoiseModel)
        assert noise_model.default_round_duration == 1.5
        assert len(noise_model.qubit_params) == 3
        assert len(noise_model.coupler_params) == 2
    
    def test_to_dict(self, sample_calibration):
        """Test converting to dictionary."""
        data = sample_calibration.to_dict()
        
        assert data["backend_name"] == "test_backend"
        assert "qubits" in data
        assert "couplers" in data
        assert len(data["qubits"]) == 3
        assert "0" in data["qubits"]
        assert data["qubits"]["0"]["t1"] == 100.0
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "backend_name": "from_dict_test",
            "timestamp": "2026-01-22T12:00:00Z",
            "qubits": {
                "0": {"t1": 100.0, "t2": 80.0, "readout_error": 0.02, "gate_error": 0.001},
                "1": {"t1": 120.0, "t2": 90.0, "readout_error": 0.015, "gate_error": 0.0008},
            },
            "couplers": {
                "0-1": {"cx_error": 0.01, "crosstalk": 0.001},
            },
            "gate_times": {"sx": 0.035},
        }
        
        cal = DeviceCalibration.from_dict(data)
        
        assert cal.backend_name == "from_dict_test"
        assert cal.num_qubits == 2
        assert 0 in cal.qubit_params
        assert 1 in cal.qubit_params
        assert (0, 1) in cal.coupler_params
    
    def test_roundtrip_to_dict_from_dict(self, sample_calibration):
        """Test that to_dict/from_dict roundtrip preserves data."""
        data = sample_calibration.to_dict()
        restored = DeviceCalibration.from_dict(data)
        
        assert restored.backend_name == sample_calibration.backend_name
        assert restored.num_qubits == sample_calibration.num_qubits
        assert restored.qubit_params[0].t1 == sample_calibration.qubit_params[0].t1
        assert restored.qubit_params[0].t2 == sample_calibration.qubit_params[0].t2


class TestDeviceCalibrationJSON:
    """Tests for JSON serialization/deserialization."""
    
    @pytest.fixture
    def sample_json_data(self):
        """Sample JSON calibration data."""
        return {
            "backend_name": "ibm_test",
            "timestamp": "2026-01-22T14:00:00Z",
            "qubits": {
                "0": {"t1": 150.0, "t2": 100.0, "readout_error_0to1": 0.01, "readout_error_1to0": 0.02, "gate_error": 0.0005},
                "1": {"t1": 140.0, "t2": 95.0, "readout_error_0to1": 0.015, "readout_error_1to0": 0.015, "gate_error": 0.0006},
            },
            "couplers": {
                "0-1": {"cx_error": 0.008},
            },
            "gate_times": {"sx": 0.035, "x": 0.035, "cx": 0.3},
            "metadata": {"calibration_version": "1.0"},
        }
    
    def test_to_json(self, sample_json_data):
        """Test saving calibration to JSON file."""
        cal = DeviceCalibration.from_dict(sample_json_data)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            cal.to_json(temp_path)
            
            # Verify file was created and is valid JSON
            assert temp_path.exists()
            with open(temp_path) as f:
                loaded = json.load(f)
            assert loaded["backend_name"] == "ibm_test"
        finally:
            temp_path.unlink()
    
    def test_from_json(self, sample_json_data):
        """Test loading calibration from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_json_data, f)
            temp_path = Path(f.name)
        
        try:
            cal = DeviceCalibration.from_json(temp_path)
            
            assert cal.backend_name == "ibm_test"
            assert cal.num_qubits == 2
            assert cal.qubit_params[0].t1 == 150.0
            assert cal.qubit_params[1].t2 == 95.0
            assert (0, 1) in cal.coupler_params
        finally:
            temp_path.unlink()
    
    def test_roundtrip_json(self, sample_json_data):
        """Test JSON roundtrip preserves all data."""
        original = DeviceCalibration.from_dict(sample_json_data)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            original.to_json(temp_path)
            restored = DeviceCalibration.from_json(temp_path)
            
            assert restored.backend_name == original.backend_name
            assert restored.timestamp == original.timestamp
            assert restored.num_qubits == original.num_qubits
            
            for q in original.qubit_indices:
                assert restored.qubit_params[q].t1 == original.qubit_params[q].t1
                assert restored.qubit_params[q].t2 == original.qubit_params[q].t2
        finally:
            temp_path.unlink()


class TestDeviceCalibrationUniform:
    """Tests for uniform calibration factory method."""
    
    def test_uniform_creation(self):
        """Test creating uniform calibration."""
        cal = DeviceCalibration.uniform(
            num_qubits=5,
            t1=100.0,
            t2=80.0,
            readout_error=0.02,
            gate_error_1q=0.001,
            gate_error_2q=0.01,
        )
        
        assert cal.backend_name == "uniform_synthetic"
        assert cal.num_qubits == 5
        
        # All qubits should have same parameters
        for q in range(5):
            assert cal.qubit_params[q].t1 == 100.0
            assert cal.qubit_params[q].t2 == 80.0
            assert cal.qubit_params[q].readout_error == 0.02
    
    def test_uniform_all_to_all_connectivity(self):
        """Test uniform creates all-to-all connectivity by default."""
        cal = DeviceCalibration.uniform(num_qubits=4)
        
        # Should have C(4,2) = 6 couplers
        assert len(cal.coupler_params) == 6
        assert (0, 1) in cal.coupler_params
        assert (0, 2) in cal.coupler_params
        assert (0, 3) in cal.coupler_params
        assert (1, 2) in cal.coupler_params
        assert (1, 3) in cal.coupler_params
        assert (2, 3) in cal.coupler_params
    
    def test_uniform_custom_connectivity(self):
        """Test uniform with custom connectivity."""
        connectivity = [(0, 1), (1, 2), (2, 3)]  # Linear chain
        cal = DeviceCalibration.uniform(
            num_qubits=4,
            connectivity=connectivity,
        )
        
        assert len(cal.coupler_params) == 3
        assert (0, 1) in cal.coupler_params
        assert (1, 2) in cal.coupler_params
        assert (2, 3) in cal.coupler_params
        assert (0, 2) not in cal.coupler_params


class TestDeviceCalibrationSummary:
    """Tests for calibration summary output."""
    
    def test_summary_output(self):
        """Test that summary produces readable output."""
        cal = DeviceCalibration.uniform(num_qubits=3, t1=100.0, t2=80.0)
        summary = cal.summary()
        
        assert "Device:" in summary
        assert "Qubits: 3" in summary
        assert "T1 range:" in summary
        assert "T2 range:" in summary


class TestDeviceCalibrationEdgeCases:
    """Tests for edge cases in calibration handling."""
    
    def test_edge_string_with_underscore(self):
        """Test parsing edge strings with underscore separator."""
        data = {
            "backend_name": "test",
            "timestamp": "2026-01-22",
            "qubits": {"0": {"t1": 100.0, "t2": 80.0}},
            "couplers": {"0_1": {"cx_error": 0.01}},  # Underscore separator
        }
        cal = DeviceCalibration.from_dict(data)
        assert (0, 1) in cal.coupler_params
    
    def test_legacy_readout_error_field(self):
        """Test handling legacy 'readout_error' field (symmetric)."""
        data = {
            "backend_name": "test",
            "timestamp": "2026-01-22",
            "qubits": {"0": {"t1": 100.0, "t2": 80.0, "readout_error": 0.02}},
        }
        cal = DeviceCalibration.from_dict(data)
        assert cal.qubit_params[0].readout_error_0to1 == 0.02
        assert cal.qubit_params[0].readout_error_1to0 == 0.02
    
    def test_missing_optional_fields(self):
        """Test that missing optional fields use defaults."""
        data = {
            "backend_name": "test",
            "timestamp": "2026-01-22",
            "qubits": {"0": {"t1": 100.0, "t2": 80.0}},
        }
        cal = DeviceCalibration.from_dict(data)
        
        # Should have default values
        assert cal.qubit_params[0].readout_error_0to1 == 0.0
        assert cal.qubit_params[0].single_qubit_gate_error == 0.0
    
    def test_coupler_normalization(self):
        """Test that coupler keys are normalized to (min, max) order."""
        data = {
            "backend_name": "test",
            "timestamp": "2026-01-22",
            "qubits": {
                "0": {"t1": 100.0, "t2": 80.0},
                "1": {"t1": 100.0, "t2": 80.0},
            },
            "couplers": {
                "1-0": {"cx_error": 0.01},  # Reverse order
            },
        }
        cal = DeviceCalibration.from_dict(data)
        
        # Should be normalized to (0, 1)
        assert (0, 1) in cal.coupler_params
        assert (1, 0) not in cal.coupler_params
