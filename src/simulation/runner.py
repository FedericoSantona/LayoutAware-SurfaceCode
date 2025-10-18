"""Monte Carlo simulation routines for logical error estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pymatching as pm
import stim

from surface_code.logical_ops import PauliFrame, parse_init_label
from surface_code import GlobalStimBuilder, PhenomenologicalStimConfig, create_single_patch_layout, MeasureRound


@dataclass
class BatchPauliFrame:
    """Batch Pauli-frame tracker built from decoder predictions."""

    bases: Tuple[str, ...]
    flips: np.ndarray

    def __post_init__(self) -> None:
        if self.flips.ndim != 2:
            raise ValueError("BatchPauliFrame expects a 2D array of flips")
        normalized = tuple(self._normalize_basis(b) for b in self.bases)
        if len(normalized) != self.flips.shape[1]:
            raise ValueError("Number of bases must match number of columns in flips array")
        self.bases = normalized

    @staticmethod
    def _normalize_basis(basis: str) -> str:
        b = basis.upper()
        if b not in {"X", "Z"}:
            raise ValueError("Tracked bases must be 'X' or 'Z'")
        return b

    def _resolve_index(self, basis: str, column: Optional[int]) -> int:
        if column is not None:
            if column < 0 or column >= len(self.bases):
                raise IndexError(f"Column {column} is out of range for BatchPauliFrame")
            return column
        normalized = self._normalize_basis(basis)
        matches = [idx for idx, label in enumerate(self.bases) if label == normalized]
        if not matches:
            raise ValueError(f"No tracked basis '{normalized}' in BatchPauliFrame")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple tracked observables share basis '{normalized}'. "
                "Specify 'column' explicitly."
            )
        return matches[0]

    def correction_bits(self, basis: str, column: Optional[int] = None) -> np.ndarray:
        """Return the decoder flip bits for a basis (optionally selecting a column)."""
        idx = self._resolve_index(basis, column)
        return self.flips[:, idx]

    def column_for_basis(self, basis: str, column: Optional[int] = None) -> int:
        """Resolve and return the column index associated with a logical basis."""
        return self._resolve_index(basis, column)

    def apply(self, samples: np.ndarray, basis: str, column: Optional[int] = None) -> np.ndarray:
        """Apply the stored Pauli-frame flips to a set of logical samples."""
        idx = self._resolve_index(basis, column)
        flips = self.flips[:, idx]
        column_samples = samples[:, idx]
        return np.bitwise_xor(column_samples, flips)

    def pauli_frames(self) -> Tuple[PauliFrame, ...]:
        """Return a tuple of PauliFrame objects, one per shot."""
        frames = []
        for row in self.flips:
            pf = PauliFrame()
            for idx, basis in enumerate(self.bases):
                pf.set_flip(basis, int(row[idx]) & 1)
            frames.append(pf)
        return tuple(frames)


@dataclass
class MonteCarloConfig:
    shots: int = 5000
    seed: Optional[int] = None


@dataclass
class SimulationResult:
    logical_error_rate: float
    avg_syndrome_weight: float
    click_rate: float
    shots: int
    predictions: np.ndarray
    logical_observables: np.ndarray
    num_detectors: int
    observable_basis: Tuple[str, ...]
    # Optional physics-based end-basis reporting (demo readout)
    demo_bits: Optional[np.ndarray] = None
    demo_basis: Optional[str] = None

    def frame_corrected_observables(self, frame_flip: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return logical observables and decoder predictions after applying a frame flip."""
        bit = int(frame_flip) & 1
        logicals = self.logical_observables[:, 0]
        preds = self.predictions[:, 0]
        if bit == 0:
            return logicals, preds
        corrected_logicals = np.bitwise_xor(logicals, bit)
        corrected_preds = np.bitwise_xor(preds, bit)
        return corrected_logicals, corrected_preds

    def decoder_frame(self) -> BatchPauliFrame:
        """Return a batch Pauli-frame tracker built from decoder predictions."""
        return BatchPauliFrame(self.observable_basis, self.predictions)

    def apply_decoder_frame(
        self,
        basis: str,
        column: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply tracked Pauli-frame flips to logical observables for a given basis."""
        frame = self.decoder_frame()
        idx = frame.column_for_basis(basis, column=column)
        flips = frame.correction_bits(basis, column=idx)
        logicals = self.logical_observables[:, idx]
        corrected = np.bitwise_xor(logicals, flips)
        return logicals, corrected


def run_logical_error_rate(
    model,
    stim_config: PhenomenologicalStimConfig,
    mc_config: MonteCarloConfig,
) -> SimulationResult:
    """Run logical error rate simulation for a single-patch code model.
    
    Args:
        model: Code model with attributes: code, z_stabilizers, x_stabilizers, logical_z, logical_x
        stim_config: Stim circuit configuration
        mc_config: Monte Carlo configuration
    """
    # Create single-patch layout
    layout = create_single_patch_layout(model)
    
    # Generate explicit timeline of measure rounds
    ops = [MeasureRound(patch_ids=None) for _ in range(stim_config.rounds)]
    
    # Create builder and build circuit
    builder = GlobalStimBuilder(layout)
    bracket_map = {"q0": "Z"}  # Default bracket basis
    circuit, observable_pairs, metadata = builder.build(ops, stim_config, bracket_map)
    
    return run_circuit_logical_error_rate(circuit, observable_pairs, stim_config, mc_config, metadata)


def run_circuit_logical_error_rate(
    circuit: stim.Circuit,
    observable_pairs: Sequence[Tuple[int, int]],
    stim_config: PhenomenologicalStimConfig,
    mc_config: MonteCarloConfig,
    metadata: Optional[dict] = None,
) -> SimulationResult:
    dem = circuit.detector_error_model()
    matcher = pm.Matching.from_detector_error_model(dem)

    dem_sampler = dem.compile_sampler(seed=mc_config.seed)
    detector_samples, observable_samples, _ = dem_sampler.sample(mc_config.shots)

    detector_samples_bool = np.asarray(detector_samples, dtype=np.bool_)
    if observable_samples is None or observable_samples.size == 0:
        logical_array = np.zeros((mc_config.shots, 1), dtype=np.uint8)
    else:
        logical_array = np.asarray(observable_samples, dtype=np.uint8)

    predictions = matcher.decode_batch(detector_samples_bool)
    predictions = np.asarray(predictions, dtype=np.uint8)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    detector_samples_uint8 = detector_samples_bool.astype(np.uint8)
    target_bits = logical_array[:, 0]
    logical_error_rate = (predictions[:, 0] ^ target_bits).mean()

    avg_syndrome_weight = detector_samples_uint8.sum(axis=1).mean()
    click_rate = (detector_samples_uint8.sum(axis=1) > 0).mean()

    # Prefer basis labels provided by builder metadata, fallback to inference
    if metadata is not None and "observable_basis" in metadata:
        basis_tuple = tuple(str(b) for b in metadata.get("observable_basis", tuple()))
        if len(basis_tuple) != predictions.shape[1]:
            observable_basis = _infer_observable_basis(predictions.shape[1], stim_config)
        else:
            observable_basis = basis_tuple
    else:
        observable_basis = _infer_observable_basis(predictions.shape[1], stim_config)

    # Optionally sample demo measurement bits from the circuit's measurement
    # record (end-only MPP in requested basis). This is independent from DEM.
    demo_bits: Optional[np.ndarray] = None
    demo_basis: Optional[str] = None
    if metadata is not None and "demo_index" in metadata:
        demo_basis = metadata.get("demo_basis")
        # Compile a circuit sampler to sample raw measurements including the demo.
        circ_sampler = circuit.compile_sampler(seed=mc_config.seed)
        # Sample measurements only (detector samples not needed here).
        m_samples = circ_sampler.sample(shots=mc_config.shots)
        # The returned array is [shots, num_measurements]; pick the column at demo_index.
        # Stim returns booleans; convert to uint8 for consistency.
        demo_col = int(metadata["demo_index"]) if metadata["demo_index"] is not None else None
        if demo_col is not None:
            demo_bits = np.asarray(m_samples[:, demo_col], dtype=np.uint8)

    return SimulationResult(
        logical_error_rate=float(logical_error_rate),
        avg_syndrome_weight=float(avg_syndrome_weight),
        click_rate=float(click_rate),
        shots=mc_config.shots,
        predictions=predictions,
        logical_observables=logical_array,
        num_detectors=dem.num_detectors,
        observable_basis=observable_basis,
        demo_bits=demo_bits,
        demo_basis=demo_basis,
    )


def _infer_observable_basis(num_observables: int, stim_config: PhenomenologicalStimConfig) -> Tuple[str, ...]:
    """Infer logical basis labels for each tracked observable column."""
    if num_observables == 0:
        return tuple()
    # Bracket basis is fixed for DEM/decoding
    if stim_config.bracket_basis is not None:
        basis = stim_config.bracket_basis.strip().upper()
    elif stim_config.init_label is not None:
        basis, _ = parse_init_label(stim_config.init_label)
    else:
        basis = "Z"
    if basis not in {"X", "Z"}:
        raise ValueError(f"Unable to infer logical basis for observables (got '{basis}')")
    return tuple(basis for _ in range(num_observables))
