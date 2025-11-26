"""Stim circuit builders for phenomenological surface-code experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import stim


@dataclass
class PhenomenologicalStimConfig:
    """Configuration values for phenomenological stabilizer sampling.

    family:
        None  -> interleave Z and X halves (measure both each round)
        "Z"   -> Z-only family (CSS split, measure only Z stabilizers)
        "X"   -> X-only family (CSS split, measure only X stabilizers)
    """
    rounds: int = 5
    p_x_error: float = 1e-3
    p_z_error: float = 1e-3
    init_label: Optional[str] = None  # one of {"0", "1", "+", "-"}
    family: Optional[str] = None
   

@dataclass
class PhaseSpec:
    """Describe one spacetime phase of the lattice-surgery protocol.

    Each phase uses a *single* CSS stabilizer family (Z and X) on a *fixed*
    layout of qubits (three patches plus any seam / intermediate qubits).
    The differences between phases come solely from which stabilizers are
    measured and for how many rounds.

    Attributes
    ----------
    name:
        Human-readable label, e.g. "pre-merge", "C+INT smooth merge".
    z_stabilizers / x_stabilizers:
        Lists of Pauli strings (on the *combined* code) describing the Z- and
        X-type checks to measure in this phase.
    rounds:
        Number of repeated stabilizer-measurement rounds in this phase.
    measure_z / measure_x:
        Optional flags to explicitly control whether Z or X stabilizers are
        measured in this phase. If None, uses the default behavior based on
        config.family and whether stabilizers are provided.
    """

    name: str
    z_stabilizers: Sequence[str]
    x_stabilizers: Sequence[str]
    rounds: int
    measure_z: bool | None = None
    measure_x: bool | None = None



class PhenomenologicalStimBuilder:
    """Construct Stim circuits for repeated stabilizer measurements."""

    def __init__(
        self,
        code,
        z_stabilizers: Sequence[str],
        x_stabilizers: Sequence[str],
        logical_z: Optional[str] = None,
        logical_x: Optional[str] = None,
    ) -> None:
        self.code = code
        self.z_stabilizers = list(z_stabilizers)
        self.x_stabilizers = list(x_stabilizers)
        self.logical_z = logical_z
        self.logical_x = logical_x

        # Debug metadata: maps absolute measurement index -> info dict
        # (family, round, stabilizer index, and Pauli string).
        self._meas_meta: Dict[int, Dict[str, Any]] = {}

    # ----- helpers -----------------------------------------------------

    @staticmethod
    def _init_intent(label: str) -> tuple[str, int]:
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
    def _mpp_from_string(circuit: stim.Circuit, pauli_str: str) -> Optional[int]:
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

    @staticmethod
    def _rec_from_abs(circuit: stim.Circuit, index: int) -> stim.GateTarget:
        return stim.target_rec(index - circuit.num_measurements)

    def _measure_list(
        self,
        circuit: stim.Circuit,
        paulies: Sequence[str],
        *,
        family: str,
        round_index: int,
    ) -> list[int]:
        """Measure a list of Paulis and record debug metadata per measurement.

        Args:
            circuit: The Stim circuit to append operations into.
            paulies: Sequence of Pauli strings to measure with MPP.
            family: Stabilizer family label, e.g. "Z" or "X".
            round_index: Logical round index for this measurement pass. Use -1
                for pre-round reference measurements.
        """
        indices: list[int] = []
        for stab_index, pauli in enumerate(paulies):
            idx = self._mpp_from_string(circuit, pauli)
            if idx is not None:
                indices.append(idx)
                # Record debug metadata for this measurement index.
                self._meas_meta[idx] = {
                    "family": family,
                    "round": round_index,
                    "stab_index": stab_index,
                    "pauli": pauli,
                }
        return indices

    def _add_detectors(self, circuit: stim.Circuit, prev: Sequence[int], curr: Sequence[int]) -> None:
        for curr_idx, prev_idx in zip(curr, prev):
            circuit.append_operation(
                "DETECTOR",
                [self._rec_from_abs(circuit, prev_idx), self._rec_from_abs(circuit, curr_idx)],
            )

        # ----- internal CSS engine --------------------------------------------

    def _measure_family_flags(
        self,
        config: "PhenomenologicalStimConfig",
        phase_measure_z: bool | None,
        phase_measure_x: bool | None,
    ) -> tuple[bool, bool]:
        fam = (config.family or "").upper()
        if fam not in {"", "Z", "X"}:
            raise ValueError("config.family must be one of None, 'Z', or 'X'")
        measure_Z = phase_measure_z if phase_measure_z is not None else (fam in {"", "Z"})
        measure_X = phase_measure_x if phase_measure_x is not None else (fam in {"", "X"})
        return measure_Z, measure_X

    def _apply_x_noise(self, circuit: stim.Circuit, config: "PhenomenologicalStimConfig") -> None:
        if config.p_x_error:
            n = self.code.n
            circuit.append_operation("X_ERROR", list(range(n)), config.p_x_error)

    def _apply_z_noise(self, circuit: stim.Circuit, config: "PhenomenologicalStimConfig") -> None:
        if config.p_z_error:
            n = self.code.n
            circuit.append_operation("Z_ERROR", list(range(n)), config.p_z_error)


    # ----- logical observable helpers ---------------------------------------

    def measure_logical_once(
        self,
        circuit: stim.Circuit,
        logical_str: Optional[str],
    ) -> Optional[int]:
        """Measure a logical operator once and return its absolute index.

        Appends a TICK and an MPP(logical_str) if logical_str is not None.
        Returns the absolute measurement index, or None if skipped/empty.
        """
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


    def _run_css_block(
        self,
        circuit: stim.Circuit,
        *,
        z_stabilizers: Sequence[str],
        x_stabilizers: Sequence[str],
        rounds: int,
        config: "PhenomenologicalStimConfig",
        phase_measure_z: bool | None,
        phase_measure_x: bool | None,
        sz_prev: list[int] | None,
        sx_prev: list[int] | None,
    ) -> tuple[list[int] | None, list[int] | None]:
    
        """Core engine: run Z/X halves with noise and time-like detectors."""
        z_stabs = list(z_stabilizers)
        x_stabs = list(x_stabilizers)

        measure_Z, measure_X = self._measure_family_flags(
            config, phase_measure_z, phase_measure_x
        )

        # Warmup round if needed
        if measure_Z and z_stabs and sz_prev is None:
            circuit.append_operation("TICK")
            sz_prev = self._measure_list(
                circuit, z_stabs, family="Z", round_index=-1
            )
        if measure_X and x_stabs and sx_prev is None:
            circuit.append_operation("TICK")
            sx_prev = self._measure_list(
                circuit, x_stabs, family="X", round_index=-1
            )

        # Main rounds
        for round_idx in range(rounds):
            if measure_Z and z_stabs:
                circuit.append_operation("TICK")
                self._apply_x_noise(circuit, config)
                sz_curr = self._measure_list(
                    circuit, z_stabs, family="Z", round_index=round_idx
                )
                if sz_prev is not None:
                    self._add_detectors(circuit, sz_prev, sz_curr)
                sz_prev = sz_curr

            if measure_X and x_stabs:
                circuit.append_operation("TICK")
                self._apply_z_noise(circuit, config)
                sx_curr = self._measure_list(
                    circuit, x_stabs, family="X", round_index=round_idx
                )
                if sx_prev is not None:
                    self._add_detectors(circuit, sx_prev, sx_curr)
                sx_prev = sx_curr

        # If this phase doesn't measure a family at all, kill its history
        if not measure_Z:
            sz_prev = None
        if not measure_X:
            sx_prev = None

        return sz_prev, sx_prev


    # ----- public API --------------------------------------------------

    def build(self, config: PhenomenologicalStimConfig) -> tuple[stim.Circuit, list[tuple[int, int]]]:
        n = self.code.n
        circuit = stim.Circuit()

        # Validate CSS family configuration
        fam = (config.family or "").upper()
        if fam not in {"", "Z", "X"}:
            raise ValueError("config.family must be one of None, 'Z', or 'X'")

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

        for q in range(n):
            circuit.append_operation("QUBIT_COORDS", [q], [q, 0])  

        observable_pairs: list[tuple[int, int]] = []

        start: Optional[int] = self.measure_logical_once(circuit, logical_string)

        # Run CSS measurement rounds using the helper method
        sz_prev: Optional[list[int]] = None
        sx_prev: Optional[list[int]] = None
        sz_prev, sx_prev = self._run_css_block(
            circuit,
            z_stabilizers=self.z_stabilizers,
            x_stabilizers=self.x_stabilizers,
            rounds=config.rounds,
            config=config,
            phase_measure_z=None,
            phase_measure_x=None,
            sz_prev=sz_prev,
            sx_prev=sx_prev,
        )

        end: Optional[int] = self.measure_logical_once(circuit, logical_string)
        self.attach_observable_pair(
            circuit,
            start_idx=start,
            end_idx=end,
            observable_index=0,
            observable_pairs=observable_pairs,
        )

        return circuit, observable_pairs

        # ----- multi-phase API --------------------------------------------------

    def run_phases(
        self,
        circuit: stim.Circuit,
        phases: Sequence["PhaseSpec"],
        config: "PhenomenologicalStimConfig",
    ) -> None:
        """Run a sequence of CSS phases on an existing circuit.

        This mirrors the internal logic of `build`, but allows each phase to
        have its own Z/X stabilizer sets and round counts, while sharing
        time-like detectors across phases when the stabilizer sets match.
        """
        fam = (config.family or "").upper()
        if fam not in {"", "Z", "X"}:
            raise ValueError("config.family must be one of None, 'Z', or 'X'")

        sz_prev: list[int] | None = None
        sx_prev: list[int] | None = None
        prev_z_set: Sequence[str] | None = None
        prev_x_set: Sequence[str] | None = None

        for phase in phases:
            z_stabs = list(phase.z_stabilizers)
            x_stabs = list(phase.x_stabilizers)

            # Reset time-like detectors when stabilizer sets change.
            if prev_z_set is not None and z_stabs != list(prev_z_set):
                sz_prev = None
            if prev_x_set is not None and x_stabs != list(prev_x_set):
                sx_prev = None

            # Run CSS measurement rounds using the helper method
            sz_prev, sx_prev = self._run_css_block(
                circuit,
                z_stabilizers=z_stabs,
                x_stabilizers=x_stabs,
                rounds=phase.rounds,
                config=config,
                phase_measure_z=phase.measure_z,
                phase_measure_x=phase.measure_x,
                sz_prev=sz_prev,
                sx_prev=sx_prev,
            )

            prev_z_set = z_stabs
            prev_x_set = x_stabs