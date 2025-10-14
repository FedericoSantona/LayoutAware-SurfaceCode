"""Stim circuit builders for phenomenological surface-code experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

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
    # Optional explicit logical correlation measurement: start/end operator choice
    # as basis labels 'Z' or 'X'. When provided, they override the default derived
    # from init_label for the start, and allow using a different operator at the end.
    logical_start: Optional[str] = None
    logical_end: Optional[str] = None
    # Select the fixed bracketing basis for the start/end logical MPP used in
    # OBSERVABLE_INCLUDE. One of None, 'Z', or 'X'. If None, derive from
    # init_label: Z for {0,1}, X for {+,-}.
    bracket_basis: Optional[str] = None
    # Optional: append a final end-only demo readout MPP in the requested basis
    # ("X" or "Z"). This extra measurement is NOT part of OBSERVABLE_INCLUDE and
    # NOT used by detectors; it is for physics-based reporting in the end basis.
    demo_basis: Optional[str] = None

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

    def _measure_list(self, circuit: stim.Circuit, paulies: Sequence[str]) -> list[int]:
        indices: list[int] = []
        for pauli in paulies:
            idx = self._mpp_from_string(circuit, pauli)
            if idx is not None:
                indices.append(idx)
        return indices

    def _add_detectors(self, circuit: stim.Circuit, prev: Sequence[int], curr: Sequence[int]) -> None:
        for curr_idx, prev_idx in zip(curr, prev):
            circuit.append_operation(
                "DETECTOR",
                [self._rec_from_abs(circuit, prev_idx), self._rec_from_abs(circuit, curr_idx)],
            )


    # ----- public API --------------------------------------------------

    def build(self, config: PhenomenologicalStimConfig) -> tuple[stim.Circuit, list[tuple[int, int]], dict]:
        n = self.code.n
        circuit = stim.Circuit()

        # Which CSS family to measure in this circuit
        fam = (config.family or "").upper()
        if fam not in {"", "Z", "X"}:
            raise ValueError("config.family must be one of None, 'Z', or 'X'")
        measure_Z = (fam in {"", "Z"})
        measure_X = (fam in {"", "X"})

        # Deterministic DEM: always bracket with a single logical operator at
        # start and end, chosen by bracket_basis (Z or X). If not provided,
        # derive from init_label: Z for {0,1}, X for {+,-}. Default to Z.
        chosen_basis: str
        if config.bracket_basis is not None:
            chosen_basis = config.bracket_basis.strip().upper()
            if chosen_basis not in {"Z", "X"}:
                raise ValueError("bracket_basis must be 'Z' or 'X'")
        elif config.init_label is not None:
            init_b, _ = self._init_intent(config.init_label.strip())
            chosen_basis = init_b
        else:
            chosen_basis = "Z"

        if chosen_basis == "Z":
            if self.logical_z is None:
                raise ValueError("Z logical operator required for Z bracketing")
            logical_start_str: Optional[str] = self.logical_z
            logical_end_str: Optional[str] = self.logical_z
        else:
            if self.logical_x is None:
                raise ValueError("X logical operator required for X bracketing")
            logical_start_str = self.logical_x
            logical_end_str = self.logical_x

        for q in range(n):
            circuit.append_operation("QUBIT_COORDS", [q], [q, 0])

        def apply_x_noise() -> None:
            if config.p_x_error:    
                circuit.append_operation("X_ERROR", list(range(n)), config.p_x_error)

        def apply_z_noise() -> None:
            if config.p_z_error:
                circuit.append_operation("Z_ERROR", list(range(n)), config.p_z_error)

        observable_pairs: list[tuple[int, int]] = []
        metadata: dict = {}

        start: Optional[int] = None
        if logical_start_str is not None:
            circuit.append_operation("TICK")
            start = self._mpp_from_string(circuit, logical_start_str)

        # Establish reference measurements before noisy cycles, per-family.
        sz_prev: Optional[list[int]] = None
        sx_prev: Optional[list[int]] = None
        if measure_Z:
            circuit.append_operation("TICK")
            sz_prev = self._measure_list(circuit, self.z_stabilizers)
        if measure_X:
            circuit.append_operation("TICK")
            sx_prev = self._measure_list(circuit, self.x_stabilizers)

        for _round in range(config.rounds):
            # Z half
            if measure_Z:
                circuit.append_operation("TICK")
              
                apply_x_noise()
                sz_curr = self._measure_list(circuit, self.z_stabilizers)
                if sz_prev is not None:
                    self._add_detectors(circuit, sz_prev, sz_curr)

                sz_prev = sz_curr

            # X half
            if measure_X:
                circuit.append_operation("TICK")

                apply_z_noise()
                sx_curr = self._measure_list(circuit, self.x_stabilizers)
                if sx_prev is not None:
                    self._add_detectors(circuit, sx_prev, sx_curr)
        
                sx_prev = sx_curr

        end: Optional[int] = None
        if logical_end_str is not None:
            circuit.append_operation("TICK")
            end = self._mpp_from_string(circuit, logical_end_str)
            obs_targets: list[stim.GateTarget] = []
            if start is not None:
                obs_targets.append(self._rec_from_abs(circuit, start))
            if end is not None:
                obs_targets.append(self._rec_from_abs(circuit, end))
            if obs_targets:
                circuit.append_operation("OBSERVABLE_INCLUDE", obs_targets, 0)
                observable_pairs.append((start, end))

        # Optional: append an end-only demo readout in requested basis for
        # physics-based reporting (not part of observables/detectors).
        if config.demo_basis is not None:
            b = config.demo_basis.strip().upper()
            if b not in {"X", "Z"}:
                raise ValueError("demo_basis must be 'X' or 'Z'")
            if b == "X":
                if self.logical_x is None:
                    raise ValueError("X logical operator required for demo readout")
                circuit.append_operation("TICK")
                demo_idx = self._mpp_from_string(circuit, self.logical_x)
                metadata["demo_basis"] = "X"
                metadata["demo_index"] = demo_idx
            else:
                # Z basis demo (use same Z_L string)
                circuit.append_operation("TICK")
                demo_idx = self._mpp_from_string(circuit, self.logical_z)
                metadata["demo_basis"] = "Z"
                metadata["demo_index"] = demo_idx

        return circuit, observable_pairs, metadata
