from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

try:
    # Qiskit 2.x
    from qiskit.transpiler import Target
except Exception as e:
    raise ImportError("Qiskit >= 2.0 is required. Install with `pip install qiskit>=2.1,<3.0`.") from e


@dataclass(frozen=True)
class TranspileConfig:
    """
    Container for all transpilation knobs. Pure data, no logic.
    """
    target: Target                                # Hardware model (heavy-hex); includes durations & direction
    basis: Tuple[str, ...] = ("rz", "sx", "x", "cx")
    seeds: int = 12                               # how many seed tries for layout/routing search
    seed_offset: int = 0                          # offset to make runs reproducible yet distinct
    schedule_mode: str = "alap"                   # "alap" or "asap"
    dd_policy: Optional[str] = None               # None | "XIX" | "XYXY"
    keep_top_k: int = 3                           # leaderboard length
    enable_gate_direction_fix: bool = True        # apply GateDirection after routing
    dd_between_rounds_only: bool = True           # caller ensures barriers; we won't insert DD inside QEC rounds
    sabre_layout_iterations: int = 5               # SabreLayout max iterations (higher = slower but maybe better)
    
    def basis_set(self) -> set[str]:
        return set(self.basis)

    def seed_stream(self) -> Iterable[int]:
        start = int(self.seed_offset)
        for i in range(self.seeds):
            yield start + i