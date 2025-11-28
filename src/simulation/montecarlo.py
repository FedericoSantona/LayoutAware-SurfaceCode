from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class MonteCarloConfig:
    shots: int = 5000
    seed: Optional[int] = None