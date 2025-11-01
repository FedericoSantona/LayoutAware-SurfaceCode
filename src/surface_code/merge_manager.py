"""Merge window and seam operation management.

This module handles merge windows, seam tracking, and joint check operations
across patches during circuit building.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .builder_state import BuilderState
from .layout import Layout


class MergeManager:
    """Manages merge windows and seam operations."""
    
    def __init__(self, layout: Layout):
        """Initialize merge manager.
        
        Args:
            layout: Layout instance for accessing seams
        """
        self.layout = layout
        
        # Active merge trackers
        self.active_rough: Optional[Tuple[str, str, int]] = None  # (a,b,remaining)
        self.active_smooth: Optional[Tuple[str, str, int]] = None
        
        # Merge window tracking
        self.merge_windows: List[Dict[str, object]] = []
        self.current_window: Optional[Dict[str, object]] = None
        self.window_id = 0
        
        # Seam tracking
        self.seam_round_counts: Dict[Tuple[str, str, str], int] = {}
        self.seam_wrap_counts: Dict[Tuple[str, str, str], int] = {}
        self.last_window_joint: Dict[Tuple[str, str, str], List[int]] = {}
        self.first_window_joint: Dict[Tuple[str, str, str], List[int]] = {}
    
    def begin_window(self, kind: str, a: str, b: str, rounds: int, state: BuilderState) -> None:
        """Start a new merge window.
        
        Args:
            kind: Merge type ("rough" or "smooth")
            a: First patch name
            b: Second patch name
            rounds: Number of rounds for this window
            state: BuilderState for updating joint_prev
        """
        self.current_window = {
            "id": self.window_id,
            "type": kind,
            "parity_type": "ZZ" if kind == "rough" else "XX",
            "a": a,
            "b": b,
            "rounds": int(rounds),
        }
        self.window_id += 1
        key = (kind, a, b)
        self.seam_round_counts[key] = 0
        
        # Seed state.prev.joint_prev[(kind, a, b)] with [None] * len(pairs) for this window
        pairs = self.layout.seams.get((kind, a, b), [])
        state.prev.joint_prev[(kind, a, b)] = [None] * len(pairs)
    
    def end_window(self) -> Optional[Dict[str, object]]:
        """End current merge window.
        
        Returns:
            The ended window dictionary, or None if no window was active
        """
        ended_window = self.current_window
        if ended_window is not None:
            self.merge_windows.append(ended_window)
            self.current_window = None
        return ended_window
    
    def measure_joint_checks(
        self,
        circuit,
        kind: str,
        a: str,
        b: str,
    ) -> List[int]:
        """Measure simple 2-body joint checks across the seam.
        
        Args:
            circuit: stim.Circuit to append measurements to
            kind: Merge type ("rough" or "smooth")
            a: First patch name
            b: Second patch name
            
        Returns:
            List of absolute measurement indices (one per pair)
        """
        from .builder_utils import mpp_from_positions
        
        key = (kind, a, b)
        pairs = sorted(self.layout.seams.get(key, []))
        if not pairs:
            return []
        
        pauli = "Z" if kind == "rough" else "X"
        indices: List[int] = []
        for ia, ib in pairs:
            global_a = self.layout.globalize_local_index(a, ia)
            global_b = self.layout.globalize_local_index(b, ib)
            idx = mpp_from_positions(circuit, [global_a, global_b], pauli)
            if idx is not None:
                indices.append(idx)
        
        return indices
    
    def get_windows(self) -> List[Dict[str, object]]:
        """Get all merge windows.
        
        Returns:
            List of merge window dictionaries
        """
        return self.merge_windows
    
    def get_seam_wrap_counts(self) -> Dict[Tuple[str, str, str], int]:
        """Get seam wrap counts for diagnostics.
        
        Returns:
            Dictionary mapping seam keys to wrap counts
        """
        return self.seam_wrap_counts
    
    def get_current_window_id(self) -> int:
        """Get current window ID (or next if no current window).
        
        Returns:
            Window ID - if current_window exists, returns its ID.
            Otherwise returns window_id - 1 (the last completed window ID).
        """
        if self.current_window is not None:
            return self.current_window["id"]
        return self.window_id - 1 if self.window_id > 0 else 0

