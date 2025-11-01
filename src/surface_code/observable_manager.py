"""Observable bracketing management.

This module handles logical observable bracketing, tracking start/end indices
and emitting OBSERVABLE_INCLUDE operations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import stim

from .builder_state import BuilderState
from .builder_utils import rec_from_abs
from .layout import Layout


class ObservableManager:
    """Manages logical observable bracketing."""
    
    def __init__(self, layout: Layout, bracket_map: Dict[str, str]):
        """Initialize observable manager.
        
        Args:
            layout: Layout instance
            bracket_map: Map from patch names to basis ('Z' or 'X')
        """
        self.layout = layout
        self.bracket_map = bracket_map
        
        # Track start and end indices for each patch
        self.start_indices: Dict[str, Optional[int]] = {}
        self.end_indices: Dict[str, Optional[int]] = {}
        
        # Track effective basis (may differ from requested due to conflicts)
        self.effective_basis_map: Dict[str, str] = {}
        
        # Initialize from bracket_map
        for name in layout.patches.keys():
            self.start_indices[name] = None
            self.end_indices[name] = None
            if name in bracket_map:
                requested_basis = bracket_map[name].upper()
                self.effective_basis_map[name] = requested_basis
    
    def capture_start(self, patch_name: str, measurement_idx: int) -> None:
        """Capture start measurement index for a patch.
        
        Args:
            patch_name: Name of the patch
            measurement_idx: Absolute measurement index
        """
        self.start_indices[patch_name] = measurement_idx
    
    def seal_end(self, patch_name: str, basis: str, last_measurement_idx: Optional[int]) -> None:
        """Seal end measurement index for a patch.
        
        Args:
            patch_name: Name of the patch
            basis: Basis ('Z' or 'X')
            last_measurement_idx: Last measurement index for this basis
        """
        if self.end_indices.get(patch_name) is None:
            self.end_indices[patch_name] = last_measurement_idx
    
    def get_start_indices(self) -> Dict[str, Optional[int]]:
        """Get start indices dictionary.
        
        Returns:
            Dictionary mapping patch names to start indices
        """
        return self.start_indices
    
    def get_end_indices(self) -> Dict[str, Optional[int]]:
        """Get end indices dictionary.
        
        Returns:
            Dictionary mapping patch names to end indices
        """
        return self.end_indices
    
    def finalize_observables(
        self,
        circuit: stim.Circuit,
        state: BuilderState,
        _last_non_none,
    ) -> Tuple[List[Tuple[int, int]], List[str], List[Tuple[Optional[int], Optional[int], int]]]:
        """Finalize observables and prepare for emission.
        
        Args:
            circuit: stim.Circuit instance
            state: BuilderState instance
            _last_non_none: Helper function to get last non-None index
            
        Returns:
            Tuple of (observable_pairs, basis_labels, deferred_observables)
        """
        observable_pairs: List[Tuple[int, int]] = []
        basis_labels: List[str] = []
        deferred_observables: List[Tuple[Optional[int], Optional[int], int]] = []
        observable_index = 0
        
        # At the very end, fallback-seal any observables that didn't conflict
        for pname, basis in self.effective_basis_map.items():
            if pname not in self.end_indices or self.end_indices[pname] is not None:
                continue
            if basis == "Z":
                self.end_indices[pname] = _last_non_none(list(state.prev.z_prev.get(pname, [])))
            else:
                self.end_indices[pname] = _last_non_none(list(state.prev.x_prev.get(pname, [])))
        
        # Only bracket patches that are explicitly in bracket_map (excludes ancillas and terminated patches)
        for name in self.bracket_map.keys():
            if name not in self.layout.patches or name in state.terminated_patches:
                continue  # Skip if patch doesn't exist or was terminated
            
            requested_basis = self.bracket_map[name].upper()
            effective_basis = self.effective_basis_map.get(name, requested_basis)
            
            # Prefer a pre-sealed end (set when a conflicting window began)
            end_idx = self.end_indices.get(name)
            if end_idx is None:
                if effective_basis == "Z":
                    end_idx = _last_non_none(list(state.prev.z_prev.get(name, [])))
                else:
                    end_idx = _last_non_none(list(state.prev.x_prev.get(name, [])))
                self.end_indices[name] = end_idx
            
            start_idx = self.start_indices[name]
            
            targets: List[stim.GateTarget] = []
            if start_idx is not None:
                targets.append(rec_from_abs(circuit, start_idx))
            if end_idx is not None:
                targets.append(rec_from_abs(circuit, end_idx))
            
            if targets:
                # Defer OBSERVABLE_INCLUDE to the very end to avoid later anti-commuting MPPs
                deferred_observables.append((start_idx, end_idx, observable_index))
                observable_pairs.append((start_idx, end_idx))
                basis_labels.append(effective_basis)
                observable_index += 1
        
        return observable_pairs, basis_labels, deferred_observables
    
    def emit_observables(
        self,
        circuit: stim.Circuit,
        deferred_observables: List[Tuple[Optional[int], Optional[int], int]],
    ) -> None:
        """Emit deferred OBSERVABLE_INCLUDE operations.
        
        Args:
            circuit: stim.Circuit to append operations to
            deferred_observables: List of (start_idx, end_idx, obs_k) tuples
        """
        if deferred_observables:
            final_m2 = circuit.num_measurements
            for s_idx, e_idx, obs_k in deferred_observables:
                obs_targets: List[stim.GateTarget] = []
                if s_idx is not None:
                    obs_targets.append(stim.target_rec(s_idx - final_m2))
                if e_idx is not None:
                    obs_targets.append(stim.target_rec(e_idx - final_m2))
                if obs_targets:
                    circuit.append_operation("OBSERVABLE_INCLUDE", obs_targets, obs_k)

