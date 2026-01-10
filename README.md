Layout-aware surface-code and heavy-hex experiments built on Qiskit QEC, Stim, and PyMatching.

## Basis sanity check
- The CSS convention follows `|0_L>` -> measure logical `Z` (sensitive to physical `X` flips) and `|+_L>` -> measure logical `X` (sensitive to physical `Z` flips). Qiskit’s heavy-hex code swaps X/Z labels relative to some papers; the pipeline already applies that swap.
- Run `scripts/basis_sanity_check.py` to confirm the bookkeeping under pure noise:  
  `PYTHONPATH=src myenv/bin/python scripts/basis_sanity_check.py --code-type heavy_hex --distance 3 --p 0.02 --shots 2000`
- Expected outcome: with pure Z noise only the `|+_L>` experiment degrades; with pure X noise only the `|0_L>` experiment degrades. A mismatch indicates a basis/observable labeling bug rather than physics.
