# Builder Comparison: Old vs New

## Key Differences

### Old Builder (`stim_builder.py`) - Simple and Direct

**Structure:**
1. Initialize qubits
2. Logical start measurement (if init_label)
3. **Reference measurements** (before noisy cycles) - establishes baseline
4. For each round:
   - Z half: apply X noise → measure Z stabilizers → create detectors (prev vs curr)
   - X half: apply Z noise → measure X stabilizers → create detectors (prev vs curr)
5. Logical end measurement → OBSERVABLE_INCLUDE

**Key Principles:**
- **Immediate detector creation**: `circuit.append_operation("DETECTOR", ...)` directly
- **Simple temporal chaining**: Compare prev vs curr measurements, create detector
- **Reference before noise**: Reference measurements establish baseline before any noise
- **Direct noise placement**: Noise applied right before measurements in each half
- **No deferred operations**: Everything happens immediately

**Detector Creation:**
```python
def _add_detectors(self, circuit, prev, curr):
    for curr_idx, prev_idx in zip(curr, prev):
        circuit.append_operation(
            "DETECTOR",
            [self._rec_from_abs(circuit, prev_idx), self._rec_from_abs(circuit, curr_idx)],
        )
```

### New Builder (`builder.py` + `MeasurementHalf`) - Complex with Lattice Surgery

**Structure:**
1. Initialize qubits
2. Logical start measurement (if explicit_logical_start)
3. **Warmup round** (no noise) - establishes reference measurements
4. For each round:
   - Z half: apply X noise → MeasurementHalf.measure_round() → defer detectors
   - X half: apply Z noise → MeasurementHalf.measure_round() → defer detectors
5. Close segments (wrap detectors)
6. Logical end measurement → ObservableManager handles
7. **Emit all deferred detectors** at the end

**Key Differences:**
- **Deferred detector creation**: `detector_manager.defer_detector()` → emitted at end
- **Complex segment tracking**: SegmentTracker manages temporal chains
- **Boundary handling**: DetectorManager handles boundary anchors
- **Merge support**: MergeManager handles lattice surgery operations
- **Observable management**: ObservableManager handles observables

**Potential Issues:**

1. **Detector Deferral**: Detectors are deferred and emitted at the end. While this should work, the timing might affect error model construction.

2. **Segment Tracking Complexity**: The SegmentTracker adds complexity that might introduce bugs in simple cases.

3. **Boundary Handling**: The new builder has complex boundary anchor logic that might interfere with simple error correction.

4. **Reference Measurement Timing**: The warmup round should work like the old reference, but the initialization of `state.prev` might not be exactly equivalent.

5. **Noise Model in Detector Emission**: The `emit_all_detectors` function receives a noise model, but temporal detectors should always be kept regardless of noise model.

## Critical Observations

### Old Builder Detector Creation Flow:
```
Round 0 (Reference): measure → store in sz_prev/sx_prev
Round 1: noise → measure → detector(reference, round1) → update prev
Round 2: noise → measure → detector(round1, round2) → update prev
```

### New Builder Detector Creation Flow:
```
Warmup: measure → store in state.prev.z_prev/x_prev
Round 1: noise → measure → defer_detector(warmup, round1) → update prev
Round 2: noise → measure → defer_detector(round1, round2) → update prev
...
End: emit_all_detectors() → convert deferred to DETECTOR operations
```

## Issues Found and Fixed

### CRITICAL BUG FIXED: Noise Applied to Warmup Round

**Problem**: The new builder was applying noise to the warmup round, which should establish clean reference measurements before any noise is introduced. This broke error correction because the reference measurements were contaminated with noise.

**Root Cause**: The builder applied noise before every `MeasureRound` operation, including the warmup round. The warmup round should be noise-free to match the old builder's behavior of creating reference measurements before noisy cycles.

**Fix**: Added a check to detect the warmup round (when `state.prev.z_prev` and `state.prev.x_prev` are empty for all patches) and skip noise application for that round. This ensures:
- Warmup round: No noise → clean reference measurements → stored in `state.prev`
- First round: Noise → measurements → detectors (warmup vs first round)
- Subsequent rounds: Noise → measurements → detectors (consecutive rounds)

**Code Change**: Modified `builder.py` to check `is_warmup` flag before applying noise:
```python
is_warmup = all(
    name not in state.prev.z_prev and name not in state.prev.x_prev
    for name in names
)
if not is_warmup and cfg.p_x_error:
    circuit.append_operation("X_ERROR", ...)
```

## Additional Verification Needed

1. **Verify temporal detectors are always created**: The condition `if a is None or b is None or a == b: continue` should not skip detectors when warmup has been done. ✓ (Warmup round has empty prev, so no detectors created - correct behavior)

2. **Check that deferred detector emission preserves detector structure**: The conversion from deferred detectors to DETECTOR operations must preserve the exact same structure as immediate creation. (Should be verified through testing)

3. **Simplify for non-merge cases**: When there are no merges, the builder should now behave exactly like the old builder. (Should be verified through testing)

