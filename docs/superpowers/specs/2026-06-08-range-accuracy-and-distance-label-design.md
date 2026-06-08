# Design: Range Accuracy Fix + Distance Label for HR Mode

**Date:** 2026-06-08
**Status:** Approved

## Problem

1. **Range measurement error:** `find_best_range_bin` returns discrete integer bin index, causing ~30-40cm error at close range due to grid quantization (picket-fence effect) + uncalibrated hardware antenna delay offset.
2. **Missing distance label:** HR mode has no distance display in SubjectTab, even though `best_range_bin` is available and logged to CSV.
3. **Re-scan updates:** When 2D-CFAR re-scan updates `_best_bin`, the distance label should reflect the new position.

## Root Cause

- `dsp_pipeline/range_bin.py:10` — return type is `int`, discrete bin index without sub-bin interpolation
- `ui/monitor_mode.py:174-181` — `HRMode.poll_and_update` doesn't compute physical distance or pass it to UI
- `ui/subject_tab.py:150-158` — `update_display` has no `target_distance_m` parameter and no distance label widget

## Design

### File Changes

| File | Change |
|---|---|
| `dsp_pipeline/range_bin.py` | Return `float` with quadratic interpolation; apply interpolation in fallback path too |
| `dsp_pipeline/pipeline.py` | `_best_bin` → `float \| None`; wrap all index usages with `int()`; 2D-CFAR paths store as `float` |
| `dsp_pipeline/phase.py` | `extract_phase` accepts float bin, casts to int for indexing |
| `ui/subject_tab.py` | Add distance label widget; extend `update_display` with `target_distance_m` param; clear on reset |
| `ui/monitor_mode.py` | Compute physical distance in `HRMode.poll_and_update` and pass to SubjectTab |
| `config/protocol.py` | Add `RANGE_HARDWARE_OFFSET_M: float = 0.35` config constant |

### Key Design Decisions

1. **Float bin storage, int indexing:** `_best_bin` stores precise float for distance calculation. All array indexing points (`_extract_rx_complex`, `extract_phase`, etc.) use `int(self._best_bin)` for safe numpy indexing.

2. **2D-CFAR paths:** `_run_2d_cfar_lock` and `_run_2d_cfar_rescan` store results as `float(best_bin)` for type consistency. No interpolation on CFAR path yet — can be added later.

3. **Distance resolution:** Derived from FrameHeader `range_resol_mm`. HR mode = 25mm (0.025m).

4. **Hardware offset:** Placed in `config/protocol.py` as `RANGE_HARDWARE_OFFSET_M` for easy calibration. Default 0.35, to be adjusted after physical measurement with a ruler.

5. **Re-scan updates work automatically:** `poll_and_update` reads `self._pipeline.best_range_bin` fresh each cycle. When CFAR re-scan (every ~100 frames) updates `_best_bin`, the distance label reflects it on the next UI refresh without any extra mechanism.

### Distance Calculation Formula

```
physical_distance_m = (best_bin * range_resolution_m) - hardware_offset_m
clamped to >= 0.01 to avoid negative display
```

### UI Layout

Distance label added to SubjectTab top row, left of SQI indicator:
```
[距离: XX.X cm]  ......  [SQI]
```

Styled with blue accent (#3498db) to distinguish from BPM numbers.

### Calibration Procedure

1. Place a fixed metal reflector (or person) at a precisely measured distance (e.g., 20cm, 40cm)
2. Observe the distance label reading
3. If reading is off by N cm, adjust `RANGE_HARDWARE_OFFSET_M` by N cm
4. Expected final accuracy: within 0.5 cm

## Scope

- HR mode only (BP mode already has distance display via `target_distance_m` in BPResult)
- No interpolation on 2D-CFAR path (future enhancement)
- No changes to ResearchTab (distance could be added later if needed)
