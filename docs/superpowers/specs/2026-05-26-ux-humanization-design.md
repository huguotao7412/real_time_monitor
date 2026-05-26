# UX Humanization Design: Vital Signs Monitor

**Date:** 2026-05-26
**Status:** Approved
**Scope:** UI layer only — DSP pipeline unchanged

## Goal

Transform the current engineer-facing raw data display into a medical-grade consumer UX with "living" visual feedback, while adding a Research Mode for algorithm validation and academic publication.

## Architecture

### Component Tree

```
MainWindow (~80 lines, thin shell)
└── QTabWidget
    ├── SubjectTab
    │   ├── CalibrationOverlay (ring progress, countdown text)
    │   ├── SqiIndicator (green/yellow/red 3-bar light)
    │   ├── BPM Center Area
    │   │   ├── QLabel(breath_bpm) + BreathingPetals (Apple Watch style)
    │   │   └── QLabel(heart_bpm) + HeartBeatIcon (QPainterPath bezier)
    │   ├── WaveWidget (filled, no axes, no grid, breath only)
    │   └── QLabel(human-readable status message)
    │
    └── ResearchTab
        ├── WaveWidget x2 (with axes + grid, breath + heart)
        ├── BPM row + SQI + raw status text
        ├── TrendPanel (5/15/30 min BPM scatter, pyqtgraph)
        └── DebugPanel (collapsible, DSP internals)
```

### File Changes

| Action | File | Notes |
|--------|------|-------|
| Refactor | `ui/main_window.py` | 460→~80 lines, becomes shell |
| Enhance | `ui/wave_widget.py` | Add fill mode, axis/grid toggle |
| New | `ui/subject_tab.py` | Subject mode container widget |
| New | `ui/research_tab.py` | Research mode container widget |
| New | `ui/calibration_overlay.py` | Ring progress overlay widget |
| New | `ui/sqi_indicator.py` | Signal quality 3-bar indicator |
| New | `ui/breathing_petals.py` | Apple Watch style breathing animation |
| New | `ui/trend_panel.py` | BPM history scatter plot |
| New | `ui/status_mapper.py` | DSP conditions → human messages (pure functions) |
| No change | `dsp_pipeline/**` | — |
| No change | `io_engine/**`, `models/**` | — |

## Feature Specifications

### 1. Calibration Overlay

- **Trigger:** `pipeline._calibration_done == False`
- **Pipeline exposes:** `_calibration_done` boolean must be readable from UI. Add a property `Pipeline.calibration_done: bool` and a progress signal (frames collected / 200).
- **Visual:** Semi-transparent dark overlay on SubjectTab. Center: ring progress bar (0→100%, 200 frames = 10s @ 20fps). Text below ring: "正在校准，请保持静止...". Countdown seconds inside ring.
- **Completion:** Ring completes → overlay fades out (500ms QPropertyAnimation opacity) → waveform + BPM fade in → brief green border flash.

### 2. DSP Status → Human Messages (status_mapper.py)

Pure function module, no class. Input: quality dict. Output: (message_str, level: "normal"|"warning"|"error").

```
phase_range < 0.005        → "未检测到微动，请确认在雷达覆盖范围内 (0.5m-1.5m)"  [warning]
breath_ratio < 0.03         → "信号较弱，请调整坐姿，正对雷达"                      [warning]
apnea_state == True         → "监测到呼吸微弱或屏息状态"                            [error]
high_freq_energy > 3*std    → "检测到体动干扰，请保持放松"                          [warning]
```

Error level: background blushes red, BPM values go semi-transparent (opacity 0.4).

### 3. SQI Indicator (sqi_indicator.py)

3-bar signal icon (like WiFi bars). Computation from existing quality fields:

| Level | Color | Condition | BPM Display |
|-------|-------|-----------|-------------|
| Good (3 bars) | Green #27ae60 | breath_ratio>0.15 AND phase_range>0.01 | Normal |
| Fair (2 bars) | Yellow #f39c12 | breath_ratio 0.05-0.15 OR phase_range 0.005-0.01 | Normal |
| Poor (1 bar) | Red #e74c3c | breath_ratio<0.05 OR phase_range<0.005 | Semi-transparent (opacity 0.4) |

Update every UI tick. Position: top-right of SubjectTab.

### 4. Breathing Petals (breathing_petals.py)

Apple Watch style: multiple petal arcs that open/close at breath_bpm frequency.

- **Implementation:** QWidget with custom paintEvent. Draw 5-6 overlapping arc paths (QPainterPath.arcTo). Petal openness controlled by a phase accumulator driven by breath_bpm.
- **Update cadence:** Every UI tick (~30fps), phase advances by `breath_bpm / 60 * 2π * dt`. If breath_bpm==0, hold last position.
- **Size:** ~80×80px, placed next to breath BPM number.
- **Color:** #27ae60 with varying alpha per petal for depth effect.

### 5. Heart Beat Icon

- QPainterPath bezier heart shape next to heart BPM number.
- Pulsing animation: scale 1.0→1.3→1.0 at heart_bpm frequency. Use QPropertyAnimation with easing.
- **Color thresholds:**
  - 60-100 bpm: green #27ae60
  - 50-60 or 100-110: orange #f39c12
  - <50 or >110: red #e74c3c
- Hold last color for 0.5s before transitioning (prevents flickering).

### 6. WaveWidget Enhancements

**New constructor parameters:**
- `fill_mode: bool = False` — fill below curve with gradient brush
- `show_axes: bool = True` — toggle axis labels
- `show_grid: bool = True` — toggle grid

**Fill implementation:** Use `pg.FillBetweenItem` or `curve.setFillLevel(0)` + `setBrush` with a QLinearGradient (bottom: dark green, top: bright green, alpha 80→180).

**Subject mode usage:** `WaveWidget("", fill_mode=True, show_axes=False, show_grid=False)` — just the filled waveform, no metadata.

### 7. Trend Panel (trend_panel.py)

- **Data storage:** Two `collections.deque(maxlen=1800)` (one per 30 min @ 1 sample/sec).
- **Display:** pyqtgraph scatter plot. Green dots for breath BPM, red dots for heart BPM.
- **Time window selector:** QComboBox with 5min / 15min / 30min. Adjusts x-axis range.
- **X-axis:** elapsed seconds, formatted as mm:ss.
- **Y-axis:** 0-150 BPM (or auto-range).
- **Reset:** Clear deques on each "Start" click.

Data fed from `_on_ui_tick`: sample latest breath_bpm and heart_bpm, append to deques once per second (use a frame counter modulo 20).

### 8. CSV Export

On save, write both `.npz` (existing) and `.csv`.

**CSV columns:**
```
Timestamp, FrameIndex, RangeBin, RawPhase, BreathBPM, HeartBPM,
PhaseRange, BreathRatio, HeartProminence, ApneaFlag, SQI_Level
```

All accumulated vitals data written at save time. Use Python `csv` module, UTF-8 BOM for Excel compatibility.

### 9. Dual-Mode Tab Switching

- QTabWidget with two tabs: "监测" (Subject) and "研究" (Research).
- Default tab on start: Subject.
- Switching is instant — both tabs share the same Pipeline reference and update from the same `_on_ui_tick` timer.
- Subject tab consumes `_latest_vitals` same as Research tab; just renders subset of data differently.

## Data Flow

```
Pipeline._process_frame(frame)
  → display_queue.put(VitalSigns)
    → MainWindow._on_ui_tick()  [every 33ms]
      → self._latest_vitals = display_queue.get()
      → SubjectTab.update(vitals)    # renders BPM, petals, filled wave, SQI, messages
      → ResearchTab.update(vitals)   # renders waveforms, trend, debug panel
```

## Pipeline Changes (minimal)

Add one property for calibration state exposure:

```python
@property
def calibration_done(self) -> bool:
    return self._calibration_done

@property
def calibration_progress(self) -> float:
    """0.0 - 1.0, based on collected calibration samples vs 200 target."""
    return min(1.0, len(self._calibration_samples) / 200.0)
```

Add `heart_waveform` output in the advanced DSP path (currently hardcoded to `np.array([])`).

Body movement detection is done in the UI layer: track `phase_range` history in a deque, flag sudden spikes >3σ as body movement. No pipeline changes needed.

## Testing Strategy

- **status_mapper.py:** Pure functions, unit-test with mock quality dicts.
- **WaveWidget:** Visual verification — run app, confirm fill renders and axes toggle.
- **CalibrationOverlay:** Verify overlay appears/disappears at correct pipeline states.
- **TrendPanel:** Feed synthetic BPM data, verify scatter plot renders and time windows switch.
- **CSV export:** Verify output columns match spec, open in Excel.
- **Mode switching:** Toggle tabs during active monitoring, confirm no crash and data continuity.

## Dependencies

No new packages required. All UI work uses existing PyQt6 + pyqtgraph.
