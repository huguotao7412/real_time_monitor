# BP/HR Mode Separation Design

## Problem

When switching between HR (heart rate) mode and BP (blood pressure) mode, the
two modes are not cleanly isolated. Symptoms:

- After switching HR→BP, data collection still runs in HR mode
- After switching, the Subject tab (监测标签页) shows HR data when it should show "--"
- Each fix adds more patches (`drain display queues`, `unified frame builder`) without addressing the root cause

**Root cause**: `_bp_mode` is a single boolean flag scattered across `MainWindow`.
Every method checks it (`_build_radar_frame`, `_serial_io_loop`, `_on_ui_tick`,
`_on_toggle_mode`). State variables are shared between modes (`_latest_vitals`,
`_latest_bp_result`, `_csv_rows`, waveform accumulators). The toggle logic has 9
sequential steps that must execute in exact order — any misstep causes mode confusion.

Specific issues:

1. `_serial_io_loop`'s old thread may still be alive after `join(timeout=3)` if
   `read_data` is blocking, leaving two threads reading the same serial port.
2. `_on_ui_tick` uses `if bp_mode: poll_bp; return` early-return — fragile.
3. No clear lifecycle boundary: old pipeline state can leak into new mode.

## Design

### Architecture: Strategy Pattern

Each mode is a self-contained object implementing `MonitorMode`. `MainWindow`
holds a single `_mode: MonitorMode` reference — no more `if self._bp_mode:` anywhere.

```
MainWindow (shared layer)
├── SerialManager / RadarMgr   — physical connection, mode-independent
├── UartParser                 — rebuilt on mode switch
├── UI skeleton (tabs, buttons, status bar)
└── _mode: MonitorMode         — the ONLY mode reference

    HRMode                      BPMode
    ├── Pipeline                 ├── BPPipeline
    ├── display_queue            ├── display_queue
    ├── waveform/csv/bpm accum   ├── bp result cache
    ├── frame: 2T4R, 128 bins    ├── frame: 1T1R, 32 bins
    └── boot: radar_mgr.boot()   └── boot: radar_mgr.boot_bp()
```

### MonitorMode Interface

```python
class MonitorMode(ABC):
    @property
    def uart_bins(self) -> int: ...
    def boot_radar(self, radar_mgr) -> bool: ...
    def build_frame(self, fft_data, frame_index) -> RadarFrame: ...
    def start(self, serial_mgr, radar_mgr, uart_parser) -> None: ...
    def stop(self) -> None: ...
    def poll_and_update(self, tabs, status_label) -> None: ...
    def get_export_data(self) -> dict: ...
    def clear_data(self) -> None: ...
```

### Tab Visibility

| Mode | Subject Tab | BP Tab | Research Tab |
|------|------------|--------|-------------|
| HR   | visible    | hidden | visible     |
| BP   | hidden     | visible| hidden      |

### Mode Toggle (simplified from 9 steps to 4)

```
1. self._mode.stop()                              — stop I/O thread, stop pipeline, shutdown radar
2. self._mode = BPMode()  (or HRMode)             — create new mode object
3. self._uart_parser = UartParser(bins=self._mode.uart_bins)
4. self._mode.start(serial_mgr, radar_mgr, uart_parser)  — boot radar, start pipeline, start I/O
```

`stop()` closes the serial port to force `read_data` to return immediately,
ensuring the old I/O thread exits before the new one starts.

### UI Timer

```python
def _on_ui_tick(self):
    self._mode.poll_and_update(tabs, status_label)
    # update frame rate / elapsed (shared)
```

### Files Changed

- `ui/main_window.py` — extract all mode-specific logic into HRMode/BPMode
- `ui/monitor_mode.py` — NEW: `MonitorMode` ABC + `HRMode` + `BPMode`
- `ui/bp_tab.py` — minor: add reset/clear method for "--" state
- `ui/subject_tab.py` — minor: add reset/clear method for "--" state

### What Stays the Same

- `bp_monitor/` — BPPipeline, models, CFAR, signal cleaner, network
- `dsp_pipeline/` — Pipeline, vital signs, filters, etc.
- `io_engine/` — SerialManager, RadarMgr, UartParser, BinFileReader
- `ui/` — WaveWidget, SqiIndicator, BreathingPetals, ResearchTab, etc.
- `main.py` — entry point (pass mode/bp_replay flags to MainWindow unchanged)
