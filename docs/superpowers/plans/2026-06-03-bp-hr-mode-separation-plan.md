# BP/HR Mode Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the `_bp_mode` boolean flag scattered across MainWindow by extracting each mode into a self-contained Strategy-pattern class. Mode toggle becomes a clean stop→swap→start sequence.

**Architecture:** Strategy pattern — `MonitorMode` ABC defines the interface; `HRMode` and `BPMode` each own their pipeline, frame builder, display polling, and data buffers. `MainWindow` delegates all mode-specific work to `self._mode` — no `if self._bp_mode:` remains anywhere. The I/O thread stays in MainWindow but delegates frame building and queue routing to the mode object.

**Tech Stack:** Python 3.12, PyQt6, numpy, scipy

**Files:**
- Create: `ui/monitor_mode.py` — `MonitorMode` ABC + `HRMode` + `BPMode`
- Modify: `ui/main_window.py` — strip mode logic, delegate to `MonitorMode`
- Modify: `ui/subject_tab.py` — add `reset_display()` method
- Modify: `ui/bp_tab.py` — add `reset_display()` method

---

### Task 1: Add `reset_display()` to SubjectTab

**Files:**
- Modify: `ui/subject_tab.py`

- [ ] **Step 1: Add `reset_display()` method to SubjectTab**

Add this method to the `SubjectTab` class (after `update_display`):

```python
def reset_display(self) -> None:
    """Clear all displayed values to '--' state."""
    self._breath_bpm_label.setText("--")
    self._heart_bpm_label.setText("--")
    self._breath_wave.set_data(np.array([], dtype=np.float32))
    self._sqi.set_level(0.0, 0.0)
    self._petals.set_breath_bpm(0, 0.0)
    self._heart_icon.set_heart_bpm(0, 0.0)
    self._status_label.setText(tr("status_standby"))
    self._status_label.setStyleSheet("color: #f39c12;")
    self._error_overlay.hide()
    self._error_start_time = None
    self.setStyleSheet("")
```

- [ ] **Step 2: Commit**

```bash
git add ui/subject_tab.py
git commit -m "feat: add reset_display() to SubjectTab for mode switch cleanup"
```

---

### Task 2: Add `reset_display()` to BPTab

**Files:**
- Modify: `ui/bp_tab.py`

- [ ] **Step 1: Add `reset_display()` method to BPTab**

Add this method to the `BPTab` class (after `update_display`):

```python
def reset_display(self) -> None:
    """Clear all BP values to '--' state."""
    self._sbp_panel.set_value(float('nan'))
    self._dbp_panel.set_value(float('nan'))
    self._wave_buffer.clear()
    self._wave.set_data(np.array([], dtype=np.float32))
    self._dist_label.setText("Distance: --")
    self._conf_dots.set_confidence(0.0)
    self._update_label.setText("")
```

Also add `import numpy as np` at the top if not already present (check existing imports — numpy is already imported).

- [ ] **Step 2: Commit**

```bash
git add ui/bp_tab.py
git commit -m "feat: add reset_display() to BPTab for mode switch cleanup"
```

---

### Task 3: Add `reset_display()` to ResearchTab

**Files:**
- Modify: `ui/research_tab.py`

- [ ] **Step 1: Add `reset_display()` method to ResearchTab**

Add this method to the `ResearchTab` class (after `start`):

```python
def reset_display(self) -> None:
    """Clear all displayed values to default state."""
    self._breath_bpm_label.setText("--")
    self._heart_bpm_label.setText("--")
    self._breath_wave.set_data(np.array([], dtype=np.float32))
    self._heart_wave.set_data(np.array([], dtype=np.float32))
    self._sqi.set_level(0.0, 0.0)
    self._trend.start()  # re-initialize trend panel
```

- [ ] **Step 2: Commit**

```bash
git add ui/research_tab.py
git commit -m "feat: add reset_display() to ResearchTab for mode switch cleanup"
```

---

### Task 4: Create `MonitorMode` ABC and `HRMode`

**Files:**
- Create: `ui/monitor_mode.py`

- [ ] **Step 1: Create the file with MonitorMode ABC and HRMode**

```python
"""Monitor mode abstraction — each vital-signs mode (HR, BP) as a Strategy object."""

import threading
import time
import queue
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime

import numpy as np

from config.protocol import RAW_QUEUE_MAXSIZE, DISPLAY_QUEUE_MAXSIZE, UI_REFRESH_MS
from config.i18n import tr
from models.radar_frame import RadarFrame, FrameHeader
from dsp_pipeline.vital_signs import VitalSigns


def _drain_queue(q) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


class MonitorMode(ABC):
    """Abstract vital-signs monitoring mode.

    Each concrete mode owns its pipeline, frame builder, display queue,
    data buffers, and tab visibility policy.
    """

    @property
    @abstractmethod
    def uart_bins(self) -> int:
        """Number of FFT bins per UART frame for this mode."""
        ...

    @abstractmethod
    def boot_radar(self, radar_mgr) -> bool:
        """Send radar boot sequence for this mode. Returns True on success."""
        ...

    @abstractmethod
    def build_frame(self, fft_data: np.ndarray, frame_index: int) -> RadarFrame:
        """Build a RadarFrame from raw FFT data in the correct format."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Create and start the processing pipeline."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Drain display queue and stop the processing pipeline."""
        ...

    @abstractmethod
    def feed_frame(self, frame: RadarFrame) -> None:
        """Push a RadarFrame into the pipeline's raw queue (non-blocking)."""
        ...

    @abstractmethod
    def poll_and_update(self, subject_tab, bp_tab, research_tab,
                        status_label, elapsed_label, frame_rate_label,
                        start_time, frame_count) -> None:
        """Poll the pipeline's display queue and update the appropriate tabs.

        Called from the UI timer on the main thread.
        """
        ...

    @abstractmethod
    def tab_visibility(self) -> tuple[bool, bool, bool]:
        """Return (show_subject, show_bp, show_research) for this mode."""
        ...

    @abstractmethod
    def get_export_data(self) -> dict:
        """Return accumulated data ready for export.

        Returns dict with keys depending on mode:
          HR: csv_rows, breath_waveform_accum, heart_waveform_accum,
              bpm_history, sqi_history, latest_vitals
          BP: bp_results (list of BPResult)
        """
        ...

    @abstractmethod
    def clear_data(self) -> None:
        """Reset all accumulated data buffers."""
        ...


class HRMode(MonitorMode):
    """Heart rate / breath rate monitoring mode (2T4R, 128 range bins)."""

    def __init__(self):
        from dsp_pipeline.pipeline import Pipeline

        self._pipeline: Pipeline | None = None
        self._latest_vitals: VitalSigns | None = None
        self._trend_tick_counter: int = 0

        # Data accumulation for export
        self._csv_rows: list[dict] = []
        self._breath_waveform_accum: list[np.ndarray] = []
        self._heart_waveform_accum: list[np.ndarray] = []
        self._bpm_history: list[tuple[float, float, float]] = []
        self._sqi_history: list[dict] = []

    # -- MonitorMode impl ------------------------------------------------

    @property
    def uart_bins(self) -> int:
        return 1024

    def boot_radar(self, radar_mgr) -> bool:
        return radar_mgr.boot()

    def build_frame(self, fft_data: np.ndarray, frame_index: int) -> RadarFrame:
        """Build HR frame: 2T4R, 128 range bins."""
        cube = fft_data.reshape(2, 4, -1)
        rx_combined = np.mean(cube[0, :, :], axis=0)
        return RadarFrame(
            timestamp=time.time(),
            frame_index=frame_index,
            header=FrameHeader(0, 1, 4, 2, 58000, 128, 1, 3000, 25, 1920, 60),
            data_cube=rx_combined.reshape(-1, 1, 1),
        )

    def start(self) -> None:
        from dsp_pipeline.pipeline import Pipeline
        self._pipeline = Pipeline()
        self._pipeline.start()

    def stop(self) -> None:
        if self._pipeline:
            _drain_queue(self._pipeline.display_queue)
            self._pipeline.stop()
            self._pipeline = None

    def feed_frame(self, frame: RadarFrame) -> None:
        if self._pipeline is None:
            return
        while True:
            try:
                self._pipeline.raw_queue.put_nowait(frame)
                break
            except queue.Full:
                try:
                    self._pipeline.raw_queue.get_nowait()
                except queue.Empty:
                    pass

    def poll_and_update(self, subject_tab, bp_tab, research_tab,
                        status_label, elapsed_label, frame_rate_label,
                        start_time, frame_count) -> None:
        if self._pipeline is None:
            return

        try:
            while not self._pipeline.display_queue.empty():
                self._latest_vitals = self._pipeline.display_queue.get_nowait()
        except queue.Empty:
            pass

        if self._latest_vitals is None:
            return

        q = self._latest_vitals.quality
        calib_done = self._pipeline.calibration_done
        calib_prog = self._pipeline.calibration_progress

        # Subject tab
        subject_tab.update_display(
            breath_bpm=self._latest_vitals.breath_bpm,
            heart_bpm=self._latest_vitals.heart_bpm,
            breath_waveform=self._latest_vitals.breath_waveform,
            quality=q,
            calibration_done=calib_done,
            calibration_progress=calib_prog,
        )

        # Research tab
        self._trend_tick_counter += 1
        trend_sample = (self._trend_tick_counter % 20 == 0)
        research_tab.update_display(
            breath_bpm=self._latest_vitals.breath_bpm,
            heart_bpm=self._latest_vitals.heart_bpm,
            breath_waveform=self._latest_vitals.breath_waveform,
            heart_waveform=self._latest_vitals.heart_waveform,
            quality=q,
            sample_for_trend=trend_sample,
        )

        # Waveform accumulation
        if self._latest_vitals.breath_waveform.size > 0:
            self._breath_waveform_accum.append(
                self._latest_vitals.breath_waveform.copy())
        if self._latest_vitals.heart_waveform.size > 0:
            self._heart_waveform_accum.append(
                self._latest_vitals.heart_waveform.copy())

        # CSV row accumulation (once per second)
        if trend_sample and q is not None:
            phase_range_raw = float(
                np.max(self._latest_vitals.breath_waveform)
                - np.min(self._latest_vitals.breath_waveform)
            ) if len(self._latest_vitals.breath_waveform) > 0 else 0.0

            sqi = 0
            br = q.get("breath_ratio", 0)
            pr = q.get("phase_range", 0)
            if pr >= 0.01 and br >= 0.15:
                sqi = 3
            elif pr >= 0.005 and br >= 0.05:
                sqi = 2
            elif pr > 0 or br > 0:
                sqi = 1

            elapsed_t = time.time() - start_time if start_time > 0 else 0
            self._bpm_history.append((
                elapsed_t,
                self._latest_vitals.breath_bpm,
                self._latest_vitals.heart_bpm,
            ))
            self._sqi_history.append({
                "phase_range": q.get("phase_range", 0.0),
                "breath_ratio": q.get("breath_ratio", 0.0),
                "sqi_level": sqi,
            })

            self._csv_rows.append({
                "Timestamp": datetime.now().isoformat(),
                "FrameIndex": self._latest_vitals.frame_index,
                "RangeBin": self._pipeline.best_range_bin if self._pipeline.best_range_bin is not None else 0,
                "RawPhase": round(phase_range_raw, 6),
                "BreathBPM": self._latest_vitals.breath_bpm,
                "HeartBPM": self._latest_vitals.heart_bpm,
                "PhaseRange": round(q.get("phase_range", 0), 6),
                "BreathRatio": round(q.get("breath_ratio", 0), 4),
                "HeartProminence": round(q.get("heart_prominence", 0), 4),
                "ApneaFlag": 1 if q.get("apnea_state") else 0,
                "SQI_Level": sqi,
            })

        # Status bar
        running = True  # called only when running
        if q and not q.get("valid") and calib_done:
            status_label.setText(tr("status_signal_error"))
            status_label.setStyleSheet("color: #e74c3c;")
        elif running:
            status_label.setText(tr("status_monitoring"))
            status_label.setStyleSheet("color: #27ae60;")

    def tab_visibility(self) -> tuple[bool, bool, bool]:
        return (True, False, True)

    def get_export_data(self) -> dict:
        return {
            "csv_rows": self._csv_rows,
            "breath_waveform_accum": self._breath_waveform_accum,
            "heart_waveform_accum": self._heart_waveform_accum,
            "bpm_history": self._bpm_history,
            "sqi_history": self._sqi_history,
            "latest_vitals": self._latest_vitals,
        }

    def clear_data(self) -> None:
        self._csv_rows.clear()
        self._breath_waveform_accum.clear()
        self._heart_waveform_accum.clear()
        self._bpm_history.clear()
        self._sqi_history.clear()
        self._latest_vitals = None
        self._trend_tick_counter = 0
```

- [ ] **Step 2: Commit**

```bash
git add ui/monitor_mode.py
git commit -m "feat: add MonitorMode ABC and HRMode implementation"
```

---

### Task 5: Add BPMode to monitor_mode.py

**Files:**
- Modify: `ui/monitor_mode.py` (append BPMode class)

- [ ] **Step 1: Add BPMode class**

Add the following class to `ui/monitor_mode.py` after the `HRMode` class:

```python
class BPMode(MonitorMode):
    """Blood pressure monitoring mode (1T1R, 32 range bins)."""

    def __init__(self):
        self._pipeline = None  # type: ignore  # BPPipeline
        self._latest_bp_result = None  # type: ignore  # BPResult
        self._bp_results: list = []

    # -- MonitorMode impl ------------------------------------------------

    @property
    def uart_bins(self) -> int:
        return 32

    def boot_radar(self, radar_mgr) -> bool:
        return radar_mgr.boot_bp()

    def build_frame(self, fft_data: np.ndarray, frame_index: int) -> RadarFrame:
        """Build BP frame: 1T1R, 32 range bins."""
        rx_combined = fft_data.ravel()[:32]
        return RadarFrame(
            timestamp=time.time(),
            frame_index=frame_index,
            header=FrameHeader(0, 1, 1, 1, 60000, 32, 1, 160, 50, 0, 0, 0, 5),
            data_cube=rx_combined.reshape(32, 1, 1),
        )

    def start(self) -> None:
        from bp_monitor.bp_pipeline import BPPipeline
        self._pipeline = BPPipeline("bp_matlab/bp_weights.mat")
        self._pipeline.start()

    def stop(self) -> None:
        if self._pipeline:
            _drain_queue(self._pipeline.display_queue)
            self._pipeline.stop()
            self._pipeline = None

    def feed_frame(self, frame: RadarFrame) -> None:
        if self._pipeline is None:
            return
        while True:
            try:
                self._pipeline.raw_queue.put_nowait(frame)
                break
            except queue.Full:
                try:
                    self._pipeline.raw_queue.get_nowait()
                except queue.Empty:
                    pass

    def poll_and_update(self, subject_tab, bp_tab, research_tab,
                        status_label, elapsed_label, frame_rate_label,
                        start_time, frame_count) -> None:
        if self._pipeline is None:
            return

        try:
            while not self._pipeline.display_queue.empty():
                self._latest_bp_result = self._pipeline.display_queue.get_nowait()
        except queue.Empty:
            pass

        if self._latest_bp_result is not None:
            bp_tab.update_display(self._latest_bp_result)
            self._bp_results.append(self._latest_bp_result)
            status_label.setText("● Monitoring")
            status_label.setStyleSheet("color: #27ae60;")

    def tab_visibility(self) -> tuple[bool, bool, bool]:
        return (False, True, False)

    def get_export_data(self) -> dict:
        return {
            "bp_results": list(self._bp_results),
        }

    def clear_data(self) -> None:
        self._bp_results.clear()
        self._latest_bp_result = None
```

- [ ] **Step 2: Commit**

```bash
git add ui/monitor_mode.py
git commit -m "feat: add BPMode implementation"
```

---

### Task 6: Refactor MainWindow to use MonitorMode

**Files:**
- Modify: `ui/main_window.py`

This is the largest change. The file goes from ~760 lines to ~400 lines. All `if self._bp_mode:` checks are removed and replaced with delegation to `self._mode`.

- [ ] **Step 1: Update imports and constructor**

Replace the imports and `__init__` to use `MonitorMode`:

In the imports, remove:
```python
from dsp_pipeline.pipeline import Pipeline
from dsp_pipeline.vital_signs import VitalSigns
from bp_monitor.bp_pipeline import BPPipeline  # (if present)
```

Add:
```python
from ui.monitor_mode import MonitorMode, HRMode, BPMode
```

Replace the `__init__` method:

```python
class MainWindow(QMainWindow):
    def __init__(self, mode: str = "replay", replay_file: str | None = None,
                 bp_replay: bool = False):
        super().__init__()
        self.setWindowTitle(tr("window_title"))
        self.resize(1200, 800)

        self._mode_type = mode  # "serial" or "replay"
        self._replay_file = replay_file

        # Mode object (Strategy pattern) — the ONLY mode reference
        self._current_mode: MonitorMode = BPMode() if bp_replay else HRMode()

        # Replay
        self._bin_reader: BinFileReader | None = None
        self._replay_timer: QTimer | None = None

        # Shared state
        self._start_time: float = 0.0
        self._frame_count: int = 0
        self._running: bool = False

        # Serial mode
        self._serial_mgr: SerialManager | None = None
        self._radar_mgr: RadarMgr | None = None
        self._uart_parser: UartParser | None = None
        self._io_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._serial_status: str = ""
        self._serial_error: bool = False
        if mode == "serial":
            self._serial_mgr = SerialManager()
            self._radar_mgr = RadarMgr(self._serial_mgr)
            self._uart_parser = UartParser(
                bins_per_frame=self._current_mode.uart_bins)

        self._setup_ui()
        self._setup_timers()
```

- [ ] **Step 2: Add mode-agnostic helper methods**

Remove `_drain_queue` (moved to monitor_mode.py). Remove `_build_radar_frame`, `_build_bp_frame`, `_build_hr_frame` (moved to mode classes).

Remove `_latest_vitals`, `_latest_bp_result`, `_csv_rows`, `_breath_waveform_accum`, `_heart_waveform_accum`, `_bpm_history`, `_sqi_history`, `_trend_tick_counter` — these are all in the mode classes now.

Remove `_bp_mode` (replaced by `isinstance(self._current_mode, BPMode)` only where needed, or better yet, avoid isinstance checks entirely).

- [ ] **Step 3: Replace `_on_start` / `_start_serial` / `_start_serial_io` / `_serial_init_thread` / `_do_serial_init`**

```python
def _on_start(self) -> None:
    if self._mode_type == "serial":
        self._start_serial()
    else:
        self._start_replay()

def _start_serial(self) -> None:
    self._start_btn.setEnabled(False)
    self._stop_btn.setEnabled(True)
    if hasattr(self, '_select_btn'):
        self._select_btn.setEnabled(False)
    if hasattr(self, '_mode_btn'):
        self._mode_btn.setEnabled(False)
    self._status_label.setText(tr("status_starting"))
    self._status_label.setStyleSheet("color: #f39c12;")

    # Boot radar + start pipeline + start I/O on background thread
    thread = threading.Thread(target=self._serial_init_thread, daemon=True)
    thread.start()

def _serial_init_thread(self) -> None:
    try:
        self._do_serial_init()
    except Exception as e:
        import traceback
        print(f"[Serial Init] CRASH: {e}")
        traceback.print_exc()
        self._serial_error = True
        self._serial_status = tr("serial_init_failed", str(e))

def _do_serial_init(self) -> None:
    import serial.tools.list_ports
    print("[Serial Init] Scanning ports...")
    ports = SerialManager.list_ports()
    ctrl_port = data_port = ""
    for p in ports:
        try:
            info = next((i for i in serial.tools.list_ports.comports()
                         if i.device == p), None)
            desc = info.description if info else ""
        except Exception:
            desc = ""
        if "Standard" in desc:
            ctrl_port = p
        if "Enhanced" in desc:
            data_port = p
    print(f"[Serial Init] Found: ctrl={ctrl_port}, data={data_port}")
    if not ctrl_port or not data_port:
        self._serial_error = True
        self._serial_status = tr("serial_not_found", ctrl_port, data_port)
        return
    print("[Serial Init] Connecting...")
    if not self._radar_mgr.connect(ctrl_port, data_port):
        self._serial_error = True
        self._serial_status = tr("serial_connect_failed", ctrl_port, data_port)
        return
    print("[Serial Init] Booting radar...")
    self._current_mode.boot_radar(self._radar_mgr)
    self._stop_event = threading.Event()
    self._uart_parser.reset()
    self._current_mode.start()
    self._current_mode.clear_data()
    self._start_time = time.time()
    self._frame_count = 0
    self._running = True
    self._research_tab.start()
    self._serial_error = False
    self._serial_status = tr("serial_capturing", ctrl_port, data_port)
    self._update_tab_visibility()
    print("[Serial Init] Starting I/O loop...")
    self._io_thread = threading.Thread(target=self._serial_io_loop, daemon=True)
    self._io_thread.start()
    print("[Serial Init] Done!")
```

- [ ] **Step 4: Replace `_serial_io_loop`**

```python
def _serial_io_loop(self) -> None:
    while self._running and not self._stop_event.is_set():
        try:
            raw = self._serial_mgr.read_data(4096)
            if not raw:
                continue
            frames = self._uart_parser.feed(raw)
            for fft_data in frames:
                self._frame_count += 1
                frame = self._current_mode.build_frame(fft_data, self._frame_count)
                self._current_mode.feed_frame(frame)
        except Exception as e:
            print(f"[Serial I/O] {e}")
            time.sleep(0.5)
```

- [ ] **Step 5: Replace `_start_replay` and `_feed_one_frame`**

```python
def _start_replay(self) -> None:
    if not self._replay_file or not os.path.exists(self._replay_file):
        QMessageBox.critical(self, tr("dialog_error"), tr("dialog_no_valid_file"))
        return
    reader = BinFileReader(self._replay_file)
    if not reader.open():
        QMessageBox.critical(self, tr("dialog_error"),
                             tr("dialog_cannot_open", self._replay_file))
        return
    self._bin_reader = reader
    self._current_mode.start()
    self._current_mode.clear_data()
    self._start_time = time.time()
    self._frame_count = 0
    self._running = True
    self._research_tab.start()
    self._update_tab_visibility()
    self._start_btn.setEnabled(False)
    self._stop_btn.setEnabled(True)
    if hasattr(self, '_select_btn'):
        self._select_btn.setEnabled(False)
    self._status_label.setText(tr("status_playing"))
    self._status_label.setStyleSheet("color: #27ae60;")
    self._replay_timer = QTimer()
    self._replay_timer.timeout.connect(self._feed_one_frame)
    self._replay_timer.start(50)

def _feed_one_frame(self) -> None:
    if not self._bin_reader:
        self._replay_timer.stop()
        return
    frames = self._bin_reader.read_frames(max_frames=1)
    if not frames:
        self._replay_timer.stop()
        self._status_label.setText(tr("status_done"))
        self._status_label.setStyleSheet("color: #3498db;")
        self._stop_btn.setEnabled(False)
        self._start_btn.setEnabled(True)
        if hasattr(self, '_select_btn'):
            self._select_btn.setEnabled(True)
        return
    self._frame_count += 1
    frame = self._current_mode.build_frame(frames[0], self._frame_count)
    self._current_mode.feed_frame(frame)
```

- [ ] **Step 6: Replace `_on_toggle_mode`**

```python
def _on_toggle_mode(self) -> None:
    """Hot-switch between HR and BP monitoring modes (serial only)."""
    from ui.monitor_mode import BPMode, HRMode

    was_running = self._running

    # 1. Stop current I/O
    if was_running:
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        # Close serial data port to unblock read_data()
        if self._serial_mgr and self._serial_mgr.data_serial:
            try:
                self._serial_mgr.data_serial.close()
            except Exception:
                pass
        if self._io_thread:
            self._io_thread.join(timeout=5)

    # 2. Stop pipeline + shutdown radar
    self._current_mode.stop()
    if self._radar_mgr:
        self._radar_mgr.shutdown()

    # 3. Swap mode
    was_bp = isinstance(self._current_mode, BPMode)
    self._current_mode = HRMode() if was_bp else BPMode()

    # 4. Rebuild UART parser for new mode
    self._uart_parser = UartParser(bins_per_frame=self._current_mode.uart_bins)

    # 5. Update UI
    self._update_tab_visibility()
    self._mode_btn.setText(
        tr("btn_mode_hr") if isinstance(self._current_mode, BPMode)
        else tr("btn_mode_bp")
    )

    # 6. Reset inactive tabs to "--"
    if was_bp:
        # Switching to HR: reset BP tab
        self._bp_tab.reset_display()
    else:
        # Switching to BP: reset subject + research tabs
        self._subject_tab.reset_display()
        self._research_tab.reset_display()

    # 7. Restart if was running
    if was_running:
        # Re-open data port
        try:
            self._serial_mgr.open_data(
                self._serial_mgr.data_port,
                baudrate=self._radar_mgr.data_baudrate,
            )
        except Exception:
            pass
        self._current_mode.boot_radar(self._radar_mgr)
        self._current_mode.start()
        self._current_mode.clear_data()
        self._running = True
        self._stop_event = threading.Event()
        self._uart_parser.reset()
        self._frame_count = 0
        self._start_time = time.time()
        self._io_thread = threading.Thread(
            target=self._serial_io_loop, daemon=True)
        self._io_thread.start()
        self._status_label.setText("● Capturing")
        self._status_label.setStyleSheet("color: #27ae60;")
    else:
        self._status_label.setText(tr("status_standby"))
        self._status_label.setStyleSheet("color: #f39c12;")
```

- [ ] **Step 7: Replace `_on_stop`**

```python
def _on_stop(self) -> None:
    self._running = False
    if self._mode_type == "serial":
        if self._radar_mgr:
            self._radar_mgr.shutdown()
        if self._stop_event:
            self._stop_event.set()
        # Close serial to unblock read_data
        if self._serial_mgr and self._serial_mgr.data_serial:
            try:
                self._serial_mgr.data_serial.close()
            except Exception:
                pass
        if self._io_thread:
            self._io_thread.join(timeout=5)
        if self._serial_mgr:
            self._serial_mgr.close()
    if self._replay_timer:
        self._replay_timer.stop()
        self._replay_timer = None
    self._current_mode.stop()
    if self._bin_reader:
        self._bin_reader.close()
        self._bin_reader = None
    self._start_btn.setEnabled(True)
    self._stop_btn.setEnabled(False)
    if hasattr(self, '_select_btn'):
        self._select_btn.setEnabled(True)
    if hasattr(self, '_mode_btn'):
        self._mode_btn.setEnabled(True)
    self._status_label.setText(tr("status_stopped"))
    self._status_label.setStyleSheet("color: #f39c12;")
```

- [ ] **Step 8: Replace `_on_save` and export methods**

```python
def _on_save(self) -> None:
    from PyQt6.QtWidgets import QInputDialog

    fmt = tr("export_format_csv")
    choice, ok = QInputDialog.getItem(
        self, tr("export_title"), "Format:",
        [tr("export_format_csv"), tr("export_format_hdf5"),
         tr("export_format_edf")],
        0, False,
    )
    if not ok or not choice:
        return

    path = QFileDialog.getExistingDirectory(self, tr("dialog_save_dir"))
    if not path:
        return

    try:
        data = self._current_mode.get_export_data()
        if choice == tr("export_format_csv"):
            breath = data["latest_vitals"].breath_waveform if data.get("latest_vitals") else np.array([])
            heart = data["latest_vitals"].heart_waveform if data.get("latest_vitals") else np.array([])
            export_csv(path, data.get("csv_rows", []), breath, heart)
        elif choice == tr("export_format_hdf5"):
            breath_hist = (np.array(data["breath_waveform_accum"])
                           if data.get("breath_waveform_accum") else np.array([]))
            heart_hist = (np.array(data["heart_waveform_accum"])
                          if data.get("heart_waveform_accum") else np.array([]))
            metadata = {
                "device": "RS6240",
                "fs": 20,
                "session_duration_s": time.time() - self._start_time if self._start_time > 0 else 0,
            }
            export_hdf5(path, breath_hist, heart_hist,
                        data.get("bpm_history", []),
                        data.get("sqi_history", []), metadata)
        elif choice == tr("export_format_edf"):
            breath = data["latest_vitals"].breath_waveform if data.get("latest_vitals") else np.array([])
            heart = data["latest_vitals"].heart_waveform if data.get("latest_vitals") else np.array([])
            export_edf(path, breath, heart, fs=20.0)
        QMessageBox.information(self, tr("dialog_save_done"),
                                tr("dialog_save_done_msg", path))
    except ImportError as e:
        QMessageBox.critical(self, tr("dialog_error"), str(e))
    except Exception as e:
        QMessageBox.critical(self, tr("dialog_error"), str(e))
```

Remove `_do_export_csv`, `_do_export_hdf5`, `_do_export_edf` — their logic is inlined in `_on_save` above.

- [ ] **Step 9: Replace `_on_ui_tick`**

```python
def _on_ui_tick(self) -> None:
    # Poll serial status
    if self._serial_status:
        s = self._serial_status
        if self._serial_error:
            self._status_label.setText(f"● {s}")
            self._status_label.setStyleSheet("color: #e74c3c;")
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            if hasattr(self, '_select_btn'):
                self._select_btn.setEnabled(True)
        else:
            self._status_label.setText(f"● {s}")
            self._status_label.setStyleSheet("color: #27ae60;")
        self._serial_status = ""

    if not self._running:
        return

    # Delegate display polling to current mode
    self._current_mode.poll_and_update(
        self._subject_tab, self._bp_tab, self._research_tab,
        self._status_label, self._elapsed_label, self._frame_rate_label,
        self._start_time, self._frame_count,
    )

    # Shared: frame rate + elapsed
    if self._start_time > 0:
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            self._frame_rate_label.setText(
                tr("frame_rate", f"{self._frame_count / elapsed:.1f}"))
        m, s_div = divmod(int(elapsed), 60)
        self._elapsed_label.setText(tr("elapsed", f"{m:02d}:{s_div:02d}"))
```

Remove `_poll_bp_results` (moved into BPMode.poll_and_update).

- [ ] **Step 10: Update `_setup_ui` mode toggle button**

In `_setup_ui`, update the mode toggle button label to use isinstance instead of `_bp_mode`:

```python
# Mode toggle button (serial only)
if self._mode_type == "serial":
    from ui.monitor_mode import BPMode
    self._mode_btn = QPushButton(
        tr("btn_mode_bp") if isinstance(self._current_mode, HRMode)
        else tr("btn_mode_hr")
    )
    ...
```

And update `update_ui_texts` to refresh the mode button text correctly.

- [ ] **Step 11: Update `_setup_ui` to set initial tab visibility**

At the end of `_setup_ui`, after creating all tabs, call:

```python
self._update_tab_visibility()
```

And add the helper:

```python
def _update_tab_visibility(self) -> None:
    show_subject, show_bp, show_research = self._current_mode.tab_visibility()
    self._tabs.setTabVisible(0, show_subject)
    self._tabs.setTabVisible(1, show_bp)
    self._tabs.setTabVisible(2, show_research)
```

Note: PyQt6 `QTabWidget.setTabVisible` was added in Qt 6.4. If using older versions, use `removeTab`/`insertTab` instead:

```python
def _update_tab_visibility(self) -> None:
    show_subject, show_bp, show_research = self._current_mode.tab_visibility()
    # Qt 6.4+:
    if hasattr(self._tabs, 'setTabVisible'):
        self._tabs.setTabVisible(0, show_subject)
        self._tabs.setTabVisible(1, show_bp)
        self._tabs.setTabVisible(2, show_research)
    else:
        # Qt < 6.4 fallback: remove/re-insert
        current = self._tabs.currentIndex()
        self._tabs.clear()
        if show_subject:
            self._tabs.addTab(self._subject_tab, tr("tab_subject"))
        if show_bp:
            self._tabs.addTab(self._bp_tab, tr("tab_bp"))
        if show_research:
            self._tabs.addTab(self._research_tab, tr("tab_research"))
        self._tabs.setCurrentIndex(min(current, self._tabs.count() - 1))
```

- [ ] **Step 12: Commit**

```bash
git add ui/main_window.py
git commit -m "refactor: extract mode-specific logic into MonitorMode strategy classes"
```

---

### Task 7: Verify — import and basic structural check

**Files:**
- No changes; verification only

- [ ] **Step 1: Run Python syntax check on all changed files**

```bash
cd F:/yan/mmradarcoverage/real_time_monitor && python -c "import ui.monitor_mode; print('monitor_mode OK')"
```

Expected: `monitor_mode OK`

```bash
cd F:/yan/mmradarcoverage/real_time_monitor && python -c "from ui.monitor_mode import HRMode, BPMode, MonitorMode; hr = HRMode(); bp = BPMode(); print(f'HR bins={hr.uart_bins}, BP bins={bp.uart_bins}'); print(f'HR tabs={hr.tab_visibility()}, BP tabs={bp.tab_visibility()}')"
```

Expected: `HR bins=1024, BP bins=32` and `HR tabs=(True, False, True), BP tabs=(False, True, False)`

- [ ] **Step 2: Run Python syntax check on MainWindow**

```bash
cd F:/yan/mmradarcoverage/real_time_monitor && python -c "from ui.main_window import MainWindow; print('MainWindow OK')"
```

Expected: `MainWindow OK` (no import errors, no syntax errors)

---

### Task 8: Commit and finalize

- [ ] **Step 1: Verify all changes are committed**

```bash
git status
```

Expected: clean working tree (only the committed changes from tasks 1-5).

- [ ] **Step 2: Run the application in replay mode to test HR**

```bash
cd F:/yan/mmradarcoverage/real_time_monitor && timeout 10 python main.py -r data/*.bin 2>&1 || true
```

Expected: No crashes, window appears, HR data shows in subject tab.

- [ ] **Step 3: Run the application in replay mode to test BP**

```bash
cd F:/yan/mmradarcoverage/real_time_monitor && timeout 10 python main.py -r data/*.bin --bp 2>&1 || true
```

Expected: No crashes, BP tab visible, subject tab hidden.
