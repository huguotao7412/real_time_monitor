# UX Humanization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform engineer-facing raw DSP display into a medical-grade consumer UX with "living" visual feedback and dual-mode (Subject/Research) tabbed interface.

**Architecture:** Refactor 460-line monolithic MainWindow into a thin shell (~80 lines) hosting a QTabWidget with SubjectTab and ResearchTab. Six new focused widget files handle calibration overlay, SQI indicator, breathing petals, trend panel, status message mapping, and waveform fill rendering. DSP pipeline receives only 2 minor property additions.

**Tech Stack:** PyQt6, pyqtgraph, numpy, Python 3.12, existing `csv` stdlib module

---

## File Map

| File | Responsibility |
|------|---------------|
| `dsp_pipeline/pipeline.py` | Add `calibration_done` + `calibration_progress` properties, fix `heart_waveform` passthrough |
| `ui/wave_widget.py` | Enhance: add `fill_mode`, `show_axes`, `show_grid` params |
| `ui/status_mapper.py` **[new]** | Pure function: DSP quality dict → (message: str, level: str) |
| `ui/sqi_indicator.py` **[new]** | 3-bar signal quality light widget |
| `ui/breathing_petals.py` **[new]** | Apple Watch style petal animation widget |
| `ui/calibration_overlay.py` **[new]** | Ring progress overlay for 10s calibration |
| `ui/trend_panel.py` **[new]** | BPM history scatter plot (5/15/30 min) |
| `ui/subject_tab.py` **[new]** | Subject mode container layout |
| `ui/research_tab.py` **[new]** | Research mode container layout |
| `ui/main_window.py` | Refactor to thin shell with QTabWidget |

---

### Task 1: Pipeline — calibration properties + heart_waveform fix

**Files:**
- Modify: `dsp_pipeline/pipeline.py`

- [ ] **Step 1: Add `calibration_done` and `calibration_progress` properties**

At the end of the `Pipeline` class (before any private method), add these two read-only properties. Insert after line 76 (after `_cached_heart_wave`):

```python
    @property
    def calibration_done(self) -> bool:
        return self._calibration_done

    @property
    def calibration_progress(self) -> float:
        """0.0 - 1.0 fraction of calibration samples collected (target 200)."""
        return min(1.0, len(self._calibration_samples) / 200.0)
```

- [ ] **Step 2: Fix heart_waveform passthrough in `_shared_signal_chain`**

Change line 342 from `heart_waveform=np.array([])` to use the computed `heart_signal_display`:

Find the `return VitalSigns(` block at approximately line 339-344. Replace:

```python
        return VitalSigns(
            timestamp=time.time(), frame_index=self._frame_count,
            breath_waveform=breath_signal_display, breath_bpm=round(breath_bpm, 1),
            heart_bpm=round(heart_bpm, 1), heart_waveform=np.array([]),
            quality=quality,
        )
```

With:

```python
        return VitalSigns(
            timestamp=time.time(), frame_index=self._frame_count,
            breath_waveform=breath_signal_display, breath_bpm=round(breath_bpm, 1),
            heart_bpm=round(heart_bpm, 1), heart_waveform=heart_signal_display,
            quality=quality,
        )
```

- [ ] **Step 3: Commit**

```bash
git add dsp_pipeline/pipeline.py
git commit -m "feat: expose calibration_done/progress properties, fix heart_waveform passthrough"
```

---

### Task 2: status_mapper.py — DSP conditions to human messages

**Files:**
- Create: `ui/status_mapper.py`

- [ ] **Step 1: Write the pure function module**

Create `ui/status_mapper.py`:

```python
"""Map DSP quality dict conditions to human-readable Chinese status messages."""

from typing import Any


def map_status(quality: dict[str, Any] | None) -> tuple[str, str]:
    """Return (message: str, level: str) where level is 'normal'|'warning'|'error'.

    Priority order: apnea > phase_range > breath_ratio. First match wins.
    """
    if quality is None:
        return ("● 待机", "normal")

    if quality.get("apnea_state"):
        return ("监测到呼吸微弱或屏息状态", "error")

    phase_range = quality.get("phase_range", 0.0)
    breath_ratio = quality.get("breath_ratio", 0.0)

    if phase_range < 0.005:
        return ("未检测到微动，请确认在雷达覆盖范围内 (0.5m-1.5m)", "warning")
    if breath_ratio < 0.03:
        return ("信号较弱，请调整坐姿，正对雷达", "warning")

    return ("● 监测中", "normal")


class BodyMovementDetector:
    """Detect sudden body movement from phase_range history."""

    def __init__(self, window_size: int = 60, sigma_threshold: float = 3.0):
        from collections import deque
        self._history: deque[float] = deque(maxlen=window_size)
        self._sigma_threshold = sigma_threshold

    def feed(self, phase_range: float) -> bool:
        """Return True if current phase_range is a body movement spike."""
        import numpy as np
        self._history.append(phase_range)
        if len(self._history) < 10:
            return False
        mean = np.mean(self._history)
        std = np.std(self._history)
        if std < 1e-9:
            return False
        return phase_range > mean + self._sigma_threshold * std


def map_status_with_movement(
    quality: dict[str, Any] | None,
    movement_detected: bool,
) -> tuple[str, str]:
    """Like map_status but with body movement detection layered in."""
    if movement_detected:
        return ("检测到体动干扰，请保持放松", "warning")
    return map_status(quality)
```

- [ ] **Step 2: Commit**

```bash
git add ui/status_mapper.py
git commit -m "feat: add status_mapper for DSP-to-human message translation"
```

---

### Task 3: WaveWidget enhancements — fill mode, axis/grid toggle

**Files:**
- Modify: `ui/wave_widget.py`

- [ ] **Step 1: Rewrite WaveWidget with new parameters**

Replace `ui/wave_widget.py` entirely:

```python
"""Single-channel waveform display with optional fill and axis toggle."""

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QLinearGradient
import numpy as np


class WaveWidget(QWidget):
    """Waveform plot widget with optional gradient fill and axis visibility."""

    def __init__(
        self,
        title: str = "",
        y_label: str = "",
        max_points: int = 200,
        fill_mode: bool = False,
        show_axes: bool = True,
        show_grid: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.max_points = max_points
        self._data = np.zeros(max_points)
        self._fill_mode = fill_mode

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if title:
            self.title_label = QLabel(title)
            self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
            layout.addWidget(self.title_label)
        else:
            self.title_label = None

        self.plot = pg.PlotWidget()
        self.plot.setMouseEnabled(x=False, y=False)

        if show_axes and y_label:
            self.plot.setLabel("left", y_label)
        elif not show_axes:
            self.plot.hideAxis("left")
            self.plot.hideAxis("bottom")

        if show_grid:
            self.plot.showGrid(x=True, y=True, alpha=0.3)
        else:
            self.plot.showGrid(x=False, y=False)

        pen_color = (39, 174, 96)  # #27ae60
        self.curve = self.plot.plot(pen=pg.mkPen(color=pen_color, width=1.5))

        if fill_mode:
            brush = pg.mkBrush(QColor(39, 174, 96, 60))
            self.fill = pg.FillBetweenItem(
                self.curve,
                pg.PlotDataItem([0], [0]),
                brush=brush,
            )
            self.plot.addItem(self.fill)

        layout.addWidget(self.plot)

    def set_data(self, y: np.ndarray) -> None:
        """Update waveform data."""
        if len(y) == 0:
            return
        self._data = y[-self.max_points:]
        x = np.arange(len(self._data))
        self.curve.setData(x, self._data)
        if self._fill_mode and hasattr(self, "fill"):
            self.fill.setCurves(
                self.curve,
                pg.PlotDataItem(x, np.zeros_like(self._data)),
            )
```

- [ ] **Step 2: Commit**

```bash
git add ui/wave_widget.py
git commit -m "feat: add fill_mode, show_axes, show_grid options to WaveWidget"
```

---

### Task 4: SqiIndicator widget

**Files:**
- Create: `ui/sqi_indicator.py`

- [ ] **Step 1: Create the SQI indicator widget**

Create `ui/sqi_indicator.py`:

```python
"""3-bar signal quality indicator (like WiFi signal icon)."""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class SqiIndicator(QWidget):
    """Signal quality indicator with 3 colored bars and a label."""

    COLORS = {
        3: "#27ae60",  # good
        2: "#f39c12",  # fair
        1: "#e74c3c",  # poor
        0: "#7f8c8d",  # no signal
    }

    LABELS = {
        3: "信号: 优",
        2: "信号: 中",
        1: "信号: 差",
        0: "信号: --",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._bars: list[QLabel] = []
        for i in range(3):
            bar = QLabel()
            bar.setFixedSize(6, 8 + i * 5)
            bar.setStyleSheet(f"background-color: #7f8c8d; border-radius: 1px;")
            layout.addWidget(bar)
            self._bars.append(bar)

        self._label = QLabel(self.LABELS[0])
        self._label.setFont(QFont("Segoe UI", 9))
        self._label.setStyleSheet("color: #95a5a6;")
        layout.addWidget(self._label)

        layout.addStretch()

    def set_level(self, breath_ratio: float, phase_range: float) -> int:
        """Update from quality metrics. Returns level 0-3."""
        if phase_range >= 0.01 and breath_ratio >= 0.15:
            level = 3
        elif phase_range >= 0.005 and breath_ratio >= 0.05:
            level = 2
        elif phase_range > 0 or breath_ratio > 0:
            level = 1
        else:
            level = 0

        color = self.COLORS[level]
        for i, bar in enumerate(self._bars):
            if i < level:
                bar.setStyleSheet(f"background-color: {color}; border-radius: 1px;")
            else:
                bar.setStyleSheet("background-color: #7f8c8d; border-radius: 1px;")

        self._label.setText(self.LABELS[level])
        self._label.setStyleSheet(f"color: {color};")
        return level
```

- [ ] **Step 2: Commit**

```bash
git add ui/sqi_indicator.py
git commit -m "feat: add SqiIndicator 3-bar signal quality widget"
```

---

### Task 5: BreathingPetals widget

**Files:**
- Create: `ui/breathing_petals.py`

- [ ] **Step 1: Create the breathing petals animation widget**

Create `ui/breathing_petals.py`:

```python
"""Apple Watch style breathing petals animation driven by breath BPM."""

import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QBrush, QPen


class BreathingPetals(QWidget):
    """Draws overlapping arc petals that open/close at breath_bpm frequency."""

    def __init__(self, size: int = 80, num_petals: int = 6, parent=None):
        super().__init__(parent)
        self._size = size
        self._num_petals = num_petals
        self._phase = 0.0
        self._breath_bpm = 0.0
        self._openness = 0.5  # 0.0=closed, 1.0=open
        self.setFixedSize(size, size)

    def set_breath_bpm(self, bpm: float, dt: float) -> None:
        """Advance animation by dt seconds at given BPM."""
        if bpm > 0:
            self._breath_bpm = bpm
            self._phase += bpm / 60.0 * 2 * math.pi * dt
        self._openness = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(self._phase))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self._size / 2, self._size / 2
        max_radius = self._size * 0.42

        for i in range(self._num_petals):
            angle = (i / self._num_petals) * 2 * math.pi
            dx = max_radius * 0.18 * self._openness * math.cos(angle)
            dy = max_radius * 0.18 * self._openness * math.sin(angle)

            alpha = 80 + int(100 * self._openness)
            color = QColor(39, 174, 96, alpha)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))

            path = QPainterPath()
            path.moveTo(QPointF(cx, cy))
            r = max_radius * (0.3 + 0.7 * self._openness)
            start_angle = angle - math.pi * 0.22
            sweep_angle = math.pi * 0.44
            rect = QPointF(cx - r + dx, cy - r + dy)
            path.arcTo(rect.x(), rect.y(), r * 2, r * 2,
                       math.degrees(start_angle), math.degrees(sweep_angle))
            path.closeSubpath()
            painter.drawPath(path)

        painter.end()
```

- [ ] **Step 2: Commit**

```bash
git add ui/breathing_petals.py
git commit -m "feat: add BreathingPetals Apple Watch style animation widget"
```

---

### Task 6: CalibrationOverlay widget

**Files:**
- Create: `ui/calibration_overlay.py`

- [ ] **Step 1: Create the calibration overlay**

Create `ui/calibration_overlay.py`:

```python
"""Ring progress overlay shown during 10-second calibration phase."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QRectF, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QPainter, QPen, QColor, QFont


class CalibrationOverlay(QWidget):
    """Semi-transparent overlay with ring progress and countdown text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0
        self._countdown = 10
        self._opacity = 1.0

        self.setStyleSheet("background-color: rgba(30, 30, 40, 220);")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._text = QLabel("正在校准，请保持静止...")
        self._text.setFont(QFont("Segoe UI", 14))
        self._text.setStyleSheet("color: #bdc3c7; background: transparent;")
        self._text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._text)

        self._countdown_label = QLabel("10")
        self._countdown_label.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
        self._countdown_label.setStyleSheet("color: #3498db; background: transparent;")
        self._countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._countdown_label)

        self.setVisible(False)

    def set_progress(self, fraction: float) -> None:
        """Update ring progress 0.0-1.0 and derived countdown."""
        self._progress = fraction
        self._countdown = max(0, int(10 * (1.0 - fraction)))
        self._countdown_label.setText(str(self._countdown))
        self.update()

    def get_opacity(self) -> float:
        return self._opacity

    def set_opacity(self, val: float) -> None:
        self._opacity = val
        self.update()

    opacity = pyqtProperty(float, get_opacity, set_opacity)

    def fade_out(self) -> None:
        self._anim = QPropertyAnimation(self, b"opacity")
        self._anim.setDuration(500)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(lambda: self.setVisible(False))
        self._anim.start()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self._opacity)

        cx = self.width() // 2
        cy = self.height() // 2 - 50
        radius = 60
        pen_width = 6

        # Background ring
        painter.setPen(QPen(QColor(60, 60, 70), pen_width))
        painter.drawArc(QRectF(cx - radius, cy - radius, radius * 2, radius * 2),
                        90 * 16, -360 * 16)

        # Progress ring
        painter.setPen(QPen(QColor(52, 152, 219), pen_width))
        painter.drawArc(QRectF(cx - radius, cy - radius, radius * 2, radius * 2),
                        90 * 16, -int(360 * self._progress) * 16)

        painter.end()
```

- [ ] **Step 2: Commit**

```bash
git add ui/calibration_overlay.py
git commit -m "feat: add CalibrationOverlay ring progress widget"
```

---

### Task 7: TrendPanel widget

**Files:**
- Create: `ui/trend_panel.py`

- [ ] **Step 1: Create the trend panel**

Create `ui/trend_panel.py`:

```python
"""BPM history scatter plot with time window selector."""

from collections import deque
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import time


class TrendPanel(QWidget):
    """Scatter plot showing breath/heart BPM over configurable time windows."""

    WINDOWS = {"5 分钟": 300, "15 分钟": 900, "30 分钟": 1800}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._breath_deque: deque[tuple[float, float]] = deque(maxlen=1800)
        self._heart_deque: deque[tuple[float, float]] = deque(maxlen=1800)
        self._start_time: float = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QHBoxLayout()
        title = QLabel("历史趋势")
        title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        header.addWidget(title)
        header.addStretch()
        self._combo = QComboBox()
        self._combo.addItems(list(self.WINDOWS.keys()))
        self._combo.setCurrentText("5 分钟")
        self._combo.currentTextChanged.connect(self._on_window_changed)
        header.addWidget(self._combo)
        layout.addLayout(header)

        # Plot
        self._plot = pg.PlotWidget()
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setLabel("left", "BPM")
        self._plot.setLabel("bottom", "时间")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setYRange(0, 150)

        self._breath_scatter = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(39, 174, 96, 120), size=6,
        )
        self._heart_scatter = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(231, 76, 60, 120), size=6,
        )
        self._plot.addItem(self._breath_scatter)
        self._plot.addItem(self._heart_scatter)
        layout.addWidget(self._plot)

    def start(self) -> None:
        self._breath_deque.clear()
        self._heart_deque.clear()
        self._start_time = time.time()

    def add_sample(self, breath_bpm: float, heart_bpm: float) -> None:
        if self._start_time == 0:
            return
        t = time.time() - self._start_time
        self._breath_deque.append((t, breath_bpm))
        self._heart_deque.append((t, heart_bpm))
        self._redraw()

    def _redraw(self) -> None:
        window_sec = self.WINDOWS[self._combo.currentText()]
        now = time.time() - self._start_time if self._start_time > 0 else 0
        cutoff = now - window_sec

        for dq, scatter in [
            (self._breath_deque, self._breath_scatter),
            (self._heart_deque, self._heart_scatter),
        ]:
            pts = [(t, bpm) for t, bpm in dq if t >= cutoff and bpm > 0]
            if pts:
                xs, ys = zip(*pts)
                scatter.setData(x=xs, y=ys)
            else:
                scatter.setData(x=[], y=[])

        self._plot.setXRange(max(0, cutoff), max(cutoff + window_sec, now))

    def _on_window_changed(self) -> None:
        self._redraw()
```

- [ ] **Step 2: Commit**

```bash
git add ui/trend_panel.py
git commit -m "feat: add TrendPanel BPM history scatter plot widget"
```

---

### Task 8: SubjectTab container

**Files:**
- Create: `ui/subject_tab.py`

- [ ] **Step 1: Create the SubjectTab container widget**

Create `ui/subject_tab.py`:

```python
"""Subject mode tab — minimal, calming UI for the person being monitored."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import time
import numpy as np

from ui.wave_widget import WaveWidget
from ui.sqi_indicator import SqiIndicator
from ui.breathing_petals import BreathingPetals
from ui.calibration_overlay import CalibrationOverlay
from ui.status_mapper import map_status_with_movement, BodyMovementDetector


class SubjectTab(QWidget):
    """Minimal subject-facing UI: BPM numbers, petal animation, filled waveform."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_update_time = time.time()
        self._movement_detector = BodyMovementDetector()
        self._sqi_level = 0
        self._calibration_was_done = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)

        # Top row: SQI indicator (right-aligned)
        top_row = QHBoxLayout()
        top_row.addStretch()
        self._sqi = SqiIndicator()
        top_row.addWidget(self._sqi)
        layout.addLayout(top_row)

        # BPM center area
        bpm_area = QVBoxLayout()
        bpm_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bpm_area.addStretch()

        # Breath BPM row
        breath_row = QHBoxLayout()
        breath_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._petals = BreathingPetals(size=80)
        breath_row.addWidget(self._petals)
        self._breath_bpm_label = QLabel("--")
        self._breath_bpm_label.setFont(QFont("Segoe UI", 42, QFont.Weight.Bold))
        self._breath_bpm_label.setStyleSheet("color: #27ae60;")
        self._breath_bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        breath_row.addWidget(self._breath_bpm_label)
        bpm_area.addLayout(breath_row)

        breath_unit = QLabel("呼吸频率 次/分钟")
        breath_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        breath_unit.setStyleSheet("color: #7f8c8d; font-size: 10pt;")
        bpm_area.addWidget(breath_unit)

        bpm_area.addSpacing(20)

        # Heart BPM row
        heart_row = QHBoxLayout()
        heart_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._heart_icon = HeartBeatIcon(size=50)
        heart_row.addWidget(self._heart_icon)
        self._heart_bpm_label = QLabel("--")
        self._heart_bpm_label.setFont(QFont("Segoe UI", 42, QFont.Weight.Bold))
        self._heart_bpm_label.setStyleSheet("color: #27ae60;")
        self._heart_bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heart_row.addWidget(self._heart_bpm_label)
        bpm_area.addLayout(heart_row)

        heart_unit = QLabel("心率 次/分钟")
        heart_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heart_unit.setStyleSheet("color: #7f8c8d; font-size: 10pt;")
        bpm_area.addWidget(heart_unit)

        bpm_area.addStretch()
        layout.addLayout(bpm_area)

        # Filled waveform (no axes, no grid, no title)
        self._breath_wave = WaveWidget(
            title="", fill_mode=True, show_axes=False, show_grid=False,
        )
        layout.addWidget(self._breath_wave, stretch=1)

        # Status message
        self._status_label = QLabel("● 待机")
        self._status_label.setFont(QFont("Segoe UI", 11))
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("color: #f39c12;")
        layout.addWidget(self._status_label)

        # Calibration overlay (stacked on top via parent layout)
        self._calibration_overlay = CalibrationOverlay(self)
        self._calibration_overlay.hide()

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #3a3a4a;")
        layout.addWidget(line)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._calibration_overlay.setGeometry(self.rect())

    def update_display(
        self,
        breath_bpm: float,
        heart_bpm: float,
        breath_waveform: np.ndarray,
        quality: dict | None,
        calibration_done: bool,
        calibration_progress: float,
    ) -> None:
        now = time.time()
        dt = now - self._last_update_time
        self._last_update_time = now
        if dt > 0.5:
            dt = 0.033

        # Calibration state
        if not calibration_done:
            if not self._calibration_overlay.isVisible():
                self._calibration_overlay.setVisible(True)
            self._calibration_overlay.set_progress(calibration_progress)
            return

        if self._calibration_overlay.isVisible():
            if not self._calibration_was_done:
                self._calibration_overlay.fade_out()
                self._calibration_was_done = True

        # SQI
        phase_range = quality.get("phase_range", 0.0) if quality else 0.0
        breath_ratio = quality.get("breath_ratio", 0.0) if quality else 0.0
        self._sqi_level = self._sqi.set_level(breath_ratio, phase_range)

        # Body movement detection
        movement = self._movement_detector.feed(phase_range)
        msg, level = map_status_with_movement(quality, movement)

        # BPM opacity
        bpm_opacity = 0.4 if level == "error" or self._sqi_level <= 1 else 1.0

        # Breath BPM + petals
        if breath_bpm > 0:
            self._breath_bpm_label.setText(f"{breath_bpm:.0f}")
            self._petals.set_breath_bpm(breath_bpm, dt)
        else:
            self._breath_bpm_label.setText("--")
            self._petals.set_breath_bpm(0, dt)

        self._breath_bpm_label.setStyleSheet(
            f"color: rgba(39, 174, 96, {int(bpm_opacity * 255)});"
        )

        # Heart BPM + icon
        if heart_bpm > 0:
            self._heart_bpm_label.setText(f"{heart_bpm:.0f}")
            self._heart_icon.set_heart_bpm(heart_bpm, dt)
        else:
            self._heart_bpm_label.setText("--")
            self._heart_icon.set_heart_bpm(0, dt)

        heart_color = self._heart_icon.current_color()
        self._heart_bpm_label.setStyleSheet(
            f"color: rgba({heart_color.red()}, {heart_color.green()}, "
            f"{heart_color.blue()}, {int(bpm_opacity * 255)});"
        )

        # Waveform
        if len(breath_waveform) > 0:
            self._breath_wave.set_data(breath_waveform)

        # Status
        status_colors = {"normal": "#27ae60", "warning": "#f39c12", "error": "#e74c3c"}
        self._status_label.setText(msg)
        self._status_label.setStyleSheet(f"color: {status_colors.get(level, '#f39c12')};")

        # Background hint for error
        if level == "error":
            self.setStyleSheet("background-color: rgba(231, 76, 60, 0.05);")
        else:
            self.setStyleSheet("")


class HeartBeatIcon(QWidget):
    """Bezier heart shape that pulses at heart_bpm frequency."""

    def __init__(self, size: int = 50, parent=None):
        super().__init__(parent)
        self._size = size
        self._phase = 0.0
        self._heart_bpm = 0.0
        self._scale = 1.0
        self._current_color = QColor(39, 174, 96)
        self.setFixedSize(size, size)

    def set_heart_bpm(self, bpm: float, dt: float) -> None:
        import math
        if bpm > 0:
            self._heart_bpm = bpm
            self._phase += bpm / 60.0 * 2 * math.pi * dt
        beat = 0.5 + 0.5 * math.sin(self._phase)
        self._scale = 1.0 + 0.3 * max(0, beat ** 8)
        self._current_color = _heart_rate_color(bpm)
        self.update()

    def current_color(self):
        from PyQt6.QtGui import QColor
        return self._current_color

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QPainterPath, QBrush, QPen
        from PyQt6.QtCore import QPointF
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self._size / 2, self._size / 2
        painter.translate(cx, cy)
        painter.scale(self._scale, self._scale)
        painter.translate(-cx, -cy)

        s = self._size * 0.35
        x = cx - s * 0.8
        y = cy - s * 0.7

        path = QPainterPath()
        path.moveTo(x + s / 2, y + s)
        path.cubicTo(x, y + s * 0.6, x, y, x + s / 2, y)
        path.cubicTo(x + s, y, x + s, y + s * 0.6, x + s / 2, y + s)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._current_color))
        painter.drawPath(path)
        painter.end()


def _heart_rate_color(bpm: float):
    from PyQt6.QtGui import QColor
    if bpm <= 0:
        return QColor(127, 140, 141)
    if 60 <= bpm <= 100:
        return QColor(39, 174, 96)
    if 50 <= bpm < 60 or 100 < bpm <= 110:
        return QColor(243, 156, 18)
    return QColor(231, 76, 60)
```

- [ ] **Step 2: Commit**

```bash
git add ui/subject_tab.py
git commit -m "feat: add SubjectTab with petals, heart icon, calibration overlay"
```

---

### Task 9: ResearchTab container

**Files:**
- Create: `ui/research_tab.py`

- [ ] **Step 1: Create the ResearchTab container**

Create `ui/research_tab.py`:

```python
"""Research mode tab — full DSP data, waveforms with axes, trend panel, debug."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, QFrame,
    QPushButton, QTextEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import numpy as np

from ui.wave_widget import WaveWidget
from ui.sqi_indicator import SqiIndicator
from ui.trend_panel import TrendPanel


class ResearchTab(QWidget):
    """Full researcher view: dual waveforms, BPM, trend, collapsible debug panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sqi_level = 0
        self._debug_expanded = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Waveforms
        wave_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._breath_wave = WaveWidget("呼吸波形 (0.1-0.6 Hz)", "Amplitude")
        self._heart_wave = WaveWidget("心率波形 (0.8-2.5 Hz)", "Amplitude")
        wave_splitter.addWidget(self._breath_wave)
        wave_splitter.addWidget(self._heart_wave)
        layout.addWidget(wave_splitter, stretch=3)

        # BPM row + SQI
        bpm_row = QHBoxLayout()
        bpm_font = QFont("Segoe UI", 22, QFont.Weight.Bold)

        self._breath_bpm_label = QLabel("--")
        self._breath_bpm_label.setFont(bpm_font)
        self._breath_bpm_label.setStyleSheet("color: #27ae60;")
        self._breath_bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bpm_row.addWidget(QLabel("呼吸:"))
        bpm_row.addWidget(self._breath_bpm_label)

        bpm_row.addSpacing(30)

        self._heart_bpm_label = QLabel("--")
        self._heart_bpm_label.setFont(bpm_font)
        self._heart_bpm_label.setStyleSheet("color: #e74c3c;")
        self._heart_bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bpm_row.addWidget(QLabel("心率:"))
        bpm_row.addWidget(self._heart_bpm_label)

        bpm_row.addStretch()
        self._sqi = SqiIndicator()
        bpm_row.addWidget(self._sqi)
        layout.addLayout(bpm_row)

        # Trend panel
        self._trend = TrendPanel()
        layout.addWidget(self._trend, stretch=2)

        # Debug panel (collapsible)
        debug_header = QHBoxLayout()
        self._debug_toggle = QPushButton("▼ 调试面板")
        self._debug_toggle.setStyleSheet(
            "QPushButton { background: #3a3a4a; color: #bdc3c7; border: none; "
            "padding: 6px 12px; text-align: left; }"
            "QPushButton:hover { background: #4a4a5a; }"
        )
        self._debug_toggle.clicked.connect(self._toggle_debug)
        debug_header.addWidget(self._debug_toggle)
        debug_header.addStretch()
        layout.addLayout(debug_header)

        self._debug_panel = QTextEdit()
        self._debug_panel.setReadOnly(True)
        self._debug_panel.setMaximumHeight(120)
        self._debug_panel.setFont(QFont("Consolas", 9))
        self._debug_panel.setStyleSheet(
            "QTextEdit { background: #1e1e2e; color: #3498db; border: 1px solid #3a3a4a; }"
        )
        self._debug_panel.setVisible(False)
        layout.addWidget(self._debug_panel)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #3a3a4a;")
        layout.addWidget(line)

    def _toggle_debug(self) -> None:
        self._debug_expanded = not self._debug_expanded
        self._debug_panel.setVisible(self._debug_expanded)
        self._debug_toggle.setText("▲ 调试面板" if self._debug_expanded else "▼ 调试面板")

    def start(self) -> None:
        self._trend.start()

    def update_display(
        self,
        breath_bpm: float,
        heart_bpm: float,
        breath_waveform: np.ndarray,
        heart_waveform: np.ndarray,
        quality: dict | None,
        sample_for_trend: bool = False,
    ) -> None:
        # Waveforms
        if len(breath_waveform) > 0:
            self._breath_wave.set_data(breath_waveform)
        if len(heart_waveform) > 0:
            self._heart_wave.set_data(heart_waveform)

        # BPM
        if breath_bpm > 0:
            self._breath_bpm_label.setText(f"{breath_bpm:.0f}")
        if heart_bpm > 0:
            self._heart_bpm_label.setText(f"{heart_bpm:.0f}")

        # SQI
        phase_range = quality.get("phase_range", 0.0) if quality else 0.0
        breath_ratio = quality.get("breath_ratio", 0.0) if quality else 0.0
        self._sqi_level = self._sqi.set_level(breath_ratio, phase_range)

        # Trend (throttled to ~1 sample/sec)
        if sample_for_trend:
            self._trend.add_sample(breath_bpm, heart_bpm)

        # Debug panel
        if self._debug_expanded and quality is not None:
            lines = [
                f"phase_range: {phase_range:.4f}",
                f"breath_ratio: {breath_ratio:.3f}",
                f"heart_prominence: {quality.get('heart_prominence', 0):.3f}",
                f"apnea_state: {quality.get('apnea_state', False)}",
                f"valid: {quality.get('valid', False)}",
                f"reason: {quality.get('reason', '')}",
                f"SQI: {self._sqi_level}/3",
            ]
            self._debug_panel.setText("  |  ".join(lines))
```

- [ ] **Step 2: Commit**

```bash
git add ui/research_tab.py
git commit -m "feat: add ResearchTab with waveforms, trend panel, debug panel"
```

---

### Task 10: Refactor MainWindow to thin shell with QTabWidget

**Files:**
- Modify: `ui/main_window.py`

- [ ] **Step 1: Rewrite MainWindow as a thin shell**

Replace `ui/main_window.py` entirely:

```python
"""Main window — thin shell hosting SubjectTab and ResearchTab via QTabWidget."""

import os
import glob
import time
import queue
import csv
import threading
from datetime import datetime
from collections import deque

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QLabel, QPushButton,
    QTabWidget,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

from config.protocol import UI_REFRESH_MS
from dsp_pipeline.pipeline import Pipeline
from dsp_pipeline.vital_signs import VitalSigns
from io_engine.bin_reader import BinFileReader
from io_engine.uart_parser import UartParser
from io_engine.serial_manager import SerialManager
from io_engine.radar_mgr import RadarMgr
from models.radar_frame import RadarFrame, FrameHeader

from ui.subject_tab import SubjectTab
from ui.research_tab import ResearchTab


class MainWindow(QMainWindow):
    def __init__(self, mode: str = "replay", replay_file: str | None = None):
        super().__init__()
        self.setWindowTitle("RS6240 毫米波雷达生命体征实时监测系统")
        self.resize(1200, 800)

        self._mode = mode
        self._replay_file = replay_file
        self._bin_reader: BinFileReader | None = None
        self._pipeline: Pipeline | None = None
        self._replay_timer: QTimer | None = None
        self._start_time: float = 0.0
        self._frame_count: int = 0
        self._running: bool = False
        self._latest_vitals: VitalSigns | None = None
        self._trend_tick_counter: int = 0

        # CSV data store — accumulate all vitals snapshots during session
        self._csv_rows: list[dict] = []

        # Serial mode
        self._serial_mgr: SerialManager | None = None
        self._radar_mgr: RadarMgr | None = None
        self._uart_parser: UartParser | None = None
        self._io_thread = None
        self._stop_event = None
        self._serial_status: str = ""
        if mode == "serial":
            self._serial_mgr = SerialManager()
            self._radar_mgr = RadarMgr(self._serial_mgr)
            self._uart_parser = UartParser(bins_per_frame=1024)

        self._setup_ui()
        self._setup_timers()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Title bar
        title_row = QHBoxLayout()
        title_row.setContentsMargins(12, 8, 12, 4)
        title = QLabel("RS6240 生命体征监测系统")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title_row.addWidget(title, stretch=1)

        self._file_label = QLabel("未选择文件")
        self._file_label.setStyleSheet("color: #95a5a6; font-size: 9pt;")
        title_row.addWidget(self._file_label)

        if self._mode == "replay":
            self._select_btn = QPushButton("选择文件")
            self._select_btn.clicked.connect(self._on_select_file)
            title_row.addWidget(self._select_btn)

        main_layout.addLayout(title_row)

        # Tab widget
        self._tabs = QTabWidget()
        self._subject_tab = SubjectTab()
        self._research_tab = ResearchTab()
        self._tabs.addTab(self._subject_tab, "监测")
        self._tabs.addTab(self._research_tab, "研究")
        main_layout.addWidget(self._tabs, stretch=1)

        # Control bar
        ctrl_row = QHBoxLayout()
        ctrl_row.setContentsMargins(12, 4, 12, 8)

        label = "▶ 开始采集" if self._mode == "serial" else "▶ 开始回放"
        self._start_btn = QPushButton(label)
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-weight: bold; "
            "padding: 8px 20px; border-radius: 4px; font-size: 11pt; }"
            "QPushButton:hover { background-color: #2ecc71; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self._start_btn.clicked.connect(self._on_start)
        ctrl_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("■ 停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; "
            "padding: 8px 20px; border-radius: 4px; font-size: 11pt; }"
            "QPushButton:hover { background-color: #c0392b; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self._stop_btn.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self._stop_btn)

        self._save_btn = QPushButton("保存数据")
        self._save_btn.clicked.connect(self._on_save)
        ctrl_row.addWidget(self._save_btn)

        ctrl_row.addStretch()

        self._status_label = QLabel("● 待机")
        self._status_label.setFont(QFont("Segoe UI", 10))
        self._status_label.setStyleSheet("color: #f39c12;")
        ctrl_row.addWidget(self._status_label)

        self._frame_rate_label = QLabel("帧率: --")
        ctrl_row.addWidget(self._frame_rate_label)

        self._elapsed_label = QLabel("运行: 00:00")
        ctrl_row.addWidget(self._elapsed_label)

        main_layout.addLayout(ctrl_row)

        # Auto-select latest file
        if self._mode == "replay" and not self._replay_file:
            self._replay_file = self._find_latest_bin()
        if self._replay_file:
            self._file_label.setText(os.path.basename(self._replay_file))
            self._file_label.setStyleSheet("color: #3498db; font-size: 9pt;")

    def _setup_timers(self) -> None:
        self._ui_timer = QTimer()
        self._ui_timer.timeout.connect(self._on_ui_tick)
        self._ui_timer.start(UI_REFRESH_MS)

    @staticmethod
    def _find_latest_bin() -> str | None:
        candidates = sorted(
            glob.glob("data/*.bin"),
            key=lambda f: os.path.getmtime(f),
            reverse=True,
        )
        return candidates[0] if candidates else None

    # === Slots ===

    def _on_select_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 .bin 文件", "data", "Bin Files (*.bin);;All Files (*)"
        )
        if path:
            self._replay_file = path
            self._file_label.setText(os.path.basename(path))
            self._file_label.setStyleSheet("color: #3498db; font-size: 9pt;")

    def _on_start(self) -> None:
        if self._mode == "serial":
            self._start_serial()
        else:
            self._start_replay()

    def _start_serial(self) -> None:
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        if hasattr(self, '_select_btn'):
            self._select_btn.setEnabled(False)
        self._status_label.setText("● 启动中...")
        self._status_label.setStyleSheet("color: #f39c12;")
        thread = threading.Thread(target=self._serial_init_thread, daemon=True)
        thread.start()

    def _serial_init_thread(self) -> None:
        try:
            self._do_serial_init()
        except Exception as e:
            import traceback
            print(f"[Serial Init] CRASH: {e}")
            traceback.print_exc()
            self._serial_status = f"启动失败: {e}"

    def _do_serial_init(self) -> None:
        import serial.tools.list_ports
        print("[Serial Init] Scanning ports...")
        ports = SerialManager.list_ports()
        ctrl_port = data_port = ""
        for p in ports:
            try:
                info = next((i for i in serial.tools.list_ports.comports() if i.device == p), None)
                desc = info.description if info else ""
            except Exception:
                desc = ""
            if "Standard" in desc:
                ctrl_port = p
            if "Enhanced" in desc:
                data_port = p
        print(f"[Serial Init] Found: ctrl={ctrl_port}, data={data_port}")
        if not ctrl_port or not data_port:
            self._serial_status = f"未找到雷达: ctrl={ctrl_port} data={data_port}"
            return
        print("[Serial Init] Connecting...")
        if not self._radar_mgr.connect(ctrl_port, data_port):
            self._serial_status = f"连接失败 {ctrl_port}/{data_port}"
            return
        print("[Serial Init] Booting radar...")
        ok = self._radar_mgr.boot()
        print(f"[Serial Init] Boot {'OK' if ok else 'PARTIAL FAIL'}")
        self._stop_event = threading.Event()
        self._uart_parser.reset()
        self._pipeline = Pipeline()
        self._pipeline.start()
        self._start_time = time.time()
        self._frame_count = 0
        self._running = True
        self._csv_rows.clear()
        self._research_tab.start()
        self._serial_status = f"采集中 ({ctrl_port}/{data_port})"
        print("[Serial Init] Starting I/O loop...")
        self._io_thread = threading.Thread(target=self._serial_io_loop, daemon=True)
        self._io_thread.start()
        print("[Serial Init] Done!")

    def _serial_io_loop(self) -> None:
        while self._running and not self._stop_event.is_set():
            try:
                raw = self._serial_mgr.read_data(4096)
                if not raw:
                    continue
                frames = self._uart_parser.feed(raw)
                for fft_data in frames:
                    self._frame_count += 1
                    cube = fft_data.reshape(2, 4, 128)
                    rx_combined = np.mean(cube[0, :, :], axis=0)
                    frame = RadarFrame(
                        timestamp=time.time(),
                        frame_index=self._frame_count,
                        header=FrameHeader(0, 1, 4, 2, 58000, 128, 1, 3000, 25, 1920, 60),
                        data_cube=rx_combined.reshape(-1, 1, 1),
                    )
                    while True:
                        try:
                            self._pipeline.raw_queue.put_nowait(frame)
                            break
                        except queue.Full:
                            try:
                                self._pipeline.raw_queue.get_nowait()
                            except queue.Empty:
                                pass
            except Exception as e:
                print(f"[Serial I/O] {e}")
                time.sleep(0.5)

    def _start_replay(self) -> None:
        if not self._replay_file or not os.path.exists(self._replay_file):
            QMessageBox.critical(self, "错误", "请先选择有效的 .bin 文件")
            return
        reader = BinFileReader(self._replay_file)
        if not reader.open():
            QMessageBox.critical(self, "错误", f"无法打开 {self._replay_file}")
            return
        self._bin_reader = reader
        self._pipeline = Pipeline()
        self._pipeline.start()
        self._start_time = time.time()
        self._frame_count = 0
        self._running = True
        self._csv_rows.clear()
        self._research_tab.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        if hasattr(self, '_select_btn'):
            self._select_btn.setEnabled(False)
        self._status_label.setText("● 回放中")
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
            self._status_label.setText("● 回放完毕")
            self._status_label.setStyleSheet("color: #3498db;")
            self._stop_btn.setEnabled(False)
            self._start_btn.setEnabled(True)
            if hasattr(self, '_select_btn'):
                self._select_btn.setEnabled(True)
            return
        self._frame_count += 1
        cube = frames[0].reshape(-1, 1, 1)
        frame = RadarFrame(
            timestamp=time.time(),
            frame_index=self._frame_count,
            header=FrameHeader(0, 1, 1, 1, 58000, 128, 0, 3000, 25, 1920, 60),
            data_cube=cube,
        )
        while True:
            try:
                self._pipeline.raw_queue.put_nowait(frame)
                break
            except queue.Full:
                try:
                    self._pipeline.raw_queue.get_nowait()
                except queue.Empty:
                    pass

    def _on_stop(self) -> None:
        self._running = False
        if self._mode == "serial":
            if self._radar_mgr:
                self._radar_mgr.shutdown()
            if self._stop_event:
                self._stop_event.set()
            if self._io_thread:
                self._io_thread.join(timeout=3)
            if self._serial_mgr:
                self._serial_mgr.close()
        if self._replay_timer:
            self._replay_timer.stop()
            self._replay_timer = None
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        if self._bin_reader:
            self._bin_reader.close()
            self._bin_reader = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if hasattr(self, '_select_btn'):
            self._select_btn.setEnabled(True)
        self._status_label.setText("● 已停止")
        self._status_label.setStyleSheet("color: #f39c12;")

    def _on_save(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not path:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save .npz (existing format)
        if self._latest_vitals is not None:
            np.savez(
                f"{path}/vital_signs_{ts}.npz",
                breath_waveform=self._latest_vitals.breath_waveform,
                heart_waveform=self._latest_vitals.heart_waveform,
                breath_bpm=self._latest_vitals.breath_bpm,
                heart_bpm=self._latest_vitals.heart_bpm,
            )

        # Save .csv
        if self._csv_rows:
            csv_path = f"{path}/vital_signs_{ts}.csv"
            fieldnames = [
                "Timestamp", "FrameIndex", "RangeBin", "RawPhase",
                "BreathBPM", "HeartBPM", "PhaseRange", "BreathRatio",
                "HeartProminence", "ApneaFlag", "SQI_Level",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._csv_rows)

        QMessageBox.information(self, "保存完成", f"数据已保存至 {path}")

    # === UI Timer ===

    def _on_ui_tick(self) -> None:
        # Poll serial status
        if self._serial_status:
            s = self._serial_status
            if "失败" in s or "未找到" in s or "连接失败" in s:
                self._status_label.setText(f"● {s}")
                self._status_label.setStyleSheet("color: #e74c3c;")
                self._start_btn.setEnabled(True)
                self._stop_btn.setEnabled(False)
                if hasattr(self, '_select_btn'):
                    self._select_btn.setEnabled(True)
            elif "采集中" in s:
                self._status_label.setText(f"● {s}")
                self._status_label.setStyleSheet("color: #27ae60;")
            self._serial_status = ""

        if not self._pipeline:
            return

        try:
            while not self._pipeline.display_queue.empty():
                self._latest_vitals = self._pipeline.display_queue.get_nowait()
        except queue.Empty:
            pass

        if self._latest_vitals is not None:
            q = self._latest_vitals.quality
            calib_done = self._pipeline.calibration_done
            calib_prog = self._pipeline.calibration_progress

            # Subject tab always gets data
            self._subject_tab.update_display(
                breath_bpm=self._latest_vitals.breath_bpm,
                heart_bpm=self._latest_vitals.heart_bpm,
                breath_waveform=self._latest_vitals.breath_waveform,
                quality=q,
                calibration_done=calib_done,
                calibration_progress=calib_prog,
            )

            # Research tab always gets data
            self._trend_tick_counter += 1
            trend_sample = (self._trend_tick_counter % 20 == 0)  # ~1 sample/sec
            self._research_tab.update_display(
                breath_bpm=self._latest_vitals.breath_bpm,
                heart_bpm=self._latest_vitals.heart_bpm,
                breath_waveform=self._latest_vitals.breath_waveform,
                heart_waveform=self._latest_vitals.heart_waveform,
                quality=q,
                sample_for_trend=trend_sample,
            )

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

                self._csv_rows.append({
                    "Timestamp": datetime.now().isoformat(),
                    "FrameIndex": self._latest_vitals.frame_index,
                    "RangeBin": 0,  # Pipeline doesn't expose best_bin; available with further work
                    "RawPhase": round(phase_range_raw, 6),
                    "BreathBPM": self._latest_vitals.breath_bpm,
                    "HeartBPM": self._latest_vitals.heart_bpm,
                    "PhaseRange": round(q.get("phase_range", 0), 6),
                    "BreathRatio": round(q.get("breath_ratio", 0), 4),
                    "HeartProminence": round(q.get("heart_prominence", 0), 4),
                    "ApneaFlag": 1 if q.get("apnea_state") else 0,
                    "SQI_Level": sqi,
                })

            # Status bar — simplified
            if q and not q.get("valid") and calib_done:
                self._status_label.setText(f"● 信号异常")
                self._status_label.setStyleSheet("color: #e74c3c;")
            elif self._running:
                self._status_label.setText("● 监测中")
                self._status_label.setStyleSheet("color: #27ae60;")

        if self._start_time > 0:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                self._frame_rate_label.setText(f"帧率: {self._frame_count / elapsed:.1f} fps")
            m, s = divmod(int(elapsed), 60)
            self._elapsed_label.setText(f"运行: {m:02d}:{s:02d}")

    def closeEvent(self, event) -> None:
        self._on_stop()
        event.accept()
```

- [ ] **Step 2: Commit**

```bash
git add ui/main_window.py
git commit -m "refactor: split MainWindow into SubjectTab + ResearchTab with QTabWidget"
```

---

### Task 11: Expose pipeline best_bin for CSV export

**Files:**
- Modify: `dsp_pipeline/pipeline.py`

- [ ] **Step 1: Add best_bin property**

Add after the calibration properties added in Task 1:

```python
    @property
    def best_range_bin(self) -> int | None:
        return self._best_bin
```

- [ ] **Step 2: Update CSV accumulation in MainWindow to use best_range_bin**

In `main_window.py`, replace the `"RangeBin": 0` line with:

```python
    "RangeBin": self._pipeline.best_range_bin if self._pipeline.best_range_bin is not None else 0,
```

- [ ] **Step 3: Commit**

```bash
git add dsp_pipeline/pipeline.py ui/main_window.py
git commit -m "feat: expose pipeline best_range_bin for CSV export"
```

---

### Task 12: Integration smoke test

**Files:**
- No changes — verification only

- [ ] **Step 1: Verify no import errors on startup**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "from ui.main_window import MainWindow; from ui.subject_tab import SubjectTab; from ui.research_tab import ResearchTab; from ui.status_mapper import map_status, BodyMovementDetector; from ui.sqi_indicator import SqiIndicator; from ui.breathing_petals import BreathingPetals; from ui.calibration_overlay import CalibrationOverlay; from ui.trend_panel import TrendPanel; print('All imports OK')"
```

Expected: "All imports OK"

- [ ] **Step 2: Verify pipeline properties work**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "
from dsp_pipeline.pipeline import Pipeline
p = Pipeline()
assert p.calibration_done == False
assert 0.0 <= p.calibration_progress <= 1.0
assert p.best_range_bin is None
print('Pipeline properties OK')
"
```

Expected: "Pipeline properties OK"

- [ ] **Step 3: Verify status_mapper returns correct tuples**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "
from ui.status_mapper import map_status, map_status_with_movement

# Apnea
msg, level = map_status({'apnea_state': True, 'phase_range': 0.01, 'breath_ratio': 0.2})
assert level == 'error', f'expected error, got {level}'

# Low phase_range
msg, level = map_status({'apnea_state': False, 'phase_range': 0.001, 'breath_ratio': 0.2})
assert level == 'warning', f'expected warning, got {level}'

# Normal
msg, level = map_status({'apnea_state': False, 'phase_range': 0.02, 'breath_ratio': 0.2})
assert level == 'normal', f'expected normal, got {level}'

# Body movement
msg, level = map_status_with_movement({'apnea_state': False}, True)
assert level == 'warning'

# None quality
msg, level = map_status(None)
assert level == 'normal'
print('status_mapper OK')
"
```

Expected: "status_mapper OK"

- [ ] **Step 4: Launch the app and check visual output**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python main.py -r
```

Manually verify:
- Two tabs visible: "监测" and "研究"
- Subject tab shows calibration overlay on start
- After 10s, overlay fades, BPM numbers appear
- Petals animate when breath_bpm > 0
- Heart icon pulses with heart_bpm
- SQI bars update
- Research tab shows dual waveforms with axes
- Trend panel accumulates scatter points
- Debug panel toggles
- Save produces both .npz and .csv

- [ ] **Step 5: Commit any fixes**

```bash
git status
# If nothing to fix, skip. Otherwise:
git add -A && git commit -m "fix: integration smoke test fixes"
```

---

## Self-Review

**1. Spec coverage:**
- Calibration overlay → Task 6 + Task 8 (SubjectTab integration) ✓
- DSP→Human messages → Task 2 (status_mapper.py) ✓
- SQI indicator → Task 4 ✓
- Breathing petals → Task 5 ✓
- Heart beat icon → Task 8 (included in subject_tab.py) ✓
- WaveWidget enhancements → Task 3 ✓
- Trend panel → Task 7 ✓
- Dual-mode tabs → Task 10 (MainWindow refactor) ✓
- Subject tab layout → Task 8 ✓
- Research tab layout → Task 9 ✓
- CSV export → Task 10 (in _on_save) ✓
- Pipeline calibration properties → Task 1 ✓
- heart_waveform fix → Task 1 ✓
- Body movement detection → Task 2 (BodyMovementDetector) ✓
- Pipeline best_bin exposure → Task 11 ✓

**2. Placeholder scan:** No TBD, TODO, or vague instructions found. Every step has concrete code.

**3. Type consistency:**
- `calibration_done` → `bool` ✓ (used consistently)
- `calibration_progress` → `float` (0.0-1.0) ✓ (used consistently)
- `best_range_bin` → `int | None` ✓
- `SqiIndicator.set_level()` returns `int` ✓
- `map_status()` returns `tuple[str, str]` ✓
- `SubjectTab.update_display()` signature matches call site in MainWindow ✓
- `ResearchTab.update_display()` signature matches call site in MainWindow ✓
