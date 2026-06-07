"""Single-channel waveform display with optional fill, axis toggle, and state-driven visual degradation."""

import time

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
import numpy as np


class WaveWidget(QWidget):
    """Waveform plot widget with optional gradient fill and axis visibility.

    Supports state-driven visual degradation: in 'error' state the waveform
    amplitude smoothly decays to near-flat and color shifts to cold gray,
    providing visual reassurance instead of showing noisy artifacts.
    """

    _COLOR_MAP = {
        "normal": QColor(39, 174, 96),     # #27ae60  green
        "warning": QColor(243, 156, 18),   # #f39c12  orange
        "error": QColor(127, 140, 141),    # #7f8c8d  cold gray
    }

    _ERROR_DECAY = 0.05        # target amplitude ratio in error state
    _ERROR_DECAY_DURATION = 1.0   # seconds to reach error decay
    _RECOVERY_DURATION = 1.5      # seconds to recover full amplitude
    _COLOR_TRANSITION = 0.8       # seconds for color to reach target

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

        # State-driven visual properties
        self._state = "normal"
        self._state_change_time = time.time()
        self._color_at_state_change = QColor(self._COLOR_MAP["normal"])
        self._current_color = QColor(self._COLOR_MAP["normal"])
        self._decay_factor = 1.0
        self._target_decay = 1.0
        self._decay_start = 1.0
        self._decay_start_time = time.time()
        self._decay_duration = 1.5

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
        self._pen = pg.mkPen(color=pen_color, width=1.5)
        self.curve = self.plot.plot(pen=self._pen)

        if fill_mode:
            brush = pg.mkBrush(QColor(39, 174, 96, 60))
            self._zero_line = pg.PlotDataItem([0], [0])
            self.fill = pg.FillBetweenItem(
                self.curve,
                self._zero_line,
                brush=brush,
            )
            self.plot.addItem(self.fill)
            self._fill_brush = brush

        layout.addWidget(self.plot)

    # ── public API ──────────────────────────────────────────────

    def set_state(self, state: str) -> None:
        """Transition waveform appearance based on signal quality level.

        Args:
            state: one of 'normal', 'warning', 'error'.
        """
        if state == self._state:
            return
        self._state = state
        now = time.time()
        self._state_change_time = now
        self._color_at_state_change = QColor(self._current_color)

        # Begin decay-factor interpolation
        self._decay_start = self._decay_factor
        self._decay_start_time = now
        if state == "error":
            self._target_decay = self._ERROR_DECAY
            self._decay_duration = self._ERROR_DECAY_DURATION
        else:
            self._target_decay = 1.0
            self._decay_duration = self._RECOVERY_DURATION

    def set_data(self, y: np.ndarray) -> None:
        """Update waveform data, applying current visual degradation."""
        if len(y) == 0:
            return
        self._update_appearance()
        self._data = y[-self.max_points:] * self._decay_factor
        x = np.arange(len(self._data))

        c = self._current_color
        self._pen.setColor(c)
        self.curve.setData(x, self._data)

        if self._fill_mode and hasattr(self, "fill"):
            self._zero_line.setData(x, np.zeros_like(self._data))
            self.fill.setCurves(self.curve, self._zero_line)
            fill_color = QColor(c.red(), c.green(), c.blue(), 60)
            self._fill_brush = pg.mkBrush(fill_color)
            self.fill.setBrush(self._fill_brush)

    # ── internal helpers ────────────────────────────────────────

    def _update_appearance(self) -> None:
        """Interpolate decay factor and pen color toward their targets."""
        now = time.time()

        # Decay factor: linear interpolation over configured duration
        elapsed = now - self._decay_start_time
        t = min(1.0, elapsed / self._decay_duration) if self._decay_duration > 0 else 1.0
        self._decay_factor = self._decay_start + (self._target_decay - self._decay_start) * t

        # Color: smoothstep over _COLOR_TRANSITION seconds
        target = self._COLOR_MAP.get(self._state, self._COLOR_MAP["normal"])
        color_elapsed = now - self._state_change_time
        ct = min(1.0, color_elapsed / self._COLOR_TRANSITION)
        ct = ct * ct * (3.0 - 2.0 * ct)  # smoothstep ease-out
        self._current_color = _lerp_color(self._color_at_state_change, target, ct)


def _lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    """Linearly interpolate between two QColors."""
    t = max(0.0, min(1.0, t))
    return QColor(
        int(c1.red() + (c2.red() - c1.red()) * t),
        int(c1.green() + (c2.green() - c1.green()) * t),
        int(c1.blue() + (c2.blue() - c1.blue()) * t),
    )
