"""BPM history scatter plot with time window selector."""

import time
from collections import deque

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
import pyqtgraph as pg


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
                scatter.setData(x=list(xs), y=list(ys))
            else:
                scatter.setData(x=[], y=[])

        self._plot.setXRange(max(0, cutoff), max(cutoff + window_sec, now))

    def _on_window_changed(self) -> None:
        self._redraw()
