"""Single-channel waveform display with optional fill and axis toggle."""

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
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
            self._zero_line = pg.PlotDataItem([0], [0])
            self.fill = pg.FillBetweenItem(
                self.curve,
                self._zero_line,
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
            self._zero_line.setData(x, np.zeros_like(self._data))
            self.fill.setCurves(
                self.curve,
                self._zero_line,
            )
