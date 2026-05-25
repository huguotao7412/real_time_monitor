import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
import numpy as np


class WaveWidget(QWidget):
    """单路波形显示组件: 标题 + PlotWidget"""

    def __init__(self, title: str, y_label: str = "", max_points: int = 200, parent=None):
        super().__init__(parent)
        self.max_points = max_points
        self._data = np.zeros(max_points)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(self.title_label)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", y_label)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setMouseEnabled(x=False, y=False)
        self.curve = self.plot.plot(pen=pg.mkPen(color=(0, 180, 100), width=1.5))
        layout.addWidget(self.plot)

    def set_data(self, y: np.ndarray) -> None:
        """更新波形数据"""
        if len(y) == 0:
            return
        self._data = y[-self.max_points:]
        x = np.arange(len(self._data))
        self.curve.setData(x, self._data)
