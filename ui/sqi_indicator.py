"""3-bar signal quality indicator (like WiFi signal icon)."""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
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
            bar.setStyleSheet("background-color: #7f8c8d; border-radius: 1px;")
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
