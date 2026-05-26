"""Apple Watch style breathing petals animation driven by breath BPM."""

import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QBrush


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
            rect_x = cx - r + dx
            rect_y = cy - r + dy
            path.arcTo(rect_x, rect_y, r * 2, r * 2,
                       math.degrees(start_angle), math.degrees(sweep_angle))
            path.closeSubpath()
            painter.drawPath(path)

        painter.end()
