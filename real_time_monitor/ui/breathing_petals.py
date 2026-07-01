"""Apple Watch style breathing petals animation driven by breath BPM.

Uses QVariantAnimation (C++ event loop) instead of manual phase tracking
in paintEvent, decoupling animation timing from Python-thread refresh jitter.
"""

import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF, QVariantAnimation, QEasingCurve
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QBrush


class BreathingPetals(QWidget):
    """Draws overlapping arc petals that open/close at breath_bpm frequency.

    Animation is driven by a looping QVariantAnimation that sweeps 0 → 2π,
    keeping the 60 fps rendering in Qt's native C++ animation thread.
    """

    def __init__(self, size: int = 80, num_petals: int = 6, parent=None):
        super().__init__(parent)
        self._size = size
        self._num_petals = num_petals
        self._breath_bpm = 0.0
        self._openness = 0.5  # 0.0=closed, 1.0=open

        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(2.0 * math.pi)
        self._anim.setLoopCount(-1)  # infinite
        self._anim.valueChanged.connect(self._on_anim_tick)

        self.setFixedSize(size, size)

    def set_breath_bpm(self, bpm: float, dt: float) -> None:
        """Update animation period from BPM. dt is kept for API compatibility
        but no longer used for manual phase integration."""
        if bpm > 0 and abs(bpm - self._breath_bpm) > 1.0:
            self._breath_bpm = bpm
            period_ms = int(60000.0 / bpm)
            self._anim.setDuration(period_ms)
            if self._anim.state() != QVariantAnimation.State.Running:
                self._anim.start()
        elif bpm <= 0:
            self._breath_bpm = 0.0
            if self._anim.state() == QVariantAnimation.State.Running:
                self._anim.stop()
            self._openness = 0.5
            self.update()

    def _on_anim_tick(self, phase: float) -> None:
        """Called by Qt animation thread at screen refresh rate."""
        self._openness = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(phase))
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
