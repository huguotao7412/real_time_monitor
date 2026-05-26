"""Subject mode tab — minimal, calming UI for the person being monitored."""

import math
import time

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QFont, QPainter, QPainterPath, QColor, QBrush

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

        # Calibration overlay (stacked on top via resizeEvent)
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
        if bpm > 0:
            self._heart_bpm = bpm
            self._phase += bpm / 60.0 * 2 * math.pi * dt
        beat = 0.5 + 0.5 * math.sin(self._phase)
        self._scale = 1.0 + 0.3 * max(0, beat ** 8)
        self._current_color = _heart_rate_color(bpm)
        self.update()

    def current_color(self) -> QColor:
        return self._current_color

    def paintEvent(self, event):
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


def _heart_rate_color(bpm: float) -> QColor:
    if bpm <= 0:
        return QColor(127, 140, 141)
    if 60 <= bpm <= 100:
        return QColor(39, 174, 96)
    if 50 <= bpm < 60 or 100 < bpm <= 110:
        return QColor(243, 156, 18)
    return QColor(231, 76, 60)
