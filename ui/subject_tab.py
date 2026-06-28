"""Subject mode tab — minimal, calming UI for the person being monitored."""

import math
import time

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt, QPointF, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QFont, QPainter, QPainterPath, QColor, QBrush

from ui.wave_widget import WaveWidget
from ui.sqi_indicator import SqiIndicator
from ui.breathing_petals import BreathingPetals
from ui.calibration_overlay import CalibrationOverlay
from ui.status_mapper import map_status_with_movement, BodyMovementDetector
from config.i18n import tr, I18n

_ERROR_OVERLAY_DELAY = 2.0  # seconds of continuous error before showing calm-down text


class SubjectTab(QWidget):
    """Minimal subject-facing UI: BPM numbers, petal animation, filled waveform."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_update_time = time.time()
        self._movement_detector = BodyMovementDetector()
        self._sqi_level = 0
        self._calibration_was_done = False
        self._error_start_time: float | None = None
        self._last_bpm_label_update: float = 0.0  # BPM 标签节流用时间戳

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)

        # Top row: distance label (left) + SQI indicator (right)
        top_row = QHBoxLayout()
        self._distance_label = QLabel(tr("目标距离: -- cm"))
        self._distance_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self._distance_label.setStyleSheet(
            "color: #3498db; background-color: rgba(52, 152, 219, 0.1);"
            "border-radius: 5px; padding: 5px;"
        )
        top_row.addWidget(self._distance_label)
        top_row.addStretch()
        self._sqi = SqiIndicator()
        top_row.addWidget(self._sqi)
        layout.addLayout(top_row)

        # BPM center area — breath + heart on same row
        bpm_area = QVBoxLayout()
        bpm_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bpm_area.addStretch()

        # Breath + Heart BPM row (并排)
        vitals_row = QHBoxLayout()
        vitals_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vitals_row.setSpacing(60)

        # -- 呼吸 (左侧) --
        breath_col = QVBoxLayout()
        breath_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        breath_col.setSpacing(0)
        self._petals = BreathingPetals(size=50)
        breath_col.addWidget(self._petals, alignment=Qt.AlignmentFlag.AlignCenter)
        self._breath_bpm_label = QLabel("--")
        self._breath_bpm_label.setFont(QFont("Segoe UI", 64, QFont.Weight.Bold))
        self._breath_bpm_label.setStyleSheet("color: #27ae60;")
        self._breath_bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        breath_col.addWidget(self._breath_bpm_label)
        self._breath_unit_label = QLabel(tr("breath_rate_unit"))
        self._breath_unit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._breath_unit_label.setStyleSheet("color: #7f8c8d; font-size: 11pt; margin-top: -2px;")
        breath_col.addWidget(self._breath_unit_label)
        vitals_row.addLayout(breath_col)

        # -- 心率 (右侧) --
        heart_col = QVBoxLayout()
        heart_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heart_col.setSpacing(0)
        self._heart_icon = HeartBeatIcon(size=50)
        heart_col.addWidget(self._heart_icon, alignment=Qt.AlignmentFlag.AlignCenter)
        self._heart_bpm_label = QLabel("--")
        self._heart_bpm_label.setFont(QFont("Segoe UI", 64, QFont.Weight.Bold))
        self._heart_bpm_label.setStyleSheet("color: #e74c3c;")
        self._heart_bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heart_col.addWidget(self._heart_bpm_label)
        self._heart_unit_label = QLabel(tr("heart_rate_unit"))
        self._heart_unit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._heart_unit_label.setStyleSheet("color: #7f8c8d; font-size: 11pt; margin-top: -2px;")
        heart_col.addWidget(self._heart_unit_label)
        vitals_row.addLayout(heart_col)

        bpm_area.addLayout(vitals_row)

        bpm_area.addStretch()
        layout.addLayout(bpm_area)

        # Waveform container (for positioning the overlay)
        self._wave_container = QWidget()
        wave_container_layout = QVBoxLayout(self._wave_container)
        wave_container_layout.setContentsMargins(0, 0, 0, 0)

        self._breath_wave = WaveWidget(
            title="", fill_mode=True, show_axes=False, show_grid=False,
        )
        wave_container_layout.addWidget(self._breath_wave)

        # Semi-transparent calm-down overlay (hidden by default)
        self._error_overlay = QLabel(tr("error_overlay_text"), self._wave_container)
        self._error_overlay.setFont(QFont("Segoe UI", 13))
        self._error_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error_overlay.setStyleSheet(
            "color: rgba(189, 195, 199, 230);"
            "background-color: rgba(30, 30, 40, 180);"
        )
        self._error_overlay.hide()

        layout.addWidget(self._wave_container, stretch=1)

        # Status message
        self._status_label = QLabel(tr("status_standby"))
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

        # i18n hot-switch
        I18n.instance().language_changed.connect(self.update_ui_texts)

    def update_ui_texts(self, _lang: str = "") -> None:
        self._breath_unit_label.setText(tr("breath_rate_unit"))
        self._heart_unit_label.setText(tr("heart_rate_unit"))
        self._error_overlay.setText(tr("error_overlay_text"))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._calibration_overlay.setGeometry(self.rect())
        # Keep error overlay centered within the wave container
        if self._wave_container.isVisible():
            cw = self._wave_container.width()
            ch = self._wave_container.height()
            ow = min(400, cw - 40)
            oh = 44
            self._error_overlay.setGeometry(
                (cw - ow) // 2, (ch - oh) // 2, ow, oh,
            )

    def update_display(
        self,
        breath_bpm: float,
        heart_bpm: float,
        breath_waveform: np.ndarray,
        quality: dict | None,
        calibration_done: bool,
        calibration_progress: float,
        target_distance_m: float = 0.0,
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

        # Distance label
        if target_distance_m > 0:
            self._distance_label.setText(
                tr("目标距离: {:.1f} cm").format(target_distance_m * 100)
            )
        else:
            self._distance_label.setText(tr("目标距离: -- cm"))

        # SQI
        phase_range = quality.get("phase_range", 0.0) if quality else 0.0
        breath_ratio = quality.get("breath_ratio", 0.0) if quality else 0.0
        self._sqi_level = self._sqi.set_level(breath_ratio, phase_range)

        # Body movement detection
        movement = self._movement_detector.feed(phase_range)
        msg, level = map_status_with_movement(quality, movement)

        # ── Waveform visual degradation ──
        self._breath_wave.set_state(level)

        # Error overlay: show after sustained error
        if level == "error":
            if self._error_start_time is None:
                self._error_start_time = now
            elif now - self._error_start_time >= _ERROR_OVERLAY_DELAY:
                self._error_overlay.show()
        else:
            self._error_start_time = None
            self._error_overlay.hide()

        # BPM opacity
        bpm_opacity = 0.4 if level == "error" else 1.0

        # Breath BPM + petals (标签 500ms 节流防闪烁)
        now = time.time()
        bpm_label_debounce = (now - self._last_bpm_label_update) >= 0.5
        if breath_bpm > 0:
            if bpm_label_debounce:
                self._breath_bpm_label.setText(f"{breath_bpm:.0f}")
            self._petals.set_breath_bpm(breath_bpm, dt)
        else:
            if bpm_label_debounce:
                self._breath_bpm_label.setText("--")
            self._petals.set_breath_bpm(0, dt)

        self._breath_bpm_label.setStyleSheet(
            f"color: rgba(39, 174, 96, {int(bpm_opacity * 255)});"
        )

        # Heart BPM + icon
        if heart_bpm > 0:
            if bpm_label_debounce:
                self._heart_bpm_label.setText(f"{heart_bpm:.0f}")
            self._heart_icon.set_heart_bpm(heart_bpm, dt)
        else:
            if bpm_label_debounce:
                self._heart_bpm_label.setText("--")
            self._heart_icon.set_heart_bpm(0, dt)
        if bpm_label_debounce:
            self._last_bpm_label_update = now

        heart_color = self._heart_icon.current_color()
        self._heart_bpm_label.setStyleSheet(
            f"color: rgba({heart_color.red()}, {heart_color.green()}, "
            f"{heart_color.blue()}, {int(bpm_opacity * 255)});"
        )

        # Waveform data (decay is applied inside WaveWidget)
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
        self._distance_label.setText(tr("目标距离: -- cm"))


class HeartBeatIcon(QWidget):
    """Bezier heart shape that pulses at heart_bpm frequency.

    Uses QPropertyAnimation on scale_factor, driven by Qt's native C++
    animation loop — no manual phase integration in Python paintEvent.
    """

    def __init__(self, size: int = 50, parent=None):
        super().__init__(parent)
        self._size = size
        self._heart_bpm = 0.0
        self._scale_factor = 1.0
        self._current_color = QColor(39, 174, 96)

        # Heartbeat animation: quick contraction + elastic recoil
        self._anim = QPropertyAnimation(self, b"scale_factor")
        self._anim.setLoopCount(-1)  # infinite — stopped when BPM drops to 0
        # Keyframes: rest → peak (systole) → recoil → rest
        self._anim.setKeyValueAt(0.0, 1.0)
        self._anim.setKeyValueAt(0.12, 1.28)
        self._anim.setKeyValueAt(0.30, 1.0)
        self._anim.setKeyValueAt(1.0, 1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutElastic)

        self.setFixedSize(size, size)

    # ── pyqtProperty for QPropertyAnimation ──────────────────

    def get_scale_factor(self) -> float:
        return self._scale_factor

    def set_scale_factor(self, val: float) -> None:
        self._scale_factor = val
        self.update()

    scale_factor = pyqtProperty(float, get_scale_factor, set_scale_factor)

    # ── public API ──────────────────────────────────────────

    def set_heart_bpm(self, bpm: float, dt: float) -> None:
        """Update animation period from BPM. dt kept for API compatibility."""
        if bpm > 0:
            if bpm != self._heart_bpm:
                self._heart_bpm = bpm
                period_ms = int(60000.0 / bpm)
                self._anim.setDuration(period_ms)
                if self._anim.state() != QPropertyAnimation.State.Running:
                    self._anim.start()
        else:
            self._heart_bpm = 0.0
            if self._anim.state() == QPropertyAnimation.State.Running:
                self._anim.stop()
            self._scale_factor = 1.0
            self.update()

        self._current_color = _heart_rate_color(bpm)

    def current_color(self) -> QColor:
        return self._current_color

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self._size / 2, self._size / 2
        painter.translate(cx, cy)
        painter.scale(self._scale_factor, self._scale_factor)
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
