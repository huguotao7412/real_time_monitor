"""Ring progress overlay shown during 10-second calibration phase."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QRectF, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QPainter, QPen, QColor, QFont

from config.i18n import tr, I18n


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

        self._text = QLabel(tr("calibration_text"))
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

        I18n.instance().language_changed.connect(self.update_ui_texts)

    def update_ui_texts(self, _lang: str = "") -> None:
        self._text.setText(tr("calibration_text"))

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
