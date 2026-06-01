"""Blood pressure monitoring tab — large SBP/DBP numbers + scrolling waveform."""

import time
from collections import deque

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.wave_widget import WaveWidget
from config.i18n import tr, I18n


class BPValuePanel(QWidget):
    """Single BP value display: large number + unit label below."""

    def __init__(self, label: str, color: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel(label)
        self._label.setFont(QFont("Segoe UI", 11))
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color: #95a5a6;")
        layout.addWidget(self._label)

        self._value = QLabel("--")
        self._value.setFont(QFont("Segoe UI", 48, QFont.Weight.Bold))
        self._value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value.setStyleSheet(f"color: {color};")
        layout.addWidget(self._value)

        self._unit = QLabel("mmHg")
        self._unit.setFont(QFont("Segoe UI", 10))
        self._unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._unit.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(self._unit)

    def set_value(self, val: float) -> None:
        if np.isnan(val):
            self._value.setText("--")
        else:
            self._value.setText(f"{val:.0f}")


class ConfidenceDots(QWidget):
    """5-dot confidence indicator (0.0-1.0 → 0-5 filled dots)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._dots = []
        for _ in range(5):
            dot = QLabel("●")
            dot.setFont(QFont("Segoe UI", 8))
            dot.setStyleSheet("color: #555555;")
            layout.addWidget(dot)
            self._dots.append(dot)

    def set_confidence(self, conf: float) -> None:
        n = int(round(np.clip(conf, 0.0, 1.0) * 5))
        for i, dot in enumerate(self._dots):
            dot.setStyleSheet(
                f"color: {'#27ae60' if i < n else '#555555'};"
            )


class BPTab(QWidget):
    """Blood pressure monitoring tab.

    Layout:
      - SBP / DBP large numbers (red/blue)
      - Scrolling BP waveform (reuses WaveWidget)
      - Bottom info bar: distance, confidence, time since last update
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_update_time = time.time()
        self._wave_buffer: deque[float] = deque(maxlen=1200)

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)

        # --- BP value row ---
        bp_row = QHBoxLayout()
        bp_row.addStretch()
        self._sbp_panel = BPValuePanel(tr("bp_sbp_label") if tr("bp_sbp_label") != "bp_sbp_label" else "SBP", "#e74c3c")
        bp_row.addWidget(self._sbp_panel)
        bp_row.addSpacing(60)
        self._dbp_panel = BPValuePanel(tr("bp_dbp_label") if tr("bp_dbp_label") != "bp_dbp_label" else "DBP", "#3498db")
        bp_row.addWidget(self._dbp_panel)
        bp_row.addStretch()
        layout.addLayout(bp_row)

        # --- Separator ---
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #3a3a4a;")
        layout.addWidget(sep)

        # --- Waveform ---
        self._wave = WaveWidget(
            title="", y_label="",
            max_points=1200, fill_mode=True,
            show_axes=False, show_grid=False,
        )
        layout.addWidget(self._wave, stretch=1)

        # --- Bottom info bar ---
        info_row = QHBoxLayout()
        info_row.setContentsMargins(8, 4, 8, 4)

        self._dist_label = QLabel(tr("bp_dist_label") if tr("bp_dist_label") != "bp_dist_label" else "Distance: --")
        self._dist_label.setFont(QFont("Segoe UI", 9))
        self._dist_label.setStyleSheet("color: #95a5a6;")
        info_row.addWidget(self._dist_label)

        info_row.addStretch()

        conf_label = QLabel(tr("bp_conf_label") if tr("bp_conf_label") != "bp_conf_label" else "Confidence:")
        conf_label.setFont(QFont("Segoe UI", 9))
        conf_label.setStyleSheet("color: #95a5a6;")
        info_row.addWidget(conf_label)
        self._conf_dots = ConfidenceDots()
        info_row.addWidget(self._conf_dots)

        info_row.addStretch()

        self._update_label = QLabel("")
        self._update_label.setFont(QFont("Segoe UI", 9))
        self._update_label.setStyleSheet("color: #555555;")
        info_row.addWidget(self._update_label)

        layout.addLayout(info_row)

        # i18n
        I18n.instance().language_changed.connect(self._on_lang_change)

    def _on_lang_change(self) -> None:
        self._sbp_panel._label.setText(tr("bp_sbp_label"))
        self._dbp_panel._label.setText(tr("bp_dbp_label"))
        self._dist_label.setText(tr("bp_dist_label"))
        # Update confidence label text
        for i in range(self._conf_dots.parent().layout().count()):
            w = self._conf_dots.parent().layout().itemAt(i).widget()
            if w and isinstance(w, QLabel) and w.text().startswith("Conf"):
                w.setText(tr("bp_conf_label"))

    def update_display(self, bp_result) -> None:
        """Accept BPResult and refresh all UI elements.

        Args:
            bp_result: BPResult dataclass from bp_monitor.bp_models
        """
        now = time.time()
        self._last_update_time = now

        r = bp_result

        # SBP / DBP
        self._sbp_panel.set_value(r.sbp)
        self._dbp_panel.set_value(r.dbp)

        # Waveform: downsample 256 -> 60 points, append to rolling buffer
        wf = r.bp_waveform
        if wf.size >= 60:
            indices = np.linspace(0, wf.size - 1, 60, dtype=int)
            chunk = wf[indices]
        else:
            chunk = wf
        self._wave_buffer.extend(float(v) for v in chunk)
        self._wave.set_data(np.array(self._wave_buffer, dtype=np.float32))

        # Info bar
        if not np.isnan(r.target_distance_m):
            self._dist_label.setText(f"Distance: {r.target_distance_m:.2f} m")
        else:
            self._dist_label.setText("Distance: --")

        conf = r.quality.get("confidence", 0.0) if r.quality else 0.0
        self._conf_dots.set_confidence(conf)

        self._update_label.setText(
            f"Updated {now - r.timestamp:.0f}s ago"
        )
