"""Blood pressure monitoring tab — large SBP/DBP numbers + trend scatter plot."""

import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.subject_tab import HeartBeatIcon
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
      - SBP/DBP trend scatter plot (pyqtgraph)
      - Bottom info bar: distance, confidence, time since last update
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_update_time = time.time()
        self._first_timestamp: float | None = None

        # Trend data caches (max 2000 points)
        self._trend_time: list[float] = []
        self._trend_sbp: list[float] = []
        self._trend_dbp: list[float] = []

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

        # --- Heart icon row (implicit HR from peak count) ---
        heart_row = QHBoxLayout()
        heart_row.addStretch()
        self._heart_icon = HeartBeatIcon(size=36)
        heart_row.addWidget(self._heart_icon)
        heart_row.addStretch()
        layout.addLayout(heart_row)

        # --- BP Trend Plot (replaces WaveWidget) ---
        self._trend_plot = pg.PlotWidget(title="SBP / DBP Trend")
        self._trend_plot.setBackground("transparent")
        self._trend_plot.showGrid(x=True, y=True, alpha=0.3)
        self._trend_plot.setLabel("left", "mmHg")
        self._trend_plot.setLabel("bottom", "Time (s)")
        self._trend_plot.setYRange(40, 180)
        self._trend_plot.setXRange(0, 60)
        # SBP: red scatter + line, DBP: blue scatter + line
        self._sbp_curve = self._trend_plot.plot(
            pen=pg.mkPen("#e74c3c", width=1.5),
            symbol='o', symbolBrush='#e74c3c', symbolPen=None, symbolSize=5,
            name="SBP",
        )
        self._dbp_curve = self._trend_plot.plot(
            pen=pg.mkPen("#3498db", width=1.5),
            symbol='o', symbolBrush='#3498db', symbolPen=None, symbolSize=5,
            name="DBP",
        )
        layout.addWidget(self._trend_plot, stretch=1)

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

    def update_display(self, bp_result) -> None:
        """Accept BPResult and refresh all UI elements."""
        now = time.time()
        self._last_update_time = now

        r = bp_result

        # Initialize first timestamp for relative time axis
        if self._first_timestamp is None:
            self._first_timestamp = r.timestamp

        # SBP / DBP values
        self._sbp_panel.set_value(r.sbp)
        self._dbp_panel.set_value(r.dbp)

        # Append to trend data (valid readings only)
        if not np.isnan(r.sbp) and not np.isnan(r.dbp):
            elapsed = r.timestamp - self._first_timestamp
            self._trend_time.append(elapsed)
            self._trend_sbp.append(r.sbp)
            self._trend_dbp.append(r.dbp)

            # Keep max 300 points rolling window
            MAX_TREND_POINTS = 2000
            if len(self._trend_time) > MAX_TREND_POINTS:
                self._trend_time.pop(0)
                self._trend_sbp.pop(0)
                self._trend_dbp.pop(0)

            self._sbp_curve.setData(self._trend_time, self._trend_sbp)
            self._dbp_curve.setData(self._trend_time, self._trend_dbp)

            if elapsed > 60:
                self._trend_plot.setXRange(elapsed - 60, elapsed)
            else:
                self._trend_plot.setXRange(0, 60)

        # Implicit heart rate from systolic peak count
        n_peaks = r.quality.get("n_peaks", 0) if r.quality else 0
        if n_peaks > 0:
            implicit_hr = n_peaks / 5.12 * 60.0
            self._heart_icon.set_heart_bpm(implicit_hr, 0)
        else:
            self._heart_icon.set_heart_bpm(0, 0)

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

    def reset_display(self) -> None:
        """Clear all BP values to '--' state."""
        self._sbp_panel.set_value(float('nan'))
        self._dbp_panel.set_value(float('nan'))
        self._first_timestamp = None
        self._trend_time.clear()
        self._trend_sbp.clear()
        self._trend_dbp.clear()
        self._sbp_curve.setData([], [])
        self._dbp_curve.setData([], [])
        self._dist_label.setText("Distance: --")
        self._conf_dots.set_confidence(0.0)
        self._update_label.setText("")
        self._heart_icon.set_heart_bpm(0, 0)
