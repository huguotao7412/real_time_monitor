"""Research mode tab — full DSP data, waveforms with axes, trend panel, debug."""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, QFrame,
    QPushButton, QTextEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.wave_widget import WaveWidget
from ui.sqi_indicator import SqiIndicator
from ui.trend_panel import TrendPanel
from config.i18n import tr, I18n


class ResearchTab(QWidget):
    """Full researcher view: dual waveforms, BPM, trend, collapsible debug panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sqi_level = 0
        self._debug_expanded = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Waveforms
        wave_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._breath_wave = WaveWidget(tr("resp_wave_title"), "Amplitude")
        self._heart_wave = WaveWidget(tr("heart_wave_title"), "Amplitude")
        wave_splitter.addWidget(self._breath_wave)
        wave_splitter.addWidget(self._heart_wave)
        layout.addWidget(wave_splitter, stretch=3)

        # BPM row + SQI
        bpm_row = QHBoxLayout()
        bpm_font = QFont("Segoe UI", 22, QFont.Weight.Bold)

        self._breath_bpm_label = QLabel("--")
        self._breath_bpm_label.setFont(bpm_font)
        self._breath_bpm_label.setStyleSheet("color: #27ae60;")
        self._breath_bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._breath_label = QLabel(tr("label_breath"))
        bpm_row.addWidget(self._breath_label)
        bpm_row.addWidget(self._breath_bpm_label)

        bpm_row.addSpacing(30)

        self._heart_bpm_label = QLabel("--")
        self._heart_bpm_label.setFont(bpm_font)
        self._heart_bpm_label.setStyleSheet("color: #e74c3c;")
        self._heart_bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._heart_label = QLabel(tr("label_heart"))
        bpm_row.addWidget(self._heart_label)
        bpm_row.addWidget(self._heart_bpm_label)

        bpm_row.addStretch()
        self._sqi = SqiIndicator()
        bpm_row.addWidget(self._sqi)
        layout.addLayout(bpm_row)

        # Trend panel
        self._trend = TrendPanel()
        layout.addWidget(self._trend, stretch=2)

        # Debug panel (collapsible)
        debug_header = QHBoxLayout()
        self._debug_toggle = QPushButton(tr("debug_collapsed"))
        self._debug_toggle.setStyleSheet(
            "QPushButton { background: #3a3a4a; color: #bdc3c7; border: none; "
            "padding: 6px 12px; text-align: left; }"
            "QPushButton:hover { background: #4a4a5a; }"
        )
        self._debug_toggle.clicked.connect(self._toggle_debug)
        debug_header.addWidget(self._debug_toggle)
        debug_header.addStretch()
        layout.addLayout(debug_header)

        self._debug_panel = QTextEdit()
        self._debug_panel.setReadOnly(True)
        self._debug_panel.setMaximumHeight(120)
        self._debug_panel.setFont(QFont("Consolas", 9))
        self._debug_panel.setStyleSheet(
            "QTextEdit { background: #1e1e2e; color: #3498db; border: 1px solid #3a3a4a; }"
        )
        self._debug_panel.setVisible(False)
        layout.addWidget(self._debug_panel)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #3a3a4a;")
        layout.addWidget(line)

        I18n.instance().language_changed.connect(self.update_ui_texts)

    def update_ui_texts(self, _lang: str = "") -> None:
        self._breath_wave.title_label.setText(tr("resp_wave_title"))
        self._heart_wave.title_label.setText(tr("heart_wave_title"))
        self._breath_label.setText(tr("label_breath"))
        self._heart_label.setText(tr("label_heart"))
        self._debug_toggle.setText(
            tr("debug_expanded") if self._debug_expanded else tr("debug_collapsed")
        )

    def _toggle_debug(self) -> None:
        self._debug_expanded = not self._debug_expanded
        self._debug_panel.setVisible(self._debug_expanded)
        self._debug_toggle.setText(
            tr("debug_expanded") if self._debug_expanded else tr("debug_collapsed")
        )

    def start(self) -> None:
        self._trend.start()

    def reset_display(self) -> None:
        """Clear all displayed values to default state."""
        self._breath_bpm_label.setText("--")
        self._heart_bpm_label.setText("--")
        self._breath_wave.set_data(np.array([], dtype=np.float32))
        self._heart_wave.set_data(np.array([], dtype=np.float32))
        self._sqi.set_level(0.0, 0.0)
        self._trend.start()
        self._debug_panel.clear()

    def update_display(
        self,
        breath_bpm: float,
        heart_bpm: float,
        breath_waveform: np.ndarray,
        heart_waveform: np.ndarray,
        quality: dict | None,
        sample_for_trend: bool = False,
    ) -> None:
        # Waveforms
        if len(breath_waveform) > 0:
            self._breath_wave.set_data(breath_waveform)
        if len(heart_waveform) > 0:
            self._heart_wave.set_data(heart_waveform)

        # BPM
        if breath_bpm > 0:
            self._breath_bpm_label.setText(f"{breath_bpm:.0f}")
        if heart_bpm > 0:
            self._heart_bpm_label.setText(f"{heart_bpm:.0f}")

        # SQI
        phase_range = quality.get("phase_range", 0.0) if quality else 0.0
        breath_ratio = quality.get("breath_ratio", 0.0) if quality else 0.0
        self._sqi_level = self._sqi.set_level(breath_ratio, phase_range)

        # Trend (throttled to ~1 sample/sec)
        if sample_for_trend:
            self._trend.add_sample(breath_bpm, heart_bpm)

        # Debug panel
        if self._debug_expanded and quality is not None:
            lines = [
                f"phase_range: {phase_range:.4f}",
                f"breath_ratio: {breath_ratio:.3f}",
                f"heart_prominence: {quality.get('heart_prominence', 0):.3f}",
                f"apnea_state: {quality.get('apnea_state', False)}",
                f"valid: {quality.get('valid', False)}",
                f"reason: {quality.get('reason', '')}",
                f"SQI: {self._sqi_level}/3",
            ]
            self._debug_panel.setText("  |  ".join(lines))
