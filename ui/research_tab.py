"""Research mode tab — full DSP data, waveforms with axes, trend panel, debug."""

import time

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, QFrame,
    QPushButton, QTextEdit, QComboBox,
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
        self._last_bpm_label_update: float = 0.0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # ── Algorithm Control Panel ──
        algo_row = QHBoxLayout()
        algo_row.setContentsMargins(0, 0, 0, 4)

        algo_row.addWidget(QLabel(tr("algo_panel_label") + ":"))
        self._algo_combo = QComboBox()
        self._algo_combo.addItems([
            tr("algo_adaptive"),
            tr("algo_vmd_wpd"),
            tr("algo_emd_wpd"),
            tr("algo_passthrough_sos"),
        ])
        self._algo_combo.setCurrentIndex(0)
        algo_row.addWidget(self._algo_combo)

        algo_row.addSpacing(12)
        algo_row.addWidget(QLabel(tr("ab_panel_label") + ":"))
        self._ab_combo = QComboBox()
        self._ab_combo.addItems([
            tr("ab_off"),
            tr("algo_vmd_wpd"),
            tr("algo_emd_wpd"),
            tr("algo_passthrough_sos"),
        ])
        self._ab_combo.setCurrentIndex(0)
        algo_row.addWidget(self._ab_combo)

        algo_row.addSpacing(12)
        self._record_btn = QPushButton(tr("btn_record_start"))
        self._record_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; font-size: 10pt; }"
            "QPushButton:hover { background-color: #c0392b; }"
        )
        algo_row.addWidget(self._record_btn)

        self._record_timer_label = QLabel("")
        self._record_timer_label.setFont(QFont("Consolas", 10))
        self._record_timer_label.setStyleSheet("color: #e74c3c;")
        self._record_timer_label.setVisible(False)
        algo_row.addWidget(self._record_timer_label)

        algo_row.addStretch()
        layout.addLayout(algo_row)

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

    # ── Public API for MainWindow wiring ──

    @property
    def algo_selection(self) -> int:
        return self._algo_combo.currentIndex()

    @property
    def ab_selection(self) -> int:
        return self._ab_combo.currentIndex()

    def set_recording_state(self, is_recording: bool) -> None:
        """Update record button appearance and timer visibility."""
        if is_recording:
            self._record_btn.setText(tr("btn_record_stop"))
            self._record_btn.setStyleSheet(
                "QPushButton { background-color: #95a5a6; color: white; font-weight: bold; "
                "padding: 6px 16px; border-radius: 4px; font-size: 10pt; }"
            )
            self._record_timer_label.setVisible(True)
        else:
            self._record_btn.setText(tr("btn_record_start"))
            self._record_btn.setStyleSheet(
                "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; "
                "padding: 6px 16px; border-radius: 4px; font-size: 10pt; }"
            )
            self._record_timer_label.setVisible(False)

    def update_record_timer(self, elapsed_sec: float) -> None:
        """Update the recording timer label."""
        if elapsed_sec > 0:
            h = int(elapsed_sec // 3600)
            m = int((elapsed_sec % 3600) // 60)
            s = int(elapsed_sec % 60)
            self._record_timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
        else:
            self._record_timer_label.setText("")

    def set_bp_mode_labels(self) -> None:
        """Update combo box labels for BP mode (cleaner-only, no separator)."""
        self._algo_combo.clear()
        self._algo_combo.addItems([
            "EMD Pulse (default)",
            "VMD+RLS Cleaner",
            "EMD Harmonic",
            "Passthrough (baseline)",
        ])
        self._ab_combo.clear()
        self._ab_combo.addItems([
            tr("ab_off"),
            "VMD+RLS Cleaner",
            "EMD Harmonic",
            "Passthrough (baseline)",
        ])

    def set_hr_mode_labels(self) -> None:
        """Update combo box labels for HR mode (cleaner+separator pairs)."""
        self._algo_combo.clear()
        self._algo_combo.addItems([
            tr("algo_adaptive"),
            tr("algo_vmd_wpd"),
            tr("algo_emd_wpd"),
            tr("algo_passthrough_sos"),
        ])
        self._ab_combo.clear()
        self._ab_combo.addItems([
            tr("ab_off"),
            tr("algo_vmd_wpd"),
            tr("algo_emd_wpd"),
            tr("algo_passthrough_sos"),
        ])

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
        dsp_telemetry: dict | None = None,
        benchmark_elapsed: float = 0.0,
    ) -> None:
        # Waveforms
        if len(breath_waveform) > 0:
            self._breath_wave.set_data(breath_waveform)
        if len(heart_waveform) > 0:
            self._heart_wave.set_data(heart_waveform)

        # BPM (标签 500ms 节流防闪烁)
        now = time.time()
        bpm_debounce = (now - self._last_bpm_label_update) >= 0.5
        if bpm_debounce:
            if breath_bpm > 0:
                self._breath_bpm_label.setText(f"{breath_bpm:.0f}")
            if heart_bpm > 0:
                self._heart_bpm_label.setText(f"{heart_bpm:.0f}")
            self._last_bpm_label_update = now

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
            if dsp_telemetry:
                algo = dsp_telemetry.get("current_algo", "--")
                lat = dsp_telemetry.get("current_latency_ms", 0.0)
                snr = dsp_telemetry.get("current_snr_gain_db", 0.0)
                lines.append(
                    tr("debug_dsp_current", algo, f"{lat:.0f}", snr)
                )
                if dsp_telemetry.get("ab_enabled"):
                    ab_algo = dsp_telemetry.get("ab_algo", "--")
                    ab_lat = dsp_telemetry.get("ab_latency_ms", 0.0)
                    ab_snr = dsp_telemetry.get("ab_snr_gain_db", 0.0)
                    lines.append(
                        tr("debug_dsp_ab", ab_algo, f"{ab_lat:.0f}", ab_snr)
                    )
            self._debug_panel.setText("  |  ".join(lines))
