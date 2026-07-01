"""Research mode tab — full DSP data, pulse waveform, quality dashboard, trend panel, debug."""

import time
from collections import deque
from scipy.signal import find_peaks

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, QFrame,
    QPushButton, QTextEdit, QComboBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.wave_widget import WaveWidget
from ui.sqi_indicator import SqiIndicator
import pyqtgraph as pg
from ui.trend_panel import TrendPanel
from config.i18n import tr, I18n

class _PulseWaveWidget(QWidget):
    def __init__(self, title="", parent=None, fs=20.0):
        super().__init__(parent)
        self._fs=fs; self._max_pts=int(3*fs)
        self._data=np.zeros(self._max_pts,dtype=np.float32)
        l=QVBoxLayout(self); l.setContentsMargins(0,0,0,0)
        self._tl=QLabel(title)
        self._tl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tl.setStyleSheet("font-weight:bold;font-size:12pt")
        l.addWidget(self._tl)
        self._plt=pg.PlotWidget()
        self._plt.setMouseEnabled(0,0); self._plt.hideAxis("bottom"); self._plt.hideAxis("left")
        self._plt.showGrid(1,1,alpha=0.25)
        self._wv=self._plt.plot(pen=pg.mkPen((200,80,80),width=1.5))
        self._pk=pg.ScatterPlotItem(size=8,symbol="t1",brush=pg.mkBrush(255,50,50))
        self._nc=pg.ScatterPlotItem(size=7,symbol="d",brush=pg.mkBrush(50,200,220))
        self._plt.addItem(self._pk); self._plt.addItem(self._nc)
        l.addWidget(self._plt)
    def set_data(self,wf):
        if len(wf)<2: return
        if len(wf)>self._max_pts: wf=wf[-self._max_pts:]
        elif len(wf)<self._max_pts: wf=np.concatenate([np.zeros(self._max_pts-len(wf)),wf])
        self._data=wf.astype(np.float32); x=np.arange(len(self._data))
        self._wv.setData(x,self._data)
        try:
            p,_=find_peaks(self._data,10,prominence=0.05)
            if len(p)>0:
                nx,ny=[],[]
                for pk in p:
                    s=int(pk)+1; e=min(int(pk)+int(self._fs*0.3),len(self._data))
                    if e-s<2: continue
                    seg=self._data[s:e]; lm=np.argmin(seg)
                    nv=seg[lm]; pv=self._data[pk]
                    if nv<pv-0.15*(pv-np.min(self._data)):
                        nx.append(float(s+lm)); ny.append(float(nv))
                self._pk.setData(x[p],self._data[p])
                self._nc.setData(nx,ny)
            else: self._pk.clear(); self._nc.clear()
        except: self._pk.clear(); self._nc.clear()

class _QualityDashboard(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self._m={}
        l=QHBoxLayout(self); l.setContentsMargins(4,4,4,4)
        for k,lb,cl in [("pr","PhaseRng","#27ae60"),("br","BreathSNR","#f39c12"),
                        ("hp","HeartSNR","#e74c3c"),("td","Target","#3498db")]:
            c=QWidget(); c.setStyleSheet("QWidget{background:#2a2a3a;border-radius:4px}")
            cl=QVBoxLayout(c); cl.setContentsMargins(6,4,6,4)
            cl.addWidget(QLabel(lb))
            v=QLabel("--"); v.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v.setStyleSheet(f"color:{cl};font-size:14pt;font-weight:bold")
            cl.addWidget(v); l.addWidget(c); self._m[k]=(v,cl)
        l.addStretch()
    def set_mode(self, is_bp_mode: bool) -> None:
        if is_bp_mode:
            self._m["br"][1].itemAt(0).widget().setText("Net Conf")
            self._m["hp"][1].itemAt(0).widget().setText("Peaks/HR")
        else:
            self._m["br"][1].itemAt(0).widget().setText("BreathSNR")
            self._m["hp"][1].itemAt(0).widget().setText("HeartSNR")

    def update(self,pr=0.0,br=0.0,hp=0.0,td=0.0):
        vals={"pr":pr,"br":br*100,"hp":hp*10,"td":td}
        for k,v in vals.items():
            if k in self._m:
                self._m[k][0].setText(f"{v:.1f}" if v>=0 else "--")



class ResearchTab(QWidget):
    """Full researcher view: dual waveforms, BPM, trend, collapsible debug panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sqi_level = 0
        self._pulse_wave = _PulseWaveWidget(fs=20.0)
        self._debug_expanded = False
        self._last_bpm_label_update: float = 0.0
        self._bpm_widgets: list[QWidget] = []
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

        # Quality dashboard
        self._qd = _QualityDashboard(self)
        layout.addWidget(self._qd)

        # Waveforms
        wave_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._breath_wave = WaveWidget(tr("resp_wave_title"), "Amplitude")
        self._heart_wave = WaveWidget(tr("heart_wave_title"), "Amplitude")
        wave_splitter.addWidget(self._breath_wave)
        wave_splitter.addWidget(self._heart_wave)
        layout.addWidget(wave_splitter, stretch=3)
        layout.addWidget(self._pulse_wave, stretch=3)
        self._pulse_wave.setVisible(False)  # 默认 HR 模式隐藏

        # BPM row + SQI (wrapped for mode-aware visibility toggle)
        self._bpm_container = QWidget()
        bpm_row = QHBoxLayout(self._bpm_container)
        bpm_row.setContentsMargins(0, 0, 0, 0)
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
        layout.addWidget(self._bpm_container)

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
        self._pulse_wave.set_data(np.array([], dtype=np.float32))
        self._sqi.set_level(0.0, 0.0)
        self._qd.update(0,0,0,0)
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
        target_distance_m: float = 0.0,
        benchmark_elapsed: float = 0.0,
    ) -> None:
        # Waveforms
        if len(breath_waveform) > 0:
            self._breath_wave.set_data(breath_waveform)
        if len(heart_waveform) > 0:
            self._heart_wave.set_data(heart_waveform)
            self._pulse_wave.set_data(heart_waveform)

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

        # Quality Dashboard
        heart_prom = quality.get("heart_prominence", 0.0) if quality else 0.0
        self._qd.update(
            pr=phase_range,
            br=breath_ratio,
            hp=heart_prom,
            td=target_distance_m,
        )

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

    # ── BP mode support ──

    def set_mode(self, is_bp_mode: bool) -> None:
        """切换界面布局以适应不同的监测模式。"""
        self._qd.set_mode(is_bp_mode)
        if is_bp_mode:
            # 隐藏双波形 splitter，显示单路脉搏波
            self._breath_wave.parentWidget().setVisible(False)
            self._pulse_wave.setVisible(True)
            # 隐藏下方的呼吸/心率大字 BPM 区域及 SQI
            self._bpm_container.setVisible(False)
        else:
            self._breath_wave.parentWidget().setVisible(True)
            self._pulse_wave.setVisible(False)
            self._bpm_container.setVisible(True)

    def update_display_bp(self, bp_result) -> None:
        """专门用于血压模式的数据更新。"""
        r = bp_result
        # 1. 更新脉搏波形
        if len(r.bp_waveform) > 0:
            self._pulse_wave.set_data(r.bp_waveform)

        # 2. 提取质量指标
        quality = r.quality or {}
        phase_range = quality.get("phase_range", 0.0)
        conf = quality.get("confidence", 0.0)
        n_peaks = quality.get("n_peaks", 0)

        # 将峰值数量转换为隐含心率展示 (与 BP_Tab 逻辑一致)
        implicit_hr = (n_peaks / 5.12 * 60.0) if n_peaks > 0 else 0.0

        # 3. 更新 Dashboard
        self._qd.update(
            pr=phase_range,
            br=conf / 100.0,
            hp=implicit_hr / 10.0,
            td=r.target_distance_m,
        )

        # 4. 更新 Debug 面板
        if self._debug_expanded:
            lines = [
                f"phase_range: {phase_range:.4f}",
                f"confidence: {conf:.2f}",
                f"n_peaks: {n_peaks}",
                f"SBP: {r.sbp:.1f} | DBP: {r.dbp:.1f}",
            ]
            self._debug_panel.setText("  |  ".join(lines))
