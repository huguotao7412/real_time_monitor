"""主窗口 — 离线回放模式: 读取 .bin 文件 → DSP → 实时波形显示"""

import os
import glob
import time
import queue
import threading
from datetime import datetime

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QLabel, QSplitter, QPushButton,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

from config.protocol import UI_REFRESH_MS
from dsp_pipeline.pipeline import Pipeline
from dsp_pipeline.vital_signs import VitalSigns
from io_engine.bin_reader import BinFileReader
from io_engine.uart_parser import UartParser
from io_engine.serial_manager import SerialManager
from io_engine.radar_mgr import RadarMgr
from models.radar_frame import RadarFrame, FrameHeader
from ui.wave_widget import WaveWidget


class MainWindow(QMainWindow):
    def __init__(self, mode: str = "replay", replay_file: str | None = None):
        super().__init__()
        self.setWindowTitle("RS6240 毫米波雷达生命体征实时监测系统")
        self.resize(1200, 800)

        self._mode = mode
        self._replay_file = replay_file
        self._bin_reader: BinFileReader | None = None
        self._pipeline: Pipeline | None = None
        self._replay_timer: QTimer | None = None
        self._start_time: float = 0.0
        self._frame_count: int = 0
        self._running: bool = False
        self._latest_vitals: VitalSigns | None = None

        # Serial mode
        self._serial_mgr: SerialManager | None = None
        self._radar_mgr: RadarMgr | None = None
        self._uart_parser: UartParser | None = None
        self._io_thread = None
        self._stop_event = None
        self._serial_status: str = ""
        if mode == "serial":
            self._serial_mgr = SerialManager()
            self._radar_mgr = RadarMgr(self._serial_mgr)
            self._uart_parser = UartParser(bins_per_frame=1024)

        self._setup_ui()
        self._setup_timers()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Title + file selector
        title_row = QHBoxLayout()
        title = QLabel("RS6240 毫米波雷达生命体征实时监测系统")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title_row.addWidget(title, stretch=1)

        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("color: #95a5a6; font-size: 10pt;")
        title_row.addWidget(self.file_label)

        self.select_btn = QPushButton("选择文件")
        self.select_btn.clicked.connect(self._on_select_file)
        title_row.addWidget(self.select_btn)
        main_layout.addLayout(title_row)

        # Waveforms
        wave_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.breath_wave = WaveWidget("呼吸波形 (0.1-0.6 Hz)", "Amplitude")
        self.heart_wave = WaveWidget("心率波形 (0.8-2.5 Hz)", "Amplitude")
        wave_splitter.addWidget(self.breath_wave)
        wave_splitter.addWidget(self.heart_wave)
        main_layout.addWidget(wave_splitter, stretch=3)

        # BPM display
        bpm_widget = QWidget()
        bpm_layout = QHBoxLayout(bpm_widget)
        bpm_font = QFont("Segoe UI", 28, QFont.Weight.Bold)

        for name, color in [("呼吸频率", "#27ae60"), ("心率", "#e74c3c")]:
            group = QVBoxLayout()
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFont(QFont("Segoe UI", 12))
            group.addWidget(lbl)
            val = QLabel("--")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setFont(bpm_font)
            val.setStyleSheet(f"color: {color};")
            group.addWidget(val)
            if name == "呼吸频率":
                self.breath_bpm_value = val
            else:
                self.heart_bpm_value = val
            bpm_layout.addLayout(group)
        main_layout.addWidget(bpm_widget)

        # Controls
        ctrl_widget = QWidget()
        ctrl_layout = QHBoxLayout(ctrl_widget)

        label = "▶ 开始采集" if self._mode == "serial" else "▶ 开始回放"
        self.start_btn = QPushButton(label)
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-weight: bold; "
            "padding: 10px 24px; border-radius: 4px; font-size: 12pt; }"
            "QPushButton:hover { background-color: #2ecc71; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self.start_btn.clicked.connect(self._on_start)
        ctrl_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; "
            "padding: 10px 24px; border-radius: 4px; font-size: 12pt; }"
            "QPushButton:hover { background-color: #c0392b; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self.stop_btn.clicked.connect(self._on_stop)
        ctrl_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("💾 保存当前数据")
        self.save_btn.clicked.connect(self._on_save)
        ctrl_layout.addWidget(self.save_btn)

        ctrl_layout.addStretch()

        self.status_label = QLabel("● 待机")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setStyleSheet("color: #f39c12;")
        ctrl_layout.addWidget(self.status_label)

        self.frame_rate_label = QLabel("帧率: --")
        ctrl_layout.addWidget(self.frame_rate_label)

        self.elapsed_label = QLabel("运行: 00:00")
        ctrl_layout.addWidget(self.elapsed_label)

        main_layout.addWidget(ctrl_widget)

        # Auto-select latest file on startup
        if not self._replay_file:
            self._replay_file = self._find_latest_bin()
        if self._replay_file:
            self.file_label.setText(os.path.basename(self._replay_file))
            self.file_label.setStyleSheet("color: #3498db; font-size: 10pt;")

    def _setup_timers(self) -> None:
        self._ui_timer = QTimer()
        self._ui_timer.timeout.connect(self._on_ui_tick)
        self._ui_timer.start(UI_REFRESH_MS)  # ~30 fps

    @staticmethod
    def _find_latest_bin() -> str | None:
        candidates = sorted(
            glob.glob("data/*.bin"),
            key=lambda f: os.path.getmtime(f),
            reverse=True,
        )
        return candidates[0] if candidates else None

    # === Slots ===

    def _on_select_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 .bin 文件", "data", "Bin Files (*.bin);;All Files (*)"
        )
        if path:
            self._replay_file = path
            self.file_label.setText(os.path.basename(path))
            self.file_label.setStyleSheet("color: #3498db; font-size: 10pt;")

    def _on_start(self) -> None:
        if self._mode == "serial":
            self._start_serial()
        else:
            self._start_replay()

    def _start_serial(self) -> None:
        # Disable button immediately
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.select_btn.setEnabled(False)
        self.status_label.setText("● 启动中...")
        self.status_label.setStyleSheet("color: #f39c12;")

        # Move ALL serial init to background thread (prevents UI freeze)
        thread = threading.Thread(target=self._serial_init_thread, daemon=True)
        thread.start()

    def _serial_init_thread(self) -> None:
        try:
            self._do_serial_init()
        except Exception as e:
            import traceback
            print(f"[Serial Init] CRASH: {e}")
            traceback.print_exc()
            self._serial_status = f"启动失败: {e}"

    def _do_serial_init(self) -> None:
        import serial.tools.list_ports
        print("[Serial Init] Scanning ports...")

        ports = SerialManager.list_ports()
        ctrl_port = data_port = ""
        for p in ports:
            try:
                info = next((i for i in serial.tools.list_ports.comports() if i.device == p), None)
                desc = info.description if info else ""
            except Exception:
                desc = ""
            if "Standard" in desc:
                ctrl_port = p
            if "Enhanced" in desc:
                data_port = p

        print(f"[Serial Init] Found: ctrl={ctrl_port}, data={data_port}")

        if not ctrl_port or not data_port:
            self._serial_status = f"未找到雷达: ctrl={ctrl_port} data={data_port}"
            return

        print("[Serial Init] Connecting...")
        if not self._radar_mgr.connect(ctrl_port, data_port):
            self._serial_status = f"连接失败 {ctrl_port}/{data_port}"
            return

        print("[Serial Init] Booting radar...")
        ok = self._radar_mgr.boot()
        print(f"[Serial Init] Boot {'OK' if ok else 'PARTIAL FAIL'}")

        self._stop_event = threading.Event()
        self._uart_parser.reset()
        self._pipeline = Pipeline()
        self._pipeline.start()

        self._start_time = time.time()
        self._frame_count = 0
        self._running = True
        self._serial_status = f"采集中 ({ctrl_port}/{data_port})"

        print("[Serial Init] Starting I/O loop...")
        self._io_thread = threading.Thread(target=self._serial_io_loop, daemon=True)
        self._io_thread.start()
        print("[Serial Init] Done!")

    def _serial_io_loop(self) -> None:
        while self._running and not self._stop_event.is_set():
            try:
                raw = self._serial_mgr.read_data(4096)
                if not raw:
                    continue
                frames = self._uart_parser.feed(raw)
                for fft_data in frames:
                    self._frame_count += 1
                    # Reshape [1024] → [TX=2, RX=4, BIN=128]
                    cube = fft_data.reshape(2, 4, 128)
                    # Combine all RX channels for TX0 (better SNR)
                    rx_combined = np.mean(cube[0, :, :], axis=0)  # [128] avg over 4 RX
                    frame = RadarFrame(
                        timestamp=time.time(),
                        frame_index=self._frame_count,
                        header=FrameHeader(0, 1, 4, 2, 58000, 128, 1, 3000, 25, 1920, 60),
                        data_cube=rx_combined.reshape(-1, 1, 1),  # [128, 1, 1]
                    )
                    while True:
                        try:
                            self._pipeline.raw_queue.put_nowait(frame)
                            break
                        except queue.Full:
                            try:
                                self._pipeline.raw_queue.get_nowait()
                            except queue.Empty:
                                pass
            except Exception as e:
                print(f"[Serial I/O] {e}")
                time.sleep(0.5)

    def _start_replay(self) -> None:
        if not self._replay_file or not os.path.exists(self._replay_file):
            QMessageBox.critical(self, "错误", "请先选择有效的 .bin 文件")
            return

        reader = BinFileReader(self._replay_file)
        if not reader.open():
            QMessageBox.critical(self, "错误", f"无法打开 {self._replay_file}")
            return

        self._bin_reader = reader
        self._pipeline = Pipeline()
        self._pipeline.start()

        self._start_time = time.time()
        self._frame_count = 0
        self._running = True

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.select_btn.setEnabled(False)
        self.status_label.setText("● 回放中")
        self.status_label.setStyleSheet("color: #27ae60;")

        # Feed frames at 20 fps via timer
        self._replay_timer = QTimer()
        self._replay_timer.timeout.connect(self._feed_one_frame)
        self._replay_timer.start(50)  # 50ms = 20 fps

    def _feed_one_frame(self) -> None:
        if not self._bin_reader:
            self._replay_timer.stop()
            return

        frames = self._bin_reader.read_frames(max_frames=1)
        if not frames:
            self._replay_timer.stop()
            self.status_label.setText("● 回放完毕")
            self.status_label.setStyleSheet("color: #3498db;")
            self.stop_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
            self.select_btn.setEnabled(True)
            return

        self._frame_count += 1
        cube = frames[0].reshape(-1, 1, 1)  # [bins, 1, 1]

        frame = RadarFrame(
            timestamp=time.time(),
            frame_index=self._frame_count,
            header=FrameHeader(0, 1, 1, 1, 58000, 128, 0, 3000, 25, 1920, 60),
            data_cube=cube,
        )

        while True:
            try:
                self._pipeline.raw_queue.put_nowait(frame)
                break
            except queue.Full:
                try:
                    self._pipeline.raw_queue.get_nowait()
                except queue.Empty:
                    pass

    def _on_stop(self) -> None:
        self._running = False

        if self._mode == "serial":
            if self._radar_mgr:
                self._radar_mgr.shutdown()
            if self._stop_event:
                self._stop_event.set()
            if self._io_thread:
                self._io_thread.join(timeout=3)
            if self._serial_mgr:
                self._serial_mgr.close()

        if self._replay_timer:
            self._replay_timer.stop()
            self._replay_timer = None

        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None

        if self._bin_reader:
            self._bin_reader.close()
            self._bin_reader = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.select_btn.setEnabled(True)
        self.status_label.setText("● 已停止")
        self.status_label.setStyleSheet("color: #f39c12;")

    def _on_save(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not path:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self._latest_vitals is not None:
            np.savez(
                f"{path}/vital_signs_{ts}.npz",
                breath_waveform=self._latest_vitals.breath_waveform,
                heart_waveform=self._latest_vitals.heart_waveform,
                breath_bpm=self._latest_vitals.breath_bpm,
                heart_bpm=self._latest_vitals.heart_bpm,
            )
        QMessageBox.information(self, "保存完成", f"数据已保存至 {path}")

    # === UI Timer ===

    def _on_ui_tick(self) -> None:
        # Poll serial status from main thread
        if self._serial_status:
            s = self._serial_status
            if "失败" in s or "未找到" in s or "连接失败" in s:
                self.status_label.setText(f"● {s}")
                self.status_label.setStyleSheet("color: #e74c3c;")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.select_btn.setEnabled(True)
            elif "采集中" in s:
                self.status_label.setText(f"● {s}")
                self.status_label.setStyleSheet("color: #27ae60;")
            self._serial_status = ""  # Consumed

        if not self._pipeline:
            return

        try:
            while not self._pipeline.display_queue.empty():
                self._latest_vitals = self._pipeline.display_queue.get_nowait()
        except queue.Empty:
            pass

        if self._latest_vitals is not None:
            q = self._latest_vitals.quality
            if q and not q.get("valid"):
                # Signal quality failed — show why
                self.breath_bpm_value.setText("--")
                self.heart_bpm_value.setText("--")
                self.status_label.setText(f"● 信号无效 ({q.get('reason', '')})")
                self.status_label.setStyleSheet("color: #e74c3c;")
            else:
                self.breath_wave.set_data(self._latest_vitals.breath_waveform)
                if len(self._latest_vitals.heart_waveform) > 0:
                    self.heart_wave.set_data(self._latest_vitals.heart_waveform)
                if self._latest_vitals.breath_bpm > 0:
                    self.breath_bpm_value.setText(f"{self._latest_vitals.breath_bpm:.0f}")
                if self._latest_vitals.heart_bpm > 0:
                    self.heart_bpm_value.setText(f"{self._latest_vitals.heart_bpm:.0f}")
                self.status_label.setText("● 运行中")
                self.status_label.setStyleSheet("color: #27ae60;")

        if self._start_time > 0:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                self.frame_rate_label.setText(f"帧率: {self._frame_count / elapsed:.1f} fps")
            m, s = divmod(int(elapsed), 60)
            self.elapsed_label.setText(f"运行: {m:02d}:{s:02d}")

    def closeEvent(self, event) -> None:
        self._on_stop()
        event.accept()
