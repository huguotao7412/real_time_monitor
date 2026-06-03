"""Main window — thin shell hosting SubjectTab and ResearchTab via QTabWidget."""

import os
import glob
import time
import queue
import threading
from datetime import datetime

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QLabel, QPushButton,
    QTabWidget,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

from config.protocol import UI_REFRESH_MS
from config.i18n import tr, I18n
from dsp_pipeline.pipeline import Pipeline
from dsp_pipeline.vital_signs import VitalSigns
from io_engine.bin_reader import BinFileReader
from io_engine.uart_parser import UartParser
from io_engine.serial_manager import SerialManager
from io_engine.radar_mgr import RadarMgr
from io_engine.data_exporter import export_csv, export_hdf5, export_edf
from models.radar_frame import RadarFrame, FrameHeader

from ui.subject_tab import SubjectTab
from ui.research_tab import ResearchTab


class MainWindow(QMainWindow):
    def __init__(self, mode: str = "replay", replay_file: str | None = None):
        super().__init__()
        self.setWindowTitle(tr("window_title"))
        self.resize(1200, 800)

        self._mode = mode
        self._bp_mode = False  # Start in HR mode, toggle button switches
        self._replay_file = replay_file
        self._bin_reader: BinFileReader | None = None
        self._pipeline: Pipeline | None = None
        self._bp_pipeline = None  # type: ignore  # BPPipeline, lazy import
        self._replay_timer: QTimer | None = None
        self._start_time: float = 0.0
        self._frame_count: int = 0
        self._running: bool = False
        self._latest_vitals: VitalSigns | None = None
        self._latest_bp_result = None  # type: ignore  # BPResult
        self._trend_tick_counter: int = 0

        # Data accumulation for export (CSV + HDF5 + EDF)
        self._csv_rows: list[dict] = []
        self._breath_waveform_accum: list[np.ndarray] = []
        self._heart_waveform_accum: list[np.ndarray] = []
        self._bpm_history: list[tuple[float, float, float]] = []   # (t, rpm, bpm)
        self._sqi_history: list[dict] = []  # {phase_range, breath_ratio, sqi_level}

        # Serial mode
        self._serial_mgr: SerialManager | None = None
        self._radar_mgr: RadarMgr | None = None
        self._uart_parser: UartParser | None = None
        self._io_thread = None
        self._stop_event = None
        self._serial_status: str = ""
        self._serial_error: bool = False
        if mode == "serial":
            self._serial_mgr = SerialManager()
            self._radar_mgr = RadarMgr(self._serial_mgr)
            self._uart_parser = UartParser(bins_per_frame=1024)

        self._setup_ui()
        self._setup_timers()

    def _setup_ui(self) -> None:
        # Menu bar with language switch
        menubar = self.menuBar()
        lang_menu = menubar.addMenu(tr("menu_language"))
        zh_action = lang_menu.addAction(tr("lang_zh"))
        zh_action.triggered.connect(lambda: I18n.set_language("zh"))
        en_action = lang_menu.addAction(tr("lang_en"))
        en_action.triggered.connect(lambda: I18n.set_language("en"))
        self._lang_menu = lang_menu
        self._zh_action = zh_action
        self._en_action = en_action

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Title bar
        title_row = QHBoxLayout()
        title_row.setContentsMargins(12, 8, 12, 4)
        self._title_label = QLabel(tr("app_title"))
        self._title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title_row.addWidget(self._title_label, stretch=1)

        self._file_label = QLabel(tr("file_not_selected"))
        self._file_label.setStyleSheet("color: #95a5a6; font-size: 9pt;")
        title_row.addWidget(self._file_label)

        if self._mode == "replay":
            self._select_btn = QPushButton(tr("btn_select_file"))
            self._select_btn.clicked.connect(self._on_select_file)
            title_row.addWidget(self._select_btn)

        main_layout.addLayout(title_row)

        # Tab widget — always create all 3 tabs
        self._tabs = QTabWidget()
        self._subject_tab = SubjectTab()
        from ui.bp_tab import BPTab
        self._bp_tab = BPTab()
        self._research_tab = ResearchTab()
        self._tabs.addTab(self._subject_tab, tr("tab_subject"))
        self._tabs.addTab(self._bp_tab, tr("tab_bp"))
        self._tabs.addTab(self._research_tab, tr("tab_research"))
        main_layout.addWidget(self._tabs, stretch=1)

        # Control bar
        ctrl_row = QHBoxLayout()
        ctrl_row.setContentsMargins(12, 4, 12, 8)

        label = tr("btn_start_capture") if self._mode == "serial" else tr("btn_start_replay")
        self._start_btn = QPushButton(label)
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-weight: bold; "
            "padding: 8px 20px; border-radius: 4px; font-size: 11pt; }"
            "QPushButton:hover { background-color: #2ecc71; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self._start_btn.clicked.connect(self._on_start)
        ctrl_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton(tr("btn_stop"))
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; "
            "padding: 8px 20px; border-radius: 4px; font-size: 11pt; }"
            "QPushButton:hover { background-color: #c0392b; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self._stop_btn.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self._stop_btn)

        self._save_btn = QPushButton(tr("btn_save"))
        self._save_btn.clicked.connect(self._on_save)
        ctrl_row.addWidget(self._save_btn)

        # Mode toggle button (serial only)
        if self._mode == "serial":
            self._mode_btn = QPushButton(
                tr("btn_mode_bp") if self._bp_mode else tr("btn_mode_hr")
            )
            self._mode_btn.setStyleSheet(
                "QPushButton { background-color: #8e44ad; color: white; font-weight: bold; "
                "padding: 8px 16px; border-radius: 4px; font-size: 10pt; }"
                "QPushButton:hover { background-color: #9b59b6; }"
                "QPushButton:disabled { background-color: #95a5a6; }"
            )
            self._mode_btn.clicked.connect(self._on_toggle_mode)
            ctrl_row.addWidget(self._mode_btn)

        ctrl_row.addStretch()

        self._status_label = QLabel(tr("status_standby"))
        self._status_label.setFont(QFont("Segoe UI", 10))
        self._status_label.setStyleSheet("color: #f39c12;")
        ctrl_row.addWidget(self._status_label)

        self._frame_rate_label = QLabel(tr("frame_rate_na"))
        ctrl_row.addWidget(self._frame_rate_label)

        self._elapsed_label = QLabel(tr("elapsed_na"))
        ctrl_row.addWidget(self._elapsed_label)

        main_layout.addLayout(ctrl_row)

        # Auto-select latest file
        if self._mode == "replay" and not self._replay_file:
            self._replay_file = self._find_latest_bin()
        if self._replay_file:
            self._file_label.setText(os.path.basename(self._replay_file))
            self._file_label.setStyleSheet("color: #3498db; font-size: 9pt;")

        I18n.instance().language_changed.connect(self.update_ui_texts)

    def update_ui_texts(self, _lang: str = "") -> None:
        self.setWindowTitle(tr("window_title"))
        self._title_label.setText(tr("app_title"))
        self._file_label.setText(tr("file_not_selected"))
        if hasattr(self, '_select_btn'):
            self._select_btn.setText(tr("btn_select_file"))
        self._tabs.setTabText(0, tr("tab_subject"))
        self._tabs.setTabText(1, tr("tab_research"))
        label = tr("btn_start_capture") if self._mode == "serial" else tr("btn_start_replay")
        self._start_btn.setText(label)
        self._stop_btn.setText(tr("btn_stop"))
        self._save_btn.setText(tr("btn_save"))
        self._lang_menu.setTitle(tr("menu_language"))
        self._zh_action.setText(tr("lang_zh"))
        self._en_action.setText(tr("lang_en"))
        # Status labels are set dynamically; keep current state

    def _setup_timers(self) -> None:
        self._ui_timer = QTimer()
        self._ui_timer.timeout.connect(self._on_ui_tick)
        self._ui_timer.start(UI_REFRESH_MS)

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
            self, tr("dialog_select_bin"), "data", "Bin Files (*.bin);;All Files (*)"
        )
        if path:
            self._replay_file = path
            self._file_label.setText(os.path.basename(path))
            self._file_label.setStyleSheet("color: #3498db; font-size: 9pt;")

    def _on_start(self) -> None:
        if self._mode == "serial":
            self._start_serial()
        else:
            self._start_replay()

    def _start_serial(self) -> None:
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        if hasattr(self, '_select_btn'):
            self._select_btn.setEnabled(False)
        if hasattr(self, '_mode_btn'):
            self._mode_btn.setEnabled(False)
        self._status_label.setText(tr("status_starting"))
        self._status_label.setStyleSheet("color: #f39c12;")
        self._start_serial_io()

    def _serial_init_thread(self) -> None:
        try:
            self._do_serial_init()
        except Exception as e:
            import traceback
            print(f"[Serial Init] CRASH: {e}")
            traceback.print_exc()
            self._serial_error = True
            self._serial_status = tr("serial_init_failed", str(e))

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
            self._serial_error = True
            self._serial_status = tr("serial_not_found", ctrl_port, data_port)
            return
        print("[Serial Init] Connecting...")
        if not self._radar_mgr.connect(ctrl_port, data_port):
            self._serial_error = True
            self._serial_status = tr("serial_connect_failed", ctrl_port, data_port)
            return
        print("[Serial Init] Booting radar...")
        if self._bp_mode:
            ok = self._radar_mgr.boot_bp()
        else:
            ok = self._radar_mgr.boot()
        print(f"[Serial Init] Boot {'OK' if ok else 'PARTIAL FAIL'}")
        self._stop_event = threading.Event()
        self._uart_parser.reset()
        if self._bp_mode:
            from bp_monitor.bp_pipeline import BPPipeline
            self._bp_pipeline = BPPipeline("bp_matlab/bp_weights.mat")
            self._bp_pipeline.start()
            self._pipeline = None
        else:
            self._pipeline = Pipeline()
            self._pipeline.start()
            self._bp_pipeline = None
        self._start_time = time.time()
        self._frame_count = 0
        self._running = True
        self._csv_rows.clear()
        self._breath_waveform_accum.clear()
        self._heart_waveform_accum.clear()
        self._bpm_history.clear()
        self._sqi_history.clear()
        self._research_tab.start()
        self._serial_error = False
        self._serial_status = tr("serial_capturing", ctrl_port, data_port)
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
                    frame = self._build_radar_frame(fft_data)
                    if self._bp_mode:
                        target_queue = self._bp_pipeline.raw_queue
                    else:
                        target_queue = self._pipeline.raw_queue
                    while True:
                        try:
                            target_queue.put_nowait(frame)
                            break
                        except queue.Full:
                            try:
                                target_queue.get_nowait()
                            except queue.Empty:
                                pass
            except Exception as e:
                print(f"[Serial I/O] {e}")
                time.sleep(0.5)

    def _build_bp_frame(self, fft_data: np.ndarray) -> RadarFrame:
        """Build a RadarFrame from raw FFT data in BP mode (1T1R, 32 bins)."""
        rx_combined = fft_data.ravel()[:32]
        return RadarFrame(
            timestamp=time.time(),
            frame_index=self._frame_count,
            header=FrameHeader(0, 1, 1, 1, 60000, 32, 1, 160, 50, 0, 0, 0, 5),
            data_cube=rx_combined.reshape(32, 1, 1),
        )

    def _build_hr_frame(self, fft_data: np.ndarray) -> RadarFrame:
        """Build a RadarFrame from raw FFT data in HR mode (2T4R, 128 bins)."""
        cube = fft_data.reshape(2, 4, -1)
        rx_combined = np.mean(cube[0, :, :], axis=0)
        return RadarFrame(
            timestamp=time.time(),
            frame_index=self._frame_count,
            header=FrameHeader(0, 1, 4, 2, 58000, 128, 1, 3000, 25, 1920, 60),
            data_cube=rx_combined.reshape(-1, 1, 1),
        )

    def _build_radar_frame(self, fft_data: np.ndarray) -> RadarFrame:
        """Build RadarFrame for current mode (BP or HR)."""
        if self._bp_mode:
            return self._build_bp_frame(fft_data)
        return self._build_hr_frame(fft_data)

    def _start_replay(self) -> None:
        if not self._replay_file or not os.path.exists(self._replay_file):
            QMessageBox.critical(self, tr("dialog_error"), tr("dialog_no_valid_file"))
            return
        reader = BinFileReader(self._replay_file)
        if not reader.open():
            QMessageBox.critical(self, tr("dialog_error"), tr("dialog_cannot_open", self._replay_file))
            return
        self._bin_reader = reader
        self._pipeline = Pipeline()
        self._pipeline.start()
        self._start_time = time.time()
        self._frame_count = 0
        self._running = True
        self._csv_rows.clear()
        self._breath_waveform_accum.clear()
        self._heart_waveform_accum.clear()
        self._bpm_history.clear()
        self._sqi_history.clear()
        self._research_tab.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        if hasattr(self, '_select_btn'):
            self._select_btn.setEnabled(False)
        self._status_label.setText(tr("status_playing"))
        self._status_label.setStyleSheet("color: #27ae60;")
        self._replay_timer = QTimer()
        self._replay_timer.timeout.connect(self._feed_one_frame)
        self._replay_timer.start(50)

    def _feed_one_frame(self) -> None:
        if not self._bin_reader:
            self._replay_timer.stop()
            return
        frames = self._bin_reader.read_frames(max_frames=1)
        if not frames:
            self._replay_timer.stop()
            self._status_label.setText(tr("status_done"))
            self._status_label.setStyleSheet("color: #3498db;")
            self._stop_btn.setEnabled(False)
            self._start_btn.setEnabled(True)
            if hasattr(self, '_select_btn'):
                self._select_btn.setEnabled(True)
            return
        self._frame_count += 1
        frame = self._build_radar_frame(frames[0])
        while True:
            try:
                self._pipeline.raw_queue.put_nowait(frame)
                break
            except queue.Full:
                try:
                    self._pipeline.raw_queue.get_nowait()
                except queue.Empty:
                    pass

    def _on_toggle_mode(self) -> None:
        """Hot-switch between HR and BP monitoring modes."""
        was_running = self._running

        # 1. Stop current I/O
        if was_running:
            self._running = False
            if self._stop_event:
                self._stop_event.set()
            if self._io_thread:
                self._io_thread.join(timeout=3)

        # 2. Stop pipeline
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        if self._bp_pipeline:
            self._bp_pipeline.stop()
            self._bp_pipeline = None

        # 3. Shutdown radar (keep serial ports open)
        if self._radar_mgr:
            self._radar_mgr.shutdown()

        # 4. Switch mode
        self._bp_mode = not self._bp_mode

        # 5. Update UART parser
        from io_engine.uart_parser import UartParser
        bins = 32 if self._bp_mode else 1024
        self._uart_parser = UartParser(bins_per_frame=bins)

        # 6. Update UI
        self._mode_btn.setText(
            tr("btn_mode_hr") if self._bp_mode else tr("btn_mode_bp")
        )

        # 7-8. If was running, restart radar + pipeline in new mode.
        # If not running, defer to _on_serial_start.
        if was_running:
            if self._bp_mode:
                self._radar_mgr.boot_bp()
                from bp_monitor.bp_pipeline import BPPipeline
                self._bp_pipeline = BPPipeline("bp_matlab/bp_weights.mat")
                self._bp_pipeline.start()
                self._latest_vitals = None
            else:
                self._radar_mgr.boot()
                self._pipeline = Pipeline()
                self._pipeline.start()
                self._latest_bp_result = None

        # 9. Restart I/O if was running
        if was_running:
            self._running = True
            self._stop_event = threading.Event()
            self._uart_parser.reset()
            self._frame_count = 0
            self._start_time = time.time()
            self._io_thread = threading.Thread(
                target=self._serial_io_loop, daemon=True)
            self._io_thread.start()
            self._status_label.setText(
                tr("serial_capturing", "COM?", "COM?") if "COM?" in tr("serial_capturing", "COM?", "COM?") else "Capturing")
            self._status_label.setStyleSheet("color: #27ae60;")
        else:
            self._status_label.setText(tr("status_standby"))
            self._status_label.setStyleSheet("color: #f39c12;")

    def _start_serial_io(self) -> None:
        """Start radar boot + pipeline in current mode (non-blocking thread)."""
        thread = threading.Thread(target=self._serial_init_thread, daemon=True)
        thread.start()

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
        if self._bp_pipeline:
            self._bp_pipeline.stop()
            self._bp_pipeline = None
        if self._bin_reader:
            self._bin_reader.close()
            self._bin_reader = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if hasattr(self, '_select_btn'):
            self._select_btn.setEnabled(True)
        if hasattr(self, '_mode_btn'):
            self._mode_btn.setEnabled(True)
        self._status_label.setText(tr("status_stopped"))
        self._status_label.setStyleSheet("color: #f39c12;")

    def _on_save(self) -> None:
        from PyQt6.QtWidgets import QInputDialog

        formats = [
            tr("export_format_csv"),
            tr("export_format_hdf5"),
            tr("export_format_edf"),
        ]
        choice, ok = QInputDialog.getItem(
            self, tr("export_title"), "Format:", formats, 0, False,
        )
        if not ok or not choice:
            return

        path = QFileDialog.getExistingDirectory(self, tr("dialog_save_dir"))
        if not path:
            return

        try:
            if choice == tr("export_format_csv"):
                self._do_export_csv(path)
            elif choice == tr("export_format_hdf5"):
                self._do_export_hdf5(path)
            elif choice == tr("export_format_edf"):
                self._do_export_edf(path)
            QMessageBox.information(
                self, tr("dialog_save_done"), tr("dialog_save_done_msg", path)
            )
        except ImportError as e:
            QMessageBox.critical(self, tr("dialog_error"), str(e))
        except Exception as e:
            QMessageBox.critical(self, tr("dialog_error"), str(e))

    def _do_export_csv(self, path: str) -> None:
        breath = self._latest_vitals.breath_waveform if self._latest_vitals else np.array([])
        heart = self._latest_vitals.heart_waveform if self._latest_vitals else np.array([])
        export_csv(path, self._csv_rows, breath, heart)

    def _do_export_hdf5(self, path: str) -> None:
        breath_hist = (
            np.array(self._breath_waveform_accum)
            if self._breath_waveform_accum
            else np.array([])
        )
        heart_hist = (
            np.array(self._heart_waveform_accum)
            if self._heart_waveform_accum
            else np.array([])
        )
        metadata = {
            "device": "RS6240",
            "fs": 20,
            "session_duration_s": time.time() - self._start_time if self._start_time > 0 else 0,
        }
        export_hdf5(path, breath_hist, heart_hist,
                    self._bpm_history, self._sqi_history, metadata)

    def _do_export_edf(self, path: str) -> None:
        breath = self._latest_vitals.breath_waveform if self._latest_vitals else np.array([])
        heart = self._latest_vitals.heart_waveform if self._latest_vitals else np.array([])
        export_edf(path, breath, heart, fs=20.0)

    # === UI Timer ===

    def _poll_bp_results(self) -> None:
        """Poll BP pipeline display queue and route to BPTab."""
        try:
            while not self._bp_pipeline.display_queue.empty():
                self._latest_bp_result = self._bp_pipeline.display_queue.get_nowait()
        except queue.Empty:
            pass

        if self._latest_bp_result is not None and self._bp_tab is not None:
            self._bp_tab.update_display(self._latest_bp_result)
            self._status_label.setText("● Monitoring")
            self._status_label.setStyleSheet("color: #27ae60;")

    def _on_ui_tick(self) -> None:
        # Poll serial status
        if self._serial_status:
            s = self._serial_status
            if self._serial_error:
                self._status_label.setText(f"● {s}")
                self._status_label.setStyleSheet("color: #e74c3c;")
                self._start_btn.setEnabled(True)
                self._stop_btn.setEnabled(False)
                if hasattr(self, '_select_btn'):
                    self._select_btn.setEnabled(True)
            else:
                self._status_label.setText(f"● {s}")
                self._status_label.setStyleSheet("color: #27ae60;")
            self._serial_status = ""

        if self._bp_mode and self._bp_pipeline is not None:
            self._poll_bp_results()
            return

        if not self._pipeline:
            return

        try:
            while not self._pipeline.display_queue.empty():
                self._latest_vitals = self._pipeline.display_queue.get_nowait()
        except queue.Empty:
            pass

        if self._latest_vitals is not None:
            q = self._latest_vitals.quality
            calib_done = self._pipeline.calibration_done
            calib_prog = self._pipeline.calibration_progress

            # Subject tab always gets data
            self._subject_tab.update_display(
                breath_bpm=self._latest_vitals.breath_bpm,
                heart_bpm=self._latest_vitals.heart_bpm,
                breath_waveform=self._latest_vitals.breath_waveform,
                quality=q,
                calibration_done=calib_done,
                calibration_progress=calib_prog,
            )

            # Research tab always gets data
            self._trend_tick_counter += 1
            trend_sample = (self._trend_tick_counter % 20 == 0)  # ~1 sample/sec
            self._research_tab.update_display(
                breath_bpm=self._latest_vitals.breath_bpm,
                heart_bpm=self._latest_vitals.heart_bpm,
                breath_waveform=self._latest_vitals.breath_waveform,
                heart_waveform=self._latest_vitals.heart_waveform,
                quality=q,
                sample_for_trend=trend_sample,
            )

            # Waveform accumulation (every display tick ~30fps for export)
            if self._latest_vitals.breath_waveform.size > 0:
                self._breath_waveform_accum.append(
                    self._latest_vitals.breath_waveform.copy()
                )
            if self._latest_vitals.heart_waveform.size > 0:
                self._heart_waveform_accum.append(
                    self._latest_vitals.heart_waveform.copy()
                )

            # CSV row accumulation (once per second)
            if trend_sample and q is not None:
                phase_range_raw = float(
                    np.max(self._latest_vitals.breath_waveform)
                    - np.min(self._latest_vitals.breath_waveform)
                ) if len(self._latest_vitals.breath_waveform) > 0 else 0.0

                sqi = 0
                br = q.get("breath_ratio", 0)
                pr = q.get("phase_range", 0)
                if pr >= 0.01 and br >= 0.15:
                    sqi = 3
                elif pr >= 0.005 and br >= 0.05:
                    sqi = 2
                elif pr > 0 or br > 0:
                    sqi = 1

                # BPM & SQI history for HDF5/EDF export
                elapsed_t = time.time() - self._start_time if self._start_time > 0 else 0
                self._bpm_history.append((
                    elapsed_t,
                    self._latest_vitals.breath_bpm,
                    self._latest_vitals.heart_bpm,
                ))
                self._sqi_history.append({
                    "phase_range": q.get("phase_range", 0.0),
                    "breath_ratio": q.get("breath_ratio", 0.0),
                    "sqi_level": sqi,
                })

                self._csv_rows.append({
                    "Timestamp": datetime.now().isoformat(),
                    "FrameIndex": self._latest_vitals.frame_index,
                    "RangeBin": self._pipeline.best_range_bin if self._pipeline.best_range_bin is not None else 0,
                    "RawPhase": round(phase_range_raw, 6),
                    "BreathBPM": self._latest_vitals.breath_bpm,
                    "HeartBPM": self._latest_vitals.heart_bpm,
                    "PhaseRange": round(q.get("phase_range", 0), 6),
                    "BreathRatio": round(q.get("breath_ratio", 0), 4),
                    "HeartProminence": round(q.get("heart_prominence", 0), 4),
                    "ApneaFlag": 1 if q.get("apnea_state") else 0,
                    "SQI_Level": sqi,
                })

            # Status bar — simplified
            if q and not q.get("valid") and calib_done:
                self._status_label.setText(tr("status_signal_error"))
                self._status_label.setStyleSheet("color: #e74c3c;")
            elif self._running:
                self._status_label.setText(tr("status_monitoring"))
                self._status_label.setStyleSheet("color: #27ae60;")

        if self._start_time > 0:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                self._frame_rate_label.setText(tr("frame_rate", f"{self._frame_count / elapsed:.1f}"))
            m, s = divmod(int(elapsed), 60)
            self._elapsed_label.setText(tr("elapsed", f"{m:02d}:{s:02d}"))

    def closeEvent(self, event) -> None:
        self._on_stop()
        event.accept()
