"""Main window — thin shell hosting SubjectTab, BPTab, and ResearchTab via QTabWidget.

Mode-specific logic (HR vs BP) is delegated to MonitorMode strategy objects.
"""

import os
import glob
import time
import threading
from datetime import datetime

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QLabel, QPushButton,
    QTabWidget, QInputDialog,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

from config.protocol import UI_REFRESH_MS
from config.i18n import tr, I18n
from io_engine.bin_reader import BinFileReader
from io_engine.uart_parser import UartParser
from io_engine.serial_manager import SerialManager
from io_engine.radar_mgr import RadarMgr
from io_engine.data_exporter import export_csv, export_hdf5, export_edf

from ui.subject_tab import SubjectTab
from ui.research_tab import ResearchTab
from ui.monitor_mode import MonitorMode, HRMode, BPMode


class MainWindow(QMainWindow):
    def __init__(self, mode: str = "replay", replay_file: str | None = None,
                 bp_replay: bool = False):
        super().__init__()
        self.setWindowTitle(tr("window_title"))
        self.resize(1200, 800)

        self._mode_type = mode  # "serial" or "replay"
        self._replay_file = replay_file

        # Mode object (Strategy pattern) — the ONLY mode reference
        self._current_mode: MonitorMode = BPMode() if bp_replay else HRMode()

        # Replay
        self._bin_reader: BinFileReader | None = None
        self._replay_timer: QTimer | None = None

        # Shared state
        self._start_time: float = 0.0
        self._frame_count: int = 0
        self._running: bool = False

        # Serial mode
        self._serial_mgr: SerialManager | None = None
        self._radar_mgr: RadarMgr | None = None
        self._uart_parser: UartParser | None = None
        self._io_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._serial_status: str = ""
        self._serial_error: bool = False
        if mode == "serial":
            self._serial_mgr = SerialManager()
            self._radar_mgr = RadarMgr(self._serial_mgr)
            self._uart_parser = UartParser(
                bins_per_frame=self._current_mode.uart_bins)

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

        if self._mode_type == "replay":
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

        label = tr("btn_start_capture") if self._mode_type == "serial" else tr("btn_start_replay")
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
        if self._mode_type == "serial":
            self._mode_btn = QPushButton(
                tr("btn_mode_bp") if isinstance(self._current_mode, HRMode)
                else tr("btn_mode_hr")
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

        # Apply initial tab visibility
        self._update_tab_visibility()

        # Auto-select latest file
        if self._mode_type == "replay" and not self._replay_file:
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
        self._tabs.setTabText(1, tr("tab_bp"))
        self._tabs.setTabText(2, tr("tab_research"))
        label = tr("btn_start_capture") if self._mode_type == "serial" else tr("btn_start_replay")
        self._start_btn.setText(label)
        self._stop_btn.setText(tr("btn_stop"))
        self._save_btn.setText(tr("btn_save"))
        self._lang_menu.setTitle(tr("menu_language"))
        self._zh_action.setText(tr("lang_zh"))
        self._en_action.setText(tr("lang_en"))
        # Update mode button text
        if hasattr(self, '_mode_btn'):
            self._mode_btn.setText(
                tr("btn_mode_hr") if isinstance(self._current_mode, BPMode)
                else tr("btn_mode_bp")
            )

    def _setup_timers(self) -> None:
        self._ui_timer = QTimer()
        self._ui_timer.timeout.connect(self._on_ui_tick)
        self._ui_timer.start(UI_REFRESH_MS)

    def _update_tab_visibility(self) -> None:
        show_subject, show_bp, show_research = self._current_mode.tab_visibility()
        self._tabs.setTabVisible(0, show_subject)
        self._tabs.setTabVisible(1, show_bp)
        self._tabs.setTabVisible(2, show_research)

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
        if self._mode_type == "serial":
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
        # Boot radar + start pipeline + start I/O on background thread
        thread = threading.Thread(target=self._serial_init_thread, daemon=True)
        thread.start()

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
                info = next((i for i in serial.tools.list_ports.comports()
                             if i.device == p), None)
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
        self._current_mode.boot_radar(self._radar_mgr)
        self._stop_event = threading.Event()
        self._uart_parser.reset()
        self._current_mode.start()
        self._current_mode.clear_data()
        self._start_time = time.time()
        self._frame_count = 0
        self._running = True
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
                    frame = self._current_mode.build_frame(fft_data, self._frame_count)
                    self._current_mode.feed_frame(frame)
            except Exception as e:
                print(f"[Serial I/O] {e}")
                time.sleep(0.5)

    def _start_replay(self) -> None:
        if not self._replay_file or not os.path.exists(self._replay_file):
            QMessageBox.critical(self, tr("dialog_error"), tr("dialog_no_valid_file"))
            return
        reader = BinFileReader(self._replay_file)
        if not reader.open():
            QMessageBox.critical(self, tr("dialog_error"),
                                 tr("dialog_cannot_open", self._replay_file))
            return
        self._bin_reader = reader
        self._current_mode.start()
        self._current_mode.clear_data()
        self._start_time = time.time()
        self._frame_count = 0
        self._running = True
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
        frame = self._current_mode.build_frame(frames[0], self._frame_count)
        self._current_mode.feed_frame(frame)

    def _on_toggle_mode(self) -> None:
        """Hot-switch between HR and BP monitoring modes (serial only)."""
        was_running = self._running

        # 1. Stop current I/O
        if was_running:
            self._running = False
            if self._stop_event:
                self._stop_event.set()
            # Close serial data port to unblock read_data()
            if self._serial_mgr and self._serial_mgr.data_serial:
                try:
                    self._serial_mgr.data_serial.close()
                except Exception:
                    pass
            if self._io_thread:
                self._io_thread.join(timeout=5)

        # 2. Stop pipeline + shutdown radar
        self._current_mode.stop()
        if self._radar_mgr:
            self._radar_mgr.shutdown()

        # 3. Swap mode
        was_bp = isinstance(self._current_mode, BPMode)
        self._current_mode = HRMode() if was_bp else BPMode()

        # 4. Rebuild UART parser for new mode
        self._uart_parser = UartParser(bins_per_frame=self._current_mode.uart_bins)

        # 5. Update UI
        self._update_tab_visibility()
        self._mode_btn.setText(
            tr("btn_mode_hr") if isinstance(self._current_mode, BPMode)
            else tr("btn_mode_bp")
        )

        # 6. Reset inactive tabs to "--"
        if was_bp:
            self._bp_tab.reset_display()
        else:
            self._subject_tab.reset_display()
            self._research_tab.reset_display()

        # 7. Restart if was running
        if was_running:
            # Re-open data port
            try:
                self._serial_mgr.open_data(
                    self._serial_mgr.data_port,
                    baudrate=self._radar_mgr.data_baudrate,
                )
            except Exception:
                pass
            self._current_mode.boot_radar(self._radar_mgr)
            self._current_mode.start()
            self._current_mode.clear_data()
            self._running = True
            self._stop_event = threading.Event()
            self._uart_parser.reset()
            self._frame_count = 0
            self._start_time = time.time()
            self._io_thread = threading.Thread(
                target=self._serial_io_loop, daemon=True)
            self._io_thread.start()
            self._status_label.setText("● Capturing")
            self._status_label.setStyleSheet("color: #27ae60;")
        else:
            self._status_label.setText(tr("status_standby"))
            self._status_label.setStyleSheet("color: #f39c12;")

    def _on_stop(self) -> None:
        self._running = False
        if self._mode_type == "serial":
            if self._radar_mgr:
                self._radar_mgr.shutdown()
            if self._stop_event:
                self._stop_event.set()
            # Close serial to unblock read_data
            if self._serial_mgr and self._serial_mgr.data_serial:
                try:
                    self._serial_mgr.data_serial.close()
                except Exception:
                    pass
            if self._io_thread:
                self._io_thread.join(timeout=5)
            if self._serial_mgr:
                self._serial_mgr.close()
        if self._replay_timer:
            self._replay_timer.stop()
            self._replay_timer = None
        self._current_mode.stop()
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
            data = self._current_mode.get_export_data()
            if choice == tr("export_format_csv"):
                vitals = data.get("latest_vitals")
                breath = vitals.breath_waveform if vitals else np.array([])
                heart = vitals.heart_waveform if vitals else np.array([])
                export_csv(path, data.get("csv_rows", []), breath, heart)
            elif choice == tr("export_format_hdf5"):
                breath_hist = (np.array(data["breath_waveform_accum"])
                               if data.get("breath_waveform_accum") else np.array([]))
                heart_hist = (np.array(data["heart_waveform_accum"])
                              if data.get("heart_waveform_accum") else np.array([]))
                metadata = {
                    "device": "RS6240",
                    "fs": 20,
                    "session_duration_s": time.time() - self._start_time if self._start_time > 0 else 0,
                }
                export_hdf5(path, breath_hist, heart_hist,
                            data.get("bpm_history", []),
                            data.get("sqi_history", []), metadata)
            elif choice == tr("export_format_edf"):
                vitals = data.get("latest_vitals")
                breath = vitals.breath_waveform if vitals else np.array([])
                heart = vitals.heart_waveform if vitals else np.array([])
                export_edf(path, breath, heart, fs=20.0)
            QMessageBox.information(
                self, tr("dialog_save_done"), tr("dialog_save_done_msg", path)
            )
        except ImportError as e:
            QMessageBox.critical(self, tr("dialog_error"), str(e))
        except Exception as e:
            QMessageBox.critical(self, tr("dialog_error"), str(e))

    # === UI Timer ===

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

        if not self._running:
            return

        # Delegate display polling to current mode
        self._current_mode.poll_and_update(
            self._subject_tab, self._bp_tab, self._research_tab,
            self._status_label, self._elapsed_label, self._frame_rate_label,
            self._start_time, self._frame_count,
        )

        # Shared: frame rate + elapsed
        if self._start_time > 0:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                self._frame_rate_label.setText(
                    tr("frame_rate", f"{self._frame_count / elapsed:.1f}"))
            m, s_div = divmod(int(elapsed), 60)
            self._elapsed_label.setText(tr("elapsed", f"{m:02d}:{s_div:02d}"))

    def closeEvent(self, event) -> None:
        self._on_stop()
        event.accept()
