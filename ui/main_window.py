"""Main window — thin shell hosting SubjectTab, BPTab, and ResearchTab via QTabWidget.

Mode-specific logic (HR vs BP) is delegated to MonitorMode strategy objects.
"""

import time
import threading
import copy

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QLabel, QPushButton,
    QTabWidget, QInputDialog,
)
from PyQt6.QtCore import QTimer, Qt,QMetaObject,Q_ARG, pyqtSignal
from PyQt6.QtGui import QFont

from config.protocol import UI_REFRESH_MS
from config.i18n import tr, I18n
from io_engine.uart_parser import UartParser
from io_engine.serial_manager import SerialManager
from io_engine.radar_mgr import RadarMgr
from io_engine.data_exporter import export_csv, export_hdf5, export_edf, export_bp_csv, export_bp_hdf5

from ui.subject_tab import SubjectTab
from ui.research_tab import ResearchTab
from ui.monitor_mode import MonitorMode, HRMode, BPMode


class MainWindow(QMainWindow):
    _start_research_signal = pyqtSignal()
    def __init__(self, bp_mode: bool = False):
        super().__init__()
        self.setWindowTitle(tr("window_title"))
        self.resize(1200, 800)

        # Mode object (Strategy pattern)
        self._current_mode: MonitorMode = BPMode() if bp_mode else HRMode()

        # Shared state
        self._start_time: float = 0.0
        self._frame_count: int = 0
        self._running: bool = False

        # Serial mode (always)
        self._serial_mgr: SerialManager | None = None
        self._radar_mgr: RadarMgr | None = None
        self._uart_parser: UartParser | None = None
        self._io_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._serial_status: str = ""
        self._serial_error: bool = False
        self._serial_mgr = SerialManager()
        self._radar_mgr = RadarMgr(self._serial_mgr)
        self._uart_parser = UartParser(
            bins_per_frame=self._current_mode.uart_bins)

        self._setup_ui()
        self._setup_timers()
        self._start_research_signal.connect(self._research_tab.start)

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

        label = tr("btn_start_capture")
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

        # Mode toggle button
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

        I18n.instance().language_changed.connect(self.update_ui_texts)

    def update_ui_texts(self, _lang: str = "") -> None:
        self.setWindowTitle(tr("window_title"))
        self._title_label.setText(tr("app_title"))
        self._tabs.setTabText(0, tr("tab_subject"))
        self._tabs.setTabText(1, tr("tab_bp"))
        self._tabs.setTabText(2, tr("tab_research"))
        self._start_btn.setText(tr("btn_start_capture"))
        self._stop_btn.setText(tr("btn_stop"))
        self._save_btn.setText(tr("btn_save"))
        self._lang_menu.setTitle(tr("menu_language"))
        self._zh_action.setText(tr("lang_zh"))
        self._en_action.setText(tr("lang_en"))
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

    # === Slots ===

    def _on_start(self) -> None:
        self._start_serial()

    def _start_serial(self) -> None:
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._mode_btn.setEnabled(False)
        self._status_label.setText(tr("status_starting"))
        self._status_label.setStyleSheet("color: #f39c12;")
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
        self._start_research_signal.emit()
        self._serial_error = False
        self._serial_status = tr("serial_capturing", ctrl_port, data_port)
        print("[Serial Init] Starting I/O loop...")
        self._io_thread = threading.Thread(target=self._serial_io_loop, daemon=True)
        self._io_thread.start()
        print("[Serial Init] Done!")

    def _serial_io_loop(self) -> None:
        try:
            while self._running and not self._stop_event.is_set():
                try:
                    # 这里的 read_data 依赖 serial 的 timeout 参数（不可为 None，否则会永久阻塞）
                    raw = self._serial_mgr.read_data(4096)
                    if not raw:
                        continue  # 超时唤醒后，如果没有数据，会重新判断 while 循环条件
                    frames = self._uart_parser.feed(raw)
                    for fft_data in frames:
                        self._frame_count += 1
                        frame = self._current_mode.build_frame(fft_data, self._frame_count)
                        self._current_mode.feed_frame(frame)
                except Exception as e:
                    print(f"[Serial I/O] {e}")
                    time.sleep(0.5)
        finally:
            # 循环结束（不论是正常停止还是异常退出），由 IO 线程自行安全关闭底层串口
            if self._serial_mgr:
                print("[Serial I/O] Thread exiting, safely closing serial ports...")
                self._serial_mgr.close()

    def _on_toggle_mode(self) -> None:
        """Hot-switch between HR and BP monitoring modes (serial only)."""
        was_running = self._running

        # 1. Stop current I/O safely
        if was_running:
            if self._radar_mgr:
                self._radar_mgr.shutdown()
            self._running = False
            if self._stop_event:
                self._stop_event.set()
            if self._io_thread:
                self._io_thread.join(timeout=1.0) # 等待 IO 线程自身干净地结束并关闭串口

        # 2. Stop pipeline + shutdown radar
        self._current_mode.stop()

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
        if self._radar_mgr:
            self._radar_mgr.shutdown()

        self._running = False

        if self._stop_event:
            self._stop_event.set()  # 发出停止信号

        if self._io_thread:
            # timeout 应该稍微大于你 config 中设置的 DATA_TIMEOUT_SEC
            # 假设 DATA_TIMEOUT_SEC 是 0.5，这里给 1.0 秒以保证线程能响应超时并退出
            self._io_thread.join(timeout=1.0)

        self._current_mode.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._mode_btn.setEnabled(True)
        self._status_label.setText(tr("status_stopped"))
        self._status_label.setStyleSheet("color: #f39c12;")

    def _on_save(self) -> None:
        is_bp = isinstance(self._current_mode, BPMode)
        formats = [
            tr("export_format_csv"),
            tr("export_format_hdf5"),
        ]
        if not is_bp:
            formats.append(tr("export_format_edf"))

        choice, ok = QInputDialog.getItem(
            self, tr("export_title"), "Format:", formats, 0, False,
        )
        if not ok or not choice:
            return

        path = QFileDialog.getExistingDirectory(self, tr("dialog_save_dir"))
        if not path:
            return




        self._save_btn.setEnabled(False)
        self._save_btn.setText(tr("btn_saving") if tr("btn_saving") != "btn_saving" else "Saving...")

        # 3. 定义后台异步导出任务
        def export_task():
            try:
                data = self._current_mode.get_export_data()

                if choice == tr("export_format_csv"):
                    if is_bp:
                        export_bp_csv(path, data.get("csv_rows", []))
                    else:
                        vitals = data.get("latest_vitals")
                        breath = vitals.breath_waveform if vitals else np.array([])
                        heart = vitals.heart_waveform if vitals else np.array([])
                        export_csv(path, data.get("csv_rows", []), breath, heart)

                elif choice == tr("export_format_hdf5"):
                    if is_bp:
                        metadata = {
                            "device": "RS6240",
                            "session_duration_s": time.time() - self._start_time if self._start_time > 0 else 0,
                        }
                        export_bp_hdf5(path, data.get("csv_rows", []), metadata)
                    else:
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

                # 4. 导出成功，通过单次定时器安全地切回主线程进行弹窗
                QTimer.singleShot(0, lambda: self._on_save_success(path))

            except Exception as e:
                # 导出失败，同样切回主线程弹窗报错
                error_msg = str(e)
                QTimer.singleShot(0, lambda: self._on_save_error(error_msg))

        # 启动后台守护线程执行保存任务
        threading.Thread(target=export_task, daemon=True).start()

    # === 新增：线程安全的 UI 恢复回调 ===

    def _on_save_success(self, path: str) -> None:
        """导出成功后在主线程恢复按钮并弹窗"""
        self._save_btn.setEnabled(True)
        self._save_btn.setText(tr("btn_save"))
        QMessageBox.information(
            self, tr("dialog_save_done"), tr("dialog_save_done_msg", path)
        )

    def _on_save_error(self, error_msg: str) -> None:
        """导出失败后在主线程恢复按钮并报错"""
        self._save_btn.setEnabled(True)
        self._save_btn.setText(tr("btn_save"))
        QMessageBox.critical(self, tr("dialog_error"), error_msg)

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
