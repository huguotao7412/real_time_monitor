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
from ui.calibration_overlay import CalibrationDialog
from config.calibration_mgr import CalibrationMgr
from dsp_pipeline.strategies import (
    VMDRLSCleaner, EMDHarmonicCleaner, PassthroughCleaner,
    WPDSeparator, SOSFilterSeparator, EMDPulseCleaner,
)


class MainWindow(QMainWindow):
    _start_research_signal = pyqtSignal()
    _export_success_signal = pyqtSignal(str)
    _export_error_signal = pyqtSignal(str)
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

        # Calibration manager (survives mode hot-switches)
        self._calib_mgr = CalibrationMgr.instance()
        # Scale+Bias model: 从 CalibrationMgr 读取初始四项参数
        s_scale, s_bias, d_scale, d_bias = self._calib_mgr.get_calibration_params()
        self._calib_sbp_scale: float = s_scale
        self._calib_sbp_bias: float = s_bias
        self._calib_dbp_scale: float = d_scale
        self._calib_dbp_bias: float = d_bias

        self._setup_ui()
        self._setup_timers()
        self._start_research_signal.connect(self._research_tab.start)
        self._export_success_signal.connect(self._on_save_success)
        self._export_error_signal.connect(self._on_save_error)

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
        # Wire algorithm panel signals
        self._research_tab._algo_combo.currentIndexChanged.connect(
            self._on_research_algo_changed
        )
        self._research_tab._ab_combo.currentIndexChanged.connect(
            self._on_research_ab_changed
        )
        self._research_tab._record_btn.clicked.connect(
            self._on_research_record_toggle
        )
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

        # Set initial research tab layout for current mode
        from ui.monitor_mode import BPMode
        self._research_tab.set_mode(isinstance(self._current_mode, BPMode))

        I18n.instance().language_changed.connect(self.update_ui_texts)

        # ── Calibration signal wiring ──
        self._bp_tab.calibrate_clicked.connect(self._show_calibration_dialog)
        self._bp_tab.profile_changed.connect(self._on_profile_selected)
        self._bp_tab.profile_add_requested.connect(self._on_profile_add)
        self._calib_mgr.profile_changed.connect(self._on_calibration_offset_changed)

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

        # 10-second countdown calibration timer (100ms tick)
        self._calib_timer = QTimer()
        self._calib_timer.timeout.connect(self._on_calib_tick)
        self._calib_duration = 10.0
        self._calib_elapsed = 0.0

    def _update_tab_visibility(self) -> None:
        show_subject, show_bp, show_research = self._current_mode.tab_visibility()
        self._tabs.setTabVisible(0, show_subject)
        self._tabs.setTabVisible(1, show_bp)
        self._tabs.setTabVisible(2, show_research)
        # Update research tab labels for current mode
        if isinstance(self._current_mode, BPMode):
            self._research_tab.set_bp_mode_labels()
        else:
            self._research_tab.set_hr_mode_labels()

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
        if isinstance(self._current_mode, BPMode):
            self._current_mode.start(
                sbp_scale=self._calib_sbp_scale,
                sbp_bias=self._calib_sbp_bias,
                dbp_scale=self._calib_dbp_scale,
                dbp_bias=self._calib_dbp_bias,
            )
        else:
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
            print("[Serial I/O] Thread exiting...")

    def _on_toggle_mode(self) -> None:
        """Hot-switch between HR and BP monitoring modes (serial only)."""
        was_running = self._running

        self._mode_btn.setEnabled(False)

        # 1. 停止当前所有活动
        if was_running:
            self._on_stop()
            # 【修复点】：强制确保旧的 IO 线程彻底退出并释放串口，避免 Access Denied
            if self._io_thread and self._io_thread.is_alive():
                self._io_thread.join(timeout=2.0)
        else:
            self._current_mode.stop()

        # 2. Swap mode
        was_bp = isinstance(self._current_mode, BPMode)
        self._current_mode = HRMode() if was_bp else BPMode()

        # 3. Rebuild UART parser for new mode
        self._uart_parser = UartParser(bins_per_frame=self._current_mode.uart_bins)

        # 4. Update UI
        self._update_tab_visibility()
        if isinstance(self._current_mode, BPMode):
            self._tabs.setCurrentWidget(self._bp_tab)  # 切换到血压模式时，强制选中血压标签页
        else:
            self._tabs.setCurrentWidget(self._subject_tab)  # 切换到心率模式时，强制选中受试者标签页
        self._mode_btn.setText(
            tr("btn_mode_hr") if isinstance(self._current_mode, BPMode)
            else tr("btn_mode_bp")
        )

        # Sync research tab layout for current mode
        is_bp = isinstance(self._current_mode, BPMode)
        self._research_tab.set_mode(is_bp)

        # 5. Reset inactive tabs to "--"
        if was_bp:
            self._bp_tab.reset_display()
        else:
            self._subject_tab.reset_display()
            self._research_tab.reset_display()

        # 6. 重新启动
        if was_running:
            self._start_serial()
        else:
            self._mode_btn.setEnabled(True)  # 若没在运行，马上恢复按钮

    def _on_stop(self) -> None:
        self._pending_calib_start = False
        if hasattr(self, '_calib_timer') and self._calib_timer.isActive():
            self._calib_timer.stop()
        if hasattr(self._bp_tab, 'overlay') and self._bp_tab.overlay.isVisible():
            self._bp_tab.overlay.fade_out()

        if self._radar_mgr:
            self._radar_mgr.shutdown()

        self._running = False

        if self._stop_event:
            self._stop_event.set()  # 发出停止信号

        if self._serial_mgr:
            self._serial_mgr.close()

        if self._io_thread:
            self._io_thread.join(timeout=1.0)

        self._current_mode.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._mode_btn.setEnabled(True)
        self._status_label.setText(tr("status_stopped"))
        self._status_label.setStyleSheet("color: #f39c12;")

    # ── Research tab algorithm signal handlers ──

    def _on_research_algo_changed(self, index: int) -> None:
        """Apply primary algorithm selection to the current mode."""
        if isinstance(self._current_mode, HRMode):
            mode: HRMode = self._current_mode
            if index == 0:
                mode.set_adaptive_mode()
            elif index == 1:
                mode.set_strategies(VMDRLSCleaner(), WPDSeparator())
            elif index == 2:
                mode.set_strategies(EMDHarmonicCleaner(), WPDSeparator())
            elif index == 3:
                mode.set_strategies(PassthroughCleaner(), SOSFilterSeparator())
        elif isinstance(self._current_mode, BPMode):
            mode: BPMode = self._current_mode
            if index == 1:
                mode.set_strategies(VMDRLSCleaner())
            elif index == 2:
                mode.set_strategies(EMDHarmonicCleaner())
            elif index == 3:
                mode.set_strategies(PassthroughCleaner())
            # index 0 (Adaptive) maps to default EMDPulseCleaner for BP
            elif index == 0:
                mode.set_strategies(EMDPulseCleaner())

    def _on_research_ab_changed(self, index: int) -> None:
        """Apply A/B algorithm selection to the current mode."""
        if isinstance(self._current_mode, HRMode):
            mode: HRMode = self._current_mode
            pair_map = {
                0: (None, None),
                1: (VMDRLSCleaner(), WPDSeparator()),
                2: (EMDHarmonicCleaner(), WPDSeparator()),
                3: (PassthroughCleaner(), SOSFilterSeparator()),
            }
            cleaner, separator = pair_map.get(index, (None, None))
            mode.set_ab_strategy(cleaner, separator)
        elif isinstance(self._current_mode, BPMode):
            mode: BPMode = self._current_mode
            ab_map = {
                0: None,
                1: VMDRLSCleaner(),
                2: EMDHarmonicCleaner(),
                3: PassthroughCleaner(),
            }
            cleaner = ab_map.get(index, None)
            mode.set_ab_strategy(cleaner)

    def _on_research_record_toggle(self) -> None:
        """Start or stop benchmark recording (works for both HR and BP modes)."""
        if isinstance(self._current_mode, HRMode):
            mode: HRMode = self._current_mode
        elif isinstance(self._current_mode, BPMode):
            mode: BPMode = self._current_mode
        else:
            return
        is_recording = mode.toggle_benchmark()
        self._research_tab.set_recording_state(is_recording)

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
                self._export_success_signal.emit(path)

            except Exception as e:
                # 导出失败，同样切回主线程弹窗报错
                error_msg = str(e)
                self._export_error_signal.emit(error_msg)

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
        if getattr(self, '_pending_calib_start', False):
            # 异常打断：如果雷达启动报错，中止校准
            if getattr(self, '_serial_error', False):
                self._pending_calib_start = False
                self._bp_tab.overlay.fade_out()

            # 等待成功：雷达正在运行，且拿到了有效的（非 NaN）第一帧数据
            elif self._running and isinstance(self._current_mode, BPMode):
                mode: BPMode = self._current_mode
                if mode._latest_bp_result is not None:
                    import numpy as np
                    if not np.isnan(mode._latest_bp_result.sbp):
                        # 成功锁定！立刻关闭等待标记，进入 10 秒倒计时
                        self._pending_calib_start = False
                        self._start_calibration_phase()
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

        # Update benchmark recording timer (works for both modes)
        if isinstance(self._current_mode, (HRMode, BPMode)):
            elapsed = self._current_mode.get_benchmark_elapsed()
            self._research_tab.update_record_timer(elapsed)

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

    # ── Calibration Slots ──────────────────────────────────────────

    def _show_calibration_dialog(self) -> None:
        """Open the calibration dialog — user enters true BP values first."""
        dlg = CalibrationDialog(measured_sbp=None, measured_dbp=None, parent=self)
        dlg.calibration_submitted.connect(self._on_calibration_confirmed)
        dlg.exec()

    def _on_calibration_confirmed(
        self, true_sbp: float, true_dbp: float, save: bool,
    ) -> None:
        """接收弹窗发来的真实值，进入自动化校准流水线"""
        # 1. 暂存用户的目标真实值
        self._calib_target_sbp = true_sbp
        self._calib_target_dbp = true_dbp
        self._calib_save_flag = save

        # 2. 如果当前不在血压模式，自动切过去
        if not isinstance(self._current_mode, BPMode):
            self._on_toggle_mode()

        # 3. 开启全屏遮罩层，提前安抚用户
        self._bp_tab.overlay.set_progress(0.0)
        self._bp_tab.overlay.set_opacity(1.0)
        self._bp_tab.overlay.setVisible(True)

        # 4. 判断雷达状态，进入等待队列
        self._pending_calib_start = True  # 开启自动等待锁定状态机

        if not self._running:
            self._bp_tab.overlay.set_text("正在启动雷达并搜索目标...\n(请正对雷达保持静坐)")
            self._start_serial() # 自动帮你点“启动”按钮
        else:
            self._bp_tab.overlay.set_text("正在等待雷达数据稳定...\n(请正对雷达保持静坐)")

    def _start_calibration_phase(self) -> None:
        """雷达已锁定且有数据，正式开始 10 秒倒计时采样"""
        self._bp_tab.overlay.set_text("基线采样中，请保持平稳呼吸...")
        self._calib_elapsed = 0.0
        self._calib_timer.start(100)  # 启动 10 秒计时器

        self._status_label.setText("基线采样中...")
        self._status_label.setStyleSheet("color: #f39c12;")

    def _on_calib_tick(self) -> None:
        """Calibration countdown tick (every 100ms)."""
        self._calib_elapsed += 0.1
        fraction = self._calib_elapsed / self._calib_duration

        if fraction < 1.0:
            # Update ring progress and countdown number
            self._bp_tab.overlay.set_progress(fraction)
        else:
            # 10 seconds reached — stop timer, hide overlay, settle
            self._calib_timer.stop()
            self._bp_tab.overlay.set_progress(1.0)
            self._bp_tab.overlay.fade_out()
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        """10-second sampling complete: compute radar average, settle params, persist."""
        import numpy as np
        mode: BPMode = self._current_mode

        # 1. Retrieve both calibrated and raw BP averages over last 10 seconds
        (measured_sbp, measured_dbp, std_sbp, std_dbp,
         raw_sbp_mean, raw_dbp_mean) = mode.get_recent_bp_raw_stats(self._calib_duration)

        # 2. Guard: no valid radar lock during the 10-second window
        if measured_sbp is None or measured_dbp is None:
            self._status_label.setText("采样失败：雷达信号丢失，请保持静坐并重试")
            self._status_label.setStyleSheet("color: #e74c3c;")
            return

        # 2.5 Guard (Quality Control): 检查采样期间血压波动是否过大
        if std_sbp is not None and std_dbp is not None:
            if std_sbp > 15.0 or std_dbp > 10.0:
                print(
                    f"[Calibration] Rejected due to high variance. "
                    f"SBP std: {std_sbp:.1f}, DBP std: {std_dbp:.1f}")
                self._status_label.setText("采样失败：期间体征波动过大，请保持静坐并重新校准")
                self._status_label.setStyleSheet("color: #e74c3c;")
                return

        # 3. Auto-create default profile if user chose to save but no profile selected
        if self._calib_save_flag and self._calib_mgr.active_profile_name is None:
            self._calib_mgr.add_profile("默认用户")

        # 4. Persist calibration record with raw values (Scale+Bias model)
        if self._calib_save_flag:
            active = self._calib_mgr.active_profile_name
            if active is not None:
                self._calib_mgr.add_record(
                    user_name=active,
                    true_sbp=self._calib_target_sbp,
                    true_dbp=self._calib_target_dbp,
                    measured_sbp=measured_sbp,
                    measured_dbp=measured_dbp,
                    raw_sbp=raw_sbp_mean,
                    raw_dbp=raw_dbp_mean,
                )

        # 5. Read updated Scale+Bias from CalibrationMgr and apply to pipeline
        s_scale, s_bias, d_scale, d_bias = self._calib_mgr.get_calibration_params()
        self._calib_sbp_scale = s_scale
        self._calib_sbp_bias = s_bias
        self._calib_dbp_scale = d_scale
        self._calib_dbp_bias = d_bias

        if mode._pipeline is not None:
            mode._pipeline.set_calibration(s_scale, s_bias, d_scale, d_bias)

        # 6. Status feedback
        if self._calib_save_flag:
            self._status_label.setText(
                f"校准成功: 雷达实测均值 {measured_sbp:.0f}/{measured_dbp:.0f}"
            )
        else:
            self._status_label.setText("临时基线校准已应用")
        self._status_label.setStyleSheet("color: #27ae60;")

    def _on_profile_selected(self, name: str) -> None:
        """Handle profile selection from BPTab combo."""
        if not name:
            self._calib_mgr.select_profile(None)
        else:
            self._calib_mgr.select_profile(name)

    def _on_profile_add(self) -> None:
        """Handle 'New User...' from BPTab combo."""
        name, ok = QInputDialog.getText(
            self, tr("dlg_new_user_title"), tr("dlg_new_user_prompt"),
        )
        if ok and name.strip():
            self._calib_mgr.add_profile(name.strip())

    def _on_calibration_offset_changed(self) -> None:
        """Re-read Scale+Bias from CalibrationMgr and apply to running pipeline."""
        s_scale, s_bias, d_scale, d_bias = self._calib_mgr.get_calibration_params()
        self._calib_sbp_scale = s_scale
        self._calib_sbp_bias = s_bias
        self._calib_dbp_scale = d_scale
        self._calib_dbp_bias = d_bias

        if isinstance(self._current_mode, BPMode):
            mode: BPMode = self._current_mode
            if mode._pipeline is not None:
                mode._pipeline.set_calibration(s_scale, s_bias, d_scale, d_bias)

        # Refresh BPTab profile combo
        self._bp_tab._refresh_profile_combo()

    def closeEvent(self, event) -> None:
        self._on_stop()
        event.accept()
