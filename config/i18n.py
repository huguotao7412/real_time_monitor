"""Academic-grade i18n system with hot-switch support (no restart required).

Usage:
    from config.i18n import I18n, tr

    # In widget __init__:
    I18n.instance().language_changed.connect(self.update_ui_texts)

    # At point of use:
    label.setText(tr("status_standby"))
"""

from PyQt6.QtCore import QObject, pyqtSignal

# ── TRANSLATION DICTIONARY ──────────────────────────────────
# Keys are semantic identifiers; values are localized strings.
# Add new languages by adding a top-level key (e.g. "ja", "fr").

TRANSLATIONS: dict[str, dict[str, str]] = {
    "zh": {
        # Window / tabs
        "window_title": "RS6240 毫米波雷达生命体征实时监测系统",
        "app_title": "RS6240 生命体征监测系统",
        "tab_subject": "监测",
        "tab_research": "研究",
        "tab_bp": "血压",
        "bp_sbp_label": "SBP",
        "bp_dbp_label": "DBP",
        "bp_dist_label": "距离",
        "bp_conf_label": "置信度",
        "btn_mode_hr": "♥ 心率模式",
        "btn_mode_bp": "💓 血压模式",
        "status_switching": "切换模式中...",

        # Buttons
        "btn_start_capture": "▶ 开始采集",
        "btn_start_replay": "▶ 开始回放",
        "btn_stop": "■ 停止",
        "btn_save": "保存数据",
        "btn_select_file": "选择文件",

        # File / status labels
        "file_not_selected": "未选择文件",
        "status_standby": "● 待机",
        "status_starting": "● 启动中...",
        "status_playing": "● 回放中",
        "status_done": "● 回放完毕",
        "status_stopped": "● 已停止",
        "status_signal_error": "● 信号异常",
        "status_monitoring": "● 监测中",
        "status_running": "● 运行中",
        "status_no_data": "● 无数据",
        "status_dsp_error": "● DSP异常",
        "frame_rate": "帧率: {} fps",
        "frame_rate_na": "帧率: --",
        "elapsed": "运行: {}",
        "elapsed_na": "运行: 00:00",

        # File dialog
        "dialog_select_bin": "选择 .bin 文件",
        "dialog_save_dir": "选择保存目录",
        "dialog_save_done": "保存完成",
        "dialog_save_done_msg": "数据已保存至 {}",
        "dialog_error": "错误",
        "dialog_no_valid_file": "请先选择有效的 .bin 文件",
        "dialog_cannot_open": "无法打开 {}",

        # Serial status
        "serial_init_failed": "启动失败: {}",
        "serial_not_found": "未找到雷达: ctrl={} data={}",
        "serial_connect_failed": "连接失败 {}/{}",
        "serial_capturing": "采集中 ({}/{})",

        # Subject tab
        "breath_rate_unit": "呼吸频率 次/分钟",
        "heart_rate_unit": "心率 次/分钟",
        "error_overlay_text": "设备重新校准中，请保持平稳...",

        # Research tab
        "resp_wave_title": "呼吸波形 (0.1-0.6 Hz)",
        "heart_wave_title": "心率波形 (0.8-2.5 Hz)",
        "label_breath": "呼吸:",
        "label_heart": "心率:",
        "debug_collapsed": "▼ 调试面板",
        "debug_expanded": "▲ 调试面板",

        # Status messages (status_mapper)
        "msg_apnea": "呼吸较浅，请放松",
        "msg_signal_extreme_weak": "信号极弱，设备重新校准中，请保持平稳...",
        "msg_signal_severe_degraded": "信号质量严重下降，请调整坐姿，正对雷达",
        "msg_no_micro_motion": "未检测到微动，请确认在雷达覆盖范围内 (0.5m-1.5m)",
        "msg_signal_weak": "信号较弱，请调整坐姿，正对雷达",
        "msg_body_movement": "检测到体动干扰，请保持放松",

        # SQI indicator
        "sqi_excellent": "信号: 优",
        "sqi_good": "信号: 中",
        "sqi_poor": "信号: 差",
        "sqi_none": "信号: --",

        # Trend panel
        "trend_title": "历史趋势",
        "trend_5min": "5 分钟",
        "trend_15min": "15 分钟",
        "trend_30min": "30 分钟",
        "trend_axis_bpm": "BPM",
        "trend_axis_time": "时间",

        # Calibration overlay
        "calibration_text": "正在校准，请保持静止...",

        # Controls widget
        "group_tcp": "TCP 连接",
        "group_serial": "串口设置",
        "label_ctrl_port": "控制口:",
        "label_data_port": "数据口:",
        "group_status": "状态",

        # Language menu
        "menu_language": "Language / 语言",
        "lang_zh": "中文",
        "lang_en": "English",

        # Export dialog (Phase 4)
        "export_title": "导出数据",
        "export_format_csv": "Export to CSV (Basic)",
        "export_format_hdf5": "Export to HDF5 (Research)",
        "export_format_edf": "Export to EDF (Clinical)",
    },

    "en": {
        # Window / tabs
        "window_title": "RS6240 mmWave Radar Vital Signs Real-Time Monitor",
        "app_title": "RS6240 Vital Signs Monitor",
        "tab_subject": "Subject",
        "tab_research": "Research",
        "tab_bp": "BP",
        "bp_sbp_label": "SBP",
        "bp_dbp_label": "DBP",
        "bp_dist_label": "Distance",
        "bp_conf_label": "Confidence",
        "btn_mode_hr": "♥ Heart Rate",
        "btn_mode_bp": "💓 Blood Pressure",
        "status_switching": "Switching mode...",

        # Buttons
        "btn_start_capture": "▶ Start Capture",
        "btn_start_replay": "▶ Start Replay",
        "btn_stop": "■ Stop",
        "btn_save": "Save Data",
        "btn_select_file": "Select File",

        # File / status labels
        "file_not_selected": "No file selected",
        "status_standby": "● Standby",
        "status_starting": "● Starting...",
        "status_playing": "● Playing",
        "status_done": "● Replay Complete",
        "status_stopped": "● Stopped",
        "status_signal_error": "● Signal Error",
        "status_monitoring": "● Monitoring",
        "status_running": "● Running",
        "status_no_data": "● No Data",
        "status_dsp_error": "● DSP Error",
        "frame_rate": "FPS: {}",
        "frame_rate_na": "FPS: --",
        "elapsed": "Elapsed: {}",
        "elapsed_na": "Elapsed: 00:00",

        # File dialog
        "dialog_select_bin": "Select .bin file",
        "dialog_save_dir": "Select save directory",
        "dialog_save_done": "Save Complete",
        "dialog_save_done_msg": "Data saved to {}",
        "dialog_error": "Error",
        "dialog_no_valid_file": "Please select a valid .bin file",
        "dialog_cannot_open": "Cannot open {}",

        # Serial status
        "serial_init_failed": "Init failed: {}",
        "serial_not_found": "Radar not found: ctrl={} data={}",
        "serial_connect_failed": "Connection failed {}/{}",
        "serial_capturing": "Capturing ({}/{})",

        # Subject tab
        "breath_rate_unit": "Breath Rate bpm",
        "heart_rate_unit": "Heart Rate bpm",
        "error_overlay_text": "Calibrating signal, please hold still...",

        # Research tab
        "resp_wave_title": "Respiratory Waveform (0.1-0.6 Hz)",
        "heart_wave_title": "Heartbeat Waveform (0.8-2.5 Hz)",
        "label_breath": "Breath:",
        "label_heart": "Heart:",
        "debug_collapsed": "▼ Debug Panel",
        "debug_expanded": "▲ Debug Panel",

        # Status messages (status_mapper)
        "msg_apnea": "Shallow breathing, please relax",
        "msg_signal_extreme_weak": "Signal extremely weak, recalibrating, please hold still...",
        "msg_signal_severe_degraded": "Signal severely degraded, adjust posture to face radar",
        "msg_no_micro_motion": "No micro-motion detected, confirm within radar range (0.5m-1.5m)",
        "msg_signal_weak": "Weak signal, adjust posture to face radar",
        "msg_body_movement": "Body movement detected, please stay relaxed",

        # SQI indicator
        "sqi_excellent": "Signal: Excellent",
        "sqi_good": "Signal: Good",
        "sqi_poor": "Signal: Poor",
        "sqi_none": "Signal: --",

        # Trend panel
        "trend_title": "Trend",
        "trend_5min": "5 min",
        "trend_15min": "15 min",
        "trend_30min": "30 min",
        "trend_axis_bpm": "BPM",
        "trend_axis_time": "Time",

        # Calibration overlay
        "calibration_text": "Calibrating, please hold still...",

        # Controls widget
        "group_tcp": "TCP Connection",
        "group_serial": "Serial Ports",
        "label_ctrl_port": "Control:",
        "label_data_port": "Data:",
        "group_status": "Status",

        # Language menu
        "menu_language": "Language / 语言",
        "lang_zh": "中文",
        "lang_en": "English",

        # Export dialog (Phase 4)
        "export_title": "Export Data",
        "export_format_csv": "Export to CSV (Basic)",
        "export_format_hdf5": "Export to HDF5 (Research)",
        "export_format_edf": "Export to EDF (Clinical)",
    },
}


class I18n(QObject):
    """Singleton i18n manager with hot-switch signal."""

    language_changed = pyqtSignal(str)

    _instance: "I18n | None" = None
    _lang: str = "zh"

    def __init__(self):
        super().__init__()

    @classmethod
    def instance(cls) -> "I18n":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def tr(cls, key: str, *fmt_args) -> str:
        """Return the localized string for *key* in the current language.

        If *fmt_args* are provided, the string is formatted with them.
        Falls back to the key itself if translation is missing.
        """
        text = TRANSLATIONS.get(cls._lang, {}).get(key)
        if text is None:
            # Fallback to English, then to key itself
            text = TRANSLATIONS.get("en", {}).get(key, key)
        if fmt_args:
            text = text.format(*fmt_args)
        return text

    @classmethod
    def current_language(cls) -> str:
        return cls._lang

    @classmethod
    def set_language(cls, lang: str) -> None:
        if lang not in TRANSLATIONS:
            return
        cls._lang = lang
        if cls._instance:
            cls._instance.language_changed.emit(lang)


# Shortcut for convenience
def tr(key: str, *fmt_args) -> str:
    return I18n.tr(key, *fmt_args)
