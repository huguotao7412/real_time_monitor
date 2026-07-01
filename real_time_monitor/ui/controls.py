from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QComboBox,
    QLabel, QVBoxLayout, QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class ControlsWidget(QWidget):
    """底部控制栏: 开始/停止/保存 按钮 + 连接配置 + 状态指示"""

    start_clicked = pyqtSignal(str, str)  # params depend on mode
    stop_clicked = pyqtSignal()
    save_clicked = pyqtSignal()

    def __init__(self, mode: str = "serial", parent=None):
        super().__init__(parent)
        self._mode = mode

        layout = QHBoxLayout(self)

        # 连接配置区
        if mode == "tcp":
            cfg_group = QGroupBox("TCP 连接")
            cfg_layout = QVBoxLayout(cfg_group)
            self.tcp_info_label = QLabel("127.0.0.1:9000")
            self.tcp_info_label.setFont(QFont("Consolas", 11))
            self.tcp_info_label.setStyleSheet("color: #3498db;")
            cfg_layout.addWidget(self.tcp_info_label)
            layout.addWidget(cfg_group)
            self.control_combo = None
            self.data_combo = None
        else:
            port_group = QGroupBox("串口设置")
            port_layout = QVBoxLayout(port_group)
            port_row = QHBoxLayout()
            port_row.addWidget(QLabel("控制口:"))
            self.control_combo = QComboBox()
            self.control_combo.setMinimumWidth(100)
            port_row.addWidget(self.control_combo)
            port_layout.addLayout(port_row)

            port_row2 = QHBoxLayout()
            port_row2.addWidget(QLabel("数据口:"))
            self.data_combo = QComboBox()
            self.data_combo.setMinimumWidth(100)
            port_row2.addWidget(self.data_combo)
            port_layout.addLayout(port_row2)
            layout.addWidget(port_group)

        # 操作按钮
        btn_layout = QVBoxLayout()
        self.start_btn = QPushButton("开始采集")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2ecc71; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self.start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #c0392b; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self.stop_btn.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("保存数据")
        self.save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)

        # 状态指示
        status_group = QGroupBox("状态")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("● 待机")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setStyleSheet("color: #f39c12;")
        status_layout.addWidget(self.status_label)

        self.frame_rate_label = QLabel("帧率: --")
        status_layout.addWidget(self.frame_rate_label)

        self.elapsed_label = QLabel("运行: 00:00")
        status_layout.addWidget(self.elapsed_label)
        layout.addWidget(status_group)

    def _on_start(self) -> None:
        if self._mode == "tcp":
            self.start_clicked.emit("", "")
        else:
            ctrl = self.control_combo.currentText() if self.control_combo else ""
            data = self.data_combo.currentText() if self.data_combo else ""
            if ctrl and data:
                self.start_clicked.emit(ctrl, data)

    def _on_stop(self) -> None:
        self.stop_clicked.emit()

    def _on_save(self) -> None:
        self.save_clicked.emit()

    def set_tcp_info(self, host: str, port: int) -> None:
        self.tcp_info_label.setText(f"{host}:{port}")

    def set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if self._mode == "serial":
            if self.control_combo:
                self.control_combo.setEnabled(not running)
            if self.data_combo:
                self.data_combo.setEnabled(not running)
        if running:
            self.status_label.setText("● 运行中")
            self.status_label.setStyleSheet("color: #27ae60;")
        else:
            self.status_label.setText("● 已停止")
            self.status_label.setStyleSheet("color: #f39c12;")

    def set_ports(self, ports: list[str]) -> None:
        current_ctrl = self.control_combo.currentText()
        current_data = self.data_combo.currentText()
        self.control_combo.clear()
        self.data_combo.clear()
        self.control_combo.addItems(ports)
        self.data_combo.addItems(ports)
        if current_ctrl in ports:
            self.control_combo.setCurrentText(current_ctrl)
        if current_data in ports:
            self.data_combo.setCurrentText(current_data)

    def set_frame_rate(self, fps: float) -> None:
        self.frame_rate_label.setText(f"帧率: {fps:.1f} fps")

    def set_elapsed(self, seconds: int) -> None:
        m, s = divmod(seconds, 60)
        self.elapsed_label.setText(f"运行: {m:02d}:{s:02d}")

    def set_warning(self, level: str) -> None:
        """level: 'normal', 'warning' (yellow), 'error' (red)"""
        if level == "warning":
            self.status_label.setText("● 无数据")
            self.status_label.setStyleSheet("color: #f39c12;")
        elif level == "error":
            self.status_label.setText("● DSP异常")
            self.status_label.setStyleSheet("color: #e74c3c;")
        else:
            self.status_label.setText("● 运行中")
            self.status_label.setStyleSheet("color: #27ae60;")
