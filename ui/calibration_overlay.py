"""Ring progress overlay shown during 10-second calibration phase,
and CalibrationDialog for user-driven BP baseline calibration."""

import math
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QDialog, QFrame,
    QGroupBox, QGridLayout, QScrollArea,
    QListWidget, QListWidgetItem, QAbstractItemView,
)
from PyQt6.QtCore import (
    Qt, QRectF, QPropertyAnimation, QEasingCurve, pyqtProperty, pyqtSignal,
)
from PyQt6.QtGui import QPainter, QPen, QColor, QFont

from config.i18n import tr, I18n
from config.calibration_mgr import CalibrationMgr


class CalibrationOverlay(QWidget):
    """Semi-transparent overlay with ring progress and countdown text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0
        self._countdown = 10
        self._opacity = 1.0

        self.setStyleSheet("background-color: rgba(30, 30, 40, 220);")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._text = QLabel(tr("calibration_text"))
        self._text.setFont(QFont("Segoe UI", 14))
        self._text.setStyleSheet("color: #bdc3c7; background: transparent;")
        self._text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._text)

        self._countdown_label = QLabel("10")
        self._countdown_label.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
        self._countdown_label.setStyleSheet("color: #3498db; background: transparent;")
        self._countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._countdown_label)

        self.setVisible(False)

        I18n.instance().language_changed.connect(self.update_ui_texts)

    def update_ui_texts(self, _lang: str = "") -> None:
        self._text.setText(tr("calibration_text"))

    def set_text(self, text: str) -> None:
        self._text.setText(text)

    def set_progress(self, fraction: float) -> None:
        """Update ring progress 0.0-1.0 and derived countdown."""
        self._progress = fraction
        self._countdown = max(0, int(10 * (1.0 - fraction)))
        self._countdown_label.setText(str(self._countdown))
        self.update()

    def get_opacity(self) -> float:
        return self._opacity

    def set_opacity(self, val: float) -> None:
        self._opacity = val
        self.update()

    opacity = pyqtProperty(float, get_opacity, set_opacity)

    def fade_out(self) -> None:
        self._anim = QPropertyAnimation(self, b"opacity")
        self._anim.setDuration(500)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(lambda: self.setVisible(False))
        self._anim.start()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self._opacity)

        cx = self.width() // 2
        cy = self.height() // 2 - 50
        radius = 60
        pen_width = 6

        # Background ring
        painter.setPen(QPen(QColor(60, 60, 70), pen_width))
        painter.drawArc(QRectF(cx - radius, cy - radius, radius * 2, radius * 2),
                        90 * 16, -360 * 16)

        # Progress ring
        painter.setPen(QPen(QColor(52, 152, 219), pen_width))
        painter.drawArc(QRectF(cx - radius, cy - radius, radius * 2, radius * 2),
                        90 * 16, -int(360 * self._progress) * 16)

        painter.end()


class CalibrationDialog(QDialog):
    """Modal dialog for BP baseline calibration with user profile management.

    Users enter reference SBP/DBP from a sphygmomanometer. The dialog
    displays the current 5-second average reading, manages calibration
    history records, and emits calibration_submitted on confirm.
    """

    calibration_submitted = pyqtSignal(float, float, bool)
    # true_sbp, true_dbp, save_to_profile

    def __init__(
        self,
        measured_sbp: float | None,
        measured_dbp: float | None,
        parent=None,
    ):
        super().__init__(parent)
        self._measured_sbp = measured_sbp
        self._measured_dbp = measured_dbp
        self._calib_mgr = CalibrationMgr.instance()
        self._setup_ui()
        self._set_button_enabled_state()
        self._refresh_history()
        self.setWindowTitle(tr("dlg_calib_title"))
        self.setMinimumWidth(480)
        self.setModal(True)

        # Listen for profile changes to refresh UI
        self._calib_mgr.profile_changed.connect(self._on_profile_changed)

    # ── internal state ──────────────────────────────────────────

    def _has_valid_measurements(self) -> bool:
        if self._measured_sbp is None or self._measured_dbp is None:
            return False
        if math.isnan(self._measured_sbp) or math.isnan(self._measured_dbp):
            return False
        return True

    def _inputs_valid(self) -> bool:
        sbp = self._sbp_spin.value()
        dbp = self._dbp_spin.value()
        return (sbp - dbp) >= 15

    def _set_button_enabled_state(self) -> None:
        """Enable confirm buttons only when both measurements exist AND inputs valid."""
        if not hasattr(self, '_btn_save'):
            return
        enabled = self._inputs_valid()

        self._btn_save.setEnabled(enabled)
        self._btn_temp.setEnabled(enabled)

        # 显示/隐藏验证错误提示
        if not self._inputs_valid():
            self._validation_error.setText(tr("msg_sbp_gt_dbp"))
            self._validation_error.setVisible(True)
        else:
            self._validation_error.setVisible(False)

    # ── UI setup ────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        # -- current user label --
        self._user_label = QLabel()
        self._user_label.setStyleSheet("color: #95a5a6; font-size: 10pt;")
        layout.addWidget(self._user_label)
        self._refresh_user_label()

        # -- measured values display (read-only) --
        measured_group = QGroupBox(tr("lbl_measured_bp"))
        measured_grid = QGridLayout(measured_group)
        self._meas_sbp_label = QLabel(
            f"{self._measured_sbp:.0f}" if self._measured_sbp is not None else "--"
        )
        self._meas_sbp_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self._meas_sbp_label.setStyleSheet("color: #e74c3c;")
        self._meas_dbp_label = QLabel(
            f"{self._measured_dbp:.0f}" if self._measured_dbp is not None else "--"
        )
        self._meas_dbp_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self._meas_dbp_label.setStyleSheet("color: #3498db;")
        measured_grid.addWidget(QLabel("SBP:"), 0, 0)
        measured_grid.addWidget(self._meas_sbp_label, 0, 1)
        measured_grid.addWidget(QLabel("mmHg"), 0, 2)
        measured_grid.addWidget(QLabel("DBP:"), 1, 0)
        measured_grid.addWidget(self._meas_dbp_label, 1, 1)
        measured_grid.addWidget(QLabel("mmHg"), 1, 2)
        layout.addWidget(measured_group)

        # -- reference input group --
        input_group = QGroupBox("参考血压 (水银血压计实测值)")
        input_grid = QGridLayout(input_group)

        input_grid.addWidget(QLabel(tr("lbl_true_sbp")), 0, 0)
        self._sbp_spin = QSpinBox()
        self._sbp_spin.setRange(90, 200)
        self._sbp_spin.setValue(120)
        self._sbp_spin.setSuffix(" mmHg")
        self._sbp_spin.valueChanged.connect(self._set_button_enabled_state)
        input_grid.addWidget(self._sbp_spin, 0, 1)

        input_grid.addWidget(QLabel(tr("lbl_true_dbp")), 1, 0)
        self._dbp_spin = QSpinBox()
        self._dbp_spin.setRange(50, 130)
        self._dbp_spin.setValue(80)
        self._dbp_spin.setSuffix(" mmHg")
        self._dbp_spin.valueChanged.connect(self._set_button_enabled_state)
        input_grid.addWidget(self._dbp_spin, 1, 1)

        layout.addWidget(input_group)

        # -- validation error label --
        self._validation_error = QLabel()
        self._validation_error.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self._validation_error.setVisible(False)
        layout.addWidget(self._validation_error)

        # -- history section (collapsible) --
        self._history_group = QGroupBox(tr("lbl_history"))
        self._history_group.setCheckable(True)
        self._history_group.setChecked(False)
        self._history_group.toggled.connect(self._refresh_history)
        history_layout = QVBoxLayout(self._history_group)

        self._history_list = QListWidget()
        self._history_list.setMaximumHeight(150)
        self._history_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._history_list.setStyleSheet(
            "QListWidget { background: #2b2b36; border: 1px solid #3a3a4a; "
            "border-radius: 4px; padding: 4px; color: #bdc3c7; font-size: 9pt; }"
            "QListWidget::item:selected { background: #34495e; border-radius: 3px; }"
        )
        history_layout.addWidget(self._history_list)

        history_btn_row = QHBoxLayout()
        self._btn_delete_record = QPushButton(tr("btn_delete_record"))
        self._btn_delete_record.clicked.connect(self._on_delete_selected)
        history_btn_row.addWidget(self._btn_delete_record)
        history_btn_row.addStretch()
        self._btn_apply_record = QPushButton(tr("btn_apply_record"))
        self._btn_apply_record.clicked.connect(self._on_apply_selected)
        history_btn_row.addWidget(self._btn_apply_record)
        history_layout.addLayout(history_btn_row)

        layout.addWidget(self._history_group)

        # -- separator --
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #3a3a4a;")
        layout.addWidget(sep)

        # -- confirm button row --
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_save = QPushButton("确认并开始基线采样")
        self._btn_save.clicked.connect(lambda: self._on_confirm(save=True))
        btn_row.addWidget(self._btn_save)

        self._btn_temp = QPushButton(tr("btn_calib_only"))
        self._btn_temp.clicked.connect(lambda: self._on_confirm(save=False))
        btn_row.addWidget(self._btn_temp)
        layout.addLayout(btn_row)

    # ── slots ───────────────────────────────────────────────────

    def _on_confirm(self, save: bool) -> None:
        sbp = float(self._sbp_spin.value())
        dbp = float(self._dbp_spin.value())
        self.calibration_submitted.emit(sbp, dbp, save)
        self.accept()

    def _on_delete_selected(self) -> None:
        """Delete the selected history record."""
        selected_items = self._history_list.selectedItems()
        if not selected_items:
            return  # 没有选中任何行则不操作

        item = selected_items[0]
        idx = item.data(Qt.ItemDataRole.UserRole)

        if idx is not None:
            active = self._calib_mgr.active_profile_name
            if active is not None:
                self._calib_mgr.delete_record(active, idx)

    def _on_apply_selected(self) -> None:
        """Apply the selected history record as active."""
        selected_items = self._history_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        idx = item.data(Qt.ItemDataRole.UserRole)

        if idx is not None:
            active = self._calib_mgr.active_profile_name
            if active is not None:
                self._calib_mgr.select_record(active, idx)

    def _on_profile_changed(self) -> None:
        self._refresh_user_label()
        self._refresh_history()

    # ── UI refresh helpers ──────────────────────────────────────

    def _refresh_user_label(self) -> None:
        active = self._calib_mgr.active_profile_name
        if active:
            self._user_label.setText(f"{tr('lbl_current_user')}：{active}")
        else:
            self._user_label.setText(f"{tr('lbl_current_user')}：--")

    def _refresh_history(self) -> None:
        """Rebuild history list for the active profile."""
        if not hasattr(self, '_history_list'):
            return

        self._history_list.clear()

        active = self._calib_mgr.active_profile_name
        if active is None:
            item = QListWidgetItem(tr("lbl_no_records"))
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._history_list.addItem(item)
            return

        profile = self._calib_mgr._find_profile(active)
        records = profile["records"] if profile else []
        if not records:
            item = QListWidgetItem(tr("lbl_no_records"))
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._history_list.addItem(item)
            return

        active_idx = self._calib_mgr.active_record_index
        for i, r in enumerate(records):
            marker = "★ [生效中] " if i == active_idx else "    "
            text = (
                f"{marker}{r['timestamp']}  "
                f"参考: {r['true_sbp']}/{r['true_dbp']}  "
                f"测量: {r['measured_sbp']}/{r['measured_dbp']}"
            )
            item = QListWidgetItem(text)

            # [修复 Bug 1] 将这条记录的真实索引隐藏绑定到 item，实现精准操作
            item.setData(Qt.ItemDataRole.UserRole, i)

            # 视觉优化：高亮当前生效的校准记录
            if i == active_idx:
                item.setForeground(QColor("#f1c40f"))
                font = item.font()
                font.setBold(True)
                item.setFont(font)

            self._history_list.addItem(item)

    # ── public helpers ──────────────────────────────────────────

    def update_measured_values(self, sbp: float | None, dbp: float | None) -> None:
        """Update the displayed measured values (called when readings change)."""
        self._measured_sbp = sbp
        self._measured_dbp = dbp
        self._meas_sbp_label.setText(
            f"{sbp:.0f}" if sbp is not None else "--"
        )
        self._meas_dbp_label.setText(
            f"{dbp:.0f}" if dbp is not None else "--"
        )
        self._set_button_enabled_state()
