"""Multi-user calibration profile manager with JSON persistence.

Singleton pattern (matching config/i18n.py I18n style).
Survives mode hot-switches because it lives outside algorithm instances.
"""

import json
import os
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal


STORAGE_DIR = Path(__file__).parent.parent / "data"
STORAGE_PATH = STORAGE_DIR / "calibration_profiles.json"

EMPTY_STATE: dict = {
    "profiles": [],
    "active_profile": None,
    "active_record_index": None,
}


class CalibrationMgr(QObject):
    """Singleton manager for multi-user BP calibration profiles.

    Usage:
        mgr = CalibrationMgr.instance()
        mgr.add_profile("张三")
        mgr.add_record("张三", 120.0, 80.0, 105.0, 72.0)
        offset_sbp = mgr.current_sbp_offset  # 15.0
    """

    profile_changed = pyqtSignal()

    _instance: "CalibrationMgr | None" = None

    def __init__(self):
        super().__init__()
        self._data: dict = {}
        self._load()

    # ── singleton ─────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "CalibrationMgr":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── current offset (read-only, DEPRECATED: prefer get_calibration_params) ──

    @property
    def current_sbp_offset(self) -> float:
        """DEPRECATED: 返回当前生效的 SBP Bias 值。新代码请使用 get_calibration_params()。"""
        _, bias_s, _, _ = self.get_calibration_params()
        return bias_s

    @property
    def current_dbp_offset(self) -> float:
        """DEPRECATED: 返回当前生效的 DBP Bias 值。新代码请使用 get_calibration_params()。"""
        _, _, _, bias_d = self.get_calibration_params()
        return bias_d

    # ── profile query ─────────────────────────────────────────────

    @property
    def profiles(self) -> list[dict]:
        return self._data.get("profiles", [])

    @property
    def active_profile_name(self) -> str | None:
        return self._data.get("active_profile")

    @property
    def active_record_index(self) -> int | None:
        return self._data.get("active_record_index")

    # ── profile CRUD ──────────────────────────────────────────────

    def add_profile(self, user_name: str) -> None:
        """Create a new empty profile. Automatically selects it as active."""
        user_name = user_name.strip()
        if not user_name:
            return
        # Check for duplicate
        for p in self.profiles:
            if p["user_name"] == user_name:
                return
        self.profiles.append({"user_name": user_name, "records": []})
        self._data["active_profile"] = user_name
        self._data["active_record_index"] = None
        self._save()
        self.profile_changed.emit()

    def delete_profile(self, user_name: str) -> None:
        """Remove a profile. If active, switch to the first remaining profile."""
        before_len = len(self.profiles)
        self._data["profiles"] = [
            p for p in self.profiles if p["user_name"] != user_name
        ]
        if len(self.profiles) < before_len:
            if self._data["active_profile"] == user_name:
                if self.profiles:
                    self._data["active_profile"] = self.profiles[0]["user_name"]
                    self._data["active_record_index"] = None
                else:
                    self._data["active_profile"] = None
                    self._data["active_record_index"] = None
            self._save()
            self.profile_changed.emit()

    def select_profile(self, user_name: str | None) -> None:
        """Switch active profile. Pass None to deactivate all profiles."""
        self._data["active_profile"] = user_name
        # 自动选择最新的记录
        if user_name is not None:
            profile = self._find_profile(user_name)
            if profile and profile["records"]:
                # [修改点] 将 0 改为 len(...) - 1，确保默认应用最新一次的血管基线校准
                self._data["active_record_index"] = len(profile["records"]) - 1
            else:
                self._data["active_record_index"] = None
        else:
            self._data["active_record_index"] = None
        self._save()
        self.profile_changed.emit()

    # ── record CRUD ───────────────────────────────────────────────

    def add_record(
        self,
        user_name: str,
        true_sbp: float,
        true_dbp: float,
        measured_sbp: float,
        measured_dbp: float,
        raw_sbp: float,
        raw_dbp: float,
    ) -> None:
        """Add a calibration record for a user.

        measured_sbp/dbp: 管线最终校准输出（用于 UI 历史展示）
        raw_sbp/dbp:      网络原始预测 extract_bp(offset=0)（用于拟合）
        """
        profile = self._find_profile(user_name)
        if profile is None:
            return

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "true_sbp": round(true_sbp, 1),
            "true_dbp": round(true_dbp, 1),
            "measured_sbp": round(measured_sbp, 1),
            "measured_dbp": round(measured_dbp, 1),
            "offset_sbp": round(true_sbp - raw_sbp, 1),
            "offset_dbp": round(true_dbp - raw_dbp, 1),
            "raw_measured_sbp": round(raw_sbp, 1),
            "raw_measured_dbp": round(raw_dbp, 1),
        }
        profile["records"].append(record)
        self._data["active_profile"] = user_name
        self._data["active_record_index"] = len(profile["records"]) - 1
        self._save()
        self.profile_changed.emit()

    def select_record(self, user_name: str, index: int) -> None:
        """Activate a specific record by index."""
        profile = self._find_profile(user_name)
        if profile is None:
            return
        if 0 <= index < len(profile["records"]):
            self._data["active_profile"] = user_name
            self._data["active_record_index"] = index
            self._save()
            self.profile_changed.emit()

    def delete_record(self, user_name: str, index: int) -> None:
        """Delete a record. Adjusts active_record_index as needed."""
        profile = self._find_profile(user_name)
        if profile is None:
            return
        if 0 <= index < len(profile["records"]):
            del profile["records"][index]
            if not profile["records"]:
                self._data["active_record_index"] = None
            elif (
                self._data["active_profile"] == user_name
                and self._data["active_record_index"] == index
            ):
                # Deleted record was active → move to previous
                self._data["active_record_index"] = max(0, index - 1)
            elif (
                self._data["active_profile"] == user_name
                and self._data["active_record_index"] is not None
            ):
                if self._data["active_record_index"] > index:
                    self._data["active_record_index"] -= 1
            self._save()
            self.profile_changed.emit()

    # ── Scale+Bias calibration params ──────────────────────────────

    def get_calibration_params(
        self, user_name: str | None = None, window_size: int = 5
    ) -> tuple[float, float, float, float]:
        """基于最近 window_size 条记录计算线性校准参数。

        返回: (sbp_scale, sbp_bias, dbp_scale, dbp_bias)

        三级策略：
          - 0 条记录 → (1.0, 0.0, 1.0, 0.0)  完全信任网络
          - 1 条记录 → (1.0, bias, 1.0, bias)  退化 Offset
          - ≥2 条记录 → 滑动窗口最小二乘 + Scale clamp [0.5, 2.0] + Bias 重算
        """
        name = user_name or self._data.get("active_profile")
        if name is None:
            return (1.0, 0.0, 1.0, 0.0)

        profile = self._find_profile(name)
        if profile is None:
            return (1.0, 0.0, 1.0, 0.0)

        records = profile.get("records", [])
        if not records:
            return (1.0, 0.0, 1.0, 0.0)

        # 单点校准：退化为 Offset 模式
        if len(records) == 1:
            r = records[0]
            raw_s = r.get("raw_measured_sbp")
            if raw_s is None:
                raw_s = r["measured_sbp"] - r.get("offset_sbp", 0.0)
            raw_d = r.get("raw_measured_dbp")
            if raw_d is None:
                raw_d = r["measured_dbp"] - r.get("offset_dbp", 0.0)
            bias_s = r["true_sbp"] - raw_s
            bias_d = r["true_dbp"] - raw_d
            return (1.0, bias_s, 1.0, bias_d)

        # 多点回归：滑动窗口 + 最小二乘
        recent = records[-window_size:]

        # 提取 true 和 raw 数组（向后兼容反推）
        true_s = np.array([r["true_sbp"] for r in recent], dtype=np.float64)
        true_d = np.array([r["true_dbp"] for r in recent], dtype=np.float64)

        raw_s_list = []
        raw_d_list = []
        for r in recent:
            rs = r.get("raw_measured_sbp")
            if rs is None:
                rs = r["measured_sbp"] - r.get("offset_sbp", 0.0)
            raw_s_list.append(rs)
            rd = r.get("raw_measured_dbp")
            if rd is None:
                rd = r["measured_dbp"] - r.get("offset_dbp", 0.0)
            raw_d_list.append(rd)
        raw_s = np.array(raw_s_list, dtype=np.float64)
        raw_d = np.array(raw_d_list, dtype=np.float64)

        def _fit(raw_vals, true_vals):
            """最小二乘拟合 y = scale*x + bias，含安全钳制。"""
            if np.std(raw_vals) < 1e-9:
                # 所有 raw 值几乎相同 → 矩阵奇异，fallback 为 Offset
                return 1.0, float(np.mean(true_vals) - np.mean(raw_vals))
            A = np.column_stack([raw_vals, np.ones_like(raw_vals)])
            scale, bias = np.linalg.lstsq(A, true_vals, rcond=None)[0]
            # 安全钳制
            scale = float(np.clip(scale, 0.5, 2.0))
            # 用钳制后的 scale 重算 bias，保持配对
            bias = float(np.mean(true_vals) - scale * np.mean(raw_vals))
            return scale, bias

        s_scale, s_bias = _fit(raw_s, true_s)
        d_scale, d_bias = _fit(raw_d, true_d)
        return (s_scale, s_bias, d_scale, d_bias)

    # ── internal ──────────────────────────────────────────────────

    def _find_profile(self, user_name: str) -> dict | None:
        for p in self.profiles:
            if p["user_name"] == user_name:
                return p
        return None

    # DEPRECATED: 新代码请使用 get_calibration_params()
    def _compute_current_offset(self) -> tuple[float, float]:
        """Derive current offset from active profile + record index.

        DEPRECATED: 仅保留向后兼容。新代码应使用 get_calibration_params()。
        """
        active = self._data.get("active_profile")
        idx = self._data.get("active_record_index")
        if active is None or idx is None:
            return (0.0, 0.0)
        profile = self._find_profile(active)
        if profile is None:
            return (0.0, 0.0)
        records = profile.get("records", [])
        if not records or not (0 <= idx < len(records)):
            return (0.0, 0.0)
        r = records[idx]
        return (r.get("offset_sbp", 0.0), r.get("offset_dbp", 0.0))

    def _load(self) -> None:
        """Load from JSON. Gracefully degrade if file missing or corrupt."""
        if not STORAGE_PATH.exists():
            self._data = copy.deepcopy(EMPTY_STATE)  # [修改] 使用深拷贝防止内存污染全局状态
            return
        try:
            with open(STORAGE_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # Validate structure
            if not isinstance(loaded, dict):
                raise ValueError("Root is not a dict")
            self._data = {
                "profiles": loaded.get("profiles", []),
                "active_profile": loaded.get("active_profile"),
                "active_record_index": loaded.get("active_record_index"),
            }
        except (json.JSONDecodeError, ValueError, OSError) as e:
            print(f"[CalibrationMgr] Failed to load {STORAGE_PATH}: {e}")
            self._data = copy.deepcopy(EMPTY_STATE)  # [修改] 使用深拷贝

    def _save(self) -> None:
        """Write current state to JSON file atomically."""
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = STORAGE_PATH.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, STORAGE_PATH)  # atomic on same fs
        except OSError as e:
            print(f"[CalibrationMgr] Failed to save: {e}")
