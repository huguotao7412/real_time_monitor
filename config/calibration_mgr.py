"""Multi-user calibration profile manager with JSON persistence.

Singleton pattern (matching config/i18n.py I18n style).
Survives mode hot-switches because it lives outside algorithm instances.
"""

import json
import os
import copy
from datetime import datetime
from pathlib import Path

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

    # ── current offset (read-only) ────────────────────────────────

    @property
    def current_sbp_offset(self) -> float:
        offset, _ = self._compute_current_offset()
        return offset

    @property
    def current_dbp_offset(self) -> float:
        _, offset = self._compute_current_offset()
        return offset

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
    ) -> None:
        """Add a calibration record for a user. Computes offset automatically.

        offset = true_value - measured_value
        """
        profile = self._find_profile(user_name)
        if profile is None:
            return

        # 1. 获取当前系统正在使用的旧偏移量
        old_offset_sbp, old_offset_dbp = self._compute_current_offset()

        # 2. 核心修复：计算新偏移量时，把旧偏移量加回来，抵消掉 measured 中包含的旧 offset
        new_offset_sbp = true_sbp - measured_sbp + old_offset_sbp
        new_offset_dbp = true_dbp - measured_dbp + old_offset_dbp

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "true_sbp": round(true_sbp, 1),
            "true_dbp": round(true_dbp, 1),
            "measured_sbp": round(measured_sbp, 1),
            "measured_dbp": round(measured_dbp, 1),
            "offset_sbp": round(new_offset_sbp, 1),
            "offset_dbp": round(new_offset_dbp, 1),
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

    # ── internal ──────────────────────────────────────────────────

    def _find_profile(self, user_name: str) -> dict | None:
        for p in self.profiles:
            if p["user_name"] == user_name:
                return p
        return None

    def _compute_current_offset(self) -> tuple[float, float]:
        """Derive current offset from active profile + record index."""
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
