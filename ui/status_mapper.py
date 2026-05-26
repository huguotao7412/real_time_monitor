"""Map DSP quality dict conditions to human-readable Chinese status messages."""

from collections import deque
from typing import Any

import numpy as np


def map_status(quality: dict[str, Any] | None) -> tuple[str, str]:
    """Return (message: str, level: str) where level is 'normal'|'warning'|'error'.

    Priority order: apnea > phase_range > breath_ratio. First match wins.
    """
    if quality is None:
        return ("● 待机", "normal")

    if quality.get("apnea_state"):
        return ("呼吸较浅，请放松", "warning")

    phase_range = quality.get("phase_range", 0.0)
    breath_ratio = quality.get("breath_ratio", 0.0)

    if phase_range < 0.005:
        return ("未检测到微动，请确认在雷达覆盖范围内 (0.5m-1.5m)", "warning")
    if breath_ratio < 0.03:
        return ("信号较弱，请调整坐姿，正对雷达", "warning")

    return ("● 监测中", "normal")


class BodyMovementDetector:
    """Detect sudden body movement from phase_range history."""

    def __init__(self, window_size: int = 60, sigma_threshold: float = 3.0):
        self._history: deque[float] = deque(maxlen=window_size)
        self._sigma_threshold = sigma_threshold

    def feed(self, phase_range: float) -> bool:
        """Return True if current phase_range is a body movement spike."""
        self._history.append(phase_range)
        if len(self._history) < 10:
            return False
        mean = float(np.mean(self._history))
        std = float(np.std(self._history))
        if std < 1e-9:
            return False
        return phase_range > mean + self._sigma_threshold * std


def map_status_with_movement(
    quality: dict[str, Any] | None,
    movement_detected: bool,
) -> tuple[str, str]:
    """Like map_status but with body movement detection layered in."""
    if movement_detected:
        return ("检测到体动干扰，请保持放松", "warning")
    return map_status(quality)
