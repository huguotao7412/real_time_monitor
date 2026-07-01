"""Map DSP quality dict conditions to human-readable status messages (i18n-aware)."""

from collections import deque
from typing import Any

import numpy as np

from config.i18n import tr

from config.protocol import BREATH_RATIO_MIN, PHASE_RANGE_MIN_NORMAL


def map_status(quality: dict[str, Any] | None) -> tuple[str, str]:
    """Return (message: str, level: str) where level is 'normal'|'warning'|'error'.

    Priority order: apnea > phase_range > breath_ratio. First match wins.
    Error level is reserved for conditions severe enough to trigger waveform
    visual degradation (amplitude decay + color desaturation).
    """
    if quality is None:
        return (tr("status_standby"), "normal")

    if quality.get("apnea_state"):
        return (tr("msg_apnea"), "warning")

    phase_range = quality.get("phase_range", 0.0)
    breath_ratio = quality.get("breath_ratio", 0.0)

    # Error thresholds: signal too weak to produce meaningful waveform
    if phase_range < 0.002:
        return (tr("msg_signal_extreme_weak"), "error")
    if breath_ratio < 0.01:
        return (tr("msg_signal_severe_degraded"), "error")

    # Warning thresholds: degraded but still usable
    if phase_range < PHASE_RANGE_MIN_NORMAL:
        return (tr("msg_no_micro_motion"), "warning")
    if breath_ratio < BREATH_RATIO_MIN:
        return (tr("msg_signal_weak"), "warning")

    return (tr("status_monitoring"), "normal")


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
        return (tr("msg_body_movement"), "warning")
    return map_status(quality)
