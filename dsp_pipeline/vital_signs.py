from dataclasses import dataclass
import numpy as np


from typing import Any


@dataclass
class VitalSigns:
    timestamp: float
    frame_index: int
    breath_waveform: np.ndarray  # 呼吸时域波形 (200 点)
    breath_bpm: float
    heart_bpm: float
    heart_waveform: np.ndarray   # 预留, MVP 为空数组
    quality: dict[str, Any] | None = None  # 信号质量指标
