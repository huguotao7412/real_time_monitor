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
    quality: dict[str, Any] | None = None
    # quality 字段:
    #   valid: bool              — 信号是否有效
    #   reason: str              — 无效/异常原因 ("apnea", "low_signal", ...)
    #   phase_range: float       — 当前窗口峰峰值
    #   apnea_state: bool        — 是否处于屏息/微弱信号状态
    #   harmonic_overlap: bool   — 心率峰值是否落在呼吸谐波衰减区内
    #   heart_prominence: float  — 心率谱峰显著性 (0.1~1.0)
    #   breath_ratio: float      — 呼吸频带能量占比
