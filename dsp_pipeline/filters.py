"""滤波器模块 — 基于 MATLAB Filter.m 的 SOS (二阶节) 实现

SOS 级联比单次 butter+filtfilt 更稳定，不易出现数值溢出。
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


class VitalSignFilter:
    """生命体征带通滤波器组 (MATLAB Filter.m 移植, SOS 级联)"""

    def __init__(self, fs: float = 20.0):
        self.fs = fs

        # 呼吸: 0.1-0.8 Hz, 降阶为 2（使用 sosfiltfilt 前后零相位, 相当于 4 阶实际响应）
        # 降阶能显著减轻滤波器自激振铃 (ringing) 的风险
        self.sos_breath = butter(2, [0.1, 0.8], btype="band", fs=fs, output="sos")

        # 心跳: 0.8-2.5 Hz, 3阶（比原 4 阶略温和）
        self.sos_heart = butter(3, [0.8, 2.0], btype="band", fs=fs, output="sos")

        # 综合: 0.08-2.5 Hz, 10阶 (MATLAB MUSIC 预滤波, 上限对齐心跳频带)
        self.sos_all = butter(10, [0.08, 2.0], btype="band", fs=fs, output="sos")

    def filter_breath(self, x: np.ndarray) -> np.ndarray:
        return sosfiltfilt(self.sos_breath, x)

    def filter_heart(self, x: np.ndarray) -> np.ndarray:
        return sosfiltfilt(self.sos_heart, x)

    def filter_all(self, x: np.ndarray) -> np.ndarray:
        return sosfiltfilt(self.sos_all, x)


# 保留旧接口兼容
def remove_dc(signal: np.ndarray, window: int = 200) -> np.ndarray:
    """Remove DC component via zero-phase high-pass filter.

    Uses 2nd-order Butterworth high-pass at 0.05 Hz with forward-backward
    filtering to eliminate window-edge artifacts that a simple mean subtraction
    would introduce.
    """
    from scipy.signal import filtfilt

    b, a = butter(2, 0.05 / (20.0 / 2), btype='high')
    return filtfilt(b, a, signal)


def butter_bandpass(signal, lowcut, highcut, fs=20.0, order=4):
    """单次 butter (向后兼容)"""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    from scipy.signal import filtfilt
    return filtfilt(b, a, signal)
