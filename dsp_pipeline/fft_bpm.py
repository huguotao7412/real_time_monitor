"""BPM 估计 — 混合 FFT + 时域峰值检测 (MATLAB breath_test.m + PhaseProcess.m)"""

import numpy as np


def estimate_bpm(
    signal: np.ndarray,
    fs: float,
    valid_band: tuple[float, float],
    n_fft: int = 4096,
) -> float:
    """FFT 峰值估计 (用于心率等弱周期信号)"""
    n = len(signal)
    if n < 16:
        return 0.0

    t = np.arange(n)
    poly = np.polyfit(t, signal, 3)
    detrended = signal - np.polyval(poly, t)

    windowed = detrended * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    mask = (freqs >= valid_band[0]) & (freqs <= valid_band[1])
    if not np.any(mask):
        return 0.0

    peak_idx = np.argmax(spectrum[mask])
    peak_freq = freqs[mask][peak_idx]

    delta = peak_freq * 0.05
    refine_mask = (freqs >= peak_freq - delta) & (freqs <= peak_freq + delta)
    if np.any(refine_mask):
        weights = spectrum[refine_mask] ** 2
        refined_freq = np.average(freqs[refine_mask], weights=weights + 1e-10)
    else:
        refined_freq = peak_freq

    return refined_freq * 60.0


def estimate_breath_bpm_time_domain(
    signal: np.ndarray,
    fs: float = 20.0,
    min_interval_sec: float = 2.0,
) -> float:
    """
    时域峰值检测法算呼吸 BPM (MATLAB breath_test.m 移植)

    呼吸 BPM 范围: 6-40 (对应 1.5s-10s 周期)
    最小间隔 2.0s → 最高 30 BPM (滤除心跳和噪声峰)
    """
    n = len(signal)
    if n < fs * 3:
        return 0.0

    # 去趋势
    t = np.arange(n)
    detrended = signal - np.polyval(np.polyfit(t, signal, 1), t)

    # 振幅检查: 无显著波动 → 屏息/无目标, 返回 0
    envelope = np.abs(detrended)
    # 将信号分成前后两半，比较振幅: 真人呼吸时振幅有变化
    half = n // 2
    amp_first = np.mean(envelope[:half])
    amp_second = np.mean(envelope[half:])
    max_amp = max(amp_first, amp_second)
    if max_amp < np.std(signal) * 0.5:
        return 0.0  # 信号太弱

    # 找峰值: 间隔 ≥ 2.5秒, 显著性 ≥ 信号标准差的 50%
    min_distance = int(min_interval_sec * fs)
    signal_std = np.std(detrended)
    from scipy.signal import find_peaks
    peaks, props = find_peaks(
        detrended,
        distance=min_distance,
        prominence=signal_std * 0.5,
        height=signal_std * 0.2,
    )

    if len(peaks) < 2:
        return 0.0

    # 峰值间隔 → BPM
    intervals = np.diff(peaks) / fs

    # 呼吸周期 1.5s~10s → 6~40 BPM
    valid = (intervals >= 1.5) & (intervals <= 10.0)
    if np.sum(valid) < 2:
        return 0.0

    mean_interval = np.mean(intervals[valid])
    bpm = 60.0 / mean_interval

    # 最终范围检查
    if bpm < 6 or bpm > 40:
        return 0.0

    return bpm


def kalman_smooth(measurements: list[float], q: float = 1e-4, r: float = 0.1) -> float:
    """卡尔曼平滑 — 响应更快的参数"""
    if not measurements:
        return 0.0

    # 只用最近的 15 个值 (更快响应)
    recent = measurements[-15:]

    x_est = recent[0]
    p = 1.0
    x_filtered = np.zeros(len(recent))

    for k, z in enumerate(recent):
        x_pred = x_est
        p_pred = p + q
        k_gain = p_pred / (p_pred + r)
        x_est = x_pred + k_gain * (z - x_pred)
        p = (1 - k_gain) * p_pred
        x_filtered[k] = x_est

    # 返回最近的 Kalman 估计值 (不是历史平均)
    return float(x_filtered[-1])


# 向后兼容
def fft_peak_to_bpm(signal, fs=20.0, valid_band=(0.1, 3.0)):
    return estimate_bpm(signal, fs, valid_band)
