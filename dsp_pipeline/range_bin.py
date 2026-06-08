"""Range Bin 选择 — 基于 MATLAB DataProcess.m 的自适应 CFAR 阈值检测"""

import numpy as np
from scipy.signal import find_peaks


def _interpolate_peak(power: np.ndarray, idx: int) -> float:
    """二次插值求亚区间峰值偏移量，返回精确的小数 bin 索引."""
    alpha = power[idx - 1] if idx > 0 else power[idx]
    beta = power[idx]
    gamma = power[idx + 1] if idx < len(power) - 1 else power[idx]
    denom = alpha - 2 * beta + gamma
    if abs(denom) < 1e-15:
        p = 0.0
    else:
        p = 0.5 * (alpha - gamma) / denom
    return float(np.clip(idx + p, 0, len(power) - 1))


def find_best_range_bin(
    data_cube: np.ndarray,
    fs: float = 20.0,
    breath_band: tuple[float, float] = (0.1, 0.8),
) -> float:
    """
    自适应 CFAR 目标检测 (移植自 MATLAB DataProcess.findTargetBin)

    策略:
      1. 计算每个 Range Bin 的平均功率
      2. 取底部 70% 的低功率点估计噪声参数 (mean, std)
      3. 动态阈值 = noise_mean + 3 * noise_std
      4. 找超过阈值的峰值, 选最近的一个 (离人体最近的目标)
      5. 二次插值获得亚区间精度
    """
    n_range, n_doppler, n_rx = data_cube.shape
    if n_doppler > 0:
        static_slice = data_cube[:, 0, :]
    else:
        static_slice = data_cube[:, 0:1, :].squeeze(1)

    # 1. 聚合: 平均功率 (跨 RX 通道), 跳过 bin 0
    abs_data = np.abs(static_slice[1:, :])  # [n_range-1, n_rx]
    power_profile = np.mean(abs_data, axis=1)  # [n_range-1]

    # 2. 噪声估计: 底部 70% 的点
    sorted_power = np.sort(power_profile)
    noise_samples = sorted_power[: int(0.7 * len(sorted_power))]
    noise_mean = np.mean(noise_samples)
    noise_std = np.std(noise_samples)

    # 3. 自适应阈值 (alpha=3.0 = 99.7% 置信度, MATLAB 原版)
    threshold = noise_mean + 3.0 * noise_std

    # 4. 寻找超过阈值的峰值
    peaks, props = find_peaks(power_profile, height=threshold)

    if len(peaks) > 0:
        # 选峰值最高的那个 (对应最强反射 → 最近的人体)
        best_local = peaks[np.argmax(props["peak_heights"])]
        return _interpolate_peak(power_profile, best_local) + 1  # +1 补偿跳过 bin 0

    # 5. 找不到峰值 → 降级: 仅在中间距离找功率最大的 bin
    search_start = max(2, int(n_range * 0.04))
    search_end = min(n_range - 3, int(n_range * 0.94))
    if search_end <= search_start:
        return float(np.argmax(power_profile) + 1)
    mid_power = power_profile[search_start - 1 : search_end - 1]
    best_mid = int(np.argmax(mid_power))
    return _interpolate_peak(mid_power, best_mid) + search_start
