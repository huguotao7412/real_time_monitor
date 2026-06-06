import numpy as np


def extract_phase(data_cube: np.ndarray, range_bin_idx: int) -> np.ndarray:
    """
    从指定 Range Bin 提取复数相位序列。

    选择幅度最大的天线而非直接平均复数 IQ，避免多天线间
    相位差接近 180° 时的相消干涉导致振幅归零。

    Args:
        data_cube: shape [range_bins, doppler_bins, rx_antennas]
        range_bin_idx: 目标 Range Bin 索引

    Returns:
        相位值 (弧度)
    """
    complex_vals = data_cube[range_bin_idx, 0, :]  # [rx]
    best_idx = np.argmax(np.abs(complex_vals))
    best_complex = complex_vals[best_idx]
    return np.arctan2(best_complex.imag, best_complex.real)


def unwrap_phase(phase_seq: np.ndarray) -> np.ndarray:
    """相位解缠: 消除 2π 跳变"""
    return np.unwrap(phase_seq)
