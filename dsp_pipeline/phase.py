import numpy as np


def extract_phase(data_cube: np.ndarray, range_bin_idx: int) -> np.ndarray:
    """
    从指定 Range Bin 提取复数相位序列。

    Args:
        data_cube: shape [range_bins, doppler_bins, rx_antennas]
        range_bin_idx: 目标 Range Bin 索引

    Returns:
        相位值 (弧度)
    """
    n_doppler = data_cube.shape[1]
    if n_doppler > 0:
        complex_vals = data_cube[range_bin_idx, 0, :]  # [rx]
    else:
        complex_vals = data_cube[range_bin_idx, 0, :]

    avg_complex = np.mean(complex_vals)
    return np.arctan2(avg_complex.imag, avg_complex.real)


def unwrap_phase(phase_seq: np.ndarray) -> np.ndarray:
    """相位解缠: 消除 2π 跳变"""
    return np.unwrap(phase_seq)
