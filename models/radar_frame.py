from dataclasses import dataclass
import numpy as np


@dataclass
class FrameHeader:
    """从二进制帧头解析出的元信息"""
    data_type: int       # 0=FFT, 1=PointCloud
    frame_type: int      # 1=1DFFT, 2=2DFFT
    rx_ant_num: int
    tx_ant_num: int
    start_freq_mhz: int
    range_fft_num: int
    doppler_fft_num: int
    max_range_cm: int
    range_resol_mm: int
    max_velocity_cm_s: int
    velocity_resol_mm_s: int
    interval_period_us: int = 0   # 仅实时 UART 包中有
    frame_period_ms: int = 0      # 仅实时 UART 包中有


@dataclass
class RadarFrame:
    """解包后的单帧雷达数据"""
    timestamp: float
    frame_index: int
    header: FrameHeader
    data_cube: np.ndarray  # shape: [range_bins, doppler_bins, rx_antennas]
