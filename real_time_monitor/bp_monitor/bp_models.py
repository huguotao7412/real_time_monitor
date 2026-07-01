"""Blood pressure monitoring data types."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BPResult:
    """Single blood pressure measurement result.

    Produced by BPPipeline every ~5.12 seconds (1024 frames at 200Hz).
    """

    timestamp: float          # time.time() when result was produced
    frame_index: int          # cumulative frame count
    sbp: float                # systolic blood pressure (mmHg), NaN if invalid
    dbp: float                # diastolic blood pressure (mmHg), NaN if invalid
    raw_sbp: float            # 网络原始预测 SBP（extract_bp offset=0，含 MATLAB 常数 -10）
    raw_dbp: float            # 网络原始预测 DBP（extract_bp offset=0，含 MATLAB 常数 -20）
    bp_waveform: np.ndarray   # reconstructed BP waveform, 256 points at 50Hz, in mmHg
    target_distance_m: float  # target range in meters
    quality: dict = field(default_factory=dict)
    # quality fields:
    #   n_peaks: int         — number of systolic peaks detected
    #   n_valleys: int       — number of diastolic valleys detected
    #   confidence: float    — 0.0-1.0, higher with more peaks/valleys
    #   noise_floor: float   — CFAR noise floor estimate
