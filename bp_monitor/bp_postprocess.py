"""SBP/DBP extraction from reconstructed BP waveform.

Ported from MATLAB main.m peak/valley detection logic.
"""

import numpy as np
from scipy.signal import find_peaks


def extract_bp(
    waveform_mmhg: np.ndarray, fs: float = 50.0,
user_calib_sbp: float = 0.0, user_calib_dbp: float = 0.0
) -> tuple[float, float, dict]:
    """Extract SBP and DBP via peak-valley detection.

    MATLAB logic:
      - prominence >= max(2.0, 0.05 * range)
      - min peak distance = round(50 * 0.5) = 25 samples
      - need >= 3 peaks AND >= 3 valleys
      - SBP = max(sort(peaks)(2:end-1)) - 10  (trim extremes, take max)
      - DBP = min(sort(valleys)(2:end-1)) - 20  (trim extremes, take min)

    Args:
        waveform_mmhg: 256-point BP waveform in mmHg (50 Hz sampling)
        fs: waveform sampling rate (50 Hz)

    Returns:
        sbp: systolic blood pressure (mmHg), NaN if insufficient peaks
        dbp: diastolic blood pressure (mmHg), NaN if insufficient valleys
        info: dict with detection metadata
    """
    wf = np.asarray(waveform_mmhg, dtype=np.float64).ravel()
    wf_range = float(np.max(wf) - np.min(wf))

    prom = max(2.0, 0.05 * wf_range)
    min_dist = int(round(fs * 0.5))  # 25 samples at 50Hz

    # Systolic peaks
    peaks, _ = find_peaks(wf, distance=min_dist, prominence=prom)

    # Diastolic valleys (peaks on inverted signal)
    valleys, _ = find_peaks(-wf, distance=min_dist, prominence=prom)

    info: dict = {
        "n_peaks": len(peaks),
        "n_valleys": len(valleys),
        "confidence": 0.0,
    }

    print(f"[extract_bp] wf_range={wf_range:.2f} mmHg  "
          f"peaks={len(peaks)}  valleys={len(valleys)}  "
          f"prom={prom:.2f}")

    if len(peaks) < 3 or len(valleys) < 3:
        print(f"[extract_bp] FAIL: need >=3 peaks and >=3 valleys")
        return np.nan, np.nan, info

        # 按照 MATLAB 逻辑：去除两端极端值，取最高/最低
    sorted_peaks = np.sort(wf[peaks])
    sorted_valleys = np.sort(wf[valleys])

        # 确保有足够数据去头掐尾
    if len(sorted_peaks) >= 3:
            raw_sbp = float(np.max(sorted_peaks[1:-1]))
    else:
            raw_sbp = float(np.max(sorted_peaks))

    if len(sorted_valleys) >= 3:
            raw_dbp = float(np.min(sorted_valleys[1:-1]))
    else:
            raw_dbp = float(np.min(sorted_valleys))

        # 应用 MATLAB 的经验补偿，并加入个人校准偏移量
        # 建议在 UI 层面设计一个输入框，让用户输入水银血压计的真实值来动态计算 calib 值
    sbp = raw_sbp - 10.0 + user_calib_sbp
    dbp = raw_dbp - 20.0 + user_calib_dbp

    info["confidence"] = min(1.0, min(len(peaks), len(valleys)) / 10.0)

    return sbp, dbp, info
