"""SBP/DBP extraction from reconstructed BP waveform.

Ported from MATLAB main.m peak/valley detection logic.
"""

import numpy as np
from scipy.signal import find_peaks


def extract_bp(
    waveform_mmhg: np.ndarray, fs: float = 50.0
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

        # 提取所有收缩期峰值的中位数
    sbp = float(np.median(wf[peaks]))

        # 提取所有舒张期谷值的中位数
    dbp = float(np.median(wf[valleys]))

    info["confidence"] = min(1.0, min(len(peaks), len(valleys)) / 10.0)

    return sbp, dbp, info
