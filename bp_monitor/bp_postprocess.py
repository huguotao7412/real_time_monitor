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
      - SBP = mean(sorted(peaks)[1:-1]) - 10  (trim extremes, empirical correction)
      - DBP = mean(sorted(valleys)[1:-1]) - 20

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

    if len(peaks) < 3 or len(valleys) < 3:
        return np.nan, np.nan, info

    # Trim extremes, compute means, apply empirical corrections
    pks_sorted = np.sort(wf[peaks])
    if len(pks_sorted) >= 5:
        sbp = float(np.mean(pks_sorted[1:-1]) - 10.0)
    else:
        sbp = float(np.mean(pks_sorted) - 10.0)

    vlys_sorted = np.sort(wf[valleys])
    if len(vlys_sorted) >= 5:
        dbp = float(np.mean(vlys_sorted[1:-1]) - 20.0)
    else:
        dbp = float(np.mean(vlys_sorted) - 20.0)

    info["confidence"] = min(1.0, min(len(peaks), len(valleys)) / 10.0)

    return sbp, dbp, info
