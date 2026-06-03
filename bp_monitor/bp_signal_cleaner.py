"""Phase signal cleaning — ported from MATLAB PhaseProcess.m.

Pipeline:
  1. EMD harmonic interference removal (remove_harmonic_interference_local)
  2. Wavelet denoising + baseline drift removal (RadarSignalCleaner)
"""

import numpy as np
from PyEMD import EMD
import pywt
from scipy.signal import periodogram


def clean_pulse_wave(raw_phase: np.ndarray, fs: float = 200.0) -> np.ndarray:
    """Full signal cleaning matching MATLAB RadarSignalCleaner.

    Steps:
      1. EMD decomposition -> detect breath fundamental freq -> LS remove harmonics
      2. Sym8 wavelet decomposition, zero approx coeffs, reconstruct from details

    Args:
        raw_phase: 1D unwrapped phase signal at target range bin
        fs: sampling frequency (200 Hz in BP mode)

    Returns:
        cleaned pulse wave (same length as input)
    """
    # Step 1: Harmonic removal
    signal_notch = _remove_harmonics(raw_phase, fs)

    # Step 2: Wavelet denoising (0.5× matches MATLAB)
    clean = _wavelet_denoise(signal_notch, fs)
    return 0.5 * clean


def _remove_harmonics(signal: np.ndarray, fs: float) -> np.ndarray:
    """EMD-based breath harmonic detection + least-squares removal.

    MATLAB: PhaseProcess.remove_harmonic_interference_local
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = len(signal)
    if n < 128:
        return signal.copy()

    # EMD (max 6 IMFs)
    emd = EMD(max_imf=6)
    imfs = emd.emd(signal)
    if imfs.ndim == 1:
        imfs = imfs.reshape(1, -1)

    n_imfs = min(imfs.shape[0], 6)

    # Dominant frequency + correlation per IMF
    imf_freqs = np.zeros(n_imfs)
    imf_corrs = np.zeros(n_imfs)
    for i in range(n_imfs):
        imf = imfs[i]
        freqs, psd = periodogram(imf, fs, nfft=n)
        if len(psd) > 0:
            imf_freqs[i] = freqs[np.argmax(psd)]
        corr = np.corrcoef(signal, imf)[0, 1]
        imf_corrs[i] = 0.0 if np.isnan(corr) else corr

    # Find breath-related IMF: 0.1-0.6 Hz, highest correlation
    candidates = np.where((imf_freqs >= 0.1) & (imf_freqs <= 0.6))[0]
    if len(candidates) == 0:
        return signal.copy()

    best = candidates[np.argmax(imf_corrs[candidates])]
    f_rr = imf_freqs[best]

    # Build 2nd/3rd/4th harmonic bases
    t = np.arange(n) / fs
    X = []
    for k in [2, 3, 4]:
        freq = k * f_rr
        X.append(np.cos(2 * np.pi * freq * t))
        X.append(np.sin(2 * np.pi * freq * t))
    X = np.column_stack(X)

    # Least-squares fit and subtract
    beta = np.linalg.lstsq(X, signal, rcond=None)[0]
    return signal - X @ beta


def _wavelet_denoise(signal: np.ndarray, fs: float) -> np.ndarray:
    """Wavelet denoising + baseline drift removal.

    MATLAB: PhaseProcess.RadarSignalCleaner (wavedec/wrcoef chain)
    Uses sym8 wavelet, zeroes approximation coefs, keeps top 4 detail levels.
    """
    n = len(signal)

    # Desired level based on fs (MATLAB convention)
    if 180 <= fs <= 250:
        level = 8
    elif 90 <= fs < 180:
        level = 7
    else:
        level = 6

    # Clamp to maximum useful level for this signal length + wavelet
    max_level = pywt.dwt_max_level(n, "sym8")
    level = min(level, max_level)

    coeffs = pywt.wavedec(signal, "sym8", level=level)

    # Zero approximation (baseline drift)
    coeffs[0] = np.zeros_like(coeffs[0])

    # MATLAB reconstructs D_level-3 .. D_level (coarser, lower-freq details).
    # PyWavelets coeffs: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    # MATLAB detail level m maps to PyWavelets index: level - m + 1
    # e.g. level=8: D5→coeffs[4], D6→coeffs[3], D7→coeffs[2], D8→coeffs[1]
    result = np.zeros(n)
    matlab_start = max(1, level - 3)
    for m_level in range(matlab_start, level + 1):
        py_idx = level - m_level + 1
        single = [np.zeros_like(c) for c in coeffs]
        single[0] = np.zeros_like(coeffs[0])
        single[py_idx] = coeffs[py_idx]
        result += pywt.waverec(single, "sym8")[:n]

    return result
