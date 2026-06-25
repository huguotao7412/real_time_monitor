"""EMD-based harmonic interference cancellation — port of PhaseProcess.remove_harmonic_interference_algo

Decomposes signal into IMFs, identifies the breathing IMF (dominant freq in 0.1-0.6 Hz),
builds sin/cos basis for f_RR and its first 3 harmonics, then subtracts via least squares.
"""

import numpy as np
from scipy.signal import welch
from config.protocol import EMD_RESP_CORR_RATIO


def emd_harmonic_clean(
    signal: np.ndarray,
    fs: float,
    max_imf: int = 6,
    resp_band: tuple[float, float] = (0.1, 0.6),
    harmonics: list[int] | None = None,
) -> np.ndarray:
    """Remove respiratory harmonics from signal using EMD + least squares.

    Args:
        signal: 1D real-valued signal (SOS pre-filtered displacement).
        fs: Sampling rate in Hz.
        max_imf: Maximum number of IMFs to extract.
        resp_band: Frequency range for identifying breathing IMF (Hz).
        harmonics: Harmonic orders to cancel (default: [2, 3, 4]).

    Returns:
        Cleaned signal, same shape as input. On failure, returns original signal.
    """
    if harmonics is None:
        harmonics = [2, 3, 4]

    n = len(signal)
    if n < 128:
        return signal.copy()

    sig_col = signal.ravel()

    # 1. EMD decomposition
    try:
        from PyEMD import EMD
        emd = EMD()
        imfs = emd.emd(sig_col, max_imf=max_imf)
    except Exception:
        return signal.copy()

    if imfs.shape[0] == 0:
        return signal.copy()

    n_imfs = imfs.shape[0]

    # 2. Compute dominant frequency and correlation for each IMF
    imf_freqs = np.zeros(n_imfs)
    imf_corrs = np.zeros(n_imfs)

    for i in range(n_imfs):
        imf = imfs[i]
        try:
            nperseg_val = min(256, len(imf))
            freqs, psd = welch(imf, fs, nperseg=nperseg_val)
            idx = np.argmax(psd)
            imf_freqs[i] = freqs[idx]

            corr_mat = np.corrcoef(sig_col, imf)
            val = corr_mat[0, 1]
            imf_corrs[i] = val if not np.isnan(val) else 0.0
        except Exception:
            imf_freqs[i] = 0.0
            imf_corrs[i] = 0.0

    # 3. Find breathing IMF: dominant freq in resp_band, highest correlation
    resp_candidates = np.where(
        (imf_freqs >= resp_band[0]) & (imf_freqs <= resp_band[1])
    )[0]

    if len(resp_candidates) == 0:
        return signal.copy()

    valid_corrs = np.abs(imf_corrs[resp_candidates])
    max_corr = np.max(valid_corrs)
    strong = resp_candidates[valid_corrs > max_corr * EMD_RESP_CORR_RATIO]
    if len(strong) == 0:
        return signal.copy()
    best_idx = strong[np.argmin(imf_freqs[strong])]
    f_rr = imf_freqs[best_idx]

    if f_rr <= 0:
        return signal.copy()

    # 4. Build harmonic basis: sin/cos(1*f_RR), sin/cos(2*f_RR), sin/cos(3*f_RR)
    t = np.arange(n) / fs
    X_cols = []
    nyquist = fs / 2

    for k in harmonics:
        f_k = k * f_rr
        if f_k >= nyquist:
            continue
        X_cols.append(np.cos(2 * np.pi * f_k * t))
        X_cols.append(np.sin(2 * np.pi * f_k * t))

    if not X_cols:
        return signal.copy()

    X = np.column_stack(X_cols)

    # 5. Least squares fit: beta = argmin ||X*beta - y||, then subtract
    try:
        beta, _, _, _ = np.linalg.lstsq(X, sig_col, rcond=None)
    except np.linalg.LinAlgError:
        return signal.copy()

    harmonic_estimate = X @ beta
    clean = sig_col - harmonic_estimate

    return clean
