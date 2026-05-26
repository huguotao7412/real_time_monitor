"""MUSIC spatial spectrum angle estimation — port of DataProcess.calculateMusicSpectrum

Single-target mode: scans [-60°, 60°], returns the angle of the strongest peak.
"""

import numpy as np
from scipy.signal import sosfiltfilt


def estimate_angle_music(
    rx_matrix: np.ndarray,
    fs: float,
    sos_all: np.ndarray,
    num_signals: int = 1,
    angle_min: float = -60.0,
    angle_max: float = 60.0,
    angle_step: float = 1.0,
    d_spacing: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Estimate target angle using MUSIC pseudo-spectrum.

    Args:
        rx_matrix: Complex IQ data, shape [n_snapshots, n_channels].
        fs: Sampling rate (Hz), used for SOS pre-filtering.
        sos_all: SOS filter coefficients for 0.08–1.8 Hz bandpass,
                 shape [n_sections, 6]. Pre-filters each channel before
                 covariance estimation.
        num_signals: Expected number of signal sources (typically 1).
        angle_min, angle_max, angle_step: Scan grid in degrees.
        d_spacing: Antenna spacing in wavelengths (lambda/2 = 0.5).

    Returns:
        (angle_deg, spectrum): Estimated angle in degrees and the
        normalized MUSIC spectrum over the scan grid.

    Raises:
        ValueError: If rx_matrix has < 2 channels or < 16 snapshots.
    """
    n_snapshots, n_channels = rx_matrix.shape

    if n_channels < 2:
        raise ValueError(f"Need at least 2 channels, got {n_channels}")
    if n_snapshots < 16:
        raise ValueError(f"Need at least 16 snapshots, got {n_snapshots}")

    # Pre-filter each channel (MATLAB: filterObj.apply_all_filter)
    X_filt = np.zeros_like(rx_matrix)
    for ch in range(n_channels):
        X_filt[:, ch] = sosfiltfilt(sos_all, rx_matrix[:, ch])

    # Build covariance matrix [n_channels, n_channels]
    # MATLAB: X = X_avg'; R = (X * X') / num_snapshots;
    X = X_filt.T  # [n_channels, n_snapshots]
    R = (X @ X.conj().T) / n_snapshots

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(R)
    # np.linalg.eigh returns ascending order; reverse for descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # Noise subspace
    if num_signals >= n_channels:
        num_signals = n_channels - 1
    En = eigvecs[:, num_signals:]  # [n_channels, n_noise]
    Q = En @ En.conj().T  # noise subspace projection matrix [n_channels, n_channels]

    # Scan angles
    scan_angles = np.arange(angle_min, angle_max + angle_step, angle_step)
    spectrum = np.zeros(len(scan_angles))

    rx_indices = np.arange(n_channels)

    for i, theta in enumerate(scan_angles):
        theta_rad = np.deg2rad(theta)
        # Steering vector
        a = np.exp(-1j * 2 * np.pi * d_spacing * rx_indices * np.sin(theta_rad))
        # a'H * Q * a -> scalar (a is [n_channels], result is [1, 1])
        a_col = a.reshape(-1, 1)
        denominator = (a_col.conj().T @ Q @ a_col).item()
        spectrum[i] = 1.0 / (abs(denominator) + np.finfo(float).eps)

    # Normalize
    smax = np.max(spectrum)
    if smax > 0:
        spectrum = spectrum / smax

    # Find strongest peak in search range
    peak_idx = np.argmax(spectrum)
    angle_deg = float(scan_angles[peak_idx])

    return angle_deg, spectrum
