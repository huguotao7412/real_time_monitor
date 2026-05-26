"""Robust LCMV beamforming with derivative constraint — port of DataProcess.getDisplacement_LCMV

Produces a flat-top beam pattern around the target angle, making the system
robust to small angle estimation errors and body sway.
"""

import numpy as np


def lcmv_displacement(
    rx_matrix: np.ndarray,
    angle_deg: float,
    fc: float = 60.325e9,
    d_factor: float = 0.5,
    loading_factor: float = 0.01,
) -> np.ndarray:
    """LCMV beamforming with derivative constraint → normalized displacement.

    Args:
        rx_matrix: Complex IQ data, shape [n_snapshots, n_channels].
        angle_deg: Target angle in degrees.
        fc: Radar center frequency in Hz (default: 60.325 GHz for RS6240).
        d_factor: Antenna spacing as fraction of wavelength (lambda/2 = 0.5).
        loading_factor: Diagonal loading factor for covariance inversion
                        stability (fraction of mean trace).

    Returns:
        Displacement signal, shape [n_snapshots], normalized to [0, 1]
        (matches MATLAB rescale(unwrapped_phase)).

    Raises:
        ValueError: If rx_matrix has < 2 channels or < 16 snapshots.
    """
    n_snapshots, n_channels = rx_matrix.shape

    if n_channels < 2:
        raise ValueError(f"Need at least 2 channels, got {n_channels}")
    if n_snapshots < 16:
        raise ValueError(f"Need at least 16 snapshots, got {n_snapshots}")

    # Physical constants
    c = 3e8
    lambda_m = c / fc
    d = lambda_m * d_factor

    # Standard array processing format: X = [n_channels, n_snapshots]
    X = rx_matrix.T

    # Sample covariance matrix with diagonal loading
    R = (X @ X.conj().T) / n_snapshots
    trace_R = np.trace(R).real
    R_dl = R + np.eye(n_channels) * (trace_R / n_channels) * loading_factor

    # Invert once
    try:
        R_inv = np.linalg.inv(R_dl)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R_dl)

    # Steering vector and derivative constraint
    theta_rad = np.deg2rad(angle_deg)
    rx_idx = np.arange(n_channels)

    # Constraint 1: standard steering vector a(theta) -> gain = 1
    a_theta = np.exp(-1j * 2 * np.pi * d * rx_idx * np.sin(theta_rad) / lambda_m)

    # Constraint 2: derivative constraint b(theta) -> flat-top at theta
    # d(a)/d(theta) = a * (-j * 2*pi * d * n * cos(theta) / lambda)
    deriv_coeff = -1j * 2 * np.pi * d * np.cos(theta_rad) / lambda_m
    b_theta = deriv_coeff * rx_idx * a_theta

    # Joint constraint matrix C and response vector f
    C = np.column_stack([a_theta, b_theta])  # [n_channels, 2]
    f = np.array([1.0, 0.0])                 # w'a = 1, w'b = 0

    # Optimal LCMV weight: w = R^{-1} C (C^H R^{-1} C)^{-1} f
    C_H_R_inv_C = C.conj().T @ R_inv @ C  # [2, 2]
    try:
        w_opt = R_inv @ C @ np.linalg.solve(C_H_R_inv_C, f)
    except np.linalg.LinAlgError:
        w_opt = R_inv @ C @ np.linalg.lstsq(C_H_R_inv_C, f, rcond=None)[0]

    # Beamforming: y = w^H X
    beamformed = w_opt.conj() @ X  # [n_snapshots]

    # Phase extraction
    raw_phase = np.angle(beamformed)
    unwrapped = np.unwrap(raw_phase)

    # Normalize to [0, 1] (MATLAB rescale)
    umin = np.min(unwrapped)
    umax = np.max(unwrapped)
    if umax - umin > 1e-10:
        displacement = (unwrapped - umin) / (umax - umin)
    else:
        displacement = unwrapped - umin

    return displacement
