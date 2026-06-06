"""Wavelet Packet Decomposition for vital sign separation — port of PhaseProcess.process_signals

Uses db6 wavelet for breath (0.1-0.8 Hz) and sym8 for heart (1.0-2.0 Hz).
Reconstructs signals by summing frequency-ordered nodes in the target bands.
"""

import numpy as np
import pywt
from scipy.signal import savgol_filter


def _compute_wpd_level(signal_len: int) -> int:
    """Dynamic WPD decomposition level, matching MATLAB formula.

    MATLAB: max_level = floor(log2(L)); wpt_level = min(max(max_level - 1, 4), 7)
    """
    max_level = int(np.floor(np.log2(signal_len)))
    return min(max(max_level - 1, 4), 7)


def _reconstruct_band(
    wp: pywt.WaveletPacket,
    level: int,
    fs: float,
    freq_band: tuple[float, float],
) -> np.ndarray:
    """Reconstruct signal by summing WPD nodes whose center freq falls in freq_band.

    Collects all target leaf paths first, zeros non-target leaves once,
    then calls wp.reconstruct() a single time to avoid the O(N) performance trap.
    """
    nodes = wp.get_level(level, order='freq')
    num_nodes = len(nodes)  # 2^level
    f_node_bw = (fs / 2) / num_nodes
    f_lo, f_hi = freq_band

    leaf_nodes = wp.get_leaf_nodes()

    # Collect all target leaf paths across matching nodes
    all_target_leaves: set[str] = set()
    for i, node in enumerate(nodes):
        center_freq = (i + 0.5) * f_node_bw
        if not (f_lo <= center_freq <= f_hi):
            continue
        for leaf in leaf_nodes:
            if leaf.path.startswith(node.path):
                all_target_leaves.add(leaf.path)

    if not all_target_leaves:
        return np.zeros(len(wp.reconstruct(update=False)))

    # Backup all leaf data, zero non-target leaves, reconstruct once, restore
    backup = {leaf.path: leaf.data.copy() for leaf in leaf_nodes}

    for leaf in leaf_nodes:
        if leaf.path not in all_target_leaves:
            n_leaf = len(leaf.data)
            if n_leaf > 0:
                window = np.hanning(n_leaf)
                leaf.data[:] = leaf.data * (1.0 - window)

    result = wp.reconstruct(update=False)

    for leaf in leaf_nodes:
        leaf.data[:] = backup[leaf.path]

    return result


def wpd_separate(
    signal: np.ndarray,
    fs: float,
    breath_band: tuple[float, float] = (0.1, 0.8),
    heart_band: tuple[float, float] = (1.0, 2.0),
    heart_input_signal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate breath and heart signals using wavelet packet decomposition.

    Args:
        signal: 1D displacement signal (used for breath path).
        fs: Sampling rate in Hz.
        breath_band: Frequency range for breath reconstruction (Hz).
        heart_band: Frequency range for heart reconstruction (Hz).
        heart_input_signal: Optional separate input for heart path.
                            If None, uses diff(signal). Use this when heart
                            needs different preprocessing (e.g. EMD) than breath.

    Returns:
        (breath_waveform, heart_waveform): Two 1D arrays same length as input.
    """
    n = len(signal)
    if n < 16:
        raise ValueError(f"Signal too short for WPD: {n} samples")

    wpt_level = _compute_wpd_level(n)

    # Breath: db6 wavelet, no diff, no EMD
    breath_input = signal.copy()
    try:
        wp_breath = pywt.WaveletPacket(
            breath_input, 'db6', mode='symmetric', maxlevel=wpt_level
        )
        breath_wave = _reconstruct_band(wp_breath, wpt_level, fs, breath_band)
    except Exception:
        breath_wave = signal.copy()

    # Heart: sym8 wavelet, on SG derivative (MATLAB: sig_heart_pre = diff(FiltedData))
    if heart_input_signal is None:
        heart_input = savgol_filter(signal, window_length=9, polyorder=3, deriv=1)
    else:
        heart_input = savgol_filter(heart_input_signal, window_length=9, polyorder=3, deriv=1)

    try:
        wp_heart = pywt.WaveletPacket(
            heart_input, 'sym8', mode='symmetric', maxlevel=wpt_level
        )
        heart_wave = _reconstruct_band(wp_heart, wpt_level, fs, heart_band)
    except Exception:
        heart_wave = np.zeros(n)

    return breath_wave, heart_wave
