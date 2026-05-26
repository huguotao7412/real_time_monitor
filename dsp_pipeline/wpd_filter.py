"""Wavelet Packet Decomposition for vital sign separation — port of PhaseProcess.process_signals

Uses db6 wavelet for breath (0.1-0.8 Hz) and sym8 for heart (1.0-2.0 Hz).
Reconstructs signals by summing frequency-ordered nodes in the target bands.
"""

import numpy as np
import pywt


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

    Matches MATLAB: sum(wprcoef(node)) for nodes where center_freq in band.
    """
    nodes = wp.get_level(level, order='freq')
    num_nodes = len(nodes)  # 2^level
    f_node_bw = (fs / 2) / num_nodes
    f_lo, f_hi = freq_band

    leaf_nodes = wp.get_leaf_nodes()
    n_signal = len(wp.reconstruct(update=False))
    result = np.zeros(n_signal)

    # Build mapping: frequency-order node path -> list of its leaf paths
    for i, node in enumerate(nodes):
        center_freq = (i + 0.5) * f_node_bw  # 0-indexed freq order
        if not (f_lo <= center_freq <= f_hi):
            continue

        target_leaves = [
            leaf.path for leaf in leaf_nodes
            if leaf.path.startswith(node.path)
        ]

        if not target_leaves:
            continue

        # Backup all leaf data, zero non-target leaves, reconstruct, restore
        backup = {leaf.path: leaf.data.copy() for leaf in leaf_nodes}

        for leaf in leaf_nodes:
            if leaf.path not in target_leaves:
                leaf.data[:] = 0.0

        result += wp.reconstruct(update=False)

        # Restore all leaf data for next iteration
        for leaf in leaf_nodes:
            leaf.data[:] = backup[leaf.path]

    return result


def wpd_separate(
    signal: np.ndarray,
    fs: float,
    breath_band: tuple[float, float] = (0.1, 0.8),
    heart_band: tuple[float, float] = (1.0, 2.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Separate breath and heart signals using wavelet packet decomposition.

    Args:
        signal: 1D real-valued cleaned displacement signal.
        fs: Sampling rate in Hz.
        breath_band: Frequency range for breath reconstruction (Hz).
        heart_band: Frequency range for heart reconstruction (Hz).

    Returns:
        (breath_waveform, heart_waveform): Two 1D arrays same length as input.
        On WPD failure, returns (signal, zeros).

    Raises:
        ValueError: If signal length < 16 (too short for WPD).
    """
    n = len(signal)
    if n < 16:
        raise ValueError(f"Signal too short for WPD: {n} samples")

    wpt_level = _compute_wpd_level(n)

    # Breath: db6 wavelet, no diff (MATLAB: sig_enhanced_nodiff = FiltedData(2:end))
    breath_input = signal[1:] if len(signal) > 1 else signal
    try:
        wp_breath = pywt.WaveletPacket(
            breath_input, 'db6', mode='symmetric', maxlevel=wpt_level
        )
        breath_wave = _reconstruct_band(wp_breath, wpt_level, fs, breath_band)
        # Pad to match input length (MATLAB: rec_breath = zeros(size(sig_enhanced)))
        if len(breath_wave) < n:
            breath_wave = np.pad(breath_wave, (0, n - len(breath_wave)))
    except Exception:
        breath_wave = signal.copy()

    # Heart: sym8 wavelet, on diff(signal) (MATLAB: sig_heart_pre = diff(FiltedData))
    heart_input = np.diff(signal, prepend=signal[0])
    try:
        wp_heart = pywt.WaveletPacket(
            heart_input, 'sym8', mode='symmetric', maxlevel=wpt_level
        )
        heart_wave = _reconstruct_band(wp_heart, wpt_level, fs, heart_band)
    except Exception:
        heart_wave = np.zeros(n)

    return breath_wave, heart_wave
