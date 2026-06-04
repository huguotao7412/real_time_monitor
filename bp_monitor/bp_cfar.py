"""2D CFAR target detection — ported from MATLAB DataProcess.m.

Implements:
  find_target_bins_1d  — adaptive-threshold range profile peak detection
  adaptive_2d_cfar      — 2D CA-CFAR on Range-Doppler Map with adaptive beta
"""

import numpy as np
from scipy.signal import find_peaks


def find_target_bins_1d(
    mean_bin_frame_rx: np.ndarray,
    distance_per_bin: float = 0.05,
    num_targets: int = 3,
    start_bin: int = 2,
    sensitivity_alpha: float = 3.0,
    verbose: bool = False,
) -> np.ndarray:
    """1D range profile CFAR — adaptive threshold on energy profile.

    MATLAB: DataProcess.findTargetBin

    Args:
        mean_bin_frame_rx: [n_bins, n_frames, n_rx] complex 1D-FFT data
        distance_per_bin: meters per range bin
        num_targets: max targets to return
        start_bin: skip near-field DC clutter bins
        sensitivity_alpha: threshold = noise_mean + alpha * noise_std
        verbose: print debug info

    Returns:
        candidate_bins: sorted range bin indices (nearest first), empty if none found
    """
    n_bins = mean_bin_frame_rx.shape[0]

    # Energy profile: average abs over RX and time
    raw = mean_bin_frame_rx[start_bin:, :, :]
    profile = np.mean(np.mean(np.abs(raw), axis=2), axis=1)  # [n_bins-start_bin]

    # Adaptive threshold: bottom 70% = noise
    sorted_e = np.sort(profile)
    n_noise = max(1, int(0.7 * len(sorted_e)))
    noise = sorted_e[:n_noise]
    noise_mean = float(np.mean(noise))
    noise_std = float(np.std(noise))
    threshold = noise_mean + sensitivity_alpha * noise_std

    peaks, props = find_peaks(profile, height=threshold)
    if len(peaks) == 0:
        if verbose:
            print(f"  [CFAR] No peaks above threshold {threshold:.4f}")
        return np.array([], dtype=int)

    # Top num_targets by height, then sort by distance (nearest first)
    order = np.argsort(props["peak_heights"])[::-1][:num_targets]
    found = np.sort((peaks[order] + start_bin).astype(int))

    if verbose:
        print(f"  [CFAR] noise_mean={noise_mean:.4f} std={noise_std:.4f} thr={threshold:.4f}")
        for i, b in enumerate(found):
            print(f"    Target {i}: bin={b}, dist={b * distance_per_bin:.2f}m")

    return found


def adaptive_2d_cfar(
    mean_bin_frame_rx: np.ndarray,
    candidate_bins: np.ndarray,
    cfar_state: dict | None = None,
    ref_cells_rng: int = 2,
    guard_cells_rng: int = 1,
    ref_cells_dop: int = 6,
    guard_cells_dop: int = 2,
    dop_search_width: int = 15,
) -> tuple[np.ndarray, dict]:
    """2D CA-CFAR on Range-Doppler Map.

    MATLAB: DataProcess.adaptive_2d_cfar_findTargetBin

    For each candidate range bin, searches locally in range-Doppler space,
    estimates noise from a guard-band window, and confirms targets that
    exceed the adaptive threshold.

    Args:
        mean_bin_frame_rx: [n_bins, n_frames, n_rx]
        candidate_bins: from find_target_bins_1d
        cfar_state: persistent dict with 'last_beta', 'last_noise'
        ref_cells_rng, guard_cells_rng: range-dim CFAR window params
        ref_cells_dop, guard_cells_dop: Doppler-dim CFAR window params
        dop_search_width: half-width of Doppler search around DC

    Returns:
        confirmed_bins: verified range bin indices
        cfar_state: updated state for next call
    """
    n_bins, n_doppler, _ = mean_bin_frame_rx.shape

    # Generate Range-Doppler Map
    rdm = np.fft.fft(mean_bin_frame_rx, axis=1)
    rdm = np.fft.fftshift(rdm, axes=1)
    rdm_power = np.sum(np.abs(rdm) ** 2, axis=2)

    center_dop = n_doppler // 2

    # Global noise (80th percentile)
    current_noise = float(np.percentile(rdm_power, 80))

    # Adaptive beta scaling
    if cfar_state is None:
        beta = 6.0
        last_noise = max(current_noise, 1e-10)
    else:
        beta = float(cfar_state["last_beta"])
        last_noise = max(float(cfar_state["last_noise"]), 1e-10)

    beta = max(5.0, min(beta * (current_noise / last_noise), 30.0))

    # Doppler search window
    d_start = max(0, center_dop - dop_search_width)
    d_end = min(n_doppler, center_dop + dop_search_width + 1)

    confirmed = []  # list of (bin, power, snr, dop_idx)

    for cand_bin in candidate_bins:
        ci = int(cand_bin)
        if ci < ref_cells_rng + guard_cells_rng + 1:
            continue
        if ci > n_bins - ref_cells_rng - guard_cells_rng - 1:
            continue

        r_start = max(0, ci - 1)
        r_end = min(n_bins, ci + 2)

        best_power = -np.inf
        best_bin = ci
        best_dop = center_dop
        best_noise_est = 0.0

        for r in range(r_start, r_end):
            for d in range(d_start, d_end):
                cut = float(rdm_power[r, d])
                if cut <= best_power:
                    continue

                # Local noise window
                rs = max(0, r - ref_cells_rng - guard_cells_rng)
                re = min(n_bins, r + ref_cells_rng + guard_cells_rng + 1)
                ds = max(0, d - ref_cells_dop - guard_cells_dop)
                de = min(n_doppler, d + ref_cells_dop + guard_cells_dop + 1)

                window = rdm_power[rs:re, ds:de]

                grs = max(0, r - guard_cells_rng)
                gre = min(n_bins, r + guard_cells_rng + 1)
                gds = max(0, d - guard_cells_dop)
                gde = min(n_doppler, d + guard_cells_dop + 1)
                guard = rdm_power[grs:gre, gds:gde]

                n_ref = window.size - guard.size
                if n_ref <= 0:
                    continue

                noise_level = (window.sum() - guard.sum()) / n_ref
                best_power = cut
                best_bin = r
                best_dop = d
                best_noise_est = noise_level

        if best_power > best_noise_est * beta:
            snr = best_power / max(best_noise_est, 1e-10)
            confirmed.append((best_bin, best_power, snr, best_dop))

    if not confirmed:
        return np.array([], dtype=int), {
            "last_beta": beta, "last_noise": current_noise
        }

    arr = np.array(confirmed)
    return arr[:, 0].astype(int), {
        "last_beta": beta, "last_noise": current_noise
    }
