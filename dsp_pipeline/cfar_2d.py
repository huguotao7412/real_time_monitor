"""2D CFAR target detection on Range-Doppler Map (MATLAB adaptive_2d_cfar_findTargetBin)

Generates RDM from buffered range × slow-time data, applies 2D CFAR with
adaptive threshold factor β, and confirms targets detected by coarse 1D CFAR.
"""

import numpy as np
from config.protocol import (
    CFAR_BETA_INITIAL, CFAR_BETA_MIN, CFAR_BETA_MAX, CFAR_NOISE_PERCENTILE,
    CFAR_1D_ALPHA, CFAR_1D_NOISE_RATIO,
    CFAR_2D_REF_RNG, CFAR_2D_GUARD_RNG, CFAR_2D_REF_DOP, CFAR_2D_GUARD_DOP, CFAR_2D_DOP_SEARCH
)


def adaptive_2d_cfar(
    mean_bin_frame_rx: np.ndarray,
    distance_per_bin: float,
    prev_state: dict | None = None,
    overall_target_bins: np.ndarray | None = None,
    ref_cells_rng: int = CFAR_2D_REF_RNG,
    guard_cells_rng: int = CFAR_2D_GUARD_RNG,
    ref_cells_dop: int = CFAR_2D_REF_DOP,
    guard_cells_dop: int = CFAR_2D_GUARD_DOP,
    dop_search_width: int = CFAR_2D_DOP_SEARCH,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """2D CFAR detection with adaptive threshold.

    Args:
        mean_bin_frame_rx: Background-subtracted data [bins, frames, rx_channels].
        distance_per_bin: Range resolution in meters.
        prev_state: CFAR state from previous call {'last_beta', 'last_noise_level'}.
        overall_target_bins: Coarse 1D CFAR candidate bin indices (0-indexed).
        ref_cells_rng, guard_cells_rng: Range-dimension CFAR window params.
        ref_cells_dop, guard_cells_dop: Doppler-dimension CFAR window params.
        dop_search_width: Half-width of Doppler search around zero-Doppler.

    Returns:
        final_target_bins: Confirmed target bin indices.
        rdm_power: Range-Doppler Map power [bins, doppler].
        debug_info: Detection metadata.
        next_state: CFAR state for next call.
    """
    # ---- 1. RDM generation (MATLAB lines 199-202) ----
    rdm_complex = np.fft.fft(mean_bin_frame_rx, axis=1)
    rdm_complex = np.fft.fftshift(rdm_complex, axes=1)
    rdm_power = np.sum(np.abs(rdm_complex) ** 2, axis=2)  # [bins, doppler]

    num_bins, num_doppler = rdm_power.shape
    center_dop_idx = num_doppler // 2  # Python 0-indexed equivalent of floor(N/2)+1

    # ---- 2. Global noise estimate & adaptive beta (MATLAB lines 206-218) ----
    current_global_noise = float(np.percentile(rdm_power, CFAR_NOISE_PERCENTILE))

    if prev_state is None:
        current_beta = CFAR_BETA_INITIAL
        last_global_noise = current_global_noise if current_global_noise > 0 else 1e-10
    else:
        current_beta = prev_state["last_beta"]
        last_global_noise = prev_state["last_noise_level"]
        if last_global_noise == 0:
            last_global_noise = 1e-10

    scaling_ratio = current_global_noise / last_global_noise
    new_beta = max(CFAR_BETA_MIN, min(current_beta * scaling_ratio, CFAR_BETA_MAX))

    # ---- 3. 2D CFAR scan (MATLAB lines 220-263) ----
    doppler_half = dop_search_width
    d_start = max(0, center_dop_idx - doppler_half)
    d_end = min(num_doppler, center_dop_idx + doppler_half + 1)
    dop_indices = list(range(d_start, d_end))

    edge_margin = ref_cells_rng + guard_cells_rng

    confirmed_targets = []  # list of [bin, peak_power, snr, dop_idx]

    if overall_target_bins is not None and len(overall_target_bins) > 0:
        candidates = np.atleast_1d(overall_target_bins).astype(int)
    else:
        # Fallback: search middle range bins
        candidates = np.arange(edge_margin + 1, num_bins - edge_margin, dtype=int)

    for cand_bin in candidates:
        if cand_bin <= edge_margin or cand_bin >= num_bins - edge_margin:
            continue

        search_bins = [cand_bin - 1, cand_bin, cand_bin + 1]
        local_max_power = -np.inf
        best_local_bin = cand_bin
        best_local_dop = center_dop_idx
        local_noise_est = 0.0

        for r in search_bins:
            if r < 0 or r >= num_bins:
                continue
            for d in dop_indices:
                cut_power = rdm_power[r, d]
                if cut_power <= local_max_power:
                    continue

                # Compute local noise around CUT
                r_s = r - ref_cells_rng - guard_cells_rng
                r_e = r + ref_cells_rng + guard_cells_rng
                d_s = d - ref_cells_dop - guard_cells_dop
                d_e = d + ref_cells_dop + guard_cells_dop

                if r_s < 0 or r_e >= num_bins or d_s < 0 or d_e >= num_doppler:
                    continue

                window_sum = float(np.sum(rdm_power[r_s:r_e + 1, d_s:d_e + 1]))

                gr_s = r - guard_cells_rng
                gr_e = r + guard_cells_rng
                gd_s = d - guard_cells_dop
                gd_e = d + guard_cells_dop
                guard_sum = float(np.sum(rdm_power[gr_s:gr_e + 1, gd_s:gd_e + 1]))

                window_area = (r_e - r_s + 1) * (d_e - d_s + 1)
                guard_area = (gr_e - gr_s + 1) * (gd_e - gd_s + 1)
                num_ref_cells = window_area - guard_area

                noise_level = (window_sum - guard_sum) / num_ref_cells

                local_max_power = cut_power
                best_local_bin = r
                best_local_dop = d
                local_noise_est = noise_level

        adaptive_threshold = local_noise_est * new_beta
        if local_max_power > adaptive_threshold and local_noise_est > 0:
            confirmed_targets.append([
                best_local_bin,
                local_max_power,
                local_max_power / local_noise_est,
                best_local_dop,
            ])

    # ---- 4. Detection map (MATLAB lines 266-293) ----
    detection_map = np.zeros_like(rdm_power)

    if confirmed_targets:
        arr = np.array(confirmed_targets)
        final_target_bins = arr[:, 0]
        final_doppler_idxs = arr[:, 3].astype(int)
        final_snrs = arr[:, 2]
        for k in range(len(final_target_bins)):
            r = int(final_target_bins[k])
            d = int(final_doppler_idxs[k])
            if 0 <= r < num_bins and 0 <= d < num_doppler:
                detection_map[r, d] = 1
        has_detection = True
    else:
        if overall_target_bins is not None and len(overall_target_bins) > 0:
            final_target_bins = np.array([overall_target_bins[0]], dtype=float)
        else:
            final_target_bins = np.array([10.0], dtype=float)
        final_doppler_idxs = np.array([center_dop_idx], dtype=float)
        final_snrs = np.array([0.0])
        has_detection = False

    # ---- 5. Build output ----
    next_state = {
        "last_noise_level": current_global_noise,
        "last_beta": new_beta,
    }

    approx_threshold = current_global_noise * new_beta
    debug_info = {
        "peak_doppler_idx": final_doppler_idxs,
        "detection_map": detection_map,
        "threshold_map": np.full_like(rdm_power, approx_threshold),
        "confirmed_list": np.array(confirmed_targets) if confirmed_targets else np.array([]),
        "has_detection": has_detection,
    }

    return final_target_bins, rdm_power, debug_info, next_state


def coarse_1d_cfar_candidates(
    data_cube: np.ndarray,
    num_candidates: int = 3,
    sensitivity_alpha: float =CFAR_1D_ALPHA,
    start_bin: int = 1,
) -> np.ndarray:
    """1D CFAR coarse search for candidate range bins (MATLAB findTargetBin).

    Args:
        data_cube: Radar data [range_bins, doppler_bins, rx_channels].
        num_candidates: Max number of candidates to return.
        sensitivity_alpha: Threshold multiplier (noise_mean + alpha * noise_std).
        start_bin: First range bin to consider (skip DC).

    Returns:
        Candidate bin indices (0-indexed), sorted by distance ascending.
    """
    n_range = data_cube.shape[0]

    # Aggregate: mean abs across RX channels, across static doppler bin
    doppler_idx = 0 if data_cube.shape[1] > 0 else 0
    raw_slice = data_cube[start_bin:, doppler_idx, :]
    abs_slice = np.abs(raw_slice)
    profile = np.mean(abs_slice, axis=1)  # [bins]

    # Adaptive threshold from lower 70%
    sorted_energy = np.sort(profile)
    n_noise = int(CFAR_1D_NOISE_RATIO * len(sorted_energy))
    noise_samples = sorted_energy[:max(1, n_noise)]
    noise_mean = float(np.mean(noise_samples))
    noise_std = float(np.std(noise_samples))
    threshold = noise_mean + sensitivity_alpha * noise_std

    # Find peaks above threshold
    from scipy.signal import find_peaks
    peaks, _props = find_peaks(profile)
    if len(peaks) == 0:
        return np.array([], dtype=int)

    # Filter by threshold, sort by peak height descending
    peak_heights = profile[peaks]
    valid = peak_heights > threshold
    valid_peaks = peaks[valid]
    valid_heights = peak_heights[valid]

    if len(valid_peaks) == 0:
        return np.array([], dtype=int)

    # Take top N, restore real bin indices, sort by distance
    order = np.argsort(valid_heights)[::-1][:num_candidates]
    found_bins = valid_peaks[order] + start_bin
    found_bins = np.sort(found_bins)
    return found_bins.astype(int)
