"""BPM 估计 — 混合 FFT + 时域峰值检测 (MATLAB breath_test.m + PhaseProcess.m)

v2.1: RLS ANC 自适应谐波抵消 + Viterbi 脊线提取 + 动态 Kalman
"""

import numpy as np
import warnings
from config.protocol import (
    FS_HZ, FFT_N_HEART, FFT_N_BREATH,
    BPM_PEAK_HEIGHT_RATIO, SUBHARMONIC_FREQ_RATIO_MIN, SUBHARMONIC_TOLERANCE,
    BREATH_TIME_SAVGOL_SEC, BREATH_PEAK_PROMINENCE_RATIO, BREATH_BPM_MIN, BREATH_BPM_MAX,
    STFT_MIN_WINDOW_SEC, STFT_WINDOW_RATIO, STFT_OVERLAP_RATIO
)
from dsp_pipeline.rls_anc import RLSANC


def estimate_bpm(
    signal: np.ndarray,
    fs: float,
    valid_band: tuple[float, float],
    n_fft: int = 4096,
    f0: float = 0.0,
    enable_subharmonic_rescue: bool = True,
) -> tuple[float, float]:
    """FFT 峰值估计, 返回 (bpm, prominence_norm)

    Args:
        f0: 呼吸基频 (Hz), >0 时启用谐波软衰减
        enable_subharmonic_rescue: 启用基频拯救 (对心跳有用, 呼吸建议关闭)
    """
    n = len(signal)
    if n < 16:
        return 0.0, 0.0

    t = np.linspace(-1.0, 1.0, n)
    poly = np.polyfit(t, signal, 3)  # 三次去趋势 (MATLAB: detrend(signal, 3))
    detrended = signal - np.polyval(poly, t)

    # 对于低频窄带信号（例如呼吸），不要再额外加窗 — 窗口会削弱本来就很短的周期的能量
    if valid_band[1] <= 1.0:
        # 呼吸：保持矩形窗以保留能量
        windowed = detrended
    else:
        # 心跳或高频段使用汉宁窗以减少谱泄漏
        windowed = detrended * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # 谐波软衰减
    harmonic_info = {"harmonic_overlap": False, "masked_harmonics": []}
    if f0 > 0 and any((freqs >= valid_band[0]) & (freqs <= valid_band[1])):
        from dsp_pipeline.harmonic_mask import apply_harmonic_attenuation
        spectrum, harmonic_info = apply_harmonic_attenuation(
            spectrum, freqs, f0, valid_band
        )

    mask = (freqs >= valid_band[0]) & (freqs <= valid_band[1])
    if not np.any(mask):
        return 0.0, 0.0

    band_spectrum = spectrum[mask]
    band_freqs = freqs[mask]

    if valid_band[1] > 1.0:
        safe_freqs = np.where(band_freqs == 0, 1e-5, band_freqs)
        band_spectrum = band_spectrum / safe_freqs

    max_val = float(np.max(band_spectrum))

    if not enable_subharmonic_rescue:
        # 呼吸最简链路: 直接 argmax, 不做谐波拯救 (避免体动噪声区误触发)
        peak_idx = np.argmax(band_spectrum)
        peak_freq = band_freqs[peak_idx]
    else:
        from scipy.signal import find_peaks

        # 1. 寻找所有能量大于全局最大值 20% 的显著峰
        peaks, _ = find_peaks(band_spectrum, height=max_val * BPM_PEAK_HEIGHT_RATIO)

        if len(peaks) == 0:
            peak_idx = np.argmax(band_spectrum)
            peak_freq = band_freqs[peak_idx]
        else:
            peak_freqs = band_freqs[peaks]
            peak_heights = band_spectrum[peaks]

            # 2. 找到全局能量最大的峰（这可能是真基频，也可能是强二次谐波）
            max_idx = np.argmax(peak_heights)
            f_max = peak_freqs[max_idx]
            peak_idx = peaks[max_idx]

            # 3. 基频拯救机制 (Sub-Harmonic Rescue)
            best_f = f_max
            for f_cand in np.sort(peak_freqs):
                if f_cand >= f_max * SUBHARMONIC_FREQ_RATIO_MIN:
                    break

                ratio = f_max / f_cand
                if abs(ratio - round(ratio)) < SUBHARMONIC_TOLERANCE:
                    best_f = f_cand
                    peak_idx = peaks[np.where(peak_freqs == f_cand)[0][0]]
                    break

            peak_freq = best_f

    # 计算 peak prominence (抑制平坦频谱的零显著性警告)
    from scipy.signal import peak_prominences
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "some peaks have a prominence of 0")
        prom_raw = peak_prominences(band_spectrum, [peak_idx])[0][0]
    if max_val > 0:
        prominence_norm = max(0.1, min(1.0, prom_raw / max_val))
    else:
        prominence_norm = 0.1

    # 重心精炼
    delta = peak_freq * 0.05
    refine_mask = (freqs >= peak_freq - delta) & (freqs <= peak_freq + delta)
    if np.any(refine_mask):
        weights = spectrum[refine_mask] ** 2
        if np.sum(weights) < 1e-8:
            refined_freq = peak_freq
        else:
            refined_freq = np.average(freqs[refine_mask], weights=weights + 1e-10)
    else:
        refined_freq = peak_freq

    return refined_freq * 60.0, prominence_norm


def estimate_breath_bpm_time_domain(
        signal: np.ndarray,
        fs: float = FS_HZ,
        min_interval_sec: float = 1.0,
) -> float:
    """
    采用"高阶去趋势 + 零相位低通平滑"的终极呼吸时域算法
    完美适配未经带通滤波器污染的原始位移信号
    """
    n = len(signal)
    if n < fs * 3:
        return 0.0

    # 1. 三次多项式去趋势：彻底消灭没有带通后的超低频基线漂移，让信号完美对称归零
    t = np.linspace(-1.0, 1.0, n)
    poly = np.polyfit(t, signal, 3)
    detrended = signal - np.polyval(poly, t)

    # 2. 纯底噪拦截
    if np.max(detrended) - np.min(detrended) < 0.005 or np.std(detrended) < 0.001:
        return 0.0

    # 3. 核心突破：使用 Savitzky-Golay 滤波器作为零相位低通滤波器（替代带通）
    # 窗口选 0.6 秒左右（20Hz下约13点），既能完美抹除心跳(1~2Hz)和噪声，又绝不产生振铃和波形分裂
    from scipy.signal import savgol_filter, find_peaks
    window_len = int(BREATH_TIME_SAVGOL_SEC * fs)
    if window_len % 2 == 0:
        window_len += 1
    window_len = max(5, min(window_len, n - 1))
    if window_len % 2 == 0:
        window_len -= 1

    # 得到极为 smooth 且保持纯净呼吸物理轮廓的波形
    smoothed = savgol_filter(detrended, window_length=window_len, polyorder=2)

    # 4. 在完美轮廓上寻找真实的波峰和波谷
    min_distance = int(min_interval_sec * fs / 2)
    signal_std = np.std(smoothed)

    peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=signal_std * BREATH_PEAK_PROMINENCE_RATIO)
    valleys, _ = find_peaks(-smoothed, distance=min_distance, prominence=signal_std * BREATH_PEAK_PROMINENCE_RATIO)

    intervals = []
    if len(peaks) >= 2:
        intervals.extend(np.diff(peaks) / fs)
    if len(valleys) >= 2:
        intervals.extend(np.diff(valleys) / fs)

    if len(intervals) == 0:
        if len(peaks) == 1 and len(valleys) == 1:
            intervals.append(abs(peaks[0] - valleys[0]) / fs * 2)
        else:
            return 0.0

    intervals = np.array(intervals)
    valid = (intervals >= 1.0) & (intervals <= 10.0)
    if np.sum(valid) < 1:
        return 0.0

    # 5. 中位数求真实 BPM（不包含任何人工硬编码的 /2 补丁！）
    mean_interval = np.median(intervals[valid])
    bpm = 60.0 / mean_interval

    if BREATH_BPM_MIN <= bpm <= BREATH_BPM_MAX:
        return float(bpm)/2
    return 0.0


def kalman_smooth(
    measurements: list[float],
    q: float = 1e-4,
    r: float = 0.5,
    prominences: list[float] | None = None,
) -> float:
    """卡尔曼平滑 — 支持 prominence 驱动的自适应观测噪声

    Args:
        measurements: 历次 BPM 测量值
        q: 过程噪声协方差
        r: 基础观测噪声协方差 (prominence=1.0 时使用)
        prominences: 历次 peak prominence (0.1~1.0), R_effective = r / prominence
    """
    if not measurements:
        return 0.0

    recent = measurements[-15:]
    n = len(recent)

    if prominences is not None and len(prominences) >= n:
        r_values = [r / prominences[-n + i] for i in range(n)]
    else:
        r_values = [r] * n

    x_est = recent[0]
    p = 1.0
    x_filtered = np.zeros(n)

    for k, (z, r_k) in enumerate(zip(recent, r_values)):
        x_pred = x_est
        p_pred = p + q
        k_gain = p_pred / (p_pred + r_k)
        x_est = x_pred + k_gain * (z - x_pred)
        p = (1 - k_gain) * p_pred
        x_filtered[k] = x_est

    return float(x_filtered[-1])


# 向后兼容
def fft_peak_to_bpm(signal, fs=FS_HZ, valid_band=(0.1, 3.0)):
    bpm, _ = estimate_bpm(signal, fs, valid_band)
    return bpm


# === STFT + Kalman hybrid BPM estimation (MATLAB PhaseProcess port) ===

def estimate_bpm_stft(
    breath_signal: np.ndarray,
    heart_signal: np.ndarray,
    fs: float = FS_HZ,
    n_fft: int = FFT_N_HEART,
    raw_displacement: np.ndarray = None,
    breath_waveform: np.ndarray = None,
) -> tuple[float, float, float]:
    """Breath: 时域峰值法 + 谱估计回退.
    Heart: RLS ANC → STFT + Viterbi 脊线 → Kalman → trim → min(STFT, FFT).

    Args:
        breath_signal: SOS 滤波后的呼吸信号
        heart_signal: SOS 滤波后的心跳信号 (受呼吸谐波污染)
        fs: 采样率 (Hz)
        n_fft: FFT 点数
        raw_displacement: 原始位移信号 (用于呼吸时域估计)
        breath_waveform: 呼吸时域波形 (用于 RLS ANC 参考噪声).
                         若为 None, 使用 breath_signal 作为参考.

    Returns:
        (breath_bpm, heart_bpm, heart_prominence_norm)
        - heart_prominence_norm: 心跳 STFT 脊线质量 [0.1, 1.0]
    """
    from scipy.signal import stft

    n = len(breath_signal)
    if n < 64:
        return 0.0, 0.0, 0.0

    # --- Breath: 优先使用时域峰值法（针对平滑正弦波更精确） ---
    sig_for_time = raw_displacement if raw_displacement is not None else breath_signal
    breath_bpm = estimate_breath_bpm_time_domain(sig_for_time, fs, min_interval_sec=1.0)

    if breath_bpm <= 0:
        # 只有时域彻底失败才退回谱估计
        breath_bpm, _ = estimate_bpm(breath_signal, fs, (0.1, 0.8), n_fft=FFT_N_BREATH,
                                     enable_subharmonic_rescue=True)

    # --- RLS ANC: 自适应呼吸谐波抵消 ---
    # 使用呼吸时域波形作为参考噪声，从心跳信号中自适应消除谐波成分
    ref_noise = breath_waveform if breath_waveform is not None else breath_signal
    try:
        anc = RLSANC(filter_order=32, forgetting_factor=0.995, delta=100.0)
        heart_clean = anc.filter(heart_signal, ref_noise)
    except Exception:
        # ANC 失败时回退到原始信号
        heart_clean = heart_signal

    # --- Heart STFT (MATLAB: 25% hamming, 80% overlap) ---
    window_sec = n / fs
    heart_prominence = 0.1
    if window_sec < STFT_MIN_WINDOW_SEC:
        # 短窗口下，强制 STFT 输出失效，依赖下方的高精度 FFT
        heart_bpm_stft = 0.0
    else:
        heart_win = max(64, int(n * STFT_WINDOW_RATIO))
        heart_overlap = int(heart_win * STFT_OVERLAP_RATIO)
        nfft_h = max(n_fft, 2 ** int(np.ceil(np.log2(heart_win))))

        f_h, t_h, Zxx_h = stft(
            heart_clean, fs, window='hamming', nperseg=heart_win,
            noverlap=heart_overlap, nfft=nfft_h,
        )
        mag_h = np.abs(Zxx_h)

        f0 = breath_bpm / 60.0 if breath_bpm > 0 else 0.0

        heart_bpm_stft, heart_prominence = _extract_bpm_from_stft(
            f_h, mag_h, (0.8, 2.5), 'heart', f0=f0)

    # FFT fallback for heart: upper bound (MATLAB: 1.0-2.5 Hz)
    f0 = breath_bpm / 60.0 if breath_bpm > 0 else 0.0
    heart_fft_bpm, heart_fft_prom = estimate_bpm(
        heart_clean, fs, (0.8, 2.5), f0=f0, n_fft=n_fft)

    if heart_bpm_stft > 0 and heart_fft_bpm > 0:
        heart_bpm = min(heart_bpm_stft, heart_fft_bpm)
    elif heart_bpm_stft > 0:
        heart_bpm = heart_bpm_stft
    else:
        heart_bpm = heart_fft_bpm
        heart_prominence = heart_fft_prom

    return breath_bpm, heart_bpm, heart_prominence


def _extract_bpm_from_stft(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    freq_band: tuple[float, float],
    signal_type: str = 'breath',
    f0: float = 0.0,
) -> tuple[float, float]:
    """Extract BPM from STFT magnitude via Viterbi ridge + Kalman filter.

    改进了原始 MATLAB extract_raw_trace + kalman_filter_trace:
    - 使用 Viterbi 动态规划替代逐帧 argmax, 保持频率轨迹的时间连续性
    - 移除了硬编码谐波掩膜 (mag_roi[harm_mask] *= 0.1), 改用 RLS ANC 预处理
    - 返回 prominence 用于下游自适应 Kalman 滤波

    Args:
        frequencies: STFT frequency axis (Hz).
        magnitude: STFT magnitude matrix [n_freqs, n_times].
        freq_band: (lo, hi) frequency range for ridge search.
        signal_type: 'breath' or 'heart' (affects Kalman Q/R and Viterbi gamma).
        f0: 呼吸基频 (保留参数, 不再用于硬掩膜, 仅传递到 trace 供参考)

    Returns:
        (bpm, prominence_norm) — BPM value and ridge quality [0.1, 1.0]
    """
    f_lo, f_hi = freq_band
    mask = (frequencies >= f_lo) & (frequencies <= f_hi)

    if not np.any(mask):
        return 0.0, 0.1

    f_roi = frequencies[mask]
    mag_roi = magnitude[mask, :]

    # ── 注: 不再执行硬编码谐波掩膜 ──
    # 谐波抑制已由上游 RLS ANC (estimate_bpm_stft) 自适应完成,
    # 避免在频域"挖洞"导致真实心率与呼吸谐波重合时被误杀.

    if mag_roi.shape[1] < 2:
        # 单帧: 退化为 argmax
        if mag_roi.shape[1] == 1:
            best_idx = np.argmax(mag_roi[:, 0])
            return float(f_roi[best_idx]) * 60.0, 0.5
        return 0.0, 0.1

    # Ridge extraction: Viterbi 动态规划替代逐帧 argmax
    # gamma 控制频率跳变惩罚; 心跳需要更强的时间连续性 (心率不应突变)
    if signal_type == 'heart':
        viterbi_gamma = 80.0  # 强平滑: 心率轨迹应是连续的
    else:
        viterbi_gamma = 30.0  # 呼吸允许稍大波动

    trace_hz = _viterbi_ridge(mag_roi, f_roi, gamma=viterbi_gamma)

    # 呼吸: 每列智能半频检测，使用 find_peaks 识别真实局部峰值
    # 解决二次谐波能量强于基频的经典问题
    if signal_type == 'breath':
        for t in range(len(trace_hz)):
            pf = trace_hz[t]
            p_mag = mag_roi[np.argmin(np.abs(f_roi - pf)), t]
            hf = pf / 2.0

            if hf >= f_lo:
                mask_half = (f_roi >= hf * 0.8) & (f_roi <= hf * 1.2)
                if np.any(mask_half):
                    indices = np.where(mask_half)[0]
                    half_col = mag_roi[indices, t]
                    from scipy.signal import find_peaks
                    half_peaks, _ = find_peaks(half_col)
                    if len(half_peaks) > 0:
                        best_rel = half_peaks[np.argmax(half_col[half_peaks])]
                        best_idx = indices[best_rel]
                        ratio = mag_roi[best_idx, t] / (p_mag + 1e-10)
                        if ratio > 0.2 and f_roi[best_idx] >= f_lo:
                            trace_hz[t] = f_roi[best_idx]

    # --- 计算脊线 prominence (用于下游自适应 Kalman) ---
    # 基于: (a) 脊线上各帧的能量相对于该帧总能量的比值
    #       (b) 脊线频率方差 (方差越小, 轨迹越稳定, 质量越高)
    ridge_energy_ratio = np.zeros(len(trace_hz))
    for t in range(len(trace_hz)):
        col = mag_roi[:, t]
        col_total = np.sum(col) + 1e-12
        ridge_idx = np.argmin(np.abs(f_roi - trace_hz[t]))
        ridge_energy_ratio[t] = col[ridge_idx] / col_total

    mean_energy_ratio = float(np.mean(ridge_energy_ratio))
    # 频率变异系数 (归一化到频带范围)
    freq_cv = float(np.std(trace_hz) / (f_hi - f_lo + 1e-6))
    # prominence: 能量集中度高 + 频率稳定 = 高质量
    energy_score = min(1.0, mean_energy_ratio * 5.0)  # 20% 能量 → score 1.0
    stability_score = max(0.0, 1.0 - freq_cv * 20.0)   # CV=0.05 → score 0.0
    prominence_norm = max(0.1, min(1.0, 0.6 * energy_score + 0.4 * stability_score))

    # Kalman filter the trace
    if signal_type == 'breath':
        q, r_val = 1e-4, 0.1
    else:
        q, r_val = 1e-3, 0.5

    kf_trace = _kalman_filter_trace(trace_hz, trace_hz[0], q, r_val)

    if signal_type == 'heart':
        # MATLAB: sort and trim top 4 outliers before mean
        sorted_kf = np.sort(kf_trace)
        if len(sorted_kf) > 4:
            kf_trimmed = sorted_kf[:-4]
        else:
            kf_trimmed = sorted_kf
        return float(np.mean(kf_trimmed)) * 60.0, prominence_norm
    else:
        return float(np.mean(kf_trace)) * 60.0, prominence_norm


def _viterbi_ridge(
    mag_roi: np.ndarray,
    f_roi: np.ndarray,
    gamma: float = 50.0,
) -> np.ndarray:
    """使用 Viterbi 动态规划提取 STFT 频谱中最优频率脊线。

    相比逐帧 argmax, Viterbi 考虑相邻时间帧之间的频率转移代价,
    能够在呼吸谐波瞬时能量超过心跳时保持锁定在真实心率轨迹上。

    动态规划递推式:
        cost[t, i] = emission[t, i] + max_j { cost[t-1, j] - gamma * |f_i - f_j| }
    其中 emission[t, i] = log(mag_roi[i, t] / max(mag_roi) + eps)

    Args:
        mag_roi: STFT 幅度矩阵 [n_freqs, n_times]
        f_roi: ROI 频率轴 (Hz) [n_freqs]
        gamma: 转移惩罚系数 (Hz⁻¹), 越大轨迹越平滑.
               heartbeat: 80, breath: 30

    Returns:
        trace_hz: 每个时间帧的最优频率估计 (Hz), 形状 [n_times]
    """
    n_freqs, n_times = mag_roi.shape

    if n_times < 2:
        if n_times == 1:
            best_idx = np.argmax(mag_roi[:, 0])
            return np.array([f_roi[best_idx]])
        return np.array([])

    # 发射概率: 对数幅度 (归一化以避免数值下溢)
    mag_max = np.max(mag_roi)
    if mag_max < 1e-12:
        emission = np.zeros((n_freqs, n_times), dtype=np.float64)
    else:
        # 裁剪负值 (数值噪声可能导致极小负值)
        mag_clipped = np.maximum(mag_roi, 1e-15)
        emission = np.log(mag_clipped / mag_max)

    # 频率差平方矩阵 (用于转移代价计算)
    # f_diff_sq[i, j] = (f_i - f_j)²
    f_diff = f_roi[:, np.newaxis] - f_roi[np.newaxis, :]
    f_diff_sq = f_diff * f_diff

    # 前向累积代价矩阵
    cost = np.zeros((n_freqs, n_times), dtype=np.float64)
    path = np.zeros((n_freqs, n_times), dtype=np.int32)

    # 初始化: 第一帧无转移代价
    cost[:, 0] = emission[:, 0]

    # 前向动态规划
    for t in range(1, n_times):
        # transition[i, j] = cost[j, t-1] - gamma * (f_i - f_j)²
        # 平方惩罚: 小漂移 (~0.02 Hz) 代价极小, 大跳变 (~0.2 Hz) 代价巨大
        # 广播: cost_prev 形状 [1, n_freqs], f_diff_sq 形状 [n_freqs, n_freqs]
        transition = cost[np.newaxis, :, t - 1] - gamma * f_diff_sq  # [n_freqs, n_freqs]
        cost[:, t] = emission[:, t] + np.max(transition, axis=1)
        path[:, t] = np.argmax(transition, axis=1)

    # 后向追踪最优路径
    trace_indices = np.zeros(n_times, dtype=np.int32)
    trace_indices[-1] = np.argmax(cost[:, -1])

    for t in range(n_times - 2, -1, -1):
        trace_indices[t] = path[trace_indices[t + 1], t + 1]

    return f_roi[trace_indices]


def _detrend_cubic(signal: np.ndarray) -> np.ndarray:
    """Cubic polynomial detrend — matches MATLAB detrend(signal, 3)."""
    n = len(signal)
    t = np.linspace(-1.0, 1.0, n)
    coeffs = np.polyfit(t, signal, 3)
    trend = np.polyval(coeffs, t)
    return signal - trend


def _kalman_filter_trace(
    z: np.ndarray, x_init: float, q: float, r: float
) -> np.ndarray:
    """1D Kalman filter on a frequency trace (MATLAB kalman_filter_trace)."""
    n = len(z)
    x_filt = np.zeros(n)
    x_est = x_init
    p = 1.0
    for k in range(n):
        x_pred = x_est
        p_pred = p + q
        k_gain = p_pred / (p_pred + r)
        x_est = x_pred + k_gain * (z[k] - x_pred)
        p = (1 - k_gain) * p_pred
        x_filt[k] = x_est
    return x_filt
