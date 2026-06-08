"""DSP 处理管线 — MATLAB PhaseProcess.m + DataProcess.m 移植"""

import threading
import time
import queue
from collections import deque
import numpy as np

# Safe import of scipy.signal functions with lightweight fallbacks if SciPy is not available
try:
    from scipy.signal import sosfiltfilt, savgol_filter
except Exception:
    # Fallbacks: keep pipeline running even if SciPy isn't installed
    def _sosfiltfilt_passthrough(sos, x):
        return x
    sosfiltfilt = _sosfiltfilt_passthrough

    def savgol_filter(x, window_length=9, polyorder=3, deriv=0):
        import numpy as _np
        # Basic moving-average fallback for smoothing, gradient for derivative
        if window_length < 3:
            window_length = 3
        if deriv == 0:
            w = _np.ones(window_length) / float(window_length)
            return _np.convolve(x, w, mode='same')
        else:
            sm = _np.convolve(x, _np.ones(window_length) / float(window_length), mode='same')
            return _np.gradient(sm)

from config.protocol import (
    RAW_QUEUE_MAXSIZE, DISPLAY_QUEUE_MAXSIZE,
    WINDOW_SIZE, FS_HZ, BPM_UPDATE_INTERVAL,
    BREATH_RAW_HISTORY_MAXLEN, BREATH_HISTORY_MAXLEN,
    BREATH_USE_NEW_SMOOTHER, HEART_USE_NEW_SMOOTHER,
)
from models.radar_frame import RadarFrame
from dsp_pipeline.vital_signs import VitalSigns
from dsp_pipeline.range_bin import find_best_range_bin
from dsp_pipeline.cfar_2d import coarse_1d_cfar_candidates, adaptive_2d_cfar
from dsp_pipeline.phase import extract_phase, unwrap_phase
from dsp_pipeline.filters import remove_dc, VitalSignFilter
from dsp_pipeline.fft_bpm import estimate_bpm, kalman_smooth, estimate_breath_bpm_time_domain, estimate_bpm_stft
from dsp_pipeline.music_angle import estimate_angle_music
from dsp_pipeline.lcmv_beamformer import lcmv_displacement
from dsp_pipeline.smoothers import SmootherState, apply_smoothing_chain, compute_sqi


class Pipeline:
    def __init__(self, use_beamforming: bool = True):
        self.raw_queue = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
        self.display_queue = queue.Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_count = 0

        # Per-RX complex buffer (replaces scalar _phase_buffer for beamforming)
        self._rx_buffer: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
        # Scalar phase buffer kept as fallback
        self._phase_buffer: deque[float] = deque(maxlen=WINDOW_SIZE)

        self._best_bin: float | None = None
        self._last_bpm_update = 0
        self.last_heartbeat = 0.0

        # 2D-CFAR state (MATLAB adaptive_2d_cfar_findTargetBin)
        self._cfar_accumulator: list[np.ndarray] = []
        self._cfar_state: dict | None = None
        self._cfar_rolling_buffer: deque[np.ndarray] = deque(maxlen=int(FS_HZ * 2.5))
        self._cfar_initial_frames: int = int(FS_HZ * 2.5)
        self._cfar_rescan_interval: int = int(FS_HZ * 5.0)
        self._current_bin_snr: float = 0.0
        self.DISTANCE_PER_BIN: float = 0.05  # RS6240 range resolution

        # Feature toggles
        self._use_beamforming = use_beamforming
        self._use_advanced_dsp: bool = True  # Step 2: EMD + WPD + STFT

        # Beamforming state
        self._angle_deg: float = 0.0  # initial guess: boresight
        self._angle_initialized: bool = False
        self._beamforming_ok: bool = True  # set False on failure -> fallback
        self._last_music_update: int = -50
        self._music_update_interval: int = int(FS_HZ * 2.5)

        # MATLAB Filter.m: SOS 滤波器组
        self._filter = VitalSignFilter(fs=FS_HZ)

        # Kalman 追踪历史
        self._breath_history: list[float] = []
        self._heart_history: list[float] = []
        # raw history deque for median prefilter; size moved to config default
        from config.protocol import BREATH_RAW_HISTORY_MAXLEN, BREATH_HISTORY_MAXLEN
        self._breath_raw_history: deque[float] = deque(maxlen=BREATH_RAW_HISTORY_MAXLEN)  # 中值预滤波
        self._heart_raw_history: deque[float] = deque(maxlen=3)

        # Smoother state for breath
        self._breath_smoother = SmootherState()
        # Optional smoother for heart (off by default)
        self._heart_smoother = SmootherState()

        # 弱信号计数 (用于 Range Bin 重捕获)
        self._low_signal_frame_count: int = 0
        self._last_valid_breath_bpm: float = 0.0
        self._last_valid_heart_bpm: float = 0.0

        # EMA 平滑状态 + 生理区间拦截
        self._breath_ema: float = 0.0
        self._heart_ema: float = 0.0

        # 自适应 Kalman: 心率 prominence 历史

        # 自适应 Kalman: 心率 prominence 历史
        self._heart_prominence_history: list[float] = []

        # Cached advanced-DSP waveforms for display between BPM updates
        self._cached_breath_wave: np.ndarray | None = None
        self._cached_heart_wave: np.ndarray | None = None

        # Phase unwrapping continuity state (prevents 2π jumps across sliding windows)
        self._last_unwrapped_phase: float | None = None

    @property
    def calibration_done(self) -> bool:
        return True

    @property
    def calibration_progress(self) -> float:
        return 1.0

    @property
    def best_range_bin(self) -> float | None:
        return self._best_bin

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame: RadarFrame = self.raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self._process_frame(frame)
                if result is not None:
                    self._push_to_display(result)
            except Exception as e:
                print(f"[DSP] Error: {e}")

    # MUSIC/LCMV 通道选择: MATLAB 用 [1,2,5,6] (2T4R, 取前两个TX各前两个RX)
    _MUSIC_CHANNELS = [0, 1, 4, 5]

    def _extract_rx_complex(self, data_cube: np.ndarray) -> np.ndarray:
        """Extract per-RX complex IQ at target range bin with +/-2 bin averaging.

        Matches MATLAB calculateMusicSpectrum range bin smoothing.
        Returns shape [rx_antennas] complex array.
        """
        n_range = data_cube.shape[0]
        bin_idx = int(self._best_bin)
        start_bin = max(1, bin_idx - 2)
        end_bin = min(n_range - 1, bin_idx + 2)
        rx_slice = data_cube[start_bin:end_bin + 1, 0, :]  # [n_bins, rx]
        full_rx = np.mean(rx_slice, axis=0)  # [rx]
        # Select channels [0,1,4,5] for MUSIC/LCMV (MATLAB [1,2,5,6])
        if len(full_rx) > max(self._MUSIC_CHANNELS):
            return full_rx[self._MUSIC_CHANNELS]
        return full_rx

    def _process_frame(self, frame: RadarFrame) -> VitalSigns | None:
        data_cube = frame.data_cube

        # 1. Range Bin锁定 (2-stage CFAR: 1D coarse + 2D refinement)
        self._cfar_rolling_buffer.append(data_cube)

        if self._best_bin is None:
            self._cfar_accumulator.append(data_cube)
            if len(self._cfar_accumulator) >= self._cfar_initial_frames:
                self._best_bin, self._current_bin_snr = self._run_2d_cfar_lock()
                self._cfar_accumulator.clear()
            if self._best_bin is None:
                return None
        elif self._frame_count > 0 and self._frame_count % self._cfar_rescan_interval == 0:
            new_bin, new_snr, current_actual_snr = self._run_2d_cfar_rescan()
            if new_bin is not None and current_actual_snr > 0 and new_snr > current_actual_snr * 1.5:
                self._best_bin = new_bin
                self._current_bin_snr = new_snr

        # 2. Extract per-RX complex data and buffer
        rx_complex = None
        try:
            rx_complex = self._extract_rx_complex(data_cube)
            self._rx_buffer.append(rx_complex)
        except (IndexError, ValueError):
            pass

        # 3. Fallback: simple scalar phase (always buffered)
        phase = extract_phase(data_cube, self._best_bin)
        self._phase_buffer.append(phase)
        self._frame_count += 1

        if len(self._phase_buffer) < WINDOW_SIZE:
            return None

        # 4. Choose signal path
        should_update_bpm = (
            self._frame_count - self._last_bpm_update >= BPM_UPDATE_INTERVAL
        )

        if self._use_beamforming and self._beamforming_ok and rx_complex is not None:
            displacement = self._beamforming_path(should_update_bpm)
        else:
            displacement = self._fallback_phase_path()

        # 5. Shared downstream: detrend -> diff -> SOS -> BPM -> quality
        return self._shared_signal_chain(displacement, should_update_bpm)

    def _build_mean_bin_frame_rx(self, cubes: list[np.ndarray]) -> np.ndarray:
        """Concatenate frames → [bins, frames, rx] with background subtraction."""
        bin_frame_rx = np.concatenate(cubes, axis=1)  # [bins, N, rx]
        background = np.mean(bin_frame_rx, axis=1, keepdims=True)
        return bin_frame_rx - background

    def _run_2d_cfar_lock(self) -> tuple[float | None, float]:
        """1D coarse + 2D refinement CFAR for initial target lock."""
        mean_bin_frame_rx = self._build_mean_bin_frame_rx(self._cfar_accumulator)
        candidates = coarse_1d_cfar_candidates(mean_bin_frame_rx)
        final_bins, _, debug, self._cfar_state = adaptive_2d_cfar(
            mean_bin_frame_rx, self.DISTANCE_PER_BIN, self._cfar_state, candidates
        )
        confirmed = debug.get("confirmed_list", np.array([]))
        if len(confirmed) > 0:
            best_idx = np.argmin(confirmed[:, 0])  # closest bin
            best_bin = float(confirmed[best_idx, 0])
            snr = float(confirmed[best_idx, 2])
            return best_bin, snr
        # Fallback: 1D CFAR only
        if len(candidates) > 0:
            return float(candidates[0]), 0.0
        # Ultimate fallback
        best_bin = find_best_range_bin(mean_bin_frame_rx, fs=FS_HZ)
        return best_bin, 0.0

    def _run_2d_cfar_rescan(self) -> tuple[float | None, float, float]:
        """Periodic re-scan using rolling buffer.

        Returns (new_bin, new_snr, current_actual_snr). current_actual_snr is the
        SNR of self._best_bin measured from the rolling buffer data at this moment.
        """
        cubes = list(self._cfar_rolling_buffer)
        if len(cubes) < 20:
            return None, 0.0, 0.0
        mean_bin_frame_rx = self._build_mean_bin_frame_rx(cubes)
        candidates = coarse_1d_cfar_candidates(mean_bin_frame_rx)
        _, _, debug, _ = adaptive_2d_cfar(
            mean_bin_frame_rx, self.DISTANCE_PER_BIN, self._cfar_state, candidates
        )
        confirmed = debug.get("confirmed_list", np.array([]))

        # Compute current best_bin's actual SNR from the rolling buffer
        current_actual_snr = 0.0
        if len(confirmed) > 0:
            for entry in confirmed:
                if int(entry[0]) == int(self._best_bin):
                    current_actual_snr = float(entry[2])
                    break

        if len(confirmed) > 0:
            best_idx = np.argmin(confirmed[:, 0])
            best_bin = float(confirmed[best_idx, 0])
            snr = float(confirmed[best_idx, 2])
            if best_bin != self._best_bin:
                return best_bin, snr, current_actual_snr
        return None, 0.0, current_actual_snr

    def _beamforming_path(self, update_angle: bool) -> np.ndarray | None:
        """Run MUSIC + LCMV on the RX buffer. Returns displacement or None."""
        try:
            rx_matrix = np.array(self._rx_buffer)  # [window, rx]

            # Periodic MUSIC angle update (~every 50 frames), MATLAB-style per-frame
            if self._frame_count - self._last_music_update >= self._music_update_interval:
                try:
                    angle, _, peak_q = estimate_angle_music(
                        rx_matrix, FS_HZ, self._filter.sos_all,
                        num_signals=1,
                    )
                    if peak_q > 2.5:
                        self._angle_deg = angle
                        self._angle_initialized = True
                    self._last_music_update = self._frame_count
                except Exception:
                    pass

            # LCMV beamforming (fast enough to run every frame)
            displacement = lcmv_displacement(rx_matrix, self._angle_deg)
            return displacement

        except Exception:
            self._beamforming_ok = False
            return self._fallback_phase_path()

    def _fallback_phase_path(self) -> np.ndarray:
        """Simple phase path: unwrap with cross-frame continuity -> detrend.

        The sliding window shifts by 1 sample per frame. Independent np.unwrap
        on each window causes 2π jumps between frames. We align overlapping
        samples to produce a temporally continuous phase stream.

        Returns displacement-like array for downstream compatibility.
        """
        phase_arr = np.array(self._phase_buffer)
        unwrapped = unwrap_phase(phase_arr)

        if self._last_unwrapped_phase is not None:
            # unwrapped[-2] and _last_unwrapped_phase represent the same raw
            # sample (the window shifted by 1). Align to remove any 2π jump.
            offset = self._last_unwrapped_phase - unwrapped[-2]
            n2pi = np.round(offset / (2 * np.pi)) * (2 * np.pi)
            unwrapped = unwrapped + n2pi

        self._last_unwrapped_phase = unwrapped[-1]
        return remove_dc(unwrapped)

    def _advanced_dsp_path(
        self, displacement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """SOS → WPD (breath: SOS only, heart: diff+WPD) → BPM. MATLAB PhaseProcess port."""
        # SOS pre-filter (MATLAB: filterObj.apply_all_filter)
        try:
            filted = sosfiltfilt(self._filter.sos_all, displacement)
        except Exception:
            filted = displacement.copy()

        # Breath: 跳过 WPD，直接用 SOS 带通 (呼吸能量强，无需小波重建)
        breath_wave = self._filter.filter_breath(filted)
        breath_wave = breath_wave[1:]  # MATLAB: sig_enhanced_nodiff = FiltedData(2:end)

        # Heart: diff → WPD sym8 (MATLAB: sig_heart_pre = diff(FiltedData); wpdec(sig_heart_pre))
        from dsp_pipeline.emd_cleaner import emd_harmonic_clean
        from dsp_pipeline.wpd_filter import wpd_separate
        try:
            # 清除 displacement 中的呼吸谐波后再提取心率
            clean_disp = emd_harmonic_clean(filted, FS_HZ, max_imf=4)
            heart_diff = np.diff(clean_disp)
            _, heart_wave = wpd_separate(
                clean_disp, FS_HZ, heart_input_signal=heart_diff
            )
        except Exception:
            print(f"[Advanced DSP] WPD Fallback triggered: {e}")
            enhanced = savgol_filter(filted, window_length=9, polyorder=3, deriv=1)
            heart_wave = self._filter.filter_heart(enhanced)

        # BPM estimation: FFT breath + STFT heart (MATLAB PhaseProcess port)
        try:
            breath_bpm, heart_bpm = estimate_bpm_stft(
                breath_wave, heart_wave, FS_HZ,1024, raw_displacement=displacement
            )
        except Exception:
            breath_bpm, _ = estimate_bpm(
                breath_wave, FS_HZ, (0.1, 0.8), n_fft=4096,
                enable_subharmonic_rescue=True,
            )
            if breath_bpm <= 0:
                breath_bpm = estimate_breath_bpm_time_domain(breath_wave, FS_HZ)
            f0 = breath_bpm / 60.0 if breath_bpm > 0 else 0.0
            heart_bpm, _ = estimate_bpm(
                heart_wave, FS_HZ, (0.8, 2.0), f0=f0
            )

        return breath_wave, heart_wave, breath_bpm, heart_bpm

    def _shared_signal_chain(
            self, displacement: np.ndarray | None, update_bpm: bool
    ) -> VitalSigns | None:
        """Common signal processing chain: diff -> SOS -> apnea -> BPM -> quality."""
        if displacement is None:
            return None

        no_dc = remove_dc(displacement)
        enhanced = savgol_filter(no_dc, window_length=9, polyorder=3, deriv=1)

        # SOS 带通滤波: 呼吸用原始位移防振铃分裂, 心跳用差分放大高频脉冲
        breath_signal = self._filter.filter_breath(no_dc)
        heart_signal = self._filter.filter_heart(enhanced)

        # 信号能量指标
        phase_range = float(np.max(no_dc) - np.min(no_dc))
        breath_energy = float(np.var(breath_signal))
        total_energy = float(np.var(enhanced)) + 1e-10
        breath_power_ratio = breath_energy / total_energy

        # 提取最近 30 帧 (1.5秒) 短时能量，避免 10 秒窗口导致的响应迟钝
        recent_frames = int(FS_HZ * 1.5)
        if len(no_dc) >= recent_frames:
            recent_no_dc = no_dc[-recent_frames:]
            recent_phase_range = float(np.max(recent_no_dc) - np.min(recent_no_dc))
        else:
            recent_phase_range = phase_range

        # 弱信号检测 → Range Bin 重捕获
        in_low_signal = (recent_phase_range < 0.005)

        if in_low_signal:
            self._low_signal_frame_count += 1
        else:
            self._low_signal_frame_count = 0

        if self._low_signal_frame_count >= int(FS_HZ * 1.5) and self._best_bin is not None:
            self._best_bin = None

        # BPM 估计
        breath_bpm = self._last_valid_breath_bpm
        heart_bpm = self._last_valid_heart_bpm
        breath_signal_display = breath_signal
        heart_signal_display = heart_signal

        if update_bpm:
            if self._use_advanced_dsp:
                try:
                    adv_breath, adv_heart, adv_breath_bpm, adv_heart_bpm = \
                        self._advanced_dsp_path(displacement)
                    breath_signal_display = adv_breath
                    heart_signal_display = adv_heart
                    breath_bpm = adv_breath_bpm
                    heart_bpm = adv_heart_bpm

                    # 中值去飞点 → Kalman 平滑 → EMA 稳显示
                    if breath_bpm > 0:
                        # Use combined smoother: median -> kalman -> jump-filter -> adaptive EMA
                        self._breath_raw_history.append(breath_bpm)
                        # compute short-term breath power ratio and phase_range for SQI
                        sqi_val = compute_sqi(recent_phase_range, breath_power_ratio, self._current_bin_snr)
                        breath_bpm = apply_smoothing_chain(self._breath_smoother, breath_bpm,
                                                           recent_phase_range, breath_power_ratio,
                                                           self._current_bin_snr)
                        self._last_valid_breath_bpm = breath_bpm

                    if heart_bpm > 0:
                        # Heart smoothing: either old median->kalman or new smoother
                        if HEART_USE_NEW_SMOOTHER:
                            self._heart_raw_history.append(heart_bpm)
                            heart_bpm = apply_smoothing_chain(self._heart_smoother, heart_bpm,
                                                              recent_phase_range, breath_power_ratio,
                                                              self._current_bin_snr)
                            self._last_valid_heart_bpm = heart_bpm
                        else:
                            self._heart_raw_history.append(heart_bpm)
                            heart_bpm = float(np.median(list(self._heart_raw_history)))
                            self._heart_history.append(heart_bpm)
                            if len(self._heart_history) > 10:
                                self._heart_history = self._heart_history[-10:]
                            heart_bpm = kalman_smooth(self._heart_history, q=1e-3, r=0.5)
                            self._last_valid_heart_bpm = heart_bpm
                    self._cached_breath_wave = adv_breath
                    self._cached_heart_wave = adv_heart
                except Exception:
                    self._use_advanced_dsp = False

            if not self._use_advanced_dsp or breath_bpm <= 0:
                breath_bpm = estimate_breath_bpm_time_domain(
                    no_dc, fs=FS_HZ, min_interval_sec=1.0
                )
                if breath_bpm <= 0:
                    breath_bpm, _ = estimate_bpm(
                        breath_signal, FS_HZ, (0.1, 0.8), n_fft=1024,
                        enable_subharmonic_rescue=True,
                    )
                if breath_bpm > 0:
                    self._breath_raw_history.append(breath_bpm)
                    sqi_val = compute_sqi(recent_phase_range, breath_power_ratio, self._current_bin_snr)
                    breath_bpm = apply_smoothing_chain(self._breath_smoother, breath_bpm,
                                                       recent_phase_range, breath_power_ratio,
                                                       self._current_bin_snr)
                    self._last_valid_breath_bpm = breath_bpm

                f0 = breath_bpm / 60.0 if breath_bpm > 0 else 0.0
                heart_bpm_raw, prominence = estimate_bpm(
                    heart_signal, FS_HZ, (0.8, 2.0), f0=f0
                )
                if heart_bpm_raw > 0:
                    if HEART_USE_NEW_SMOOTHER:
                        self._heart_raw_history.append(heart_bpm_raw)
                        heart_bpm = apply_smoothing_chain(self._heart_smoother, heart_bpm_raw,
                                                          recent_phase_range, breath_power_ratio,
                                                          self._current_bin_snr)
                        self._last_valid_heart_bpm = heart_bpm
                    else:
                        self._heart_raw_history.append(heart_bpm_raw)
                        heart_bpm = float(np.median(list(self._heart_raw_history)))
                        self._heart_history.append(heart_bpm)
                        if len(self._heart_history) > 10:
                            self._heart_history = self._heart_history[-10:]
                        heart_bpm = kalman_smooth(self._heart_history, q=1e-3, r=0.5)
                        self._last_valid_heart_bpm = heart_bpm

            self._last_bpm_update = self._frame_count

        quality = self._check_quality(no_dc)

        quality["phase_range"] = phase_range
        quality["breath_ratio"] = breath_power_ratio

        # 信号质量不通过 → BPM 置 0、波形清空，所有模式统一生效
        if not quality.get("valid"):
            breath_bpm = 0.0
            heart_bpm = 0.0
            breath_signal_display = np.array([])
            heart_signal_display = np.array([])

        # 重置 EMA 状态当 BPM 归零时
        if breath_bpm <= 0:
            self._breath_ema = 0.0
        if heart_bpm <= 0:
            self._heart_ema = 0.0

        self.last_heartbeat = time.time()

        return VitalSigns(
            timestamp=time.time(), frame_index=self._frame_count,
            breath_waveform=breath_signal_display,
            breath_bpm=round(breath_bpm, 1),
            heart_bpm=round(heart_bpm, 1),
            heart_waveform=heart_signal_display,
            quality=quality,
        )

    def _check_quality(self, signal: np.ndarray) -> dict:
        from scipy.signal import welch
        phase_range = float(np.max(signal) - np.min(signal))

        if phase_range < 0.005:
            return {"valid": False, "reason": f"phase_range={phase_range:.4f}"}

        freqs, psd = welch(signal, fs=FS_HZ, nperseg=min(128, len(signal)))
        breath_mask = (freqs >= 0.1) & (freqs <= 0.8)
        total_power = np.sum(psd) + 1e-10
        breath_ratio = np.sum(psd[breath_mask]) / total_power

        if breath_ratio < 0.03:
            return {"valid": False, "reason": f"breath_ratio={breath_ratio:.3f}"}

        return {"valid": True, "phase_range": phase_range, "breath_ratio": breath_ratio}

    def _push_to_display(self, vitals: VitalSigns) -> None:
        try:
            self.display_queue.put_nowait(vitals)
        except queue.Full:
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                pass
            self.display_queue.put_nowait(vitals)
