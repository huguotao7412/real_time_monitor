"""DSP 处理管线 — MATLAB PhaseProcess.m + DataProcess.m 移植"""

import threading
import time
import queue
from collections import deque
from scipy.signal import sosfiltfilt, savgol_filter
import numpy as np
import concurrent.futures

from config.protocol import (
    RAW_QUEUE_MAXSIZE, DISPLAY_QUEUE_MAXSIZE,
    WINDOW_SIZE, FS_HZ, BPM_UPDATE_INTERVAL,
    BREATH_RAW_HISTORY_MAXLEN, BREATH_HISTORY_MAXLEN,
    BREATH_USE_NEW_SMOOTHER, HEART_USE_NEW_SMOOTHER,
    RANGE_RESOLUTION_M, MIN_VALID_RANGE_BIN, BEAMFORMING_RX_CHANNELS,
    CFAR_ROLLING_BUFFER_SEC, CFAR_INITIAL_SEC, CFAR_RESCAN_SEC, CFAR_SNR_UPDATE_RATIO,
    DSP_STARTUP_SEC, MUSIC_UPDATE_SEC, EMD_MAX_IMF, FFT_N_BREATH, FFT_N_HEART,
    SQI_RECENT_SEC, PHASE_RANGE_MIN_NORMAL, BREATH_RATIO_MIN,
    CFAR_RESCAN_MIN_FRAMES, SAVGOL_WINDOW_LENGTH, SAVGOL_POLYORDER,
    ADAPTIVE_HIGH_PHASE_THRESHOLD, ADAPTIVE_LOW_PHASE_THRESHOLD,
    ADAPTIVE_HIGH_SNR_DB, ADAPTIVE_LOW_SNR_DB, ADAPTIVE_EVAL_INTERVAL,
    WELCH_NPERSEG, HEART_KALMAN_HISTORY_MAXLEN
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
from dsp_pipeline.strategies import (
    SignalCleanerStrategy, VitalSignSeparator,
    VMDRLSCleaner, EMDHarmonicCleaner, PassthroughCleaner,
    WPDSeparator, SOSFilterSeparator,
    AdaptiveStrategySelector,
)
from utils.benchmark_logger import AlgorithmBenchmarker, BenchmarkRecord


class Pipeline:
    _MUSIC_CHANNELS = BEAMFORMING_RX_CHANNELS
    def __init__(
        self,
        use_beamforming: bool = True,
        cleaner: SignalCleanerStrategy | None = None,
        separator: VitalSignSeparator | None = None,
        use_adaptive: bool = True,
    ):
        self.raw_queue = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
        self.display_queue = queue.Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_count = 0

        # Strategy injection — defaults preserve current VMD+RLS → WPD behavior
        self._cleaner: SignalCleanerStrategy = cleaner or VMDRLSCleaner()
        self._separator: VitalSignSeparator = separator or WPDSeparator()
        self._current_algo_name: str = "Adaptive"
        self._current_latency_ms: float = 0.0
        self._current_snr_gain_db: float = 0.0

        # Adaptive routing (replaces _use_advanced_dsp flag)
        self._use_adaptive = use_adaptive
        self._adaptive_selector: AdaptiveStrategySelector | None = (
            AdaptiveStrategySelector(
                light=(PassthroughCleaner(), SOSFilterSeparator()),
                standard=(VMDRLSCleaner(), WPDSeparator()),
                heavy=(EMDHarmonicCleaner(), WPDSeparator()),
                evaluation_interval=ADAPTIVE_EVAL_INTERVAL,
                high_phase_threshold=ADAPTIVE_HIGH_PHASE_THRESHOLD,
                low_phase_threshold=ADAPTIVE_LOW_PHASE_THRESHOLD,
                high_snr_db=ADAPTIVE_HIGH_SNR_DB,
                low_snr_db=ADAPTIVE_LOW_SNR_DB,
            ) if use_adaptive else None
        )

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
        self._cfar_rolling_buffer: deque[np.ndarray] = deque(maxlen=int(FS_HZ * CFAR_ROLLING_BUFFER_SEC))
        self._cfar_initial_frames: int = int(FS_HZ * CFAR_INITIAL_SEC)
        self._cfar_rescan_interval: int = int(FS_HZ * CFAR_RESCAN_SEC)
        self._current_bin_snr: float = 0.0
        self.DISTANCE_PER_BIN: float = RANGE_RESOLUTION_M  # RS6240 range resolution
        self._MIN_RANGE_BIN: int = MIN_VALID_RANGE_BIN  # 跳过近场天线耦合杂波 (bins 1-9 ≈ 2.5-22.5cm)

        # Feature toggles
        self._use_beamforming = use_beamforming

        # Beamforming state
        self._angle_deg: float = 0.0  # initial guess: boresight
        self._angle_initialized: bool = False
        self._beamforming_ok: bool = True  # set False on failure -> fallback
        self._last_music_update: int = -50
        self._music_update_interval: int = int(FS_HZ * MUSIC_UPDATE_SEC)

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
        self._heart_prominence_history: list[float] = []

        # Cached advanced-DSP waveforms for display between BPM updates
        self._cached_breath_wave: np.ndarray | None = None
        self._cached_heart_wave: np.ndarray | None = None

        # Phase unwrapping continuity state (prevents 2π jumps across sliding windows)
        self._last_unwrapped_phase: float | None = None

        # ── Async DSP + A/B ──
        self._dsp_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._dsp_future: concurrent.futures.Future | None = None
        self._ab_enabled: bool = False
        self._ab_cleaner: SignalCleanerStrategy | None = None
        self._ab_separator: VitalSignSeparator | None = None
        self._ab_future: concurrent.futures.Future | None = None
        self._ab_algo_name: str = ""
        self._ab_latency_ms: float = 0.0
        self._ab_snr_gain_db: float = 0.0
        self._pending_displacement = None

        # ── Benchmarker ──
        self._benchmarker: AlgorithmBenchmarker | None = None

        # 弱信号计数 (用于 Range Bin 重捕获)
        self._low_signal_frame_count: int = 0
        self._last_valid_breath_bpm: float = 0.0
        self._last_valid_heart_bpm: float = 0.0

        # ========== 【新增：跨距离元相位补偿量】 ==========
        # 用于在目标移动发生 Range Bin 切换时，无缝对齐历史波形的基准线
        self._cross_bin_phase_offset: float = 0.0

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
        if len(full_rx) >= max(self._MUSIC_CHANNELS) + 1:
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
                return VitalSigns(
                    timestamp=time.time(),
                    frame_index=self._frame_count,
                    breath_waveform=np.array([]),
                    breath_bpm=0.0,
                    heart_bpm=0.0,
                    heart_waveform=np.array([]),
                    quality={"valid": False, "reason": "Searching Target"}
                )
        elif self._frame_count > 0 and self._frame_count % self._cfar_rescan_interval == 0:
            new_bin, new_snr, current_actual_snr = self._run_2d_cfar_rescan()
            if new_bin is not None:
                MIN_NEW_TARGET_SNR = 12.0
                if new_snr > MIN_NEW_TARGET_SNR and (
                        current_actual_snr < 5.0 or new_snr > current_actual_snr * CFAR_SNR_UPDATE_RATIO):
                    print(f"[DSP] Target moved! Range updated to bin: {new_bin} ...")

                    # ========== 【核心修改：计算跨 Bin 瞬时相位差】 ==========
                    if self._best_bin is not None:
                        # 提取在切换这一帧时，旧 Bin 和新 Bin 的空间绝对相位
                        old_phase_inst = extract_phase(data_cube, self._best_bin)
                        new_phase_inst = extract_phase(data_cube, new_bin)

                        # 将这个阶跃差值累加到全局补偿量中
                        self._cross_bin_phase_offset += (old_phase_inst - new_phase_inst)
                    # ========================================================

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
        raw_phase = extract_phase(data_cube, self._best_bin)
        compensated_phase = raw_phase + self._cross_bin_phase_offset
        self._phase_buffer.append(compensated_phase)
        self._frame_count += 1

        MIN_STARTUP_FRAMES = int(FS_HZ * DSP_STARTUP_SEC)
        if len(self._phase_buffer) < MIN_STARTUP_FRAMES:
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
        # Filter: skip near-field bins (antenna coupling / enclosure reflections)
        if len(confirmed) > 0:
            valid_mask = confirmed[:, 0] >= self._MIN_RANGE_BIN
            confirmed = confirmed[valid_mask]
        if len(confirmed) > 0:
            best_idx = np.argmin(confirmed[:, 0])  # closest valid bin
            best_bin = float(confirmed[best_idx, 0])
            snr = float(confirmed[best_idx, 2])
            return best_bin, snr
        # Fallback: 1D CFAR only (also skip near-field bins)
        valid_candidates = [c for c in candidates if c >= self._MIN_RANGE_BIN] if len(candidates) > 0 else []
        if len(valid_candidates) > 0:
            return float(valid_candidates[0]), 0.0
        # Ultimate fallback
        best_bin = find_best_range_bin(mean_bin_frame_rx, fs=FS_HZ)
        return best_bin, 0.0

    def _run_2d_cfar_rescan(self) -> tuple[float | None, float, float]:
        """Periodic re-scan using rolling buffer.

        Returns (new_bin, new_snr, current_actual_snr). current_actual_snr is the
        SNR of self._best_bin measured from the rolling buffer data at this moment.
        """
        cubes = list(self._cfar_rolling_buffer)
        if len(cubes) < CFAR_RESCAN_MIN_FRAMES:
            return None, 0.0, 0.0
        mean_bin_frame_rx = self._build_mean_bin_frame_rx(cubes)
        candidates = coarse_1d_cfar_candidates(mean_bin_frame_rx)
        _, _, debug, _ = adaptive_2d_cfar(
            mean_bin_frame_rx, self.DISTANCE_PER_BIN, self._cfar_state, candidates
        )
        confirmed = debug.get("confirmed_list", np.array([]))

        # Filter: skip near-field bins (antenna coupling / enclosure reflections)
        if len(confirmed) > 0:
            valid_mask = confirmed[:, 0] >= self._MIN_RANGE_BIN
            confirmed = confirmed[valid_mask]

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
            offset = self._last_unwrapped_phase - unwrapped[-2]
            n2pi = np.round(offset / (2 * np.pi)) * (2 * np.pi)
            unwrapped = unwrapped + n2pi

        # 物理限制：人类胸腔最大运动速度阈值判定。突变超出阈值视为体动干扰
        diff_phase = np.abs(np.diff(unwrapped))
        MA_THRESHOLD = 0.8  # 可视实际波长调节（rad/frame），24GHz或77GHz表现不同

        raw_last_phase = unwrapped[-1]

        if np.any(diff_phase > MA_THRESHOLD):
            # 引入快速 Hampel 滤波（局部中值异常点剔除）
            window_size = 5
            pad_width = window_size // 2
            padded = np.pad(unwrapped, pad_width, mode='edge')

            # 使用 numpy strides 加速滑动窗口中值计算
            shape = (len(unwrapped), window_size)
            strides = (padded.strides[0], padded.strides[0])
            windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

            local_medians = np.median(windows, axis=1)
            local_mads = np.median(np.abs(windows - local_medians[:, None]), axis=1)

            # 识别异常点并用局部中值替换
            outliers = np.abs(unwrapped - local_medians) > (3 * local_mads)
            unwrapped[outliers] = local_medians[outliers]

        self._last_unwrapped_phase = raw_last_phase
        return remove_dc(unwrapped)

    def _advanced_dsp_path(
        self, displacement: np.ndarray,
        cleaner: SignalCleanerStrategy | None = None,
        separator: VitalSignSeparator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, dict]:
        """SOS → Cleaner → Separator → BPM. Returns 6-tuple with metrics.

        Strategy objects handle their own errors internally — no try/except needed.
        """
        use_cleaner = cleaner if cleaner is not None else self._cleaner
        use_sep = separator if separator is not None else self._separator

        # SOS pre-filter
        try:
            filted = sosfiltfilt(self._filter.sos_all, displacement)
        except Exception:
            filted = displacement.copy()

        # Cleaner (harmonic removal)
        clean_disp, clean_metrics = use_cleaner.clean(filted, FS_HZ)

        # Separator (breath / heart)
        heart_diff = np.diff(clean_disp)
        breath_wave, heart_wave, sep_metrics = use_sep.separate(
            clean_disp, FS_HZ, heart_input_signal=heart_diff,
        )

        # BPM estimation
        heart_prominence = 0.1
        try:
            breath_bpm, heart_bpm, heart_prominence = estimate_bpm_stft(
                breath_wave, heart_wave, FS_HZ, FFT_N_HEART,
                raw_displacement=displacement,
                breath_waveform=breath_wave,
            )
        except Exception:
            breath_bpm, _ = estimate_bpm(
                breath_wave, FS_HZ, (0.1, 0.8), n_fft=FFT_N_BREATH,
                enable_subharmonic_rescue=True,
            )
            if breath_bpm <= 0:
                breath_bpm = estimate_breath_bpm_time_domain(breath_wave, FS_HZ)
            f0 = breath_bpm / 60.0 if breath_bpm > 0 else 0.0
            heart_bpm, heart_prominence = estimate_bpm(
                heart_wave, FS_HZ, (0.8, 2.0), f0=f0,
            )

        # Combine metrics
        combined = {
            **clean_metrics,
            **sep_metrics,
            "breath_bpm": breath_bpm,
            "heart_bpm": heart_bpm,
            "heart_prominence": heart_prominence,
        }
        return breath_wave, heart_wave, breath_bpm, heart_bpm, heart_prominence, combined

    def _shared_signal_chain(
            self, displacement: np.ndarray | None, update_bpm: bool
    ) -> VitalSigns | None:
        """Common signal processing chain: diff -> SOS -> apnea -> BPM -> quality."""
        if displacement is None:
            return None

        no_dc = remove_dc(displacement)
        enhanced = savgol_filter(no_dc, window_length=SAVGOL_WINDOW_LENGTH,
                                 polyorder=SAVGOL_POLYORDER, deriv=1)

        # SOS 带通滤波: 呼吸用原始位移防振铃分裂, 心跳用差分放大高频脉冲
        breath_signal = self._filter.filter_breath(no_dc)
        heart_signal = self._filter.filter_heart(enhanced)

        # 信号能量指标
        phase_range = float(np.max(no_dc) - np.min(no_dc))
        breath_energy = float(np.var(breath_signal))
        total_energy = float(np.var(enhanced)) + 1e-10
        breath_power_ratio = breath_energy / total_energy

        # 提取最近 30 帧 (1.5秒) 短时能量，避免 10 秒窗口导致的响应迟钝
        recent_frames = int(FS_HZ * SQI_RECENT_SEC)
        if len(no_dc) >= recent_frames:
            recent_no_dc = no_dc[-recent_frames:]
            recent_phase_range = float(np.max(recent_no_dc) - np.min(recent_no_dc))
        else:
            recent_phase_range = phase_range

        # 弱信号检测 → Range Bin 重捕获
        SNR_THRESHOLD_EMPTY = 10.0  # 根据实际雷达表现微调经验值
        in_low_signal = (recent_phase_range < PHASE_RANGE_MIN_NORMAL) and (self._current_bin_snr < SNR_THRESHOLD_EMPTY)

        if in_low_signal:
            self._low_signal_frame_count += 1
        else:
            self._low_signal_frame_count = 0

        if self._low_signal_frame_count >= int(FS_HZ * SQI_RECENT_SEC) and self._best_bin is not None:
            print("[DSP] Target lost! Resetting buffers...")
            self._best_bin = None
            self._phase_buffer.clear()
            self._rx_buffer.clear()
            self._last_unwrapped_phase = None
            self._low_signal_frame_count = 0
            self._cross_bin_phase_offset = 0.0

        # BPM 估计
        breath_bpm = self._last_valid_breath_bpm
        heart_bpm = self._last_valid_heart_bpm
        breath_signal_display = breath_signal
        heart_signal_display = heart_signal

        if update_bpm:
            # ── Run AdaptiveSelector if enabled ──
            if self._use_adaptive and self._adaptive_selector is not None:
                self._cleaner, self._separator = self._adaptive_selector.select(
                    phase_range=recent_phase_range,
                    snr_db=self._current_bin_snr,
                )
                cname = self._cleaner.clean(np.array([0.0]), FS_HZ)[1].get("algorithm", "?")
                sname = self._separator.separate(np.array([0.0, 0.1]), FS_HZ)[2].get("algorithm", "?")
                self._current_algo_name = f"{cname}+{sname}"

            # ── Primary strategy: check async result then dispatch new ──
            if self._dsp_future is not None and self._dsp_future.done():
                try:
                    (adv_breath, adv_heart, adv_breath_bpm, adv_heart_bpm,
                     adv_heart_prom, adv_metrics) = self._dsp_future.result()

                    breath_signal_display = adv_breath
                    heart_signal_display = adv_heart
                    breath_bpm = adv_breath_bpm
                    heart_bpm_raw = adv_heart_bpm

                    self._current_latency_ms = adv_metrics.get("latency_ms", 0.0)
                    self._current_snr_gain_db = adv_metrics.get("snr_estimate_db", 0.0) or 0.0

                    if breath_bpm > 0:
                        self._breath_raw_history.append(breath_bpm)
                        sqi_val = compute_sqi(recent_phase_range, breath_power_ratio, self._current_bin_snr)
                        breath_bpm = apply_smoothing_chain(self._breath_smoother, breath_bpm,
                                                           recent_phase_range, breath_power_ratio,
                                                           self._current_bin_snr)
                        self._last_valid_breath_bpm = breath_bpm

                    if heart_bpm_raw > 0:
                        self._heart_prominence_history.append(adv_heart_prom)
                        if len(self._heart_prominence_history) > HEART_KALMAN_HISTORY_MAXLEN * 2:
                            self._heart_prominence_history = self._heart_prominence_history[
                                -HEART_KALMAN_HISTORY_MAXLEN * 2:]

                        heart_bpm_accepted = True
                        if self._last_valid_heart_bpm > 0:
                            bpm_jump = abs(heart_bpm_raw - self._last_valid_heart_bpm)
                            if bpm_jump > 20.0 and adv_heart_prom < 0.3:
                                heart_bpm_accepted = False

                        if heart_bpm_accepted:
                            if HEART_USE_NEW_SMOOTHER:
                                self._heart_raw_history.append(heart_bpm_raw)
                                heart_bpm = apply_smoothing_chain(self._heart_smoother, heart_bpm_raw,
                                                                  recent_phase_range, breath_power_ratio,
                                                                  self._current_bin_snr)
                                self._last_valid_heart_bpm = heart_bpm
                            else:
                                self._heart_raw_history.append(heart_bpm_raw)
                                heart_bpm_raw_median = float(np.median(list(self._heart_raw_history)))
                                self._heart_history.append(heart_bpm_raw_median)
                                if len(self._heart_history) > HEART_KALMAN_HISTORY_MAXLEN:
                                    self._heart_history = self._heart_history[-HEART_KALMAN_HISTORY_MAXLEN:]
                                prom_slice = self._heart_prominence_history[-len(self._heart_history):]
                                heart_bpm = kalman_smooth(
                                    self._heart_history, q=1e-3, r=0.5,
                                    prominences=prom_slice)
                                self._last_valid_heart_bpm = heart_bpm
                        else:
                            heart_bpm = self._last_valid_heart_bpm

                    self._cached_breath_wave = adv_breath
                    self._cached_heart_wave = adv_heart

                    # ── Benchmark log (primary) ──
                    self._log_benchmark(adv_metrics, displacement, phase_range, is_primary=True)

                except Exception as e:
                    print(f"[DSP] Async advanced path failed: {e}")

            # Dispatch new primary task if idle
            if self._dsp_future is None or self._dsp_future.done():
                self._dsp_future = self._dsp_executor.submit(
                    self._advanced_dsp_path, displacement.copy(),
                    self._cleaner, self._separator,
                )

            # ── A/B comparison: check result then dispatch ──
            if self._ab_enabled:
                if self._ab_future is not None and self._ab_future.done():
                    try:
                        (_, _, _, _, _, ab_metrics) = self._ab_future.result()
                        self._ab_algo_name = ab_metrics.get("algorithm", "?")
                        self._ab_latency_ms = ab_metrics.get("latency_ms", 0.0)
                        self._ab_snr_gain_db = ab_metrics.get("snr_estimate_db", 0.0) or 0.0
                        self._log_benchmark(ab_metrics, displacement, phase_range, is_primary=False)
                    except Exception:
                        pass

                if self._ab_future is None or self._ab_future.done():
                    if self._ab_cleaner is not None and self._ab_separator is not None:
                        self._ab_future = self._dsp_executor.submit(
                            self._advanced_dsp_path, displacement.copy(),
                            self._ab_cleaner, self._ab_separator,
                        )

            # Maintain last valid BPMs while computing
            breath_bpm = self._last_valid_breath_bpm
            heart_bpm = self._last_valid_heart_bpm
            if self._cached_breath_wave is not None:
                breath_signal_display = self._cached_breath_wave
                heart_signal_display = self._cached_heart_wave

            # ── Fallback baseline path (when adaptive off and BPM invalid) ──
            if not self._use_adaptive and breath_bpm <= 0:
                breath_bpm = estimate_breath_bpm_time_domain(
                    no_dc, fs=FS_HZ, min_interval_sec=1.0,
                )
                if breath_bpm <= 0:
                    breath_bpm, _ = estimate_bpm(
                        breath_signal, FS_HZ, (0.1, 0.8), n_fft=FFT_N_HEART,
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
                    heart_signal, FS_HZ, (0.8, 2.0), f0=f0,
                )
                if heart_bpm_raw > 0:
                    self._heart_prominence_history.append(prominence)
                    if len(self._heart_prominence_history) > HEART_KALMAN_HISTORY_MAXLEN * 2:
                        self._heart_prominence_history = self._heart_prominence_history[
                            -HEART_KALMAN_HISTORY_MAXLEN * 2:]

                    heart_bpm_accepted = True
                    if self._last_valid_heart_bpm > 0:
                        bpm_jump = abs(heart_bpm_raw - self._last_valid_heart_bpm)
                        if bpm_jump > 20.0 and prominence < 0.3:
                            heart_bpm_accepted = False

                    if heart_bpm_accepted:
                        if HEART_USE_NEW_SMOOTHER:
                            self._heart_raw_history.append(heart_bpm_raw)
                            heart_bpm = apply_smoothing_chain(self._heart_smoother, heart_bpm_raw,
                                                              recent_phase_range, breath_power_ratio,
                                                              self._current_bin_snr)
                            self._last_valid_heart_bpm = heart_bpm
                        else:
                            self._heart_raw_history.append(heart_bpm_raw)
                            heart_bpm_raw_median = float(np.median(list(self._heart_raw_history)))
                            self._heart_history.append(heart_bpm_raw_median)
                            if len(self._heart_history) > HEART_KALMAN_HISTORY_MAXLEN:
                                self._heart_history = self._heart_history[-HEART_KALMAN_HISTORY_MAXLEN:]
                            prom_slice = self._heart_prominence_history[-len(self._heart_history):]
                            heart_bpm = kalman_smooth(
                                self._heart_history, q=1e-3, r=0.5,
                                prominences=prom_slice,
                            )
                            self._last_valid_heart_bpm = heart_bpm
                    else:
                        heart_bpm = self._last_valid_heart_bpm

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

    # ── Strategy hot-swap API ──────────────────────────────────

    def set_strategies(
        self, cleaner: SignalCleanerStrategy, separator: VitalSignSeparator
    ) -> None:
        """Hot-swap primary strategies. Takes effect on next update_bpm dispatch."""
        self._cleaner = cleaner
        self._separator = separator
        self._use_adaptive = False  # manual override disables adaptive
        cname = cleaner.clean(np.array([0.0]), FS_HZ)[1].get("algorithm", "?")
        sname = separator.separate(np.array([0.0, 0.1]), FS_HZ)[2].get("algorithm", "?")
        self._current_algo_name = f"{cname}+{sname}"

    def set_ab_strategy(
        self,
        cleaner: SignalCleanerStrategy | None,
        separator: VitalSignSeparator | None,
    ) -> None:
        """Set or disable A/B comparison strategy. None disables."""
        self._ab_cleaner = cleaner
        self._ab_separator = separator
        self._ab_enabled = (cleaner is not None and separator is not None)

    @property
    def benchmarker(self) -> AlgorithmBenchmarker | None:
        return self._benchmarker

    def set_benchmarker(self, benchmarker: AlgorithmBenchmarker | None) -> None:
        self._benchmarker = benchmarker

    def _log_benchmark(
        self, metrics: dict, displacement: np.ndarray, phase_range_val: float,
        is_primary: bool,
    ) -> None:
        """Build and enqueue a BenchmarkRecord if recording is active."""
        if self._benchmarker is None or not self._benchmarker.is_recording:
            return
        try:
            input_pr = float(np.max(displacement) - np.min(displacement))
        except Exception:
            input_pr = 0.0
        record = BenchmarkRecord(
            timestamp=time.time(),
            frame_index=self._frame_count,
            elapsed_sec=self._frame_count / FS_HZ,
            algorithm_name=metrics.get("algorithm",
                         self._current_algo_name if is_primary else self._ab_algo_name),
            is_primary=is_primary,
            latency_ms=metrics.get("latency_ms", 0.0),
            input_phase_range=input_pr,
            input_snr_db=metrics.get("snr_estimate_db"),
            output_phase_range=phase_range_val,
            output_snr_db=None,
            snr_gain_db=None,
            breath_bpm=metrics.get("breath_bpm", 0.0),
            heart_bpm=metrics.get("heart_bpm", 0.0),
            heart_prominence=metrics.get("heart_prominence", 0.0),
            imf_count=metrics.get("imf_count"),
            convergence_iter=metrics.get("convergence_iter"),
            retained_harmonics=(
                ",".join(str(h) for h in metrics["retained_harmonics"])
                if metrics.get("retained_harmonics") else None
            ),
            dominant_freq_hz=metrics.get("dominant_freq_hz"),
        )
        self._benchmarker.log(record)

    def _check_quality(self, signal: np.ndarray) -> dict:
        from scipy.signal import welch
        phase_range = float(np.max(signal) - np.min(signal))

        if phase_range < PHASE_RANGE_MIN_NORMAL:
            return {"valid": False, "reason": f"phase_range={phase_range:.4f}"}

        freqs, psd = welch(signal, fs=FS_HZ, nperseg=min(WELCH_NPERSEG, len(signal)))
        breath_mask = (freqs >= 0.1) & (freqs <= 0.8)
        total_power = np.sum(psd) + 1e-10
        breath_ratio = np.sum(psd[breath_mask]) / total_power

        if breath_ratio < BREATH_RATIO_MIN:
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
