"""DSP 处理管线 — MATLAB PhaseProcess.m + DataProcess.m 移植"""

import threading
import time
import queue
from collections import deque
import numpy as np

from config.protocol import (
    RAW_QUEUE_MAXSIZE, DISPLAY_QUEUE_MAXSIZE,
    WINDOW_SIZE, FS_HZ, BPM_UPDATE_INTERVAL,
)
from models.radar_frame import RadarFrame
from dsp_pipeline.vital_signs import VitalSigns
from dsp_pipeline.range_bin import find_best_range_bin
from dsp_pipeline.phase import extract_phase, unwrap_phase
from dsp_pipeline.filters import remove_dc, VitalSignFilter
from dsp_pipeline.fft_bpm import estimate_bpm, kalman_smooth, estimate_breath_bpm_time_domain, estimate_bpm_stft
from dsp_pipeline.music_angle import estimate_angle_music
from dsp_pipeline.lcmv_beamformer import lcmv_displacement
from dsp_pipeline.emd_cleaner import emd_harmonic_clean
from dsp_pipeline.wpd_filter import wpd_separate
from scipy.signal import sosfiltfilt


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

        self._best_bin: int | None = None
        self._last_bpm_update = 0
        self.last_heartbeat = 0.0

        # Feature toggles
        self._use_beamforming = use_beamforming
        self._use_advanced_dsp: bool = True  # Step 2: EMD + WPD + STFT

        # Beamforming state
        self._angle_deg: float = 0.0  # initial guess: boresight
        self._angle_initialized: bool = False
        self._beamforming_ok: bool = True  # set False on failure -> fallback

        # MATLAB Filter.m: SOS 滤波器组
        self._filter = VitalSignFilter(fs=FS_HZ)

        # Kalman 追踪历史
        self._breath_history: list[float] = []
        self._heart_history: list[float] = []

        # 屏息状态机
        self._apnea_threshold: float | None = 0.005  # 校准前默认值
        self._calibration_samples: list[float] = []
        self._calibration_done: bool = False
        self._in_apnea: bool = False
        self._apnea_start_time: float = 0.0
        self._apnea_hold_breath: float = 0.0
        self._apnea_hold_heart: float = 0.0
        self._recovery_count: int = 0
        self._last_valid_breath_bpm: float = 0.0
        self._last_valid_heart_bpm: float = 0.0

        # 自适应 Kalman: 心率 prominence 历史
        self._heart_prominence_history: list[float] = []

        # Cached advanced-DSP waveforms for display between BPM updates
        self._cached_breath_wave: np.ndarray | None = None
        self._cached_heart_wave: np.ndarray | None = None

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

    def _extract_rx_complex(self, data_cube: np.ndarray) -> np.ndarray:
        """Extract per-RX complex IQ at target range bin with +/-2 bin averaging.

        Matches MATLAB calculateMusicSpectrum range bin smoothing.
        Returns shape [rx_antennas] complex array.
        """
        n_range = data_cube.shape[0]
        start_bin = max(1, self._best_bin - 2)
        end_bin = min(n_range - 1, self._best_bin + 2)
        rx_slice = data_cube[start_bin:end_bin + 1, 0, :]  # [n_bins, rx]
        return np.mean(rx_slice, axis=0)  # [rx]

    def _process_frame(self, frame: RadarFrame) -> VitalSigns | None:
        data_cube = frame.data_cube

        # 1. Range Bin 锁定 (CFAR, 前10帧)
        if self._best_bin is None and self._frame_count < 10:
            self._best_bin = find_best_range_bin(data_cube, fs=FS_HZ)
        if self._best_bin is None:
            return None

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

    def _beamforming_path(self, update_angle: bool) -> np.ndarray | None:
        """Run MUSIC + LCMV on the RX buffer. Returns displacement [200] or None."""
        try:
            rx_matrix = np.array(self._rx_buffer)  # [200, rx]

            # Periodic MUSIC angle update
            if update_angle and not self._angle_initialized:
                try:
                    angle, _ = estimate_angle_music(
                        rx_matrix, FS_HZ, self._filter.sos_all,
                        num_signals=1,
                    )
                    self._angle_deg = angle
                    self._angle_initialized = True
                except Exception:
                    # MUSIC failed; keep current angle guess
                    pass

            # LCMV beamforming (fast enough to run every frame)
            displacement = lcmv_displacement(rx_matrix, self._angle_deg)
            return displacement

        except Exception:
            self._beamforming_ok = False
            return self._fallback_phase_path()

    def _fallback_phase_path(self) -> np.ndarray:
        """Existing simple phase path: unwrap -> detrend.

        Returns displacement-like array for downstream compatibility.
        """
        phase_arr = np.array(self._phase_buffer)
        unwrapped = unwrap_phase(phase_arr)
        return remove_dc(unwrapped)

    def _advanced_dsp_path(
        self, displacement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """EMD -> WPD -> STFT chain. Returns (breath_wave, heart_wave, breath_bpm, heart_bpm).

        Each stage has independent try-catch fallback to existing SOS/FFT behavior.
        """
        # SOS pre-filter (MATLAB: filterObj.apply_all_filter)
        try:
            filted = sosfiltfilt(self._filter.sos_all, displacement)
        except Exception:
            filted = displacement.copy()

        # EMD harmonic cancellation
        try:
            clean = emd_harmonic_clean(filted, FS_HZ)
        except Exception:
            clean = filted

        # WPD separation
        try:
            breath_wave, heart_wave = wpd_separate(clean, FS_HZ)
        except Exception:
            no_dc = remove_dc(clean)
            enhanced = np.diff(no_dc, prepend=no_dc[0])
            breath_wave = self._filter.filter_breath(enhanced)
            heart_wave = self._filter.filter_heart(enhanced)

        # STFT + Kalman BPM estimation
        try:
            breath_bpm, heart_bpm = estimate_bpm_stft(
                breath_wave, heart_wave, FS_HZ
            )
        except Exception:
            breath_bpm, _ = estimate_bpm(
                breath_wave, FS_HZ, (0.1, 0.8), n_fft=1024
            )
            f0 = breath_bpm / 60.0 if breath_bpm > 0 else 0.0
            heart_bpm, _ = estimate_bpm(
                heart_wave, FS_HZ, (1.0, 2.5), f0=f0
            )

        return breath_wave, heart_wave, breath_bpm, heart_bpm

    def _shared_signal_chain(
        self, displacement: np.ndarray | None, update_bpm: bool
    ) -> VitalSigns | None:
        """Common signal processing chain: diff -> SOS -> apnea -> BPM -> quality."""
        if displacement is None:
            return None

        # 一阶差分增强 (MATLAB PhaseProcess.m: diff)
        no_dc = remove_dc(displacement)
        enhanced = np.diff(no_dc, prepend=no_dc[0])

        # SOS 带通滤波 (MATLAB Filter.m)
        breath_signal = self._filter.filter_breath(enhanced)
        heart_signal = self._filter.filter_heart(enhanced)

        # 信号能量指标
        phase_range = float(np.max(enhanced) - np.min(enhanced))
        breath_energy = float(np.var(breath_signal))
        total_energy = float(np.var(enhanced)) + 1e-10
        breath_power_ratio = breath_energy / total_energy

        # 自动校准
        if not self._calibration_done and len(self._phase_buffer) >= WINDOW_SIZE:
            self._calibration_samples.append(phase_range)
            if len(self._calibration_samples) >= 200:
                arr = np.array(self._calibration_samples)
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr))
                self._apnea_threshold = max(0.001, mean_val - 2 * std_val)
                self._calibration_done = True

        # 屏息状态机
        in_low_signal = (
            breath_power_ratio < 0.15
            or phase_range < 0.005
        )
        if self._calibration_done:
            in_low_signal = in_low_signal or (phase_range < self._apnea_threshold)

        if in_low_signal:
            return self._handle_apnea(breath_signal, phase_range, breath_power_ratio)
        elif self._in_apnea:
            self._recovery_count += 1
            if self._recovery_count >= 3:
                self._in_apnea = False
                self._recovery_count = 0

        # BPM 估计
        breath_bpm = 0.0
        heart_bpm = 0.0
        breath_signal_display = breath_signal
        heart_signal_display = heart_signal

        if update_bpm:
            if self._use_advanced_dsp:
                # Step 2: EMD -> WPD -> STFT (full replacement of SOS+BPM)
                try:
                    adv_breath, adv_heart, adv_breath_bpm, adv_heart_bpm = \
                        self._advanced_dsp_path(displacement)
                    breath_signal_display = adv_breath
                    heart_signal_display = adv_heart
                    breath_bpm = adv_breath_bpm
                    heart_bpm = adv_heart_bpm
                    self._cached_breath_wave = adv_breath
                    self._cached_heart_wave = adv_heart
                except Exception:
                    # Fall through to legacy path
                    self._use_advanced_dsp = False

            if not self._use_advanced_dsp or breath_bpm <= 0:
                # Legacy SOS + FFT path (also serves as fallback)
                breath_bpm = estimate_breath_bpm_time_domain(
                    breath_signal, fs=FS_HZ, min_interval_sec=1.5
                )
                if breath_bpm <= 0:
                    breath_bpm, _ = estimate_bpm(
                        breath_signal, FS_HZ, (0.1, 0.8), n_fft=1024
                    )
                if breath_bpm > 0:
                    self._last_valid_breath_bpm = breath_bpm
                    self._breath_history.append(breath_bpm)
                    if len(self._breath_history) > 15:
                        self._breath_history = self._breath_history[-15:]
                    breath_bpm = kalman_smooth(self._breath_history, q=5e-3, r=0.3)

                f0 = breath_bpm / 60.0 if breath_bpm > 0 else 0.0
                heart_bpm_raw, prominence = estimate_bpm(
                    heart_signal, FS_HZ, (0.8, 2.5), f0=f0
                )
                if heart_bpm_raw > 0:
                    self._last_valid_heart_bpm = heart_bpm_raw
                    self._heart_history.append(heart_bpm_raw)
                    self._heart_prominence_history.append(prominence)
                    if len(self._heart_history) > 15:
                        self._heart_history = self._heart_history[-15:]
                    if len(self._heart_prominence_history) > 15:
                        self._heart_prominence_history = self._heart_prominence_history[-15:]
                    heart_bpm = kalman_smooth(
                        self._heart_history, q=1e-3, r=0.5,
                        prominences=self._heart_prominence_history,
                    )

            self._last_bpm_update = self._frame_count

        quality = self._check_quality(enhanced)
        self.last_heartbeat = time.time()

        return VitalSigns(
            timestamp=time.time(), frame_index=self._frame_count,
            breath_waveform=breath_signal_display, breath_bpm=round(breath_bpm, 1),
            heart_bpm=round(heart_bpm, 1), heart_waveform=np.array([]),
            quality=quality,
        )

    def _handle_apnea(
        self, breath_signal: np.ndarray, phase_range: float, breath_power_ratio: float
    ) -> VitalSigns:
        if not self._in_apnea:
            self._in_apnea = True
            self._apnea_start_time = time.time()
            self._apnea_hold_breath = self._last_valid_breath_bpm
            self._apnea_hold_heart = self._last_valid_heart_bpm
        elapsed = time.time() - self._apnea_start_time
        if elapsed < 4.0:
            decay = 1.0 - (elapsed / 4.0)
            breath_bpm = self._apnea_hold_breath * decay
            heart_bpm = self._apnea_hold_heart * decay
        else:
            breath_bpm = 0.0
            heart_bpm = 0.0
        self._recovery_count = 0
        self.last_heartbeat = time.time()
        return VitalSigns(
            timestamp=time.time(), frame_index=0,
            breath_waveform=breath_signal, breath_bpm=round(breath_bpm, 1),
            heart_bpm=round(heart_bpm, 1), heart_waveform=np.array([]),
            quality={
                "valid": True, "reason": "apnea", "phase_range": phase_range,
                "apnea_state": True, "harmonic_overlap": False,
                "heart_prominence": 0.0, "breath_ratio": breath_power_ratio,
            },
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
