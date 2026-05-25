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
from dsp_pipeline.fft_bpm import estimate_bpm, kalman_smooth, estimate_breath_bpm_time_domain


class Pipeline:
    def __init__(self):
        self.raw_queue = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
        self.display_queue = queue.Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._phase_buffer: deque[float] = deque(maxlen=WINDOW_SIZE)
        self._best_bin: int | None = None
        self._last_bpm_update = 0
        self.last_heartbeat = 0.0

        # MATLAB Filter.m: SOS 滤波器组
        self._filter = VitalSignFilter(fs=FS_HZ)

        # Kalman 追踪历史
        self._breath_history: list[float] = []
        self._heart_history: list[float] = []

        # 屏息状态机
        self._apnea_threshold: float | None = 0.003  # 校准前默认值
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

    def _process_frame(self, frame: RadarFrame) -> VitalSigns | None:
        data_cube = frame.data_cube

        # 1. Range Bin 锁定 (CFAR, 前10帧)
        if self._best_bin is None and self._frame_count < 10:
            self._best_bin = find_best_range_bin(data_cube, fs=FS_HZ)
        if self._best_bin is None:
            return None

        # 2. 相位提取
        phase = extract_phase(data_cube, self._best_bin)
        self._phase_buffer.append(phase)
        self._frame_count += 1

        if len(self._phase_buffer) < WINDOW_SIZE:
            return None

        phase_arr = np.array(self._phase_buffer)

        # 3. 解缠 + 去直流
        unwrapped = unwrap_phase(phase_arr)
        no_dc = remove_dc(unwrapped)

        # 4. 一阶差分增强 (MATLAB PhaseProcess.m: diff)
        enhanced = np.diff(no_dc, prepend=no_dc[0])

        # 5. SOS 带通滤波 (MATLAB Filter.m)
        breath_signal = self._filter.filter_breath(enhanced)
        heart_signal = self._filter.filter_heart(enhanced)

        # 6. 计算 phase_range (用于自动校准和屏息检测)
        phase_range = float(np.max(enhanced) - np.min(enhanced))

        # 7. 自动校准 (系统启动后在前200个完整窗口中收集样本)
        if not self._calibration_done and len(self._phase_buffer) >= WINDOW_SIZE:
            self._calibration_samples.append(phase_range)
            if len(self._calibration_samples) >= 200:
                arr = np.array(self._calibration_samples)
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr))
                self._apnea_threshold = max(0.001, mean_val - 2 * std_val)
                self._calibration_done = True

        # 8. 屏息状态机
        if self._calibration_done and phase_range < self._apnea_threshold:
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
                timestamp=time.time(), frame_index=frame.frame_index,
                breath_waveform=breath_signal, breath_bpm=round(breath_bpm, 1),
                heart_bpm=round(heart_bpm, 1), heart_waveform=np.array([]),
                quality={
                    "valid": True, "reason": "apnea", "phase_range": phase_range,
                    "apnea_state": True, "harmonic_overlap": False,
                    "heart_prominence": 0.0, "breath_ratio": 0.0,
                },
            )
        elif self._in_apnea:
            self._recovery_count += 1
            if self._recovery_count >= 3:
                self._in_apnea = False
                self._recovery_count = 0

        # 9. BPM 估计
        breath_bpm = 0.0
        heart_bpm = 0.0
        if self._frame_count - self._last_bpm_update >= BPM_UPDATE_INTERVAL:
            # 呼吸: 时域峰值检测 (替代原有 FFT 方法)
            breath_bpm = estimate_breath_bpm_time_domain(breath_signal, fs=FS_HZ)
            if breath_bpm > 0:
                self._last_valid_breath_bpm = breath_bpm
                self._breath_history.append(breath_bpm)
                if len(self._breath_history) > 15:
                    self._breath_history = self._breath_history[-15:]
                breath_bpm = kalman_smooth(self._breath_history, q=5e-3, r=0.3)

            # 心率: FFT + 谐波掩码 + 自适应 Kalman
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

        # 10. 质量评估
        quality = self._check_quality(enhanced)

        self.last_heartbeat = time.time()
        return VitalSigns(
            timestamp=time.time(), frame_index=frame.frame_index,
            breath_waveform=breath_signal, breath_bpm=breath_bpm,
            heart_bpm=heart_bpm, heart_waveform=np.array([]),
            quality=quality,
        )

    def _check_quality(self, signal: np.ndarray) -> dict:
        from scipy.signal import welch
        phase_range = float(np.max(signal) - np.min(signal))

        if phase_range < 0.003:
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
