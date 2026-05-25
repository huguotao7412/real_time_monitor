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
from dsp_pipeline.fft_bpm import estimate_bpm, kalman_smooth


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

        # 6. 信号质量检查
        quality = self._check_quality(enhanced)
        if not quality["valid"]:
            self.last_heartbeat = time.time()
            return VitalSigns(
                timestamp=time.time(), frame_index=frame.frame_index,
                breath_waveform=breath_signal, breath_bpm=0.0, heart_bpm=0.0,
                heart_waveform=np.array([]), quality=quality,
            )

        # 7. BPM 估计 (FFT, 用更短窗口提高响应速度)
        breath_bpm = 0.0
        heart_bpm = 0.0
        if self._frame_count - self._last_bpm_update >= BPM_UPDATE_INTERVAL:
            # 呼吸: FFT on last 8 seconds (160 samples) — 分辨率 0.125Hz=7.5BPM
            breath_short = breath_signal[-160:] if len(breath_signal) >= 160 else breath_signal
            breath_bpm = estimate_bpm(breath_short, FS_HZ, (0.1, 0.8), n_fft=1024)
            if breath_bpm > 0:
                self._breath_history.append(breath_bpm)
                if len(self._breath_history) > 15:
                    self._breath_history = self._breath_history[-15:]
                breath_bpm = kalman_smooth(self._breath_history, q=5e-3, r=0.3)

            # 心率: FFT on full window — 更好的频率分辨率
            heart_bpm = estimate_bpm(heart_signal, FS_HZ, (0.8, 2.5))
            if heart_bpm > 0:
                self._heart_history.append(heart_bpm)
                if len(self._heart_history) > 15:
                    self._heart_history = self._heart_history[-15:]
                heart_bpm = kalman_smooth(self._heart_history, q=1e-3, r=0.5)

            self._last_bpm_update = self._frame_count

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

        if phase_range > 5.0:
            return {"valid": False, "reason": f"phase_range={phase_range:.2f}"}
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
