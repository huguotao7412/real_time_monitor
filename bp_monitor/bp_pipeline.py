"""Blood pressure monitoring pipeline — strictly ported from bp_matlab/main.m.

MATLAB processing chain (preserved exactly):
  1. Accumulate 1024 frames of complex 1D-FFT data  →  [32, 1024, 1]
  2. Background subtraction: subtract temporal mean per bin
  3. 1D CFAR on range energy profile
  4. 2D CFAR on Range-Doppler Map with adaptive beta
  5. Phase extraction at target bin → angle → unwrap along time
  6. Frequency scaling: × (24/60)  (60GHz → 24GHz mapping)
  7. EMD harmonic removal + sym8 wavelet denoising → ×0.5
  8. Downsample 200→50Hz (1024→256 points)
  9. Global min-max normalize → network inference → denormalize → mmHg
  10. Peak/valley detection → SBP/DBP

Usage:
    pipeline = BPPipeline("bp_matlab/bp_weights.mat")
    pipeline.start()
    pipeline.raw_queue.put(radar_frame)   # feed per-frame
    result = pipeline.display_queue.get()  # read BPResult
    pipeline.stop()
"""

import threading
import time
import queue
from collections import deque

import numpy as np
from scipy.signal import resample_poly

from config.protocol import (RAW_QUEUE_MAXSIZE, DISPLAY_QUEUE_MAXSIZE, RANGE_HARDWARE_OFFSET_M,
                              RADAR_BP_FPS, DSP_BP_TARGET_FS,RANGE_RESOLUTION_M,
                              BP_BATCH_SEC, BP_STEP_SEC, BP_NETWORK_INPUT_LEN,
                              BP_CFAR_INITIAL_FRAMES, BP_CFAR_INTERVAL, BP_CFAR_FALLBACK_FRAMES, BP_COLD_START_FRAMES,
                              FREQ_SCALE_60G_TO_24G, PHASE_RANGE_MIN_BP, BP_MAX_BAD_SIGNAL_COUNT, MIN_REAL_DISTANCE_M
                             )
from models.radar_frame import RadarFrame
from bp_monitor.bp_models import BPResult
from bp_monitor.bp_cfar import find_target_bins_1d, adaptive_2d_cfar
from bp_monitor.bp_signal_cleaner import clean_pulse_wave
from bp_monitor.bp_network import BPInference
from bp_monitor.bp_postprocess import extract_bp


class BPPipeline:
    """Blood pressure processing pipeline — MATLAB main.m strict port.

    Runs in a dedicated daemon thread.  Accumulates 1024 complex frames
    (~5.12 s at 200 Hz), then runs the full MATLAB processing chain:
    background subtraction → 1D+2D CFAR → phase extraction → cleaning →
    network inference → SBP/DBP.
    """

    FS = float(RADAR_BP_FPS)
    FS_TARGET = float(DSP_BP_TARGET_FS)

    # 物理时间联动：原来的 1024 是基于 5.12 秒 * 200Hz 得来的
    MAX_FRAMES = int(BP_BATCH_SEC * FS)

    # 神经网络固定输入维度 (这个不随采样率变化，是由 bp_weights.mat 模型决定的)
    N_INPUT = BP_NETWORK_INPUT_LEN

    # 滑动窗口步长：原来的 100 是基于 0.5 秒 * 200Hz 得来的
    STEP_FRAMES = int(BP_STEP_SEC * FS)

    # 距离分辨率：直接引用配置文件的全局变量
    DISTANCE_PER_BIN = RANGE_RESOLUTION_M

    def __init__(self, weights_path: str = "bp_matlab/bp_weights.mat"):
        self.raw_queue: queue.Queue[RadarFrame] = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
        self.display_queue: queue.Queue[BPResult] = queue.Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)

        self._weights_path = weights_path
        self._bp: BPInference | None = None

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Internal state
        self._buffer = np.zeros((32, self.MAX_FRAMES, 1), dtype=complex)
        self._valid_frames = 0  # list of [32, 1, 1] complex arrays
        self._frame_count = 0
        self._target_bin: int | None = None
        self._cfar_state: dict | None = None   # persistent CFAR state (adaptive beta)

        # Temporal smoothing: sliding windows for SBP/DBP (median → EMA)
        self._sbp_history: deque[float] = deque(maxlen=10)
        self._dbp_history: deque[float] = deque(maxlen=10)
        self._sbp_ema: float | None = None
        self._dbp_ema: float | None = None
        self._bad_signal_count: int = 0

    # -- public API ---------------------------------------------------------

    @property
    def target_bin(self) -> int | None:
        return self._target_bin

    def start(self) -> None:
        if self._bp is None:
            self._bp = BPInference(self._weights_path)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    # -- internal -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame: RadarFrame = self.raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_frame(frame)
            except Exception:
                import traceback
                print(f"[BPPipeline] Error in frame {self._frame_count}:")
                traceback.print_exc()

    def _process_frame(self, frame: RadarFrame) -> None:
        data_cube = frame.data_cube  # [32, 1, 1] = [bins, doppler=1, rx=1]

        # ==========================================
        # 1. 数组环形存入逻辑（替代 list.append）
        # ==========================================
        if self._valid_frames < self.MAX_FRAMES:
            # 填满初始阶段
            self._buffer[:, self._valid_frames:self._valid_frames + 1, :] = data_cube
            self._valid_frames += 1
        else:
            # 队列满后，整体左移 (利用切片赋值替代频繁的内存分配)
            self._buffer[:, :-1, :] = self._buffer[:, 1:, :]
            self._buffer[:, -1:, :] = data_cube

        self._frame_count += 1

        # ==========================================
        # 2. 获取当前有效数据（零拷贝/引用，替代 np.concatenate）
        # ==========================================
        current_data = self._buffer[:, :self._valid_frames, :]
        n = self._valid_frames  # 替代 len(self._complex_buffer)

        # ---- Phase 1: CFAR lock ----
        if self._target_bin is None:
            # Try CFAR every 16 frames starting from 64 frames
            if n >= BP_CFAR_INITIAL_FRAMES and n % BP_CFAR_INTERVAL == 0:
                acc = current_data
                acc_bg = acc - np.mean(acc, axis=1, keepdims=True)
                candidates = find_target_bins_1d(
                    acc_bg, self.DISTANCE_PER_BIN, num_targets=1
                )
                if len(candidates) > 0:
                    self._target_bin = int(candidates[0])
                    real_dist = max(MIN_REAL_DISTANCE_M, self._target_bin * self.DISTANCE_PER_BIN - RANGE_HARDWARE_OFFSET_M)
                    print(f"[BPPipeline] Target locked: bin={self._target_bin} ({real_dist:.2f}m)")

            # Fallback after 256 frames: pick strongest bin
            if self._target_bin is None and n >= BP_CFAR_FALLBACK_FRAMES:
                acc = current_data
                acc_bg = acc - np.mean(acc, axis=1, keepdims=True)
                energy = np.mean(np.abs(acc_bg), axis=(1, 2))
                energy[:2] = 0  # skip near-field DC
                self._target_bin = int(np.argmax(energy))
                real_dist = max(MIN_REAL_DISTANCE_M, self._target_bin * self.DISTANCE_PER_BIN - RANGE_HARDWARE_OFFSET_M)
                print(f"[BPPipeline] Fallback lock: bin={self._target_bin} ({real_dist:.2f}m)")

            if self._target_bin is None:
                return  # keep accumulating

        # ---- Phase 2: Wait for full batch ----
        is_cold_start = (self._sbp_ema is None)
        required_frames = BP_COLD_START_FRAMES if is_cold_start else self.MAX_FRAMES
        if self._valid_frames < required_frames:
            return

        # ---- Full MATLAB pipeline on 1024 frames ----
        print(f"[BPPipeline] Processing batch at frame {self._frame_count}...")

        # 拷贝一份处理数据，防止后续背景相减污染原 buffer
        mean_bin_frame_rx = current_data.copy()

        # MATLAB: background = mean(bin_frame_rx, 2); mean_bin_frame_rx = bin_frame_rx - background
        background = np.mean(mean_bin_frame_rx, axis=1, keepdims=True)  # [32, 1, 1]
        mean_bin_frame_rx = mean_bin_frame_rx - background

        # ---- Step 1+2: 1D CFAR + 2D CFAR ----
        overall_target_bins = find_target_bins_1d(
            mean_bin_frame_rx, self.DISTANCE_PER_BIN, num_targets=3, verbose=True
        )
        if len(overall_target_bins) == 0:
            print("[BPPipeline] CFAR: no target found, re-acquiring...")
            self._target_bin = None
            self._valid_frames = 0  # 替代 self._complex_buffer.clear()
            return

        target_bins, self._cfar_state = adaptive_2d_cfar(
            mean_bin_frame_rx, overall_target_bins, self._cfar_state
        )
        if len(target_bins) == 0:
            print("[BPPipeline] 2D CFAR: no target confirmed, re-acquiring...")
            self._target_bin = None
            self._valid_frames = 0  # 替代 self._complex_buffer.clear()
            return

        print(f"[BPPipeline] 2D CFAR confirmed {len(target_bins)} target(s): bins={list(target_bins)}")
        target_bin = int(target_bins[0])

        if target_bin != self._target_bin:
            print(f"[BPPipeline] Target bin updated: {self._target_bin} → {target_bin}")
            self._target_bin = target_bin

        # ---- Step 3: Phase extraction ----
        complex_data = mean_bin_frame_rx[target_bin, :, :]  # [N, 1]
        phase_data = np.angle(complex_data)
        unwrapped = np.unwrap(phase_data, axis=0)
        unwrapped = unwrapped.squeeze()

        # ---- Step 4: Frequency scaling ----
        unwrapped_scaled = unwrapped * FREQ_SCALE_60G_TO_24G

        # ---- Step 4b: Low-signal detection → re-acquire ----
        phase_range = float(np.max(unwrapped_scaled) - np.min(unwrapped_scaled))
        if phase_range < PHASE_RANGE_MIN_BP:
            print("[BPPipeline] Low signal, re-acquiring target...")
            self._target_bin = None
            self._valid_frames = 0  # 替代 self._complex_buffer.clear()
            return

        # ---- Step 5: Downsample 200→50Hz ----
        wave_50hz_raw = resample_poly(unwrapped_scaled,
                                      up=int(self.FS_TARGET),
                                      down=int(self.FS))

        # ---- Step 6: Signal cleaning at 50Hz ----
        clean = clean_pulse_wave(wave_50hz_raw, fs=self.FS_TARGET)

        if len(clean) >= self.N_INPUT:
            input_seq = clean[-self.N_INPUT:]
        else:
            input_seq = np.pad(clean, (self.N_INPUT - len(clean), 0))

        # ---- Step 7: Network inference ----
        print(f"[BPPipeline] phase_range={phase_range:.4f}  clean_range={float(np.max(clean) - np.min(clean)):.4f}")
        bp_waveform = self._bp.predict(input_seq.astype(np.float32))

        # ---- Step 8: SBP / DBP extraction ----
        sbp, dbp, info = extract_bp(bp_waveform, fs=self.FS_TARGET)

        if np.isnan(sbp):
            self._bad_signal_count += 1
        else:
            self._bad_signal_count = 0

        if self._bad_signal_count >= BP_MAX_BAD_SIGNAL_COUNT:
            print("[BPPipeline] Target lost or moved! Forcing re-acquire...")
            self._target_bin = None
            self._cfar_state = None
            self._bad_signal_count = 0
            self._valid_frames = 0  # 替代 self._complex_buffer.clear()
            return

        # ---- Step 9: Temporal smoothing (median → EMA) ----
        if not np.isnan(sbp):
            self._sbp_history.append(sbp)
        if not np.isnan(dbp):
            self._dbp_history.append(dbp)

        sbp_smooth = float(sbp)
        dbp_smooth = float(dbp)

        if len(self._sbp_history) > 0:
            sbp_median = float(np.median(list(self._sbp_history)))
            if self._sbp_ema is None:
                self._sbp_ema = sbp_median
            else:
                self._sbp_ema = 0.3 * sbp_median + 0.7 * self._sbp_ema
            sbp_smooth = self._sbp_ema

        if len(self._dbp_history) > 0:
            dbp_median = float(np.median(list(self._dbp_history)))
            if self._dbp_ema is None:
                self._dbp_ema = dbp_median
            else:
                self._dbp_ema = 0.3 * dbp_median + 0.7 * self._dbp_ema
            dbp_smooth = self._dbp_ema

        # ---- Step 10: Push result ----
        raw_distance = target_bin * self.DISTANCE_PER_BIN
        real_distance = max(MIN_REAL_DISTANCE_M, raw_distance - RANGE_HARDWARE_OFFSET_M)
        result = BPResult(
            timestamp=time.time(),
            frame_index=self._frame_count,
            sbp=sbp_smooth,
            dbp=dbp_smooth,
            bp_waveform=bp_waveform.astype(np.float32),
            target_distance_m=real_distance,
            quality=info,
        )
        self._push_to_display(result)

        # ==========================================
        # 3. 滑动窗口前进 (替代原有的切片自赋值)
        # ==========================================
        shift = self.STEP_FRAMES
        if self._valid_frames > shift:
            # 将后面的数据平移到最前面
            self._buffer[:, :-shift, :] = self._buffer[:, shift:, :]
            # 有效帧数减少，腾出空间给接下来的新数据
            self._valid_frames -= shift
        else:
            self._valid_frames = 0

        if not np.isnan(sbp):
            real_dist = max(0.01, target_bin * self.DISTANCE_PER_BIN - RANGE_HARDWARE_OFFSET_M)
            print(f"[BPPipeline] Result: SBP={sbp:.1f} DBP={dbp:.1f} mmHg dist={real_dist:.2f}m")

    def _push_to_display(self, result: BPResult) -> None:
        if self.display_queue.full():
            try:
                # 扔掉最旧的一帧，给新数据腾地方
                self.display_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.display_queue.put_nowait(result)
        except queue.Full:
            # 极端情况：如果就在这一瞬间又满了，说明消费端卡死，直接丢弃本次更新
            print("[BPPipeline] UI线程卡顿，丢弃过载帧")
