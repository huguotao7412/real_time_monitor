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

import numpy as np
from scipy.signal import resample_poly

from config.protocol import RAW_QUEUE_MAXSIZE, DISPLAY_QUEUE_MAXSIZE
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

    MAX_FRAMES = 1024
    FS = 200.0
    FS_TARGET = 50.0
    N_INPUT = 256
    DISTANCE_PER_BIN = 0.05

    def __init__(self, weights_path: str = "bp_matlab/bp_weights.mat"):
        self.raw_queue: queue.Queue[RadarFrame] = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
        self.display_queue: queue.Queue[BPResult] = queue.Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)

        self._weights_path = weights_path
        self._bp: BPInference | None = None

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Internal state
        self._complex_buffer: list[np.ndarray] = []  # list of [32, 1, 1] complex arrays
        self._frame_count = 0
        self._target_bin: int | None = None
        self._cfar_state: dict | None = None   # persistent CFAR state (adaptive beta)

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
        """Accumulate complex frames, process batch when 1024 frames collected.

        Strictly follows MATLAB main.m while-loop."""
        data_cube = frame.data_cube  # [32, 1, 1] = [bins, doppler=1, rx=1]

        # Always accumulate complex frame FIRST
        self._complex_buffer.append(data_cube.copy())
        self._frame_count += 1

        # ---- Phase 1: CFAR lock ----
        if self._target_bin is None:
            n = len(self._complex_buffer)
            # Try CFAR every 16 frames starting from 64 frames
            if n >= 64 and n % 16 == 0:
                acc = np.concatenate(self._complex_buffer, axis=1)  # [32, N, 1]
                candidates = find_target_bins_1d(
                    acc, self.DISTANCE_PER_BIN, num_targets=1
                )
                if len(candidates) > 0:
                    self._target_bin = int(candidates[0])
                    print(f"[BPPipeline] Target locked: bin={self._target_bin} "
                          f"({self._target_bin * self.DISTANCE_PER_BIN:.2f}m)")
            # Fallback after 256 frames: pick strongest bin
            if self._target_bin is None and n >= 256:
                acc = np.concatenate(self._complex_buffer, axis=1)
                energy = np.mean(np.abs(acc), axis=(1, 2))
                energy[:2] = 0  # skip near-field DC
                self._target_bin = int(np.argmax(energy))
                print(f"[BPPipeline] Fallback lock: bin={self._target_bin} "
                      f"({self._target_bin * self.DISTANCE_PER_BIN:.2f}m)")
            if self._target_bin is None:
                return  # keep accumulating

        # ---- Phase 2: Wait for full batch ----
        if len(self._complex_buffer) < self.MAX_FRAMES:
            return

        # ---- Full MATLAB pipeline on 1024 frames ----
        print(f"[BPPipeline] Processing batch at frame {self._frame_count}...")

        # Build mean_bin_frame_rx: [32, 1024, 1]  (MATLAB extract_3d_data)
        mean_bin_frame_rx = np.concatenate(self._complex_buffer, axis=1)

        # MATLAB: background = mean(bin_frame_rx, 2); mean_bin_frame_rx = bin_frame_rx - background
        background = np.mean(mean_bin_frame_rx, axis=1, keepdims=True)  # [32, 1, 1]
        mean_bin_frame_rx = mean_bin_frame_rx - background

        # ---- Step 1+2: 1D CFAR + 2D CFAR (MATLAB lines 87-89) ----
        overall_target_bins = find_target_bins_1d(
            mean_bin_frame_rx, self.DISTANCE_PER_BIN, num_targets=3, verbose=True
        )
        if len(overall_target_bins) == 0:
            print("[BPPipeline] CFAR: no target found, re-acquiring...")
            self._target_bin = None
            self._complex_buffer.clear()
            return

        target_bins, self._cfar_state = adaptive_2d_cfar(
            mean_bin_frame_rx, overall_target_bins, self._cfar_state
        )
        if len(target_bins) == 0:
            print("[BPPipeline] 2D CFAR: no target confirmed, re-acquiring...")
            self._target_bin = None
            self._complex_buffer.clear()
            return

        target_bin = int(target_bins[0])
        # Update target bin if CFAR found a better one
        if target_bin != self._target_bin:
            print(f"[BPPipeline] Target bin updated: {self._target_bin} → {target_bin}")
            self._target_bin = target_bin

        # ---- Step 3: Phase extraction (MATLAB extract_target_phase) ----
        # complex_data = mean_bin_frame_rx(target_bin, :, :) → [1024, 1]
        complex_data = mean_bin_frame_rx[target_bin, :, :]  # [1024, 1]
        phase_data = np.angle(complex_data)                   # [1024, 1]
        unwrapped = np.unwrap(phase_data, axis=0)             # unwrap along time
        unwrapped = unwrapped.squeeze()                       # [1024]

        # ---- Step 4: Frequency scaling (MATLAB: × 24/60) ----
        unwrapped_scaled = unwrapped * (24.0 / 60.0)

        # ---- Step 5: Signal cleaning (MATLAB PhaseProcess.RadarSignalCleaner) ----
        clean = clean_pulse_wave(unwrapped_scaled, fs=self.FS)

        # ---- Step 5b: Low-signal detection → re-acquire ----
        phase_range = float(np.max(unwrapped_scaled) - np.min(unwrapped_scaled))
        if phase_range < 0.001:
            print("[BPPipeline] Low signal, re-acquiring target...")
            self._target_bin = None
            self._complex_buffer.clear()
            return

        # ---- Step 6: Downsample 200→50Hz (MATLAB: resample(wave, 50, 200)) ----
        wave_50hz = resample_poly(clean, up=50, down=200)

        # Take last 256 points (MATLAB: wave_50hz(end-255:end))
        if len(wave_50hz) >= self.N_INPUT:
            input_seq = wave_50hz[-self.N_INPUT:]
        else:
            input_seq = np.pad(wave_50hz, (self.N_INPUT - len(wave_50hz), 0))

        # ---- Step 7: Network inference ----
        bp_waveform = self._bp.predict(input_seq.astype(np.float32))

        # ---- Step 8: SBP / DBP extraction ----
        sbp, dbp, info = extract_bp(bp_waveform, fs=self.FS_TARGET)

        # ---- Step 9: Push result ----
        result = BPResult(
            timestamp=time.time(),
            frame_index=self._frame_count,
            sbp=sbp,
            dbp=dbp,
            bp_waveform=bp_waveform.astype(np.float32),
            target_distance_m=target_bin * self.DISTANCE_PER_BIN,
            quality=info,
        )
        self._push_to_display(result)

        # ---- Clear buffer for next batch (MATLAB: loops back to read_data) ----
        self._complex_buffer.clear()

        if not np.isnan(sbp):
            print(f"[BPPipeline] Result: SBP={sbp:.1f} DBP={dbp:.1f} mmHg "
                  f"dist={target_bin * self.DISTANCE_PER_BIN:.2f}m")

    def _push_to_display(self, result: BPResult) -> None:
        try:
            self.display_queue.put_nowait(result)
        except queue.Full:
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.display_queue.put_nowait(result)
            except queue.Full:
                pass
