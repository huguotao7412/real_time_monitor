"""Blood pressure monitoring pipeline — independent thread, 200Hz mode.

Collects 1024 frames of phase data, cleans the signal, runs neural network
inference, extracts SBP/DBP, and pushes BPResult to the display queue.

Usage:
    pipeline = BPPipeline("bp_matlab/bp_weights.mat")
    pipeline.start()
    # Feed: pipeline.raw_queue.put(radar_frame)
    # Read: pipeline.display_queue.get()
    pipeline.stop()
"""

import threading
import time
import queue
from collections import deque

import numpy as np
from scipy.signal import resample

from config.protocol import RAW_QUEUE_MAXSIZE, DISPLAY_QUEUE_MAXSIZE
from models.radar_frame import RadarFrame
from bp_monitor.bp_models import BPResult
from bp_monitor.bp_cfar import find_target_bins_1d
from bp_monitor.bp_signal_cleaner import clean_pulse_wave
from bp_monitor.bp_network import BPInference
from bp_monitor.bp_postprocess import extract_bp
from dsp_pipeline.phase import extract_phase, unwrap_phase


class BPPipeline:
    """Blood pressure processing pipeline.

    Runs in a dedicated daemon thread.  Accumulates 1024 frames of unwrapped
    phase (~5.12 s at 200 Hz), cleans the signal, runs the waveform
    reconstruction network, and extracts SBP/DBP.

    Thread model matches existing dsp_pipeline.Pipeline.
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
        self._phase_buffer: deque[float] = deque(maxlen=self.MAX_FRAMES)
        self._frame_count = 0
        self._target_bin: int | None = None

    # -- public API ---------------------------------------------------------

    @property
    def target_bin(self) -> int | None:
        return self._target_bin

    def start(self) -> None:
        """Start the processing thread."""
        if self._bp is None:
            self._bp = BPInference(self._weights_path)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal stop and wait for thread to exit."""
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
                # Log and continue — single bad frame shouldn't crash the pipeline
                pass

    def _process_frame(self, frame: RadarFrame) -> None:
        data_cube = frame.data_cube  # [bins, doppler, rx]

        # 1. Acquire target bin (first few frames, or periodic re-acquisition)
        if self._target_bin is None:
            candidates = find_target_bins_1d(
                data_cube[:, :, :], self.DISTANCE_PER_BIN, num_targets=1
            )
            if len(candidates) > 0:
                self._target_bin = int(candidates[0])

        if self._target_bin is None:
            self._frame_count += 1
            return

        # 2. Extract phase (MATLAB: 60 GHz -> 24 GHz frequency scaling)
        phase = extract_phase(data_cube, self._target_bin)
        phase_scaled = phase * (24.0 / 60.0)
        self._phase_buffer.append(phase_scaled)
        self._frame_count += 1

        # 3. Process when buffer is full
        if len(self._phase_buffer) < self.MAX_FRAMES:
            return

        # Unwrap
        phase_arr = np.array(self._phase_buffer, dtype=np.float64)
        unwrapped = unwrap_phase(phase_arr)

        # 4. Signal cleaning
        clean = clean_pulse_wave(unwrapped, fs=self.FS)

        # 5. Downsample 200 Hz -> 50 Hz -> 256 points
        n_target = int(len(clean) * self.FS_TARGET / self.FS)
        wave_50hz = resample(clean, n_target)

        if len(wave_50hz) >= self.N_INPUT:
            input_seq = wave_50hz[-self.N_INPUT:]
        else:
            input_seq = np.pad(wave_50hz, (self.N_INPUT - len(wave_50hz), 0))

        # 6. Neural network inference
        bp_waveform = self._bp.predict(input_seq.astype(np.float32))

        # 7. SBP / DBP
        sbp, dbp, info = extract_bp(bp_waveform, fs=self.FS_TARGET)

        # 8. Push result
        result = BPResult(
            timestamp=time.time(),
            frame_index=self._frame_count,
            sbp=sbp,
            dbp=dbp,
            bp_waveform=bp_waveform.astype(np.float32),
            target_distance_m=self._target_bin * self.DISTANCE_PER_BIN,
            quality=info,
        )
        self._push_to_display(result)

        # Clear buffer for next batch
        self._phase_buffer.clear()

    def _push_to_display(self, result: BPResult) -> None:
        try:
            self.display_queue.put_nowait(result)
        except queue.Full:
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                pass
            self.display_queue.put_nowait(result)
