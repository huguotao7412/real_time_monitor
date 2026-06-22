"""Blood pressure monitoring pipeline — Collector-Worker architecture with ring buffer.

Architecture (v2 — real-time optimised):
  Collector Thread (_run):         O(1) frame ingestion into ring buffer
  Worker Thread (_inference_worker): O(N) DSP + PyTorch network inference

Key improvements over v1:
  1. Collector-Worker decoupling: data ingestion and heavy inference run in
     separate threads, eliminating frame drops caused by synchronous EMD +
     network latency.
  2. Ring buffer:  a head-cursor replaces the per-frame and per-batch
     physical array shift (memcpy), reducing CPU/memory-bandwidth waste.
  3. Soft reset + Alpha-Beta range-bin tracking:  when the target is
     temporarily lost the buffer keeps sliding and the tracker predicts the
     next likely bin.  On re-acquisition a phase-continuity compensation
     aligns the old and new bin phases so the noisy window recovers in
     1–2 s instead of the old 5 s cold restart.

Usage (unchanged public API):
    pipeline = BPPipeline("bp_matlab/bp_weights.mat")
    pipeline.start()
    pipeline.raw_queue.put(radar_frame)   # feed per-frame (200 Hz)
    result = pipeline.display_queue.get()  # read BPResult
    pipeline.stop()
"""

import concurrent.futures
import threading
import time
import queue
from collections import deque
from enum import Enum

import numpy as np
from scipy.signal import resample_poly

from config.protocol import (
    RAW_QUEUE_MAXSIZE, DISPLAY_QUEUE_MAXSIZE, RANGE_HARDWARE_OFFSET_M,
    RADAR_BP_FPS, DSP_BP_TARGET_FS, RANGE_RESOLUTION_M,
    BP_BATCH_SEC, BP_STEP_SEC, BP_NETWORK_INPUT_LEN,
    BP_CFAR_INITIAL_FRAMES, BP_CFAR_INTERVAL, BP_CFAR_FALLBACK_FRAMES,
    BP_COLD_START_FRAMES,
    FREQ_SCALE_60G_TO_24G, PHASE_RANGE_MIN_BP, BP_MAX_BAD_SIGNAL_COUNT,
    MIN_REAL_DISTANCE_M,
)
from models.radar_frame import RadarFrame
from bp_monitor.bp_models import BPResult
from bp_monitor.bp_cfar import find_target_bins_1d, adaptive_2d_cfar
from bp_monitor.bp_network import BPInference
from bp_monitor.bp_postprocess import extract_bp
from dsp_pipeline.strategies import SignalCleanerStrategy, EMDPulseCleaner, PassthroughCleaner
from utils.benchmark_logger import AlgorithmBenchmarker, BenchmarkRecord


# ---------------------------------------------------------------------------
# Alpha-Beta range-bin tracker  (Optimisation 2 — soft reset)
# ---------------------------------------------------------------------------

class TrackerState(Enum):
    TRACKING = "tracking"
    LOST = "lost"
    RECOVERING = "recovering"


class AlphaBetaTracker:
    """Alpha-Beta filter for smooth range-bin target tracking.

    Predicts the next likely range bin every frame (constant-velocity model).
    On CFAR observation the estimate is corrected with alpha (position) and
    beta (velocity) gains.

    Parameters
    ----------
    alpha : float  – position gain  (0 < alpha < 1, higher = trust observation more)
    beta  : float  – velocity gain  (0 < beta  < alpha)
    """

    def __init__(self, alpha: float = 0.85, beta: float = 0.5):
        self.x_hat: float | None = None   # smoothed bin position
        self.v_hat: float = 0.0            # smoothed bin velocity (bins / frame)
        self.alpha = alpha
        self.beta = beta

    # -- public -----------------------------------------------------------

    def predict(self,dt: float = 1.0) -> float | None:
        """Advance the state by dt frames."""
        if self.x_hat is None:
            return None
        self.x_hat += self.v_hat * dt
        return self.x_hat

    def update(self, observed: float) -> float:
        """Correct the estimate with a new CFAR observation.

        Returns the smoothed bin estimate.
        """
        if self.x_hat is None:
            self.x_hat = float(observed)
            self.v_hat = 0.0
            return self.x_hat

        residual = observed - self.x_hat
        self.x_hat += self.alpha * residual
        self.v_hat += self.beta * residual
        return self.x_hat

    def reset(self) -> None:
        """Clear all state (hard reset — used on first lock only)."""
        self.x_hat = None
        self.v_hat = 0.0


# ---------------------------------------------------------------------------
# BP Pipeline  (v2 — Collector-Worker + ring buffer + soft reset)
# ---------------------------------------------------------------------------

class BPPipeline:
    """Blood pressure processing pipeline — MATLAB main.m strict port.

    Runs two daemon threads:

    * Collector  — fast O(1) ring-buffer write; triggers inference batches.
    * Worker     — runs the full MATLAB chain (CFAR → phase → EMD → wavelet
                   → network → SBP/DBP) on buffer snapshots.
    """

    FS = float(RADAR_BP_FPS)               # 200 Hz
    FS_TARGET = float(DSP_BP_TARGET_FS)    # 50 Hz
    MAX_FRAMES = int(BP_BATCH_SEC * FS)    # 1024
    N_INPUT = BP_NETWORK_INPUT_LEN         # 256
    STEP_FRAMES = int(BP_STEP_SEC * FS)    # 100
    DISTANCE_PER_BIN = RANGE_RESOLUTION_M  # 0.039 m

    def __init__(
        self,
        weights_path: str = "bp_matlab/bp_weights.mat",
        cleaner: SignalCleanerStrategy | None = None,
        benchmarker: AlgorithmBenchmarker | None = None,
    ):
        # -- public queues (API compatible with v1) --
        self.raw_queue: queue.Queue[RadarFrame] = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
        self.display_queue: queue.Queue[BPResult] = queue.Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)

        self._weights_path = weights_path
        self._bp: BPInference | None = None

        # -- strategy injection --
        self._cleaner: SignalCleanerStrategy = cleaner or EMDPulseCleaner()
        self._current_algo_name: str = "EMD_Pulse"
        self._current_latency_ms: float = 0.0

        # -- A/B comparison --
        self._ab_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._ab_enabled: bool = False
        self._ab_cleaner: SignalCleanerStrategy | None = None
        self._ab_future: concurrent.futures.Future | None = None
        self._ab_algo_name: str = ""
        self._ab_latency_ms: float = 0.0

        # -- benchmarker --
        self._benchmarker: AlgorithmBenchmarker | None = benchmarker

        # -- threads --
        self._collector_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # ==================================================================
        # Ring buffer  (Optimisation 3 — zero-copy / no physical shift)
        # ==================================================================
        self._buffer = np.zeros((32, self.MAX_FRAMES, 1), dtype=complex)
        self._head: int = 0
        self._frame_count: int = 0

        # ==================================================================
        # Collector → Worker channel
        # ==================================================================
        self._inference_queue: queue.Queue = queue.Queue(maxsize=2)

        # ==================================================================
        # Shared state
        # ==================================================================
        self._state_lock = threading.Lock()
        self._target_bin: int | None = None
        self._tracker_state: TrackerState = TrackerState.TRACKING
        self._cfar_state: dict | None = None
        self._cold_start: bool = True

        # ==================================================================
        # Alpha-Beta tracker
        # ==================================================================
        self._tracker = AlphaBetaTracker(alpha=0.85, beta=0.5)

        # Phase-continuity reference
        self._last_phase_ref: tuple[int, float] | None = None

        # ==================================================================
        # Worker-owned state
        # ==================================================================
        self._last_inference_frame: int = 0
        self._bad_signal_count: int = 0

        # Temporal smoothing
        self._sbp_history: deque[float] = deque(maxlen=10)
        self._dbp_history: deque[float] = deque(maxlen=10)
        self._sbp_ema: float | None = None
        self._dbp_ema: float | None = None

    # ======================================================================
    # Public API  (unchanged)
    # ======================================================================

    @property
    def benchmarker(self) -> AlgorithmBenchmarker | None:
        return self._benchmarker

    def set_benchmarker(self, bm: AlgorithmBenchmarker | None) -> None:
        self._benchmarker = bm

    # ── Strategy control ───────────────────────────────────────

    def set_strategies(self, cleaner: SignalCleanerStrategy) -> None:
        """Hot-swap primary cleaner. Takes effect on next worker batch."""
        self._cleaner = cleaner
        cname = cleaner.clean(np.array([0.0]), self.FS_TARGET)[1].get("algorithm", "?")
        self._current_algo_name = cname

    def set_ab_strategy(self, cleaner: SignalCleanerStrategy | None) -> None:
        """Set or disable A/B comparison cleaner. None disables."""
        self._ab_cleaner = cleaner
        self._ab_enabled = (cleaner is not None)

    def get_dsp_telemetry(self) -> dict:
        """Return current DSP engine telemetry for debug panel."""
        return {
            "current_algo": self._current_algo_name,
            "current_latency_ms": self._current_latency_ms,
            "current_snr_gain_db": 0.0,  # BP doesn't compute SNR gain yet
            "ab_algo": self._ab_algo_name if self._ab_enabled else "",
            "ab_latency_ms": self._ab_latency_ms if self._ab_enabled else 0.0,
            "ab_snr_gain_db": 0.0,
            "ab_enabled": self._ab_enabled,
        }

    @property
    def target_bin(self) -> int | None:
        with self._state_lock:
            return self._target_bin

    def start(self) -> None:
        if self._bp is None:
            self._bp = BPInference(self._weights_path)
        self._stop_event.clear()
        self._collector_thread = threading.Thread(
            target=self._run, daemon=True, name="bp-collector"
        )
        self._worker_thread = threading.Thread(
            target=self._inference_worker, daemon=True, name="bp-worker"
        )
        self._collector_thread.start()
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._collector_thread is not None:
            self._collector_thread.join(timeout=3)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3)

    # ======================================================================
    # Collector Thread  (Optimisation 1)
    # ======================================================================

    def _run(self) -> None:
        """Collector — O(1) frame ingestion into ring buffer.

        Sole responsibility:  pull frames from the hardware queue, write
        them into the ring buffer, and decide *when* to fire an inference
        batch.  Every heavy operation is delegated to the Worker thread.
        """
        while not self._stop_event.is_set():
            try:
                frame: RadarFrame = self.raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._collect_frame(frame)
            except Exception:
                import traceback
                print(f"[BPPipeline] Collector error at frame {self._frame_count}:")
                traceback.print_exc()

    def _collect_frame(self, frame: RadarFrame) -> None:
        """Write one frame into the ring buffer and check trigger condition.

        This is the *only* place that touches _buffer / _head / _frame_count.
        It MUST be O(1) — no allocations beyond the raw frame write.
        """
        data_cube = frame.data_cube  # [32, 1, 1]

        # ---- ring-buffer write  (O(1), no memcpy) ----
        self._buffer[:, self._head:self._head + 1, :] = data_cube
        self._head = (self._head + 1) % self.MAX_FRAMES
        self._frame_count += 1

        n_valid = min(self._frame_count, self.MAX_FRAMES)

        # ---- read shared state once ----
        with self._state_lock:
            target_bin = self._target_bin
            cold_start = self._cold_start

        # ---- Phase 1: CFAR lock (no target yet) ----
        if target_bin is None:
            if n_valid >= BP_CFAR_INITIAL_FRAMES and n_valid % BP_CFAR_INTERVAL == 0:
                self._trigger_inference()
            elif n_valid >= BP_CFAR_FALLBACK_FRAMES and n_valid % BP_CFAR_INTERVAL == 0:
                self._trigger_inference()
            return

        # ---- Phase 2: sliding-window inference trigger ----
        required = BP_COLD_START_FRAMES if cold_start else self.MAX_FRAMES
        if n_valid < required:
            return

        # Trigger every STEP_FRAMES new frames (~0.5 s at 200 Hz)
        frames_since_last = self._frame_count - self._last_inference_frame
        if frames_since_last >= self.STEP_FRAMES:
            self._trigger_inference()

    def _trigger_inference(self) -> None:
        """Snapshot the ring buffer and hand it to the Worker thread.

        Non-blocking:  if the worker is still busy with the previous batch
        we simply skip this one (natural back-pressure).
        """
        snapshot = self._get_valid_snapshot()
        if snapshot is None:
            return

        with self._state_lock:
            payload = {
                "snapshot": snapshot,
                "frame_count": self._frame_count,
                "target_bin": self._target_bin,
                "tracker_state": self._tracker_state,
            }

        try:
            self._inference_queue.put_nowait(payload)
        except queue.Full:
            pass  # worker is busy — drop this batch, pick up next cycle
        finally:
            # 无论推流是否成功，都必须更新游标，严格维持 STEP_FRAMES 的滑动步长
            self._last_inference_frame = self._frame_count

    # ------------------------------------------------------------------
    # Ring-buffer read helpers
    # ------------------------------------------------------------------

    def _get_valid_snapshot(self) -> np.ndarray | None:
        """Return a time-ordered **copy** of all valid frames.

        The copy is intentional: it decouples the Collector's write cursor
        from the Worker's (potentially long) processing window, so the
        Collector never blocks on memcpy contention.
        """
        n_valid = min(self._frame_count, self.MAX_FRAMES)
        if n_valid == 0:
            return None

        n_bins = self._buffer.shape[0]
        head = self._head
        # start of the oldest valid frame in logical time order
        start = (head - n_valid) % self.MAX_FRAMES

        if start + n_valid <= self.MAX_FRAMES:
            # contiguous segment — single slice copy
            return self._buffer[:, start:start + n_valid, :].copy()
        else:
            # wrapped segment — stitch two slices
            first_n = self.MAX_FRAMES - start
            second_n = n_valid - first_n
            result = np.zeros((n_bins, n_valid, 1), dtype=complex)
            result[:, :first_n, :] = self._buffer[:, start:, :]
            result[:, first_n:, :] = self._buffer[:, :second_n, :]
            return result

    # ======================================================================
    # Worker Thread  (Optimisation 1)
    # ======================================================================

    def _inference_worker(self) -> None:
        """Worker — O(N) DSP + network inference on buffer snapshots.

        Blocks on the inference queue.  When a snapshot arrives it runs the
        full MATLAB processing chain and pushes a BPResult to the display
        queue.
        """
        while not self._stop_event.is_set():
            try:
                payload = self._inference_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_snapshot(
                    snapshot=payload["snapshot"],
                    frame_count=payload["frame_count"],
                    target_bin=payload["target_bin"],
                    tracker_state=payload["tracker_state"],
                )
            except Exception:
                import traceback
                print("[BPPipeline] Worker error:")
                traceback.print_exc()

    def _process_snapshot(
        self,
        snapshot: np.ndarray,
        frame_count: int,
        target_bin: int | None,
        tracker_state: TrackerState,
    ) -> None:
        """Full MATLAB processing chain on a single buffer snapshot.

        Parameters
        ----------
        snapshot : [32, N, 1] complex  – time-ordered frame cube
        frame_count : int              – collector frame_count at trigger time
        target_bin : int | None        – current locked bin (or None)
        tracker_state : TrackerState   – TRACKING / LOST / RECOVERING
        """
        n = snapshot.shape[1]  # number of frames in this snapshot
        is_recovery = (tracker_state in (TrackerState.LOST, TrackerState.RECOVERING))

        # ================================================================
        # Phase 1: CFAR lock  (no target yet)
        # ================================================================
        if target_bin is None:
            acc_bg = snapshot - np.mean(snapshot, axis=1, keepdims=True)
            candidates = find_target_bins_1d(
                acc_bg, self.DISTANCE_PER_BIN, num_targets=1
            )
            if len(candidates) > 0:
                target_bin = int(candidates[0])
                if tracker_state == TrackerState.LOST:
                    self._tracker.reset()
                self._tracker.update(float(target_bin))
                real_dist = max(
                    MIN_REAL_DISTANCE_M,
                    target_bin * self.DISTANCE_PER_BIN - RANGE_HARDWARE_OFFSET_M,
                )
                print(
                    f"[BPPipeline] Target locked: bin={target_bin}"
                    f" ({real_dist:.2f} m)"
                )
            elif n >= BP_CFAR_FALLBACK_FRAMES:
                # Fallback: pick strongest bin (skip near-field DC)
                acc = snapshot
                acc_bg = acc - np.mean(acc, axis=1, keepdims=True)
                energy = np.mean(np.abs(acc_bg), axis=(1, 2))
                energy[:2] = 0
                target_bin = int(np.argmax(energy))
                self._tracker.update(float(target_bin))
                real_dist = max(
                    MIN_REAL_DISTANCE_M,
                    target_bin * self.DISTANCE_PER_BIN - RANGE_HARDWARE_OFFSET_M,
                )
                print(
                    f"[BPPipeline] Fallback lock: bin={target_bin}"
                    f" ({real_dist:.2f} m)"
                )

            if target_bin is not None:
                with self._state_lock:
                    self._target_bin = target_bin
                    self._tracker_state = TrackerState.TRACKING
                return  # wait for next batch with full data

            return  # keep accumulating

        # ================================================================
        # Phase 2: Full MATLAB pipeline on locked target
        # ================================================================

        # --- tracker predict (run every batch, even when tracking) ---
        self._tracker.predict(dt=1.0)

        print(f"[BPPipeline] Processing batch at frame {frame_count}...")

        # --- background subtraction ---
        mean_bin_frame_rx = snapshot.copy()
        background = np.mean(mean_bin_frame_rx, axis=1, keepdims=True)  # [32, 1, 1]
        mean_bin_frame_rx = mean_bin_frame_rx - background

        # --- 1D + 2D CFAR ---
        overall_target_bins = find_target_bins_1d(
            mean_bin_frame_rx, self.DISTANCE_PER_BIN, num_targets=3, verbose=True
        )
        if len(overall_target_bins) == 0:
            self._handle_target_lost(frame_count)
            return

        target_bins, self._cfar_state = adaptive_2d_cfar(
            mean_bin_frame_rx, overall_target_bins, self._cfar_state
        )
        if len(target_bins) == 0:
            self._handle_target_lost(frame_count)
            return

        print(
            f"[BPPipeline] 2D CFAR confirmed {len(target_bins)} target(s):"
            f" bins={list(target_bins)}"
        )
        new_target_bin = int(target_bins[0])

        # --- update tracker with observation ---
        self._tracker.update(float(new_target_bin))

        # ================================================================
        # Phase-continuity compensation  (Optimisation 2)
        #
        # When the target bin shifts (subject moved slightly) we compute
        # the instantaneous phase difference between the old and new bins
        # and apply it as a constant offset.  This keeps the unwrapped
        # phase stream continuous across bin jumps, allowing the noisy
        # window to recover into a valid network input in 1–2 s instead
        # of the full 5 s cold restart.
        # ================================================================
        phase_offset = 0.0
        if target_bin is not None and new_target_bin != target_bin and self._last_phase_ref is not None:
            ref_bin, _ref_phase = self._last_phase_ref
            # extract phase at the *same* latest frame for both bins
            transition_idx = -self.STEP_FRAMES
            old_complex = mean_bin_frame_rx[target_bin, transition_idx, :]
            new_complex = mean_bin_frame_rx[new_target_bin, transition_idx, :]

            old_phase = np.angle(old_complex[np.argmax(np.abs(old_complex))])
            new_phase = np.angle(new_complex[np.argmax(np.abs(new_complex))])
            raw_diff = np.angle(np.exp(1j * (old_phase - new_phase)))
            phase_offset = raw_diff * FREQ_SCALE_60G_TO_24G
            print(
                f"[BPPipeline] Phase alignment: offset={phase_offset:.4f} rad"
                f" (bin {target_bin} → {new_target_bin})"
            )

        if target_bin is not None and new_target_bin != target_bin:
            transition_idx = -self.STEP_FRAMES

            # 提取边界处单点复数 (确保维度为标量复数)
            old_c = mean_bin_frame_rx[target_bin, transition_idx, :][0]
            new_c = mean_bin_frame_rx[new_target_bin, transition_idx, :][0]

            # 计算复数域的精确相位差 (Old - New)
            phase_shift_rad = np.angle(old_c * np.conj(new_c))

            complex_old = mean_bin_frame_rx[target_bin, :-self.STEP_FRAMES, :]
            complex_new = mean_bin_frame_rx[new_target_bin, -self.STEP_FRAMES:, :]

            # 【核心修复】在拼接与解包前，先在复数域旋转新信号以对齐旧信号的相位
            complex_new = complex_new * np.exp(1j * phase_shift_rad)
            complex_data = np.concatenate([complex_old, complex_new], axis=0)

            print(f"[BPPipeline] Target bin updated: {target_bin} → {new_target_bin}, Phase aligned")
            target_bin = new_target_bin
            with self._state_lock:
                self._target_bin = target_bin
        else:
            complex_data = mean_bin_frame_rx[target_bin, :, :]  # [N, 1]

        phase_data = np.angle(complex_data)
        unwrapped = np.unwrap(phase_data, axis=0).squeeze()
        unwrapped_scaled = unwrapped * FREQ_SCALE_60G_TO_24G

        # 存储参考点以备后用
        self._last_phase_ref = (target_bin, float(unwrapped_scaled[-1]))

        # --- low-signal detection → soft reset ---
        phase_range = float(np.max(unwrapped_scaled) - np.min(unwrapped_scaled))
        if phase_range < PHASE_RANGE_MIN_BP:
            print("[BPPipeline] Low signal — soft reset (buffer preserved)")
            self._handle_target_lost(frame_count)
            return

        # --- downsample 200 → 50 Hz ---
        wave_50hz_raw = resample_poly(
            unwrapped_scaled, up=int(self.FS_TARGET), down=int(self.FS)
        )

        # --- signal cleaning (strategy-injected) ---
        clean, metrics = self._cleaner.clean(wave_50hz_raw, fs=self.FS_TARGET)

        # --- benchmark log ---
        if self._benchmarker and self._benchmarker.is_recording:
            try:
                input_pr = float(np.max(wave_50hz_raw) - np.min(wave_50hz_raw))
            except Exception:
                input_pr = 0.0
            self._benchmarker.log(BenchmarkRecord(
                timestamp=time.time(),
                frame_index=frame_count,
                elapsed_sec=frame_count / self.FS,
                algorithm_name=metrics.get("algorithm", "EMD_Pulse"),
                is_primary=True,
                latency_ms=metrics.get("latency_ms", 0.0),
                input_phase_range=input_pr,
                input_snr_db=None,
                output_phase_range=float(np.max(clean) - np.min(clean)) if len(clean) > 0 else 0.0,
                output_snr_db=None,
                snr_gain_db=None,
                breath_bpm=0.0,
                heart_bpm=0.0,
                heart_prominence=0.0,
                imf_count=metrics.get("imf_count"),
                convergence_iter=metrics.get("convergence_iter"),
                retained_harmonics=(
                    ",".join(str(h) for h in metrics["retained_harmonics"])
                    if metrics.get("retained_harmonics") else None
                ),
                dominant_freq_hz=metrics.get("dominant_freq_hz"),
            ))
            # Update telemetry
            self._current_algo_name = metrics.get("algorithm", "EMD_Pulse")
            self._current_latency_ms = metrics.get("latency_ms", 0.0)

        # ── A/B comparison: dispatch alternative cleaner ──
        if self._ab_enabled and self._ab_cleaner is not None:
            # Check previous A/B result
            if self._ab_future is not None and self._ab_future.done():
                try:
                    ab_clean, ab_metrics = self._ab_future.result()
                    self._ab_algo_name = ab_metrics.get("algorithm", "?")
                    self._ab_latency_ms = ab_metrics.get("latency_ms", 0.0)
                    # Log A/B result if benchmarker is recording
                    if self._benchmarker and self._benchmarker.is_recording:
                        try:
                            ab_input_pr = float(np.max(wave_50hz_raw) - np.min(wave_50hz_raw))
                        except Exception:
                            ab_input_pr = 0.0
                        self._benchmarker.log(BenchmarkRecord(
                            timestamp=time.time(),
                            frame_index=frame_count,
                            elapsed_sec=frame_count / self.FS,
                            algorithm_name=ab_metrics.get("algorithm", "AB"),
                            is_primary=False,
                            latency_ms=ab_metrics.get("latency_ms", 0.0),
                            input_phase_range=ab_input_pr,
                            input_snr_db=None,
                            output_phase_range=float(np.max(ab_clean) - np.min(ab_clean)) if len(ab_clean) > 0 else 0.0,
                            output_snr_db=None,
                            snr_gain_db=None,
                            breath_bpm=0.0,
                            heart_bpm=0.0,
                            heart_prominence=0.0,
                            imf_count=ab_metrics.get("imf_count"),
                            convergence_iter=ab_metrics.get("convergence_iter"),
                            retained_harmonics=(
                                ",".join(str(h) for h in ab_metrics["retained_harmonics"])
                                if ab_metrics.get("retained_harmonics") else None
                            ),
                            dominant_freq_hz=ab_metrics.get("dominant_freq_hz"),
                        ))
                except Exception:
                    self._ab_algo_name = "error"
                    self._ab_latency_ms = 0.0

            # Dispatch new A/B task if idle
            if self._ab_future is None or self._ab_future.done():
                self._ab_future = self._ab_executor.submit(
                    self._ab_cleaner.clean, wave_50hz_raw.copy(), self.FS_TARGET,
                )

        if len(clean) >= self.N_INPUT:
            input_seq = clean[-self.N_INPUT:]
        else:
            input_seq = np.pad(clean, (self.N_INPUT - len(clean), 0))

        # --- network inference ---
        print(
            f"[BPPipeline] phase_range={phase_range:.4f}"
            f"  clean_range={float(np.max(clean) - np.min(clean)):.4f}"
        )
        bp_waveform = self._bp.predict(input_seq.astype(np.float32))

        # --- SBP / DBP extraction ---
        sbp, dbp, info = extract_bp(bp_waveform, fs=self.FS_TARGET)

        # --- bad-signal handling (soft) ---
        if np.isnan(sbp):
            self._bad_signal_count += 1
        else:
            self._bad_signal_count = 0

        if self._bad_signal_count >= BP_MAX_BAD_SIGNAL_COUNT:
            self._handle_target_lost(frame_count)
            return

        # --- temporal smoothing (median → EMA) ---
        sbp_smooth, dbp_smooth = self._smooth_bp(sbp, dbp)

        # --- push result ---
        raw_distance = target_bin * self.DISTANCE_PER_BIN
        real_distance = max(
            MIN_REAL_DISTANCE_M, raw_distance - RANGE_HARDWARE_OFFSET_M
        )
        result = BPResult(
            timestamp=time.time(),
            frame_index=frame_count,
            sbp=sbp_smooth,
            dbp=dbp_smooth,
            bp_waveform=bp_waveform.astype(np.float32),
            target_distance_m=real_distance,
            quality=info,
        )
        self._push_to_display(result)

        # --- recovery complete → back to tracking ---
        if is_recovery:
            print("[BPPipeline] Recovery complete — tracking resumed")
        with self._state_lock:
            self._tracker_state = TrackerState.TRACKING
            self._cold_start = False

        if not np.isnan(sbp):
            real_dist = max(
                0.01,
                target_bin * self.DISTANCE_PER_BIN - RANGE_HARDWARE_OFFSET_M,
            )
            print(
                f"[BPPipeline] Result: SBP={sbp:.1f}  DBP={dbp:.1f} mmHg"
                f"  dist={real_dist:.2f} m"
            )

    # ======================================================================
    # Soft-reset helpers  (Optimisation 2)
    # ======================================================================

    def _handle_target_lost(self, frame_count: int) -> None:
        """Soft reset: mark target lost but **preserve** buffer & tracker.

        Unlike v1 which called ``_valid_frames = 0`` (discarding 5.12 s of
        data), this only clears the lock so the next CFAR pass can
        re-acquire quickly.  The Alpha-Beta tracker keeps predicting, so
        when the subject moves back the recovery time is 1–2 s instead of
        the full cold-start delay.
        """
        print(
            f"[BPPipeline] Target lost at frame {frame_count}"
            f" — soft reset, tracker predicting …"
        )
        with self._state_lock:
            self._tracker_state = TrackerState.LOST
            predicted_bin = self._tracker.x_hat
            if predicted_bin is not None:
                max_bin = self._buffer.shape[0] - 1  # 31
                safe_bin = int(round(predicted_bin))
                self._target_bin = max(0, min(safe_bin, max_bin))
            else:
                self._target_bin = None

            self._cfar_state = None
        self._bad_signal_count = 0
        # NOTE: _buffer, _head, _frame_count, _last_phase_ref are preserved

    # ======================================================================
    # Temporal smoothing  (unchanged from v1)
    # ======================================================================

    def _smooth_bp(self, sbp: float, dbp: float) -> tuple[float, float]:
        """Median → EMA smoothing for SBP / DBP."""
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

        return sbp_smooth, dbp_smooth

    # ======================================================================
    # Display output
    # ======================================================================

    def _push_to_display(self, result: BPResult) -> None:
        """Push a result to the display queue, dropping oldest if full."""
        if self.display_queue.full():
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.display_queue.put_nowait(result)
        except queue.Full:
            print("[BPPipeline] UI thread stuck, dropping frame")
