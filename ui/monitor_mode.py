"""Monitor mode abstraction — each vital-signs mode (HR, BP) as a Strategy object."""

import threading
import time
import queue
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime

import numpy as np

from config.protocol import UI_REFRESH_MS, RANGE_HARDWARE_OFFSET_M, MIN_REAL_DISTANCE_M
from config.i18n import tr
from models.radar_frame import RadarFrame, FrameHeader
from dsp_pipeline.vital_signs import VitalSigns
from dsp_pipeline.strategies import (
    SignalCleanerStrategy, VitalSignSeparator,
    VMDRLSCleaner, EMDHarmonicCleaner, PassthroughCleaner,
    WPDSeparator, SOSFilterSeparator, EMDPulseCleaner,
)
from utils.benchmark_logger import AlgorithmBenchmarker


def _drain_queue(q) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


class MonitorMode(ABC):
    """Abstract vital-signs monitoring mode.

    Each concrete mode owns its pipeline, frame builder, display queue,
    data buffers, and tab visibility policy.
    """

    @property
    @abstractmethod
    def uart_bins(self) -> int:
        """Number of FFT bins per UART frame for this mode."""
        ...

    @abstractmethod
    def boot_radar(self, radar_mgr) -> bool:
        """Send radar boot sequence for this mode. Returns True on success."""
        ...

    @abstractmethod
    def build_frame(self, fft_data: np.ndarray, frame_index: int) -> RadarFrame:
        """Build a RadarFrame from raw FFT data in the correct format."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Create and start the processing pipeline."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Drain display queue and stop the processing pipeline."""
        ...

    @abstractmethod
    def feed_frame(self, frame: RadarFrame) -> None:
        """Push a RadarFrame into the pipeline's raw queue (non-blocking)."""
        ...

    @abstractmethod
    def poll_and_update(self, subject_tab, bp_tab, research_tab,
                        status_label, elapsed_label, frame_rate_label,
                        start_time, frame_count) -> None:
        """Poll the pipeline's display queue and update the appropriate tabs.

        Called from the UI timer on the main thread.
        """
        ...

    @abstractmethod
    def tab_visibility(self) -> tuple[bool, bool, bool]:
        """Return (show_subject, show_bp, show_research) for this mode."""
        ...

    @abstractmethod
    def get_export_data(self) -> dict:
        """Return accumulated data ready for export.

        Returns dict with keys depending on mode:
          HR: csv_rows, breath_waveform_accum, heart_waveform_accum,
              bpm_history, sqi_history, latest_vitals
          BP: bp_results (list of BPResult)
        """
        ...

    @abstractmethod
    def clear_data(self) -> None:
        """Reset all accumulated data buffers."""
        ...


class HRMode(MonitorMode):
    """Heart rate / breath rate monitoring mode (2T4R, 128 range bins)."""

    def __init__(self):
        from dsp_pipeline.pipeline import Pipeline

        self._pipeline: Pipeline | None = None
        self._latest_vitals: VitalSigns | None = None
        self._trend_tick_counter: int = 0

        # Data accumulation for export (bounded: ~1 hour at 1 row/s)
        self._csv_rows: deque[dict] = deque(maxlen=3600)
        self._breath_waveform_accum: deque[np.ndarray] = deque(maxlen=3600)
        self._heart_waveform_accum: deque[np.ndarray] = deque(maxlen=3600)
        self._bpm_history: list[tuple[float, float, float]] = []
        self._sqi_history: list[dict] = []

        # Strategy + benchmarker state (applied on next start())
        self._pending_cleaner: SignalCleanerStrategy | None = None
        self._pending_separator: VitalSignSeparator | None = None
        self._pending_ab_cleaner: SignalCleanerStrategy | None = None
        self._pending_ab_separator: VitalSignSeparator | None = None
        self._use_adaptive: bool = True
        self._benchmarker: AlgorithmBenchmarker | None = None


    # -- MonitorMode impl ------------------------------------------------

    @property
    def uart_bins(self) -> int:
        return 1024

    def boot_radar(self, radar_mgr) -> bool:
        return radar_mgr.boot()

    def build_frame(self, fft_data: np.ndarray, frame_index: int) -> RadarFrame:
        """Build HR frame: 2T4R, 128 range bins."""
        cube = fft_data.reshape(2, 4, -1)
        rx_combined = np.mean(cube[0, :, :], axis=0)
        return RadarFrame(
            timestamp=time.time(),
            frame_index=frame_index,
            header=FrameHeader(0, 1, 4, 2, 58000, 128, 1, 3000, 50, 1920, 60),
            data_cube=rx_combined.reshape(-1, 1, 1),
        )

    def start(self) -> None:
        from dsp_pipeline.pipeline import Pipeline
        cleaner = self._pending_cleaner or VMDRLSCleaner()
        separator = self._pending_separator or WPDSeparator()
        self._pipeline = Pipeline(
            cleaner=cleaner,
            separator=separator,
            use_adaptive=self._use_adaptive,
        )
        if self._pending_ab_cleaner is not None and self._pending_ab_separator is not None:
            self._pipeline.set_ab_strategy(
                self._pending_ab_cleaner, self._pending_ab_separator
            )
        if self._benchmarker is not None:
            self._pipeline.set_benchmarker(self._benchmarker)
        self._pipeline.start()

    def stop(self) -> None:
        if self._pipeline:
            _drain_queue(self._pipeline.display_queue)
            self._pipeline.stop()
            self._pipeline = None

    # ── Strategy control ─────────────────────────────────────────

    def set_strategies(
        self, cleaner: SignalCleanerStrategy, separator: VitalSignSeparator,
    ) -> None:
        self._use_adaptive = False
        self._pending_cleaner = cleaner
        self._pending_separator = separator
        if self._pipeline is not None:
            self._pipeline.set_strategies(cleaner, separator)

    def set_adaptive_mode(self) -> None:
        self._use_adaptive = True
        if self._pipeline is not None:
            self._pipeline._use_adaptive = True

    def set_ab_strategy(
        self,
        cleaner: SignalCleanerStrategy | None,
        separator: VitalSignSeparator | None,
    ) -> None:
        self._pending_ab_cleaner = cleaner
        self._pending_ab_separator = separator
        if self._pipeline is not None:
            self._pipeline.set_ab_strategy(cleaner, separator)

    def toggle_benchmark(self) -> bool:
        if self._benchmarker is None:
            self._benchmarker = AlgorithmBenchmarker()
        if self._benchmarker.is_recording:
            self._benchmarker.stop()
            return False
        else:
            self._benchmarker.start()
            if self._pipeline is not None:
                self._pipeline.set_benchmarker(self._benchmarker)
            return True

    def get_dsp_telemetry(self) -> dict:
        if self._pipeline is None:
            return {
                "current_algo": "--",
                "current_latency_ms": 0.0,
                "current_snr_gain_db": 0.0,
                "ab_algo": "",
                "ab_latency_ms": 0.0,
                "ab_snr_gain_db": 0.0,
                "ab_enabled": False,
            }
        return {
            "current_algo": self._pipeline._current_algo_name,
            "current_latency_ms": self._pipeline._current_latency_ms,
            "current_snr_gain_db": self._pipeline._current_snr_gain_db,
            "ab_algo": self._pipeline._ab_algo_name,
            "ab_latency_ms": self._pipeline._ab_latency_ms,
            "ab_snr_gain_db": self._pipeline._ab_snr_gain_db,
            "ab_enabled": self._pipeline._ab_enabled,
        }

    def get_benchmarker(self) -> AlgorithmBenchmarker | None:
        return self._benchmarker

    def get_benchmark_elapsed(self) -> float:
        if self._benchmarker is None or not self._benchmarker.is_recording:
            return 0.0
        return time.time() - self._benchmarker._start_time

    def feed_frame(self, frame: RadarFrame) -> None:
        if self._pipeline is None:
            return
        try:
            self._pipeline.raw_queue.put_nowait(frame)
        except queue.Full:
            with self._pipeline.raw_queue.mutex:
                self._pipeline.raw_queue.queue.clear()
            self._pipeline.raw_queue.put_nowait(frame)

    def poll_and_update(self, subject_tab, bp_tab, research_tab,
                        status_label, elapsed_label, frame_rate_label,
                        start_time, frame_count) -> None:
        if self._pipeline is None:
            return

        try:
            while not self._pipeline.display_queue.empty():
                self._latest_vitals = self._pipeline.display_queue.get_nowait()
        except queue.Empty:
            pass

        if self._latest_vitals is None:
            return

        q = self._latest_vitals.quality
        calib_done = self._pipeline.calibration_done
        calib_prog = self._pipeline.calibration_progress

        # Compute physical distance from best_range_bin
        best_bin = self._pipeline.best_range_bin
        if best_bin is not None and best_bin > 0:
            target_distance_m = (best_bin * 0.039) - RANGE_HARDWARE_OFFSET_M
            target_distance_m = max(MIN_REAL_DISTANCE_M, target_distance_m)
        else:
            target_distance_m = 0.0

        # Subject tab
        subject_tab.update_display(
            breath_bpm=self._latest_vitals.breath_bpm,
            heart_bpm=self._latest_vitals.heart_bpm,
            breath_waveform=self._latest_vitals.breath_waveform,
            quality=q,
            calibration_done=calib_done,
            calibration_progress=calib_prog,
            target_distance_m=target_distance_m,
        )

        # Research tab
        self._trend_tick_counter += 1
        trend_sample = (self._trend_tick_counter % 20 == 0)
        dsp_telemetry = self.get_dsp_telemetry()
        benchmark_elapsed = self.get_benchmark_elapsed()
        research_tab.update_display(
            breath_bpm=self._latest_vitals.breath_bpm,
            heart_bpm=self._latest_vitals.heart_bpm,
            breath_waveform=self._latest_vitals.breath_waveform,
            heart_waveform=self._latest_vitals.heart_waveform,
            quality=q,
            sample_for_trend=trend_sample,
            dsp_telemetry=dsp_telemetry,
            benchmark_elapsed=benchmark_elapsed,
        )

        # Waveform accumulation
        if self._latest_vitals.breath_waveform.size > 0:
            self._breath_waveform_accum.append(
                self._latest_vitals.breath_waveform[-1])
        if self._latest_vitals.heart_waveform.size > 0:
            self._heart_waveform_accum.append(
                self._latest_vitals.heart_waveform[-1])

        # CSV row accumulation (once per second)
        if trend_sample and q is not None:
            phase_range_raw = float(
                np.max(self._latest_vitals.breath_waveform)
                - np.min(self._latest_vitals.breath_waveform)
            ) if len(self._latest_vitals.breath_waveform) > 0 else 0.0

            sqi = 0
            br = q.get("breath_ratio", 0)
            pr = q.get("phase_range", 0)
            if pr >= 0.01 and br >= 0.15:
                sqi = 3
            elif pr >= 0.005 and br >= 0.05:
                sqi = 2
            elif pr > 0 or br > 0:
                sqi = 1

            elapsed_t = time.time() - start_time if start_time > 0 else 0
            self._bpm_history.append((
                elapsed_t,
                self._latest_vitals.breath_bpm,
                self._latest_vitals.heart_bpm,
            ))
            self._sqi_history.append({
                "phase_range": q.get("phase_range", 0.0),
                "breath_ratio": q.get("breath_ratio", 0.0),
                "sqi_level": sqi,
            })

            self._csv_rows.append({
                "Timestamp": datetime.now().isoformat(),
                "FrameIndex": self._latest_vitals.frame_index,
                "RangeBin": self._pipeline.best_range_bin if self._pipeline.best_range_bin is not None else 0,
                "RawPhase": round(phase_range_raw, 6),
                "BreathBPM": self._latest_vitals.breath_bpm,
                "HeartBPM": self._latest_vitals.heart_bpm,
                "PhaseRange": round(q.get("phase_range", 0), 6),
                "BreathRatio": round(q.get("breath_ratio", 0), 4),
                "HeartProminence": round(q.get("heart_prominence", 0), 4),
                "ApneaFlag": 1 if q.get("apnea_state") else 0,
                "SQI_Level": sqi,
            })

        # Status bar
        if q and not q.get("valid") and calib_done:
            status_label.setText(tr("status_signal_error"))
            status_label.setStyleSheet("color: #e74c3c;")
        else:
            status_label.setText(tr("status_monitoring"))
            status_label.setStyleSheet("color: #27ae60;")

    def tab_visibility(self) -> tuple[bool, bool, bool]:
        return (True, False, True)

    def get_export_data(self) -> dict:
        return {
            "csv_rows": list(self._csv_rows),
            "breath_waveform_accum": list(self._breath_waveform_accum),
            "heart_waveform_accum": list(self._heart_waveform_accum),
            "bpm_history": list(self._bpm_history),
            "sqi_history": list(self._sqi_history),
            "latest_vitals": self._latest_vitals,
        }

    def clear_data(self) -> None:
        self._csv_rows.clear()
        self._breath_waveform_accum.clear()
        self._heart_waveform_accum.clear()
        self._bpm_history.clear()
        self._sqi_history.clear()
        self._latest_vitals = None
        self._trend_tick_counter = 0


class BPMode(MonitorMode):
    """Blood pressure monitoring mode (1T1R, 32 range bins)."""

    def __init__(self):
        self._pipeline = None  # type: ignore  # BPPipeline
        self._latest_bp_result = None  # type: ignore  # BPResult
        self._bp_results: deque = deque(maxlen=720)
        self._csv_rows: deque = deque(maxlen=3600)

        # Strategy + benchmarker state
        self._pending_cleaner: SignalCleanerStrategy | None = None
        self._pending_ab_cleaner: SignalCleanerStrategy | None = None
        self._benchmarker: AlgorithmBenchmarker | None = None


    # -- MonitorMode impl ------------------------------------------------

    @property
    def uart_bins(self) -> int:
        return 32

    def boot_radar(self, radar_mgr) -> bool:
        return radar_mgr.boot_bp()

    def build_frame(self, fft_data: np.ndarray, frame_index: int) -> RadarFrame:
        """Build BP frame: 1T1R, 32 range bins."""
        rx_combined = fft_data.ravel()[:32]
        return RadarFrame(
            timestamp=time.time(),
            frame_index=frame_index,
            header=FrameHeader(0, 1, 1, 1, 60000, 32, 1, 160, 50, 0, 0, 0, 5),
            data_cube=rx_combined.reshape(32, 1, 1),
        )

    def start(self, calib_sbp: float = 0.0, calib_dbp: float = 0.0) -> None:
        """Create and start BP pipeline, optionally injecting calibration offsets."""
        from bp_monitor.bp_pipeline import BPPipeline
        cleaner = self._pending_cleaner or EMDPulseCleaner()
        self._pipeline = BPPipeline(
            "bp_matlab/bp_weights.mat",
            cleaner=cleaner,
        )
        self._pipeline.set_calibration(calib_sbp, calib_dbp)
        if self._pending_ab_cleaner is not None:
            self._pipeline.set_ab_strategy(self._pending_ab_cleaner)
        if self._benchmarker is not None:
            self._pipeline.set_benchmarker(self._benchmarker)
        self._pipeline.start()

    def stop(self) -> None:
        if self._pipeline:
            _drain_queue(self._pipeline.display_queue)
            self._pipeline.stop()
            self._pipeline = None

    # ── Strategy control ─────────────────────────────────────────

    def set_strategies(self, cleaner: SignalCleanerStrategy) -> None:
        self._pending_cleaner = cleaner
        if self._pipeline is not None:
            self._pipeline.set_strategies(cleaner)

    def set_ab_strategy(self, cleaner: SignalCleanerStrategy | None) -> None:
        self._pending_ab_cleaner = cleaner
        if self._pipeline is not None:
            self._pipeline.set_ab_strategy(cleaner)

    def toggle_benchmark(self) -> bool:
        if self._benchmarker is None:
            self._benchmarker = AlgorithmBenchmarker()
        if self._benchmarker.is_recording:
            self._benchmarker.stop()
            return False
        else:
            self._benchmarker.start()
            if self._pipeline is not None:
                self._pipeline.set_benchmarker(self._benchmarker)
            return True

    def get_dsp_telemetry(self) -> dict:
        if self._pipeline is None:
            return {
                "current_algo": "--",
                "current_latency_ms": 0.0,
                "current_snr_gain_db": 0.0,
                "ab_algo": "",
                "ab_latency_ms": 0.0,
                "ab_snr_gain_db": 0.0,
                "ab_enabled": False,
            }
        return self._pipeline.get_dsp_telemetry()

    def get_benchmarker(self) -> AlgorithmBenchmarker | None:
        return self._benchmarker

    def get_benchmark_elapsed(self) -> float:
        if self._benchmarker is None or not self._benchmarker.is_recording:
            return 0.0
        return time.time() - self._benchmarker._start_time

    def feed_frame(self, frame: RadarFrame) -> None:
        if self._pipeline is None:
            return
        try:
            self._pipeline.raw_queue.put_nowait(frame)
        except queue.Full:
            with self._pipeline.raw_queue.mutex:
                self._pipeline.raw_queue.queue.clear()
            self._pipeline.raw_queue.put_nowait(frame)

    def poll_and_update(self, subject_tab, bp_tab, research_tab,
                        status_label, elapsed_label, frame_rate_label,
                        start_time, frame_count) -> None:
        if self._pipeline is None:
            return

        new_result = False
        try:
            while not self._pipeline.display_queue.empty():
                self._latest_bp_result = self._pipeline.display_queue.get_nowait()
                new_result = True
        except queue.Empty:
            pass

        if self._latest_bp_result is not None:
            r = self._latest_bp_result

            # 【修复点】：只有在获取到新结果时，才更新图表和追加记录
            if new_result:
                bp_tab.update_display(r)
                self._bp_results.append(r)

                # Record valid BP readings for export
                if not np.isnan(r.sbp):
                    self._csv_rows.append({
                        "Timestamp": datetime.now().isoformat(),
                        "FrameIndex": r.frame_index,
                        "SBP": round(r.sbp, 2),
                        "DBP": round(r.dbp, 2),
                        "Distance_m": round(r.target_distance_m, 2),
                        "Confidence": round(r.quality.get("confidence", 0.0), 4) if r.quality else 0.0,
                    })

            # Research tab 和 telemetry 可以继续每帧刷新
            dsp_telemetry = self.get_dsp_telemetry()
            benchmark_elapsed = self.get_benchmark_elapsed()
            research_tab.update_display(
                breath_bpm=0.0,
                heart_bpm=0.0,
                breath_waveform=np.array([]),
                heart_waveform=np.array([]),
                quality=r.quality,
                sample_for_trend=False,
                dsp_telemetry=dsp_telemetry,
                benchmark_elapsed=benchmark_elapsed,
            )

            status_label.setText("● Monitoring")
            status_label.setStyleSheet("color: #27ae60;")

    def tab_visibility(self) -> tuple[bool, bool, bool]:
        return (False, True, True)  # show BP tab + Research tab

    def get_export_data(self) -> dict:
        return {
            "csv_rows": list(self._csv_rows),
            "bp_results": list(self._bp_results),
        }

    def get_recent_bp_stats(self, seconds: float = 5.0) -> tuple[
        float | None, float | None, float | None, float | None]:
        """Return mean and standard deviation for SBP/DBP from _bp_results within the last N seconds.

        Returns (mean_sbp, mean_dbp, std_sbp, std_dbp).
        Returns (None, None, None, None) if no valid data in the window.
        """
        now = time.time()
        cutoff = now - seconds
        sbp_vals = []
        dbp_vals = []
        for r in self._bp_results:
            if r.timestamp >= cutoff:
                if not np.isnan(r.sbp):
                    sbp_vals.append(r.sbp)
                if not np.isnan(r.dbp):
                    dbp_vals.append(r.dbp)
        if not sbp_vals or not dbp_vals:
            return None, None, None, None

        # 计算并返回：收缩压均值、舒张压均值、收缩压标准差、舒张压标准差
        return (
            float(np.mean(sbp_vals)), float(np.mean(dbp_vals)),
            float(np.std(sbp_vals)), float(np.std(dbp_vals))
        )

    def clear_data(self) -> None:
        self._bp_results.clear()
        self._csv_rows.clear()
        self._latest_bp_result = None
