"""Monitor mode abstraction — each vital-signs mode (HR, BP) as a Strategy object."""

import threading
import time
import queue
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from config.protocol import UI_REFRESH_MS
from config.i18n import tr
from models.radar_frame import RadarFrame, FrameHeader
from dsp_pipeline.vital_signs import VitalSigns


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

        # Data accumulation for export
        self._csv_rows: list[dict] = []
        self._breath_waveform_accum: list[np.ndarray] = []
        self._heart_waveform_accum: list[np.ndarray] = []
        self._bpm_history: list[tuple[float, float, float]] = []
        self._sqi_history: list[dict] = []

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
            header=FrameHeader(0, 1, 4, 2, 58000, 128, 1, 3000, 25, 1920, 60),
            data_cube=rx_combined.reshape(-1, 1, 1),
        )

    def start(self) -> None:
        from dsp_pipeline.pipeline import Pipeline
        self._pipeline = Pipeline()
        self._pipeline.start()

    def stop(self) -> None:
        if self._pipeline:
            _drain_queue(self._pipeline.display_queue)
            self._pipeline.stop()
            self._pipeline = None

    def feed_frame(self, frame: RadarFrame) -> None:
        if self._pipeline is None:
            return
        while True:
            try:
                self._pipeline.raw_queue.put_nowait(frame)
                break
            except queue.Full:
                try:
                    self._pipeline.raw_queue.get_nowait()
                except queue.Empty:
                    pass

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

        # Subject tab
        subject_tab.update_display(
            breath_bpm=self._latest_vitals.breath_bpm,
            heart_bpm=self._latest_vitals.heart_bpm,
            breath_waveform=self._latest_vitals.breath_waveform,
            quality=q,
            calibration_done=calib_done,
            calibration_progress=calib_prog,
        )

        # Research tab
        self._trend_tick_counter += 1
        trend_sample = (self._trend_tick_counter % 20 == 0)
        research_tab.update_display(
            breath_bpm=self._latest_vitals.breath_bpm,
            heart_bpm=self._latest_vitals.heart_bpm,
            breath_waveform=self._latest_vitals.breath_waveform,
            heart_waveform=self._latest_vitals.heart_waveform,
            quality=q,
            sample_for_trend=trend_sample,
        )

        # Waveform accumulation
        if self._latest_vitals.breath_waveform.size > 0:
            self._breath_waveform_accum.append(
                self._latest_vitals.breath_waveform.copy())
        if self._latest_vitals.heart_waveform.size > 0:
            self._heart_waveform_accum.append(
                self._latest_vitals.heart_waveform.copy())

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
            "csv_rows": self._csv_rows,
            "breath_waveform_accum": self._breath_waveform_accum,
            "heart_waveform_accum": self._heart_waveform_accum,
            "bpm_history": self._bpm_history,
            "sqi_history": self._sqi_history,
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
