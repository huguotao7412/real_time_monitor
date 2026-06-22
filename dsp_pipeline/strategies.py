"""Pluggable DSP strategy interfaces and concrete algorithm implementations.

Strategy Pattern for academic ablation studies:
  - SignalCleanerStrategy: single-channel denoising / harmonic removal
  - VitalSignSeparator:  breath / heartbeat waveform separation
  - AdaptiveStrategySelector: SQI-driven routing between strategy pairs
"""

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.signal import savgol_filter


class SignalCleanerStrategy(ABC):
    """Single-channel signal denoising / harmonic removal.

    Each concrete implementation wraps one algorithm (VMD+RLS, EMD, etc.)
    and is responsible for its own error handling.
    """

    @abstractmethod
    def clean(self, signal: np.ndarray, fs: float) -> tuple[np.ndarray, dict[str, Any]]:
        """Clean the input signal and return metrics for academic analysis.

        Args:
            signal: 1D real-valued input signal.
            fs: Sampling rate in Hz.

        Returns:
            cleaned: Cleaned signal, same shape as input. On error, returns input.
            metrics: Dictionary with keys:
                algorithm (str), latency_ms (float),
                imf_count (int|None), convergence_iter (int|None),
                retained_harmonics (list[int]|None), dominant_freq_hz (float|None),
                snr_estimate_db (float|None).
        """


class VitalSignSeparator(ABC):
    """Breath / heartbeat waveform separation from cleaned displacement."""

    @abstractmethod
    def separate(
        self,
        signal: np.ndarray,
        fs: float,
        heart_input_signal: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Separate breath and heart waveforms.

        Args:
            signal: 1D cleaned displacement signal.
            fs: Sampling rate in Hz.
            heart_input_signal: Optional pre-processed heart input (e.g. diff).
                If None, the separator derives its own.

        Returns:
            breath_wave: 1D respiratory waveform.
            heart_wave: 1D heartbeat waveform.
            metrics: Dictionary with keys:
                algorithm (str), latency_ms (float),
                breath_band_energy_ratio (float|None),
                heart_band_energy_ratio (float|None).
        """


# ─────────────────────────────────────────────────────────────
# Concrete SignalCleanerStrategy implementations
# ─────────────────────────────────────────────────────────────


class VMDRLSCleaner(SignalCleanerStrategy):
    """VMD decomposition + RLS adaptive harmonic cancellation."""

    def clean(self, signal: np.ndarray, fs: float) -> tuple[np.ndarray, dict[str, Any]]:
        from dsp_pipeline.vmd_rls_cleaner import vmd_rls_harmonic_clean

        t0 = time.perf_counter()
        metrics: dict[str, Any] = {
            "algorithm": "VMD_RLS",
            "imf_count": None,
            "convergence_iter": None,
            "retained_harmonics": [2, 3, 4],
            "dominant_freq_hz": None,
            "snr_estimate_db": None,
        }
        try:
            cleaned = vmd_rls_harmonic_clean(signal, fs, harmonics=[2, 3, 4])
        except Exception as e:
            metrics["error"] = str(e)
            cleaned = signal.copy()
        metrics["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return cleaned.astype(np.float64), metrics


class EMDHarmonicCleaner(SignalCleanerStrategy):
    """EMD-based harmonic interference cancellation via least squares."""

    def clean(self, signal: np.ndarray, fs: float) -> tuple[np.ndarray, dict[str, Any]]:
        from dsp_pipeline.emd_cleaner import emd_harmonic_clean

        t0 = time.perf_counter()
        metrics: dict[str, Any] = {
            "algorithm": "EMD_Harmonic",
            "imf_count": 6,
            "convergence_iter": None,
            "retained_harmonics": [2, 3, 4],
            "dominant_freq_hz": None,
            "snr_estimate_db": None,
        }
        try:
            cleaned = emd_harmonic_clean(signal, fs)
        except Exception as e:
            metrics["error"] = str(e)
            cleaned = signal.copy()
        metrics["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return cleaned.astype(np.float64), metrics


class EMDPulseCleaner(SignalCleanerStrategy):
    """EMD + wavelet denoising for pulse wave extraction (BP pipeline)."""

    def clean(self, signal: np.ndarray, fs: float) -> tuple[np.ndarray, dict[str, Any]]:
        from bp_monitor.bp_signal_cleaner import clean_pulse_wave

        t0 = time.perf_counter()
        metrics: dict[str, Any] = {
            "algorithm": "EMD_Pulse",
            "imf_count": 6,
            "convergence_iter": None,
            "retained_harmonics": [2, 3, 4],
            "dominant_freq_hz": None,
            "snr_estimate_db": None,
        }
        try:
            cleaned = clean_pulse_wave(signal, fs=fs)
        except Exception as e:
            metrics["error"] = str(e)
            cleaned = signal.copy()
        metrics["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return cleaned.astype(np.float64), metrics


class PassthroughCleaner(SignalCleanerStrategy):
    """Identity cleaner — ablation baseline (no harmonic removal)."""

    def clean(self, signal: np.ndarray, fs: float) -> tuple[np.ndarray, dict[str, Any]]:
        t0 = time.perf_counter()
        metrics: dict[str, Any] = {
            "algorithm": "Passthrough",
            "imf_count": 0,
            "convergence_iter": None,
            "retained_harmonics": None,
            "dominant_freq_hz": None,
            "snr_estimate_db": None,
        }
        metrics["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return signal.copy().astype(np.float64), metrics


# ─────────────────────────────────────────────────────────────
# Concrete VitalSignSeparator implementations
# ─────────────────────────────────────────────────────────────


class WPDSeparator(VitalSignSeparator):
    """Wavelet Packet Decomposition for breath/heart separation."""

    def separate(
        self,
        signal: np.ndarray,
        fs: float,
        heart_input_signal: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        from dsp_pipeline.wpd_filter import wpd_separate

        t0 = time.perf_counter()
        metrics: dict[str, Any] = {
            "algorithm": "WPD",
            "breath_band_energy_ratio": None,
            "heart_band_energy_ratio": None,
        }
        try:
            breath_wave, heart_wave = wpd_separate(
                signal, fs, heart_input_signal=heart_input_signal,
            )
        except Exception as e:
            metrics["error"] = str(e)
            breath_wave = signal.copy()
            heart_wave = np.zeros_like(signal)
        metrics["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return breath_wave.astype(np.float64), heart_wave.astype(np.float64), metrics


class SOSFilterSeparator(VitalSignSeparator):
    """SOS bandpass filter separation — lightweight baseline."""

    def __init__(self, fs: float = 20.0):
        from dsp_pipeline.filters import VitalSignFilter
        self._filter = VitalSignFilter(fs=fs)

    def separate(
        self,
        signal: np.ndarray,
        fs: float,
        heart_input_signal: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        t0 = time.perf_counter()
        metrics: dict[str, Any] = {
            "algorithm": "SOS",
            "breath_band_energy_ratio": None,
            "heart_band_energy_ratio": None,
        }
        try:
            breath_wave = self._filter.filter_breath(signal)
            if heart_input_signal is not None:
                heart_input = heart_input_signal
            else:
                heart_input = savgol_filter(signal, window_length=9, polyorder=3, deriv=1)
            heart_wave = self._filter.filter_heart(heart_input)
        except Exception as e:
            metrics["error"] = str(e)
            breath_wave = signal.copy()
            heart_wave = np.zeros_like(signal)
        metrics["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return breath_wave.astype(np.float64), heart_wave.astype(np.float64), metrics


# ─────────────────────────────────────────────────────────────
# Adaptive SQI-driven strategy selector
# ─────────────────────────────────────────────────────────────


class AdaptiveStrategySelector:
    """SQI-driven dynamic routing of cleaner + separator pairs.

    Decision thresholds (publishable innovation):
      - phase_range > HIGH_THRESH & snr > HIGH_SNR → light (Passthrough + SOS)
      - LOW_THRESH < phase_range ≤ HIGH_THRESH        → standard (VMD+RLS + WPD)
      - phase_range ≤ LOW_THRESH or snr < LOW_SNR     → heavy (EMD + WPD)

    Re-evaluates every `evaluation_interval` calls to select() to
    prevent output jitter from frequent switching.
    """

    def __init__(
        self,
        light: tuple[SignalCleanerStrategy, VitalSignSeparator],
        standard: tuple[SignalCleanerStrategy, VitalSignSeparator],
        heavy: tuple[SignalCleanerStrategy, VitalSignSeparator],
        evaluation_interval: int = 50,
        high_phase_threshold: float = 0.02,
        low_phase_threshold: float = 0.005,
        high_snr_db: float = 15.0,
        low_snr_db: float = 5.0,
    ):
        self._light = light
        self._standard = standard
        self._heavy = heavy
        self._interval = evaluation_interval
        self._high_phase = high_phase_threshold
        self._low_phase = low_phase_threshold
        self._high_snr = high_snr_db
        self._low_snr = low_snr_db
        self._call_count = 0
        self._cached: tuple[SignalCleanerStrategy, VitalSignSeparator] = standard

    def select(
        self, phase_range: float, snr_db: float
    ) -> tuple[SignalCleanerStrategy, VitalSignSeparator]:
        """Return the optimal (cleaner, separator) pair for current SQI.

        Only re-evaluates every `evaluation_interval` calls.
        """
        self._call_count += 1
        if self._call_count % self._interval != 0:
            return self._cached

        if phase_range > self._high_phase and snr_db > self._high_snr:
            self._cached = self._light
        elif phase_range <= self._low_phase or snr_db < self._low_snr:
            self._cached = self._heavy
        else:
            self._cached = self._standard
        return self._cached
