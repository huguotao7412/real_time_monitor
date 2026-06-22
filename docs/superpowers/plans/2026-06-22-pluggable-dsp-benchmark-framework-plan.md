# Pluggable DSP Benchmark Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor hardcoded DSP pipelines into a Strategy Pattern-driven benchmark framework with real-time algorithm comparison, adaptive routing, and automated academic data logging.

**Architecture:** Two new files (`dsp_pipeline/strategies.py`, `utils/benchmark_logger.py`) define abstract strategy interfaces, 6 concrete algorithm wrappers, an adaptive SQI-driven selector, and an async benchmark recorder. Pipeline classes (`pipeline.py`, `bp_pipeline.py`) accept injected strategies via constructor; the UI layer gains an algorithm control panel for live switching and A/B comparison recording.

**Tech Stack:** Python 3.12.10 · NumPy · SciPy · PyQt6 · h5py · vmdpy · PyEMD · PyWavelets · PyTorch

## Global Constraints

- Target platform: Python 3.12.10 on Windows 11
- No new PyPI packages beyond those already in `requirements.txt`
- Strategy error handling: each strategy catches its own exceptions internally; callers do NOT wrap in `try...except`
- `_use_advanced_dsp` flag is removed and replaced by `AdaptiveStrategySelector`
- Benchmarker always starts in stopped state; requires explicit UI click to begin recording
- A/B comparison is HR-mode only (not BP mode)
- Hot-swap deferred: strategy changes take effect on next `update_bpm` dispatch, never mid-computation

---

### Task 1: Create Strategy Module — ABCs and 6 Concrete Classes

**Files:**
- Create: `dsp_pipeline/strategies.py`
- Test: `dsp_pipeline/test_strategies.py`

**Interfaces:**
- Produces:
  - `SignalCleanerStrategy` (ABC) — `clean(signal, fs) -> tuple[np.ndarray, dict]`
  - `VitalSignSeparator` (ABC) — `separate(signal, fs, heart_input_signal=None) -> tuple[np.ndarray, np.ndarray, dict]`
  - `VMDRLSCleaner(SignalCleanerStrategy)`
  - `EMDHarmonicCleaner(SignalCleanerStrategy)`
  - `EMDPulseCleaner(SignalCleanerStrategy)`
  - `PassthroughCleaner(SignalCleanerStrategy)`
  - `WPDSeparator(VitalSignSeparator)`
  - `SOSFilterSeparator(VitalSignSeparator)`
  - `AdaptiveStrategySelector` — `select(phase_range, snr_db) -> tuple[SignalCleanerStrategy, VitalSignSeparator]`

- [ ] **Step 1: Create the test file with tests for all strategies**

```python
# dsp_pipeline/test_strategies.py
import numpy as np
import pytest
from dsp_pipeline.strategies import (
    SignalCleanerStrategy, VitalSignSeparator,
    VMDRLSCleaner, EMDHarmonicCleaner, EMDPulseCleaner, PassthroughCleaner,
    WPDSeparator, SOSFilterSeparator, AdaptiveStrategySelector,
)


def _make_test_signal(n: int = 500, fs: float = 20.0) -> np.ndarray:
    """Generate a synthetic displacement signal with breath + heart + noise."""
    t = np.arange(n) / fs
    breath = 0.5 * np.sin(2 * np.pi * 0.3 * t)       # 18 bpm
    heart = 0.1 * np.sin(2 * np.pi * 1.2 * t)         # 72 bpm
    noise = 0.02 * np.random.randn(n)
    return breath + heart + noise


# ── SignalCleanerStrategy tests ──
CLEANER_CLASSES = [VMDRLSCleaner, EMDHarmonicCleaner, EMDPulseCleaner, PassthroughCleaner]

@pytest.mark.parametrize("cls", CLEANER_CLASSES)
def test_cleaner_returns_correct_shape(cls):
    """Every cleaner must return (cleaned_signal, metrics_dict)."""
    signal = _make_test_signal()
    cleaner = cls()
    cleaned, metrics = cleaner.clean(signal, fs=20.0)
    assert cleaned.shape == signal.shape
    assert isinstance(cleaned, np.ndarray)
    assert isinstance(metrics, dict)
    assert "algorithm" in metrics
    assert "latency_ms" in metrics
    assert metrics["latency_ms"] >= 0.0


@pytest.mark.parametrize("cls", CLEANER_CLASSES)
def test_cleaner_handles_short_signal(cls):
    """Cleaners must not crash on short signals (< 128 samples)."""
    signal = np.random.randn(50)
    cleaner = cls()
    cleaned, metrics = cleaner.clean(signal, fs=20.0)
    assert cleaned.shape == signal.shape
    assert "algorithm" in metrics


@pytest.mark.parametrize("cls", CLEANER_CLASSES)
def test_cleaner_handles_empty_signal(cls):
    """Cleaners must handle zero-length input gracefully."""
    signal = np.array([])
    cleaner = cls()
    try:
        cleaned, metrics = cleaner.clean(signal, fs=20.0)
        assert cleaned.shape == signal.shape
    except ValueError:
        pass  # acceptable: some algorithms require min length


def test_passthrough_cleaner_identity():
    """PassthroughCleaner must return input unchanged."""
    signal = _make_test_signal()
    cleaner = PassthroughCleaner()
    cleaned, _ = cleaner.clean(signal, fs=20.0)
    np.testing.assert_array_equal(cleaned, signal)


# ── VitalSignSeparator tests ──
SEPARATOR_CLASSES = [WPDSeparator, SOSFilterSeparator]

@pytest.mark.parametrize("cls", SEPARATOR_CLASSES)
def test_separator_returns_two_waveforms(cls):
    """Every separator must return (breath_wave, heart_wave, metrics)."""
    signal = _make_test_signal()
    sep = cls()
    breath, heart, metrics = sep.separate(signal, fs=20.0)
    assert isinstance(breath, np.ndarray)
    assert isinstance(heart, np.ndarray)
    assert isinstance(metrics, dict)
    assert "algorithm" in metrics
    assert "latency_ms" in metrics


@pytest.mark.parametrize("cls", SEPARATOR_CLASSES)
def test_separator_heart_input_signal(cls):
    """Separator must accept optional heart_input_signal."""
    signal = _make_test_signal()
    heart_in = np.diff(signal)
    sep = cls()
    breath, heart, metrics = sep.separate(signal, fs=20.0, heart_input_signal=heart_in)
    assert breath.shape[0] == signal.shape[0]


# ── AdaptiveStrategySelector tests ──

def test_adaptive_selector_light_path():
    """Strong signal -> light path (Passthrough + SOS)."""
    selector = AdaptiveStrategySelector(
        light=(PassthroughCleaner(), SOSFilterSeparator()),
        standard=(VMDRLSCleaner(), WPDSeparator()),
        heavy=(EMDHarmonicCleaner(), WPDSeparator()),
    )
    cleaner, sep = selector.select(phase_range=0.03, snr_db=20.0)
    assert isinstance(cleaner, PassthroughCleaner)
    assert isinstance(sep, SOSFilterSeparator)


def test_adaptive_selector_standard_path():
    """Moderate signal -> standard path (VMD+RLS + WPD)."""
    selector = AdaptiveStrategySelector(
        light=(PassthroughCleaner(), SOSFilterSeparator()),
        standard=(VMDRLSCleaner(), WPDSeparator()),
        heavy=(EMDHarmonicCleaner(), WPDSeparator()),
    )
    cleaner, sep = selector.select(phase_range=0.01, snr_db=10.0)
    assert isinstance(cleaner, VMDRLSCleaner)
    assert isinstance(sep, WPDSeparator)


def test_adaptive_selector_heavy_path():
    """Weak signal -> heavy path (EMD + WPD)."""
    selector = AdaptiveStrategySelector(
        light=(PassthroughCleaner(), SOSFilterSeparator()),
        standard=(VMDRLSCleaner(), WPDSeparator()),
        heavy=(EMDHarmonicCleaner(), WPDSeparator()),
    )
    cleaner, sep = selector.select(phase_range=0.002, snr_db=3.0)
    assert isinstance(cleaner, EMDHarmonicCleaner)
    assert isinstance(sep, WPDSeparator)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -m pytest dsp_pipeline/test_strategies.py -v --tb=short 2>&1 | head -30
```

Expected: all tests FAIL with `ModuleNotFoundError: No module named 'dsp_pipeline.strategies'`

- [ ] **Step 3: Create `dsp_pipeline/strategies.py` — imports and ABCs**

```python
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
```

- [ ] **Step 4: Append concrete cleaner implementations to `strategies.py`**

```python
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
```

- [ ] **Step 5: Append concrete separator implementations to `strategies.py`**

```python
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
```

- [ ] **Step 6: Append AdaptiveStrategySelector to `strategies.py`**

```python
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
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -m pytest dsp_pipeline/test_strategies.py -v --tb=short
```

Expected: at least 11 tests PASS (3 parametrized cleaner tests × 4 classes = 12, but empty-signal test may xfail for some; plus 2 separator tests × 2 classes = 4; plus 3 AdaptiveSelector tests; plus 1 passthrough identity test).

- [ ] **Step 8: Commit**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && git add dsp_pipeline/strategies.py dsp_pipeline/test_strategies.py && git commit -m "feat: add pluggable DSP strategy interfaces and 6 concrete algorithm wrappers"
```

---

### Task 2: Create Benchmark Logger Module

**Files:**
- Create: `utils/benchmark_logger.py`
- Test: `utils/test_benchmark_logger.py`

**Interfaces:**
- Produces:
  - `BenchmarkRecord` (dataclass) — all fields as defined below
  - `AlgorithmBenchmarker` — `start()`, `stop()`, `log(record)`, `mark_ground_truth(label, ts)`, `is_recording`, `record_count`, `session_path`

- [ ] **Step 1: Create the test file**

```python
# utils/test_benchmark_logger.py
import os
import time
import tempfile
import numpy as np
import pytest
from utils.benchmark_logger import BenchmarkRecord, AlgorithmBenchmarker


def _make_record(frame_idx: int = 0, algo: str = "VMD_RLS+WPD") -> BenchmarkRecord:
    return BenchmarkRecord(
        timestamp=time.time(),
        frame_index=frame_idx,
        elapsed_sec=float(frame_idx) * 0.05,
        algorithm_name=algo,
        is_primary=True,
        latency_ms=12.5,
        input_phase_range=0.015,
        input_snr_db=8.0,
        output_phase_range=0.018,
        output_snr_db=12.0,
        snr_gain_db=4.0,
        breath_bpm=15.2,
        heart_bpm=72.0,
        heart_prominence=0.65,
        imf_count=4,
        convergence_iter=200,
        retained_harmonics="2,3,4",
        dominant_freq_hz=0.25,
    )


def test_record_all_fields_defaultable():
    """Verify BenchmarkRecord optional fields default to None."""
    r = BenchmarkRecord(
        timestamp=time.time(),
        frame_index=0,
        elapsed_sec=0.0,
        algorithm_name="Test",
        is_primary=True,
        latency_ms=0.0,
        input_phase_range=0.0,
        input_snr_db=None,
        output_phase_range=0.0,
        output_snr_db=None,
        snr_gain_db=None,
        breath_bpm=0.0,
        heart_bpm=0.0,
        heart_prominence=0.0,
    )
    assert r.imf_count is None
    assert r.convergence_iter is None
    assert r.retained_harmonics is None
    assert r.dominant_freq_hz is None


def test_benchmarker_start_stop():
    """Benchmarker must start and stop cleanly, creating output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bm = AlgorithmBenchmarker(output_dir=tmpdir)
        assert not bm.is_recording
        bm.start()
        assert bm.is_recording
        assert os.path.isdir(bm.session_path)

        # Log a few records
        for i in range(5):
            bm.log(_make_record(frame_idx=i))

        bm.stop()
        assert not bm.is_recording

        # Verify CSV file was created with correct columns
        csv_path = os.path.join(bm.session_path, "benchmark.csv")
        assert os.path.isfile(csv_path)
        with open(csv_path, "r") as f:
            header = f.readline().strip()
        expected_cols = "timestamp,elapsed_sec,frame_index,algorithm,is_primary"
        assert header.startswith(expected_cols)


def test_benchmarker_queue_overflow():
    """When queue is full, oldest records are dropped (no blocking)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bm = AlgorithmBenchmarker(output_dir=tmpdir)
        bm.start()
        # Send many records quickly
        for i in range(100):
            bm.log(_make_record(frame_idx=i))
        # Should not hang or crash
        bm.stop()
        # Some records should have been written
        assert bm.record_count > 0


def test_ground_truth_mark():
    """Ground truth marks must create a separate CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bm = AlgorithmBenchmarker(output_dir=tmpdir)
        bm.start()
        bm.mark_ground_truth("cuff_sbp=118")
        bm.mark_ground_truth("cuff_dbp=76")
        bm.stop()
        gt_path = os.path.join(bm.session_path, "ground_truth.csv")
        assert os.path.isfile(gt_path)
        with open(gt_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 entries


def test_benchmarker_idempotent_start():
    """Calling start() twice must not crash or create duplicate threads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bm = AlgorithmBenchmarker(output_dir=tmpdir)
        bm.start()
        bm.start()  # second call should be a no-op
        bm.log(_make_record())
        bm.stop()
        assert bm.record_count >= 1


def test_benchmarker_stop_when_not_recording():
    """Calling stop() when not recording must not crash."""
    bm = AlgorithmBenchmarker()
    bm.stop()  # no-op
    assert not bm.is_recording
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -m pytest utils/test_benchmark_logger.py -v --tb=short 2>&1 | head -20
```

Expected: all FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `utils/__init__.py` if missing**

```bash
ls "F:\yan\mmradarcoverage\real_time_monitor\utils" 2>/dev/null || mkdir "F:\yan\mmradarcoverage\real_time_monitor\utils"
```

If the `utils/` directory does not exist, create it. Also create `utils/__init__.py`:

```python
# utils/__init__.py
```

- [ ] **Step 4: Create `utils/benchmark_logger.py` — imports and BenchmarkRecord**

```python
"""Academic benchmark logger for DSP ablation studies.

Non-blocking, thread-safe recording of per-frame per-algorithm metrics
to CSV (streaming) and HDF5 (batch flush). Designed for direct use in
paper figures — no post-processing required.
"""

import csv
import os
import queue
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class BenchmarkRecord:
    """Single-frame, single-algorithm benchmark data point."""

    # ── Time ──
    timestamp: float
    frame_index: int
    elapsed_sec: float

    # ── Algorithm identity ──
    algorithm_name: str
    is_primary: bool              # True = drives UI; False = A/B silent
    latency_ms: float

    # ── Signal features ──
    input_phase_range: float
    input_snr_db: float | None
    output_phase_range: float
    output_snr_db: float | None
    snr_gain_db: float | None

    # ── Extracted vitals ──
    breath_bpm: float
    heart_bpm: float
    heart_prominence: float

    # ── Algorithm internals (None = N/A) ──
    imf_count: int | None = None
    convergence_iter: int | None = None
    retained_harmonics: str | None = None   # "2,3,4"
    dominant_freq_hz: float | None = None

    # CSV column order, kept in sync with __post_init__ / log writing
    CSV_COLUMNS = [
        "timestamp", "elapsed_sec", "frame_index", "algorithm_name",
        "is_primary", "latency_ms",
        "input_phase_range", "input_snr_db", "output_phase_range",
        "output_snr_db", "snr_gain_db",
        "breath_bpm", "heart_bpm", "heart_prominence",
        "imf_count", "convergence_iter", "retained_harmonics",
        "dominant_freq_hz",
    ]
```

- [ ] **Step 5: Append AlgorithmBenchmarker class to `utils/benchmark_logger.py`**

```python
class AlgorithmBenchmarker:
    """Async benchmark recorder — queue → writer thread → CSV + HDF5.

    Thread model:
        DSP thread(s) → queue.Queue(maxsize=5000) → Writer thread
                           (drop oldest if full)       ├─ CSV (per-record append)
                                                       └─ HDF5 (batch flush / 100)

    Usage:
        bm = AlgorithmBenchmarker("data/benchmarks")
        bm.start()
        bm.log(record)          # non-blocking
        bm.mark_ground_truth("cuff_sbp=118")
        bm.stop()
    """

    _QUEUE_MAXSIZE = 5000
    _HDF5_FLUSH_BATCH = 100

    def __init__(
        self,
        output_dir: str = "data/benchmarks",
        session_id: str | None = None,
    ):
        self._output_dir = output_dir
        self._session_id = session_id or datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        self._session_path = os.path.join(self._output_dir, self._session_id)

        self._queue: queue.Queue[BenchmarkRecord | str] = queue.Queue(
            maxsize=self._QUEUE_MAXSIZE
        )
        self._writer_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._recording = False
        self._record_count = 0
        self._start_time: float = 0.0

        # File handles (opened by writer thread)
        self._csv_file = None
        self._csv_writer = None
        self._h5_file = None
        self._h5_batch: list[BenchmarkRecord] = []
        self._h5_lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def session_path(self) -> str:
        return self._session_path

    def start(self) -> None:
        """Start the writer thread and open output files."""
        if self._recording:
            return  # idempotent
        os.makedirs(self._session_path, exist_ok=True)
        self._start_time = time.time()
        self._stop_event.clear()
        self._recording = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="benchmark-writer"
        )
        self._writer_thread.start()

    def stop(self) -> None:
        """Flush all buffers, close files, join writer thread."""
        if not self._recording:
            return
        self._stop_event.set()
        # Send sentinel to unblock writer
        try:
            self._queue.put_nowait("__STOP__")
        except queue.Full:
            pass
        if self._writer_thread:
            self._writer_thread.join(timeout=5.0)
        self._recording = False

    def log(self, record: BenchmarkRecord) -> None:
        """Enqueue a record for writing. Non-blocking.

        If the queue is full, the oldest record is silently dropped
        to prevent DSP pipeline back-pressure.
        """
        if not self._recording:
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # Drop oldest, retry
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                pass  # truly cannot keep up — drop this record

    def mark_ground_truth(
        self, label: str, timestamp: float | None = None
    ) -> None:
        """Record an external reference event (e.g., cuff measurement).

        Enqueues a special string token that the writer thread writes
        to ground_truth.csv.
        """
        if not self._recording:
            return
        ts = timestamp if timestamp is not None else time.time()
        try:
            self._queue.put_nowait(f"GT:{ts:.6f}:{label}")
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(f"GT:{ts:.6f}:{label}")
            except queue.Full:
                pass

    # ── Writer thread ───────────────────────────────────────

    def _writer_loop(self) -> None:
        """Writer thread main loop. Opens files, drains queue, flushes."""
        self._open_files()
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if isinstance(item, str):
                    if item == "__STOP__":
                        break
                    elif item.startswith("GT:"):
                        self._write_ground_truth(item)
                elif isinstance(item, BenchmarkRecord):
                    self._write_csv(item)
                    self._buffer_hdf5(item)
                    self._record_count += 1
        finally:
            self._flush_hdf5()
            self._close_files()

    def _open_files(self) -> None:
        """Open CSV and HDF5 file handles."""
        # CSV
        csv_path = os.path.join(self._session_path, "benchmark.csv")
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=BenchmarkRecord.CSV_COLUMNS
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()

        # Ground truth CSV
        gt_path = os.path.join(self._session_path, "ground_truth.csv")
        self._gt_file = open(gt_path, "w", newline="", encoding="utf-8-sig")
        self._gt_writer = csv.writer(self._gt_file)
        self._gt_writer.writerow(["timestamp", "elapsed_sec", "label"])
        self._gt_file.flush()

        # HDF5 (opened lazily on first record to avoid empty files)
        self._h5_file = None

    def _write_csv(self, record: BenchmarkRecord) -> None:
        """Stream a single row to the CSV file."""
        d = asdict(record)
        # asdict uses field names directly; ensure they match CSV_COLUMNS
        row = {k: d[k] for k in BenchmarkRecord.CSV_COLUMNS}
        self._csv_writer.writerow(row)
        self._csv_file.flush()  # streaming append

    def _write_ground_truth(self, token: str) -> None:
        """Parse a GT token and write to ground_truth.csv."""
        # Format: "GT:1700000000.123456:cuff_sbp=118"
        parts = token[3:].split(":", 1)  # skip "GT:"
        if len(parts) != 2:
            return
        ts_str, label = parts
        try:
            ts = float(ts_str)
            elapsed = ts - self._start_time if self._start_time > 0 else 0.0
        except ValueError:
            return
        self._gt_writer.writerow([ts, elapsed, label])
        self._gt_file.flush()

    def _buffer_hdf5(self, record: BenchmarkRecord) -> None:
        """Accumulate records; flush to HDF5 every _HDF5_FLUSH_BATCH."""
        if self._h5_file is None:
            self._open_hdf5()
        with self._h5_lock:
            self._h5_batch.append(record)
            if len(self._h5_batch) >= self._HDF5_FLUSH_BATCH:
                self._flush_hdf5()

    def _open_hdf5(self) -> None:
        import h5py
        h5_path = os.path.join(self._session_path, "benchmark.h5")
        self._h5_file = h5py.File(h5_path, "w")
        self._h5_file.attrs["session_id"] = self._session_id
        self._h5_file.attrs["created"] = datetime.now().isoformat()

    def _flush_hdf5(self) -> None:
        """Write accumulated records to HDF5, grouped by algorithm_name."""
        if self._h5_file is None or not self._h5_batch:
            return
        import numpy as np

        # Group by algorithm
        groups: dict[str, list[BenchmarkRecord]] = {}
        for r in self._h5_batch:
            groups.setdefault(r.algorithm_name, []).append(r)

        for algo, recs in groups.items():
            grp_name = algo.replace("+", "_").replace(" ", "_")
            if grp_name in self._h5_file:
                del self._h5_file[grp_name]
            grp = self._h5_file.create_group(grp_name)

            n = len(recs)
            grp.create_dataset(
                "timestamp", data=np.array([r.timestamp for r in recs]),
                maxshape=(None,), compression="gzip",
            )
            grp.create_dataset(
                "elapsed_sec", data=np.array([r.elapsed_sec for r in recs]),
                maxshape=(None,), compression="gzip",
            )
            grp.create_dataset(
                "frame_index", data=np.array([r.frame_index for r in recs]),
                maxshape=(None,), compression="gzip",
            )
            grp.create_dataset(
                "latency_ms", data=np.array([r.latency_ms for r in recs]),
                maxshape=(None,), compression="gzip",
            )
            grp.create_dataset(
                "snr_gain_db",
                data=np.array([r.snr_gain_db if r.snr_gain_db else 0.0 for r in recs]),
                maxshape=(None,), compression="gzip",
            )
            grp.create_dataset(
                "breath_bpm", data=np.array([r.breath_bpm for r in recs]),
                maxshape=(None,), compression="gzip",
            )
            grp.create_dataset(
                "heart_bpm", data=np.array([r.heart_bpm for r in recs]),
                maxshape=(None,), compression="gzip",
            )
            grp.create_dataset(
                "heart_prominence", data=np.array([r.heart_prominence for r in recs]),
                maxshape=(None,), compression="gzip",
            )

        self._h5_batch.clear()

    def _close_files(self) -> None:
        """Close all open file handles."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        if hasattr(self, '_gt_file') and self._gt_file:
            self._gt_file.close()
        if self._h5_file:
            self._h5_file.close()
            self._h5_file = None
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -m pytest utils/test_benchmark_logger.py -v --tb=short
```

Expected: 6 tests PASS

- [ ] **Step 7: Commit**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && git add utils/__init__.py utils/benchmark_logger.py utils/test_benchmark_logger.py && git commit -m "feat: add async benchmark logger with CSV + HDF5 output"
```

---

### Task 3: Update `dsp_pipeline/__init__.py` Exports

**Files:**
- Modify: `dsp_pipeline/__init__.py`

**Interfaces:**
- Consumes: strategy classes from Task 1
- Produces: Importable `from dsp_pipeline import SignalCleanerStrategy, ...`

- [ ] **Step 1: Read current `__init__.py`** (already read — currently 3 exports)

- [ ] **Step 2: Replace `dsp_pipeline/__init__.py`**

```python
"""DSP pipeline — signal processing, beamforming, and strategy interfaces."""

# Existing exports (unchanged)
from dsp_pipeline.harmonic_mask import apply_harmonic_attenuation
from dsp_pipeline.music_angle import estimate_angle_music
from dsp_pipeline.lcmv_beamformer import lcmv_displacement

# Strategy layer (new)
from dsp_pipeline.strategies import (
    SignalCleanerStrategy,
    VitalSignSeparator,
    VMDRLSCleaner,
    EMDHarmonicCleaner,
    EMDPulseCleaner,
    PassthroughCleaner,
    WPDSeparator,
    SOSFilterSeparator,
    AdaptiveStrategySelector,
)
```

- [ ] **Step 3: Verify imports work**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "from dsp_pipeline import SignalCleanerStrategy, VMDRLSCleaner, WPDSeparator, AdaptiveStrategySelector; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && git add dsp_pipeline/__init__.py && git commit -m "feat: add strategy exports to dsp_pipeline.__init__"
```

---

### Task 4: Add AdaptiveSelector Threshold Constants

**Files:**
- Modify: `config/protocol.py`

- [ ] **Step 1: Append constants to `config/protocol.py`**

Add after the existing SQI constants block (after line 131 `SQI_BREATH_RATIO_REF = 0.1`):

```python
# ==============================================================================
# 1.4 自适应策略选择器阈值 (AdaptiveStrategySelector thresholds)
# ==============================================================================
ADAPTIVE_HIGH_PHASE_THRESHOLD = 0.02     # 信号强 → 轻量策略 (Passthrough+SOS)
ADAPTIVE_LOW_PHASE_THRESHOLD = 0.005    # 信号弱 → 抗干扰策略 (EMD+WPD)
ADAPTIVE_HIGH_SNR_DB = 15.0             # SNR 高 → 轻量策略
ADAPTIVE_LOW_SNR_DB = 5.0               # SNR 低 → 抗干扰策略
ADAPTIVE_EVAL_INTERVAL = 50             # 策略重评估帧间隔 (防止决策抖动)
```

- [ ] **Step 2: Verify import**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "from config.protocol import ADAPTIVE_HIGH_PHASE_THRESHOLD; print(ADAPTIVE_HIGH_PHASE_THRESHOLD)"
```

Expected: `0.02`

- [ ] **Step 3: Commit**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && git add config/protocol.py && git commit -m "feat: add AdaptiveStrategySelector threshold constants"
```

---

### Task 5: Add I18n Keys for Algorithm Panel

**Files:**
- Modify: `config/i18n.py`

- [ ] **Step 1: Append translation keys to each language block**

In the `"zh"` dictionary, add after `"debug_expanded"` (line 85):

```python
        # Algorithm control panel
        "algo_panel_label": "算法",
        "algo_adaptive": "自适应 (Adaptive)",
        "algo_vmd_wpd": "VMD+RLS + WPD",
        "algo_emd_wpd": "EMD + WPD",
        "algo_passthrough_sos": "Passthrough + SOS",
        "ab_panel_label": "对比",
        "ab_off": "关闭",
        "btn_record_start": "🔴 记录",
        "btn_record_stop": "⏹ 停止记录",
        "debug_dsp_current": "[DSP Engine] Current: {} | Latency: {}ms | SNR_Gain: {:+.1f}dB",
        "debug_dsp_ab": "[DSP Engine] A/B: {} | Latency: {}ms | SNR_Gain: {:+.1f}dB",
```

In the `"en"` dictionary, add after `"debug_expanded"` (line 195):

```python
        # Algorithm control panel
        "algo_panel_label": "Algorithm",
        "algo_adaptive": "Adaptive",
        "algo_vmd_wpd": "VMD+RLS + WPD",
        "algo_emd_wpd": "EMD + WPD",
        "algo_passthrough_sos": "Passthrough + SOS",
        "ab_panel_label": "A/B",
        "ab_off": "Off",
        "btn_record_start": "🔴 Record",
        "btn_record_stop": "⏹ Stop Recording",
        "debug_dsp_current": "[DSP Engine] Current: {} | Latency: {}ms | SNR_Gain: {:+.1f}dB",
        "debug_dsp_ab": "[DSP Engine] A/B: {} | Latency: {}ms | SNR_Gain: {:+.1f}dB",
```

- [ ] **Step 2: Verify**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "from config.i18n import tr; I18n.set_language('en'); print(tr('algo_adaptive')); I18n.set_language('zh'); print(tr('algo_adaptive'))"
```

Expected:
```
Adaptive
自适应 (Adaptive)
```

- [ ] **Step 3: Commit**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && git add config/i18n.py && git commit -m "feat: add i18n keys for algorithm control panel"
```

---

### Task 6: Refactor `pipeline.py` — Strategy Injection + AdaptiveSelector + A/B

**Files:**
- Modify: `dsp_pipeline/pipeline.py`

**Interfaces:**
- Consumes: `SignalCleanerStrategy`, `VitalSignSeparator`, `AdaptiveStrategySelector` (Task 1), `AlgorithmBenchmarker` (Task 2), config constants (Task 4)
- Produces: `Pipeline` with constructor injection, strategy hot-swap API, A/B dispatch, benchmark logging

- [ ] **Step 1: Add new imports at top of `pipeline.py`**

Add after the existing `from dsp_pipeline.smoothers import ...` line (line 32):

```python
from dsp_pipeline.strategies import (
    SignalCleanerStrategy, VitalSignSeparator,
    VMDRLSCleaner, EMDHarmonicCleaner, PassthroughCleaner,
    WPDSeparator, SOSFilterSeparator,
    AdaptiveStrategySelector,
)
from utils.benchmark_logger import AlgorithmBenchmarker, BenchmarkRecord
```

Also add the new config imports to the existing config block — add after `SAVGOL_WINDOW_LENGTH` line:

```python
    ADAPTIVE_HIGH_PHASE_THRESHOLD, ADAPTIVE_LOW_PHASE_THRESHOLD,
    ADAPTIVE_HIGH_SNR_DB, ADAPTIVE_LOW_SNR_DB, ADAPTIVE_EVAL_INTERVAL,
```

- [ ] **Step 2: Replace Pipeline `__init__` constructor**

Replace the `__init__` method (lines 37-113) with the new version:

```python
    def __init__(
        self,
        use_beamforming: bool = True,
        cleaner: SignalCleanerStrategy | None = None,
        separator: VitalSignSeparator | None = None,
        use_adaptive: bool = True,
    ):
        self.raw_queue = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
        self.display_queue = queue.Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_count = 0

        # Strategy injection — defaults preserve current VMD+RLS → WPD behavior
        self._cleaner: SignalCleanerStrategy = cleaner or VMDRLSCleaner()
        self._separator: VitalSignSeparator = separator or WPDSeparator()
        self._current_algo_name: str = "Adaptive"
        self._current_latency_ms: float = 0.0
        self._current_snr_gain_db: float = 0.0

        # Adaptive routing (replaces _use_advanced_dsp flag)
        self._use_adaptive = use_adaptive
        self._adaptive_selector = AdaptiveStrategySelector(
            light=(PassthroughCleaner(), SOSFilterSeparator()),
            standard=(VMDRLSCleaner(), WPDSeparator()),
            heavy=(EMDHarmonicCleaner(), WPDSeparator()),
            evaluation_interval=ADAPTIVE_EVAL_INTERVAL,
            high_phase_threshold=ADAPTIVE_HIGH_PHASE_THRESHOLD,
            low_phase_threshold=ADAPTIVE_LOW_PHASE_THRESHOLD,
            high_snr_db=ADAPTIVE_HIGH_SNR_DB,
            low_snr_db=ADAPTIVE_LOW_SNR_DB,
        ) if use_adaptive else None

        # Per-RX complex buffer
        self._rx_buffer: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
        self._phase_buffer: deque[float] = deque(maxlen=WINDOW_SIZE)

        self._best_bin: float | None = None
        self._last_bpm_update = 0
        self.last_heartbeat = 0.0

        # 2D-CFAR state
        self._cfar_accumulator: list[np.ndarray] = []
        self._cfar_state: dict | None = None
        self._cfar_rolling_buffer: deque[np.ndarray] = deque(maxlen=int(FS_HZ * CFAR_ROLLING_BUFFER_SEC))
        self._cfar_initial_frames: int = int(FS_HZ * CFAR_INITIAL_SEC)
        self._cfar_rescan_interval: int = int(FS_HZ * CFAR_RESCAN_SEC)
        self._current_bin_snr: float = 0.0
        self.DISTANCE_PER_BIN: float = RANGE_RESOLUTION_M
        self._MIN_RANGE_BIN: int = MIN_VALID_RANGE_BIN

        # Feature toggles
        self._use_beamforming = use_beamforming

        # Beamforming state
        self._angle_deg: float = 0.0
        self._angle_initialized: bool = False
        self._beamforming_ok: bool = True
        self._last_music_update: int = -50
        self._music_update_interval: int = int(FS_HZ * MUSIC_UPDATE_SEC)

        # MATLAB Filter.m: SOS
        self._filter = VitalSignFilter(fs=FS_HZ)

        # Kalman tracking history
        self._breath_history: list[float] = []
        self._heart_history: list[float] = []
        self._breath_raw_history: deque[float] = deque(maxlen=BREATH_RAW_HISTORY_MAXLEN)
        self._heart_raw_history: deque[float] = deque(maxlen=3)

        # Smoother state
        self._breath_smoother = SmootherState()
        self._heart_smoother = SmootherState()

        # Weak signal / target reacquisition
        self._low_signal_frame_count: int = 0
        self._last_valid_breath_bpm: float = 0.0
        self._last_valid_heart_bpm: float = 0.0

        # EMA state
        self._breath_ema: float = 0.0
        self._heart_ema: float = 0.0

        # Heart prominence history for adaptive Kalman
        self._heart_prominence_history: list[float] = []

        # Cached waveforms
        self._cached_breath_wave: np.ndarray | None = None
        self._cached_heart_wave: np.ndarray | None = None

        # Phase unwrapping continuity
        self._last_unwrapped_phase: float | None = None

        # ── Async DSP + A/B ──
        self._dsp_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._dsp_future = None
        self._ab_enabled: bool = False
        self._ab_cleaner: SignalCleanerStrategy | None = None
        self._ab_separator: VitalSignSeparator | None = None
        self._ab_future = None
        self._ab_algo_name: str = ""
        self._ab_latency_ms: float = 0.0
        self._ab_snr_gain_db: float = 0.0
        self._pending_displacement = None

        # ── Benchmarker ──
        self._benchmarker: AlgorithmBenchmarker | None = None
```

- [ ] **Step 3: Replace `_advanced_dsp_path` method (lines 373-423)**

Replace the entire method:

```python
    def _advanced_dsp_path(
        self, displacement: np.ndarray,
        cleaner: SignalCleanerStrategy | None = None,
        separator: VitalSignSeparator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, dict]:
        """SOS → Cleaner → Separator → BPM. Returns 6-tuple with metrics.

        Strategy objects handle their own errors internally — no try/except needed.
        """
        use_cleaner = cleaner if cleaner is not None else self._cleaner
        use_sep = separator if separator is not None else self._separator

        # SOS pre-filter
        try:
            filted = sosfiltfilt(self._filter.sos_all, displacement)
        except Exception:
            filted = displacement.copy()

        # Cleaner (harmonic removal)
        clean_disp, clean_metrics = use_cleaner.clean(filted, FS_HZ)

        # Separator (breath / heart)
        heart_diff = np.diff(clean_disp)
        breath_wave, heart_wave, sep_metrics = use_sep.separate(
            clean_disp, FS_HZ, heart_input_signal=heart_diff,
        )

        # BPM estimation
        heart_prominence = 0.1
        try:
            breath_bpm, heart_bpm, heart_prominence = estimate_bpm_stft(
                breath_wave, heart_wave, FS_HZ, FFT_N_HEART,
                raw_displacement=displacement,
                breath_waveform=breath_wave,
            )
        except Exception:
            breath_bpm, _ = estimate_bpm(
                breath_wave, FS_HZ, (0.1, 0.8), n_fft=FFT_N_BREATH,
                enable_subharmonic_rescue=True,
            )
            if breath_bpm <= 0:
                breath_bpm = estimate_breath_bpm_time_domain(breath_wave, FS_HZ)
            f0 = breath_bpm / 60.0 if breath_bpm > 0 else 0.0
            heart_bpm, heart_prominence = estimate_bpm(
                heart_wave, FS_HZ, (0.8, 2.0), f0=f0,
            )

        # Combine metrics
        combined = {
            **clean_metrics,
            **sep_metrics,
            "breath_bpm": breath_bpm,
            "heart_bpm": heart_bpm,
            "heart_prominence": heart_prominence,
        }
        return breath_wave, heart_wave, breath_bpm, heart_bpm, heart_prominence, combined
```

- [ ] **Step 4: Replace the `update_bpm` block in `_shared_signal_chain`**

Replace lines 477-606 (the entire `if update_bpm:` block) with:

```python
        if update_bpm:
            # ── Run AdaptiveSelector if enabled ──
            if self._use_adaptive and self._adaptive_selector is not None:
                self._cleaner, self._separator = self._adaptive_selector.select(
                    phase_range=recent_phase_range,
                    snr_db=self._current_bin_snr,
                )
                cname = self._cleaner.clean(np.array([0.0]), FS_HZ)[1].get("algorithm", "?")
                sname = self._separator.separate(np.array([0.0, 0.1]), FS_HZ)[2].get("algorithm", "?")
                self._current_algo_name = f"{cname}+{sname}"

            # ── Primary strategy: check async result then dispatch new ──
            # 1. Check if previous async computation finished
            if self._dsp_future is not None and self._dsp_future.done():
                try:
                    (adv_breath, adv_heart, adv_breath_bpm, adv_heart_bpm,
                     adv_heart_prom, adv_metrics) = self._dsp_future.result()

                    breath_signal_display = adv_breath
                    heart_signal_display = adv_heart
                    breath_bpm = adv_breath_bpm
                    heart_bpm_raw = adv_heart_bpm

                    # Cache latency + SNR for debug panel
                    self._current_latency_ms = adv_metrics.get("latency_ms", 0.0)
                    self._current_snr_gain_db = adv_metrics.get("snr_estimate_db", 0.0) or 0.0

                    # [保留] 中值去飞点 → Kalman → EMA
                    if breath_bpm > 0:
                        self._breath_raw_history.append(breath_bpm)
                        breath_bpm = apply_smoothing_chain(
                            self._breath_smoother, breath_bpm,
                            recent_phase_range, breath_power_ratio,
                            self._current_bin_snr,
                        )
                        self._last_valid_breath_bpm = breath_bpm

                    if heart_bpm_raw > 0:
                        self._heart_prominence_history.append(adv_heart_prom)
                        if len(self._heart_prominence_history) > HEART_KALMAN_HISTORY_MAXLEN * 2:
                            self._heart_prominence_history = self._heart_prominence_history[-HEART_KALMAN_HISTORY_MAXLEN * 2:]

                        heart_bpm_accepted = True
                        if self._last_valid_heart_bpm > 0:
                            bpm_jump = abs(heart_bpm_raw - self._last_valid_heart_bpm)
                            if bpm_jump > 20.0 and adv_heart_prom < 0.3:
                                heart_bpm_accepted = False

                        if heart_bpm_accepted:
                            if HEART_USE_NEW_SMOOTHER:
                                self._heart_raw_history.append(heart_bpm_raw)
                                heart_bpm = apply_smoothing_chain(
                                    self._heart_smoother, heart_bpm_raw,
                                    recent_phase_range, breath_power_ratio,
                                    self._current_bin_snr,
                                )
                                self._last_valid_heart_bpm = heart_bpm
                            else:
                                self._heart_raw_history.append(heart_bpm_raw)
                                heart_bpm_raw_median = float(np.median(list(self._heart_raw_history)))
                                self._heart_history.append(heart_bpm_raw_median)
                                if len(self._heart_history) > HEART_KALMAN_HISTORY_MAXLEN:
                                    self._heart_history = self._heart_history[-HEART_KALMAN_HISTORY_MAXLEN:]
                                prom_slice = self._heart_prominence_history[-len(self._heart_history):]
                                heart_bpm = kalman_smooth(
                                    self._heart_history, q=1e-3, r=0.5,
                                    prominences=prom_slice,
                                )
                                self._last_valid_heart_bpm = heart_bpm
                        else:
                            heart_bpm = self._last_valid_heart_bpm

                    self._cached_breath_wave = adv_breath
                    self._cached_heart_wave = adv_heart

                    # ── Benchmark log (primary) ──
                    self._log_benchmark(adv_metrics, displacement, is_primary=True)

                except Exception as e:
                    print(f"[DSP] Async advanced path failed: {e}")

            # 2. Dispatch new primary task if idle
            if self._dsp_future is None or self._dsp_future.done():
                self._dsp_future = self._dsp_executor.submit(
                    self._advanced_dsp_path, displacement.copy(),
                    self._cleaner, self._separator,
                )

            # ── A/B comparison: check result then dispatch ──
            if self._ab_enabled:
                if self._ab_future is not None and self._ab_future.done():
                    try:
                        (_, _, _, _, _, ab_metrics) = self._ab_future.result()
                        self._ab_algo_name = f"{ab_metrics.get('algorithm', '?')}"
                        self._ab_latency_ms = ab_metrics.get("latency_ms", 0.0)
                        self._ab_snr_gain_db = ab_metrics.get("snr_estimate_db", 0.0) or 0.0
                        self._log_benchmark(ab_metrics, displacement, is_primary=False)
                    except Exception:
                        pass

                if self._ab_future is None or self._ab_future.done():
                    if self._ab_cleaner is not None and self._ab_separator is not None:
                        self._ab_future = self._dsp_executor.submit(
                            self._advanced_dsp_path, displacement.copy(),
                            self._ab_cleaner, self._ab_separator,
                        )

            # 3. Maintain last valid BPMs while computing
            breath_bpm = self._last_valid_breath_bpm
            heart_bpm = self._last_valid_heart_bpm
            if self._cached_breath_wave is not None:
                breath_signal_display = self._cached_breath_wave
                heart_signal_display = self._cached_heart_wave

            # ── Fallback baseline path (only when adaptive is off and BPM invalid) ──
            if not self._use_adaptive and breath_bpm <= 0:
                breath_bpm = estimate_breath_bpm_time_domain(
                    no_dc, fs=FS_HZ, min_interval_sec=1.0,
                )
                if breath_bpm <= 0:
                    breath_bpm, _ = estimate_bpm(
                        breath_signal, FS_HZ, (0.1, 0.8), n_fft=FFT_N_HEART,
                        enable_subharmonic_rescue=True,
                    )
                if breath_bpm > 0:
                    self._breath_raw_history.append(breath_bpm)
                    breath_bpm = apply_smoothing_chain(
                        self._breath_smoother, breath_bpm,
                        recent_phase_range, breath_power_ratio,
                        self._current_bin_snr,
                    )
                    self._last_valid_breath_bpm = breath_bpm

                f0 = breath_bpm / 60.0 if breath_bpm > 0 else 0.0
                heart_bpm_raw, prominence = estimate_bpm(
                    heart_signal, FS_HZ, (0.8, 2.0), f0=f0,
                )
                if heart_bpm_raw > 0:
                    self._heart_prominence_history.append(prominence)
                    if len(self._heart_prominence_history) > HEART_KALMAN_HISTORY_MAXLEN * 2:
                        self._heart_prominence_history = self._heart_prominence_history[-HEART_KALMAN_HISTORY_MAXLEN * 2:]

                    heart_bpm_accepted = True
                    if self._last_valid_heart_bpm > 0:
                        bpm_jump = abs(heart_bpm_raw - self._last_valid_heart_bpm)
                        if bpm_jump > 20.0 and prominence < 0.3:
                            heart_bpm_accepted = False

                    if heart_bpm_accepted:
                        if HEART_USE_NEW_SMOOTHER:
                            self._heart_raw_history.append(heart_bpm_raw)
                            heart_bpm = apply_smoothing_chain(
                                self._heart_smoother, heart_bpm_raw,
                                recent_phase_range, breath_power_ratio,
                                self._current_bin_snr,
                            )
                            self._last_valid_heart_bpm = heart_bpm
                        else:
                            self._heart_raw_history.append(heart_bpm_raw)
                            heart_bpm_raw_median = float(np.median(list(self._heart_raw_history)))
                            self._heart_history.append(heart_bpm_raw_median)
                            if len(self._heart_history) > HEART_KALMAN_HISTORY_MAXLEN:
                                self._heart_history = self._heart_history[-HEART_KALMAN_HISTORY_MAXLEN:]
                            prom_slice = self._heart_prominence_history[-len(self._heart_history):]
                            heart_bpm = kalman_smooth(
                                self._heart_history, q=1e-3, r=0.5,
                                prominences=prom_slice,
                            )
                            self._last_valid_heart_bpm = heart_bpm
                    else:
                        heart_bpm = self._last_valid_heart_bpm

            self._last_bpm_update = self._frame_count
```

- [ ] **Step 5: Add helper methods to Pipeline class**

Insert these methods before `_check_quality`:

```python
    def set_strategies(
        self, cleaner: SignalCleanerStrategy, separator: VitalSignSeparator
    ) -> None:
        """Hot-swap primary strategies. Takes effect on next update_bpm dispatch."""
        self._cleaner = cleaner
        self._separator = separator
        self._use_adaptive = False  # manual override disables adaptive
        cname = cleaner.clean(np.array([0.0]), FS_HZ)[1].get("algorithm", "?")
        sname = separator.separate(np.array([0.0, 0.1]), FS_HZ)[2].get("algorithm", "?")
        self._current_algo_name = f"{cname}+{sname}"

    def set_ab_strategy(
        self,
        cleaner: SignalCleanerStrategy | None,
        separator: VitalSignSeparator | None,
    ) -> None:
        """Set or disable A/B comparison strategy. None disables."""
        self._ab_cleaner = cleaner
        self._ab_separator = separator
        self._ab_enabled = (cleaner is not None and separator is not None)

    @property
    def benchmarker(self) -> AlgorithmBenchmarker | None:
        return self._benchmarker

    def set_benchmarker(self, benchmarker: AlgorithmBenchmarker | None) -> None:
        self._benchmarker = benchmarker

    def _log_benchmark(
        self, metrics: dict, displacement: np.ndarray, is_primary: bool
    ) -> None:
        """Build and enqueue a BenchmarkRecord if recording is active."""
        if self._benchmarker is None or not self._benchmarker.is_recording:
            return
        try:
            input_pr = float(np.max(displacement) - np.min(displacement))
        except Exception:
            input_pr = 0.0
        record = BenchmarkRecord(
            timestamp=time.time(),
            frame_index=self._frame_count,
            elapsed_sec=self._frame_count / FS_HZ,
            algorithm_name=metrics.get("algorithm",
                         self._current_algo_name if is_primary else self._ab_algo_name),
            is_primary=is_primary,
            latency_ms=metrics.get("latency_ms", 0.0),
            input_phase_range=input_pr,
            input_snr_db=metrics.get("snr_estimate_db"),
            output_phase_range=phase_range if is_primary else 0.0,  # phase_range from outer scope
            output_snr_db=None,
            snr_gain_db=None,
            breath_bpm=metrics.get("breath_bpm", 0.0),
            heart_bpm=metrics.get("heart_bpm", 0.0),
            heart_prominence=metrics.get("heart_prominence", 0.0),
            imf_count=metrics.get("imf_count"),
            convergence_iter=metrics.get("convergence_iter"),
            retained_harmonics=(
                ",".join(str(h) for h in metrics["retained_harmonics"])
                if metrics.get("retained_harmonics") else None
            ),
            dominant_freq_hz=metrics.get("dominant_freq_hz"),
        )
        self._benchmarker.log(record)
```

Note: The `_log_benchmark` references `phase_range` from the enclosing `_shared_signal_chain` scope. We'll fix the reference — pass it explicitly or refactor to use the correct value. For a clean fix, modify the `_log_benchmark` call sites to include `phase_range` as a parameter:

In the call sites in Step 4, replace `self._log_benchmark(adv_metrics, displacement, is_primary=True)` with:

```python
self._log_benchmark(adv_metrics, displacement, phase_range, is_primary=True)
```

And update the method signature:

```python
    def _log_benchmark(
        self, metrics: dict, displacement: np.ndarray, phase_range_val: float,
        is_primary: bool,
    ) -> None:
        """Build and enqueue a BenchmarkRecord if recording is active."""
        if self._benchmarker is None or not self._benchmarker.is_recording:
            return
        try:
            input_pr = float(np.max(displacement) - np.min(displacement))
        except Exception:
            input_pr = 0.0
        record = BenchmarkRecord(
            ...
            output_phase_range=phase_range_val,
            ...
        )
```

- [ ] **Step 6: Clean up the `_dsp_future.done()` result handling — update the variable reference**

In `_shared_signal_chain`, the `quality = self._check_quality(no_dc)` block before the return already has `phase_range` and `breath_ratio`. The `_log_benchmark` calls need `phase_range` from this earlier assignment. Pass it through.

- [ ] **Step 7: Verify no import errors**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "from dsp_pipeline.pipeline import Pipeline; p = Pipeline(); print('Pipeline OK')"
```

Expected: `Pipeline OK`

- [ ] **Step 8: Commit**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && git add dsp_pipeline/pipeline.py && git commit -m "refactor: inject strategies into Pipeline, add AdaptiveSelector and A/B dispatch"
```

---

### Task 7: Refactor `bp_pipeline.py` — Strategy Injection

**Files:**
- Modify: `bp_monitor/bp_pipeline.py`

**Interfaces:**
- Consumes: `SignalCleanerStrategy`, `EMDPulseCleaner` (Task 1), `AlgorithmBenchmarker`, `BenchmarkRecord` (Task 2)

- [ ] **Step 1: Add new imports to `bp_pipeline.py`**

After the existing import block (after line 50), add:

```python
from dsp_pipeline.strategies import SignalCleanerStrategy, EMDPulseCleaner
from utils.benchmark_logger import AlgorithmBenchmarker, BenchmarkRecord
```

- [ ] **Step 2: Replace BPPipeline `__init__` signature**

Replace lines 137-193 (constructor):

```python
    def __init__(
        self,
        weights_path: str = "bp_matlab/bp_weights.mat",
        cleaner: SignalCleanerStrategy | None = None,
        benchmarker: AlgorithmBenchmarker | None = None,
    ):
        # -- public queues --
        self.raw_queue: queue.Queue[RadarFrame] = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
        self.display_queue: queue.Queue[BPResult] = queue.Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)

        self._weights_path = weights_path
        self._bp: BPInference | None = None

        # -- strategy injection --
        self._cleaner: SignalCleanerStrategy = cleaner or EMDPulseCleaner()

        # -- benchmarker --
        self._benchmarker: AlgorithmBenchmarker | None = benchmarker

        # -- threads --
        self._collector_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Ring buffer
        self._buffer = np.zeros((32, self.MAX_FRAMES, 1), dtype=complex)
        self._head: int = 0
        self._frame_count: int = 0

        # Collector → Worker channel
        self._inference_queue: queue.Queue = queue.Queue(maxsize=2)

        # Shared state
        self._state_lock = threading.Lock()
        self._target_bin: int | None = None
        self._tracker_state: TrackerState = TrackerState.TRACKING
        self._cfar_state: dict | None = None
        self._cold_start: bool = True

        # Alpha-Beta tracker
        self._tracker = AlphaBetaTracker(alpha=0.85, beta=0.5)

        # Phase-continuity
        self._last_phase_ref: tuple[int, float] | None = None

        # Worker-owned state
        self._last_inference_frame: int = 0
        self._bad_signal_count: int = 0

        # Temporal smoothing
        self._sbp_history: deque[float] = deque(maxlen=10)
        self._dbp_history: deque[float] = deque(maxlen=10)
        self._sbp_ema: float | None = None
        self._dbp_ema: float | None = None
```

- [ ] **Step 3: Replace signal cleaning call in `_process_snapshot`**

Replace lines 539-541:

```python
        # --- signal cleaning (EMD + wavelet) ---
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
```

- [ ] **Step 4: Add benchmarker property accessor**

Insert into BPPipeline class:

```python
    @property
    def benchmarker(self) -> AlgorithmBenchmarker | None:
        return self._benchmarker

    def set_benchmarker(self, bm: AlgorithmBenchmarker | None) -> None:
        self._benchmarker = bm
```

- [ ] **Step 5: Verify**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "from bp_monitor.bp_pipeline import BPPipeline; p = BPPipeline(); print('BPPipeline OK')"
```

Expected: `BPPipeline OK`

- [ ] **Step 6: Commit**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && git add bp_monitor/bp_pipeline.py && git commit -m "refactor: inject cleaner strategy and benchmarker into BPPipeline"
```

---

### Task 8: Update `monitor_mode.py` — Wire Strategies Through HRMode and BPMode

**Files:**
- Modify: `ui/monitor_mode.py`

- [ ] **Step 1: Add imports to `monitor_mode.py`**

After existing imports (around line 16), add:

```python
from dsp_pipeline.strategies import (
    SignalCleanerStrategy, VitalSignSeparator,
    VMDRLSCleaner, EMDHarmonicCleaner, PassthroughCleaner,
    WPDSeparator, SOSFilterSeparator,
)
from utils.benchmark_logger import AlgorithmBenchmarker
```

- [ ] **Step 2: Update HRMode `__init__`**

Replace HRMode's `__init__` (lines 99-111):

```python
    def __init__(self):
        from dsp_pipeline.pipeline import Pipeline

        self._pipeline: Pipeline | None = None
        self._latest_vitals: VitalSigns | None = None
        self._trend_tick_counter: int = 0

        # Data accumulation for export
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
```

- [ ] **Step 3: Update HRMode `start()` to pass strategies**

Replace HRMode `start()` method (lines 134-137):

```python
    def start(self) -> None:
        from dsp_pipeline.pipeline import Pipeline
        cleaner = self._pending_cleaner or VMDRLSCleaner()
        separator = self._pending_separator or WPDSeparator()
        self._pipeline = Pipeline(
            cleaner=cleaner,
            separator=separator,
            use_adaptive=self._use_adaptive,
        )
        # Re-apply pending A/B and benchmarker
        if self._pending_ab_cleaner is not None and self._pending_ab_separator is not None:
            self._pipeline.set_ab_strategy(
                self._pending_ab_cleaner, self._pending_ab_separator
            )
        if self._benchmarker is not None:
            self._pipeline.set_benchmarker(self._benchmarker)
        self._pipeline.start()
```

- [ ] **Step 4: Add strategy control methods to HRMode**

Insert into HRMode class:

```python
    def set_strategies(
        self, cleaner: SignalCleanerStrategy, separator: VitalSignSeparator,
    ) -> None:
        """Set primary strategies. Applies immediately if running, else on next start."""
        self._use_adaptive = False
        self._pending_cleaner = cleaner
        self._pending_separator = separator
        if self._pipeline is not None:
            self._pipeline.set_strategies(cleaner, separator)

    def set_adaptive_mode(self) -> None:
        """Enable adaptive strategy routing."""
        self._use_adaptive = True
        if self._pipeline is not None:
            self._pipeline._use_adaptive = True

    def set_ab_strategy(
        self,
        cleaner: SignalCleanerStrategy | None,
        separator: VitalSignSeparator | None,
    ) -> None:
        """Set or disable A/B comparison."""
        self._pending_ab_cleaner = cleaner
        self._pending_ab_separator = separator
        if self._pipeline is not None:
            self._pipeline.set_ab_strategy(cleaner, separator)

    def toggle_benchmark(self) -> bool:
        """Start or stop the benchmarker. Returns new is_recording state."""
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
        """Return current DSP engine telemetry for debug panel."""
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
        """Return elapsed recording seconds. Used by UI timer."""
        if self._benchmarker is None or not self._benchmarker.is_recording:
            return 0.0
        return time.time() - self._benchmarker._start_time
```

- [ ] **Step 5: Update `poll_and_update` to pass telemetry to UI**

In HRMode `poll_and_update` method, modify the research_tab call to include telemetry:

Replace lines 196-203:

```python
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
```

- [ ] **Step 6: Verify**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "from ui.monitor_mode import HRMode; m = HRMode(); print('HRMode OK'); print(m.get_dsp_telemetry())"
```

Expected: `HRMode OK` and a dict with telemetry keys.

- [ ] **Step 7: Commit**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && git add ui/monitor_mode.py && git commit -m "refactor: wire strategy injection and benchmarker through HRMode"
```

---

### Task 9: Update `research_tab.py` — Algorithm Panel + Debug Extension

**Files:**
- Modify: `ui/research_tab.py`

- [ ] **Step 1: Add the algorithm control panel widgets in `_setup_ui`**

Add after the line `layout = QVBoxLayout(self)` and `layout.setContentsMargins(8, 8, 8, 8)`:

```python
        # ── Algorithm Control Panel ──
        algo_row = QHBoxLayout()
        algo_row.setContentsMargins(0, 0, 0, 4)

        algo_row.addWidget(QLabel(tr("algo_panel_label") + ":"))
        self._algo_combo = QComboBox()
        self._algo_combo.addItems([
            tr("algo_adaptive"),
            tr("algo_vmd_wpd"),
            tr("algo_emd_wpd"),
            tr("algo_passthrough_sos"),
        ])
        self._algo_combo.setCurrentIndex(0)  # default: Adaptive
        self._algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        algo_row.addWidget(self._algo_combo)

        algo_row.addSpacing(12)
        algo_row.addWidget(QLabel(tr("ab_panel_label") + ":"))
        self._ab_combo = QComboBox()
        self._ab_combo.addItems([
            tr("ab_off"),
            tr("algo_vmd_wpd"),
            tr("algo_emd_wpd"),
            tr("algo_passthrough_sos"),
        ])
        self._ab_combo.setCurrentIndex(0)  # default: Off
        self._ab_combo.currentIndexChanged.connect(self._on_ab_changed)
        algo_row.addWidget(self._ab_combo)

        algo_row.addSpacing(12)
        self._record_btn = QPushButton(tr("btn_record_start"))
        self._record_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; font-size: 10pt; }"
            "QPushButton:hover { background-color: #c0392b; }"
        )
        self._record_btn.clicked.connect(self._on_record_toggle)
        algo_row.addWidget(self._record_btn)

        self._record_timer_label = QLabel("")
        self._record_timer_label.setFont(QFont("Consolas", 10))
        self._record_timer_label.setStyleSheet("color: #e74c3c;")
        self._record_timer_label.setVisible(False)
        algo_row.addWidget(self._record_timer_label)

        algo_row.addStretch()
        layout.addLayout(algo_row)
```

Also need to add `QComboBox` to the PyQt6 imports at top — add to the existing import line:

```python
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter, QFrame,
    QPushButton, QTextEdit, QComboBox,
)
```

- [ ] **Step 2: Update `update_display` method signature and body**

Replace the method signature and add telemetry to the debug panel:

```python
    def update_display(
        self,
        breath_bpm: float,
        heart_bpm: float,
        breath_waveform: np.ndarray,
        heart_waveform: np.ndarray,
        quality: dict | None,
        sample_for_trend: bool = False,
        dsp_telemetry: dict | None = None,
        benchmark_elapsed: float = 0.0,
    ) -> None:
        # Waveforms
        ...

        # Debug panel
        if self._debug_expanded and quality is not None:
            lines = [
                f"phase_range: {phase_range:.4f}",
                f"breath_ratio: {breath_ratio:.3f}",
                f"heart_prominence: {quality.get('heart_prominence', 0):.3f}",
                f"apnea_state: {quality.get('apnea_state', False)}",
                f"valid: {quality.get('valid', False)}",
                f"reason: {quality.get('reason', '')}",
                f"SQI: {self._sqi_level}/3",
            ]
            # DSP telemetry lines
            if dsp_telemetry:
                algo = dsp_telemetry.get("current_algo", "--")
                lat = dsp_telemetry.get("current_latency_ms", 0.0)
                snr = dsp_telemetry.get("current_snr_gain_db", 0.0)
                lines.append(
                    tr("debug_dsp_current", algo, f"{lat:.0f}", snr)
                )
                if dsp_telemetry.get("ab_enabled"):
                    ab_algo = dsp_telemetry.get("ab_algo", "--")
                    ab_lat = dsp_telemetry.get("ab_latency_ms", 0.0)
                    ab_snr = dsp_telemetry.get("ab_snr_gain_db", 0.0)
                    lines.append(
                        tr("debug_dsp_ab", ab_algo, f"{ab_lat:.0f}", ab_snr)
                    )
            self._debug_panel.setText("  |  ".join(lines))
```

- [ ] **Step 3: Add signal handler methods and slot connections**

Insert into ResearchTab class:

```python
    # ── Algorithm panel slots ──

    def _on_algo_changed(self, index: int) -> None:
        """Notify parent (MainWindow) that algorithm selection changed."""
        self._algo_selection = index  # 0=Adaptive, 1=VMD+WPD, 2=EMD+WPD, 3=Passthrough+SOS

    def _on_ab_changed(self, index: int) -> None:
        """Notify parent that A/B selection changed."""
        self._ab_selection = index  # 0=Off, 1=VMD+WPD, 2=EMD+WPD, 3=Passthrough+SOS

    def _on_record_toggle(self) -> None:
        """Notify parent to start/stop benchmark recording."""

    # ── Public accessors for parent wiring ──

    @property
    def algo_selection(self) -> int:
        return getattr(self, '_algo_selection', 0)

    @property
    def ab_selection(self) -> int:
        return getattr(self, '_ab_selection', 0)

    def set_recording_state(self, is_recording: bool) -> None:
        """Update record button appearance and timer visibility."""
        if is_recording:
            self._record_btn.setText(tr("btn_record_stop"))
            self._record_btn.setStyleSheet(
                "QPushButton { background-color: #95a5a6; color: white; font-weight: bold; "
                "padding: 6px 16px; border-radius: 4px; font-size: 10pt; }"
            )
            self._record_timer_label.setVisible(True)
        else:
            self._record_btn.setText(tr("btn_record_start"))
            self._record_btn.setStyleSheet(
                "QPushButton { background-color: #e74c3c; color: white; font-weight: bold; "
                "padding: 6px 16px; border-radius: 4px; font-size: 10pt; }"
            )
            self._record_timer_label.setVisible(False)

    def update_record_timer(self, elapsed_sec: float) -> None:
        """Update the recording timer label."""
        if elapsed_sec > 0:
            h = int(elapsed_sec // 3600)
            m = int((elapsed_sec % 3600) // 60)
            s = int(elapsed_sec % 60)
            self._record_timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
        else:
            self._record_timer_label.setText("")
```

- [ ] **Step 4: Verify syntax**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "from ui.research_tab import ResearchTab; print('ResearchTab OK')"
```

Expected: `ResearchTab OK`

- [ ] **Step 5: Commit**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && git add ui/research_tab.py && git commit -m "feat: add algorithm control panel and extended debug telemetry to ResearchTab"
```

---

### Task 10: Update `main_window.py` — Wire Algorithm Signals

**Files:**
- Modify: `ui/main_window.py`

- [ ] **Step 1: Add strategy imports**

After existing imports, add:

```python
from dsp_pipeline.strategies import (
    VMDRLSCleaner, EMDHarmonicCleaner, PassthroughCleaner,
    WPDSeparator, SOSFilterSeparator,
)
```

- [ ] **Step 2: Wire algorithm change signals in `_setup_ui`**

After `self._research_tab = ResearchTab()` (line 97), add:

```python
        # Wire algorithm panel signals
        self._research_tab._algo_combo.currentIndexChanged.connect(
            self._on_research_algo_changed
        )
        self._research_tab._ab_combo.currentIndexChanged.connect(
            self._on_research_ab_changed
        )
        self._research_tab._record_btn.clicked.connect(
            self._on_research_record_toggle
        )
```

- [ ] **Step 3: Add slot methods to MainWindow**

Insert into MainWindow class:

```python
    # ── Research tab algorithm signal handlers ──

    def _on_research_algo_changed(self, index: int) -> None:
        """Apply primary algorithm selection to the current mode."""
        if not isinstance(self._current_mode, HRMode):
            return
        mode: HRMode = self._current_mode
        if index == 0:
            # Adaptive
            mode.set_adaptive_mode()
        elif index == 1:
            # VMD+RLS + WPD
            mode.set_strategies(VMDRLSCleaner(), WPDSeparator())
        elif index == 2:
            # EMD + WPD
            mode.set_strategies(EMDHarmonicCleaner(), WPDSeparator())
        elif index == 3:
            # Passthrough + SOS
            mode.set_strategies(PassthroughCleaner(), SOSFilterSeparator())

    def _on_research_ab_changed(self, index: int) -> None:
        """Apply A/B algorithm selection to the current mode."""
        if not isinstance(self._current_mode, HRMode):
            return
        mode: HRMode = self._current_mode
        pair_map = {
            0: (None, None),               # Off
            1: (VMDRLSCleaner(), WPDSeparator()),
            2: (EMDHarmonicCleaner(), WPDSeparator()),
            3: (PassthroughCleaner(), SOSFilterSeparator()),
        }
        cleaner, separator = pair_map.get(index, (None, None))
        mode.set_ab_strategy(cleaner, separator)

    def _on_research_record_toggle(self) -> None:
        """Start or stop benchmark recording."""
        if not isinstance(self._current_mode, HRMode):
            return
        mode: HRMode = self._current_mode
        is_recording = mode.toggle_benchmark()
        self._research_tab.set_recording_state(is_recording)
```

- [ ] **Step 4: Update `_on_ui_tick` to refresh record timer**

In `_on_ui_tick`, inside the `if not self._running: return` check, add:

```python
        # Update benchmark recording timer
        if isinstance(self._current_mode, HRMode):
            elapsed = self._current_mode.get_benchmark_elapsed()
            self._research_tab.update_record_timer(elapsed)
```

- [ ] **Step 5: Verify**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "from ui.main_window import MainWindow; print('MainWindow OK')"
```

Expected: `MainWindow OK`

- [ ] **Step 6: Commit**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && git add ui/main_window.py && git commit -m "feat: wire algorithm panel signals through MainWindow to HRMode"
```

---

### Task 11: Integration Smoke Test

**Files:**
- None (verification only)

- [ ] **Step 1: Full import chain verification**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "
from dsp_pipeline.strategies import (
    SignalCleanerStrategy, VitalSignSeparator,
    VMDRLSCleaner, EMDHarmonicCleaner, EMDPulseCleaner, PassthroughCleaner,
    WPDSeparator, SOSFilterSeparator, AdaptiveStrategySelector,
)
from utils.benchmark_logger import AlgorithmBenchmarker, BenchmarkRecord
from dsp_pipeline.pipeline import Pipeline
from bp_monitor.bp_pipeline import BPPipeline
from ui.monitor_mode import HRMode, BPMode
from ui.research_tab import ResearchTab
from ui.main_window import MainWindow
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 2: Strategy instantiation and basic test**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "
import numpy as np
from dsp_pipeline.strategies import (
    VMDRLSCleaner, WPDSeparator, PassthroughCleaner, SOSFilterSeparator,
    AdaptiveStrategySelector,
)
from dsp_pipeline.pipeline import Pipeline

# Test with Passthrough + SOS (lightweight baseline)
p = Pipeline(
    cleaner=PassthroughCleaner(),
    separator=SOSFilterSeparator(),
    use_adaptive=False,
)
print('Pipeline created with Passthrough+SOS')

# Test with VMD+WPD (default, but adaptive off)
p2 = Pipeline(use_adaptive=False)
print('Pipeline created with defaults (VMD+WPD)')

# Check adaptive selector
selector = AdaptiveStrategySelector(
    light=(PassthroughCleaner(), SOSFilterSeparator()),
    standard=(VMDRLSCleaner(), WPDSeparator()),
    heavy=(VMDRLSCleaner(), WPDSeparator()),
)
c, s = selector.select(phase_range=0.03, snr_db=20.0)
print(f'Adaptive selected: {type(c).__name__} + {type(s).__name__}')
print('Integration sanity check PASSED')
"
```

Expected: `Integration sanity check PASSED`

- [ ] **Step 3: Benchmarker smoke test**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "
import tempfile, os
from utils.benchmark_logger import AlgorithmBenchmarker, BenchmarkRecord
import time
with tempfile.TemporaryDirectory() as d:
    bm = AlgorithmBenchmarker(output_dir=d)
    bm.start()
    for i in range(50):
        bm.log(BenchmarkRecord(
            timestamp=time.time(), frame_index=i, elapsed_sec=i*0.05,
            algorithm_name='Test', is_primary=True, latency_ms=10.0,
            input_phase_range=0.01, input_snr_db=8.0,
            output_phase_range=0.015, output_snr_db=12.0,
            snr_gain_db=4.0, breath_bpm=15.0, heart_bpm=72.0,
            heart_prominence=0.5,
        ))
    bm.stop()
    csv_path = os.path.join(bm.session_path, 'benchmark.csv')
    with open(csv_path) as f:
        lines = f.readlines()
    print(f'CSV: {len(lines)} lines (1 header + {len(lines)-1} records)')
    assert len(lines) >= 20  # at least some records
    print('Benchmarker smoke test PASSED')
"
```

Expected: `Benchmarker smoke test PASSED`

- [ ] **Step 4: Pipeline with synthetic frame processing**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && python -c "
import numpy as np
from dsp_pipeline.pipeline import Pipeline
from dsp_pipeline.strategies import PassthroughCleaner, SOSFilterSeparator
from models.radar_frame import RadarFrame, FrameHeader

p = Pipeline(
    cleaner=PassthroughCleaner(),
    separator=SOSFilterSeparator(),
    use_adaptive=False,
)
# Feed synthetic frames to fill the window
for i in range(400):
    cube = np.random.randn(128, 1, 1) * 0.01 + 0.1 * np.sin(np.arange(128) * 0.1).reshape(-1, 1, 1)
    frame = RadarFrame(
        timestamp=i/20.0, frame_index=i,
        header=FrameHeader(0,1,4,2,58000,128,1,3000,50,1920,60),
        data_cube=cube,
    )
    p.raw_queue.put(frame)

# Run a few process cycles
import time
time.sleep(2.0)
p.stop()

# Drain display queue
results = []
while not p.display_queue.empty():
    try:
        results.append(p.display_queue.get_nowait())
    except:
        break
print(f'Pipeline produced {len(results)} VitalSigns results')
assert len(results) > 0
print('Pipeline integration test PASSED')
"
```

Expected: `Pipeline integration test PASSED` with 1+ VitalSigns results.

- [ ] **Step 5: Commit final verification**

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && git add -A && git diff --cached --stat
```

Review what's staged, then:

```bash
cd "F:\yan\mmradarcoverage\real_time_monitor" && git commit -m "test: integration smoke tests for DSP benchmark framework"
```

---

## Completion Checklist

- [ ] All 6 concrete strategies instantiate and `clean()`/`separate()` return correct shapes
- [ ] AdaptiveStrategySelector routes to correct strategy pair at boundary SQI values
- [ ] AlgorithmBenchmarker creates CSV + HDF5 files, survives queue overflow, idempotent start/stop
- [ ] Pipeline accepts injected strategies via constructor and `set_strategies()`
- [ ] A/B dispatch runs alternative strategy silently without affecting UI output
- [ ] BPPipeline accepts injected cleaner and logs to benchmarker when recording
- [ ] HRMode `get_dsp_telemetry()` returns real values after processing
- [ ] ResearchTab algorithm combo box triggers strategy changes
- [ ] Record button toggles benchmarker state and updates UI
- [ ] Debug panel shows DSP Engine telemetry lines
- [ ] All imports resolve without errors
- [ ] Pipeline processes synthetic frames and produces VitalSigns output
