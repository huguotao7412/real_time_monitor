# Pluggable DSP Benchmark Framework — Design Specification

**Date**: 2026-06-22
**Status**: Approved
**Target**: Python 3.12.10

---

## 1. Motivation & Goals

The current `pipeline.py` and `bp_pipeline.py` hardcode specific DSP algorithms (`vmd_rls_harmonic_clean`, `wpd_separate`, `clean_pulse_wave`) with `try...except` fallback chains. This coupling prevents systematic ablation studies and algorithm comparisons required for Q2/Q3-level signal processing journal submissions.

### Goals

1. **Strategy Pattern refactoring**: Decouple algorithm selection from pipeline orchestration via abstract interfaces for signal cleaners and vital-sign separators.
2. **Automated benchmark data generation**: Per-frame, per-algorithm metrics (latency, SNR, internal algorithm stats) streamed to CSV + HDF5 for direct use in paper figures.
3. **Adaptive strategy routing**: An `AdaptiveStrategySelector` that switches between lightweight/heavy DSP chains based on real-time SQI — itself a publishable innovation.
4. **A/B parallel comparison**: Run a primary strategy (drives UI) and an alternative strategy (silent recording) concurrently during live monitoring.
5. **UI controls for research workflow**: Algorithm dropdown, A/B selector, record button, extended debug panel with DSP engine telemetry.

---

## 2. Architecture Overview

Five layers, ~10 files changed (2 new, 8 modified):

```
 UI Layer          research_tab.py (+Algorithm Panel, +Debug extension)
                   main_window.py (wiring)
 ───────────────────────────────────────────
 Mode Layer        monitor_mode.py (strategy injection into HRMode / BPMode)
 ───────────────────────────────────────────
 Pipeline Layer    pipeline.py (cleaner + separator slots, AdaptiveSelector, A/B executor)
                   bp_pipeline.py (cleaner slot only)
 ───────────────────────────────────────────
 Strategy Layer    dsp_pipeline/strategies.py [NEW]
                   ├─ SignalCleanerStrategy (ABC)
                   ├─ VitalSignSeparator (ABC)
                   ├─ VMDRLSCleaner, EMDHarmonicCleaner, EMDPulseCleaner, PassthroughCleaner
                   ├─ WPDSeparator, SOSFilterSeparator
                   └─ AdaptiveStrategySelector
 ───────────────────────────────────────────
 Benchmark Layer   utils/benchmark_logger.py [NEW]
                   ├─ BenchmarkRecord (dataclass)
                   └─ AlgorithmBenchmarker (async queue → writer thread)
```

### Data Flow (live A/B mode)

```
RadarFrame → Pipeline._process_frame()
  → displacement
  → AdaptiveStrategySelector → (active_cleaner, active_separator)    [primary, drives UI]
  → ThreadPoolExecutor.submit(alt_cleaner, alt_separator)            [secondary, silent]
  → Benchmarker.log(primary_record)
  → Benchmarker.log(secondary_record)
  → UI.display(primary_result)
```

---

## 3. Strategy Layer (`dsp_pipeline/strategies.py`) [NEW]

### 3.1 Abstract Base Classes

Two separate ABCs for fine-grained ablation (cleaner × separator matrix):

```python
class SignalCleanerStrategy(ABC):
    """Single-channel denoising / harmonic removal."""

    @abstractmethod
    def clean(self, signal: np.ndarray, fs: float) -> tuple[np.ndarray, dict]:
        """
        Returns:
            cleaned: np.ndarray (same shape as input)
            metrics: {
                'algorithm': str,
                'latency_ms': float,
                'imf_count': int | None,
                'convergence_iter': int | None,
                'retained_harmonics': list[int] | None,
                'dominant_freq_hz': float | None,
                'snr_estimate_db': float | None,
            }
        """


class VitalSignSeparator(ABC):
    """Breath / heartbeat waveform separation."""

    @abstractmethod
    def separate(self, signal: np.ndarray, fs: float,
                 heart_input_signal: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Returns:
            breath_wave: np.ndarray
            heart_wave: np.ndarray
            metrics: {
                'algorithm': str,
                'latency_ms': float,
                'breath_band_energy_ratio': float | None,
                'heart_band_energy_ratio': float | None,
            }
        """
```

### 3.2 Concrete Strategy Classes

| Class | Base | Wraps | Pipeline |
|-------|------|-------|----------|
| `VMDRLSCleaner` | `SignalCleanerStrategy` | `vmd_rls_harmonic_clean` | HR |
| `EMDHarmonicCleaner` | `SignalCleanerStrategy` | `emd_harmonic_clean` | HR |
| `EMDPulseCleaner` | `SignalCleanerStrategy` | `clean_pulse_wave` | BP |
| `PassthroughCleaner` | `SignalCleanerStrategy` | identity (ablation baseline) | HR |
| `WPDSeparator` | `VitalSignSeparator` | `wpd_separate` | HR |
| `SOSFilterSeparator` | `VitalSignSeparator` | `savgol_filter` + `filter_heart` | HR |

**Design rules**:
- Each concrete class catches its own internal exceptions and returns the original signal + empty metrics on failure. Callers never need `try...except` wrappers.
- `ICACleaner` reserved as a stub for future blind-source-separation experiments.
- `SavGolHarmonicCleaner` is intentionally omitted — SavGol already runs in `_shared_signal_chain` for all paths; a separate cleaner would double-filter.

### 3.3 Ablation Matrix (direct paper material)

```
               │ WPDSeparator      │ SOSFilterSeparator
───────────────┼────────────────────┼────────────────────
VMDRLSCleaner  │ VMD+RLS → WPD      │ VMD+RLS → SOS
EMDCleaner     │ EMD → WPD          │ EMD → SOS
Passthrough    │ WPD only           │ SOS only (baseline)
```

### 3.4 AdaptiveStrategySelector

Not a `SignalCleanerStrategy` itself — sits in the Pipeline as a routing layer:

```python
class AdaptiveStrategySelector:
    """SQI-driven dynamic routing of cleaner + separator pairs."""

    def __init__(self,
                 light: tuple[SignalCleanerStrategy, VitalSignSeparator],
                 standard: tuple[SignalCleanerStrategy, VitalSignSeparator],
                 heavy: tuple[SignalCleanerStrategy, VitalSignSeparator],
                 evaluation_interval: int = 50):
        ...

    def select(self, phase_range: float, snr_db: float
               ) -> tuple[SignalCleanerStrategy, VitalSignSeparator]:
        # phase_range > 0.02 & snr > 15 dB → light (Passthrough + SOS)
        # 0.005 < phase_range ≤ 0.02        → standard (VMD+RLS + WPD)
        # phase_range ≤ 0.005 or snr < 5 dB → heavy (EMD + WPD)
```

Decision is re-evaluated every `evaluation_interval` frames to prevent output jitter.

---

## 4. Benchmark Layer (`utils/benchmark_logger.py`) [NEW]

### 4.1 Data Model

```python
@dataclass
class BenchmarkRecord:
    timestamp: float
    frame_index: int
    elapsed_sec: float
    algorithm_name: str          # "VMD_RLS+WPD" | "EMD+WPD" | "Adaptive" | ...
    is_primary: bool             # True = drives UI; False = A/B silent
    latency_ms: float

    # Signal features
    input_phase_range: float
    input_snr_db: float | None
    output_phase_range: float
    output_snr_db: float | None
    snr_gain_db: float | None

    # Extracted vitals
    breath_bpm: float
    heart_bpm: float
    heart_prominence: float

    # Algorithm internals (None = N/A for this algorithm)
    imf_count: int | None
    convergence_iter: int | None
    retained_harmonics: str | None   # "2,3,4"
    dominant_freq_hz: float | None
```

### 4.2 AlgorithmBenchmarker API

```python
class AlgorithmBenchmarker:
    def __init__(self, output_dir: str = "data/benchmarks",
                 session_id: str | None = None): ...

    def log(self, record: BenchmarkRecord) -> None:
        """Non-blocking enqueue. Drops oldest on queue full."""

    def mark_ground_truth(self, label: str,
                          timestamp: float | None = None) -> None:
        """External reference event (e.g., cuff SBP=118)."""

    def start(self) -> None: ...
    def stop(self) -> None: ...

    @property
    def is_recording(self) -> bool: ...
    @property
    def record_count(self) -> int: ...
    @property
    def session_path(self) -> str: ...
```

### 4.3 Thread Model

```
DSP Thread(s) ──→ queue.Queue(maxsize=5000) ──→ Writer Thread
                     (drop oldest if full)         ├─ CSV (streaming append per record)
                                                   └─ HDF5 (batch flush per 100 records)
```

### 4.4 Output Layout

```
data/benchmarks/2026-06-22_14-30-15/
├── benchmark.csv           # All algorithms, all frames
├── benchmark.h5            # HDF5 grouped by algorithm_name
│   ├── /VMDRLS_WPD/
│   ├── /EMD_WPD/
│   ├── /Adaptive/
│   └── /ground_truth/
└── ground_truth.csv        # Human-readable copy of external labels
```

### 4.5 CSV Columns

```
timestamp, elapsed_sec, frame_index, algorithm, is_primary, latency_ms,
input_phase_range, output_phase_range, snr_gain_db,
breath_bpm, heart_bpm, heart_prominence,
imf_count, convergence_iter, retained_harmonics, dominant_freq_hz
```

---

## 5. Pipeline Refactoring

### 5.1 `pipeline.py` — HR Pipeline

**Constructor changes**:
```python
class Pipeline:
    def __init__(self,
                 cleaner: SignalCleanerStrategy | None = None,
                 separator: VitalSignSeparator | None = None,
                 use_adaptive: bool = True):
        self._cleaner = cleaner or VMDRLSCleaner()
        self._separator = separator or WPDSeparator()

        # Adaptive routing (replaces _use_advanced_dsp flag)
        self._adaptive_selector = AdaptiveStrategySelector(
            light=(PassthroughCleaner(), SOSFilterSeparator()),
            standard=(VMDRLSCleaner(), WPDSeparator()),
            heavy=(EMDHarmonicCleaner(), WPDSeparator()),
        ) if use_adaptive else None

        # Thread pool: max_workers=2 for A/B parallelism
        self._dsp_executor = ThreadPoolExecutor(max_workers=2)
        self._ab_enabled: bool = False
        self._ab_cleaner: SignalCleanerStrategy | None = None
        self._ab_separator: VitalSignSeparator | None = None

        # Benchmarker
        self._benchmarker: AlgorithmBenchmarker | None = None
```

**`_advanced_dsp_path` changes**:
- Remove `try...except` fallback that hardcodes `savgol_filter`
- Replace `vmd_rls_harmonic_clean()` + `wpd_separate()` with `self._cleaner.clean()` + `self._separator.separate()`
- Return signature expands from 5-tuple to 6-tuple (adds `combined_metrics: dict`)
- Each strategy handles its own errors internally

**`_shared_signal_chain` changes**:
- AdaptiveSelector evaluation: on each `update_bpm`, call `self._adaptive_selector.select(phase_range, snr)` to get current (cleaner, separator) pair
- A/B dispatch: if `_ab_enabled`, submit alternative `_advanced_dsp_path` to executor with alt strategies
- A/B result handling: collect metrics only, do NOT affect UI display
- Metric forwarding: call `self._benchmarker.log(record)` for both primary and A/B results when recording

**Strategy hot-swap API**:
```python
def set_strategies(self, cleaner: SignalCleanerStrategy,
                   separator: VitalSignSeparator) -> None:
    """Thread-safe: applies on next update_bpm dispatch."""

def set_ab_strategy(self, cleaner: SignalCleanerStrategy | None,
                    separator: VitalSignSeparator | None) -> None:
    """None disables A/B; otherwise sets alternative pair."""
```

### 5.2 `bp_pipeline.py` — BP Pipeline

**Minimal changes**:
```python
class BPPipeline:
    def __init__(self, weights_path: str,
                 cleaner: SignalCleanerStrategy | None = None,
                 benchmarker: AlgorithmBenchmarker | None = None):
        self._cleaner = cleaner or EMDPulseCleaner()
        self._benchmarker = benchmarker
        ...
```

**`_process_snapshot` change**:
```python
# BEFORE:  clean = clean_pulse_wave(wave_50hz_raw, fs=self.FS_TARGET)
# AFTER:
clean, metrics = self._cleaner.clean(wave_50hz_raw, fs=self.FS_TARGET)
if self._benchmarker and self._benchmarker.is_recording:
    self._benchmarker.log(BenchmarkRecord(
        algorithm_name=metrics.get('algorithm', 'Unknown'),
        is_primary=True,
        latency_ms=metrics.get('latency_ms', 0.0),
        ...
    ))
```

### 5.3 Error Handling Philosophy

- **No more `try...except` at the pipeline level**. Each strategy catches its own internal errors and returns `(original_signal, {'algorithm': ..., 'error': str})`.
- **No more `_use_advanced_dsp` flag**. The AdaptiveSelector subsumes this: when SQI is critically low it routes to `PassthroughCleaner + SOSFilterSeparator`, which is the functional equivalent of the old fallback path.
- **No more auto-disable on error**. The old code set `self._use_advanced_dsp = False` on async failure. The new code retries with the current strategy on the next frame — if a strategy consistently fails, the AdaptiveSelector naturally routes away.

---

## 6. UI Layer

### 6.1 Algorithm Control Panel (`research_tab.py`)

A horizontal control bar inserted above the waveform area:

```
┌──────────────────────────────────────────────────────────────────┐
│ Algorithm: [Adaptive ▼]  A/B: [Off ▼]  [🔴 Record]  00:03:42    │
└──────────────────────────────────────────────────────────────────┘
```

| Control | Widget | Behavior |
|---------|--------|----------|
| Algorithm | `QComboBox` | Options: Adaptive, VMD+RLS+WPD, EMD+WPD, Passthrough+SOS. Emits signal to `MonitorMode.set_strategies()` |
| A/B | `QComboBox` | Options: Off, VMD+RLS+WPD, EMD+WPD, Passthrough+SOS. Duplicate of primary is disabled. Off = `set_ab_strategy(None, None)` |
| Record | `QPushButton` | Toggle: "🔴 Record" ↔ "⏹ Stop Recording". Controls `benchmarker.start()` / `benchmarker.stop()` |
| Timer | `QLabel` | Visible only when recording. Shows `HH:MM:SS` elapsed |

### 6.2 Debug Panel Extension

Add 2 lines below existing 7 lines:

```
[DSP Engine] Current: VMD_RLS+WPD | Latency: 24ms | SNR_Gain: +3.2dB
[DSP Engine] A/B: EMD+WPD | Latency: 31ms | SNR_Gain: +2.1dB
```

These lines are only shown when the debug panel is expanded. When A/B is Off, the second line is hidden.

### 6.3 Signal Flow

```
ResearchTab.comboBox_changed
  → HRMode.set_cleaner(c) / set_separator(s)
    → Pipeline.set_strategies(c, s)

ResearchTab.ab_combo_changed
  → HRMode.set_ab_strategy(c, s)
    → Pipeline.set_ab_strategy(c, s)

ResearchTab.record_clicked
  → HRMode.toggle_benchmark()
    → Pipeline.benchmarker.start() / .stop()

ResearchTab.record_timer (QTimer, 1s interval)
  → poll Pipeline.benchmarker.elapsed_sec → update QLabel
```

### 6.4 Safety Constraints

- **Hot-swap deferred**: Strategy changes take effect on next `update_bpm` dispatch, never mid-computation.
- **A/B dedup**: UI disables selecting the same algorithm pair for both primary and A/B.
- **Disk guard**: Benchmarker starts in stopped state; requires explicit user click to record.
- **No A/B in BP mode**: The BP pipeline runs on a simpler single-algorithm model; A/B is HR-mode only for the current version.

---

## 7. Interaction with Existing Systems

### 7.1 Compatibility

| Existing Feature | Impact |
|-----------------|--------|
| `MonitorMode` (HRMode / BPMode) | HRMode injects strategies into Pipeline; BPMode injects cleaner into BPPipeline |
| `data_exporter.py` | Unchanged. Benchmark data is separate from user export data |
| `config/protocol.py` | May add 2-3 new constants for AdaptiveSelector thresholds |
| Serial I/O, UART parsing | No changes |
| BP network inference | No changes; only the signal cleaning step is abstracted |
| `VitalSigns` / `BPResult` dataclasses | No changes |

### 7.2 New Dependencies

- No new PyPI packages. All algorithms (VMD, EMD, WPD) are already in `requirements.txt` (`vmdpy`, `PyEMD`, `pywt`).
- `h5py` for HDF5 benchmark output (already available via existing export infrastructure).

---

## 8. File Change Summary

| File | Action | Scope |
|------|--------|-------|
| `dsp_pipeline/strategies.py` | **NEW** | ~250 lines: ABCs + 6 concrete classes + AdaptiveStrategySelector |
| `utils/benchmark_logger.py` | **NEW** | ~200 lines: BenchmarkRecord + AlgorithmBenchmarker |
| `dsp_pipeline/pipeline.py` | **MODIFY** | ~80 lines changed: constructor, `_advanced_dsp_path`, `_shared_signal_chain`, + hot-swap API, + A/B dispatch |
| `dsp_pipeline/__init__.py` | **MODIFY** | Add strategy exports |
| `bp_monitor/bp_pipeline.py` | **MODIFY** | ~20 lines changed: constructor + cleaner injection in `_process_snapshot` |
| `ui/research_tab.py` | **MODIFY** | ~100 lines added: Algorithm control panel + debug panel extension |
| `ui/monitor_mode.py` | **MODIFY** | ~40 lines changed: HRMode strategy/benchmarker wiring |
| `ui/main_window.py` | **MODIFY** | ~10 lines: wire algorithm change signals |
| `config/protocol.py` | **MODIFY** | ~5 lines: AdaptiveSelector threshold constants |
| `config/i18n.py` | **MODIFY** | ~10 lines: new UI text keys |

---

## 9. Testing Strategy

### Unit Tests
- Each strategy class: verify `clean()`/`separate()` returns correct shapes and metrics dict keys
- `PassthroughCleaner`: verify output equals input
- `AdaptiveStrategySelector`: verify routing decisions at boundary SQI values
- `AlgorithmBenchmarker`: verify CSV/HDF5 file creation, queue overflow behavior, start/stop idempotency

### Integration Tests
- Pipeline with `PassthroughCleaner + SOSFilterSeparator` vs `VMDRLSCleaner + WPDSeparator` — verify both produce valid BPM on a known test signal
- A/B mode: verify both primary and secondary records appear in benchmark output
- Hot-swap: change strategy mid-stream, verify no data corruption or crash

### Smoke Tests (Manual)
- Launch app, switch algorithm dropdown, verify Debug panel shows correct current algorithm
- Click Record, wait 30s, stop, verify CSV file exists with expected columns
- Run adaptive mode on recorded data, verify it switches strategies when signal quality changes

---

## 10. Rollout Phases (within single implementation session)

1. Create `strategies.py` + `benchmark_logger.py` (no pipeline changes yet)
2. Refactor `pipeline.py` — inject strategies, remove `try...except`, add AdaptiveSelector
3. Refactor `bp_pipeline.py` — inject cleaner
4. Update `monitor_mode.py` — wire strategies through HRMode/BPMode
5. Update UI (`research_tab.py` + `main_window.py`)
6. Add config constants + i18n keys
7. Integration test on recorded radar data
