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
        evaluation_interval=1,
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
        evaluation_interval=1,
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
        evaluation_interval=1,
    )
    cleaner, sep = selector.select(phase_range=0.002, snr_db=3.0)
    assert isinstance(cleaner, EMDHarmonicCleaner)
    assert isinstance(sep, WPDSeparator)
