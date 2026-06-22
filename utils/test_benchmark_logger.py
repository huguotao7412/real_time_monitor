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
        expected_cols = "timestamp,elapsed_sec,frame_index,algorithm_name,is_primary"
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
