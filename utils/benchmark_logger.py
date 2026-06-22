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

    # CSV column order, kept in sync with log writing
    CSV_COLUMNS = [
        "timestamp", "elapsed_sec", "frame_index", "algorithm_name",
        "is_primary", "latency_ms",
        "input_phase_range", "input_snr_db", "output_phase_range",
        "output_snr_db", "snr_gain_db",
        "breath_bpm", "heart_bpm", "heart_prominence",
        "imf_count", "convergence_iter", "retained_harmonics",
        "dominant_freq_hz",
    ]


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

        # Ground truth file
        self._gt_file = None
        self._gt_writer = None

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
        # Enqueue sentinel to flush queue naturally
        try:
            self._queue.put_nowait("__STOP__")
        except queue.Full:
            self._stop_event.set()  # fallback: force stop if queue backed up
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
        try:
            self._open_files()
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
        except Exception as e:
            import traceback
            print(f"[Benchmarker] Writer thread error: {e}")
            traceback.print_exc()
        finally:
            self._flush_hdf5()
            self._close_files()

    def _open_files(self) -> None:
        """Open CSV and HDF5 file handles."""
        # CSV
        csv_path = os.path.join(self._session_path, "benchmark.csv")
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=BenchmarkRecord.CSV_COLUMNS
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()

        # Ground truth CSV
        gt_path = os.path.join(self._session_path, "ground_truth.csv")
        self._gt_file = open(gt_path, "w", newline="", encoding="utf-8")
        self._gt_writer = csv.writer(self._gt_file)
        self._gt_writer.writerow(["timestamp", "elapsed_sec", "label"])
        self._gt_file.flush()

        # HDF5 (opened lazily on first record to avoid empty files)
        self._h5_file = None

    def _write_csv(self, record: BenchmarkRecord) -> None:
        """Stream a single row to the CSV file."""
        d = asdict(record)
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
        try:
            if self._h5_file is None:
                self._open_hdf5()
            if self._h5_file is None:
                return  # HDF5 unavailable, skip
            with self._h5_lock:
                self._h5_batch.append(record)
                if len(self._h5_batch) >= self._HDF5_FLUSH_BATCH:
                    self._flush_hdf5()
        except Exception:
            pass  # HDF5 is best-effort; never block CSV recording

    def _open_hdf5(self) -> None:
        try:
            import h5py
            h5_path = os.path.join(self._session_path, "benchmark.h5")
            self._h5_file = h5py.File(h5_path, "w")
            self._h5_file.attrs["session_id"] = self._session_id
            self._h5_file.attrs["created"] = datetime.now().isoformat()
        except Exception:
            self._h5_file = None  # mark HDF5 as unavailable

    def _flush_hdf5(self) -> None:
        """Write accumulated records to HDF5, grouped by algorithm_name."""
        if self._h5_file is None or not self._h5_batch:
            return
        try:
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
        except Exception:
            self._h5_batch.clear()
            self._h5_file = None

    def _close_files(self) -> None:
        """Close all open file handles."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        if self._gt_file:
            self._gt_file.close()
            self._gt_file = None
        if self._h5_file:
            self._h5_file.close()
            self._h5_file = None
