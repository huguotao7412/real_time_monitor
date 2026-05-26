"""Medical-grade data export: HDF5 (research), EDF (clinical), CSV (basic).

HDF5 schema follows BIDS-derivative conventions for physiological signals:
    /metadata           — subject, device, sampling rate, timestamp
    /processed_signals  — breath_waveform, heart_waveform (1D arrays)
    /vital_signs        — BPM_history, RPM_history (time-synced)
    /sqi                — signal quality index arrays

EDF export stores breath + heart waveforms as standard EDF channels
compatible with EDFbrowser, EEGLAB, and MNE-Python.
"""

import os
import csv
from datetime import datetime
from typing import Any

import numpy as np


def export_csv(
    path: str,
    csv_rows: list[dict],
    breath_waveform: np.ndarray,
    heart_waveform: np.ndarray,
) -> str:
    """Export vital signs snapshot rows as CSV (legacy format).

    Returns the written file path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save .npz waveforms
    npz_path = os.path.join(path, f"vital_signs_{ts}.npz")
    np.savez(
        npz_path,
        breath_waveform=breath_waveform,
        heart_waveform=heart_waveform,
    )

    # Save .csv rows
    csv_path = os.path.join(path, f"vital_signs_{ts}.csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    return csv_path


def export_hdf5(
    path: str,
    breath_waveform_history: np.ndarray,
    heart_waveform_history: np.ndarray,
    bpm_history: list[tuple[float, float, float]],
    sqi_history: list[dict],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Export session data as structured HDF5 file.

    Args:
        path: output directory path.
        breath_waveform_history: (N,) or (N, window_size) array.
        heart_waveform_history: (N,) or (N, window_size) array.
        bpm_history: list of (timestamp, breath_bpm, heart_bpm).
        sqi_history: list of dicts with keys phase_range, breath_ratio, sqi_level.
        metadata: optional dict with subject_id, device, fs, etc.

    Returns the written HDF5 file path.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 export. Install with: pip install h5py"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_path = os.path.join(path, f"vital_signs_{ts}.h5")

    with h5py.File(h5_path, "w") as f:
        # ── /metadata ──
        meta = f.create_group("metadata")
        meta.attrs["export_timestamp"] = datetime.now().isoformat()
        meta.attrs["format_version"] = "1.0"
        meta.attrs["sampling_rate_hz"] = metadata.get("fs", 20) if metadata else 20
        meta.attrs["device_model"] = metadata.get("device", "RS6240") if metadata else "RS6240"
        if metadata and "subject_id" in metadata:
            meta.attrs["subject_id"] = metadata["subject_id"]
        if metadata and "session_duration_s" in metadata:
            meta.attrs["session_duration_s"] = metadata["session_duration_s"]

        # ── /processed_signals ──
        sig = f.create_group("processed_signals")
        if breath_waveform_history.ndim == 2:
            # Store as 2D (time × window)
            sig.create_dataset(
                "breath_waveform", data=breath_waveform_history.astype(np.float32),
                compression="gzip", compression_opts=4,
            )
            sig["breath_waveform"].attrs["description"] = (
                "Respiratory waveform (sliding window). Shape: (time_frames, window_size)."
            )
        else:
            sig.create_dataset(
                "breath_waveform", data=breath_waveform_history.astype(np.float32),
                compression="gzip", compression_opts=4,
            )
            sig["breath_waveform"].attrs["description"] = "Respiratory waveform (1D time series)."

        if heart_waveform_history.size > 0:
            if heart_waveform_history.ndim == 2:
                sig.create_dataset(
                    "heart_waveform", data=heart_waveform_history.astype(np.float32),
                    compression="gzip", compression_opts=4,
                )
            else:
                sig.create_dataset(
                    "heart_waveform", data=heart_waveform_history.astype(np.float32),
                    compression="gzip", compression_opts=4,
                )
            sig["heart_waveform"].attrs["description"] = "Heartbeat waveform."

        # ── /vital_signs ──
        vs = f.create_group("vital_signs")
        if bpm_history:
            bpm_arr = np.array(bpm_history, dtype=np.float32)
            vs.create_dataset("bpm_timestamps", data=bpm_arr[:, 0], compression="gzip")
            vs["bpm_timestamps"].attrs["unit"] = "seconds from session start"
            vs.create_dataset("breath_bpm", data=bpm_arr[:, 1], compression="gzip")
            vs["breath_bpm"].attrs["unit"] = "breaths per minute"
            vs.create_dataset("heart_bpm", data=bpm_arr[:, 2], compression="gzip")
            vs["heart_bpm"].attrs["unit"] = "beats per minute"

        # ── /sqi ──
        sqi_g = f.create_group("sqi")
        if sqi_history:
            sqi_keys = ["phase_range", "breath_ratio", "sqi_level"]
            for key in sqi_keys:
                vals = [s.get(key, 0.0) for s in sqi_history]
                sqi_g.create_dataset(key, data=np.array(vals, dtype=np.float32),
                                     compression="gzip")

    return h5_path


def export_edf(
    path: str,
    breath_waveform: np.ndarray,
    heart_waveform: np.ndarray,
    fs: float = 20.0,
) -> str:
    """Export breath and heart waveforms as EDF+ file.

    Maps respiratory waveform to standard EDF channel label 'Resp'
    and heartbeat waveform to 'ECG' for compatibility with clinical
    viewers (EDFbrowser, EEGLAB, MNE-Python).

    Args:
        path: output directory path.
        breath_waveform: 1D array of respiratory signal.
        heart_waveform: 1D array of heartbeat signal (can be empty).
        fs: sampling frequency in Hz (default 20).

    Returns the written EDF file path.
    """
    try:
        import pyedflib
    except ImportError:
        raise ImportError(
            "pyedflib is required for EDF export. Install with: pip install pyedflib"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    edf_path = os.path.join(path, f"vital_signs_{ts}.edf")

    # Build channel list
    signals = []
    labels = []
    phys_mins = []
    phys_maxs = []
    dig_mins = []
    dig_maxs = []

    if breath_waveform.size > 0:
        signals.append(breath_waveform.astype(np.float64).ravel())
        labels.append("Resp")
        phys_mins.append(-1.0)
        phys_maxs.append(1.0)
        dig_mins.append(-32768)
        dig_maxs.append(32767)

    if heart_waveform.size > 0:
        signals.append(heart_waveform.astype(np.float64).ravel())
        labels.append("ECG")
        phys_mins.append(-1.0)
        phys_maxs.append(1.0)
        dig_mins.append(-32768)
        dig_maxs.append(32767)

    if not signals:
        raise ValueError("No signal data to export.")

    # All signals must have the same length; pad if needed
    max_len = max(s.size for s in signals)
    padded = []
    for s in signals:
        if s.size < max_len:
            s = np.pad(s, (0, max_len - s.size), mode="constant")
        padded.append(s)

    n_channels = len(padded)
    signal_data = np.array(padded)

    with pyedflib.EdfWriter(edf_path, n_channels=n_channels) as writer:
        header = {
            "technician": "RS6240 Radar",
            "recording_additional": "mmWave vital signs monitor",
            "patientname": "Subject",
            "patient_additional": "",
            "patientcode": "",
            "equipment": "RS6240",
            "admincode": "",
            "gender": 0,  # not specified
            "startdate": datetime.now(),
        }
        writer.setHeader(header)
        writer.setSampleFrequency(0, fs)
        writer.setPhysicalMaximum(0, phys_maxs[0])
        writer.setPhysicalMinimum(0, phys_mins[0])
        writer.setDigitalMaximum(0, dig_maxs[0])
        writer.setDigitalMinimum(0, dig_mins[0])
        writer.setLabel(0, labels[0])

        if n_channels > 1:
            writer.setSampleFrequency(1, fs)
            writer.setPhysicalMaximum(1, phys_maxs[1])
            writer.setPhysicalMinimum(1, phys_mins[1])
            writer.setDigitalMaximum(1, dig_maxs[1])
            writer.setDigitalMinimum(1, dig_mins[1])
            writer.setLabel(1, labels[1])

        writer.writeSamples(signal_data.astype(np.int16))

    return edf_path
