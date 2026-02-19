"""
Sync and alignment utilities (the "Sync-Tap").

Goal:
- Scan the accelerometer Z-axis for three high-magnitude spikes within a 2-second window.
- Treat the **3rd spike** as **T=0** for the entire session.
- Shift the 50 Hz session index so the 3rd tap is at 0.0 s; align manual logs with 50 Hz PPG.

Inputs:
- data/processed/session_50hz.csv    (50 Hz grid with accel_z; or use 100 Hz accel if provided)
- data/processed/accelerometer.csv   (optional; 100 Hz accel for finer tap detection)
- data/datasets/manual_labels.csv   (optional; timestamps to be aligned)

Outputs:
- data/processed/session_50hz_aligned.csv
- data/datasets/manual_labels_aligned.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


FS_50 = 50.0   # Hz, session_50hz
FS_100 = 100.0 # Hz, native accelerometer
SYNC_WINDOW_S = 2.0  # Three spikes must occur within this window (seconds)


def load_session_50hz(path: str | Path | None = None) -> pd.DataFrame:
    """Load 50 Hz session CSV with datetime index."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "processed" / "session_50hz.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"session_50hz.csv not found: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def load_accel_100hz(path: str | Path | None = None) -> Optional[pd.DataFrame]:
    """Load 100 Hz accelerometer CSV (timestamp_s, accel_x, accel_y, accel_z). Return None if missing."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "processed" / "accelerometer.csv"
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "accel_z" not in df.columns or "timestamp_s" not in df.columns:
        return None
    return df


def _accel_z_signal(df: pd.DataFrame, z_column: str = "accel_z") -> np.ndarray:
    """Z-axis signal for tap detection (absolute value for spike detection)."""
    if z_column not in df.columns:
        raise ValueError(f"Column '{z_column}' not found; need accelerometer Z-axis for sync taps.")
    return np.abs(df[z_column].astype(float).to_numpy())


def _find_three_taps_in_signal(
    z_abs: np.ndarray,
    fs: float,
    window_seconds: float = SYNC_WINDOW_S,
) -> int:
    """
    Find first occurrence of 3 high-magnitude spikes within a 2-second window on Z-axis.
    Returns sample index of the 3rd spike (T=0).
    """
    n = len(z_abs)
    max_span_samples = int(window_seconds * fs)
    if n < max_span_samples:
        raise ValueError(f"Not enough samples for a {window_seconds}s window (need >= {max_span_samples} at {fs} Hz).")

    m = float(np.nanmean(z_abs))
    s = float(np.nanstd(z_abs))
    if s > 0:
        thresh = m + 3.0 * s
    else:
        thresh = float(np.nanpercentile(z_abs, 95))

    min_distance = max(1, int(0.08 * fs))  # ~80 ms between taps
    peaks, _ = find_peaks(z_abs, height=thresh, distance=min_distance)
    if len(peaks) < 3:
        raise ValueError("Fewer than 3 high-magnitude Z-axis peaks found; cannot locate sync taps.")

    for i in range(len(peaks) - 2):
        p0, p1, p2 = peaks[i], peaks[i + 1], peaks[i + 2]
        if p2 - p0 <= max_span_samples:
            return int(p2)  # 3rd spike index
    raise ValueError(f"Could not find 3 taps within a {window_seconds}s window.")


def find_three_tap_anchor(
    df_session: pd.DataFrame,
    accel_100hz_path: str | Path | None = None,
) -> Tuple[int, pd.Timestamp]:
    """
    Find the 3-tap sync anchor using the accelerometer Z-axis.

    - If 100 Hz accelerometer CSV is available (data/processed/accelerometer.csv or path),
      use it for detection; then map the 3rd tap time onto the session timeline so the
      session can be aligned.
    - Otherwise use accel_z from the 50 Hz session and require 3 spikes within 2 seconds.

    Returns
    -------
    (anchor_idx, anchor_time)
        anchor_idx  : row index in df_session corresponding to the 3rd tap (T=0).
        anchor_time : Timestamp of the 3rd tap (for shifting labels).
    """
    df_accel_100 = load_accel_100hz(accel_100hz_path)
    if df_accel_100 is not None and len(df_accel_100) > int(SYNC_WINDOW_S * FS_100):
        # Use 100 Hz Z-axis; then map 3rd tap time onto session timeline
        z_abs = _accel_z_signal(df_accel_100)
        anchor_idx_100 = _find_three_taps_in_signal(z_abs, FS_100, SYNC_WINDOW_S)
        t_anchor_s = float(df_accel_100["timestamp_s"].iloc[anchor_idx_100])
        t0_accel_s = float(df_accel_100["timestamp_s"].min())
        # Session start = first row; assume accel timestamp_s aligns (same run).
        session_start = df_session.index[0]
        anchor_time = session_start + pd.Timedelta(seconds=(t_anchor_s - t0_accel_s))
        # Clamp to session range
        if anchor_time > df_session.index[-1]:
            anchor_time = df_session.index[-1]
        elif anchor_time < session_start:
            anchor_time = session_start
        # Closest session row to anchor_time
        anchor_idx = int(np.searchsorted(df_session.index, anchor_time, side="left"))
        if anchor_idx >= len(df_session):
            anchor_idx = len(df_session) - 1
        if anchor_idx > 0 and (anchor_time - df_session.index[anchor_idx - 1]).total_seconds() < (df_session.index[anchor_idx] - anchor_time).total_seconds():
            anchor_idx -= 1
        anchor_time = df_session.index[anchor_idx]
        return anchor_idx, anchor_time

    # Use 50 Hz session accel_z
    if "accel_z" not in df_session.columns:
        raise ValueError("No accel_z in session and no 100 Hz accelerometer CSV; cannot detect sync taps.")
    z_abs = _accel_z_signal(df_session)
    anchor_idx = _find_three_taps_in_signal(z_abs, FS_50, SYNC_WINDOW_S)
    anchor_time = df_session.index[anchor_idx]
    return anchor_idx, anchor_time


def shift_session_index_to_anchor(df: pd.DataFrame, anchor_time: pd.Timestamp) -> pd.DataFrame:
    """
    Shift DataFrame index so that the anchor time becomes 0.0 seconds.

    Resulting index is a Float64Index in seconds (elapsed from anchor).
    """
    # Compute time delta from anchor in seconds
    delta = (df.index - anchor_time).total_seconds()
    df_aligned = df.copy()
    df_aligned.index = pd.Index(delta, name="time_s")
    return df_aligned


def load_manual_labels(path: str | Path | None = None) -> Optional[pd.DataFrame]:
    """Load manual_labels.csv if present; return None if missing."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "datasets" / "manual_labels.csv"
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def align_manual_labels(df_labels: pd.DataFrame, anchor_time: pd.Timestamp) -> pd.DataFrame:
    """
    Apply the same time shift to manual labels so timestamps match the aligned session.

    Segment format (from label_generator.py): columns start_time, end_time, label.
    - If both start_time and end_time exist: add start_time_s and end_time_s (seconds from anchor).
    - Otherwise look for a single time column: 'timestamp', 'datetime', or 'time', and add time_s.
    """
    df = df_labels.copy()

    if "start_time" in df.columns and "end_time" in df.columns:
        start = pd.to_datetime(df["start_time"])
        end = pd.to_datetime(df["end_time"])
        df["start_time_s"] = (start - anchor_time).total_seconds()
        df["end_time_s"] = (end - anchor_time).total_seconds()
        return df

    # Single time column
    time_col = None
    for cand in ("timestamp", "datetime", "time"):
        if cand in df.columns:
            time_col = cand
            break

    if time_col is not None:
        times = pd.to_datetime(df[time_col])
    else:
        try:
            times = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(
                "Could not find a datetime column ('timestamp', 'datetime', 'time', "
                "'start_time'/'end_time') or parse the index as datetime in manual_labels.csv"
            ) from e

    df["time_s"] = (times - anchor_time).total_seconds()
    return df


def main(
    session_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    accel_100hz_path: str | Path | None = None,
) -> None:
    """
    High-level entry point:
    - Load 50 Hz session (and optional 100 Hz accelerometer).
    - Scan accel Z-axis for 3 high-magnitude spikes within a 2 s window; 3rd spike = T=0.
    - Shift session index so anchor is at 0.0 s and save aligned CSV.
    - Load manual labels (if present), apply same shift, and save aligned labels.
    """
    df_session = load_session_50hz(session_path)
    anchor_idx, anchor_time = find_three_tap_anchor(df_session, accel_100hz_path=accel_100hz_path)
    print(f"Sync anchor (3 sharp taps, Z-axis, 2 s window): idx={anchor_idx}, time={anchor_time} → T=0")

    # Shift session index to anchor
    df_aligned = shift_session_index_to_anchor(df_session, anchor_time)
    session_out = Path(__file__).resolve().parents[2] / "data" / "processed" / "session_50hz_aligned.csv"
    session_out.parent.mkdir(parents=True, exist_ok=True)
    df_aligned.to_csv(session_out)
    print(f"Wrote aligned session to {session_out}")

    # Align manual labels if available
    df_labels = load_manual_labels(labels_path)
    if df_labels is None:
        print("No manual_labels.csv found; skipping label alignment.")
        return

    df_labels_aligned = align_manual_labels(df_labels, anchor_time)
    labels_out = Path(__file__).resolve().parents[2] / "data" / "datasets" / "manual_labels_aligned.csv"
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    df_labels_aligned.to_csv(labels_out, index=False)
    print(f"Wrote aligned manual labels to {labels_out}")


if __name__ == "__main__":
    import sys

    session_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    labels_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    accel_arg = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    main(session_path=session_arg, labels_path=labels_arg, accel_100hz_path=accel_arg)

