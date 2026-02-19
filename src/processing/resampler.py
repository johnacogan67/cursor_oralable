"""
Resample raw CSV to a strict 50 Hz (20 ms) grid using linear interpolation.

Loads the raw CSV (e.g. from log_parser --raw), converts timestamp_s to a datetime index,
resamples to 20 ms bins, and interpolates to fill gaps. Saves session_50hz.csv to data/processed/.
"""

from pathlib import Path

import pandas as pd


def resample_raw_to_50hz(
    raw_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load raw CSV, convert timestamp to datetime index, resample to 20 ms (50 Hz), interpolate, and save.

    Parameters
    ----------
    raw_path : str or Path, optional
        Path to raw CSV (default: data/processed/raw.csv).
    out_path : str or Path, optional
        Path for output (default: data/processed/session_50hz.csv).

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with datetime index at 50 Hz.
    """
    if raw_path is None:
        raw_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "raw.csv"
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_path}")

    if out_path is None:
        out_path = raw_path.parent / "session_50hz.csv"
    out_path = Path(out_path)

    df = pd.read_csv(raw_path)
    if "timestamp_s" not in df.columns:
        raise ValueError("Raw CSV must have a 'timestamp_s' column")

    # Convert timestamp_s (seconds since midnight or similar) to datetime index.
    # Use a fixed reference so resample has a proper datetime range.
    ref = pd.Timestamp("1970-01-01 00:00:00")
    df["datetime"] = ref + pd.to_timedelta(df["timestamp_s"], unit="s")
    df = df.set_index("datetime").drop(columns=["timestamp_s"], errors="ignore")

    # Resample to 20 ms (50 Hz), take mean per bin, then linear interpolate gaps.
    resampled = df.resample("20ms").mean().interpolate(method="linear")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_csv(out_path)
    return resampled


if __name__ == "__main__":
    import sys

    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    df = resample_raw_to_50hz(raw_path=raw, out_path=out)
    print(f"Resampled to 50 Hz: {len(df)} rows")
    print(f"Saved to {out or (Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'session_50hz.csv')}")
    print(df.head(10))
