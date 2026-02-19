"""
Generate verification plots for Oralable MAM test sequence.

Applies digital filtering to reveal the true signal beneath raw noise:
- Butterworth bandpass (0.5–8 Hz) for PPG heartbeat (per .cursorrules)
- Median filter for accelerometer (keeps sync taps, removes jitter)
- Rolling mean for IR DC trend (clench occlusion dip)

Creates high-resolution PNG with 3 panels:
1. Accelerometer Z (median filtered) – sync taps = 3 sharp spikes
2. PPG AC (bandpass filtered) – rhythmic heartbeat
3. IR DC trend – clench dip / trough

Usage:
    python -m src.analysis.visualize_test
    python -m src.analysis.visualize_test data/processed/session_50hz.csv
    python -m src.analysis.visualize_test --ppg data/processed/ppg_50hz.csv --accel data/processed/accelerometer.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PLOTS_DIR = Path(__file__).resolve().parents[2] / "data" / "plots"
FS_PPG = 50.0  # Hz (per .cursorrules)


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Butterworth bandpass (clinical standard). Removes DC drift and high-freq noise."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data.astype(float))


def median_filter_1d(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Median filter. Keeps sharp spikes (sync taps) but removes random jitter."""
    return median_filter(data.astype(float), size=window, mode="nearest")


def _load_session(session_path: Path | None = None) -> pd.DataFrame:
    """Load session CSV. Returns df with elapsed_s index (seconds from start) for plotting."""
    def to_elapsed(df: pd.DataFrame) -> pd.DataFrame:
        """Convert df to elapsed_s index."""
        if "elapsed_s" in df.columns:
            out = df.set_index("elapsed_s").sort_index()
            return out
        if "datetime" in df.columns:
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["elapsed_s"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds()
            return df.drop(columns=["datetime"]).set_index("elapsed_s").sort_index()
        if "timestamp_s" in df.columns:
            t0 = df["timestamp_s"].iloc[0]
            df = df.copy()
            df["elapsed_s"] = df["timestamp_s"] - t0
            return df.set_index("elapsed_s").sort_index()
        return df

    if session_path and session_path.exists():
        df = pd.read_csv(session_path)
        return to_elapsed(df)

    session = PROCESSED_DIR / "session_50hz.csv"
    if session.exists():
        df = pd.read_csv(session)
        return to_elapsed(df)

    ppg_path = PROCESSED_DIR / "ppg_50hz.csv"
    accel_path = PROCESSED_DIR / "accelerometer.csv"
    if not ppg_path.exists() or not accel_path.exists():
        raise FileNotFoundError(
            f"Need session_50hz.csv or both ppg_50hz.csv and accelerometer.csv in {PROCESSED_DIR}"
        )
    df_ppg = pd.read_csv(ppg_path)
    df_accel = pd.read_csv(accel_path)
    t0 = min(df_ppg["timestamp_s"].iloc[0], df_accel["timestamp_s"].iloc[0])
    df_ppg["elapsed_s"] = df_ppg["timestamp_s"] - t0
    df_accel["elapsed_s"] = df_accel["timestamp_s"] - t0
    df = pd.merge_asof(
        df_ppg.sort_values("elapsed_s"),
        df_accel.sort_values("elapsed_s"),
        on="elapsed_s",
        direction="nearest",
        tolerance=0.05,
    )
    return df.set_index("elapsed_s").sort_index()


def create_verification_plot(
    session_path: Path | None = None,
    out_path: Path | None = None,
    ir_window: int = 50,
    ppg_bandpass: tuple[float, float] = (0.5, 8.0),
    accel_median_window: int = 5,
) -> Path:
    """
    Create 3-panel verification plot with filtered signals.

    Parameters
    ----------
    session_path : Path, optional
        Path to session CSV (session_50hz.csv or custom).
    out_path : Path, optional
        Output PNG path (default: data/plots/test_verification.png).
    ir_window : int
        Rolling window for IR DC trend (default 50 = 1 s at 50 Hz).
    ppg_bandpass : tuple
        (lowcut, highcut) Hz for heartbeat (default 0.5–8 Hz per .cursorrules).
    accel_median_window : int
        Median filter window for accelerometer (default 5).

    Returns
    -------
    Path
        Path to saved PNG.
    """
    df = _load_session(session_path)
    if "accel_z" not in df.columns or "ir" not in df.columns:
        raise ValueError("Session must have 'accel_z' and 'ir' columns")

    if out_path is None:
        out_path = PLOTS_DIR / "test_verification.png"
    out_path = Path(out_path)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    x = df.index.values
    lowcut, highcut = ppg_bandpass

    # 1. Accelerometer: median filter (keeps sync taps, removes jitter)
    accel_z = df["accel_z"].to_numpy()
    accel_clean = median_filter_1d(accel_z, window=accel_median_window)

    # 2. PPG heartbeat: bandpass 0.5–8 Hz (Green channel per .cursorrules)
    ppg_ch = df["green"] if "green" in df.columns else df["ir"]
    ppg_raw = ppg_ch.to_numpy()
    ppg_clean = butter_bandpass_filter(ppg_raw, lowcut, highcut, FS_PPG)

    # 3. IR DC trend: rolling mean (clench occlusion dip)
    ir_trend = df["ir"].rolling(window=ir_window, center=True).mean()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Panel 1: Accelerometer Z (median filtered)
    ax1.plot(x, accel_clean, color="tab:red", alpha=0.9, label="Accel Z (median filtered)")
    ax1.set_title("Accelerometer Z: Look for 3 Sharp Spikes (Sync Taps)")
    ax1.set_ylabel("Raw Units")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Panel 2: PPG AC (bandpass filtered) – heartbeat
    ax2.plot(x, ppg_clean, color="tab:green", alpha=0.9, label=f"PPG AC ({lowcut}-{highcut} Hz)")
    ax2.set_title("Cleaned Heartbeat (AC Component) – Rhythmic Wave")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # Panel 3: IR DC trend (clench occlusion)
    ax3.plot(x, ir_trend, color="tab:purple", linewidth=1.5, label="IR DC-Trend")
    ax3.set_title('Infrared DC: Look for "Dip" during Clenching')
    ax3.set_ylabel("Intensity")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Oralable verification plots")
    parser.add_argument(
        "session",
        nargs="?",
        type=Path,
        default=None,
        help="Path to session_50hz.csv (default: auto-detect)",
    )
    parser.add_argument(
        "--ppg",
        type=Path,
        help="Path to ppg_50hz.csv (used with --accel instead of session)",
    )
    parser.add_argument(
        "--accel",
        type=Path,
        help="Path to accelerometer.csv (used with --ppg)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PLOTS_DIR / "test_verification.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--ir-window",
        type=int,
        default=50,
        help="Rolling window for IR DC trend (default: 50 = 1 s at 50 Hz)",
    )
    args = parser.parse_args()

    session_path = args.session
    if args.ppg and args.accel:
        # Merge ppg + accel on the fly
        df_ppg = pd.read_csv(args.ppg)
        df_accel = pd.read_csv(args.accel)
        t0 = min(df_ppg["timestamp_s"].iloc[0], df_accel["timestamp_s"].iloc[0])
        df_ppg = df_ppg.copy()
        df_ppg["elapsed_s"] = df_ppg["timestamp_s"] - t0
        df_accel = df_accel.copy()
        df_accel["elapsed_s"] = df_accel["timestamp_s"] - t0
        df = pd.merge_asof(
            df_ppg.sort_values("elapsed_s"),
            df_accel.sort_values("elapsed_s"),
            on="elapsed_s",
            direction="nearest",
            tolerance=0.05,
        )
        merged_path = PROCESSED_DIR / "session_merged_50hz.csv"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(merged_path, index=False)
        session_path = merged_path

    try:
        out = create_verification_plot(
            session_path=session_path,
            out_path=args.out,
            ir_window=args.ir_window,
        )
        print(f"Verification plot saved to: {out}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
