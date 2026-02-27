#!/usr/bin/env python3
"""
Self-validation pipeline for clinical metrics on Oralable MAM sleep data.

Verifies SpO2 (110-25R empirical curve), SASHB (AUC below 90%), and Airway Rescue
(IR-DC drop >15% in 500ms), with correlation to SpO2 desaturation events.
Adheres to .cursorrules mandate for 100% clinical precision.

Usage:
    PYTHONPATH=. python src/validation/self_validate.py [log_path] [-o output.png]
    python -m src.validation.self_validate data/raw/Oralable_20260223_083911.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Project paths
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "data" / "plots"

FS = 50.0
DT = 0.02
SPO2_DESAT_THRESHOLD = 90.0
RESCUE_DESAT_WINDOW_S = 10.0
IR_DC_TARGET_V_LOW = 1.8
IR_DC_TARGET_V_HIGH = 2.4
# Oralable ADC: 16-bit per channel; assume 3.3V ref for voltage conversion estimate
ADC_TO_V_SCALE = 3.3 / 65535.0
MIN_SAMPLES_FOR_VALIDATION = int(30 * FS)  # 30 seconds minimum


class SignalQualityAlert(Exception):
    """Raised when raw signal quality is insufficient for clinical validation."""

    pass


def _butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low, high = low_hz / nyq, high_hz / nyq
    return butter(order, [low, high], btype="band")


def _butter_lowpass(cutoff_hz: float, fs: float, order: int = 4):
    return butter(order, cutoff_hz / (0.5 * fs), btype="low")


def _butter_highpass(cutoff_hz: float, fs: float, order: int = 4):
    return butter(order, cutoff_hz / (0.5 * fs), btype="high")


def load_raw_ble_log(path: Path) -> pd.DataFrame:
    """Load raw BLE sensor log and produce merged 50Hz-ready dataframe."""
    from src.parser.log_parser import parse_all
    from src.processing.resampler import resample_raw_to_50hz

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw log not found: {path}")

    streams = parse_all(path)
    ppg = streams.get("ppg")
    accel = streams.get("accelerometer")
    if ppg is None or ppg.empty:
        raise SignalQualityAlert("No PPG data in log; cannot compute SpO2 or clinical metrics.")
    if accel is None or accel.empty:
        accel = pd.DataFrame({"timestamp_s": ppg["timestamp_s"], "accel_x": 0, "accel_y": 0, "accel_z": 0})

    t0 = float(ppg["timestamp_s"].iloc[0])
    ppg = ppg.copy()
    ppg["timestamp_s"] = ppg["timestamp_s"] - t0
    accel = accel.copy()
    accel["timestamp_s"] = accel["timestamp_s"] - t0

    merged = pd.merge_asof(
        ppg.sort_values("timestamp_s"),
        accel.sort_values("timestamp_s"),
        on="timestamp_s",
        direction="nearest",
        tolerance=0.1,
    )

    # Write temp merged CSV for resampler
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = PROCESSED_DIR / "self_validate_merged.csv"
    merged.to_csv(merged_path, index=False)

    # Resample to strict 50Hz
    resampled = resample_raw_to_50hz(raw_path=merged_path, out_path=PROCESSED_DIR / "self_validate_50hz.csv")
    return resampled


def compute_heart_rate_from_green(df: pd.DataFrame) -> pd.Series:
    """Estimate instantaneous HR (BPM) from Green PPG bandpass via peak detection."""
    if "green_bp" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    sig = df["green_bp"].to_numpy()
    peaks, _ = find_peaks(sig, distance=int(0.4 * FS), prominence=np.nanstd(sig) * 0.3 if np.isfinite(np.nanstd(sig)) else 0.01)
    hr = np.full(len(df), np.nan)
    for i, p in enumerate(peaks):
        if i > 0:
            rr_s = (df.index[p] - df.index[peaks[i - 1]]).total_seconds()
            if rr_s > 0.3 and rr_s < 2.0:
                hr[p] = 60.0 / rr_s
    return pd.Series(hr, index=df.index).interpolate(method="linear", limit=50)


def flag_rescue_near_desaturation(
    clinical_df: pd.DataFrame,
    desat_threshold: float = SPO2_DESAT_THRESHOLD,
    window_s: float = RESCUE_DESAT_WINDOW_S,
) -> pd.Series:
    """Flag Airway Rescue events that occur within window_s of an SpO2 desaturation."""
    n = len(clinical_df)
    rescue_near_desat = np.zeros(n, dtype=int)
    spo2 = clinical_df["spo2_pct"].to_numpy()
    is_rescue = clinical_df["is_airway_rescue"].to_numpy()
    window_samples = int(window_s * FS)

    desat_indices = np.where(np.isfinite(spo2) & (spo2 < desat_threshold))[0]
    for i in np.where(is_rescue == 1)[0]:
        lo = max(0, i - window_samples)
        hi = min(n, i + window_samples)
        if np.any((desat_indices >= lo) & (desat_indices <= hi)):
            rescue_near_desat[i] = 1
    return pd.Series(rescue_near_desat, index=clinical_df.index)


def compute_green_snr(df: pd.DataFrame) -> float:
    """SNR for Green channel: in-band (0.5-8 Hz) vs out-of-band noise."""
    if "green" not in df.columns:
        return float("nan")
    sig = np.nan_to_num(df["green"].astype(float).to_numpy(), nan=0.0) - np.nanmean(df["green"])
    b_bp, a_bp = _butter_bandpass(0.5, 8.0, FS, order=4)
    b_hp, a_hp = _butter_highpass(12.0, FS, order=4)
    signal_band = filtfilt(b_bp, a_bp, sig)
    noise_band = filtfilt(b_hp, a_hp, sig)
    sig_var = np.var(signal_band)
    noise_var = np.var(noise_band)
    if noise_var < 1e-12:
        return 40.0
    return float(10.0 * np.log10(max(1e-12, sig_var) / max(1e-12, noise_var)))


def audit_signal_quality(df: pd.DataFrame, clinical_df: pd.DataFrame, green_snr_db: float = 0.0) -> None:
    """Raise SignalQualityAlert if quality insufficient for clinical precision."""
    if len(df) < MIN_SAMPLES_FOR_VALIDATION:
        raise SignalQualityAlert(
            f"Insufficient samples: {len(df)} < {MIN_SAMPLES_FOR_VALIDATION} (need ≥30s at 50Hz)."
        )

    nan_pct = clinical_df["spo2_pct"].isna().mean() * 100
    if nan_pct > 50:
        raise SignalQualityAlert(
            f"SpO2 NaN rate too high ({nan_pct:.1f}%); check Red/IR channel coupling."
        )

    if "red" not in df.columns:
        raise SignalQualityAlert("Red channel missing; SpO2 and SASHB require Red+IR.")

    if np.isfinite(green_snr_db) and green_snr_db < -20:
        raise SignalQualityAlert(
            f"Green channel SNR critically low ({green_snr_db:.1f} dB); heart rate unstable."
        )


def run_self_validation(log_path: Path | None = None, out_plot: Path | None = None) -> dict:
    """
    Run full self-validation pipeline.
    Returns fidelity metrics dict.
    """
    from src.analysis.features import ClinicalBiometricSuite, compute_filters

    if log_path is None:
        candidates = sorted(RAW_DIR.glob("Oralable_*.txt")) + sorted(RAW_DIR.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No raw logs in {RAW_DIR}")
        log_path = candidates[0]

    df = load_raw_ble_log(log_path)

    # Apply filters (compute_filters from features adds ir_dc, green_bp, red_dc, red_ac, ir_ac)
    df = compute_filters(df)

    suite = ClinicalBiometricSuite()
    clinical_df = suite.process(df)

    # Rescue events within 10s of SpO2 desaturation
    clinical_df["rescue_near_desat"] = flag_rescue_near_desaturation(clinical_df)

    hr_series = compute_heart_rate_from_green(df)
    green_snr = compute_green_snr(df)
    audit_signal_quality(df, clinical_df, green_snr_db=green_snr)

    # Metrics
    hr_series = compute_heart_rate_from_green(df)
    total_sashb = float(clinical_df["cumulative_sashb"].iloc[-1]) if len(clinical_df) > 0 else 0.0
    rescue_count = int(clinical_df["is_airway_rescue"].sum())
    rescue_near_desat_count = int(clinical_df["rescue_near_desat"].sum())

    ir_dc_raw = df["ir_dc"].median()
    ir_dc_v = ir_dc_raw * ADC_TO_V_SCALE
    coupling_ok = IR_DC_TARGET_V_LOW <= ir_dc_v <= IR_DC_TARGET_V_HIGH
    if not coupling_ok:
        coupling_note = f"OUTSIDE target {IR_DC_TARGET_V_LOW}-{IR_DC_TARGET_V_HIGH}V"
    else:
        coupling_note = "OK"

    fidelity = {
        "total_sashb": total_sashb,
        "airway_rescue_count": rescue_count,
        "rescue_near_desat_count": rescue_near_desat_count,
        "ir_dc_median_raw": ir_dc_raw,
        "ir_dc_median_v": ir_dc_v,
        "sensor_coupling": coupling_note,
        "green_snr_db": green_snr,
    }

    # 3-Pane plot
    if out_plot is None:
        out_plot = PLOTS_DIR / "self_validation.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    elapsed_s = (df.index - df.index[0]).total_seconds().to_numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Oralable MAM Self-Validation", fontsize=14)

    # Pane 1: SpO2 with SASHB burden shaded
    spo2 = clinical_df["spo2_pct"].to_numpy()
    ax1.plot(elapsed_s, spo2, color="steelblue", linewidth=1.5, label="SpO2 (%)")
    burden_mask = np.isfinite(spo2) & (spo2 < SPO2_DESAT_THRESHOLD)
    ax1.fill_between(
        elapsed_s,
        60,
        90,
        where=burden_mask,
        alpha=0.4,
        color="red",
        label="SASHB Burden (<90%)",
    )
    ax1.axhline(90, color="gray", linestyle="--", alpha=0.7)
    ax1.set_ylabel("SpO2 (%)")
    ax1.set_ylim(55, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Pane 2: IR DC with Airway Rescue markers
    ir_dc = df["ir_dc"].to_numpy()
    ax2.plot(elapsed_s, ir_dc, color="purple", linewidth=1.5, label="IR DC Baseline")
    rescue_idx = np.where(clinical_df["is_airway_rescue"].to_numpy() == 1)[0]
    for i in rescue_idx:
        ax2.axvline(elapsed_s[i], color="red", alpha=0.6, linewidth=0.8)
    ax2.set_ylabel("IR DC (raw)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # Pane 3: Green PPG HR + Accel Z
    hr = hr_series.to_numpy()
    ax3_twin = ax3.twinx()
    ln1 = ax3.plot(elapsed_s, hr, color="green", linewidth=1, alpha=0.9, label="HR (BPM)")
    accel_z = df["accel_z"].to_numpy() if "accel_z" in df.columns else np.zeros(len(df))
    ln2 = ax3_twin.plot(elapsed_s, accel_z, color="orange", linewidth=0.8, alpha=0.7, label="Accel Z")
    ax3.set_ylabel("HR (BPM)", color="green")
    ax3_twin.set_ylabel("Accel Z (raw)", color="orange")
    ax3.grid(True, alpha=0.3)
    lns = ln1 + ln2
    ax3.legend(lns, [l.get_label() for l in lns], loc="upper right")

    ax3.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()

    return fidelity


def print_fidelity_report(fidelity: dict) -> None:
    """Print summary to console."""
    print("\n" + "=" * 60)
    print("FIDELITY REPORT - Oralable MAM Self-Validation")
    print("=" * 60)
    print(f"  Total SASHB Score (Cumulative Burden): {fidelity['total_sashb']:.2f} %·s")
    print(f"  Airway Rescue Event Count:              {fidelity['airway_rescue_count']}")
    print(f"  Rescue within 10s of SpO2 Desaturation: {fidelity['rescue_near_desat_count']}")
    print(f"  Sensor Coupling (IR-DC median):         {fidelity['ir_dc_median_v']:.3f} V ({fidelity['ir_dc_median_raw']:.0f} raw) - {fidelity['sensor_coupling']}")
    print(f"  Green SNR (Heart Rate stability):       {fidelity['green_snr_db']:.2f} dB")
    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Oralable MAM self-validation pipeline")
    parser.add_argument("log_path", nargs="?", type=Path, help="Raw BLE log (.txt or .csv)")
    parser.add_argument("-o", "--output", type=Path, help="Output plot path")
    args = parser.parse_args()

    try:
        fidelity = run_self_validation(log_path=args.log_path, out_plot=args.output)
        print_fidelity_report(fidelity)
        return 0
    except SignalQualityAlert as e:
        print(f"Signal Quality Alert: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
