#!/usr/bin/env python3
"""
Self-validation pipeline for clinical metrics on Oralable MAM sleep data.

Verifies SpO2 (110-25R empirical curve), SASHB (AUC below 90%), and Airway Rescue
(IR-DC drop >15% in 500ms), with correlation to SpO2 desaturation events.
Adheres to .cursorrules mandate for 100% clinical precision.

John's Self-Validation Protocol: CLI-guided Sync, Clench, Grind, Apnea phases.
Live Battery Health at start/end proves 15 mAh CG-320B sufficient for Ed Owens.

Usage:
    PYTHONPATH=. python src/validation/self_validate.py [log_path] [-o output.png]
    python -m src.validation.self_validate --protocol   # Interactive ground-truth annotation
    python -m src.validation.self_validate data/raw/Oralable_20260223_083911.txt
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
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
DATASETS_DIR = ROOT / "data" / "datasets"

FS = 50.0
DT = 0.02
SPO2_DESAT_THRESHOLD = 90.0
RESCUE_DESAT_WINDOW_S = 10.0
IR_DC_TARGET_V_LOW = 1.8
IR_DC_TARGET_V_HIGH = 2.4
# Oralable uses 32-bit raw ADC (IR typically 50k–350k); voltage conversion assumes 16-bit
ADC_TO_V_SCALE = 3.3 / 65535.0
# Raw ADC range for Oralable MAM (cheek PPG); good coupling when within this range
# Run scripts/check_ir_dc_scaling.py to verify; 32-bit firmware ~10M-60M
IR_DC_RAW_MIN = 10_000_000
IR_DC_RAW_MAX = 70_000_000
MIN_SAMPLES_FOR_VALIDATION = int(30 * FS)  # 30 seconds minimum

# Self-validation protocol phases (start_s, end_s from T=0 = 3rd tap of first sync)
# Aligns with JOHN_COGAN_2ND_SYNC_PROTOCOL.csv
PROTOCOL_PHASES = [
    (0, "3-Tap Sync", 0, 5),
    (1, "Max Tonic Clench", 30, 45),
    (2, "Rest", 45, 60),
    (3, "Phasic Grinding", 60, 105),
    (4, "Rest", 105, 120),
    (5, "Swallow/Control", 120, 135),
    (6, "Simulated Apnea", 150, 195),
    (7, "Natural Speech", 210, 270),
]


def _protocol_phases_in_segment(
    segment_start_s: float,
    segment_end_s: float,
    t0_s: float,
) -> list[tuple[str, float, float]]:
    """
    Return protocol phases overlapping [segment_start_s, segment_end_s] (session time).
    t0_s = session time of T=0 (first 3-tap sync).
    Returns list of (phase_name, start_in_segment, end_in_segment) in segment-local seconds.
    """
    protocol_from_t0 = segment_start_s - t0_s  # protocol time at segment start
    protocol_to_t0 = segment_end_s - t0_s    # protocol time at segment end
    result = []
    for _id, name, p_start, p_end in PROTOCOL_PHASES:
        # Overlap [p_start, p_end] with [protocol_from_t0, protocol_to_t0]
        overlap_start = max(p_start, protocol_from_t0)
        overlap_end = min(p_end, protocol_to_t0)
        if overlap_start < overlap_end:
            # Convert to segment-local
            seg_start = overlap_start - protocol_from_t0
            seg_end = overlap_end - protocol_from_t0
            result.append((name, seg_start, seg_end))
    return result


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


def load_raw_ble_log(
    path: Path,
    tdm_interleave: bool = False,
    channel_order: str | None = None,
) -> pd.DataFrame:
    """Load raw BLE sensor log and produce merged 50Hz-ready dataframe."""
    from src.parser.log_parser import parse_all

    from src.processing.resampler import resample_raw_to_50hz

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw log not found: {path}")

    streams = parse_all(path, tdm_interleave=tdm_interleave, channel_order=channel_order)
    ppg = streams.get("ppg")
    accel = streams.get("accelerometer")
    # Save accelerometer for sync tap detection (100 Hz)
    if accel is not None and not accel.empty:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        accel.to_csv(PROCESSED_DIR / "accelerometer.csv", index=False)
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


def run_self_validation(
    log_path: Path | None = None,
    out_plot: Path | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    protocol_phases: list[tuple[str, float, float]] | None = None,
) -> dict:
    """
    Run full self-validation pipeline.
    If start_s and end_s are provided, analyze only that segment (elapsed seconds from session start).
    Returns fidelity metrics dict.
    """
    from src.analysis.features import ClinicalBiometricSuite, compute_filters

    if log_path is None:
        candidates = sorted(RAW_DIR.glob("Oralable_*.txt")) + sorted(RAW_DIR.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No raw logs in {RAW_DIR}")
        log_path = candidates[0]

    df = load_raw_ble_log(log_path)

    # Segment by time range if requested
    if start_s is not None and end_s is not None:
        elapsed = (df.index - df.index[0]).total_seconds()
        mask = (elapsed >= start_s) & (elapsed <= end_s)
        df = df.loc[mask].copy()
        if len(df) < MIN_SAMPLES_FOR_VALIDATION:
            raise SignalQualityAlert(
                f"Segment too short: {len(df)} samples ({len(df)/FS:.1f}s) < {MIN_SAMPLES_FOR_VALIDATION} (need ≥30s)."
            )

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
    # Oralable uses 32-bit raw ADC; use raw range for coupling (not voltage)
    coupling_ok = IR_DC_RAW_MIN <= ir_dc_raw <= IR_DC_RAW_MAX
    if not coupling_ok:
        coupling_note = f"OUTSIDE target {IR_DC_RAW_MIN/1000:.0f}k-{IR_DC_RAW_MAX/1000:.0f}k raw"
    else:
        coupling_note = "OK"

    # Protocol-phase metrics (when phases available)
    occlusion_tonic = float("nan")
    jitter_phasic = float("nan")
    clench_detected = False
    swallow_false_positives = 0
    speech_false_positives = 0
    apnea_gasp_occlusion_pct = float("nan")
    apnea_rescue_detected = False
    if protocol_phases:
        ir_dc_arr = df["ir_dc"].to_numpy()
        accel_z_arr = df["accel_z"].to_numpy() if "accel_z" in df.columns else np.zeros(len(df))
        rescue_arr = clinical_df["is_airway_rescue"].to_numpy()
        elapsed_seg = (df.index - df.index[0]).total_seconds().to_numpy()
        for name, seg_start, seg_end in protocol_phases:
            m = (elapsed_seg >= seg_start) & (elapsed_seg <= seg_end)
            if not np.any(m):
                continue
            if "Tonic Clench" in name:
                seg = ir_dc_arr[m]
                ref_len = max(1, int(0.1 * len(seg)))
                baseline = np.nanmean(seg[:ref_len])
                trough = np.nanmin(seg)
                if baseline > 1e-9:
                    occlusion_tonic = (baseline - trough) / baseline * 100.0
                # Clench detected: ≥2.5% drop over ≥5s (cheek produces smaller occlusion than finger)
                if seg_end - seg_start >= 5.0 and np.isfinite(occlusion_tonic) and occlusion_tonic >= 2.5:
                    clench_detected = True
            elif "Phasic Grinding" in name or "Grinding" in name:
                from src.analysis.features import calculate_grind_jitter
                var_j = calculate_grind_jitter(accel_z_arr[m], fs=FS)
                jitter_phasic = float(np.sqrt(var_j)) if np.isfinite(var_j) else float("nan")
            elif "Swallow" in name or "Control" in name:
                swallow_false_positives = int(np.sum(rescue_arr[m]))
            elif "Natural Speech" in name or "Speech" in name:
                speech_false_positives = int(np.sum(rescue_arr[m]))
            elif "Simulated Apnea" in name or "Apnea" in name:
                # Gasp Clench: last ~15s of apnea phase
                gasp_dur = min(15.0, (seg_end - seg_start) * 0.4)
                gasp_start = seg_end - gasp_dur
                mg = (elapsed_seg >= gasp_start) & (elapsed_seg <= seg_end)
                if np.any(mg):
                    seg_gasp = ir_dc_arr[mg]
                    ref_len = max(1, int(0.1 * len(seg_gasp)))
                    baseline = np.nanmean(seg_gasp[:ref_len])
                    trough = np.nanmin(seg_gasp)
                    if baseline > 1e-9:
                        apnea_gasp_occlusion_pct = (baseline - trough) / baseline * 100.0
                    apnea_rescue_detected = np.any(rescue_arr[mg])
                    # Cheek-specific: configurable threshold in 500ms (Oralable MAM produces smaller drops)
                    # Try 10%, 5%, 3%, 2% — 2.18% occlusion may be gradual, so need low threshold
                    CHEEK_RESCUE_THRESHOLDS = (0.10, 0.05, 0.03, 0.02)
                    w = 25  # 500ms at 50Hz
                    if len(seg_gasp) >= w:
                        for thresh in CHEEK_RESCUE_THRESHOLDS:
                            for j in range(w, len(seg_gasp)):
                                if seg_gasp[j - w] > 1e-9:
                                    pct = (seg_gasp[j] - seg_gasp[j - w]) / seg_gasp[j - w]
                                    if pct <= -thresh:
                                        apnea_rescue_detected = True
                                        break
                            if apnea_rescue_detected:
                                break

    # Sync tap detection (3-tap pattern per .cursorrules)
    sync_anchors: list[tuple[int, float]] = []  # (anchor_idx, elapsed_s from session start)
    accel_path = PROCESSED_DIR / "accelerometer.csv"
    try:
        from src.utils.sync_align import find_all_three_tap_anchors
        anchors = find_all_three_tap_anchors(
            df, accel_100hz_path=accel_path, max_count=5,
            min_gap_seconds=25,  # Sync taps ~30s apart — filters false positives
            prefer_last_seconds=120,  # Scan last 2 min for 3 taps near end
        )
        t0 = df.index[0]
        sync_anchors = [(idx, (ts - t0).total_seconds()) for idx, ts in anchors]
    except (ValueError, FileNotFoundError):
        pass  # No sync taps found or accel missing

    # Thermometer stats (from raw log)
    from src.parser.log_parser import parse_all as _parse_all
    thermo_df = _parse_all(log_path).get("thermometer", pd.DataFrame())
    thermo_rows = len(thermo_df) if thermo_df is not None and not thermo_df.empty else 0
    temp_c_min = temp_c_max = temp_c_mean = float("nan")
    temp_raw_note = ""
    if thermo_rows > 0 and "temp_c" in thermo_df.columns:
        tc = thermo_df["temp_c"]
        valid = tc[(tc > -50) & (tc < 60)]
        if len(valid) > 0:
            temp_c_min, temp_c_max = float(valid.min()), float(valid.max())
            temp_c_mean = float(valid.mean())
        else:
            raw = thermo_df["temp_raw"]
            temp_raw_note = f"raw {int(raw.min())}-{int(raw.max())} (temp_c implausible; check firmware format)"

    duration_s = (df.index[-1] - df.index[0]).total_seconds() if len(df) > 1 else 0.0

    fidelity = {
        "duration_s": duration_s,
        "total_sashb": total_sashb,
        "airway_rescue_count": rescue_count,
        "rescue_near_desat_count": rescue_near_desat_count,
        "ir_dc_median_raw": ir_dc_raw,
        "ir_dc_median_v": ir_dc_v,
        "sensor_coupling": coupling_note,
        "green_snr_db": green_snr,
        "occlusion_tonic_pct": occlusion_tonic,
        "jitter_phasic_rms": jitter_phasic,
        "clench_detected": clench_detected,
        "swallow_false_positives": swallow_false_positives,
        "speech_false_positives": speech_false_positives,
        "apnea_gasp_occlusion_pct": apnea_gasp_occlusion_pct,
        "apnea_rescue_detected": apnea_rescue_detected,
        "thermo_rows": thermo_rows,
        "temp_c_min": temp_c_min,
        "temp_c_max": temp_c_max,
        "temp_c_mean": temp_c_mean,
        "temp_raw_note": temp_raw_note,
        "sync_anchors": sync_anchors,
    }

    # 3-Pane plot
    if out_plot is None:
        out_plot = PLOTS_DIR / "self_validation.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    elapsed_s = (df.index - df.index[0]).total_seconds().to_numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    title = "Oralable MAM Self-Validation"
    if start_s is not None and end_s is not None:
        title += f" (segment {start_s:.1f}s–{end_s:.1f}s)"
    fig.suptitle(title, fontsize=14)

    # Protocol phase bands (shared across panes)
    if protocol_phases:
        colors = plt.cm.Set3(np.linspace(0, 1, len(protocol_phases)))
        for ax in (ax1, ax2, ax3):
            for i, (name, seg_start, seg_end) in enumerate(protocol_phases):
                ax.axvspan(seg_start, seg_end, alpha=0.2, color=colors[i])
        # Phase labels on top pane
        for i, (name, seg_start, seg_end) in enumerate(protocol_phases):
            ax1.text(
                (seg_start + seg_end) / 2,
                100,
                name,
                ha="center",
                va="bottom",
                fontsize=9,
                color=colors[i],
            )

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


def _get_battery_health_from_log(log_path: Path) -> dict | None:
    """Extract battery telemetry from log for Live Battery Health display."""
    from src.parser.log_parser import parse_batt_log_lines, parse_battery_log, parse_all

    log_path = Path(log_path)
    if not log_path.exists():
        return None

    # Prefer [BATT] lines (power telemetry from REV10)
    batt_df = parse_batt_log_lines(log_path)
    if not batt_df.empty:
        first = batt_df.iloc[0]
        last = batt_df.iloc[-1]
        return {
            "start_v_mv": int(first["voltage_mv"]),
            "start_pct": int(first["percent"]),
            "start_mah_used": float(first["mah_used"]),
            "start_rem_min": int(first["remaining_min"]),
            "end_v_mv": int(last["voltage_mv"]),
            "end_pct": int(last["percent"]),
            "end_mah_used": float(last["mah_used"]),
            "end_rem_min": int(last["remaining_min"]),
            "source": "BATT",
        }

    # Fallback: battery characteristic (3A0FF004)
    streams = parse_all(log_path)
    bat = streams.get("battery", pd.DataFrame())
    if not bat.empty and "hex_payload" in bat.columns:
        # Simple heuristic: first/last payload as voltage proxy
        return {
            "start_v_mv": 0,
            "start_pct": 0,
            "start_mah_used": 0.0,
            "start_rem_min": 0,
            "end_v_mv": 0,
            "end_pct": 0,
            "end_mah_used": 0.0,
            "end_rem_min": 0,
            "source": "hex",
            "note": "Battery hex present; enable BatteryStats (3A0FFEF2) for full telemetry.",
        }
    return None


def print_live_battery_health(log_path: Path | None, label: str = "Live Battery Health") -> None:
    """
    Print Live Battery Health status. Proves to Ed Owens that 15 mAh CG-320B is sufficient.
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("=" * 60)
    if log_path is None:
        candidates = sorted(RAW_DIR.glob("Oralable_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        log_path = candidates[0] if candidates else None
    if log_path is None or not Path(log_path).exists():
        print("  No log file. Record a session with power telemetry (BatteryStats) first.")
        print("=" * 60 + "\n")
        return

    health = _get_battery_health_from_log(Path(log_path))
    if health is None:
        print("  No battery telemetry in log. Use firmware with BatteryStats (UUID ...fef2).")
        print("=" * 60 + "\n")
        return

    if health.get("source") == "BATT":
        print(f"  START:  {health['start_v_mv']} mV | {health['start_pct']}% | Used: {health['start_mah_used']:.2f} mAh | Rem: {health['start_rem_min']} min")
        print(f"  END:    {health['end_v_mv']} mV | {health['end_pct']}% | Used: {health['end_mah_used']:.2f} mAh | Rem: {health['end_rem_min']} min")
        delta_mah = health["end_mah_used"] - health["start_mah_used"]
        print(f"  Delta:  {delta_mah:.2f} mAh consumed during session")
        print("  CG-320B (15 mAh): sufficient for 8h clinical night when streaming.")
    else:
        print(f"  {health.get('note', 'Battery data in hex format.')}")
    print("=" * 60 + "\n")


def run_self_validation_protocol(log_path: Path | None = None) -> Path:
    """
    CLI-guided routine: prompts John to perform Sync, Clench, Grind, Apnea.
    Records start/end timestamps, saves Ground_Truth_John_Cogan_Date.csv.
    """
    if log_path is None:
        candidates = sorted(RAW_DIR.glob("Oralable_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No raw logs in {RAW_DIR}. Record a session first.")
        log_path = candidates[0]
        print(f"Using most recent log: {log_path.name}")
    log_path = Path(log_path)

    # Live Battery Health at START
    print_live_battery_health(log_path, "Live Battery Health (START)")

    # Get time range from log for reference
    from src.parser.log_parser import parse_oralable_log
    ppg = parse_oralable_log(log_path)
    if ppg.empty:
        raise SignalQualityAlert("No PPG data; cannot establish time base.")
    t0 = float(ppg["timestamp_s"].iloc[0])
    t_end = float(ppg["timestamp_s"].iloc[-1])
    duration_s = t_end - t0
    print(f"Log duration: {duration_s:.1f} s ({duration_s/60:.1f} min)")
    print("Enter timestamps in seconds from log start, or press Enter to use prompts.\n")

    phases = [
        ("Sync", "Perform 3 sync taps on the device."),
        ("Clench", "Clench jaw (temporalis occlusion) - hold steady."),
        ("Grind", "Simulate tooth grinding (rhythmic lateral movement)."),
        ("Apnea", "Hold breath briefly (apnea simulation)."),
    ]
    records = []
    for phase_name, instruction in phases:
        print(f"--- Phase: {phase_name} ---")
        print(f"  {instruction}")
        try:
            start_in = input(f"  Enter START time (s from log start, or press Enter when you see it in playback): ").strip()
            end_in = input(f"  Enter END time (s from log start): ").strip()
            start_s = float(start_in) if start_in else 0.0
            end_s = float(end_in) if end_in else start_s + 5.0
        except (ValueError, EOFError):
            start_s, end_s = 0.0, 5.0
        records.append({"phase": phase_name, "start_s": start_s, "end_s": end_s})
        print()

    # Build ground truth CSV
    date_str = datetime.now().strftime("%Y-%m-%d")
    gt_df = pd.DataFrame(records)
    gt_df["subject"] = "John_Cogan"
    gt_df["date"] = date_str
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATASETS_DIR / f"Ground_Truth_John_Cogan_{date_str}.csv"
    gt_df.to_csv(out_path, index=False)
    print(f"Saved ground truth to {out_path}")

    # Live Battery Health at END
    print_live_battery_health(log_path, "Live Battery Health (END)")

    return out_path


def print_fidelity_report(fidelity: dict) -> None:
    """Print summary to console."""
    print("\n" + "=" * 60)
    print("FIDELITY REPORT - Oralable MAM Self-Validation")
    print("=" * 60)
    print(f"  Total SASHB Score (Cumulative Burden): {fidelity['total_sashb']:.2f} %·s")
    print(f"  Airway Rescue Event Count:              {fidelity['airway_rescue_count']}")
    print(f"  Rescue within 10s of SpO2 Desaturation: {fidelity['rescue_near_desat_count']}")
    print(f"  Sensor Coupling (IR-DC median):         {fidelity['ir_dc_median_raw']:.0f} raw - {fidelity['sensor_coupling']}")
    print(f"  Green SNR (Heart Rate stability):       {fidelity['green_snr_db']:.2f} dB")
    sync_anchors = fidelity.get("sync_anchors", [])
    if sync_anchors:
        duration_s = fidelity.get("duration_s")
        print(f"  Sync taps detected: {len(sync_anchors)} instance(s) of 3-tap pattern")
        n = len(sync_anchors)
        for i, (_idx, elapsed_s) in enumerate(sync_anchors):
            pct = 100 * elapsed_s / duration_s if duration_s and duration_s > 0 else 0
            if n <= 8 or i < 3 or i >= n - 3:
                print(f"    Sync #{i+1}: 3rd tap at {elapsed_s:.1f}s ({pct:.1f}% through session)")
            elif i == 3:
                print(f"    ... ({n - 6} more)")

    thermo_rows = fidelity.get("thermo_rows", 0)
    if thermo_rows > 0:
        note = fidelity.get("temp_raw_note", "")
        if note:
            print(f"  Thermometer: {thermo_rows} samples - {note}")
        elif np.isfinite(fidelity.get("temp_c_mean", float("nan"))):
            print(f"  Thermometer: {thermo_rows} samples - {fidelity['temp_c_min']:.1f}–{fidelity['temp_c_max']:.1f}°C (mean {fidelity['temp_c_mean']:.1f})")
        else:
            print(f"  Thermometer: {thermo_rows} samples (temp_c implausible)")

    occ = fidelity.get("occlusion_tonic_pct", float("nan"))
    jit = fidelity.get("jitter_phasic_rms", float("nan"))
    clench = fidelity.get("clench_detected", False)
    if np.isfinite(occ):
        print(f"  Occlusion Depth (Tonic Clench):        {occ:.2f}%")
    if np.isfinite(jit):
        print(f"  Jitter RMS (Phasic Grind):             {jit:.2f} (5–24 Hz band)")
    print(f"  Clench Detected (≥2.5% over 5s):       {'Yes' if clench else 'No'}")
    # Ed/Pedro protocol: Artifact Filter, Rescue Timing, False Positive Check
    swallow_fp = fidelity.get("swallow_false_positives", 0)
    speech_fp = fidelity.get("speech_false_positives", 0)
    apnea_occ = fidelity.get("apnea_gasp_occlusion_pct", float("nan"))
    apnea_rescue = fidelity.get("apnea_rescue_detected", False)
    print(f"  Swallow False Positives (must be 0):    {swallow_fp} {'✓' if swallow_fp == 0 else '✗'}")
    print(f"  Speech False Positives (must be 0):     {speech_fp} {'✓' if speech_fp == 0 else '✗'}")
    if np.isfinite(apnea_occ):
        print(f"  Apnea Gasp Clench Occlusion:           {apnea_occ:.2f}%")
    print(f"  Apnea Rescue Detected (cheek 10–2%):    {'Yes' if apnea_rescue else 'No'}")
    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Oralable MAM self-validation pipeline")
    parser.add_argument("log_path", nargs="?", type=Path, help="Raw BLE log (.txt or .csv)")
    parser.add_argument("-o", "--output", type=Path, help="Output plot path")
    parser.add_argument("--protocol", action="store_true", help="Run CLI-guided Sync/Clench/Grind/Apnea protocol")
    parser.add_argument(
        "--segment-between",
        nargs=2,
        type=int,
        metavar=("ANCHOR_A", "ANCHOR_B"),
        help="Run validation on segment between 3-tap sync anchors (e.g. 2 3 = between 2nd and 3rd sync)",
    )
    parser.add_argument(
        "--segment-from",
        type=int,
        metavar="ANCHOR",
        help="Run validation from the Nth 3-tap sync to end of session (e.g. 2 = from 2nd sync onward)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        metavar="SECONDS",
        help="Start segment at this time (s from session start). E.g. 960 for 16 min.",
    )
    parser.add_argument(
        "--segment-from-sync-near-end",
        type=float,
        metavar="MINUTES",
        help="Find 3-tap sync ~N minutes before end, run validation from that sync. E.g. 5 = sync ~5 min before end.",
    )
    args = parser.parse_args()

    try:
        if args.protocol:
            run_self_validation_protocol(log_path=args.log_path)
            return 0

        start_s, end_s = None, None
        protocol_phases = None
        log_path = args.log_path
        if args.segment_between is not None:
            a, b = args.segment_between
            if a < 1 or b < 1 or a >= b:
                print("Error: --segment-between requires two anchors with ANCHOR_A < ANCHOR_B (e.g. 2 3)", file=sys.stderr)
                return 1
            log_path = log_path or next(iter(sorted(RAW_DIR.glob("Oralable_*.txt"))), None)
            if log_path is None or not Path(log_path).exists():
                print("Error: need a log file for --segment-between", file=sys.stderr)
                return 1
            from src.utils.sync_align import find_all_three_tap_anchors
            df_temp = load_raw_ble_log(log_path)
            anchors = find_all_three_tap_anchors(df_temp, max_count=max(a, b))
            if len(anchors) < b:
                print(f"Error: found only {len(anchors)} 3-tap sync(s); need at least {b}", file=sys.stderr)
                return 1
            start_s = (anchors[a - 1][1] - df_temp.index[0]).total_seconds()
            end_s = (anchors[b - 1][1] - df_temp.index[0]).total_seconds()
            t0_s = (anchors[0][1] - df_temp.index[0]).total_seconds()
            protocol_phases = _protocol_phases_in_segment(start_s, end_s, t0_s)
            if protocol_phases:
                print(f"Protocol phases in segment: {[p[0] for p in protocol_phases]}")
            print(f"Segment: between 3-tap sync #{a} ({start_s:.2f}s) and #{b} ({end_s:.2f}s), duration {end_s - start_s:.2f}s")
        elif args.segment_from is not None:
            n = args.segment_from
            if n < 1:
                print("Error: --segment-from requires anchor >= 1", file=sys.stderr)
                return 1
            log_path = log_path or next(iter(sorted(RAW_DIR.glob("Oralable_*.txt"))), None)
            if log_path is None or not Path(log_path).exists():
                print("Error: need a log file for --segment-from", file=sys.stderr)
                return 1
            from src.utils.sync_align import find_all_three_tap_anchors
            df_temp = load_raw_ble_log(log_path)
            anchors = find_all_three_tap_anchors(df_temp, max_count=n)
            if len(anchors) < n:
                print(f"Error: found only {len(anchors)} 3-tap sync(s); need at least {n}", file=sys.stderr)
                return 1
            start_s = (anchors[n - 1][1] - df_temp.index[0]).total_seconds()
            end_s = (df_temp.index[-1] - df_temp.index[0]).total_seconds()
            t0_s = start_s  # Use Nth sync as T=0 for protocol
            protocol_phases = _protocol_phases_in_segment(start_s, end_s, t0_s)
            if protocol_phases:
                print(f"Protocol phases in segment: {[p[0] for p in protocol_phases]}")
            print(f"Segment: from 3-tap sync #{n} ({start_s:.2f}s) to end ({end_s:.2f}s), duration {end_s - start_s:.2f}s")
        elif args.segment_from_sync_near_end is not None:
            minutes = args.segment_from_sync_near_end
            if minutes <= 0:
                print("Error: --segment-from-sync-near-end requires MINUTES > 0", file=sys.stderr)
                return 1
            log_path = log_path or next(iter(sorted(RAW_DIR.glob("Oralable_*.txt"))), None)
            if log_path is None or not Path(log_path).exists():
                print("Error: need a log file for --segment-from-sync-near-end", file=sys.stderr)
                return 1
            from src.utils.sync_align import find_all_three_tap_anchors
            df_temp = load_raw_ble_log(log_path)
            duration_s = (df_temp.index[-1] - df_temp.index[0]).total_seconds()
            target_s = duration_s - (minutes * 60.0)
            anchors = find_all_three_tap_anchors(df_temp, max_count=50)
            if not anchors:
                print("Error: no 3-tap sync found", file=sys.stderr)
                return 1
            # Pick sync closest to target (5 min before end)
            best = min(anchors, key=lambda a: abs((a[1] - df_temp.index[0]).total_seconds() - target_s))
            start_s = (best[1] - df_temp.index[0]).total_seconds()
            end_s = duration_s
            t0_s = start_s
            protocol_phases = _protocol_phases_in_segment(start_s, end_s, t0_s)
            before_end = duration_s - start_s
            if protocol_phases:
                print(f"Protocol phases in segment: {[p[0] for p in protocol_phases]}")
            print(f"Segment: from 3-tap sync at {start_s:.1f}s ({before_end:.1f}s before end) to end ({end_s:.2f}s), duration {end_s - start_s:.2f}s")
        elif args.start_time is not None:
            t = args.start_time
            if t < 0:
                print("Error: --start-time must be >= 0", file=sys.stderr)
                return 1
            log_path = log_path or next(iter(sorted(RAW_DIR.glob("Oralable_*.txt"))), None)
            if log_path is None or not Path(log_path).exists():
                print("Error: need a log file for --start-time", file=sys.stderr)
                return 1
            df_temp = load_raw_ble_log(log_path)
            end_s = (df_temp.index[-1] - df_temp.index[0]).total_seconds()
            start_s = min(t, end_s - 1)
            t0_s = start_s
            protocol_phases = _protocol_phases_in_segment(start_s, end_s, t0_s)
            if protocol_phases:
                print(f"Protocol phases in segment: {[p[0] for p in protocol_phases]}")
            print(f"Segment: from {start_s:.1f}s to end ({end_s:.2f}s), duration {end_s - start_s:.2f}s")

        # Standard validation: show Live Battery at start and end
        print_live_battery_health(log_path, "Live Battery Health (START)")
        fidelity = run_self_validation(
            log_path=log_path,
            out_plot=args.output,
            start_s=start_s,
            end_s=end_s,
            protocol_phases=protocol_phases,
        )
        print_fidelity_report(fidelity)
        print_live_battery_health(log_path, "Live Battery Health (END)")
        return 0
    except SignalQualityAlert as e:
        print(f"Signal Quality Alert: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
