#!/usr/bin/env python3
"""
Basic evaluation of an Oralable BLE log (nRF Connect or ble_logger format).
Reports: parse stats, PPG/accel quality, IR DC range, sync taps, duration.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parser.log_parser import parse_all, check_ppg_truncation


FS = 50.0


def _butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low, high = low_hz / nyq, high_hz / nyq
    return butter(order, [low, high], btype="band")


def _butter_lowpass(cutoff_hz: float, fs: float, order: int = 4):
    return butter(order, cutoff_hz / (0.5 * fs), btype="low")


def eval_log(log_path: str | Path) -> dict:
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    # Parse
    trunc = check_ppg_truncation(log_path)
    streams = parse_all(log_path, tdm_interleave=False)
    ppg = streams.get("ppg")
    accel = streams.get("accelerometer")
    thermo = streams.get("thermometer")
    battery = streams.get("battery")

    result = {
        "log_file": log_path.name,
        "ppg_truncation": trunc.get("truncated", False),
        "ppg_samples_per_packet": trunc.get("samples_per_packet", 0),
        "ppg_rows": len(ppg) if ppg is not None and not ppg.empty else 0,
        "accel_rows": len(accel) if accel is not None and not accel.empty else 0,
        "thermo_rows": len(thermo) if thermo is not None and not thermo.empty else 0,
        "battery_rows": len(battery) if battery is not None and not battery.empty else 0,
    }

    if ppg is None or ppg.empty:
        result["error"] = "No PPG data"
        return result

    # Duration
    t0 = ppg["timestamp_s"].iloc[0]
    t1 = ppg["timestamp_s"].iloc[-1]
    duration_s = t1 - t0
    result["duration_s"] = round(duration_s, 2)
    result["duration_min"] = round(duration_s / 60, 2)

    # PPG stats (raw ADC)
    for ch in ["green", "red", "ir"]:
        if ch in ppg.columns:
            col = ppg[ch].astype(float)
            result[f"{ch}_mean"] = int(col.mean())
            result[f"{ch}_std"] = int(col.std())
            result[f"{ch}_min"] = int(col.min())
            result[f"{ch}_max"] = int(col.max())

    # IR DC (low-pass <1 Hz) - hemodynamic occlusion indicator
    sig_ir = ppg["ir"].astype(float).to_numpy()
    b_lp, a_lp = _butter_lowpass(0.8, FS, order=4)
    ir_dc = filtfilt(b_lp, a_lp, np.nan_to_num(sig_ir, nan=0.0))
    result["ir_dc_mean"] = int(np.mean(ir_dc))
    result["ir_dc_min"] = int(np.min(ir_dc))
    result["ir_dc_max"] = int(np.max(ir_dc))
    result["ir_dc_range"] = int(np.max(ir_dc) - np.min(ir_dc))

    # Green SNR (bandpass 0.5-8 Hz vs high-freq noise)
    sig_g = ppg["green"].astype(float).to_numpy() - np.mean(ppg["green"])
    b_bp, a_bp = _butter_bandpass(0.5, 8.0, FS, order=4)
    b_hp, a_hp = butter(4, 12.0 / (0.5 * FS), btype="high")
    sig_band = filtfilt(b_bp, a_bp, sig_g)
    noise_band = filtfilt(b_hp, a_hp, sig_g)
    sig_var = np.var(sig_band)
    noise_var = max(np.var(noise_band), 1e-12)
    result["green_snr_db"] = round(10.0 * np.log10(max(sig_var, 1e-12) / noise_var), 2)

    # Heart rate estimate from Green peaks
    peaks, _ = find_peaks(
        sig_band,
        distance=int(0.4 * FS),
        prominence=np.nanstd(sig_band) * 0.3 if np.isfinite(np.nanstd(sig_band)) else 0.01,
    )
    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) / FS  # seconds
        rr_valid = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
        if len(rr_valid) > 0:
            hr_bpm = 60.0 / np.median(rr_valid)
            result["heart_rate_bpm"] = round(hr_bpm, 1)
            result["peaks_detected"] = len(peaks)

    # Sync taps: 3 consecutive high-G spikes in accelerometer (per .cursorrules)
    if accel is not None and not accel.empty and "accel_x" in accel.columns:
        g_mag = np.sqrt(
            accel["accel_x"].astype(float) ** 2
            + accel["accel_y"].astype(float) ** 2
            + accel["accel_z"].astype(float) ** 2
        )
        # High-G threshold: e.g. > 2 std above mean
        thresh = np.mean(g_mag) + 2 * np.std(g_mag)
        high = (g_mag > thresh).astype(int)
        # Consecutive 3
        sync_count = 0
        i = 0
        while i < len(high) - 2:
            if high[i] and high[i + 1] and high[i + 2]:
                sync_count += 1
                i += 3
            else:
                i += 1
        result["sync_taps_3consecutive"] = sync_count
        result["accel_g_mean"] = round(float(np.mean(g_mag)), 0)
        result["accel_g_std"] = round(float(np.std(g_mag)), 0)

    # Thermometer sanity (temp_raw is centidegrees; 29450 = 294.5°C is likely raw ADC)
    if thermo is not None and not thermo.empty and "temp_c" in thermo.columns:
        tc = thermo["temp_c"]
        result["temp_c_min"] = round(float(tc.min()), 1)
        result["temp_c_max"] = round(float(tc.max()), 1)
        result["temp_c_mean"] = round(float(tc.mean()), 1)

    return result


def main():
    log_path = ROOT / "data" / "raw" / "Oralable 3.txt"
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])

    r = eval_log(log_path)
    print("=" * 60)
    print(f"Oralable Log Evaluation: {r.get('log_file', '?')}")
    print("=" * 60)
    print(f"PPG: {r.get('ppg_rows', 0)} samples | {r.get('duration_min', 0):.1f} min")
    print(f"  Samples/packet: {r.get('ppg_samples_per_packet', 0)} | Truncated: {r.get('ppg_truncation', False)}")
    if "green_mean" in r:
        print(f"  Green: mean={r['green_mean']:,} std={r['green_std']:,}")
        print(f"  Red:   mean={r['red_mean']:,} std={r['red_std']:,}")
        print(f"  IR:    mean={r['ir_mean']:,} std={r['ir_std']:,}")
    print(f"  IR DC (low-pass): mean={r.get('ir_dc_mean', 0):,} range={r.get('ir_dc_range', 0):,}")
    print(f"  Green SNR: {r.get('green_snr_db', 0):.1f} dB")
    if "heart_rate_bpm" in r:
        print(f"  Heart rate (est): {r['heart_rate_bpm']} BPM ({r.get('peaks_detected', 0)} peaks)")
    print(f"Accelerometer: {r.get('accel_rows', 0)} samples")
    if "sync_taps_3consecutive" in r:
        print(f"  Sync taps (3-consec high-G): {r['sync_taps_3consecutive']}")
    if "accel_g_mean" in r:
        print(f"  G magnitude: mean={r['accel_g_mean']:.0f} std={r['accel_g_std']:.0f}")
    print(f"Thermometer: {r.get('thermo_rows', 0)} samples")
    if "temp_c_mean" in r:
        print(f"  Temp: {r['temp_c_min']:.1f}–{r['temp_c_max']:.1f}°C (mean {r['temp_c_mean']:.1f})")
    print(f"Battery: {r.get('battery_rows', 0)} updates")
    if "error" in r:
        print(f"ERROR: {r['error']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
