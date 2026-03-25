#!/usr/bin/env python3
"""
Researcher summary: load GOLD_STANDARD_VALIDATION.csv and print / save clinical metrics
(TFI, SASHB, event counts, SpO2 nadir vs temporalis clench timing) for IEEE / patent reporting.

Usage:
  python scripts/generate_clinical_report.py
  python scripts/generate_clinical_report.py --input data/validation/GOLD_STANDARD_VALIDATION.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.features import FS, calculate_tfi


def _accel_magnitude(df: pd.DataFrame) -> np.ndarray:
    ax = df["temporalis_accel_x"].astype(float).to_numpy()
    ay = df["temporalis_accel_y"].astype(float).to_numpy()
    az = df["temporalis_accel_z"].astype(float).to_numpy()
    return np.sqrt(ax * ax + ay * ay + az * az)


def count_phasic_events(df: pd.DataFrame, fs: float = FS) -> int:
    """Rhythmic grinding: count accelerometer magnitude peaks during phasic_grinding."""
    sub = df[df["label_name"] == "phasic_grinding"]
    if sub.empty or len(sub) < 5:
        return 0
    mag = _accel_magnitude(sub)
    prom = max(float(np.nanstd(mag)) * 0.25, 1e-6)
    dist = max(1, int(0.2 * fs))
    peaks, _ = find_peaks(mag, distance=dist, prominence=prom)
    return int(len(peaks))


def count_tonic_events(df: pd.DataFrame) -> int:
    """One protocol block for max tonic (02:00–02:10): present = one event."""
    sub = df[df["label_name"] == "tonic_max"]
    return 1 if not sub.empty else 0


def count_rescue_events(df: pd.DataFrame) -> int:
    """One protocol block for tonic rescue (04:20–04:30): present = one event."""
    sub = df[df["label_name"] == "tonic_rescue"]
    return 1 if not sub.empty else 0


def total_hypoxic_burden_sashb(df: pd.DataFrame) -> float:
    """Cumulative SASHB at end of recording (area-like burden, SpO2 < 90% threshold)."""
    if "cumulative_sashb" not in df.columns or df.empty:
        return float("nan")
    return float(df["cumulative_sashb"].iloc[-1])


def _spo2_nadir_elapsed_times(df: pd.DataFrame, fs: float = FS) -> np.ndarray:
    s = df["spo2_pct"].to_numpy(dtype=float)
    e = df["elapsed_s"].to_numpy(dtype=float)
    if s.size < int(4 * fs):
        return np.array([], dtype=float)
    s_clean = np.where(np.isfinite(s), s, np.nan)
    if np.all(~np.isfinite(s_clean)):
        return np.array([], dtype=float)
    fill = float(np.nanpercentile(s_clean[np.isfinite(s_clean)], 50)) if np.any(np.isfinite(s_clean)) else 95.0
    s_clean = np.where(np.isfinite(s_clean), s_clean, fill)
    # light smooth
    k = max(3, int(1.5 * fs))
    if k % 2 == 0:
        k += 1
    pad = np.pad(s_clean, (k // 2, k // 2), mode="edge")
    kernel = np.ones(k) / k
    sm = np.convolve(pad, kernel, mode="valid")
    dist = max(1, int(2.5 * fs))
    prom = max(0.15, float(np.nanstd(sm)) * 0.2)
    troughs, _ = find_peaks(-sm, distance=dist, prominence=prom)
    return e[troughs]


def _temporalis_clench_elapsed_times(df: pd.DataFrame) -> np.ndarray:
    """
    Representative clench times: IR-DC minimum within each temporalis motor block
    (tonic max, phasic grinding, tonic rescue).
    """
    times: list[float] = []
    for label in ("tonic_max", "phasic_grinding", "tonic_rescue"):
        sub = df[df["label_name"] == label]
        if sub.empty:
            continue
        ir = sub["ir_dc"].astype(float).to_numpy()
        el = sub["elapsed_s"].astype(float).to_numpy()
        i_min = int(np.nanargmin(ir))
        times.append(float(el[i_min]))
    return np.asarray(times, dtype=float)


def timing_delta_spo2_nadirs_vs_clenches(
    df: pd.DataFrame,
    fs: float = FS,
) -> dict[str, float | np.ndarray]:
    nadirs = _spo2_nadir_elapsed_times(df, fs=fs)
    clenches = _temporalis_clench_elapsed_times(df)
    if nadirs.size == 0 or clenches.size == 0:
        return {
            "n_nadirs": float(len(nadirs)),
            "n_clenches": float(len(clenches)),
            "mean_abs_delta_s": float("nan"),
            "mean_signed_delta_s": float("nan"),
            "std_abs_delta_s": float("nan"),
            "paired_deltas_s": np.array([], dtype=float),
        }

    deltas = []
    for n in nadirs:
        j = int(np.argmin(np.abs(clenches - n)))
        deltas.append(float(n - clenches[j]))
    arr = np.asarray(deltas, dtype=float)
    return {
        "n_nadirs": float(len(nadirs)),
        "n_clenches": float(len(clenches)),
        "mean_abs_delta_s": float(np.mean(np.abs(arr))),
        "mean_signed_delta_s": float(np.mean(arr)),
        "std_abs_delta_s": float(np.std(arr)),
        "paired_deltas_s": arr,
    }


def pearson_spo2_ir_dc(df: pd.DataFrame) -> float:
    a = df["spo2_pct"].astype(float).to_numpy()
    b = df["ir_dc"].astype(float).to_numpy()
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 20:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def build_report(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("=== Oralable Temporalis — Clinical Researcher Summary ===")
    lines.append("")
    site = df["registration_site"].iloc[0] if "registration_site" in df.columns and len(df) else "unknown"
    lines.append(f"Registration site: {site}")
    lines.append(f"Samples: {len(df)}")
    if "elapsed_s" in df.columns:
        lines.append(f"Duration (approx): {float(df['elapsed_s'].iloc[-1]):.2f} s")
    lines.append("")

    n_phasic = count_phasic_events(df)
    n_tonic = count_tonic_events(df)
    n_rescue = count_rescue_events(df)
    lines.append("Event counts (protocol-driven detection)")
    lines.append(f"  Phasic (grinding peak count): {n_phasic}")
    lines.append(f"  Tonic (max clench block present): {n_tonic}")
    lines.append(f"  Rescue (breath-hold clench block present): {n_rescue}")
    lines.append("")

    sashb = total_hypoxic_burden_sashb(df)
    lines.append("SASHB — total hypoxic burden (cumulative, SpO2 < 90% curve endpoint)")
    lines.append(f"  {sashb:.6f} ((90 − SpO2) % · s integrated below 90%)")
    lines.append("")

    tfi = calculate_tfi(df)
    lines.append("Temporalis Fatigue Index (TFI)")
    lines.append(f"  TFI score [0–100]: {tfi['tfi_score']:.4f}")
    lines.append(f"  IR-DC baseline mean slope (per s): {tfi['dc_baseline_slope_per_s']:.6g}")
    lines.append(f"  AC pulse RMS envelope slope (per s): {tfi['ac_pulse_amplitude_slope_per_s']:.6g}")
    lines.append(f"  DC fatigue contribution (norm): {tfi['tfi_dc_contribution']:.4f}")
    lines.append(f"  AC fatigue contribution (norm): {tfi['tfi_ac_contribution']:.4f}")
    lines.append("")

    timing = timing_delta_spo2_nadirs_vs_clenches(df)
    lines.append("SpO2 nadirs vs temporalis clench timing")
    lines.append(f"  SpO2 nadirs detected: {int(timing['n_nadirs'])}")
    lines.append(f"  Clench reference times: {int(timing['n_clenches'])}")
    lines.append(
        f"  Mean |Δt| (nadir − nearest clench): {timing['mean_abs_delta_s']:.3f} s"
        if np.isfinite(timing["mean_abs_delta_s"])
        else "  Mean |Δt|: n/a"
    )
    lines.append(
        f"  Mean signed Δt: {timing['mean_signed_delta_s']:.3f} s"
        if np.isfinite(timing["mean_signed_delta_s"])
        else "  Mean signed Δt: n/a"
    )
    lines.append(
        f"  Std |Δt|: {timing['std_abs_delta_s']:.3f} s"
        if np.isfinite(timing["std_abs_delta_s"])
        else "  Std |Δt|: n/a"
    )
    r = pearson_spo2_ir_dc(df)
    lines.append(f"  Pearson r (SpO2 vs IR-DC, session): {r:.4f}" if np.isfinite(r) else "  Pearson r: n/a")
    lines.append("")
    lines.append("--- End of report ---")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Clinical metrics summary from GOLD CSV.")
    ap.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "validation" / "GOLD_STANDARD_VALIDATION.csv",
        help="Path to GOLD_STANDARD_VALIDATION.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "validation" / "clinical_report.txt",
        help="Output text report path.",
    )
    args = ap.parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

    text = build_report(df)
    print(text, end="")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
