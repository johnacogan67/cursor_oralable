#!/usr/bin/env python3
"""
Temporalis gold pipeline: parse BLE log → 50 Hz → 4th-order Butterworth filters (via
compute_filters) → Temporalis protocol labels → GOLD_STANDARD_VALIDATION.csv + PNG plots.

Usage:
  python scripts/process_temporalis_gold.py [path/to/TEMPORALIS_RAW_01.csv]
  python scripts/process_temporalis_gold.py --input data/raw/TEMPORALIS_RAW_01.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.features import ClinicalBiometricSuite, compute_filters
from src.analysis.label_generator import apply_temporalis_labels_to_frame
from src.parser.log_parser import parse_accelerometer_log, parse_oralable_log
from src.processing.resampler import resample_raw_to_50hz


def _build_merged_raw(
    log_path: Path,
    channel_order: str | None,
) -> pd.DataFrame:
    ppg = parse_oralable_log(log_path, channel_order=channel_order)
    if ppg is None or ppg.empty:
        raise ValueError(f"No PPG rows parsed from {log_path}")
    accel = parse_accelerometer_log(log_path)
    t0 = float(ppg["timestamp_s"].iloc[0])
    ppg = ppg.copy()
    ppg["timestamp_s"] = ppg["timestamp_s"] - t0
    if accel is None or accel.empty:
        accel = pd.DataFrame({
            "timestamp_s": ppg["timestamp_s"],
            "temporalis_accel_x": 0,
            "temporalis_accel_y": 0,
            "temporalis_accel_z": 0,
        })
    else:
        accel = accel.copy()
        accel["timestamp_s"] = accel["timestamp_s"] - t0
        accel = accel.rename(
            columns={"accel_x": "temporalis_accel_x", "accel_y": "temporalis_accel_y", "accel_z": "temporalis_accel_z"}
        )
    merged = pd.merge_asof(
        ppg.sort_values("timestamp_s"),
        accel.sort_values("timestamp_s"),
        on="timestamp_s",
        direction="nearest",
        tolerance=0.12,
    )
    return merged


def _accel_mag(df: pd.DataFrame) -> np.ndarray:
    ax = df["temporalis_accel_x"].astype(float).to_numpy()
    ay = df["temporalis_accel_y"].astype(float).to_numpy()
    az = df["temporalis_accel_z"].astype(float).to_numpy()
    return np.sqrt(ax * ax + ay * ay + az * az)


def _plot_apnea_hoi(
    df: pd.DataFrame,
    out_path: Path,
    t0_s: float = 235.0,
    t1_s: float = 275.0,
) -> None:
    """Plot A: IR-DC (HOI), SpO2, label_enum during simulated apnea + tonic rescue window."""
    sub = df[(df["elapsed_s"] >= t0_s) & (df["elapsed_s"] <= t1_s)].copy()
    if sub.empty:
        sub = df.copy()

    fig, (ax0, ax1) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 7), gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    t = sub["elapsed_s"].to_numpy()
    ax0.plot(t, sub["ir_dc"].to_numpy(), color="darkred", lw=1.0, label="IR-DC (HOI)")
    ax0.set_ylabel("IR-DC (raw)")
    ax0.legend(loc="upper left")
    ax0.set_title("Temporalis validation — apnea window (IR-DC + SpO2 + labels)")
    ax0.grid(True, alpha=0.3)

    if "spo2_pct" in sub.columns:
        ax_spo2 = ax0.twinx()
        ax_spo2.plot(t, sub["spo2_pct"].to_numpy(), color="steelblue", lw=1.0, alpha=0.85, label="SpO2 %")
        ax_spo2.set_ylabel("SpO2 (%)")
        ax_spo2.legend(loc="upper right")

    ax1.step(t, sub["label_enum"].to_numpy(), where="post", color="black", lw=1.0)
    ax1.set_ylabel("label_enum")
    ax1.set_xlabel("Elapsed (s) from session start")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_phasic_vs_tonic(df: pd.DataFrame, out_path: Path) -> None:
    """Plot B: accelerometer magnitude vs IR-DC for tonic vs phasic protocol windows."""
    tonic = df[(df["elapsed_s"] >= 120.0) & (df["elapsed_s"] < 130.0)]
    phasic = df[(df["elapsed_s"] >= 180.0) & (df["elapsed_s"] < 200.0)]

    fig, ax = plt.subplots(figsize=(8, 6))
    if not tonic.empty:
        ax.scatter(
            tonic["ir_dc"].to_numpy(),
            _accel_mag(tonic),
            s=4,
            alpha=0.5,
            c="tab:orange",
            label="Tonic max (02:00–02:10)",
        )
    if not phasic.empty:
        ax.scatter(
            phasic["ir_dc"].to_numpy(),
            _accel_mag(phasic),
            s=4,
            alpha=0.5,
            c="tab:green",
            label="Phasic grind (03:00–03:20)",
        )
    ax.set_xlabel("IR-DC (HOI, raw)")
    ax.set_ylabel("Accelerometer magnitude")
    ax.set_title("Temporalis — phasic vs tonic (stability / differentiation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    log_path: Path,
    protocol_path: Path | None,
    out_gold: Path,
    plot_dir: Path,
    channel_order: str | None,
) -> Path:
    merged_raw = _build_merged_raw(log_path, channel_order=channel_order)
    tmp_raw = out_gold.parent / "_temporalis_merged_for_resample.csv"
    merged_raw.to_csv(tmp_raw, index=False)

    tmp_50 = out_gold.parent / "_temporalis_50hz.csv"
    resampled = resample_raw_to_50hz(raw_path=tmp_raw, out_path=tmp_50)
    resampled = compute_filters(resampled)
    clinical = ClinicalBiometricSuite().process(resampled)
    df = pd.concat([resampled, clinical], axis=1)

    delta = df.index - df.index[0]
    try:
        ts = delta.total_seconds()
        df["elapsed_s"] = np.asarray(ts, dtype=float)
    except (AttributeError, TypeError):
        df["elapsed_s"] = np.arange(len(df), dtype=float) / 50.0

    df = apply_temporalis_labels_to_frame(df, elapsed_col="elapsed_s", protocol_path=protocol_path)
    df["registration_site"] = "temporalis"

    _plot_apnea_hoi(df, plot_dir / "plot_a_smoking_gun_apnea_hoi.png")
    _plot_phasic_vs_tonic(df, plot_dir / "plot_b_clinical_stability_phasic_vs_tonic.png")

    rename_map = {"green": "temporalis_green", "red": "temporalis_red", "ir": "temporalis_ir"}
    df_out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    out_gold.parent.mkdir(parents=True, exist_ok=True)
    df_out.reset_index().to_csv(out_gold, index=False)

    tmp_raw.unlink(missing_ok=True)
    tmp_50.unlink(missing_ok=True)
    return out_gold


def main() -> None:
    ap = argparse.ArgumentParser(description="Temporalis gold validation pipeline.")
    ap.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=ROOT / "TEMPORALIS_RAW_01.csv",
        help="BLE log (TEMPORALIS_RAW_01.csv)",
    )
    ap.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "docs" / "TEMPORALIS_COLLECTION_PROTOCOL.md",
        help="Protocol markdown for label offsets.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "validation" / "GOLD_STANDARD_VALIDATION.csv",
        help="Output gold CSV path.",
    )
    ap.add_argument(
        "--plot-dir",
        type=Path,
        default=ROOT / "plots" / "temporalis_validation",
        help="Directory for PNG validation plots.",
    )
    ap.add_argument(
        "--channel-order",
        type=str,
        default=None,
        help="PPG channel order override (e.g. R_G_IR). Default: log_parser default.",
    )
    args = ap.parse_args()
    log_path = args.input
    if not log_path.exists():
        print(f"Input not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    out = run(
        log_path=log_path,
        protocol_path=args.protocol if args.protocol.exists() else None,
        out_gold=args.out,
        plot_dir=args.plot_dir,
        channel_order=args.channel_order,
    )
    print(f"Wrote {out}")
    print(f"Plots under {args.plot_dir}")


if __name__ == "__main__":
    main()
