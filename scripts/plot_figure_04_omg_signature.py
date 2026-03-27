#!/usr/bin/env python3
"""
IEEE-style figure: 60 s window from TEMPORALIS_RAW_01 — filtered IR-DC (grey) + MAM Rescue probability (red).

Requires: merged log → 50 Hz (same path as run_temporalis_mam_pipeline), BruxismMAM_Temporalis.mlpackage
(sibling OralableCore repo) or --keras mam_net_temporalis.h5.

Output: FIGURE_04_OMG_SIGNATURE.png in cursor_oralable root (override with --out).

Usage (from cursor_oralable):
  python scripts/plot_figure_04_omg_signature.py
  python scripts/plot_figure_04_omg_signature.py --raw data/other.csv --window-start 120
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_mam_helpers():
    spec = importlib.util.spec_from_file_location(
        "_mam", ROOT / "scripts" / "generate_mam_model.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._build_merged_raw, mod._add_elapsed_s, mod._temporalis_filter_columns


def _default_mlpackage() -> Path | None:
    for p in (
        ROOT.parent / "OralableCore" / "Sources" / "OralableCore" / "Resources" / "BruxismMAM_Temporalis.mlpackage",
        ROOT / "BruxismMAM_Temporalis.mlpackage",
    ):
        if p.exists():
            return p
    return None


def _predict_rescue_coreml(mlpackage: Path, mat: np.ndarray, window: int) -> np.ndarray:
    """mat: (N, 6) float32 in feature order; returns length-N array, NaN until first full window."""
    from coremltools.models import MLModel

    model = MLModel(str(mlpackage))
    n = mat.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        chunk = mat[i - window + 1 : i + 1][None, ...].astype(np.float32)
        pred = model.predict({"input": chunk})
        prob = None
        for key in ("probabilities", "Identity", "var_"):
            if key in pred:
                prob = pred[key]
                break
        if prob is None:
            raise RuntimeError(f"Unexpected Core ML outputs keys: {pred.keys()}")
        arr = np.asarray(prob).reshape(-1)
        if arr.size < 4:
            raise RuntimeError(f"Expected 4-class probs, got shape {arr.shape}")
        out[i] = float(arr[3])
    return out


def _predict_rescue_keras(h5: Path, mat: np.ndarray, window: int) -> np.ndarray:
    from tensorflow import keras

    model = keras.models.load_model(h5, compile=False)
    n = mat.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        chunk = mat[i - window + 1 : i + 1][None, ...].astype(np.float32)
        arr = model.predict(chunk, verbose=0).reshape(-1)
        out[i] = float(arr[3])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="FIGURE_04: IR-DC + Rescue prob (IEEE fonts)")
    ap.add_argument("--raw", type=Path, default=ROOT / "TEMPORALIS_RAW_01.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "FIGURE_04_OMG_SIGNATURE.png")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--window-start", type=float, default=0.0, help="Start offset in seconds (after t0).")
    ap.add_argument("--channel-order", type=str, default=None)
    ap.add_argument("--mlpackage", type=Path, default=None)
    ap.add_argument("--keras", type=Path, default=None)
    args = ap.parse_args()

    if not args.raw.exists():
        print(f"Raw log not found: {args.raw}", file=sys.stderr)
        return 1

    _build_merged_raw, _add_elapsed_s, _temporalis_filter_columns = _load_mam_helpers()
    from src.processing.resampler import resample_raw_to_50hz

    merged = _build_merged_raw(args.raw, args.channel_order)
    with tempfile.TemporaryDirectory() as td:
        tmp_raw = Path(td) / "merged.csv"
        tmp_50 = Path(td) / "50hz.csv"
        merged.to_csv(tmp_raw, index=False)
        df50 = resample_raw_to_50hz(raw_path=tmp_raw, out_path=tmp_50)

    df = _add_elapsed_s(df50)
    df = _temporalis_filter_columns(df)

    t0 = float(args.window_start)
    t1 = t0 + float(args.seconds)
    df = df[(df["elapsed_s"] >= t0) & (df["elapsed_s"] < t1)].copy()
    if len(df) < 55:
        print(f"Too few rows in [{t0}, {t1}) s: {len(df)} (need enough for 50-sample windows).", file=sys.stderr)
        return 1

    feat_cols = ["mam_green_ac", "mam_ir_dc", "mam_red_ac", "mam_ax", "mam_ay", "mam_az"]
    mat = df[feat_cols].to_numpy(dtype=np.float32)
    mlp = args.mlpackage or _default_mlpackage()
    if args.keras and args.keras.exists():
        rescue = _predict_rescue_keras(args.keras, mat, window=50)
    elif mlp and mlp.exists():
        rescue = _predict_rescue_coreml(mlp, mat, window=50)
    else:
        print(
            "No Core ML model found. Pass --mlpackage path/to/BruxismMAM_Temporalis.mlpackage "
            "or --keras mam_net_temporalis.h5 (train via run_temporalis_mam_pipeline.py).",
            file=sys.stderr,
        )
        return 1

    t = df["elapsed_s"].to_numpy()
    ir_dc = df["mam_ir_dc"].to_numpy()

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )

    fig, ax1 = plt.subplots(figsize=(6.5, 2.8), layout="constrained")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Filtered IR-DC (a.u.)", color="0.35")
    ax1.plot(t, ir_dc, color="0.45", linewidth=1.0, label="Filtered IR-DC")
    ax1.tick_params(axis="y", labelcolor="0.35")

    ax2 = ax1.twinx()
    ax2.set_ylabel("MAM Rescue probability", color="darkred")
    ax2.plot(t, rescue, color="darkred", linewidth=1.1, label="Rescue (softmax)")
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax2.set_ylim(0, 1.05)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right", frameon=True)

    ax1.set_title("Temporalis hemodynamic signature (60 s) — IR-DC vs MAM Rescue")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="png", dpi=300)
    plt.close(fig)
    print(f"Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
