#!/usr/bin/env python3
"""
Train Temporalis MAM Net (Keras) from a BLE session log or pre-built gold CSV.

Pipeline (raw log):
  1) parse_oralable_log + parse_accelerometer_log → merge → resample_raw_to_50hz (20 ms grid)

Strict 50 Hz file (recommended with ``src/processing/resampler.py``):
  - Build merged CSV with ``timestamp_s``, ``green``, ``red``, ``ir``, accel columns (e.g. export from step 1), then
    ``resample_raw_to_50hz(merged_path, out_path)``.
  - Train: ``python scripts/generate_mam_model.py --session-50hz out_path``.
  2) Filters aligned with iOS UnifiedBiometricProcessor / OralableCore:
     - IR DC: lowpass 0.8 Hz (filtfilt)
     - Green / Red AC: bandpass 0.5–4 Hz (Temporalis paths)
     - Accel: int16 → g (÷16384)
  3) Labels from docs/TEMPORALIS_COLLECTION_PROTOCOL.md via label_generator (10 fine segments → 4 MAM classes)

Tensor layout [1, 50, 6]: green AC, IR DC, red AC, accel x,y,z (matches ClassificationBuffer).

Output:
  mam_net_temporalis.h5 — convert with:
    python scripts/convert_temporalis_mam.py --keras mam_net_temporalis.h5

Legacy BruxismMAM [1,250,3] stub generation was removed; use convert_temporalis_mam.py for Core ML.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import filtfilt
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.features import FS, _butter_bandpass, _butter_lowpass
from src.analysis.label_generator import apply_temporalis_labels_to_frame
from src.parser.log_parser import parse_accelerometer_log, parse_oralable_log
from src.processing.resampler import resample_raw_to_50hz

WINDOW = 50
ACCEL_SCALE = 16384.0

# label_enum (Temporalis fine) → MAM class (quiet, phasic, tonic, rescue)
FINE_TO_MAM = {
    -1: 0,
    0: 0,
    1: 0,
    2: 0,
    3: 2,  # tonic_max
    4: 0,
    5: 1,  # phasic_grinding
    6: 0,
    7: 0,  # simulated_apnea → quiet (HOI pattern, not a fifth MAM UI class)
    8: 3,  # tonic_rescue
    9: 0,
}


def _build_merged_raw(log_path: Path, channel_order: str | None) -> pd.DataFrame:
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
            "temporalis_accel_x": 0.0,
            "temporalis_accel_y": 0.0,
            "temporalis_accel_z": 0.0,
        })
    else:
        accel = accel.copy()
        accel["timestamp_s"] = accel["timestamp_s"] - t0
        accel = accel.rename(columns={
            "accel_x": "temporalis_accel_x",
            "accel_y": "temporalis_accel_y",
            "accel_z": "temporalis_accel_z",
        })
    merged = pd.merge_asof(
        ppg.sort_values("timestamp_s"),
        accel.sort_values("timestamp_s"),
        on="timestamp_s",
        direction="nearest",
        tolerance=0.12,
    )
    return merged


def _add_elapsed_s(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    delta = out.index - out.index[0]
    try:
        out["elapsed_s"] = np.asarray(delta.total_seconds(), dtype=float)
    except (AttributeError, TypeError):
        out["elapsed_s"] = np.arange(len(out), dtype=float) / FS
    return out


def _temporalis_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add mam_* columns: IR DC lowpass, green/red AC 0.5–4 Hz, accel in g."""
    need = ("green", "ir", "red", "temporalis_accel_x", "temporalis_accel_y", "temporalis_accel_z")
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Expected column {c!r} on 50 Hz frame (got {list(df.columns)[:20]}…)")

    g = df["green"].astype(float).to_numpy()
    ir = df["ir"].astype(float).to_numpy()
    r = df["red"].astype(float).to_numpy()

    b_lp, a_lp = _butter_lowpass(cutoff_hz=0.8, fs=FS, order=4)
    b_bp, a_bp = _butter_bandpass(0.5, 4.0, fs=FS, order=4)

    ir_dc = filtfilt(b_lp, a_lp, np.nan_to_num(ir, nan=0.0))
    green_ac = filtfilt(b_bp, a_bp, np.nan_to_num(g - np.nanmean(g), nan=0.0))
    red_ac = filtfilt(b_bp, a_bp, np.nan_to_num(r - np.nanmean(r), nan=0.0))

    out = df.copy()
    out["mam_ir_dc"] = ir_dc
    out["mam_green_ac"] = green_ac
    out["mam_red_ac"] = red_ac
    out["mam_ax"] = df["temporalis_accel_x"].astype(float).to_numpy() / ACCEL_SCALE
    out["mam_ay"] = df["temporalis_accel_y"].astype(float).to_numpy() / ACCEL_SCALE
    out["mam_az"] = df["temporalis_accel_z"].astype(float).to_numpy() / ACCEL_SCALE
    return out


def _fine_enum_to_mam(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(len(arr), dtype=np.int64)
    for i, v in enumerate(arr):
        out[i] = FINE_TO_MAM.get(int(v), 0)
    return out


def build_training_arrays(
    df: pd.DataFrame,
    window: int = WINDOW,
    stride: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    feat_cols = ["mam_green_ac", "mam_ir_dc", "mam_red_ac", "mam_ax", "mam_ay", "mam_az"]
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    fine = df["label_enum"].to_numpy(dtype=np.int16)
    mam = _fine_enum_to_mam(fine)
    mat = df[feat_cols].to_numpy(dtype=np.float64)
    n = mat.shape[0]
    for start in range(0, n - window + 1, stride):
        end = start + window
        if np.any(fine[start:end] < 0):
            continue
        labels_win = mam[start:end]
        # majority vote on window (stabler than center-only for 50-sample bins)
        counts = np.bincount(labels_win, minlength=4)
        y = int(np.argmax(counts))
        X_list.append(mat[start:end].astype(np.float32))
        y_list.append(y)

    if not X_list:
        raise ValueError("No training windows; check label_enum coverage and log duration.")
    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


def dataframe_from_log(
    log_path: Path,
    protocol_path: Path | None,
    channel_order: str | None,
) -> pd.DataFrame:
    merged = _build_merged_raw(log_path, channel_order=channel_order)
    with tempfile.TemporaryDirectory() as td:
        tmp_raw = Path(td) / "merged.csv"
        tmp_50 = Path(td) / "session_50hz.csv"
        merged.to_csv(tmp_raw, index=False)
        resampled = resample_raw_to_50hz(raw_path=tmp_raw, out_path=tmp_50)
    df = _add_elapsed_s(resampled)
    df = apply_temporalis_labels_to_frame(
        df,
        elapsed_col="elapsed_s",
        protocol_path=protocol_path if protocol_path and protocol_path.exists() else None,
    )
    return _temporalis_filter_columns(df)


def dataframe_from_session_50hz_csv(path: Path, protocol_path: Path | None) -> pd.DataFrame:
    """
    Load output of ``resample_raw_to_50hz`` (datetime index as first CSV column; 20 ms grid).
    Expects columns from merged PPG+accel: green, red, ir, temporalis_accel_* (or accel_x/y/z).
    """
    df = pd.read_csv(path)
    time_col = df.columns[0]
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=False)
    df = df.set_index(time_col)
    df = _add_elapsed_s(df)
    if "accel_x" in df.columns and "temporalis_accel_x" not in df.columns:
        df = df.rename(columns={
            "accel_x": "temporalis_accel_x",
            "accel_y": "temporalis_accel_y",
            "accel_z": "temporalis_accel_z",
        })
    df = apply_temporalis_labels_to_frame(
        df,
        elapsed_col="elapsed_s",
        protocol_path=protocol_path if protocol_path and protocol_path.exists() else None,
    )
    return _temporalis_filter_columns(df)


def dataframe_from_gold_csv(path: Path, protocol_path: Path | None) -> pd.DataFrame:
    """Gold CSV from process_temporalis_gold: may use temporalis_* PPG names; normalize columns."""
    df = pd.read_csv(path)
    # Map temporalis_* back to green/ir/red if needed
    rename = {}
    if "temporalis_green" in df.columns:
        rename["temporalis_green"] = "green"
    if "temporalis_ir" in df.columns:
        rename["temporalis_ir"] = "ir"
    if "temporalis_red" in df.columns:
        rename["temporalis_red"] = "red"
    df = df.rename(columns=rename)

    if "green" not in df.columns or "ir" not in df.columns or "red" not in df.columns:
        raise ValueError(f"Gold CSV missing green/ir/red columns: {path}")

    if "temporalis_accel_x" not in df.columns:
        # try unprefixed
        if "accel_x" in df.columns:
            df = df.rename(columns={
                "accel_x": "temporalis_accel_x",
                "accel_y": "temporalis_accel_y",
                "accel_z": "temporalis_accel_z",
            })
        else:
            raise ValueError("Gold CSV missing accelerometer columns")

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif "timestamp_s" in df.columns:
        ref = pd.Timestamp("1970-01-01 00:00:00")
        df["datetime"] = ref + pd.to_timedelta(df["timestamp_s"], unit="s")
        df = df.set_index("datetime").drop(columns=["timestamp_s"], errors="ignore")
    elif "elapsed_s" in df.columns:
        ref = pd.Timestamp("1970-01-01 00:00:00")
        df["datetime"] = ref + pd.to_timedelta(df["elapsed_s"], unit="s")
        df = df.set_index("datetime")
    else:
        time_like = df.columns[0]
        try:
            df = df.rename(columns={time_like: "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        except (ValueError, TypeError) as e:
            raise ValueError("Gold CSV needs datetime, timestamp_s, or elapsed_s") from e

    df = _add_elapsed_s(df)
    if "label_enum" not in df.columns:
        df = apply_temporalis_labels_to_frame(
            df,
            elapsed_col="elapsed_s",
            protocol_path=protocol_path if protocol_path and protocol_path.exists() else None,
        )
    return _temporalis_filter_columns(df)


def train_and_save(X: np.ndarray, y: np.ndarray, out_h5: Path, epochs: int, val_frac: float) -> None:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as e:
        raise SystemExit(f"Install tensorflow: pip install tensorflow ({e})") from e

    tf.keras.utils.set_random_seed(42)

    n = X.shape[0]
    n_val = max(1, int(n * val_frac))
    perm = np.random.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Input scale matches device tensors; BatchNormalization learns per-channel calibration for Core ML export.
    inp = keras.layers.Input(shape=(WINDOW, 6), name="input")
    h = keras.layers.BatchNormalization(axis=-1)(inp)
    h = keras.layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(h)
    h = keras.layers.MaxPooling1D(2)(h)
    h = keras.layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(h)
    h = keras.layers.GlobalAveragePooling1D()(h)
    h = keras.layers.Dense(32, activation="relu")(h)
    out = keras.layers.Dense(4, activation="softmax", name="probabilities")(h)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    y_int = y_train.astype(np.int32)
    classes_present = np.unique(y_int)
    cw = compute_class_weight(class_weight="balanced", classes=classes_present, y=y_int)
    cw_d = {int(c): float(w) for c, w in zip(classes_present, cw)}
    for c in range(4):
        cw_d.setdefault(c, 1.0)

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=min(32, len(X_train)),
        class_weight=cw_d,
        verbose=1,
    )

    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_h5)
    print(f"Saved Keras model → {out_h5}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Temporalis MAM Net (Keras .h5)")
    ap.add_argument(
        "--input",
        type=Path,
        default=ROOT / "TEMPORALIS_RAW_01.csv",
        help="BLE log CSV (default: TEMPORALIS_RAW_01.csv)",
    )
    ap.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Optional: pre-built gold CSV instead of --input log",
    )
    ap.add_argument(
        "--session-50hz",
        type=Path,
        default=None,
        help="Strict 50 Hz CSV from resample_raw_to_50hz (overrides --input and --gold)",
    )
    ap.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "docs" / "TEMPORALIS_COLLECTION_PROTOCOL.md",
    )
    ap.add_argument(
        "--out",
        "-o",
        type=Path,
        default=ROOT / "mam_net_temporalis.h5",
        help="Output Keras H5 path",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--stride", type=int, default=25, help="Window stride (samples at 50 Hz)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--channel-order", type=str, default=None, help="PPG order, e.g. R_G_IR")
    args = ap.parse_args()

    proto = args.protocol if args.protocol.exists() else None

    if args.session_50hz:
        if not args.session_50hz.exists():
            print(f"--session-50hz not found: {args.session_50hz}", file=sys.stderr)
            return 1
        df = dataframe_from_session_50hz_csv(args.session_50hz, protocol_path=proto)
    elif args.gold:
        if not args.gold.exists():
            print(f"--gold not found: {args.gold}", file=sys.stderr)
            return 1
        df = dataframe_from_gold_csv(args.gold, protocol_path=proto)
    else:
        if not args.input.exists():
            print(f"--input not found: {args.input}", file=sys.stderr)
            return 1
        df = dataframe_from_log(args.input, protocol_path=proto, channel_order=args.channel_order)

    X, y = build_training_arrays(df, window=WINDOW, stride=args.stride)
    print(f"Windows: {X.shape[0]}, shape {X.shape}, labels: {np.bincount(y, minlength=4)}")
    train_and_save(X, y, args.out, epochs=args.epochs, val_frac=args.val_frac)
    print("Next: python scripts/convert_temporalis_mam.py --keras", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
