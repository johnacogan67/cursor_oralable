"""
Write Dual A / research EDF+ from a 50 Hz Oralable(+ANR) frame.

Claim discipline: research Dual A EDF+ — not PSG; not AHI/ODI;
SASHB_cum is engineering SpO2<90 AUC, not Azarbarzin HB.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, filtfilt

from src.export import edf_spec as spec

FS = spec.FS_HZ
# Project IR-DC: Butterworth low-pass <1 Hz (occlusion / OMG band)
_IR_DC_LP_HZ = 0.8
_IR_DC_BASELINE_S = 5.0


def _finite(a: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=float)
    return np.where(np.isfinite(out), out, 0.0)


def _pick_series(df: pd.DataFrame, names: Sequence[str]) -> np.ndarray | None:
    for n in names:
        if n in df.columns:
            return _finite(df[n].to_numpy())
    return None


def _label_annotations(df: pd.DataFrame) -> list[tuple[float, float, str]]:
    """(onset_s, duration_s, text) for Protocol A label runs."""
    if "label_name" in df.columns:
        labels = df["label_name"].astype(str).to_numpy()
    elif "label_enum" in df.columns:
        labels = df["label_enum"].astype(str).to_numpy()
    else:
        return []
    anns: list[tuple[float, float, str]] = []
    if len(labels) == 0:
        return anns
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            text = f"{spec.ANN_LABEL_PREFIX}{labels[start]}"
            anns.append((start / FS, (i - start) / FS, text[:40]))
            start = i
    return anns


def _bout_annotations(
    bouts: Iterable[tuple[int, int]],
    text: str,
) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    for a, b in bouts:
        if b < a:
            continue
        out.append((a / FS, (b - a + 1) / FS, text))
    return out


def _viewer_scale(
    series: np.ndarray,
    *,
    invert: bool = False,
    target: float = 100.0,
) -> tuple[np.ndarray, float, float]:
    """
    Median-center and scale into ±target for EDFbrowser's default ~±100 amplitude.
    invert=True makes IR-DC occlusion troughs go *up* (EMG-matched polarity).
    """
    x = _finite(series)
    med = float(np.median(x))
    y = med - x if invert else x - med
    # Robust peak (p99 of |y|) → ~0.9 * target
    peak = float(np.percentile(np.abs(y), 99)) if len(y) else 0.0
    scale = max(peak / (0.9 * target), 1e-9)
    y = y / scale
    y = np.clip(y, -target, target)
    return y, -target, target


def _spo2_viewer(series: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    SpO2 as deviation from 100%, scaled so session floor fills ~-90 of ±100 pane.
    100% → ~0; desat/rescue swings are readable next to EMG. Absolute % stays in CSV.
    """
    x = _finite(series)
    dev = x - 100.0
    deepest = float(np.min(dev))
    if deepest >= -0.5:
        deepest = -10.0
    scale = 90.0 / abs(deepest)
    y = np.clip(dev * scale, -100.0, 10.0)
    return y, -100.0, 10.0


def _ir_dc_viewer(series: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    EDFbrowser IR_DC: LP <1 Hz, occlusion = max(0, baseline − LP) vs ~5 s median.
    Quiet baseline + upward peaks (EMG-like). Absolute IR-DC stays in CSV.
    """
    x = _finite(series)
    n = len(x)
    if n >= 27:
        b, a = butter(4, _IR_DC_LP_HZ / (0.5 * FS), btype="low")
        lp = filtfilt(b, a, x)
    else:
        lp = x
    win = max(3, int(_IR_DC_BASELINE_S * FS))
    if win % 2 == 0:
        win += 1
    roll = pd.Series(lp).rolling(win, center=True, min_periods=1).median().to_numpy()
    # Drops only (occlusion); ignore overshoots so trace is not a bipolar buzz
    occ = np.maximum(0.0, roll - lp)
    # Soft floor: suppress small coupling noise (median of positive depths)
    pos = occ[occ > 0]
    if len(pos):
        floor = float(np.percentile(pos, 40))
        occ = np.maximum(0.0, occ - floor)
    return _viewer_scale(occ, invert=False, target=100.0)


def write_dual_a_edf(
    oral_df: pd.DataFrame,
    out_path: Path | str,
    *,
    emg_bouts: Sequence[tuple[int, int]] | None = None,
    spo2_desat_bouts: Sequence[tuple[int, int]] | None = None,
    session_id: str = "dual_a",
    start_dt: datetime | None = None,
    require_emg: bool = True,
    include_raw_ppg: bool = False,
) -> Path:
    """
    Write EDF+C at 50 Hz. When require_emg=True (Dual A), `emg_raw` must be present.

    Default channels: IR_DC (viewer a.u., EMG-matched polarity), EMG, SpO2 (viewer
    %dev from 100), SASHB_cum, Acc*. Absolute IR-DC / SpO2 remain in CSV.
    Raw Red/Green/IR omitted unless include_raw_ppg=True.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(oral_df)
    if n < int(FS):
        raise ValueError(f"Need at least 1 s of 50 Hz data; got n={n}")

    channels: list[tuple[str, str, np.ndarray, float, float]] = []
    # label, unit, data, phys_min, phys_max

    ir_dc = _pick_series(oral_df, ["ir_dc", "ir_dc_mean_5s"])
    if ir_dc is not None:
        # LP occlusion vs local baseline — not raw/AC (that looked like flat buzz vs EMG).
        y, lo, hi = _ir_dc_viewer(ir_dc)
        channels.append((spec.CH_IR_DC, "a.u.", y, lo, hi))

    emg = _pick_series(oral_df, ["emg_raw", "emg"])
    if require_emg and emg is None:
        raise ValueError("Dual A EDF requires emg_raw column (ANR)")
    if emg is not None:
        channels.append((spec.CH_EMG, "", emg, 0.0, 1023.0))

    spo2 = _pick_series(oral_df, ["spo2_pct", "spo2"])
    if spo2 is not None:
        y, lo, hi = _spo2_viewer(spo2)
        channels.append((spec.CH_SPO2, "%dev", y, lo, hi))

    sashb = _pick_series(oral_df, ["cumulative_sashb", "sashb_cum"])
    if sashb is not None:
        channels.append((spec.CH_SASHB, "%s", sashb, float(np.min(sashb)), float(np.max(sashb) + 1e-6)))

    if include_raw_ppg:
        for col, label in (
            ("red", spec.CH_RED),
            ("red_dc", spec.CH_RED),
            ("green", spec.CH_GREEN),
            ("green_dc", spec.CH_GREEN),
            ("ir", spec.CH_IR),
        ):
            if any(c[0] == label for c in channels):
                continue
            series = _pick_series(oral_df, [col])
            if series is None:
                continue
            y, lo, hi = _viewer_scale(series, invert=False, target=100.0)
            channels.append((label, "a.u.", y, lo, hi))

    for col, label in (
        ("accel_x", spec.CH_ACC_X),
        ("accel_y", spec.CH_ACC_Y),
        ("accel_z", spec.CH_ACC_Z),
        ("x", spec.CH_ACC_X),
        ("y", spec.CH_ACC_Y),
        ("z", spec.CH_ACC_Z),
    ):
        if any(c[0] == label for c in channels):
            continue
        series = _pick_series(oral_df, [col])
        if series is None:
            continue
        lo, hi = float(np.min(series)), float(np.max(series))
        if hi <= lo:
            hi = lo + 1.0
        channels.append((label, "", series, lo, hi))

    if not channels:
        raise ValueError("No exportable channels on frame")

    # Trim to whole seconds for clean EDF records
    n_rec = int(n // int(FS))
    n_keep = n_rec * int(FS)
    if n_keep < int(FS):
        raise ValueError("Duration < 1 s after trim")

    start = start_dt or datetime.now(timezone.utc).replace(tzinfo=None)
    if start.tzinfo is not None:
        start = start.astimezone(timezone.utc).replace(tzinfo=None)

    n_sig = len(channels)
    signal_headers: list[dict] = []
    def _phys8(v: float) -> float:
        """EDF physical min/max are 8 ASCII chars — keep short."""
        av = abs(v)
        if av >= 1e5:
            s = f"{v:.2e}"  # e.g. 1.00e+06
        elif av >= 1000:
            s = f"{v:.0f}"
        else:
            s = f"{v:.4g}"
        return float(s[:8])

    for label, unit, data, pmin, pmax in channels:
        if pmax <= pmin:
            pmax = pmin + 1.0
        signal_headers.append(
            {
                "label": label[:16],
                "dimension": (unit or "")[:8],
                "sample_frequency": FS,
                "physical_min": _phys8(pmin),
                "physical_max": _phys8(pmax),
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "Oralable-ANR",
                "prefilter": "",
            }
        )

    data_list = [ch[2][:n_keep] for ch in channels]

    annotations: list[tuple[float, float, str]] = []
    annotations.extend(_label_annotations(oral_df.iloc[:n_keep]))
    if emg is not None and emg_bouts:
        annotations.extend(_bout_annotations(emg_bouts, spec.ANN_EMG_BOUT))
    if spo2_desat_bouts:
        annotations.extend(_bout_annotations(spo2_desat_bouts, spec.ANN_SPO2_DESAT))

    if out_path.exists():
        out_path.unlink()

    writer = pyedflib.EdfWriter(
        str(out_path),
        n_sig,
        file_type=pyedflib.FILETYPE_EDFPLUS,
    )
    try:
        writer.setPatientName(session_id[:80])
        writer.setPatientCode("X")
        writer.setSex(0)
        writer.setStartdatetime(start)
        writer.setEquipment("Oralable-DualA")
        writer.setSignalHeaders(signal_headers)
        # pyedflib writeSamples expects list of arrays (full recording)
        writer.writeSamples(data_list)
        for onset, dur, text in annotations:
            if onset >= n_rec:
                continue
            writer.writeAnnotation(onset, dur if dur > 0 else -1, text[:40])
    finally:
        writer.close()

    return out_path
