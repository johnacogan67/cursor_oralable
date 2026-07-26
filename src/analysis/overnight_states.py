"""
Overnight Temporalis state classification and bout metrics.

Shared by the 3D clinical infographic and the overnight night-report pack.

States (device-inferred wellness labels — not diagnoses):
  quiet    — baseline / low load
  tonic    — IR-DC drop with stable motion, SpO2 >= rescue threshold
  phasic   — high motion power
  rescue   — IR-DC drop while SpO2 < rescue threshold
  recovery — short settle window after tonic/phasic/rescue ends
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

MOTION_STABLE_THRESHOLD_G = 0.15
IR_DC_DROP_THRESHOLD_PCT = 15.0
RESCUE_SPO2_THRESHOLD = 92.0
RECOVERY_MAX_S = 30.0
RECOVERY_DROP_PCT_MAX = 8.0  # IR-DC largely returned toward baseline

STATE_COLORS = {
    "quiet": "#D3D3D3",
    "tonic": "#4169E1",
    "phasic": "#50C878",
    "rescue": "#8B0000",
    "recovery": "#E8B86D",  # soft amber
}

EVENT_STATES = ("tonic", "phasic", "rescue", "recovery")


def infer_sample_rate(elapsed_s: np.ndarray) -> float:
    if elapsed_s.size < 3:
        return 50.0
    dt = np.diff(elapsed_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 50.0
    med = float(np.median(dt))
    if med <= 0:
        return 50.0
    fs = 1.0 / med
    return fs if 1 <= fs <= 500 else 50.0


def lowpass(signal: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    wn = min(0.99, max(1e-6, cutoff_hz / nyq))
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, signal)


def normalize_ir_dc_to_volts(ir_dc: np.ndarray) -> np.ndarray:
    vmax = float(np.nanmax(ir_dc))
    vmin = float(np.nanmin(ir_dc))
    if np.isfinite(vmax) and np.isfinite(vmin) and vmin >= 0 and vmax <= 5.0:
        return ir_dc

    lo = float(np.nanpercentile(ir_dc, 1))
    hi = float(np.nanpercentile(ir_dc, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(ir_dc, 2.1, dtype=float)
    t = (ir_dc - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)
    return 1.5 + t * (3.0 - 1.5)


def classify_states(
    elapsed_s: np.ndarray,
    ir_dc_v: np.ndarray,
    spo2_pct: np.ndarray,
    motion_power: np.ndarray,
    label_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive Quiet/Tonic/Phasic/Rescue states from IR-DC / motion / SpO2 rules."""
    if elapsed_s.size == 0:
        return np.array([], dtype=object), np.array([], dtype=float)

    baseline_mask = elapsed_s <= 60.0
    baseline_v = (
        float(np.nanmedian(ir_dc_v[baseline_mask])) if np.any(baseline_mask) else float(np.nanmedian(ir_dc_v))
    )
    baseline_v = baseline_v if np.isfinite(baseline_v) and baseline_v > 0 else float(np.nanmedian(ir_dc_v))
    baseline_v = max(1e-6, baseline_v)
    drop_pct = (baseline_v - ir_dc_v) / baseline_v * 100.0

    out = np.full(elapsed_s.shape[0], "quiet", dtype=object)
    phasic_motion_threshold = max(MOTION_STABLE_THRESHOLD_G, float(np.nanpercentile(motion_power, 75)))
    stable_motion = motion_power <= phasic_motion_threshold
    high_motion = motion_power > phasic_motion_threshold
    drop_mask = drop_pct > IR_DC_DROP_THRESHOLD_PCT

    tonic_mask = drop_mask & stable_motion & (spo2_pct >= RESCUE_SPO2_THRESHOLD)
    rescue_mask = drop_mask & (spo2_pct < RESCUE_SPO2_THRESHOLD)
    phasic_mask = high_motion & (spo2_pct >= RESCUE_SPO2_THRESHOLD)

    out[tonic_mask] = "tonic"
    out[phasic_mask] = "phasic"
    out[rescue_mask] = "rescue"

    if label_hint is not None and len(label_hint) == len(out):
        lbl = np.char.lower(label_hint.astype(str))
        out[np.char.find(lbl, "phasic") >= 0] = "phasic"
        out[np.char.find(lbl, "tonic") >= 0] = "tonic"
        out[np.char.find(lbl, "rescue") >= 0] = "rescue"

    return out, drop_pct


def apply_recovery_states(
    states: np.ndarray,
    elapsed_s: np.ndarray,
    drop_pct: np.ndarray,
    motion_power: np.ndarray,
    *,
    recovery_max_s: float = RECOVERY_MAX_S,
    recovery_drop_pct_max: float = RECOVERY_DROP_PCT_MAX,
) -> np.ndarray:
    """
    After tonic/phasic/rescue ends, re-label trailing quiet samples as recovery
    for up to recovery_max_s while motion stays low and IR-DC drop is easing
    (or already near baseline). Does not consume the rest of the night.
    """
    if states.size == 0:
        return states

    out = states.copy()
    active = {"tonic", "phasic", "rescue"}
    phasic_motion_threshold = max(
        MOTION_STABLE_THRESHOLD_G,
        float(np.nanpercentile(motion_power, 75)) if np.any(np.isfinite(motion_power)) else MOTION_STABLE_THRESHOLD_G,
    )

    i = 0
    n = len(out)
    while i < n:
        if out[i] not in active:
            i += 1
            continue
        j = i + 1
        while j < n and out[j] in active:
            j += 1
        if j >= n:
            break
        t_end = float(elapsed_s[j - 1])
        drop_at_end = float(drop_pct[j - 1]) if np.isfinite(drop_pct[j - 1]) else 0.0
        k = j
        while k < n:
            if out[k] in active:
                break
            if out[k] != "quiet":
                break
            t = float(elapsed_s[k])
            if t - t_end > recovery_max_s:
                break
            low_motion = float(motion_power[k]) <= phasic_motion_threshold
            d = float(drop_pct[k]) if np.isfinite(drop_pct[k]) else 0.0
            easing = d <= drop_at_end  # IR-DC recovering toward baseline
            # Stop once IR-DC is essentially back (avoid painting the rest of the night).
            still_settling = d > 2.0
            approaching = d <= max(recovery_drop_pct_max, drop_at_end)
            if low_motion and still_settling and easing and approaching:
                out[k] = "recovery"
                k += 1
                continue
            break
        i = max(k, j)

    return out


def required_column(df: pd.DataFrame, names: tuple[str, ...]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Missing required column. Expected one of: {names}")


def project_measurement_frame(df: pd.DataFrame) -> pd.DataFrame:
    elapsed_col = required_column(df, ("elapsed_s", "ElapsedSeconds", "elapsed"))
    ir_col = required_column(df, ("ir_dc", "temporalis_ir", "ir"))
    spo2_col = required_column(df, ("spo2_pct", "spo2", "SpO2", "spo2_percent"))
    ax_col = required_column(df, ("temporalis_accel_x", "accel_x"))
    ay_col = required_column(df, ("temporalis_accel_y", "accel_y"))
    az_col = required_column(df, ("temporalis_accel_z", "accel_z"))
    out = pd.DataFrame(
        {
            "elapsed_s": pd.to_numeric(df[elapsed_col], errors="coerce"),
            "ir_dc_raw": pd.to_numeric(df[ir_col], errors="coerce"),
            "spo2_pct": pd.to_numeric(df[spo2_col], errors="coerce"),
            "accel_x": pd.to_numeric(df[ax_col], errors="coerce"),
            "accel_y": pd.to_numeric(df[ay_col], errors="coerce"),
            "accel_z": pd.to_numeric(df[az_col], errors="coerce"),
        }
    )
    if "label_name" in df.columns:
        out["label_name"] = df["label_name"].astype(str)
    if "cumulative_sashb" in df.columns:
        out["cumulative_sashb"] = pd.to_numeric(df["cumulative_sashb"], errors="coerce")
    if "ir_dc_mean_5s" in df.columns:
        out["ir_dc_mean_5s"] = pd.to_numeric(df["ir_dc_mean_5s"], errors="coerce")
    if "green_bp" in df.columns:
        out["green_bp"] = pd.to_numeric(df["green_bp"], errors="coerce")
    return out


def enrich_overnight_frame(
    base: pd.DataFrame,
    *,
    apply_recovery: bool = True,
    use_label_hints: bool = True,
) -> pd.DataFrame:
    """Add filtered IR-DC, motion power, mam_state (+ optional recovery)."""
    out = base.dropna(subset=["elapsed_s", "ir_dc_raw", "spo2_pct"]).copy()
    out = out.sort_values("elapsed_s").reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid rows found after filtering.")

    fs = infer_sample_rate(out["elapsed_s"].to_numpy())
    ir_v = normalize_ir_dc_to_volts(out["ir_dc_raw"].to_numpy())
    ir_v_lp = lowpass(ir_v, cutoff_hz=0.8, fs=fs, order=4)

    out["ir_dc_v_filtered_unclipped"] = ir_v_lp
    out["ir_dc_v_filtered"] = np.clip(ir_v_lp, 1.5, 3.0)
    out["spo2_pct"] = np.clip(out["spo2_pct"].astype(float), 85.0, 100.0)
    mag = np.sqrt(out["accel_x"] ** 2 + out["accel_y"] ** 2 + out["accel_z"] ** 2) / 16384.0
    out["motion_g"] = np.abs(mag - 1.0)
    out["motion_power"] = np.sqrt(
        (out["motion_g"] ** 2).rolling(window=25, center=True, min_periods=1).mean()
    )
    label_hint = out["label_name"].to_numpy() if use_label_hints and "label_name" in out.columns else None
    states, drop_pct = classify_states(
        out["elapsed_s"].to_numpy(),
        out["ir_dc_v_filtered_unclipped"].to_numpy(),
        out["spo2_pct"].to_numpy(),
        out["motion_power"].to_numpy(),
        label_hint=label_hint,
    )
    out["ir_drop_pct"] = drop_pct
    if apply_recovery:
        states = apply_recovery_states(
            states,
            out["elapsed_s"].to_numpy(),
            drop_pct,
            out["motion_power"].to_numpy(),
        )
    out["mam_state"] = states
    out.attrs["fs"] = fs
    return out


@dataclass
class Bout:
    state: str
    start_s: float
    end_s: float
    duration_s: float
    min_spo2: float
    delta_spo2: float
    peak_motion: float
    recovery_s: float


def extract_bouts(df: pd.DataFrame, states: Iterable[str] = EVENT_STATES) -> list[Bout]:
    """Contiguous runs of requested states with SpO2 / motion context."""
    wanted = set(states)
    e = df["elapsed_s"].to_numpy(dtype=float)
    s = df["mam_state"].to_numpy()
    spo2 = df["spo2_pct"].to_numpy(dtype=float)
    motion = df["motion_power"].to_numpy(dtype=float)
    bouts: list[Bout] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] not in wanted:
            i += 1
            continue
        state = str(s[i])
        j = i + 1
        while j < n and s[j] == state:
            j += 1
        start_s = float(e[i])
        end_s = float(e[j - 1])
        win_spo2 = spo2[i:j]
        win_spo2 = win_spo2[np.isfinite(win_spo2)]
        min_spo2 = float(np.min(win_spo2)) if win_spo2.size else float("nan")
        first = float(spo2[i]) if np.isfinite(spo2[i]) else float("nan")
        delta = (min_spo2 - first) if np.isfinite(min_spo2) and np.isfinite(first) else float("nan")
        peak_m = float(np.nanmax(motion[i:j])) if j > i else float("nan")

        recovery_s = 0.0
        if state in {"tonic", "phasic", "rescue"} and j < n:
            k = j
            while k < n and s[k] == "recovery":
                k += 1
            if k > j:
                recovery_s = float(e[k - 1] - e[j])

        bouts.append(
            Bout(
                state=state,
                start_s=start_s,
                end_s=end_s,
                duration_s=max(0.0, end_s - start_s),
                min_spo2=min_spo2,
                delta_spo2=delta,
                peak_motion=peak_m,
                recovery_s=recovery_s,
            )
        )
        i = j
    return bouts


def bouts_to_dataframe(bouts: list[Bout]) -> pd.DataFrame:
    rows = [
        {
            "start_s": b.start_s,
            "end_s": b.end_s,
            "duration_s": b.duration_s,
            "type": b.state,
            "min_spo2": b.min_spo2,
            "delta_spo2": b.delta_spo2,
            "peak_motion": b.peak_motion,
            "recovery_s": b.recovery_s,
        }
        for b in bouts
    ]
    return pd.DataFrame(rows)


def state_minutes(df: pd.DataFrame) -> dict[str, float]:
    e = df["elapsed_s"].to_numpy(dtype=float)
    if e.size < 2:
        return {k: 0.0 for k in STATE_COLORS}
    dt = np.diff(e, prepend=e[0])
    dt[0] = float(np.median(dt[1:])) if dt.size > 1 else 0.02
    dt = np.clip(dt, 0.0, 1.0)
    out: dict[str, float] = {}
    for state in STATE_COLORS:
        mask = df["mam_state"].to_numpy() == state
        out[state] = float(np.sum(dt[mask]) / 60.0)
    return out


def compute_kpis(df: pd.DataFrame, bouts: list[Bout] | None = None) -> dict[str, float]:
    if bouts is None:
        bouts = extract_bouts(df)
    minutes = state_minutes(df)
    e = df["elapsed_s"].to_numpy(dtype=float)
    wear_s = float(e[-1] - e[0]) if e.size >= 2 else 0.0

    tonic_bouts = [b for b in bouts if b.state == "tonic"]
    phasic_bouts = [b for b in bouts if b.state == "phasic"]
    rescue_bouts = [b for b in bouts if b.state == "rescue"]
    recovery_after_rescue = [b.recovery_s for b in rescue_bouts if b.recovery_s > 0]

    spo2 = df["spo2_pct"].to_numpy(dtype=float)
    spo2 = spo2[np.isfinite(spo2)]

    sashb = float("nan")
    if "cumulative_sashb" in df.columns and not df["cumulative_sashb"].isna().all():
        sashb = float(df["cumulative_sashb"].iloc[-1])

    tfi = float("nan")
    try:
        from src.analysis.features import calculate_tfi

        tfi_df = df
        if "ir_dc_mean_5s" not in df.columns:
            tfi_df = df.copy()
            tfi_df["ir_dc_mean_5s"] = (
                df["ir_dc_raw"].astype(float).rolling(window=250, center=True, min_periods=1).mean()
            )
        if "green_bp" not in tfi_df.columns:
            # Proxy AC envelope from IR-DC high-pass residual when green_bp absent.
            ir = tfi_df["ir_dc_raw"].astype(float).to_numpy()
            lp = pd.Series(ir).rolling(window=50, center=True, min_periods=1).mean().to_numpy()
            tfi_df = tfi_df.copy()
            tfi_df["green_bp"] = ir - lp
        tfi = float(calculate_tfi(tfi_df)["tfi_score"])
    except Exception:
        tfi = float("nan")

    return {
        "wear_s": wear_s,
        "wear_min": wear_s / 60.0,
        "tonic_min": minutes.get("tonic", 0.0),
        "phasic_min": minutes.get("phasic", 0.0),
        "rescue_min": minutes.get("rescue", 0.0),
        "recovery_min": minutes.get("recovery", 0.0),
        "quiet_min": minutes.get("quiet", 0.0),
        "longest_tonic_s": max((b.duration_s for b in tonic_bouts), default=0.0),
        "phasic_bout_count": float(len(phasic_bouts)),
        "rescue_count": float(len(rescue_bouts)),
        "rescue_total_s": float(sum(b.duration_s for b in rescue_bouts)),
        "recovery_median_s": float(np.median(recovery_after_rescue)) if recovery_after_rescue else 0.0,
        "recovery_max_s": float(np.max(recovery_after_rescue)) if recovery_after_rescue else 0.0,
        "tfi": tfi,
        "sashb": sashb,
        "spo2_mean": float(np.mean(spo2)) if spo2.size else float("nan"),
        "spo2_min": float(np.min(spo2)) if spo2.size else float("nan"),
    }


def hourly_burden(df: pd.DataFrame) -> pd.DataFrame:
    """Minutes per state per hour index + hourly SASHB increment."""
    e = df["elapsed_s"].to_numpy(dtype=float)
    if e.size < 2:
        return pd.DataFrame()

    hour = np.floor(e / 3600.0).astype(int)
    dt = np.diff(e, prepend=e[0])
    dt[0] = float(np.median(dt[1:])) if dt.size > 1 else 0.02
    dt = np.clip(dt, 0.0, 1.0)

    rows = []
    sashb = df["cumulative_sashb"].to_numpy(dtype=float) if "cumulative_sashb" in df.columns else None
    for h in sorted(set(hour.tolist())):
        mask = hour == h
        rec: dict[str, float | int] = {"hour_index": int(h), "hour_start_s": float(h * 3600)}
        for state in STATE_COLORS:
            sm = mask & (df["mam_state"].to_numpy() == state)
            rec[f"{state}_min"] = float(np.sum(dt[sm]) / 60.0)
        if sashb is not None and np.any(mask):
            idx = np.where(mask)[0]
            a = sashb[idx[0]]
            b = sashb[idx[-1]]
            if np.isfinite(a) and np.isfinite(b):
                rec["sashb_delta"] = float(max(0.0, b - a))
            else:
                rec["sashb_delta"] = float("nan")
        else:
            rec["sashb_delta"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)
