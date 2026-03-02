"""
Feature extraction on 50 Hz PPG/accelerometer data.

Loads `data/processed/session_50hz.csv` (50 Hz grid), extracts beat-wise
timing features and an IR DC baseline, and writes them to
`data/datasets/features_labeled.csv`.

For each detected heartbeat (from the Green PPG channel) we compute:
- pulse_onset_time  : timestamp of foot of the upstroke (local minimum)
- systolic_peak_time: timestamp of systolic peak (local maximum)
- pulse_offset_time : timestamp of foot after the peak (next local minimum)
- falling_edge_s    : duration from onset → peak   (rise time)
- rising_edge_s     : duration from peak  → offset (decay time)

We also compute:
- ir_dc             : IR low-pass (<1 Hz) DC component
- ir_dc_mean_5s     : rolling 5 s mean of ir_dc at each beat peak

Per 5-second window biomarkers (for training / sleep bruxism vs arousal):
- ir_dc_shift_5s    : average drop in IR baseline (occlusion indicator)
- rise_fall_symmetry_5s : ratio reperfusion time / drop time (pulse morphology)
- hrv_svd_s1_5s     : leading singular value of HRV (RR-interval matrix)
- hrv_svd_s1_s2_ratio_5s : SVD s1/s2 ratio (gold-standard 2025 biomarker)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


FS = 50.0  # Hz (per .cursorrules, session_50hz is on a 50 Hz grid)


@dataclass
class BeatFeature:
    onset_idx: int
    peak_idx: int
    offset_idx: int
    onset_time: pd.Timestamp
    peak_time: pd.Timestamp
    offset_time: pd.Timestamp
    falling_edge_s: float  # onset -> peak
    rising_edge_s: float   # peak  -> offset
    ir_dc_mean_5s: float


def _butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = high_hz / nyq
    return butter(order, [low, high], btype="band")


def _butter_lowpass(cutoff_hz: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    return butter(order, cutoff_hz / nyq, btype="low")


def load_session_50hz(path: str | Path | None = None) -> pd.DataFrame:
    """Load the 50 Hz session CSV and ensure datetime index."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "processed" / "session_50hz.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"50Hz session CSV not found: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def compute_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add filtered channels:
    - ir_dc: IR low-pass (<1 Hz) to get DC / baseline (hemodynamic occlusion).
    - green_bp: Green band-pass 0.5–8 Hz for pulsatile morphology / dicrotic notch.
    - red_dc, red_ac, ir_ac: For SpO2 (ClinicalBiometricSuite); added if 'red' exists.
    """
    if "green" not in df.columns or "ir" not in df.columns:
        raise ValueError("Expected 'green' and 'ir' columns in session_50hz.csv")

    sig_green = df["green"].astype(float).to_numpy()
    sig_ir = df["ir"].astype(float).to_numpy()

    # Low-pass < 1 Hz for DC (use 0.8 Hz cutoff for some margin)
    b_lp, a_lp = _butter_lowpass(cutoff_hz=0.8, fs=FS, order=4)
    ir_dc = filtfilt(b_lp, a_lp, np.nan_to_num(sig_ir, nan=0.0))
    df["ir_dc"] = ir_dc

    # Band-pass 0.5–8 Hz on Green for beat detection and morphology
    b_bp, a_bp = _butter_bandpass(0.5, 8.0, fs=FS, order=4)
    # Optional: remove mean before filtering to reduce edge effects
    sig_green_detrended = sig_green - np.nanmean(sig_green)
    green_bp = filtfilt(b_bp, a_bp, np.nan_to_num(sig_green_detrended, nan=0.0))
    df["green_bp"] = green_bp

    # Rolling 5-second mean of IR DC (time-based, uses datetime index)
    df["ir_dc_mean_5s"] = df["ir_dc"].rolling("5s", center=True, min_periods=1).mean()

    # Red/IR AC/DC for SpO2 (ClinicalBiometricSuite)
    if "red" in df.columns:
        sig_red = df["red"].astype(float).to_numpy()
        red_dc = filtfilt(b_lp, a_lp, np.nan_to_num(sig_red, nan=0.0))
        df["red_dc"] = red_dc
        red_ac = filtfilt(b_bp, a_bp, np.nan_to_num(sig_red - np.nanmean(sig_red), nan=0.0))
        df["red_ac"] = red_ac
        ir_ac = filtfilt(b_bp, a_bp, np.nan_to_num(sig_ir - np.nanmean(sig_ir), nan=0.0))
        df["ir_ac"] = ir_ac

    return df


def detect_beats_from_green_bp(df: pd.DataFrame) -> List[BeatFeature]:
    """
    Detect beats on band-passed Green and derive onset/peak/offset indices.

    Strategy:
    - Use scipy.signal.find_peaks on green_bp with a minimum distance (~0.4 s)
      and a modest prominence threshold.
    - For each peak:
        * Onset: local minimum before the peak (between previous peak and peak).
        * Offset: local minimum after the peak (between peak and next peak).
    """
    if "green_bp" not in df.columns:
        raise ValueError("Column 'green_bp' not found; run compute_filters() first.")

    sig = df["green_bp"].to_numpy()
    n = len(sig)
    if n < 3:
        return []

    # Peak detection parameters
    min_distance_samples = int(0.4 * FS)  # at most ~150 bpm
    prom = np.nanstd(sig) * 0.5 if np.isfinite(np.nanstd(sig)) else 0.0
    peaks, props = find_peaks(sig, distance=min_distance_samples, prominence=prom if prom > 0 else None)
    if len(peaks) < 2:
        return []

    beats: List[BeatFeature] = []
    for i, p_idx in enumerate(peaks):
        # Onset search window: from previous peak to current peak
        if i == 0:
            start = max(0, p_idx - int(0.8 * FS))
        else:
            start = peaks[i - 1]
        if start >= p_idx:
            start = max(0, p_idx - int(0.8 * FS))

        seg_prev = sig[start:p_idx]
        if seg_prev.size == 0:
            continue
        onset_rel = int(np.argmin(seg_prev))
        onset_idx = start + onset_rel

        # Offset search window: from peak to next peak
        if i == len(peaks) - 1:
            end = min(n - 1, p_idx + int(0.8 * FS))
        else:
            end = peaks[i + 1]
        if end <= p_idx:
            end = min(n - 1, p_idx + int(0.8 * FS))

        seg_post = sig[p_idx:end + 1]
        if seg_post.size == 0:
            continue
        offset_rel = int(np.argmin(seg_post))
        offset_idx = p_idx + offset_rel

        # Sanity checks
        if not (0 <= onset_idx < p_idx < offset_idx < n):
            continue

        onset_time = df.index[onset_idx]
        peak_time = df.index[p_idx]
        offset_time = df.index[offset_idx]

        falling_edge_s = (p_idx - onset_idx) / FS   # onset -> peak
        rising_edge_s = (offset_idx - p_idx) / FS   # peak  -> offset

        ir_dc_mean_5s = float(df["ir_dc_mean_5s"].iloc[p_idx]) if "ir_dc_mean_5s" in df.columns else float("nan")

        beats.append(
            BeatFeature(
                onset_idx=onset_idx,
                peak_idx=p_idx,
                offset_idx=offset_idx,
                onset_time=onset_time,
                peak_time=peak_time,
                offset_time=offset_time,
                falling_edge_s=falling_edge_s,
                rising_edge_s=rising_edge_s,
                ir_dc_mean_5s=ir_dc_mean_5s,
            )
        )

    return beats


def build_feature_dataframe(
    beats: List[BeatFeature],
    df: pd.DataFrame | None = None,
    clinical_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert list of BeatFeature objects into a labeled feature DataFrame. Optionally adds spo2_pct, is_airway_rescue, cumulative_sashb from clinical_df."""
    base_cols = [
        "beat_index",
        "onset_time",
        "peak_time",
        "offset_time",
        "falling_edge_s",
        "rising_edge_s",
        "ir_dc_mean_5s",
    ]
    if not beats:
        cols = base_cols + (["spo2_pct", "is_airway_rescue", "cumulative_sashb"] if clinical_df is not None else [])
        return pd.DataFrame(columns=cols)

    records = []
    for i, b in enumerate(beats):
        rec = {
            "beat_index": i,
            "onset_time": b.onset_time,
            "peak_time": b.peak_time,
            "offset_time": b.offset_time,
            "falling_edge_s": b.falling_edge_s,
            "rising_edge_s": b.rising_edge_s,
            "ir_dc_mean_5s": b.ir_dc_mean_5s,
        }
        if clinical_df is not None:
            if df is not None and b.peak_idx < len(clinical_df):
                rec["spo2_pct"] = float(clinical_df["spo2_pct"].iloc[b.peak_idx])
                rec["is_airway_rescue"] = int(clinical_df["is_airway_rescue"].iloc[b.peak_idx])
                rec["cumulative_sashb"] = float(clinical_df["cumulative_sashb"].iloc[b.peak_idx])
            else:
                rec["spo2_pct"] = np.nan
                rec["is_airway_rescue"] = 0
                rec["cumulative_sashb"] = 0.0
        records.append(rec)
    df_feat = pd.DataFrame.from_records(records)
    df_feat = df_feat.sort_values("peak_time").reset_index(drop=True)
    return df_feat


# --- Clinical Biometric Suite ---

DT_50HZ = 1.0 / FS  # 0.02 s
SPO2_WINDOW_SAMPLES = int(3.0 * FS)  # 3 s for stable R
AIRWAY_RESCUE_WINDOW_MS = 500
AIRWAY_RESCUE_WINDOW_SAMPLES = int(AIRWAY_RESCUE_WINDOW_MS / 1000.0 * FS)  # 25 at 50 Hz
AIRWAY_RESCUE_THRESHOLD = -0.15  # -15% drop
SPO2_DIP_THRESHOLD = 90.0


class ClinicalBiometricSuite:
    """
    Clinical-grade biometric processing on 50 Hz PPG data.

    Implements:
    - SpO2 empirical curve: R = (Red_AC/Red_DC)/(IR_AC/IR_DC), SpO2 = 110 - 25*R, clamp 60-100%
    - Airway Rescue: IR DC drop exceeding -15% within 500 ms (hemodynamic occlusion)
    - SASHB: Area under curve for SpO2 < 90%, cumulative over time
    """

    def __init__(
        self,
        spo2_window_samples: int = SPO2_WINDOW_SAMPLES,
        rescue_window_samples: int = AIRWAY_RESCUE_WINDOW_SAMPLES,
        rescue_threshold: float = AIRWAY_RESCUE_THRESHOLD,
        spo2_dip_threshold: float = SPO2_DIP_THRESHOLD,
        dt_s: float = DT_50HZ,
    ):
        self.spo2_window_samples = spo2_window_samples
        self.rescue_window_samples = rescue_window_samples
        self.rescue_threshold = rescue_threshold
        self.spo2_dip_threshold = spo2_dip_threshold
        self.dt_s = dt_s

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process df_resampled (50 Hz) and return copy with spo2_pct, is_airway_rescue, cumulative_sashb.
        """
        result = pd.DataFrame(index=df.index)
        result["spo2_pct"] = self._compute_spo2(df)
        result["is_airway_rescue"] = self._compute_airway_rescue(df)
        result["cumulative_sashb"] = self._compute_sashb(result["spo2_pct"])
        return result

    def _compute_spo2(self, df: pd.DataFrame) -> pd.Series:
        """SpO2 = 110 - 25*R, R = (Red_AC/Red_DC)/(IR_AC/IR_DC), clamped 60-100%."""
        if "red_dc" not in df.columns or "red_ac" not in df.columns:
            return pd.Series(np.nan, index=df.index)

        red_dc = df["red_dc"].astype(float)
        red_ac = df["red_ac"].astype(float)
        ir_dc = df["ir_dc"].astype(float)
        ir_ac = df["ir_ac"].astype(float)

        # Rolling RMS of AC and mean DC over spo2_window_samples
        eps = 1e-9
        red_dc_roll = red_dc.rolling(window=self.spo2_window_samples, center=True, min_periods=1).mean()
        ir_dc_roll = ir_dc.rolling(window=self.spo2_window_samples, center=True, min_periods=1).mean()
        red_ac_rms = (red_ac**2).rolling(window=self.spo2_window_samples, center=True, min_periods=1).mean() ** 0.5
        ir_ac_rms = (ir_ac**2).rolling(window=self.spo2_window_samples, center=True, min_periods=1).mean() ** 0.5

        # R = (Red_AC/Red_DC) / (IR_AC/IR_DC)
        red_ratio = red_ac_rms / (red_dc_roll + eps)
        ir_ratio = ir_ac_rms / (ir_dc_roll + eps)
        r = np.where(ir_ratio > eps, red_ratio / ir_ratio, np.nan)

        # Oralable calibration: SpO2 = 110 - 25*R, clamp 60-100
        spo2 = 110.0 - 25.0 * r
        spo2 = np.clip(spo2, 60.0, 100.0)
        spo2 = np.where(np.isfinite(spo2), spo2, np.nan)
        return pd.Series(spo2, index=df.index)

    def _compute_airway_rescue(self, df: pd.DataFrame) -> pd.Series:
        """Rescue event: IR DC drop exceeding -15% within 500 ms window."""
        if "ir_dc" not in df.columns:
            return pd.Series(0, index=df.index)

        ir_dc = df["ir_dc"].astype(float)
        n = len(ir_dc)
        is_rescue = np.zeros(n, dtype=int)

        w = self.rescue_window_samples
        if w < 2 or n < w:
            return pd.Series(is_rescue, index=df.index)

        for i in range(w, n):
            start_val = ir_dc.iloc[i - w]
            end_val = ir_dc.iloc[i]
            if np.isfinite(start_val) and start_val > 1e-9:
                pct_change = (end_val - start_val) / start_val
                if pct_change <= self.rescue_threshold:
                    is_rescue[i] = 1

        return pd.Series(is_rescue, index=df.index)

    def _compute_sashb(self, spo2_pct: pd.Series) -> pd.Series:
        """Cumulative SASHB: sum of (threshold - SpO2) * dt when SpO2 < 90%."""
        cum = np.zeros(len(spo2_pct))
        for i in range(len(spo2_pct)):
            s = spo2_pct.iloc[i]
            if np.isfinite(s) and s < self.spo2_dip_threshold:
                cum[i] = (self.spo2_dip_threshold - s) * self.dt_s
            if i > 0:
                cum[i] += cum[i - 1]
        return pd.Series(cum, index=spo2_pct.index)


# --- 5-second window biomarkers ---

WINDOW_S = 5.0
SAMPLES_PER_WINDOW = int(WINDOW_S * FS)  # 250 at 50 Hz


def calculate_hemodynamic_occlusion(ir_dc: np.ndarray | pd.Series) -> float:
    """
    Measure the percentage drop in the IR-PPG DC baseline (hemodynamic occlusion).

    Baseline = mean of first 10% of samples; trough = minimum in window.
    Returns (baseline - trough) / baseline * 100 as percentage drop.
    Positive value indicates occlusion (blood flow restriction during clench).
    """
    arr = np.asarray(ir_dc, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    if arr.size < 10:
        return float("nan")
    ref_len = max(1, int(0.1 * arr.size))
    baseline = np.mean(arr[:ref_len])
    if baseline < 1e-9:
        return float("nan")
    trough = np.min(arr)
    pct_drop = (baseline - trough) / baseline * 100.0
    return float(pct_drop)


def calculate_grind_jitter(accel_z: np.ndarray | pd.Series, fs: float = 100.0) -> float:
    """
    Use Accelerometer Z-axis variance (100 Hz) to detect rhythmic "shudder" of tooth grinding.

    Grinding produces characteristic high-frequency vibration. Returns variance of accel_z
    (or band-pass filtered segment 5-25 Hz to isolate grind signature). Higher = more grind-like.
    """
    arr = np.asarray(accel_z, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    if arr.size < int(0.5 * fs):  # need at least 0.5 s
        return float("nan")
    # Band-pass 5-25 Hz to isolate grind signature (rhythmic shudder)
    b_bp, a_bp = _butter_bandpass(5.0, 25.0, fs, order=4)
    filtered = filtfilt(b_bp, a_bp, arr - np.mean(arr))
    return float(np.var(filtered))


def _ir_dc_shift_5s(ir_dc: np.ndarray) -> float:
    """
    Average drop in IR baseline over a 5s window.
    Baseline = mean of first 1s; drop = baseline - mean(ir_dc over full window).
    Positive value = baseline drops (occlusion).
    """
    if ir_dc.size < 10:
        return float("nan")
    ref_samples = min(int(1.0 * FS), ir_dc.size)  # first 1 s
    baseline = np.nanmean(ir_dc[:ref_samples])
    window_mean = np.nanmean(ir_dc)
    return float(baseline - window_mean)


def _rise_fall_symmetry_5s(beats: List[BeatFeature], t_start: pd.Timestamp, t_end: pd.Timestamp) -> float:
    """
    Rise/fall symmetry: ratio of reperfusion time (rising edge) to drop time (falling edge).
    Mean over beats whose peak falls in [t_start, t_end).
    """
    ratios = []
    for b in beats:
        if t_start <= b.peak_time < t_end and b.falling_edge_s > 0:
            ratios.append(b.rising_edge_s / b.falling_edge_s)
    if not ratios:
        return float("nan")
    return float(np.nanmean(ratios))


def _hrv_svd_5s(beats: List[BeatFeature], t_start: pd.Timestamp, t_end: pd.Timestamp) -> tuple[float, float]:
    """
    SVD of HRV (RR intervals) in the window. Gold-standard biomarker for separating
    sleep bruxism from simple arousals (2025).
    Returns (leading singular value s1, s1/s2 ratio). Uses delay-embedding of RR intervals.
    """
    # Peak times in window; include one prior and one after for complete RR intervals
    peak_times = sorted([b.peak_time for b in beats])
    in_window = [t for t in peak_times if t_start <= t < t_end]
    if len(in_window) < 2:
        return float("nan"), float("nan")
    prev = [t for t in peak_times if t < t_start]
    nxt = [t for t in peak_times if t >= t_end]
    if prev:
        in_window = [max(prev)] + in_window
    if nxt:
        in_window = in_window + [min(nxt)]
    rr = np.diff([t.timestamp() for t in in_window])  # RR in seconds
    if len(rr) < 3:
        return float("nan"), float("nan")
    # Delay embedding: rows = [RR_i, RR_{i+1}, RR_{i+2}], embedding dim 3
    emb_dim = min(3, len(rr) - 1)
    n_rows = len(rr) - emb_dim
    if n_rows < 1:
        return float("nan"), float("nan")
    M = np.array([rr[i : i + emb_dim] for i in range(n_rows)], dtype=float)
    try:
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    s1 = float(s[0])
    s2 = float(s[1]) if len(s) > 1 and s[1] > 1e-10 else float("nan")
    ratio = s1 / s2 if np.isfinite(s2) and s2 > 1e-10 else float("nan")
    return s1, ratio


def compute_window_biomarkers(
    df: pd.DataFrame,
    beats: List[BeatFeature],
    clinical_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    For every 5-second window in the 50 Hz data, compute:
    - ir_dc_shift_5s: average drop in IR baseline
    - rise_fall_symmetry_5s: mean ratio (reperfusion time / drop time)
    - hrv_svd_s1_5s: leading singular value of HRV (RR-interval SVD)
    - hrv_svd_s1_s2_ratio_5s: s1/s2 ratio (bruxism vs arousal biomarker)
    - spo2_pct, is_airway_rescue, cumulative_sashb: from ClinicalBiometricSuite (if clinical_df provided)
    """
    if "ir_dc" not in df.columns:
        raise ValueError("Column 'ir_dc' not found; run compute_filters() first.")

    index = df.index
    base_columns = [
        "window_start",
        "window_end",
        "ir_dc_shift_5s",
        "rise_fall_symmetry_5s",
        "hrv_svd_s1_5s",
        "hrv_svd_s1_s2_ratio_5s",
    ]
    if clinical_df is not None:
        base_columns.extend(["spo2_pct", "is_airway_rescue", "cumulative_sashb"])

    if len(index) < SAMPLES_PER_WINDOW:
        return pd.DataFrame(columns=base_columns)

    records = []
    n_windows = max(1, (len(index) - 1) // SAMPLES_PER_WINDOW)
    for k in range(n_windows):
        start_idx = k * SAMPLES_PER_WINDOW
        end_idx = min(start_idx + SAMPLES_PER_WINDOW, len(index))
        window_start = index[start_idx]
        window_end = index[end_idx - 1]
        ir_dc_win = df["ir_dc"].iloc[start_idx:end_idx].to_numpy()
        ir_shift = _ir_dc_shift_5s(ir_dc_win)
        sym = _rise_fall_symmetry_5s(beats, window_start, window_end)
        s1, s1_s2 = _hrv_svd_5s(beats, window_start, window_end)
        rec = {
            "window_start": window_start,
            "window_end": window_end,
            "ir_dc_shift_5s": ir_shift,
            "rise_fall_symmetry_5s": sym,
            "hrv_svd_s1_5s": s1,
            "hrv_svd_s1_s2_ratio_5s": s1_s2,
        }
        if clinical_df is not None:
            win_spo2 = clinical_df["spo2_pct"].iloc[start_idx:end_idx]
            win_rescue = clinical_df["is_airway_rescue"].iloc[start_idx:end_idx]
            rec["spo2_pct"] = float(np.nanmean(win_spo2)) if len(win_spo2) > 0 else np.nan
            rec["is_airway_rescue"] = int(win_rescue.max()) if len(win_rescue) > 0 else 0
            rec["cumulative_sashb"] = float(clinical_df["cumulative_sashb"].iloc[end_idx - 1])
        records.append(rec)
    return pd.DataFrame.from_records(records)


def extract_features(
    session_path: str | Path | None = None,
    out_path: str | Path | None = None,
    out_path_windows: str | Path | None = None,
    out_path_clinical: str | Path | None = None,
) -> pd.DataFrame:
    """
    High-level entry point:
    - Load 50 Hz session CSV.
    - Apply IR low-pass (<1 Hz) and Green band-pass (0.5–8 Hz).
    - Run ClinicalBiometricSuite (SpO2, airway rescue, SASHB) when red channel present.
    - Detect beats and compute onset/peak/offset, rising/falling edge durations.
    - Compute rolling 5 s mean IR DC baseline at each beat.
    - For every 5 s window: IR DC-shift, rise/fall symmetry, SVD of HRV, clinical metrics.
    - Save beat-level features to data/datasets/features_labeled.csv.
    - Save window-level biomarkers to data/datasets/features_windows_5s.csv.
    """
    df = load_session_50hz(session_path)
    df = compute_filters(df)

    clinical_df = None
    if "red" in df.columns:
        suite = ClinicalBiometricSuite()
        clinical_df = suite.process(df)

    beats = detect_beats_from_green_bp(df)
    feat_df = build_feature_dataframe(beats, df=df, clinical_df=clinical_df)
    window_df = compute_window_biomarkers(df, beats, clinical_df=clinical_df)

    base = Path(__file__).resolve().parents[2] / "data" / "datasets"
    if out_path is None:
        out_path = base / "features_labeled.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(out_path, index=False)

    if out_path_windows is None:
        out_path_windows = base / "features_windows_5s.csv"
    out_path_windows = Path(out_path_windows)
    window_df.to_csv(out_path_windows, index=False)

    if clinical_df is not None and out_path_clinical is not None:
        out_path_clinical = Path(out_path_clinical)
        out_path_clinical.parent.mkdir(parents=True, exist_ok=True)
        clinical_df.to_csv(out_path_clinical)

    return feat_df


if __name__ == "__main__":
    import sys

    session_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    out_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    base = Path(__file__).resolve().parents[2] / "data" / "datasets"
    features = extract_features(session_path=session_arg, out_path=out_arg)
    print(f"Extracted {len(features)} beats -> {out_arg or (base / 'features_labeled.csv')}")
    print(f"Window biomarkers -> {base / 'features_windows_5s.csv'}")
    print(features.head(10))

