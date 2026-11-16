"""
Oralable SpO2 ∩ ANR EMG nest helpers (Dual A / AcuPebble-style burden context).

Engineering phenotypes only:
- Desat events = SpO2 < 90% for ≥10 s (descriptive; not claimed ODI)
- EMG∩desat overlap within ±tol (default 5 s)
- SpO2 QC: ok | warn | fail (handoff visibility; does not abort align)
- No AHI claim; no Bruxoff/GrindCare equivalence.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

FS = 50.0
DESAT_SPO2_THRESHOLD = 90.0
MIN_DESAT_DURATION_S = 10.0
DEFAULT_OVERLAP_TOL_S = 5.0

# Post-hoc SpO2 QC (Paper A O2 handoff; Dual A muscle packs may archive with warn)
SPO2_IN_BAND_LO = 70.0
SPO2_IN_BAND_HI = 100.0
SPO2_WARN_MEAN_PCT = 92.0
SPO2_WARN_FRAC_BELOW_90 = 0.40
# Bandpass AC median |ac| below this → essentially flat (no perfusion)
AC_FLAT_ABS = 1.0


def bout_edges(mask: np.ndarray, min_samples: int) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) inclusive bouts of True mask."""
    m = np.asarray(mask, dtype=bool)
    bouts: list[tuple[int, int]] = []
    i = 0
    n = len(m)
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        if j - i >= min_samples:
            bouts.append((i, j - 1))
        i = j
    return bouts


def desat_mask(spo2: np.ndarray, threshold: float = DESAT_SPO2_THRESHOLD) -> np.ndarray:
    """True where SpO2 is finite and below threshold."""
    s = np.asarray(spo2, dtype=float)
    return np.isfinite(s) & (s < threshold)


def desat_bouts(
    spo2: np.ndarray,
    *,
    threshold: float = DESAT_SPO2_THRESHOLD,
    min_duration_s: float = MIN_DESAT_DURATION_S,
    fs: float = FS,
) -> list[tuple[int, int]]:
    """Contiguous SpO2 < threshold runs lasting ≥ min_duration_s."""
    min_samples = max(1, int(round(min_duration_s * fs)))
    return bout_edges(desat_mask(spo2, threshold=threshold), min_samples=min_samples)


def _intervals_overlap(
    a0: int,
    a1: int,
    b0: int,
    b1: int,
    tol_samples: int,
) -> bool:
    """Inclusive index intervals overlap when expanded by tol on each side."""
    a0e = a0 - tol_samples
    a1e = a1 + tol_samples
    return not (a1e < b0 or b1 < a0e)


def count_overlaps(
    left: Iterable[tuple[int, int]],
    right: Iterable[tuple[int, int]],
    tol_samples: int,
) -> tuple[int, int]:
    """
    Return (n_left_with_any_right, n_right_with_any_left).
    """
    left_list = list(left)
    right_list = list(right)
    left_hit = 0
    for ls, le in left_list:
        if any(_intervals_overlap(ls, le, rs, re, tol_samples) for rs, re in right_list):
            left_hit += 1
    right_hit = 0
    for rs, re in right_list:
        if any(_intervals_overlap(rs, re, ls, le, tol_samples) for ls, le in left_list):
            right_hit += 1
    return left_hit, right_hit


def _signal_rms(x: np.ndarray | None) -> float | None:
    """Overall RMS; used as red_ac_rms_median / ir_ac_rms_median amplitude proxy."""
    if x is None:
        return None
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(a))))


def spo2_qc_grade(
    *,
    spo2_mean: float | None,
    frac_below_90: float,
    red_ac_rms_median: float | None,
    ir_ac_rms_median: float | None,
) -> tuple[str, str]:
    """
    Return (spo2_qc, reason).
    fail = dead AC (junk SpO2); warn = temple bias / coupling; ok = otherwise.
    Does not abort align — handoff visibility only.
    """
    if red_ac_rms_median is not None and ir_ac_rms_median is not None:
        if red_ac_rms_median < AC_FLAT_ABS or ir_ac_rms_median < AC_FLAT_ABS:
            return (
                "fail",
                "red or IR AC essentially flat — SpO2 is junk (no perfusion)",
            )
    reasons: list[str] = []
    if spo2_mean is not None and spo2_mean < SPO2_WARN_MEAN_PCT:
        reasons.append(f"mean SpO2 {spo2_mean:.1f}% < {SPO2_WARN_MEAN_PCT:.0f}%")
    if frac_below_90 > SPO2_WARN_FRAC_BELOW_90:
        reasons.append(
            f"{frac_below_90:.0%} samples < 90% (>{SPO2_WARN_FRAC_BELOW_90:.0%} warn)"
        )
    if reasons:
        return "warn", "; ".join(reasons) + " — temple bias / coupling (not finger SpO2)"
    return "ok", "SpO2 series usable as engineering oxygen context"


def nest_metrics(
    spo2: np.ndarray,
    emg_bouts: list[tuple[int, int]],
    *,
    sashb_total: float | None = None,
    fs: float = FS,
    desat_threshold: float = DESAT_SPO2_THRESHOLD,
    min_desat_s: float = MIN_DESAT_DURATION_S,
    overlap_tol_s: float = DEFAULT_OVERLAP_TOL_S,
    red_ac: np.ndarray | None = None,
    ir_ac: np.ndarray | None = None,
) -> dict:
    """Build SpO2 ∩ EMG nest metrics dict for metrics.json / NEST.md."""
    s = np.asarray(spo2, dtype=float)
    n = len(s)
    duration_s = n / fs if n else 0.0
    duration_h = duration_s / 3600.0 if duration_s > 0 else 0.0

    finite_mask = np.isfinite(s)
    finite = s[finite_mask]
    spo2_finite_frac = float(finite_mask.mean()) if n else 0.0
    in_band_mask = finite_mask & (s >= SPO2_IN_BAND_LO) & (s <= SPO2_IN_BAND_HI)
    spo2_in_band_frac = float(in_band_mask.mean()) if n else 0.0
    below_90_mask = finite_mask & (s < DESAT_SPO2_THRESHOLD)
    frac_below_90 = float(below_90_mask.mean()) if n else 0.0

    spo2_mean = float(np.mean(finite)) if finite.size else None
    spo2_median = float(np.median(finite)) if finite.size else None
    spo2_min = float(np.min(finite)) if finite.size else None

    red_ac_med = _signal_rms(red_ac)
    ir_ac_med = _signal_rms(ir_ac)
    ac_present_frac: float | None = None
    if red_ac is not None and ir_ac is not None and n:
        ra = np.asarray(red_ac, dtype=float)
        ia = np.asarray(ir_ac, dtype=float)
        present = np.isfinite(ra) & np.isfinite(ia) & (np.abs(ra) >= AC_FLAT_ABS) & (
            np.abs(ia) >= AC_FLAT_ABS
        )
        ac_present_frac = float(present.mean())

    spo2_qc, spo2_qc_reason = spo2_qc_grade(
        spo2_mean=spo2_mean,
        frac_below_90=frac_below_90,
        red_ac_rms_median=red_ac_med,
        ir_ac_rms_median=ir_ac_med,
    )

    d_bouts = desat_bouts(
        s,
        threshold=desat_threshold,
        min_duration_s=min_desat_s,
        fs=fs,
    )
    tol_samp = int(round(overlap_tol_s * fs))
    emg_with_desat, desat_with_emg = count_overlaps(emg_bouts, d_bouts, tol_samp)

    n_emg = len(emg_bouts)
    n_desat = len(d_bouts)
    frac_emg = (emg_with_desat / n_emg) if n_emg else 0.0
    frac_desat = (desat_with_emg / n_desat) if n_desat else 0.0
    desat_per_h = (n_desat / duration_h) if duration_h > 0 else 0.0

    return {
        "spo2_mean": spo2_mean,
        "spo2_median": spo2_median,
        "spo2_min": spo2_min,
        "spo2_finite_frac": spo2_finite_frac,
        "spo2_in_band_frac": spo2_in_band_frac,
        "spo2_frac_below_90": frac_below_90,
        "red_ac_rms_median": red_ac_med,
        "ir_ac_rms_median": ir_ac_med,
        "ac_present_frac": ac_present_frac,
        "spo2_qc": spo2_qc,
        "spo2_qc_reason": spo2_qc_reason,
        "sashb_total": sashb_total,
        "desat_threshold_pct": desat_threshold,
        "desat_min_duration_s": min_desat_s,
        "desat_event_count": n_desat,
        "desat_events_per_hour": desat_per_h,
        "emg_bout_count": n_emg,
        "emg_bouts_with_desat": emg_with_desat,
        "desat_events_with_emg": desat_with_emg,
        "frac_emg_bouts_with_desat": frac_emg,
        "frac_desat_events_with_emg": frac_desat,
        "overlap_tol_s": overlap_tol_s,
        "note": (
            "SpO2∩EMG nest = AcuPebble-style oxygen-burden context + ANR timing. "
            "desat_events_per_hour is descriptive (not claimed ODI/AHI). "
            "spo2_qc is handoff visibility (ok|warn|fail); not finger SpO2 equivalence. "
            "Not Bruxoff/GrindCare equivalence; not PSG-AV diagnosis."
        ),
    }


def nest_markdown(metrics: dict, session: str) -> str:
    """Short NEST.md body with claim discipline."""
    def _fmt(v: object) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    return f"""# SpO₂ ∩ ANR EMG nest — {session}

**Engineering phenotype only.** AcuPebble-style oxygen-burden context nested with ANR EMG timing.
**Not** AHI/ODI equivalence. **Not** Bruxoff/GrindCare diagnosis. **Not** PSG-AV gold standard.
**Not** finger SpO₂. `spo2_qc` is Paper A O2 handoff visibility; Dual A muscle packs may archive with `warn`.

| Metric | Value |
|--------|-------|
| spo2_qc | **{_fmt(metrics.get("spo2_qc"))}** |
| spo2_qc_reason | {_fmt(metrics.get("spo2_qc_reason"))} |
| spo2_mean | {_fmt(metrics.get("spo2_mean"))} |
| spo2_median | {_fmt(metrics.get("spo2_median"))} |
| spo2_min | {_fmt(metrics.get("spo2_min"))} |
| spo2_finite_frac | {_fmt(metrics.get("spo2_finite_frac"))} |
| spo2_in_band_frac (70–100) | {_fmt(metrics.get("spo2_in_band_frac"))} |
| spo2_frac_below_90 | {_fmt(metrics.get("spo2_frac_below_90"))} |
| red_ac_rms_median | {_fmt(metrics.get("red_ac_rms_median"))} |
| ir_ac_rms_median | {_fmt(metrics.get("ir_ac_rms_median"))} |
| ac_present_frac | {_fmt(metrics.get("ac_present_frac"))} |
| sashb_total | {_fmt(metrics.get("sashb_total"))} |
| desat_event_count (SpO₂ < {metrics.get("desat_threshold_pct", 90)}% ≥ {metrics.get("desat_min_duration_s", 10)} s) | {_fmt(metrics.get("desat_event_count"))} |
| desat_events_per_hour (descriptive) | {_fmt(metrics.get("desat_events_per_hour"))} |
| emg_bout_count | {_fmt(metrics.get("emg_bout_count"))} |
| emg_bouts_with_desat (±{metrics.get("overlap_tol_s", 5)} s) | {_fmt(metrics.get("emg_bouts_with_desat"))} |
| frac_emg_bouts_with_desat | {_fmt(metrics.get("frac_emg_bouts_with_desat"))} |
| desat_events_with_emg | {_fmt(metrics.get("desat_events_with_emg"))} |
| frac_desat_events_with_emg | {_fmt(metrics.get("frac_desat_events_with_emg"))} |

AcuPebble remains Pedro’s AHI/ODI reference when available. Oralable SpO₂ burden nests beside ANR; it does not replace AcuPebble.
"""
