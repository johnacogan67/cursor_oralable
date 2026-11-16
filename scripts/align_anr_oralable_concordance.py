#!/usr/bin/env python3
"""
Align ANR M40 EMG with Oralable Protocol A log; score bout concordance.

- Load Oralable BLE log → 50 Hz IR-DC + Protocol A label_enum + SpO2
- Load ANR EMG CSV (iso_utc,unix_s,emg_raw) → resample to 50 Hz
- Align on shared wall-clock (unix) or recording-start offsets
- Event F1: tonic/phasic label windows vs EMG activity bouts
- Lag histogram: EMG onset → IR-DC trough (expect ~1–5 s hemodynamic lag)
- SpO2 ∩ EMG nest (AcuPebble-style burden context; not AHI / not Bruxoff)

Outputs under plots/concordance/<session>/

Usage:
  .venv/bin/python scripts/align_anr_oralable_concordance.py \\
      --oralable data/raw/TEMPORALIS_RAW_....txt \\
      --anr data/raw/ANR_EMG_....txt
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.emg_spo2_nest import (  # noqa: E402
    DEFAULT_OVERLAP_TOL_S,
    desat_bouts,
    nest_markdown,
    nest_metrics,
)
from src.analysis.features import ClinicalBiometricSuite, compute_filters  # noqa: E402
from src.analysis.label_generator import apply_temporalis_labels_to_frame  # noqa: E402
from src.export.edf_writer import write_dual_a_edf  # noqa: E402
from src.parser.log_parser import parse_accelerometer_log, parse_oralable_log  # noqa: E402
from src.processing.resampler import resample_raw_to_50hz  # noqa: E402

FS = 50.0
# Protocol A active muscle windows (elapsed from recording start)
ACTIVE_LABELS = {2, 4, 7}  # tonic, phasic, tonic rescue — see label_generator
# label_enum from temporalis protocol: verify names
TONIC_PHASIC_ENUMS_FALLBACK = None  # filled after load if needed


def _load_anr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    cols = {c.lower().strip(): c for c in df.columns}
    if "unix_s" not in cols and "emg_raw" not in cols:
        # allow headerless: unix_s,emg_raw
        df = pd.read_csv(path, comment="#", header=None, names=["iso_utc", "unix_s", "emg_raw"])
    else:
        rename = {}
        for want in ("unix_s", "emg_raw", "iso_utc"):
            if want in cols:
                rename[cols[want]] = want
        df = df.rename(columns=rename)
    df = df.dropna(subset=["unix_s", "emg_raw"])
    df["unix_s"] = df["unix_s"].astype(float)
    df["emg_raw"] = df["emg_raw"].astype(float)
    return df.sort_values("unix_s").reset_index(drop=True)


def _build_oralable_50hz(log_path: Path, channel_order: str | None) -> pd.DataFrame:
    ppg = parse_oralable_log(log_path, channel_order=channel_order)
    if ppg is None or ppg.empty:
        raise ValueError(f"No PPG rows parsed from {log_path}")
    accel = parse_accelerometer_log(log_path)
    t0 = float(ppg["timestamp_s"].iloc[0])
    ppg = ppg.copy()
    ppg["timestamp_s"] = ppg["timestamp_s"] - t0
    if accel is None or accel.empty:
        accel = pd.DataFrame(
            {
                "timestamp_s": ppg["timestamp_s"],
                "temporalis_accel_x": 0.0,
                "temporalis_accel_y": 0.0,
                "temporalis_accel_z": 0.0,
            }
        )
    else:
        accel = accel.copy()
        accel["timestamp_s"] = accel["timestamp_s"] - t0
        accel = accel.rename(
            columns={
                "accel_x": "temporalis_accel_x",
                "accel_y": "temporalis_accel_y",
                "accel_z": "temporalis_accel_z",
            }
        )
    merged = pd.merge_asof(
        ppg.sort_values("timestamp_s"),
        accel.sort_values("timestamp_s"),
        on="timestamp_s",
        direction="nearest",
        tolerance=0.12,
    )
    # resample_raw_to_50hz expects a CSV path (not an in-memory DataFrame)
    with tempfile.TemporaryDirectory() as td:
        tmp_raw = Path(td) / "merged.csv"
        tmp_50 = Path(td) / "session_50hz.csv"
        merged.to_csv(tmp_raw, index=False)
        df50 = resample_raw_to_50hz(raw_path=tmp_raw, out_path=tmp_50)
    # Ensure DatetimeIndex for compute_filters rolling("5s")
    if not isinstance(df50.index, pd.DatetimeIndex):
        time_col = df50.columns[0]
        df50 = df50.copy()
        df50[time_col] = pd.to_datetime(df50[time_col], utc=False)
        df50 = df50.set_index(time_col)
    df50 = df50.copy()
    df50["elapsed_s"] = np.arange(len(df50), dtype=float) / FS
    df50 = compute_filters(df50)
    df50 = apply_temporalis_labels_to_frame(df50, elapsed_col="elapsed_s")
    return df50


def _resample_emg_to_grid(
    anr: pd.DataFrame,
    n_samples: int,
    anr_t0_unix: float | None = None,
    oral_t0_unix: float | None = None,
    emg_offset_s: float = 0.0,
) -> np.ndarray:
    """Linear interpolate EMG onto Oralable elapsed grid [0, n/FS).

    Dual Protocol A: both loggers start together → default align on first ANR sample
    (elapsed_anr → Oralable elapsed). Optional unix anchors if both known.
    """
    emg = anr["emg_raw"].to_numpy()
    unix = anr["unix_s"].to_numpy()
    if oral_t0_unix is not None and anr_t0_unix is not None:
        elapsed = unix - oral_t0_unix
    else:
        # Shared session start (dual Mac capture)
        t0 = float(anr_t0_unix) if anr_t0_unix is not None else float(unix[0])
        elapsed = unix - t0
    elapsed = elapsed + emg_offset_s
    grid = np.arange(n_samples, dtype=float) / FS
    if len(elapsed) < 2:
        return np.full(n_samples, np.nan)
    return np.interp(grid, elapsed, emg, left=np.nan, right=np.nan)


def _parse_pair_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _emg_active_mask(emg: np.ndarray, z_thresh: float = 1.5) -> np.ndarray:
    """Baseline = first 60 s rest; active when z-score of |emg - median| exceeds thresh."""
    x = np.asarray(emg, dtype=float)
    valid = np.isfinite(x)
    out = np.zeros(len(x), dtype=bool)
    if valid.sum() < FS * 10:
        return out
    n_base = int(min(60 * FS, valid.sum() // 2))
    base = x[valid][:n_base]
    med = float(np.nanmedian(base))
    mad = float(np.nanmedian(np.abs(base - med))) + 1e-6
    z = np.abs(x - med) / (1.4826 * mad)
    out = valid & (z >= z_thresh)
    return out


def _label_active_mask(df: pd.DataFrame) -> np.ndarray:
    enums = df["label_enum"].to_numpy()
    names = df["label_name"].astype(str).str.lower() if "label_name" in df.columns else None
    # Prefer name match for tonic / phasic / rescue
    if names is not None:
        m = names.str.contains("tonic|phasic|grind|clench|rescue", regex=True).to_numpy()
        if m.any():
            return m
    # Fallback: common temporalis enums for active muscle (2=max tonic, 4=phasic, 7=tonic rescue)
    return np.isin(enums, [2, 4, 7])


def _bout_edges(mask: np.ndarray, min_samples: int = 25) -> list[tuple[int, int]]:
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


def _event_f1(
    ref_bouts: list[tuple[int, int]],
    hyp_bouts: list[tuple[int, int]],
    tol_samples: int,
) -> dict:
    """Match hyp to ref if midpoints within tol; compute P/R/F1."""
    matched_ref = set()
    matched_hyp = set()
    for hi, (hs, he) in enumerate(hyp_bouts):
        hm = (hs + he) / 2
        for ri, (rs, re) in enumerate(ref_bouts):
            if ri in matched_ref:
                continue
            rm = (rs + re) / 2
            if abs(hm - rm) <= tol_samples:
                matched_ref.add(ri)
                matched_hyp.add(hi)
                break
    tp = len(matched_ref)
    fp = len(hyp_bouts) - len(matched_hyp)
    fn = len(ref_bouts) - len(matched_ref)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_ref_bouts": len(ref_bouts),
        "n_hyp_bouts": len(hyp_bouts),
        "tol_s": tol_samples / FS,
    }


def _emg_to_ir_trough_lags(
    emg_bouts: list[tuple[int, int]],
    ir_dc: np.ndarray,
    search_s: tuple[float, float] = (0.0, 8.0),
) -> list[tuple[int, int, float]]:
    """For each EMG bout: (emg_start_idx, trough_idx, lag_s) using LP IR-DC argmin in [0, 8] s."""
    out: list[tuple[int, int, float]] = []
    lo, hi = int(search_s[0] * FS), int(search_s[1] * FS)
    for es, _ee in emg_bouts:
        a = es + lo
        b = min(len(ir_dc) - 1, es + hi)
        if b <= a + 5:
            continue
        seg = ir_dc[a : b + 1]
        if not np.isfinite(seg).any():
            continue
        rel = int(np.nanargmin(seg))
        trough_i = a + rel
        out.append((es, trough_i, rel / FS))
    return out


def _lag_histogram(
    emg_bouts: list[tuple[int, int]],
    ir_dc: np.ndarray,
    search_s: tuple[float, float] = (0.0, 8.0),
) -> list[float]:
    """For each EMG bout start, lag to nearest IR-DC local minimum in [0, 8] s after."""
    return [lag for _es, _ti, lag in _emg_to_ir_trough_lags(emg_bouts, ir_dc, search_s)]


def _strong_emg_bouts(
    emg_bouts: list[tuple[int, int]],
    emg: np.ndarray,
    top_n: int = 3,
) -> list[tuple[int, int]]:
    """Rank EMG bouts by peak |emg| within the bout; return top_n."""
    scored: list[tuple[float, tuple[int, int]]] = []
    for a, b in emg_bouts:
        seg = emg[a : b + 1]
        if not np.isfinite(seg).any():
            continue
        peak = float(np.nanmax(np.abs(seg)))
        scored.append((peak, (a, b)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [bout for _p, bout in scored[:top_n]]


def _write_emg_ir_lag_zoom(
    out_path: Path,
    t: np.ndarray,
    ir_lp: np.ndarray,
    emg: np.ndarray,
    emg_bouts: list[tuple[int, int]],
    lag_rows: list[tuple[int, int, float]],
    session: str,
    pad_s: float = 4.0,
) -> None:
    """Top EMG bouts: dual-trace zoom with EMG onset + IR trough lag markers."""
    strong = _strong_emg_bouts(emg_bouts, emg, top_n=3)
    lag_by_start = {es: (ti, lag) for es, ti, lag in lag_rows}
    n_panels = max(1, len(strong))
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.8 * n_panels), sharex=False)
    if n_panels == 1:
        axes = [axes]
    if not strong:
        axes[0].text(0.5, 0.5, "No strong EMG bouts", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title(f"EMG↔IR lag zoom — {session}")
    for ax, (a, b) in zip(axes, strong):
        t0 = max(0.0, a / FS - pad_s)
        t1 = min(float(t[-1]) if len(t) else b / FS + pad_s, b / FS + pad_s + 8.0)
        m = (t >= t0) & (t <= t1)
        ax_ir = ax
        ax_emg = ax.twinx()
        ax_ir.plot(t[m], ir_lp[m], color="darkred", lw=1.0, label="LP IR-DC (<0.8 Hz)")
        ax_emg.plot(t[m], emg[m], color="steelblue", lw=0.8, alpha=0.85, label="EMG")
        ax_ir.axvspan(a / FS, b / FS, color="steelblue", alpha=0.12)
        ax_ir.axvline(a / FS, color="steelblue", ls="--", lw=0.9, label="EMG onset")
        if a in lag_by_start:
            ti, lag = lag_by_start[a]
            ax_ir.axvline(ti / FS, color="darkred", ls="--", lw=0.9, label=f"IR trough (+{lag:.2f}s)")
        ax_ir.set_ylabel("LP IR-DC", color="darkred")
        ax_emg.set_ylabel("EMG", color="steelblue")
        ax_ir.set_xlim(t0, t1)
        ax_ir.legend(loc="upper left", fontsize=8)
        ax.set_title(f"EMG bout [{a / FS:.1f}–{b / FS:.1f} s]")
    axes[-1].set_xlabel("Elapsed s")
    fig.suptitle(f"EMG→IR lag zoom (top bouts) — {session}", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(
    oralable: Path,
    anr: Path,
    out_dir: Path | None,
    channel_order: str | None,
    emg_z: float,
    tol_s: float,
    pair: Path | None = None,
    emg_offset_s: float = 0.0,
    write_edf: bool = True,
    include_raw_ppg: bool = False,
) -> Path:
    oral = _build_oralable_50hz(oralable, channel_order)
    anr_df = _load_anr(anr)
    n = len(oral)
    anr_t0 = float(anr_df["unix_s"].iloc[0])
    oral_t0: float | None = None
    if pair and pair.is_file():
        meta = _parse_pair_file(pair)
        if "anr_start_unix" in meta:
            anr_t0 = float(meta["anr_start_unix"])
        if "oralable_start_unix" in meta:
            oral_t0 = float(meta["oralable_start_unix"])
    emg = _resample_emg_to_grid(
        anr_df,
        n,
        anr_t0_unix=anr_t0,
        oral_t0_unix=oral_t0,
        emg_offset_s=emg_offset_s,
    )
    oral = oral.copy()
    oral["emg_raw"] = emg

    # SpO2 / SASHB via Mac ClinicalBiometricSuite (needs red_ac/red_dc from compute_filters)
    clinical = ClinicalBiometricSuite().process(oral)
    oral["spo2_pct"] = clinical["spo2_pct"].to_numpy()
    oral["cumulative_sashb"] = clinical["cumulative_sashb"].to_numpy()
    spo2 = oral["spo2_pct"].to_numpy(dtype=float)
    sashb = oral["cumulative_sashb"].to_numpy(dtype=float)
    sashb_total = float(sashb[-1]) if len(sashb) and np.isfinite(sashb[-1]) else None

    emg_act = _emg_active_mask(emg, z_thresh=emg_z)
    lab_act = _label_active_mask(oral)
    # LP IR-DC (Butterworth 0.8 Hz via compute_filters) — occlusion troughs, not raw IR
    ir = oral["ir_dc"].to_numpy(dtype=float)
    ir_roll = pd.Series(ir).rolling(int(5 * FS), min_periods=1, center=True).median().to_numpy()
    ir_drop = (ir_roll - ir) / (np.nanstd(ir) + 1e-9)
    ir_act = ir_drop >= 1.0

    emg_bouts = _bout_edges(emg_act)
    lab_bouts = _bout_edges(lab_act)
    ir_bouts = _bout_edges(ir_act, min_samples=int(0.5 * FS))
    spo2_desat = desat_bouts(spo2)

    tol_samp = int(tol_s * FS)
    f1_emg_vs_label = _event_f1(lab_bouts, emg_bouts, tol_samp)
    f1_ir_vs_emg = _event_f1(emg_bouts, ir_bouts, int(5.0 * FS))  # allow hemodynamic lag
    lag_rows = _emg_to_ir_trough_lags(emg_bouts, ir)
    lags = [lag for _es, _ti, lag in lag_rows]
    red_ac = oral["red_ac"].to_numpy(dtype=float) if "red_ac" in oral.columns else None
    ir_ac = oral["ir_ac"].to_numpy(dtype=float) if "ir_ac" in oral.columns else None
    nest = nest_metrics(
        spo2,
        emg_bouts,
        sashb_total=sashb_total,
        overlap_tol_s=DEFAULT_OVERLAP_TOL_S,
        red_ac=red_ac,
        ir_ac=ir_ac,
    )

    session = oralable.stem.replace("TEMPORALIS_RAW_", "")
    if out_dir is None:
        out_dir = ROOT / "plots" / "concordance" / session
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "session": session,
        "oralable": str(oralable),
        "anr": str(anr),
        "n_samples_50hz": n,
        "duration_s": n / FS,
        "emg_z_thresh": emg_z,
        "f1_emg_vs_protocol_labels": f1_emg_vs_label,
        "f1_ir_dc_vs_emg": f1_ir_vs_emg,
        "emg_to_ir_dc_lag_s": {
            "n": len(lags),
            "median": float(np.median(lags)) if lags else None,
            "mean": float(np.mean(lags)) if lags else None,
            "values": lags,
        },
        "spo2_emg_nest": nest,
        "note": (
            "ANR=research comparator; lag ~1–5 s expected; not PSG-AV diagnosis. "
            "SpO2∩EMG nest is AcuPebble-style oxygen-burden context (not AHI/ODI; not Bruxoff)."
        ),
    }
    edf_path: Path | None = None
    if write_edf:
        edf_path = out_dir / "session.edf"
        write_dual_a_edf(
            oral,
            edf_path,
            emg_bouts=emg_bouts,
            spo2_desat_bouts=spo2_desat,
            session_id=session,
            require_emg=True,
            include_raw_ppg=include_raw_ppg,
        )
        metrics["edf_path"] = str(edf_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "NEST.md").write_text(nest_markdown(nest, session), encoding="utf-8")
    oral[["elapsed_s", "ir_dc", "emg_raw", "spo2_pct", "cumulative_sashb", "label_enum"]].to_csv(
        out_dir / "aligned_50hz.csv", index=False
    )

    # Plot overlay: LP IR-DC · EMG · SpO2 · labels
    t = oral["elapsed_s"].to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(t, ir, color="darkred", lw=0.9, label="LP IR-DC (Butterworth <0.8 Hz)")
    for a, b in emg_bouts:
        axes[0].axvspan(a / FS, b / FS, color="steelblue", alpha=0.12)
    for es, ti, lag in lag_rows:
        axes[0].axvline(ti / FS, color="darkred", ls="--", lw=0.7, alpha=0.65)
    axes[0].set_ylabel("LP IR-DC")
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        f"ANR ↔ Oralable concordance + SpO₂ nest — {session} "
        "(dashed = IR trough after EMG bout)"
    )

    axes[1].plot(t, emg, color="steelblue", lw=0.9, label="EMG (resampled 50 Hz)")
    for a, b in emg_bouts:
        axes[1].axvspan(a / FS, b / FS, color="steelblue", alpha=0.15)
    axes[1].set_ylabel("EMG raw")
    axes[1].legend(loc="upper right")

    axes[2].plot(t, spo2, color="teal", lw=0.9, label="SpO2 %")
    axes[2].axhline(90.0, color="gray", ls="--", lw=0.7, label="90%")
    for a, b in spo2_desat:
        axes[2].axvspan(a / FS, b / FS, color="teal", alpha=0.18)
    axes[2].set_ylabel("SpO2 %")
    axes[2].set_ylim(70, 102)
    axes[2].legend(loc="upper right")

    axes[3].step(t, oral["label_enum"].to_numpy(), where="post", color="black", lw=0.8)
    for a, b in lab_bouts:
        axes[3].axvspan(a / FS, b / FS, color="orange", alpha=0.2)
    axes[3].set_ylabel("label_enum")
    axes[3].set_xlabel("Elapsed s (Oralable recording start)")
    fig.tight_layout()
    fig.savefig(out_dir / "overlay.png", dpi=140)
    plt.close(fig)

    _write_emg_ir_lag_zoom(
        out_dir / "emg_ir_lag_zoom.png",
        t,
        ir,
        emg,
        emg_bouts,
        lag_rows,
        session,
    )

    # Lag histogram
    fig2, ax = plt.subplots(figsize=(6, 4))
    if lags:
        ax.hist(lags, bins=np.arange(0, 8.5, 0.5), color="purple", edgecolor="white")
        ax.axvline(np.median(lags), color="black", ls="--", label=f"median={np.median(lags):.2f}s")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No lag samples", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("EMG onset → LP IR-DC trough lag (s)")
    ax.set_ylabel("Count")
    ax.set_title("Hemodynamic lag (expect ~1–5 s)")
    fig2.tight_layout()
    fig2.savefig(out_dir / "lag_histogram.png", dpi=140)
    plt.close(fig2)

    ir_gate_line = ""
    spo2_pair_line = ""
    if pair and pair.is_file():
        meta = _parse_pair_file(pair)
        if "ir_preflight_passed" in meta or "ir_preflight_drop_frac" in meta:
            passed = meta.get("ir_preflight_passed", "?")
            drop = meta.get("ir_preflight_drop_frac", "?")
            rest_m = meta.get("ir_preflight_rest_median", "?")
            ir_gate_line = (
                f"- IR optical preflight (pair file): passed={passed}, "
                f"drop_frac={drop}, rest_median={rest_m}\n"
            )
        if "spo2_preflight_ac_ok" in meta:
            spo2_pair_line = (
                f"- SpO₂ AC preflight (pair file): ac_ok={meta.get('spo2_preflight_ac_ok')}, "
                f"red_ac={meta.get('spo2_preflight_red_ac', '?')}, "
                f"ir_ac={meta.get('spo2_preflight_ir_ac', '?')}\n"
            )

    spo2_qc = nest.get("spo2_qc", "?")
    spo2_qc_reason = nest.get("spo2_qc_reason", "")
    spo2_qc_line = (
        f"- SpO₂ QC: **{spo2_qc}** — {spo2_qc_reason} "
        f"(mean={nest.get('spo2_mean')}, finite={nest.get('spo2_finite_frac')}, "
        f"in_band={nest.get('spo2_in_band_frac')}; not finger SpO₂)\n"
    )

    summary = out_dir / "SUMMARY.md"
    summary.write_text(
        f"""# Concordance — {session}

- Oralable: `{oralable}`
- ANR: `{anr}`
- Duration: {n / FS:.1f} s @ 50 Hz
{ir_gate_line}{spo2_pair_line}{spo2_qc_line}
## Event F1 (EMG bouts vs Protocol A tonic/phasic labels)

- F1 = **{f1_emg_vs_label['f1']:.3f}** (P={f1_emg_vs_label['precision']:.3f}, R={f1_emg_vs_label['recall']:.3f})
- tol = {tol_s} s · ref bouts={f1_emg_vs_label['n_ref_bouts']} · hyp={f1_emg_vs_label['n_hyp_bouts']}

## LP IR-DC vs EMG (wider tol for hemodynamic lag)

- Overlay / zoom use Butterworth LP IR-DC (&lt;0.8 Hz), not raw IR
- F1 = **{f1_ir_vs_emg['f1']:.3f}**
- Median EMG→LP IR-DC lag = **{(np.median(lags) if lags else float('nan')):.2f} s** (n={len(lags)})
- See `emg_ir_lag_zoom.png` for top EMG bouts + trough markers

**Claim discipline:** research comparator only — not PSG-AV SB diagnosis.

## EDF+

- `session.edf` — research Dual A (includes ANR EMG). Not PSG / not AHI.
"""
        + (f"\n- Path: `{edf_path}`\n" if edf_path else "\n- (skipped via --no-edf)\n"),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir}")
    print(f"  F1 EMG vs labels: {f1_emg_vs_label['f1']:.3f}")
    print(f"  median lag: {(np.median(lags) if lags else float('nan')):.2f} s")
    print(f"  SpO2 QC: {spo2_qc}")
    if edf_path:
        print(f"  EDF: {edf_path}")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="ANR ↔ Oralable concordance @ 50 Hz")
    ap.add_argument("--oralable", type=Path, required=True)
    ap.add_argument("--anr", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="Output dir (default plots/concordance/<session>)")
    ap.add_argument("--pair", type=Path, help="DUAL_PAIR_*.txt from dual session (unix anchors)")
    ap.add_argument("--emg-offset-s", type=float, default=0.0, help="Manual ANR shift (s)")
    ap.add_argument("--channel-order", type=str, default=None)
    ap.add_argument("--emg-z", type=float, default=1.5, help="EMG activity z-threshold")
    ap.add_argument("--tol-s", type=float, default=3.0, help="Bout midpoint match tolerance (s)")
    ap.add_argument(
        "--edf",
        dest="write_edf",
        action="store_true",
        default=True,
        help="Write session.edf (EDF+ with ANR EMG; default on)",
    )
    ap.add_argument(
        "--no-edf",
        dest="write_edf",
        action="store_false",
        help="Skip session.edf",
    )
    ap.add_argument(
        "--edf-raw-ppg",
        action="store_true",
        help="Include AC-scaled Red/Green/IR in session.edf (default: omit for EDFbrowser)",
    )
    args = ap.parse_args()

    if not args.oralable.is_file():
        print(f"Missing Oralable log: {args.oralable}", file=sys.stderr)
        return 1
    if not args.anr.is_file():
        print(f"Missing ANR log: {args.anr}", file=sys.stderr)
        return 1

    try:
        run(
            args.oralable,
            args.anr,
            args.out,
            args.channel_order,
            args.emg_z,
            args.tol_s,
            pair=args.pair,
            emg_offset_s=args.emg_offset_s,
            write_edf=args.write_edf,
            include_raw_ppg=args.edf_raw_ppg,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
