"""
Validation Dashboard for Oralable MAM self-test.

Loads BLE log + validation CSV, segments from Nth 3-tap sync, produces occlusion plots
and clinical quantifiers. Use JOHN_COGAN_2ND_SYNC_PROTOCOL.csv when segment_from_sync=2.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

ROOT = Path(__file__).resolve().parents[1]
FS = 50.0


def load_validation_log(validation_path: Path | None = None) -> tuple[Path, pd.DataFrame]:
    """Load validation CSV: validation_logs/JOHN_COGAN_*.csv, or GOLD_STANDARD_VALIDATION.csv."""
    if validation_path is not None and Path(validation_path).is_file():
        return Path(validation_path), pd.read_csv(validation_path)
    validation_dir = ROOT / "data" / "validation_logs"
    candidates = list(validation_dir.glob("JOHN_COGAN_*.csv")) if validation_dir.exists() else []
    if candidates:
        path = max(candidates, key=lambda p: p.stat().st_mtime)
        return path, pd.read_csv(path)
    gt_path = ROOT / "GOLD_STANDARD_VALIDATION.csv"
    if gt_path.exists():
        return gt_path, pd.read_csv(gt_path)
    raise FileNotFoundError("No validation_logs/JOHN_COGAN_*.csv or GOLD_STANDARD_VALIDATION.csv found")


def _parse_validation_phases(df_val: pd.DataFrame) -> dict:
    """Parse validation CSV into labeled phase intervals."""
    phases: dict[str, list[tuple[float, float]]] = {}
    pending: dict[str, float] = {}
    for _, row in df_val.iterrows():
        ev = str(row.get("Event", ""))
        t = float(row.get("ElapsedSeconds", 0))
        if ev.startswith("START:"):
            name = ev.replace("START:", "").strip()
            if "Clench" in name and "Apnea" not in name:
                pending["Tonic Clench"] = t
            elif "Grinding" in name:
                pending["Phasic Grind"] = t
            elif "Simulated Apnea" in name or ("Apnea" in name and "Simulated" in name):
                pending["Simulated Apnea"] = t
            elif name in ("Rest", "Baseline"):
                pending["Rest"] = t
            elif "Sync" in name or "Anchor" in name:
                pending["3-Tap Sync"] = t
        elif ev.startswith("END:"):
            name = ev.replace("END:", "").strip()
            if "Clench" in name and "Apnea" not in name and "Tonic Clench" in pending:
                s = pending.pop("Tonic Clench", None)
                if s is not None:
                    phases.setdefault("Tonic Clench", []).append((s, t))
            elif "Grinding" in name and "Phasic Grind" in pending:
                s = pending.pop("Phasic Grind", None)
                if s is not None:
                    phases.setdefault("Phasic Grind", []).append((s, t))
            elif ("Apnea" in name or "Simulated" in name) and "Simulated Apnea" in pending:
                s = pending.pop("Simulated Apnea", None)
                if s is not None:
                    phases.setdefault("Simulated Apnea", []).append((s, t))
            elif name in ("Rest", "Baseline") and "Rest" in pending:
                s = pending.pop("Rest", None)
                if s is not None:
                    phases.setdefault("Rest", []).append((s, t))
            elif ("Sync" in name or "Anchor" in name) and "3-Tap Sync" in pending:
                s = pending.pop("3-Tap Sync", None)
                if s is not None:
                    phases.setdefault("3-Tap Sync", []).append((s, t))
    return phases


def run_validation_dashboard(
    log_path: Path | None = None,
    validation_path: Path | None = None,
    output_path: Path | None = None,
    segment_from_sync: int = 2,
    root: Path | None = None,
) -> dict:
    """
    Run Validation Dashboard on segment from Nth 3-tap sync onward.
    segment_from_sync=2 uses 2nd sync as T=0 (matches self-validation protocol).
    Use validation_path=ROOT/"data"/"validation_logs"/"JOHN_COGAN_2ND_SYNC_PROTOCOL.csv" for 2nd sync.
    """
    from src.analysis.features import compute_filters, calculate_grind_jitter
    from src.utils.sync_align import find_all_three_tap_anchors
    from src.validation.self_validate import load_raw_ble_log
    import matplotlib.pyplot as plt

    root = root or ROOT
    raw_dir = root / "data" / "raw"
    val_dir = root / "data" / "validation_logs"
    # Prefer 1st-sync protocol when segment_from_sync=1
    if validation_path is None and segment_from_sync == 1:
        first_sync_protocol = val_dir / "JOHN_COGAN_1ST_SYNC_PROTOCOL.csv"
        if first_sync_protocol.exists():
            validation_path = first_sync_protocol
    if log_path is None:
        logs = sorted(raw_dir.glob("Oralable_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        log_path = logs[0] if logs else None
    if log_path is None or not Path(log_path).exists():
        raise FileNotFoundError("No raw BLE log found in data/raw/")

    df = load_raw_ble_log(log_path)
    df = compute_filters(df)

    anchors = find_all_three_tap_anchors(df, max_count=max(2, segment_from_sync))
    if len(anchors) < segment_from_sync:
        raise ValueError(
            f"Found only {len(anchors)} 3-tap sync(s); need at least {segment_from_sync} for segment_from_sync={segment_from_sync}"
        )
    anchor_idx, anchor_time = anchors[segment_from_sync - 1]
    start_s = (anchor_time - df.index[0]).total_seconds()
    elapsed_raw = (df.index - df.index[0]).total_seconds().to_numpy()
    mask = elapsed_raw >= start_s
    df = df.loc[mask].copy()
    ord_suffix = {1: "st", 2: "nd", 3: "rd"}.get(segment_from_sync, "th")
    print(f"Segment from {segment_from_sync}{ord_suffix} 3-tap sync at {start_s:.2f}s: {len(df)} rows ({len(df)/FS:.1f}s)")

    val_path, df_val = load_validation_log(validation_path)
    phases = _parse_validation_phases(df_val)

    elapsed_s = (df.index - anchor_time).total_seconds().to_numpy()
    sync_phases = phases.get("3-Tap Sync", [])
    t_val_sync = sync_phases[0][0] if sync_phases else 0.0
    elapsed_s = elapsed_s + t_val_sync

    ir_raw = df["ir"].to_numpy()
    ir_dc = df["ir_dc"].to_numpy()
    accel_z = df["accel_z"].to_numpy() if "accel_z" in df.columns else np.zeros(len(df))

    def occlusion_in_interval(start_s: float, end_s: float) -> float:
        m = (elapsed_s >= start_s) & (elapsed_s <= end_s)
        if not np.any(m):
            return float("nan")
        seg = ir_dc[m]
        ref_len = max(1, int(0.1 * len(seg)))
        baseline = np.nanmean(seg[:ref_len])
        if baseline < 1e-9:
            return float("nan")
        trough = np.nanmin(seg)
        return (baseline - trough) / baseline * 100.0

    segment_results: list[tuple[str, float]] = []
    for label in ["Tonic Clench", "Phasic Grind", "Rest", "Simulated Apnea"]:
        for start, end in phases.get(label, []):
            occ = occlusion_in_interval(start, end)
            if np.isfinite(occ):
                segment_results.append((label, occ))

    apnea_clench_mask = np.zeros(len(df), dtype=bool)
    for start, end in phases.get("Simulated Apnea", []):
        dur = end - start
        clench_start = end - min(15.0, dur * 0.4)
        m = (elapsed_s >= clench_start) & (elapsed_s <= end)
        apnea_clench_mask |= m

    tonic_occlusions = [occ for lbl, occ in segment_results if lbl == "Tonic Clench"]
    mean_occlusion_tonic = float(np.nanmean(tonic_occlusions)) if tonic_occlusions else float("nan")

    jitter_values = []
    for start, end in phases.get("Phasic Grind", []):
        m = (elapsed_s >= start) & (elapsed_s <= end)
        if np.sum(m) >= int(0.5 * FS):
            j = calculate_grind_jitter(accel_z[m], fs=FS)
            if np.isfinite(j):
                jitter_values.append(j)
    jitter_rms_phasic = float(np.sqrt(np.nanmean(jitter_values))) if jitter_values else float("nan")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ord_suffix = {1: "st", 2: "nd", 3: "rd"}.get(segment_from_sync, "th")
    fig.suptitle(
        f"Oralable MAM Validation Dashboard — from {segment_from_sync}{ord_suffix} 3-Tap Sync",
        fontsize=14,
        fontweight="bold",
    )

    ax_top.plot(elapsed_s, ir_raw, color="#4A90A4", linewidth=0.8, alpha=0.7, label="Raw IR")
    ax_top.plot(elapsed_s, ir_dc, color="#2E86AB", linewidth=1.5, label="DC Baseline (IR <1 Hz)")
    if np.any(apnea_clench_mask):
        ax_top.fill_between(
            elapsed_s,
            ir_raw.min(),
            ir_raw.max(),
            where=apnea_clench_mask,
            alpha=0.4,
            color="#E94F37",
            label="Apnea clench (end breath-hold)",
        )
    colors = {"Tonic Clench": "#E94F37", "Phasic Grind": "#F39C12", "Simulated Apnea": "#9B59B6", "Rest": "#95A5A6"}
    for label, intervals in phases.items():
        if label == "3-Tap Sync":
            continue
        c = colors.get(label, "#BDC3C7")
        for start, end in intervals:
            ax_top.axvspan(start, end, alpha=0.15, color=c)
    ax_top.set_ylabel("IR (raw ADC)")
    ax_top.set_title("Raw IR Signal with DC Baseline Overlay")
    ax_top.legend(loc="upper right")
    ax_top.grid(True, alpha=0.3)

    if segment_results:
        labels_uniq = [lbl for lbl, _ in segment_results]
        occs = [occ for _, occ in segment_results]
        x_pos = range(len(labels_uniq))
        bar_colors = [colors.get(lbl, "#3498DB") for lbl in labels_uniq]
        ax_bot.bar(x_pos, occs, color=bar_colors, edgecolor="black", linewidth=0.5)
        ax_bot.set_xticks(x_pos)
        ax_bot.set_xticklabels(labels_uniq, rotation=15, ha="right")
    ax_bot.set_ylabel("Occlusion Depth (%)")
    ax_bot.set_xlabel("Ground Truth Label")
    ax_bot.set_title("Occlusion Depth (%) vs Ground Truth Label")
    ax_bot.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax_bot.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path is None:
        output_path = root / "data" / "plots" / "validation_dashboard.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Validation Dashboard saved: {output_path}")
    print(f"  Mean Occlusion Depth (Tonic Clench): {mean_occlusion_tonic:.2f}%")
    print(f"  Jitter RMS Power (Phasic Grind):     {jitter_rms_phasic:.4f}")
    return {"mean_occlusion_tonic_clench": mean_occlusion_tonic, "jitter_rms_phasic_grind": jitter_rms_phasic}
