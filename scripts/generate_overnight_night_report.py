#!/usr/bin/env python3
"""
Overnight Jaw/Oxygen Night Report — multi-panel pack for dentists and users.

Outputs under plots/overnight_report/<session_id>/:
  01_kpi_strip.png
  02_state_hypnogram.png
  03_hourly_burden.png
  04_smoking_gun_dual_rail.png
  05_event_table.csv
  night_report.pdf

Usage:
  .venv/bin/python scripts/generate_overnight_night_report.py
  .venv/bin/python scripts/generate_overnight_night_report.py \\
      --input data/validation/GOLD_STANDARD_VALIDATION.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.overnight_states import (  # noqa: E402
    STATE_COLORS,
    bouts_to_dataframe,
    compute_kpis,
    enrich_overnight_frame,
    extract_bouts,
    hourly_burden,
    project_measurement_frame,
)

FOOTER_NOTE = (
    "Device-inferred wellness states (quiet / tonic / phasic / rescue / recovery) — "
    "not a medical diagnosis. IR-DC drop, motion power, and SpO₂ gates; recovery = "
    "short settle after an event."
)


def resolve_input(path: Path | None) -> Path:
    if path is not None:
        return path
    candidates = [
        ROOT / "data" / "validation" / "GOLD_STANDARD_VALIDATION.csv",
        ROOT / "data" / "validation" / "TEMPORALIS_GOLD_STANDARD.csv",
        ROOT / "GOLD_STANDARD_VALIDATION.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No gold/validation CSV found. Pass --input.")


def session_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.upper().startswith("GOLD"):
        return "session_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return stem.replace(" ", "_")[:80]


def load_overnight_df(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, engine="python", on_bad_lines="skip")
    if "datetime" in raw.columns:
        raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
        raw = raw.sort_values("datetime").reset_index(drop=True)
    base = project_measurement_frame(raw)
    return enrich_overnight_frame(base, apply_recovery=True, use_label_hints=True)


def _fmt(v: float, nd: int = 1) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{nd}f}"


def plot_kpi_strip(kpis: dict[str, float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.8), dpi=160)
    fig.patch.set_facecolor("#0B0F14")
    ax.set_facecolor("#0B0F14")
    ax.axis("off")

    cards = [
        ("Wear", f"{_fmt(kpis['wear_min'], 1)} min"),
        ("Tonic", f"{_fmt(kpis['tonic_min'], 2)} min\nmax {_fmt(kpis['longest_tonic_s'], 0)} s"),
        ("Phasic", f"{_fmt(kpis['phasic_min'], 2)} min\n{int(kpis['phasic_bout_count'])} bouts"),
        ("Rescue", f"{int(kpis['rescue_count'])}×\n{_fmt(kpis['rescue_total_s'], 0)} s"),
        ("Recovery", f"med {_fmt(kpis['recovery_median_s'], 0)} s\nmax {_fmt(kpis['recovery_max_s'], 0)} s"),
        ("TFI", f"{_fmt(kpis['tfi'], 1)}"),
        ("SASHB", f"{_fmt(kpis['sashb'], 1)} %·s"),
        ("SpO₂", f"μ {_fmt(kpis['spo2_mean'], 1)}\nmin {_fmt(kpis['spo2_min'], 1)}"),
    ]
    n = len(cards)
    for i, (title, body) in enumerate(cards):
        x0 = i / n + 0.01
        w = 1 / n - 0.02
        ax.add_patch(
            Rectangle(
                (x0, 0.12),
                w,
                0.76,
                transform=ax.transAxes,
                facecolor="#161B22",
                edgecolor="#333A44",
                linewidth=1,
                clip_on=False,
            )
        )
        ax.text(
            x0 + w / 2,
            0.72,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#9AA4B2",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            x0 + w / 2,
            0.38,
            body,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=11,
        )

    ax.set_title("Overnight KPIs", color="white", fontsize=13, pad=8, loc="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_hypnogram(df: pd.DataFrame, out: Path) -> None:
    e = df["elapsed_s"].to_numpy(dtype=float)
    s = df["mam_state"].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 2.4), dpi=160)
    fig.patch.set_facecolor("#0B0F14")
    ax.set_facecolor("#0B0F14")

    order = ["quiet", "recovery", "phasic", "tonic", "rescue"]
    y_map = {st: i for i, st in enumerate(order)}
    i = 0
    n = len(s)
    while i < n:
        st = str(s[i])
        j = i + 1
        while j < n and s[j] == st:
            j += 1
        y = y_map.get(st, 0)
        ax.barh(
            y,
            e[j - 1] - e[i] + (e[1] - e[0] if n > 1 else 0.02),
            left=e[i],
            height=0.85,
            color=STATE_COLORS.get(st, "#888"),
            align="center",
            linewidth=0,
        )
        i = j

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o.capitalize() for o in order], color="white")
    ax.set_xlabel("Elapsed (s) from session start", color="white")
    ax.set_title("State hypnogram (jaw-load map)", color="white", loc="left")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333A44")
    ax.set_xlim(float(e[0]), float(e[-1]))
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_hourly_burden(hourly: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=160)
    fig.patch.set_facecolor("#0B0F14")
    ax.set_facecolor("#0B0F14")

    if hourly.empty:
        ax.text(0.5, 0.5, "No hourly data", ha="center", va="center", color="white", transform=ax.transAxes)
    else:
        x = hourly["hour_index"].to_numpy()
        bottom = np.zeros(len(hourly))
        stack_order = ("quiet", "recovery", "phasic", "tonic", "rescue")
        for st in stack_order:
            col = f"{st}_min"
            vals = hourly[col].to_numpy(dtype=float) if col in hourly.columns else np.zeros(len(hourly))
            ax.bar(x, vals, bottom=bottom, color=STATE_COLORS[st], width=0.8, label=st.capitalize())
            bottom = bottom + vals
        ax2 = ax.twinx()
        sashb = hourly["sashb_delta"].to_numpy(dtype=float)
        ax2.plot(x, sashb, color="#7FDBFF", marker="o", linewidth=1.8, label="SASHB Δ")
        ax2.set_ylabel("SASHB Δ (%·s)", color="#7FDBFF")
        ax2.tick_params(colors="#7FDBFF")
        ax.set_xlabel("Hour index", color="white")
        ax.set_ylabel("Minutes", color="white")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.25, facecolor="#0B0F14", labelcolor="white")

    ax.set_title("Hourly stacked burden + SASHB", color="white", loc="left")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333A44")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_smoking_gun(df: pd.DataFrame, out: Path, *, target_hz: float = 1.0) -> None:
    e = df["elapsed_s"].to_numpy(dtype=float)
    fs = 50.0
    if e.size >= 3:
        dt = np.diff(e)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            fs = 1.0 / float(np.median(dt))
    step = max(1, int(round(fs / target_hz)))
    d = df.iloc[::step].reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.5), dpi=160, sharex=True)
    fig.patch.set_facecolor("#0B0F14")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0B0F14")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333A44")

    te = d["elapsed_s"].to_numpy(dtype=float)
    ax1.plot(te, d["ir_dc_raw"].to_numpy(dtype=float), color="#C0392B", linewidth=0.8, label="IR-DC")
    for st, color in (("tonic", STATE_COLORS["tonic"]), ("phasic", STATE_COLORS["phasic"]), ("rescue", STATE_COLORS["rescue"])):
        mask = d["mam_state"].to_numpy() == st
        if np.any(mask):
            ax1.scatter(te[mask], d["ir_dc_raw"].to_numpy(dtype=float)[mask], s=8, c=color, alpha=0.85, label=st.capitalize())
    ax1.set_ylabel("IR-DC (raw)", color="white")
    ax1.set_title("Smoking-gun dual rail (events highlighted)", color="white", loc="left")
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.25, facecolor="#0B0F14", labelcolor="white", ncol=4)

    spo2 = d["spo2_pct"].to_numpy(dtype=float)
    ax2.plot(te, spo2, color="#5DADE2", linewidth=0.9, label="SpO₂")
    low = spo2 < 90.0
    if np.any(low):
        ax2.fill_between(te, spo2, 90.0, where=low, color="#5DADE2", alpha=0.25, interpolate=True, label="SASHB zone <90%")
    rescue = d["mam_state"].to_numpy() == "rescue"
    if np.any(rescue):
        ax2.scatter(te[rescue], spo2[rescue], s=18, c=STATE_COLORS["rescue"], marker="|", linewidths=1.5, label="Rescue")
    ax2.axhline(90.0, color="#888", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("SpO₂ (%)", color="white")
    ax2.set_xlabel("Elapsed (s)", color="white")
    ax2.set_ylim(84, 101)
    ax2.legend(loc="lower right", fontsize=7, framealpha=0.25, facecolor="#0B0F14", labelcolor="white")

    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _draw_image_page(pdf: PdfPages, path: Path, title: str) -> None:
    fig = plt.figure(figsize=(8.5, 11), dpi=120)
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.96, title, fontsize=12, fontweight="bold", va="top")
    if path.exists():
        img = plt.imread(str(path))
        ax = fig.add_axes([0.06, 0.18, 0.88, 0.74])
        ax.imshow(img)
        ax.axis("off")
    fig.text(0.08, 0.06, FOOTER_NOTE, fontsize=7, color="#444", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(
    pdf_path: Path,
    kpis: dict[str, float],
    paths: dict[str, Path],
    events: pd.DataFrame,
) -> None:
    with PdfPages(pdf_path) as pdf:
        # Page 1 — patient summary
        fig = plt.figure(figsize=(8.5, 11), dpi=120)
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.96, "Oralable — Overnight Jaw / Oxygen Summary", fontsize=14, fontweight="bold", va="top")
        fig.text(0.08, 0.92, "Patient morning view (wellness — not a diagnosis)", fontsize=9, color="#555", va="top")

        lines = [
            f"Wear time: {_fmt(kpis['wear_min'], 1)} min",
            f"Tonic load: {_fmt(kpis['tonic_min'], 2)} min (longest bout {_fmt(kpis['longest_tonic_s'], 0)} s)",
            f"Phasic activity: {_fmt(kpis['phasic_min'], 2)} min ({int(kpis['phasic_bout_count'])} bouts)",
            f"Airway rescue events: {int(kpis['rescue_count'])} ({_fmt(kpis['rescue_total_s'], 0)} s total)",
            f"Recovery after rescue: median {_fmt(kpis['recovery_median_s'], 0)} s / max {_fmt(kpis['recovery_max_s'], 0)} s",
            f"TFI: {_fmt(kpis['tfi'], 1)}   |   SASHB: {_fmt(kpis['sashb'], 1)} %·s",
            f"SpO₂ mean / min: {_fmt(kpis['spo2_mean'], 1)}% / {_fmt(kpis['spo2_min'], 1)}%",
        ]
        y = 0.86
        for line in lines:
            fig.text(0.08, y, line, fontsize=11, va="top", family="monospace")
            y -= 0.035

        if paths["hypnogram"].exists():
            ax = fig.add_axes([0.06, 0.22, 0.88, 0.32])
            ax.imshow(plt.imread(str(paths["hypnogram"])))
            ax.axis("off")
            fig.text(0.08, 0.55, "Your night — state map", fontsize=11, fontweight="bold")

        legend_handles = [
            Patch(facecolor=STATE_COLORS[s], label=s.capitalize())
            for s in ("quiet", "tonic", "phasic", "rescue", "recovery")
        ]
        fig.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.08, 0.10), ncol=5, fontsize=8, frameon=False)
        fig.text(0.08, 0.05, FOOTER_NOTE, fontsize=7, color="#444")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2 — dentist detail
        _draw_image_page(pdf, paths["kpi"], "Dentist detail — KPIs")
        _draw_image_page(pdf, paths["hourly"], "Dentist detail — hourly burden")
        _draw_image_page(pdf, paths["smoking"], "Dentist detail — smoking-gun dual rail")

        # Event table page
        fig = plt.figure(figsize=(8.5, 11), dpi=120)
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.96, "Event table (bouts)", fontsize=12, fontweight="bold", va="top")
        ax = fig.add_axes([0.06, 0.12, 0.88, 0.78])
        ax.axis("off")
        show = events.head(40) if not events.empty else pd.DataFrame(
            columns=["start_s", "end_s", "duration_s", "type", "min_spo2", "delta_spo2", "peak_motion", "recovery_s"]
        )
        if show.empty:
            ax.text(0.0, 0.9, "No event bouts detected.", fontsize=10, transform=ax.transAxes)
        else:
            cell = show.copy()
            for c in cell.columns:
                if cell[c].dtype.kind == "f":
                    cell[c] = cell[c].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "—")
            table = ax.table(
                cellText=cell.values.tolist(),
                colLabels=list(cell.columns),
                loc="upper center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1.0, 1.25)
        fig.text(0.08, 0.05, FOOTER_NOTE, fontsize=7, color="#444")
        pdf.savefig(fig)
        plt.close(fig)


def generate_report(input_path: Path, out_dir: Path) -> dict[str, Path]:
    df = load_overnight_df(input_path)
    bouts = extract_bouts(df)
    events = bouts_to_dataframe(bouts)
    kpis = compute_kpis(df, bouts)
    hourly = hourly_burden(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "kpi": out_dir / "01_kpi_strip.png",
        "hypnogram": out_dir / "02_state_hypnogram.png",
        "hourly": out_dir / "03_hourly_burden.png",
        "smoking": out_dir / "04_smoking_gun_dual_rail.png",
        "events": out_dir / "05_event_table.csv",
        "pdf": out_dir / "night_report.pdf",
    }

    plot_kpi_strip(kpis, paths["kpi"])
    plot_hypnogram(df, paths["hypnogram"])
    plot_hourly_burden(hourly, paths["hourly"])
    plot_smoking_gun(df, paths["smoking"])
    events.to_csv(paths["events"], index=False)
    write_pdf(paths["pdf"], kpis, paths, events)

    # Small JSON-ish sidecar for iOS / tooling parity
    summary_path = out_dir / "kpi_summary.txt"
    summary_lines = [f"{k}={v}" for k, v in kpis.items()]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate overnight night-report graphic pack.")
    ap.add_argument("--input", type=Path, default=None, help="Gold/validation CSV path.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: plots/overnight_report/<session_id>).",
    )
    ap.add_argument("--session-id", type=str, default=None, help="Folder name under plots/overnight_report/.")
    args = ap.parse_args()

    try:
        in_path = resolve_input(args.input)
        sid = args.session_id or session_id_from_path(in_path)
        out_dir = args.out_dir or (ROOT / "plots" / "overnight_report" / sid)
        paths = generate_report(in_path, out_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Input : {in_path}")
    print(f"Out   : {out_dir}")
    for k, p in paths.items():
        print(f"  {k:10s} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
