#!/usr/bin/env python3
"""
Generate a 3D interactive clinical infographic from temporalis gold-standard data.

Objective:
  3D scatter mapping Oxygen (SpO2), Motion Power, and Clench Force (IR-DC).

Axes:
  X: SpO2 saturation (85% to 100%)
  Y: Motion Power (accelerometer jitter magnitude)
  Z: Filtered IR-DC voltage (1.5V to 3.0V)

Cluster classification:
  quiet  (gray)   : low motion, stable baseline
  tonic  (blue)   : IR-DC drop > 15% with stable motion
  phasic (green)  : high motion power, variable IR-DC
  rescue (red)    : IR-DC drop > 15% during SpO2 < 92%

Visual:
  - Plotly dark aesthetic
  - Semi-transparent occlusion floor plane at IR-DC = 2.1V
  - Times New Roman typography

Outputs:
  - FIGURE_05.png (high-res)
  - OMG_CLINICAL_INFOGRAPHIC.html (interactive)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.overnight_states import (  # noqa: E402
    STATE_COLORS as _STATE_COLORS_FULL,
    enrich_overnight_frame,
    project_measurement_frame,
)

# 3D cluster view uses the four core classes (recovery folded into quiet for scatter).
STATE_COLORS = {k: _STATE_COLORS_FULL[k] for k in ("quiet", "tonic", "phasic", "rescue")}
MOTION_STABLE_THRESHOLD_G = 0.15
IR_DC_DROP_THRESHOLD_PCT = 15.0
RESCUE_SPO2_THRESHOLD = 92.0


def resolve_input_path(user_path: Path | None) -> Path:
    if user_path is not None:
        return user_path

    candidates = [
        ROOT / "TEMPORALIS_GOLD_STANDARD.csv",
        ROOT / "data" / "validation" / "TEMPORALIS_GOLD_STANDARD.csv",
        ROOT / "data" / "validation" / "GOLD_STANDARD_VALIDATION.csv",
        ROOT / "GOLD_STANDARD_VALIDATION.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No gold-standard CSV found. Expected one of: "
        "TEMPORALIS_GOLD_STANDARD.csv, data/validation/GOLD_STANDARD_VALIDATION.csv"
    )


def _looks_like_ble_raw_log(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(4):
                line = f.readline()
                if "Characteristic (" in line and "notified:" in line:
                    return True
    except Exception:
        return False
    return False


def _load_gold_like_frame(path: Path) -> pd.DataFrame:
    """
    Load either:
      - already-processed gold CSV with elapsed/ir/spo2 columns, or
      - BLE raw text log (misnamed .csv), then parse -> resample -> clinical features.
    """
    if _looks_like_ble_raw_log(path):
        sys.path.insert(0, str(ROOT))
        from src.analysis.features import ClinicalBiometricSuite, compute_filters
        from src.parser.log_parser import parse_accelerometer_log, parse_oralable_log
        from src.processing.resampler import resample_raw_to_50hz

        ppg = parse_oralable_log(path, channel_order="R_G_IR")
        if ppg is None or ppg.empty:
            raise ValueError(f"No PPG parsed from raw log: {path}")
        accel = parse_accelerometer_log(path)
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

        with tempfile.TemporaryDirectory(prefix="oralable_3d_") as td:
            td = Path(td)
            raw_path = td / "merged.csv"
            hz50_path = td / "resampled_50hz.csv"
            merged.to_csv(raw_path, index=False)
            resampled = resample_raw_to_50hz(raw_path=raw_path, out_path=hz50_path)
            resampled = compute_filters(resampled)
            clinical = ClinicalBiometricSuite().process(resampled)
            df = pd.concat([resampled, clinical], axis=1)
            delta = df.index - df.index[0]
            try:
                df["elapsed_s"] = np.asarray(delta.total_seconds(), dtype=float)
            except (AttributeError, TypeError):
                df["elapsed_s"] = np.arange(len(df), dtype=float) / 50.0
            return df

    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def prep_frame(csv_path: Path, max_points: int, validation_path: Path | None = None) -> pd.DataFrame:
    # Prefer already unified 50 Hz validation data when provided.
    if validation_path is not None and validation_path.exists():
        base_df = project_measurement_frame(_load_gold_like_frame(validation_path))
    else:
        base_df = project_measurement_frame(_load_gold_like_frame(csv_path))

    if validation_path is not None and validation_path.exists() and validation_path.resolve() != csv_path.resolve():
        val_df = project_measurement_frame(_load_gold_like_frame(validation_path))
        base_df = pd.merge_asof(
            base_df.sort_values("elapsed_s"),
            val_df.sort_values("elapsed_s"),
            on="elapsed_s",
            direction="nearest",
            tolerance=0.05,
            suffixes=("_a", "_b"),
        )
        merged = pd.DataFrame({"elapsed_s": base_df["elapsed_s"]})
        for c in ("ir_dc_raw", "spo2_pct", "accel_x", "accel_y", "accel_z"):
            merged[c] = base_df.get(f"{c}_b").combine_first(base_df.get(f"{c}_a"))
        if "label_name_b" in base_df.columns:
            merged["label_name"] = base_df["label_name_b"].combine_first(base_df.get("label_name_a"))
        elif "label_name_a" in base_df.columns:
            merged["label_name"] = base_df["label_name_a"]
        base_df = merged

    # Cluster plot: core 4 classes only (no recovery band).
    out = enrich_overnight_frame(base_df, apply_recovery=False, use_label_hints=True)
    out.loc[out["mam_state"] == "recovery", "mam_state"] = "quiet"

    if len(out) > max_points:
        step = int(np.ceil(len(out) / max_points))
        out = out.iloc[::step].reset_index(drop=True)

    return out


def plot_png(df: pd.DataFrame, out_png: Path, hide_states: set[str] | None = None) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    hide_states = {s.lower() for s in (hide_states or set())}
    plot_states = tuple(s for s in ("quiet", "tonic", "phasic", "rescue") if s not in hide_states)

    fig = plt.figure(figsize=(14, 8), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#0B0F14")
    ax.set_facecolor("#0B0F14")

    x = df["spo2_pct"].to_numpy()
    y = df["motion_power"].to_numpy()
    z = df["ir_dc_v_filtered"].to_numpy()
    s = df["mam_state"].to_numpy()
    visible = np.isin(s, plot_states)
    y_vis = y[visible] if np.any(visible) else y

    # Occlusion floor plane at IR-DC=2.1V.
    y_max = float(np.nanpercentile(y_vis, 99)) if len(y_vis) else 0.3
    y_max = max(0.1, y_max)
    xx, yy = np.meshgrid(np.linspace(85.0, 100.0, 2), np.linspace(0.0, y_max, 2))
    zz = np.full_like(xx, 2.1)
    ax.plot_surface(xx, yy, zz, color="#C9A227", alpha=0.18, linewidth=0)

    # Color-by-state 3D scatter.
    for state in plot_states:
        mask = s == state
        if np.any(mask):
            ax.scatter(
                x[mask], y[mask], z[mask],
                s=7,
                c=STATE_COLORS[state],
                alpha=0.9 if state != "quiet" else 0.65,
                label=state.capitalize(),
                depthshade=False,
            )

    ax.set_xlim(85.0, 100.0)
    ax.set_ylim(0.0, y_max)
    ax.set_zlim(1.5, 3.0)

    ax.set_xlabel("SpO2 Saturation (%)", color="white", labelpad=12)
    ax.set_ylabel("Motion Power (Accel jitter)", color="white", labelpad=12)
    ax.set_zlabel("IR-DC Voltage (Filtered)", color="white", labelpad=12)
    title = "OMG Clinical Infographic — 3D SpO2 / IR-DC / Time"
    if "quiet" in hide_states:
        title += " (events only)"
    ax.set_title(
        title,
        color="white",
        pad=18,
        fontsize=15,
    )

    ax.tick_params(colors="white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0.15)  # type: ignore[attr-defined]
        axis._axinfo["axisline"]["color"] = (1, 1, 1, 0.65)  # type: ignore[attr-defined]
        axis._axinfo["tick"]["color"] = (1, 1, 1, 0.8)  # type: ignore[attr-defined]

    state_labels = {
        "quiet": "Quiet",
        "tonic": "Tonic",
        "phasic": "Phasic",
        "rescue": "Airway Rescue",
    }
    legend_lines = [
        plt.Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=STATE_COLORS[state], markersize=7, label=state_labels[state],
        )
        for state in plot_states
    ] + [
        plt.Line2D([0], [0], color="#C9A227", lw=6, alpha=0.35, label="Occlusion Floor (2.1V)"),
    ]
    leg = ax.legend(handles=legend_lines, loc="upper left", framealpha=0.2, facecolor="#0B0F14", edgecolor="#CCCCCC")
    for txt in leg.get_texts():
        txt.set_color("white")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=320, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_html(df: pd.DataFrame, out_html: Path, hide_states: set[str] | None = None) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "plotly is required for HTML output. Install with: pip install plotly"
        ) from exc

    hide_states = {s.lower() for s in (hide_states or set())}
    plot_states = tuple(s for s in ("quiet", "tonic", "phasic", "rescue") if s not in hide_states)

    x = df["spo2_pct"].to_numpy()
    y = df["motion_power"].to_numpy()
    z = df["ir_dc_v_filtered"].to_numpy()
    s = df["mam_state"].to_numpy()
    t = df["elapsed_s"].to_numpy()
    visible = np.isin(s, plot_states)
    y_vis = y[visible] if np.any(visible) else y

    fig = go.Figure()

    # Add one trace per state for clean legend.
    for state in plot_states:
        mask = s == state
        if not np.any(mask):
            continue
        fig.add_trace(
            go.Scatter3d(
                x=x[mask],
                y=y[mask],
                z=z[mask],
                mode="markers",
                name="Airway Rescue" if state == "rescue" else state.capitalize(),
                text=[f"t={tt:.2f}s" for tt in t[mask]],
                hovertemplate=(
                    "State=%{fullData.name}<br>"
                    "SpO2=%{x:.2f}%<br>"
                    "Motion=%{y:.4f}<br>"
                    "IR-DC=%{z:.3f}V<br>"
                    "%{text}<extra></extra>"
                ),
                marker=dict(
                    color=STATE_COLORS[state],
                    size=3.5,
                    opacity=0.9 if state != "quiet" else 0.7,
                ),
            )
        )

    # Occlusion floor plane at z=2.1V.
    y_max = float(np.nanpercentile(y_vis, 99)) if len(y_vis) else 0.3
    y_max = max(0.1, y_max)
    xx = np.array([[85.0, 100.0], [85.0, 100.0]])
    yy = np.array([[0.0, 0.0], [y_max, y_max]])
    zz = np.full_like(xx, 2.1)
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            name="Occlusion Floor (2.1V)",
            showscale=False,
            opacity=0.2,
            colorscale=[[0, "#C9A227"], [1, "#C9A227"]],
        )
    )

    title = "OMG Clinical Infographic — 3D SpO2 / IR-DC / Time"
    if "quiet" in hide_states:
        title += " (events only)"
    fig.update_layout(
        template="plotly_dark",
        title=title,
        font=dict(family="Times New Roman", color="white", size=14),
        scene=dict(
            xaxis=dict(
                title="SpO2 Saturation (%)",
                range=[85.0, 100.0],
                backgroundcolor="#0B0F14",
                gridcolor="rgba(255,255,255,0.15)",
            ),
            yaxis=dict(
                title="Motion Power (Accel jitter)",
                range=[0.0, y_max],
                backgroundcolor="#0B0F14",
                gridcolor="rgba(255,255,255,0.15)",
            ),
            zaxis=dict(
                title="IR-DC Voltage (Filtered)",
                range=[1.5, 3.0],
                backgroundcolor="#0B0F14",
                gridcolor="rgba(255,255,255,0.15)",
            ),
            bgcolor="#0B0F14",
        ),
        paper_bgcolor="#0B0F14",
        plot_bgcolor="#0B0F14",
        legend=dict(bgcolor="rgba(0,0,0,0.25)"),
        margin=dict(l=10, r=10, t=55, b=10),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate 3D clinical infographic from temporalis gold data.")
    ap.add_argument("--input", type=Path, default=None, help="Gold CSV path (default: auto-resolve).")
    ap.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "data" / "validation" / "GOLD_STANDARD_VALIDATION.csv",
        help="Validation CSV used to unify/augment 50Hz frame.",
    )
    ap.add_argument(
        "--png-out",
        type=Path,
        default=ROOT / "FIGURE_05_CLUSTER_ANALYSIS.png",
        help="High-resolution PNG output path.",
    )
    ap.add_argument(
        "--html-out",
        type=Path,
        default=ROOT / "OMG_3D_CLINICAL_INFOGRAPHIC.html",
        help="Interactive HTML output path.",
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Downsample cap for plotting performance.",
    )
    ap.add_argument(
        "--hide-quiet",
        action="store_true",
        help="Omit quiet/baseline points; show tonic/phasic/rescue only.",
    )
    ap.add_argument(
        "--hide-states",
        type=str,
        default="",
        help="Comma-separated states to omit (quiet,tonic,phasic,rescue).",
    )
    args = ap.parse_args()

    hide_states: set[str] = set()
    if args.hide_quiet:
        hide_states.add("quiet")
    if args.hide_states.strip():
        hide_states.update(s.strip().lower() for s in args.hide_states.split(",") if s.strip())

    try:
        in_path = resolve_input_path(args.input)
        df = prep_frame(in_path, max_points=max(500, args.max_points), validation_path=args.validation)
        plot_png(df, args.png_out, hide_states=hide_states)
        plot_html(df, args.html_out, hide_states=hide_states)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    shown = len(df) if not hide_states else int((~df["mam_state"].isin(hide_states)).sum())
    print(f"Input : {in_path}")
    print(f"PNG   : {args.png_out}")
    print(f"HTML  : {args.html_out}")
    print(f"Rows  : {len(df)} (plotted {shown}; hidden={sorted(hide_states) or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
