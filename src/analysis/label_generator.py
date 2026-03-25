"""
Ground Truth label generator for Oralable training data.

Manually log start/end times for each activity to create labeled segments.
Timestamps are UTC (ISO) so sync_align.py can shift them to session time.

Temporalis (anterior) REV10 protocol: use automatic mode to assign label_enum from
docs/TEMPORALIS_COLLECTION_PROTOCOL.md Phase 2 offsets (elapsed time from recording start).

Protocol (legacy Masseter interactive):
  1. Resting (Laminar): 2 minutes quiet sitting
  2. Tonic Clenching (Stagnant): 10 sets of 5-second hard clenches
  3. Phasic Grinding (Oscillatory): 10 sets of 5-second rhythmic side-to-side jaw
  4. Artifacts: 2 min reading aloud (Talking), then drinking water (Swallowing)

Commands:
  r = Resting start/end
  c = Tonic Clench start/end
  g = Phasic Grind start/end
  t = Talking start/end
  s = Swallowing start/end
  q = Quit and save

Automatic Temporalis:
  python -m src.analysis.label_generator --temporalis-auto [--protocol PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL = """
=== Ground Truth Session Protocol ===

  1. Resting (Laminar): 2 min quiet sitting
  2. Tonic Clenching (Stagnant): 10 x 5 s hard clenches (teeth held together)
  3. Phasic Grinding (Oscillatory): 10 x 5 s rhythmic side-to-side jaw
  4. Artifacts:
     - Talking: 2 min reading aloud
     - Swallowing: drink water (mark start/end as needed)

Commands: r=Resting, c=Clench, g=Grind, t=Talking, s=Swallowing, q=Quit
Press the key at START and again at END of each segment.
"""

LABELS = {
    "r": "resting",
    "c": "tonic_clench",
    "g": "phasic_grind",
    "t": "talking",
    "s": "swallowing",
}

# Temporalis REV10 Phase 2: label_enum 0..9, names stable for training exports
TEMPORALIS_LABEL_NAMES: list[str] = [
    "rest_baseline",
    "sync_taps",
    "rest_post_sync",
    "tonic_max",
    "rest_hoi_recovery",
    "phasic_grinding",
    "rest_pre_apnea",
    "simulated_apnea",
    "tonic_rescue",
    "final_recovery",
]

# Fallback intervals (seconds from session start) — must match docs/TEMPORALIS_COLLECTION_PROTOCOL.md
TEMPORALIS_PROTOCOL_FALLBACK: list[tuple[float, float]] = [
    (0.0, 60.0),
    (60.0, 70.0),
    (70.0, 120.0),
    (120.0, 130.0),
    (130.0, 180.0),
    (180.0, 200.0),
    (200.0, 240.0),
    (240.0, 260.0),
    (260.0, 270.0),
    (270.0, 360.0),
]

DEFAULT_TEMPORALIS_PROTOCOL = (
    Path(__file__).resolve().parents[2] / "docs" / "TEMPORALIS_COLLECTION_PROTOCOL.md"
)


def _mmss_to_seconds(token: str) -> float:
    token = token.strip()
    parts = token.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Bad time token: {token!r}")


def _parse_time_range_cell(cell: str) -> tuple[float, float]:
    cell = cell.strip()
    for sep in ("–", "—", "-"):
        if sep in cell and cell.count(sep) >= 1:
            parts = re.split(r"\s*" + re.escape(sep) + r"\s*", cell, maxsplit=1)
            if len(parts) == 2:
                return _mmss_to_seconds(parts[0]), _mmss_to_seconds(parts[1])
    raise ValueError(f"Bad time range cell: {cell!r}")


def parse_temporalis_protocol_md(protocol_path: str | Path | None = None) -> list[dict]:
    """
    Read Phase 2 markdown table from TEMPORALIS_COLLECTION_PROTOCOL.md and return
    segment dicts: start_s, end_s, label_enum, label_name.

    On parse failure or row count mismatch, uses TEMPORALIS_PROTOCOL_FALLBACK + TEMPORALIS_LABEL_NAMES.
    """
    path = Path(protocol_path or DEFAULT_TEMPORALIS_PROTOCOL)
    if not path.exists():
        return _segments_from_fallback()

    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[tuple[float, float]] = []
    in_table = False
    for line in text.splitlines():
        line_stripped = line.strip()
        if "| Time offset" in line and "Action" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if not line_stripped.startswith("|"):
            if rows:
                break
            continue
        if line_stripped.startswith("|--") or "---" in line_stripped[:12]:
            continue
        cells = [c.strip() for c in line_stripped.split("|")]
        if len(cells) < 4:
            continue
        time_cell = cells[1]
        if not time_cell or time_cell.lower().startswith("time"):
            continue
        if not re.search(r"\d", time_cell):
            continue
        try:
            a, b = _parse_time_range_cell(time_cell)
        except (ValueError, IndexError):
            continue
        rows.append((a, b))
        if len(rows) >= len(TEMPORALIS_LABEL_NAMES):
            break

    if len(rows) != len(TEMPORALIS_LABEL_NAMES):
        return _segments_from_fallback()

    out: list[dict] = []
    for i, ((a, b), name) in enumerate(zip(rows, TEMPORALIS_LABEL_NAMES)):
        out.append({
            "start_s": float(a),
            "end_s": float(b),
            "label_enum": i,
            "label_name": name,
        })
    return out


def _segments_from_fallback() -> list[dict]:
    out: list[dict] = []
    for i, (a, b) in enumerate(TEMPORALIS_PROTOCOL_FALLBACK):
        out.append({
            "start_s": a,
            "end_s": b,
            "label_enum": i,
            "label_name": TEMPORALIS_LABEL_NAMES[i],
        })
    return out


def temporalis_label_for_elapsed(
    elapsed_s: np.ndarray | pd.Series,
    segments: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized label assignment: for each elapsed second from session start, return
    (label_enum, label_name_index as string array) for Temporalis protocol.

    Intervals are half-open [start_s, end_s) except unlabeled regions return -1 / empty.
    """
    if segments is None:
        segments = parse_temporalis_md_or_fallback()
    t = np.asarray(elapsed_s, dtype=float)
    enum_out = np.full(t.shape, -1, dtype=np.int16)
    name_out = np.empty(t.shape, dtype=object)
    name_out[:] = ""

    for seg in segments:
        lo, hi = seg["start_s"], seg["end_s"]
        m = (t >= lo) & (t < hi)
        enum_out[m] = seg["label_enum"]
        name_out[m] = seg["label_name"]

    return enum_out, name_out


def parse_temporalis_md_or_fallback(protocol_path: str | Path | None = None) -> list[dict]:
    """Prefer parsed MD; on any issue use fallback table."""
    try:
        segs = parse_temporalis_protocol_md(protocol_path)
        if len(segs) == len(TEMPORALIS_LABEL_NAMES):
            return segs
    except (OSError, ValueError, TypeError):
        pass
    return _segments_from_fallback()


def apply_temporalis_labels_to_frame(
    df: pd.DataFrame,
    elapsed_col: str = "elapsed_s",
    protocol_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add label_enum and label_name columns from elapsed time (seconds from session start)."""
    if elapsed_col not in df.columns:
        raise ValueError(f"DataFrame must have column {elapsed_col!r}")
    segments = parse_temporalis_md_or_fallback(protocol_path)
    e = df[elapsed_col].to_numpy()
    enums, names = temporalis_label_for_elapsed(e, segments=segments)
    out = df.copy()
    out["label_enum"] = enums
    out["label_name"] = names
    return out


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def run_label_generator(out_path: str | Path | None = None) -> None:
    """
    Interactive CLI: mark start/end of each activity. Segments are stored
    with start_time, end_time (UTC ISO), label, and duration_s.
    """
    if out_path is None:
        out_path = Path(__file__).resolve().parents[2] / "data" / "datasets" / "manual_labels.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(PROTOCOL)
    print("Start recording. Use same key for START then END of each segment.\n")

    segments: list[dict] = []
    pending: dict[str, datetime] = {}  # label -> start time

    while True:
        try:
            raw = input("Key (r/c/g/t/s/q): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted.")
            break
        if not raw:
            continue
        key = raw[0]

        if key == "q":
            print("Quit. Saving labels.")
            break

        if key not in LABELS:
            print(f"Unknown key '{key}'. Use r, c, g, t, s, or q.")
            continue

        label = LABELS[key]
        now = _now_utc()
        now_str = now.isoformat()

        if label not in pending:
            pending[label] = now
            print(f"  [{now_str}] START {label}")
        else:
            start = pending.pop(label)
            end = now
            dur = (end - start).total_seconds()
            segments.append({
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "label": label,
                "duration_s": round(dur, 3),
            })
            print(f"  [{end.isoformat()}] END {label} (duration {dur:.1f} s)")

    if not segments:
        print("No segments recorded. Nothing saved.")
        return

    df = pd.DataFrame(segments)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(segments)} segments to {out_path}")
    print(df.to_string(index=False))


def run_temporalis_automatic(
    out_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Emit a segment summary CSV (start_s, end_s, label_enum, label_name) from the
    Temporalis protocol document — no live session required.
    """
    if out_path is None:
        out_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "datasets"
            / "temporalis_protocol_segments.csv"
        )
    out_path = Path(out_path)
    segs = parse_temporalis_md_or_fallback(protocol_path)
    rows = []
    for s in segs:
        rows.append({
            "start_s": s["start_s"],
            "end_s": s["end_s"],
            "label_enum": s["label_enum"],
            "label_name": s["label_name"],
            "registration_site": "temporalis",
        })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} Temporalis protocol segments to {out_path}")
    return df


def _main() -> None:
    p = argparse.ArgumentParser(description="Ground truth label helper (interactive or Temporalis auto).")
    p.add_argument(
        "--temporalis-auto",
        action="store_true",
        help="Write segment table from docs/TEMPORALIS_COLLECTION_PROTOCOL.md (no interactive session).",
    )
    p.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="Path to TEMPORALIS_COLLECTION_PROTOCOL.md (default: docs/).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (auto mode default: data/datasets/temporalis_protocol_segments.csv).",
    )
    args = p.parse_args()
    if args.temporalis_auto:
        run_temporalis_automatic(out_path=args.out, protocol_path=args.protocol)
    else:
        run_label_generator(out_path=args.out)


if __name__ == "__main__":
    _main()
