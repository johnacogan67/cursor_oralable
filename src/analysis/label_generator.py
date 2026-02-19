"""
Ground Truth label generator for Oralable training data.

Manually log start/end times for each activity to create labeled segments.
Timestamps are UTC (ISO) so sync_align.py can shift them to session time.

Protocol:
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
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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


if __name__ == "__main__":
    run_label_generator()
