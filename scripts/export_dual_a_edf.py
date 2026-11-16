#!/usr/bin/env python3
"""Convenience CLI: Dual A Oralable + ANR → concordance pack including session.edf."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from align_anr_oralable_concordance import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export Dual A research EDF+ (includes ANR EMG). Not PSG / not AHI."
    )
    ap.add_argument("--oralable", type=Path, required=True)
    ap.add_argument("--anr", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pair", type=Path, default=None)
    ap.add_argument("--emg-offset-s", type=float, default=0.0)
    ap.add_argument("--channel-order", type=str, default=None)
    args = ap.parse_args()

    if not args.oralable.is_file() or not args.anr.is_file():
        print("Need valid --oralable and --anr paths", file=sys.stderr)
        return 1

    out = run(
        args.oralable,
        args.anr,
        args.out,
        args.channel_order,
        emg_z=1.5,
        tol_s=3.0,
        pair=args.pair,
        emg_offset_s=args.emg_offset_s,
        write_edf=True,
    )
    edf = out / "session.edf"
    print(f"EDF: {edf}" if edf.is_file() else f"Wrote pack (EDF missing): {out}")
    return 0 if edf.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
