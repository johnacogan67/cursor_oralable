#!/usr/bin/env python3
"""
One-shot: TEMPORALIS_RAW_01.csv → merged CSV → strict 50 Hz → mam_net_temporalis.h5 → Core ML in OralableCore.

Prereq: repo venv with requirements.txt (tensorflow, coremltools, …).

Usage (from cursor_oralable root):
  . .venv/bin/activate
  python scripts/run_temporalis_mam_pipeline.py
  python scripts/run_temporalis_mam_pipeline.py --raw path/to/log.csv --epochs 60
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Temporalis MAM: resample, train, convert to OralableCore")
    ap.add_argument("--raw", type=Path, default=ROOT / "TEMPORALIS_RAW_01.csv")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--channel-order", type=str, default=None)
    args = ap.parse_args()

    if not args.raw.exists():
        print(f"Raw log not found: {args.raw}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(ROOT))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mam_train", ROOT / "scripts" / "generate_mam_model.py"
    )
    assert spec and spec.loader
    _mam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mam)
    _build_merged_raw = _mam._build_merged_raw
    from src.processing.resampler import resample_raw_to_50hz

    data_dir = ROOT / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    merged_path = data_dir / "temporalis_merged_01.csv"
    hz50_path = data_dir / "temporalis_50hz_01.csv"
    h5_path = ROOT / "mam_net_temporalis.h5"

    merged = _build_merged_raw(args.raw, args.channel_order)
    merged.to_csv(merged_path, index=False)
    resample_raw_to_50hz(merged_path, hz50_path)
    print(f"50 Hz: {hz50_path} ({len(__import__('pandas').read_csv(hz50_path))} rows)")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_mam_model.py"),
        "--session-50hz",
        str(hz50_path),
        "-o",
        str(h5_path),
        "--epochs",
        str(args.epochs),
    ]
    if args.channel_order:
        cmd.extend(["--channel-order", args.channel_order])
    subprocess.check_call(cmd, cwd=str(ROOT))

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "convert_temporalis_mam.py"),
            "--keras",
            str(h5_path),
        ],
        cwd=str(ROOT),
    )
    print("Done. Updated OralableCore/.../BruxismMAM_Temporalis.mlpackage (if path matches sibling layout).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
