#!/usr/bin/env python3
"""
Trim BLE log to keep only lines from the Nth 3-tap sync onward.
Creates a new log file suitable for validation (no segment flags needed).

Usage: python scripts/trim_log_to_sync.py input.txt output.txt --from-sync 2
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_timestamp_to_seconds(ts_str: str) -> float:
    """Parse HH:MM:SS.ffff to seconds since midnight."""
    try:
        from datetime import datetime
        dt = datetime.strptime(ts_str.strip(), "%H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(ts_str.strip(), "%H:%M:%S")
    return dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1e6


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input log (e.g. Oralable_6.txt)")
    parser.add_argument("output", type=Path, help="Output log (e.g. Oralable_7.txt)")
    parser.add_argument("--from-sync", type=int, default=2, help="Keep from Nth 3-tap sync (default 2)")
    args = parser.parse_args()

    from src.validation.self_validate import load_raw_ble_log
    from src.utils.sync_align import find_all_three_tap_anchors

    df = load_raw_ble_log(args.input)
    anchors = find_all_three_tap_anchors(df, max_count=args.from_sync)
    if len(anchors) < args.from_sync:
        print(f"Error: found only {len(anchors)} sync(s); need at least {args.from_sync}", file=sys.stderr)
        sys.exit(1)

    sync_time = anchors[args.from_sync - 1][1]
    t0 = df.index[0]
    sync_elapsed_s = (sync_time - t0).total_seconds()

    # Get first PPG timestamp from log to compute cutoff in log time
    from src.parser.log_parser import _read_log_rows, _parse_log_timestamp, PPG_CHAR_UUID

    df_log = _read_log_rows(args.input)
    first_ppg_ts_s = None
    for _, row in df_log.iterrows():
        line = str(row["Line"])
        if PPG_CHAR_UUID.lower() not in line.lower():
            continue
        if "Updated Value of Characteristic" in line and " to " in line:
            first_ppg_ts_s = _parse_log_timestamp(str(row["Timestamp"]))
            break
    if first_ppg_ts_s is None:
        print("Error: no PPG data in log", file=sys.stderr)
        sys.exit(1)

    cutoff_ts_s = first_ppg_ts_s + sync_elapsed_s
    print(f"2nd sync at {sync_elapsed_s:.2f}s from first PPG")
    print(f"Cutoff: keep lines with timestamp >= {cutoff_ts_s:.1f}s since midnight")

    # Read raw lines and filter
    lines = args.input.read_text(encoding="utf-8", errors="replace").splitlines()
    # nRF format: [HH:MM:SS.ffff] Level: Message  or  Level\tHH:MM:SS.mmm\tMessage
    bracket_re = re.compile(r"\[(\d{2}:\d{2}:\d{2}\.\d+)\]\s*\S+:\s*(.*)")

    kept = []
    for line in lines:
        ts_s = None
        m = bracket_re.match(line)
        if m:
            ts_str = m.group(1)
            ts_s = parse_timestamp_to_seconds(ts_str)
        else:
            parts = line.split("\t", 2)
            if len(parts) >= 3 and re.match(r"^\d{2}:\d{2}:\d{2}\.\d+$", parts[1].strip()):
                ts_s = parse_timestamp_to_seconds(parts[1].strip())
        if ts_s is not None and ts_s >= cutoff_ts_s:
            kept.append(line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"Wrote {len(kept)} lines to {args.output}")


if __name__ == "__main__":
    main()
