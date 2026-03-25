#!/usr/bin/env python3
"""
Generate GOLD_STANDARD_FILTER_PARITY.csv for Swift filter parity tests.

Runs the Python research pipeline (compute_filters from features.py) on 50 Hz PPG data
and exports raw + filtered columns. The Swift ParityTests load this CSV, run the Swift
filters, and assert output matches within 0.001.

Usage:
    python scripts/generate_filter_parity_data.py [input_csv] [output_csv]

Defaults:
    input:  data/processed/ppg_50hz.csv
    output: data/validation/GOLD_STANDARD_FILTER_PARITY.csv
"""

from pathlib import Path
import sys

def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    import pandas as pd
    from analysis.features import compute_filters

    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "processed" / "ppg_50hz.csv"
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "data" / "validation" / "GOLD_STANDARD_FILTER_PARITY.csv"

    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    ppg = pd.read_csv(input_path)
    ref = pd.Timestamp("1970-01-01 00:00:00")
    ppg["datetime"] = ref + pd.to_timedelta(ppg["timestamp_s"], unit="s")
    df = ppg.set_index("datetime")[["green", "red", "ir"]]
    df = df.resample("20ms").mean().interpolate(method="linear")
    df = compute_filters(df)

    out = df.head(2000).reset_index()
    out["timestamp_s"] = (out["datetime"] - ref).dt.total_seconds()
    out = out[["timestamp_s", "green", "ir", "red", "green_bp", "ir_dc"]]
    out.columns = ["timestamp_s", "green", "ir", "red", "green_bp_expected", "ir_dc_expected"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(out)} rows")


if __name__ == "__main__":
    main()
