#!/usr/bin/env python3
"""
Generate GOLD_STANDARD_FILTER_PARITY.csv for Swift filter + SpO2 parity tests.

- Filter columns (green_bp, ir_dc): preserved from the committed Swift-matched gold
  when regenerating (pad/filtfilt edge behavior locked to that vector).
- SpO2 / rescue / SASHB columns: always from Mac ClinicalBiometricSuite (reference).

Usage:
    python scripts/generate_filter_parity_data.py [input_csv] [output_csv]
"""

from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    import numpy as np
    import pandas as pd
    from analysis.features import ClinicalBiometricSuite, compute_filters

    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "processed" / "ppg_50hz.csv"
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "data" / "validation" / "GOLD_STANDARD_FILTER_PARITY.csv"
    core_res = root.parent / "OralableCore" / "Tests" / "OralableCoreTests" / "Resources" / "GOLD_STANDARD_FILTER_PARITY.csv"

    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    ppg = pd.read_csv(input_path)
    ref = pd.Timestamp("1970-01-01 00:00:00")
    ppg["datetime"] = ref + pd.to_timedelta(ppg["timestamp_s"], unit="s")
    df = ppg.set_index("datetime")[["green", "red", "ir"]]
    df = df.resample("20ms").mean().interpolate(method="linear")
    df = compute_filters(df)
    clinical = ClinicalBiometricSuite().process(df)

    out = df.head(2000).reset_index()
    out["timestamp_s"] = (out["datetime"] - ref).dt.total_seconds()
    out["spo2_pct"] = clinical["spo2_pct"].to_numpy()[: len(out)]
    out["is_airway_rescue"] = clinical["is_airway_rescue"].to_numpy()[: len(out)]
    out["cumulative_sashb"] = clinical["cumulative_sashb"].to_numpy()[: len(out)]

    # Prefer committed Swift-matched filter vectors when present (same raw IR/green).
    prior = None
    for candidate in (core_res, output_path):
        if candidate.exists():
            try:
                prior = pd.read_csv(candidate)
                if {"green_bp_expected", "ir_dc_expected", "ir", "green"}.issubset(prior.columns):
                    if np.allclose(prior["ir"].to_numpy()[: len(out)], out["ir"].to_numpy(), equal_nan=True):
                        out["green_bp"] = prior["green_bp_expected"].to_numpy()[: len(out)]
                        out["ir_dc"] = prior["ir_dc_expected"].to_numpy()[: len(out)]
                        print(f"Preserved filter gold from {candidate}")
                        break
            except Exception as exc:
                print(f"Warning: could not load prior gold {candidate}: {exc}")

    result = out[
        [
            "timestamp_s",
            "green",
            "ir",
            "red",
            "green_bp",
            "ir_dc",
            "red_dc",
            "red_ac",
            "ir_ac",
            "spo2_pct",
            "is_airway_rescue",
            "cumulative_sashb",
        ]
    ].copy()
    result.columns = [
        "timestamp_s",
        "green",
        "ir",
        "red",
        "green_bp_expected",
        "ir_dc_expected",
        "red_dc_expected",
        "red_ac_expected",
        "ir_ac_expected",
        "spo2_pct_expected",
        "is_airway_rescue_expected",
        "cumulative_sashb_expected",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    if core_res.parent.exists():
        result.to_csv(core_res, index=False)
        print(f"Also wrote {core_res}")
    print(f"Created {output_path} with {len(result)} rows")


if __name__ == "__main__":
    main()
