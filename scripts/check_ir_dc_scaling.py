#!/usr/bin/env python3
"""
IR-DC scaling diagnostic: verify PPG raw range, tdm_interleave, and suggest coupling thresholds.

Run: python scripts/check_ir_dc_scaling.py [path/to/log.txt]
Default: data/raw/Oralable_6.txt

Checks:
1. Raw IR min/max/median with tdm_interleave=True vs False
2. IR-DC median after full pipeline (resample + filters)
3. First packet raw values for byte-order verification
4. Suggested IR_DC_RAW_MIN/MAX for self_validate.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


def main():
    raw_dir = ROOT / "data" / "raw"
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else raw_dir / "Oralable_6.txt"
    if not log_path.exists():
        log_path = raw_dir / "Oralable_6.txt"
    if not log_path.exists():
        candidates = sorted(raw_dir.glob("Oralable*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            log_path = candidates[0]
        else:
            print("No log file found. Usage: python scripts/check_ir_dc_scaling.py [path/to/log.txt]")
            sys.exit(1)

    print("=" * 60)
    print("IR-DC Scaling Diagnostic")
    print("=" * 60)
    print(f"Log: {log_path}")
    print()

    from src.parser.log_parser import (
        parse_oralable_log,
        _parse_hex_payload,
        _extract_hex_from_line,
        PPG_CHAR_UUID,
        _read_log_rows,
        _iter_characteristic_payloads,
    )

    # 1. Parse with both tdm modes and channel orders (R_IR_G vs R_G_IR)
    print("1. Raw IR stats by tdm_interleave and channel_order:")
    for order in ["R_IR_G", "R_G_IR"]:
        for tdm in [False, True]:
            df = parse_oralable_log(log_path, tdm_interleave=tdm, channel_order=order)
            if df.empty:
                continue
            ir = df["ir"].astype(float)
            label = f"order={order}, tdm={tdm}"
            print(f"  {label}: min={ir.min():,.0f}, max={ir.max():,.0f}, median={ir.median():,.0f}, n={len(df)}")
    print()

    # 2. Run full pipeline and report IR-DC median
    print("2. Full pipeline (resample + filters) -> IR-DC median:")
    try:
        from src.parser.log_parser import parse_oralable_log, parse_accelerometer_log
        from src.analysis.features import compute_filters
        from src.processing.resampler import resample_raw_to_50hz

        processed_dir = ROOT / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        for order in ["R_IR_G", "R_G_IR"]:
            for tdm in [False, True]:
                ppg = parse_oralable_log(log_path, tdm_interleave=tdm, channel_order=order)
                if ppg is None or ppg.empty:
                    continue
                accel = parse_accelerometer_log(log_path)
                if accel is None or accel.empty:
                    accel = pd.DataFrame({
                        "timestamp_s": ppg["timestamp_s"],
                        "accel_x": 0, "accel_y": 0, "accel_z": 0,
                    })
                t0 = float(ppg["timestamp_s"].iloc[0])
                ppg = ppg.copy()
                ppg["timestamp_s"] = ppg["timestamp_s"] - t0
                accel = accel.copy()
                accel["timestamp_s"] = accel["timestamp_s"] - t0
                merged = pd.merge_asof(
                    ppg.sort_values("timestamp_s"),
                    accel.sort_values("timestamp_s"),
                    on="timestamp_s", direction="nearest", tolerance=0.1,
                )
                merged_path = processed_dir / "check_ir_dc_merged.csv"
                merged.to_csv(merged_path, index=False)
                resampled = resample_raw_to_50hz(raw_path=merged_path, out_path=processed_dir / "check_ir_dc_50hz.csv")
                resampled = compute_filters(resampled)
                ir_dc_med = resampled["ir_dc"].median()
                print(f"  order={order}, tdm={tdm}: IR-DC median = {ir_dc_med:,.0f} raw")
    except Exception as e:
        import traceback
        traceback.print_exc()
    print()

    # 3. First packet raw values (payload[0]=slot0, [1]=slot1, [2]=slot2 per sample)
    print("3. First PPG packet (raw 32-bit LE, R_IR_G order):")
    df_log = _read_log_rows(log_path)
    rows = _iter_characteristic_payloads(df_log, PPG_CHAR_UUID)
    if rows:
        _, hex_str = rows[0]
        values = _parse_hex_payload(hex_str)
        if len(values) >= 1 + 9:
            fc = values[0]
            payload = values[1:]
            print(f"  Frame counter: {fc}")
            print(f"  Sample 0: slot0={payload[0]:,}, slot1={payload[1]:,}, slot2={payload[2]:,}")
            print(f"  Sample 1: slot0={payload[3]:,}, slot1={payload[4]:,}, slot2={payload[5]:,}")
            print(f"  (R_IR_G: slot0=R, slot1=IR, slot2=G. If slot1<<slot0, try R_G_IR.)")
            slot1_vals = [payload[i] for i in range(1, len(payload), 3)]
            print(f"  slot1 range: min={min(slot1_vals):,}, max={max(slot1_vals):,}")
    print()

    # 4. Suggested thresholds (use R_G_IR if it gives higher IR range)
    best_order = "R_IR_G"
    df_rirg = parse_oralable_log(log_path, tdm_interleave=False, channel_order="R_IR_G")
    df_rgir = parse_oralable_log(log_path, tdm_interleave=False, channel_order="R_G_IR")
    ir_rirg = df_rirg["ir"].median() if not df_rirg.empty else 0
    ir_rgir = df_rgir["ir"].median() if not df_rgir.empty else 0
    if ir_rgir > ir_rirg:
        best_order = "R_G_IR"
        df = df_rgir
    else:
        df = df_rirg
    if not df.empty:
        ir = df["ir"].astype(float)
        med = ir.median()
        # For 32-bit firmware (~20M-60M): use ±50% around median
        if med > 1_000_000:
            suggested_min = max(5_000_000, int(med * 0.5))
            suggested_max = min(80_000_000, int(med * 1.5))
        else:
            p25, p75 = ir.quantile(0.25), ir.quantile(0.75)
            suggested_min = max(0, int(p25 * 0.7))
            suggested_max = int(p75 * 1.5)
        print("4. Suggested IR_DC_RAW_MIN / IR_DC_RAW_MAX:")
        print(f"  Best channel_order for this log: {best_order}")
        print(f"  IR_DC_RAW_MIN = {suggested_min:,}")
        print(f"  IR_DC_RAW_MAX = {suggested_max:,}")
        print()
        print("  Update src/validation/self_validate.py lines 46–47.")
        if best_order != "R_IR_G":
            print(f"  Consider PPG_CHANNEL_ORDER = \"{best_order}\" in log_parser.py if consistent.")
    print()
    print("MAXM86161 firmware expects 19-bit (0–524,287). Values >> 524k indicate")
    print("different firmware or sensor config. Use observed range for coupling check.")


if __name__ == "__main__":
    main()
