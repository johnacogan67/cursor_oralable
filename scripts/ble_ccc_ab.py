#!/usr/bin/env python3
"""A/B: staggered vs burst CCC after mode 3. Phone off. Seat on temple.

Stagger matches nRF Connect / iOS (battery → status → FwLog → PPG → ACC).
Burst matches the old Mac Protocol A storm. Compare how long PPG lasts.

  .venv/bin/python scripts/ble_ccc_ab.py --mode stagger --seconds 60
  .venv/bin/python scripts/ble_ccc_ab.py --mode burst --seconds 60
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.mam_last_connection import (  # noqa: E402
    USER_FACING_DEVICE_NAME,
    format_last_known,
    read_and_persist_firmware,
)
from src.utils.ble_logger import (  # noqa: E402
    ACCEL_CHAR_UUID,
    BATTERY_CHAR_UUID,
    DEVICE_NAME,
    FW_LOG_CHAR_UUID,
    PPG_CHAR_UUID,
    STATUS_CHAR_UUID,
    TEMP_CHAR_UUID,
    _scan_for_oralable,
)

FW_CFG = "3A0FF00B-98C4-46B2-94AF-1AEE0FD4C48E"
STAGGER_S = 0.06
ORDER = [
    (BATTERY_CHAR_UUID, "Battery"),
    (STATUS_CHAR_UUID, "Status"),
    (FW_LOG_CHAR_UUID, "FwLog"),
    (PPG_CHAR_UUID, "PPG"),
    (ACCEL_CHAR_UUID, "ACC"),
    (TEMP_CHAR_UUID, "Temp"),
]


async def _run(mode: str, seconds: float) -> int:
    from bleak import BleakClient

    print(f"Scanning for {USER_FACING_DEVICE_NAME} (advertises as {DEVICE_NAME})...")
    print(format_last_known())
    address = await _scan_for_oralable()
    if not address:
        print(f"{USER_FACING_DEVICE_NAME} not found.", file=sys.stderr)
        return 1
    print(f"Found {USER_FACING_DEVICE_NAME}: {address}  mode={mode}  {seconds:.0f}s")

    ppg_n = 0
    acc_n = 0
    first_ppg = 0.0
    last_ppg = 0.0
    fw_lines: list[str] = []

    def on_ppg(_s, _d):
        nonlocal ppg_n, first_ppg, last_ppg
        now = time.monotonic()
        if first_ppg == 0.0:
            first_ppg = now
        last_ppg = now
        ppg_n += 1

    def on_acc(_s, _d):
        nonlocal acc_n
        acc_n += 1

    def on_fw(_s, data):
        text = bytes(data).decode("utf-8", errors="replace").rstrip("\x00").strip()
        if text:
            fw_lines.append(text)
            print(f"  fw: {text}")

    def on_other(_s, _d):
        return

    handlers = {
        PPG_CHAR_UUID.lower(): on_ppg,
        ACCEL_CHAR_UUID.lower(): on_acc,
        FW_LOG_CHAR_UUID.lower(): on_fw,
    }

    async with BleakClient(address) as client:
        await read_and_persist_firmware(client, address)
        await client.write_gatt_char(FW_CFG, bytes([0x09, 0x03]), response=True)
        await client.write_gatt_char(FW_CFG, bytes([0x06, 0x03]), response=True)
        print("  Wrote mode 3 + stream mask")
        await asyncio.sleep(0.3)

        for uuid, label in ORDER:
            h = handlers.get(uuid.lower(), on_other)
            try:
                await client.start_notify(uuid, h)
                print(f"  {label} CCC")
            except Exception as e:
                print(f"  {label} CCC skip: {e}")
            if mode == "stagger":
                await asyncio.sleep(STAGGER_S)

        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            await asyncio.sleep(0.25)
            if first_ppg and (time.monotonic() - last_ppg) >= 8.0:
                print(f"  PPG silent 8s after {ppg_n} packets — stop")
                break

    span = (last_ppg - first_ppg) if first_ppg else 0.0
    print(f"Done. PPG={ppg_n} ACC={acc_n} span={span:.1f}s")
    if ppg_n == 0:
        print("No PPG. Seat on temple, off case, phone off.", file=sys.stderr)
        return 2
    if span < 15.0:
        print("PPG died early (Failure B pattern).", file=sys.stderr)
        return 3
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="CCC stagger vs burst A/B")
    p.add_argument("--mode", choices=("stagger", "burst"), default="stagger")
    p.add_argument("--seconds", type=float, default=60.0)
    args = p.parse_args()
    try:
        return asyncio.run(_run(args.mode, args.seconds))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
