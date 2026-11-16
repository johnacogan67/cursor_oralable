#!/usr/bin/env python3
"""
ANR M40 Muscle Sense EMG logger (bleak).

Scans for ANR company ID 0x05DA (or name match), subscribes to Analog 0x2A58
(uint16 LE, 0–1023, ~10 Hz), writes wall-clock CSV to data/raw/ANR_EMG_*.txt.

Usage (from cursor_oralable root):
  .venv/bin/python scripts/run_anr_emg_session.py
  .venv/bin/python scripts/run_anr_emg_session.py --address UUID
  .venv/bin/python scripts/run_anr_emg_session.py --duration 360
  .venv/bin/python scripts/run_anr_emg_session.py --scan-only

See docs/ANR_M40_CONCORDANCE.md and https://www.anrcorp.com/documentation/
"""

from __future__ import annotations

import argparse
import asyncio
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data" / "raw"

# ANR M40 — match ANRMuscleSenseDevice.swift / BLE Design Guide
ANR_COMPANY_ID = 0x05DA
AUTOMATION_IO_SERVICE = "00001815-0000-1000-8000-00805f9b34fb"
ANALOG_EMG_CHAR = "00002a58-0000-1000-8000-00805f9b34fb"
BATTERY_LEVEL_CHAR = "00002a19-0000-1000-8000-00805f9b34fb"


def _manufacturer_company_ids(advertisement_data) -> list[int]:
    mfg = getattr(advertisement_data, "manufacturer_data", None) or {}
    return [int(k) for k in mfg.keys()]


def _name_looks_anr(name: str | None) -> bool:
    if not name:
        return False
    n = name.strip().lower()
    return any(tok in n for tok in ("anr", "muscle", "m40", "sense"))


async def scan_for_anr(timeout: float = 12.0) -> str | None:
    """Return BLE address of first ANR M40 (company ID or name)."""
    from bleak import BleakScanner

    found: list[tuple[str, str]] = []

    def _cb(device, advertisement_data) -> None:
        ids = _manufacturer_company_ids(advertisement_data)
        name = device.name or advertisement_data.local_name
        if ANR_COMPANY_ID in ids or _name_looks_anr(name):
            label = (name or "ANR").strip()
            if not any(a == device.address for a, _ in found):
                found.append((device.address, label))
                print(f"  candidate: {label} @ {device.address} (mfg={ids})")

    print(f"Scanning for ANR M40 (company 0x{ANR_COMPANY_ID:04X})...")
    async with BleakScanner(detection_callback=_cb):
        await asyncio.sleep(timeout)

    if not found:
        return None
    return found[0][0]


async def _run_logger(
    address: str,
    out_path: Path,
    duration_s: float | None,
    verbose: bool,
) -> None:
    from bleak import BleakClient

    count = 0
    t_start = asyncio.get_event_loop().time()
    stop = asyncio.Event()

    def on_emg(_sender, data: bytearray) -> None:
        nonlocal count
        if len(data) < 2:
            return
        emg = struct.unpack_from("<H", data, 0)[0]
        if emg > 1023:
            if verbose:
                print(f"  skip out-of-range emg={emg}")
            return
        ts = datetime.now(timezone.utc)
        # ISO + unix for alignment; emg_raw 0–1023
        line = f"{ts.isoformat()},{ts.timestamp():.6f},{emg}\n"
        with out_path.open("a", encoding="utf-8") as f:
            f.write(line)
        count += 1
        if verbose and count % 10 == 0:
            print(f"  emg={emg} n={count}", flush=True)

    print(f"Connecting to {address}...")
    async with BleakClient(address, timeout=30.0) as client:
        if not client.is_connected:
            raise RuntimeError("Failed to connect to ANR")
        print("Connected. Enabling Analog (2A58) notifications...")

        # Header once
        if out_path.stat().st_size == 0:
            out_path.write_text(
                "# ANR_M40 EMG log — iso_utc,unix_s,emg_raw\n"
                "iso_utc,unix_s,emg_raw\n",
                encoding="utf-8",
            )

        try:
            batt = await client.read_gatt_char(BATTERY_LEVEL_CHAR)
            if batt:
                print(f"Battery: {batt[0]}%")
        except Exception:
            pass

        await client.start_notify(ANALOG_EMG_CHAR, on_emg)
        print(f"Logging → {out_path}")
        if duration_s:
            print(f"Duration: {duration_s:.0f}s (Ctrl+C to stop early)")
        else:
            print("Duration: until Ctrl+C")

        last_status = -1
        try:
            while not stop.is_set():
                elapsed = asyncio.get_event_loop().time() - t_start
                if duration_s is not None and elapsed >= duration_s:
                    break
                tick = int(elapsed)
                if tick > 0 and tick % 30 == 0 and tick != last_status:
                    print(f"  ... {elapsed:.0f}s samples={count}", flush=True)
                    last_status = tick
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await client.stop_notify(ANALOG_EMG_CHAR)
            except Exception:
                pass

    print(f"Done. samples={count} → {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="ANR M40 EMG BLE logger")
    ap.add_argument("--address", type=str, help="BLE address / UUID")
    ap.add_argument(
        "--out",
        type=Path,
        help="Output path (default data/raw/ANR_EMG_YYYYMMDD_HHMMSS.txt)",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Seconds to log (default: until Ctrl+C)",
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--scan-only", action="store_true", help="Scan and exit")
    ap.add_argument("--scan-timeout", type=float, default=12.0)
    args = ap.parse_args()

    try:
        from bleak import BleakClient  # noqa: F401
    except ImportError:
        print("Install bleak: pip install bleak", file=sys.stderr)
        return 1

    if args.scan_only:
        addr = asyncio.run(scan_for_anr(args.scan_timeout))
        if not addr:
            print("ANR not found. Power on M40, disconnect phone apps, retry.", file=sys.stderr)
            return 1
        print(f"Found: {addr}")
        return 0

    address = args.address
    if not address:
        address = asyncio.run(scan_for_anr(args.scan_timeout))
        if not address:
            print("ANR not found. Power on M40, disconnect phone apps, retry.", file=sys.stderr)
            return 1
        print(f"Found: {address}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = RAW_DIR / f"ANR_EMG_{ts}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    try:
        asyncio.run(_run_logger(address, out_path, args.duration, args.verbose))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("\nNext (dual session preferred):")
    print("  .venv/bin/python scripts/run_dual_protocol_a_session.py")
    print("  .venv/bin/python scripts/align_anr_oralable_concordance.py \\")
    print(f"    --anr {out_path} --oralable data/raw/TEMPORALIS_RAW_....txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
