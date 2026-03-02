"""
Live BLE logger for Oralable MAM.

Connects to the Oralable device, subscribes to PPG (50 Hz), accelerometer (~100 Hz),
temperature (1 Hz), and battery, then streams notifications to a timestamped .txt
file in data/raw/. Compatible with log_parser.py.

Usage:
    python -m src.utils.ble_logger              # Scan for Oralable, connect, log
    python -m src.utils.ble_logger -v          # Stream packets to terminal in real-time
    python -m src.utils.ble_logger --scan      # Scan only (find address)
    python -m src.utils.ble_logger --address XX:XX:XX:...  # Connect by address

macOS: Grant Bluetooth permissions when prompted. Write in append mode to avoid
data loss on crash. Avoid printing every packet to prevent buffer lag at 50 Hz.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# UUIDs match log_parser.py / oralable_nrf tgm_service.h
PPG_CHAR_UUID = "3A0FF001-98C4-46B2-94AF-1AEE0FD4C48E"
ACCEL_CHAR_UUID = "3A0FF002-98C4-46B2-94AF-1AEE0FD4C48E"
TEMP_CHAR_UUID = "3A0FF003-98C4-46B2-94AF-1AEE0FD4C48E"
BATTERY_CHAR_UUID = "3A0FF004-98C4-46B2-94AF-1AEE0FD4C48E"
BATTERY_STATS_CHAR_UUID = "3A0FFEF2-98C4-46B2-94AF-1AEE0FD4C48E"  # Power telemetry: 6-byte payload
DEVICE_NAME = "Oralable"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _bytes_to_hex_line(data: bytes) -> str:
    """Format bytes as space-separated 32-bit (8-char) hex words for parser compatibility."""
    words = []
    for i in range(0, len(data), 4):
        chunk = data[i : i + 4]
        if len(chunk) < 4:
            chunk = chunk + b"\x00" * (4 - len(chunk))
        words.append(chunk.hex().upper())
    return " ".join(words)


def _format_log_line(uuid: str, data: bytes, ts: datetime | None = None) -> str:
    if ts is None:
        ts = datetime.now(timezone.utc)
    time_str = ts.strftime("%H:%M:%S.%f")[:-3]
    hex_str = _bytes_to_hex_line(data)
    return f"{time_str} - Characteristic ({uuid}) notified: <{hex_str}>\n"


def _parse_battery_stats_6byte(data: bytes) -> str | None:
    """
    Parse 6-byte BatteryStats payload: [V_HI, V_LO, Pct, mAh_HI, mAh_LO, Est_Mins].
    Returns [BATT] line for log_parser, or None if invalid.
    """
    if len(data) < 6:
        return None
    voltage_mv = (data[0] << 8) | data[1]
    percent = data[2]
    mah_scaled = (data[3] << 8) | data[4]
    mah_used = mah_scaled / 100.0
    est_min = data[5]
    return f"V: {voltage_mv}mV | %: {percent}% | Used: {mah_used:.2f}mAh | Rem: {est_min}min"


async def _scan_for_oralable() -> str | None:
    """Discover BLE devices; return address of first named 'Oralable' or similar."""
    from bleak import BleakScanner

    devices = await BleakScanner.discover(timeout=10.0)
    for d in devices:
        name = (d.name or "").strip()
        if name and "Oralable" in name:
            return d.address
    return None


async def _run_logger(
    address: str,
    out_path: Path,
    subscribe_ppg: bool = True,
    subscribe_accel: bool = True,
    subscribe_temp: bool = True,
    subscribe_battery: bool = True,
    verbose: bool = False,
) -> None:
    from bleak import BleakClient

    chars = []
    if subscribe_ppg:
        chars.append(PPG_CHAR_UUID)
    if subscribe_accel:
        chars.append(ACCEL_CHAR_UUID)
    if subscribe_temp:
        chars.append(TEMP_CHAR_UUID)
    if subscribe_battery:
        chars.append(BATTERY_CHAR_UUID)
        chars.append(BATTERY_STATS_CHAR_UUID)  # Power telemetry (60s interval)
    if not chars:
        chars = [PPG_CHAR_UUID]

    count = 0
    last_status_at = 0.0

    def notification_handler(char_uuid: str, data: bytearray):
        nonlocal count, last_status_at
        line = _format_log_line(char_uuid, bytes(data))
        with open(out_path, "a+", encoding="utf-8") as f:
            f.write(line)
            # Emit [BATT] line for BatteryStats (power telemetry) - parseable by log_parser
            if char_uuid.lower() == BATTERY_STATS_CHAR_UUID.lower():
                batt_line = _parse_battery_stats_6byte(bytes(data))
                if batt_line:
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"{ts} - [BATT] {batt_line}\n")
        count += 1
        if verbose:
            print(line.rstrip(), flush=True)
        else:
            t = asyncio.get_event_loop().time()
            if t - last_status_at >= 5.0:
                last_status_at = t
                print(f"\rLogged {count} packets...", end="", flush=True)

    print(f"Connecting to {address}...")
    print(f"Logging to {out_path}")
    print("Press Ctrl+C to stop.\n")

    async with BleakClient(address) as client:
        for uuid in chars:
            def make_handler(char_uuid):
                def h(sender, data):
                    notification_handler(char_uuid, data)
                return h
            await client.start_notify(uuid, make_handler(uuid))
        print("Subscribed. Recording...")
        try:
            while True:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        for uuid in chars:
            try:
                await client.stop_notify(uuid)
            except Exception:
                pass

    print(f"\nStopped. Total packets: {count}")
    print(f"Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live BLE logger for Oralable MAM")
    parser.add_argument("--scan", action="store_true", help="Scan for Oralable only")
    parser.add_argument("--address", type=str, help="BLE address (e.g. XX:XX:XX:XX:XX:XX)")
    parser.add_argument("--ppg-only", action="store_true", help="Subscribe to PPG only (50 Hz)")
    parser.add_argument("--accel-only", action="store_true", help="Subscribe to accelerometer only")
    parser.add_argument("--no-temp", action="store_true", help="Skip temperature (3A0FF003)")
    parser.add_argument("--no-battery", action="store_true", help="Skip battery (3A0FF004)")
    parser.add_argument("--out", type=Path, help="Output file path (default: data/raw/Oralable_YYYYMMDD_HHMMSS.txt)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Stream each packet to terminal in real-time")
    args = parser.parse_args()

    try:
        from bleak import BleakScanner  # noqa: F401
    except ImportError:
        print("Install bleak: pip install bleak")
        sys.exit(1)

    if args.scan:
        print("Scanning for BLE devices...")
        async def scan():
            devices = await BleakScanner.discover(timeout=10.0)
            for d in devices:
                print(d)
            oralable = await _scan_for_oralable()
            if oralable:
                print(f"\nOralable found: {oralable}")
            else:
                print("\nNo device named 'Oralable' found.")
        asyncio.run(scan())
        return

    address = args.address
    if not address:
        print("Scanning for Oralable...")
        address = asyncio.run(_scan_for_oralable())
        if not address:
            print("Oralable not found. Use --address XX:XX:XX:XX:XX:XX or run with --scan to list devices.")
            sys.exit(1)
        print(f"Found: {address}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = RAW_DIR / f"Oralable_{ts}.txt"

    subscribe_ppg = not args.accel_only
    subscribe_accel = not args.ppg_only
    subscribe_temp = not args.no_temp
    subscribe_battery = not args.no_battery

    try:
        asyncio.run(_run_logger(address, out_path, subscribe_ppg, subscribe_accel, subscribe_temp, subscribe_battery, args.verbose))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
