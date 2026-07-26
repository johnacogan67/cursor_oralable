#!/usr/bin/env python3
"""
Protocol A guided BLE training session.

Connects via bleak (same path as src/utils/ble_logger), logs PPG/accel/temp/battery
to data/raw/TEMPORALIS_RAW_YYYYMMDD_HHMMSS.txt, and prints timed cues so wall-clock
actions match Protocol A label offsets (recording start = T=0).

Usage (from cursor_oralable root):
  .venv/bin/python scripts/run_protocol_a_session.py
  .venv/bin/python scripts/run_protocol_a_session.py --address UUID
  .venv/bin/python scripts/run_protocol_a_session.py --dry-run   # cues only, no BLE
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.ble_logger import (  # noqa: E402
    DEVICE_NAME,
    RAW_DIR,
    _scan_for_oralable,
)

# (start_s, end_s, cue) — elapsed from recording start (Protocol A)
PROTOCOL_A_CUES: list[tuple[float, float, str]] = [
    (0.0, 60.0, "REST quiet — stay still, jaw relaxed"),
    (60.0, 70.0, "FIVE sync taps — firm, rhythmic taps on housing"),
    (70.0, 120.0, "REST — settle after taps"),
    (120.0, 130.0, "MAX TONIC clench — hold hard 10 s"),
    (130.0, 180.0, "REST recovery — release, stay still"),
    (180.0, 200.0, "PHASIC grind — rhythmic jaw motion 20 s"),
    (200.0, 240.0, "REST — accel baseline"),
    (240.0, 260.0, "BREATH HOLD (simulated apnea) — hold breath 20 s"),
    (260.0, 270.0, "TONIC RESCUE clench — clench hard 10 s (still holding or just after)"),
    (270.0, 360.0, "FINAL recovery — stillness 90 s"),
]

SESSION_DURATION_S = 360.0


def _fmt_mmss(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


async def _cue_loop(stop_event: asyncio.Event, duration_s: float) -> None:
    """Print Protocol A cues until duration or stop_event."""
    print("\n" + "=" * 60)
    print("PROTOCOL A — cues from recording start (stay within 1–2 s)")
    print("=" * 60)
    t0 = asyncio.get_event_loop().time()
    last_cue: str | None = None
    last_tick = -1

    while not stop_event.is_set():
        elapsed = asyncio.get_event_loop().time() - t0
        if elapsed >= duration_s:
            print(f"\n[{_fmt_mmss(duration_s)}] SESSION COMPLETE — stopping logger")
            stop_event.set()
            break

        active = None
        for start, end, cue in PROTOCOL_A_CUES:
            if start <= elapsed < end:
                active = cue
                break
        if active is None:
            active = "FINAL recovery — stillness"

        if active != last_cue:
            print(f"\n>>> [{_fmt_mmss(elapsed)}] {active}")
            last_cue = active

        tick = int(elapsed)
        if tick != last_tick and tick % 10 == 0:
            print(f"    ... {_fmt_mmss(elapsed)} / {_fmt_mmss(duration_s)}", flush=True)
            last_tick = tick

        await asyncio.sleep(0.25)


# Firmware config characteristic (write): opcode + payload
FW_CFG_CHAR_UUID = "3A0FF00B-98C4-46B2-94AF-1AEE0FD4C48E"
FW_CFG_SET_STREAM_ENABLE_MASK = 0x06  # [u8 mask] bit0=ppg bit1=acc
FW_CFG_SET_USER_DEVICE_MODE = 0x09  # [u8 mode] 0=auto 1=charger 2=idle 3=worn
USER_MODE_WORN = 0x03


async def _run_session(
    address: str,
    out_path: Path,
    duration_s: float,
    verbose: bool,
) -> None:
    stop_event = asyncio.Event()

    from bleak import BleakClient
    from src.utils.ble_logger import (
        ACCEL_CHAR_UUID,
        BATTERY_CHAR_UUID,
        BATTERY_STATS_CHAR_UUID,
        PPG_CHAR_UUID,
        TEMP_CHAR_UUID,
        _format_log_line,
        _parse_battery_stats_6byte,
    )

    chars = [PPG_CHAR_UUID, ACCEL_CHAR_UUID, TEMP_CHAR_UUID, BATTERY_CHAR_UUID]
    count = 0
    ppg_count = 0
    accel_count = 0
    last_status_at = 0.0
    ppg_started = asyncio.Event()

    def notification_handler(char_uuid: str, data: bytearray) -> None:
        nonlocal count, ppg_count, accel_count, last_status_at
        line = _format_log_line(char_uuid, bytes(data))
        with open(out_path, "a+", encoding="utf-8") as f:
            f.write(line)
            if char_uuid.lower() == BATTERY_STATS_CHAR_UUID.lower():
                batt_line = _parse_battery_stats_6byte(bytes(data))
                if batt_line:
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"{ts} - [BATT] {batt_line}\n")
        count += 1
        cu = char_uuid.lower()
        if PPG_CHAR_UUID.lower() in cu:
            ppg_count += 1
            if not ppg_started.is_set():
                ppg_started.set()
        elif ACCEL_CHAR_UUID.lower() in cu:
            accel_count += 1
        if verbose:
            print(line.rstrip(), flush=True)
        else:
            t = asyncio.get_event_loop().time()
            if t - last_status_at >= 5.0:
                last_status_at = t
                print(
                    f"\rLogged {count} packets (PPG={ppg_count} ACC={accel_count})...",
                    end="",
                    flush=True,
                )

    print(f"Connecting to {address} ({DEVICE_NAME})...")
    print(f"Logging to {out_path}")
    print("Seat on temple, disconnect iPhone app, then wait for cues.\n")

    async with BleakClient(address) as client:
        # Force worn + PPG/ACC stream before CCC enable (FW worn-gate otherwise
        # yields battery-only notifies — useless for Protocol A training).
        try:
            await client.write_gatt_char(
                FW_CFG_CHAR_UUID,
                bytes([FW_CFG_SET_USER_DEVICE_MODE, USER_MODE_WORN]),
                response=True,
            )
            print("  Wrote user device mode = Worn (0x09, 0x03)")
        except Exception as e:
            print(f"  WARNING: worn-mode write failed: {e}", file=sys.stderr)
        try:
            await client.write_gatt_char(
                FW_CFG_CHAR_UUID,
                bytes([FW_CFG_SET_STREAM_ENABLE_MASK, 0x03]),
                response=True,
            )
            print("  Wrote stream enable mask = PPG|ACC (0x06, 0x03)")
        except Exception as e:
            print(f"  WARNING: stream-mask write failed: {e}", file=sys.stderr)

        await asyncio.sleep(0.3)

        for uuid in chars:

            def make_handler(char_uuid: str):
                def h(sender, data):
                    notification_handler(char_uuid, data)

                return h

            await client.start_notify(uuid, make_handler(uuid))
        try:

            def make_handler(char_uuid: str):
                def h(sender, data):
                    notification_handler(char_uuid, data)

                return h

            await client.start_notify(
                BATTERY_STATS_CHAR_UUID, make_handler(BATTERY_STATS_CHAR_UUID)
            )
            print("  BatteryStats subscribed.")
        except Exception:
            pass

        print("Subscribed. Waiting for first PPG packet to start Protocol A clock...")
        try:
            await asyncio.wait_for(ppg_started.wait(), timeout=45.0)
        except asyncio.TimeoutError:
            print(
                "ERROR: No PPG notifications in 45 s "
                f"(total packets={count}, PPG={ppg_count}). "
                "Seat on skin, disconnect iOS, confirm FW >= 1.0.70.",
                file=sys.stderr,
            )
            stop_event.set()
            for uuid in chars:
                try:
                    await client.stop_notify(uuid)
                except Exception:
                    pass
            print(f"\nStopped early. Saved incomplete log to {out_path}")
            return

        print(
            f"First PPG received — Protocol A clock STARTS NOW "
            f"(PPG={ppg_count} ACC={accel_count}).\n"
        )
        cue_task = asyncio.create_task(_cue_loop(stop_event, duration_s))
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            stop_event.set()
        await cue_task

        for uuid in chars:
            try:
                await client.stop_notify(uuid)
            except Exception:
                pass

    print(f"\nStopped. Total packets: {count} (PPG={ppg_count} ACC={accel_count})")
    print(f"Saved to {out_path}")
    if ppg_count < 100:
        print(
            "WARNING: Very few PPG packets — log may be unusable for training.",
            file=sys.stderr,
        )


async def _dry_run_cues(duration_s: float) -> None:
    stop_event = asyncio.Event()
    await _cue_loop(stop_event, duration_s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Protocol A timed BLE training session")
    ap.add_argument("--address", type=str, help="BLE address / UUID")
    ap.add_argument(
        "--out",
        type=Path,
        help="Output log path (default: data/raw/TEMPORALIS_RAW_YYYYMMDD_HHMMSS.txt)",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=SESSION_DURATION_S,
        help=f"Session length seconds (default {SESSION_DURATION_S})",
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cues only (no BLE); useful to rehearse timing",
    )
    args = ap.parse_args()

    if args.dry_run:
        print("DRY RUN — cues only (no BLE)")
        try:
            asyncio.run(_dry_run_cues(args.duration))
        except KeyboardInterrupt:
            print("\nInterrupted.")
        return 0

    try:
        from bleak import BleakClient  # noqa: F401
    except ImportError:
        print("Install bleak: pip install bleak", file=sys.stderr)
        return 1

    address = args.address
    if not address:
        print("Scanning for Oralable...")
        address = asyncio.run(_scan_for_oralable())
        if not address:
            print("Oralable not found. Disconnect iOS, wake device, retry.", file=sys.stderr)
            return 1
        print(f"Found: {address}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = RAW_DIR / f"TEMPORALIS_RAW_{ts}.txt"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")  # truncate / create

    try:
        asyncio.run(_run_session(address, out_path, args.duration, args.verbose))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("\nNext:")
    print(f"  .venv/bin/python scripts/process_temporalis_gold.py {out_path}")
    print(
        "  .venv/bin/python scripts/generate_clinical_report.py "
        "--input data/validation/GOLD_STANDARD_VALIDATION.csv"
    )
    print(
        "  # (clinical report also writes plots/overnight_report/<session>/night_report.pdf)"
    )
    print(
        "  .venv/bin/python scripts/generate_overnight_night_report.py "
        "--input data/validation/GOLD_STANDARD_VALIDATION.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
