#!/usr/bin/env python3
"""
Protocol A guided BLE training session.

Connects via bleak (same path as src/utils/ble_logger), logs PPG/accel/temp/battery
to data/raw/TEMPORALIS_RAW_YYYYMMDD_HHMMSS.txt, and prints timed cues so wall-clock
actions match Protocol A label offsets (recording start = T=0).

FW 1.0.71+: Oralable collects PPG/ACC only while this BLE link is up. A drop stops
the sensors (no FIFO catch-up). Keep the Mac connected for the whole 6 min. Charge
the MAM first with no phone: flash green = still charging (below ~4.20 V);
brief green pulse ~every 10 s = hold. STAT taper alone is not full. While linked,
red/IR is PPG only (no status LED).
Status LEDs never show red.

Mac seat QC (`couple`) is separate from GATT worn (this script still forces mode 3):
  unknown — no BLE / no PPG (or waiting for ACC); never treat as desk
  0 desk / 1 on-skin poor / 2 on-skin + plausible HR or SpO2 AC
See docs/TEMPORALIS_COLLECTION_PROTOCOL.md (seat QC decision map).

Usage (from cursor_oralable root):
  .venv/bin/python scripts/run_protocol_a_session.py
  .venv/bin/python scripts/run_protocol_a_session.py --address UUID
  .venv/bin/python scripts/run_protocol_a_session.py --dry-run   # cues only, no BLE
  .venv/bin/python scripts/run_protocol_a_session.py --no-watchdog  # full 6 min even if PPG dies
  .venv/bin/python scripts/run_protocol_a_session.py --fw-log      # subscribe 00A (can stall PPG)
  .venv/bin/python scripts/run_protocol_a_session.py --no-coupling-gate
  .venv/bin/python scripts/run_protocol_a_session.py --allow-couple-1

FW 1.0.74+: STAT pad-wins will not clear mode 3 while PPG/ACC CCC is on.
FW 1.0.77+: 5 s pad-wins grace after mode 3; CCC is staggered battery → status →
PPG → ACC (nRF Connect / iOS order) so a burst does not stall the link.
FwLog (00A) is off by default — it filled the GATT queue and tripped the 4 s
stall recover (Mac abort ~8 s). Pass --fw-log only for a bench dump.
The Mac watchdog still aborts if PPG stalls ~8 s or worn drops after two re-writes.
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

from src.utils.mam_last_connection import (  # noqa: E402
    USER_FACING_DEVICE_NAME,
    firmware_log_header,
    format_last_known,
    read_and_persist_firmware,
)
from src.utils.ble_logger import (  # noqa: E402
    ACCEL_CHAR_UUID,
    BATTERY_CHAR_UUID,
    BATTERY_STATS_CHAR_UUID,
    DEVICE_NAME,
    FW_LOG_CHAR_UUID,
    PPG_CHAR_UUID,
    RAW_DIR,
    STATUS_CHAR_UUID,
    TEMP_CHAR_UUID,
    _scan_for_oralable,
    format_status_line,
    parse_status_5byte,
)
from src.utils.coupling_gate import (  # noqa: E402
    CoupleState,
    CouplingGate,
    couple_label,
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
PPG_STALL_ABORT_S = 8.0
MAX_REWORN_WRITES = 2
CCC_STAGGER_S = 0.06
COUPLE_PREFLIGHT_TIMEOUT_S = 60.0
COUPLE_DESK_ABORT_GRACE_S = 20.0
# Capture CCC: battery → status → PPG → ACC → temp. FwLog is opt-in (--fw-log).
PROTOCOL_A_CCC_ORDER: list[tuple[str, str]] = [
    (BATTERY_CHAR_UUID, "Battery"),
    (BATTERY_STATS_CHAR_UUID, "BatteryStats"),
    (STATUS_CHAR_UUID, "Status"),
    (PPG_CHAR_UUID, "PPG"),
    (ACCEL_CHAR_UUID, "ACC"),
    (TEMP_CHAR_UUID, "Temp"),
]
PROTOCOL_A_FWLOG_CCC: tuple[str, str] = (FW_LOG_CHAR_UUID, "FwLog")


async def write_worn_and_stream(client) -> None:
    """Force user mode 3 + PPG|ACC. STAT pad-wins on FW < 1.0.74 can undo this."""
    await client.write_gatt_char(
        FW_CFG_CHAR_UUID,
        bytes([FW_CFG_SET_USER_DEVICE_MODE, USER_MODE_WORN]),
        response=True,
    )
    await client.write_gatt_char(
        FW_CFG_CHAR_UUID,
        bytes([FW_CFG_SET_STREAM_ENABLE_MASK, 0x03]),
        response=True,
    )


async def _run_session(
    address: str,
    out_path: Path,
    duration_s: float,
    verbose: bool,
    watchdog: bool,
    fw_log: bool,
    coupling_gate: bool,
    allow_couple_1: bool,
) -> None:
    stop_event = asyncio.Event()

    from bleak import BleakClient
    from src.utils.ble_logger import (
        _format_log_line,
        _parse_battery_stats_6byte,
    )

    chars = [uuid for uuid, _label in PROTOCOL_A_CCC_ORDER]
    if fw_log:
        chars.append(PROTOCOL_A_FWLOG_CCC[0])
    count = 0
    ppg_count = 0
    accel_count = 0
    last_status_at = 0.0
    last_ppg_mono = 0.0
    ppg_started = asyncio.Event()
    acc_started = asyncio.Event()
    couple_ready = asyncio.Event()
    clock_started = False
    clock_t0: float | None = None
    worn_seen = False
    need_reworn = False
    reworn_writes = 0
    last_dev_status: dict[str, int] | None = None
    abort_reason: str | None = None
    desk_since: float | None = None
    last_couple_warn: CoupleState | None = None
    loop = asyncio.get_event_loop()
    gate = CouplingGate(require_acc=True) if coupling_gate else None

    def _append(line: str) -> None:
        with open(out_path, "a+", encoding="utf-8") as f:
            f.write(line)

    def _maybe_set_couple_ready() -> None:
        if gate is None or couple_ready.is_set():
            return
        st = gate.state
        if st == CoupleState.GOOD or (allow_couple_1 and st == CoupleState.POOR):
            couple_ready.set()

    def notification_handler(char_uuid: str, data: bytearray) -> None:
        nonlocal count, ppg_count, accel_count, last_status_at
        nonlocal last_ppg_mono, worn_seen, need_reworn, last_dev_status
        nonlocal last_couple_warn
        raw = bytes(data)
        cu = char_uuid.lower()
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        mono = loop.time()

        if cu == STATUS_CHAR_UUID.lower():
            parsed = parse_status_5byte(raw)
            if parsed:
                last_dev_status = parsed
                _append(f"{ts} - [STATUS] {format_status_line(parsed)}\n")
                if gate is not None:
                    gate.set_on_dock(parsed["on_dock"] == 1)
                if parsed["worn"] == 1:
                    worn_seen = True
                elif worn_seen and parsed["worn"] == 0:
                    need_reworn = True
                    print(
                        f"\nWARNING: worn dropped "
                        f"({format_status_line(parsed)}) — will re-write mode 3",
                        flush=True,
                    )
            return

        if cu == FW_LOG_CHAR_UUID.lower():
            text = raw.decode("utf-8", errors="replace").rstrip("\x00").strip()
            if text:
                _append(f"{ts} - [FWLOG] {text}\n")
                print(f"    fw: {text}", flush=True)
            return

        line = _format_log_line(char_uuid, raw)
        extra = ""
        if cu == BATTERY_STATS_CHAR_UUID.lower():
            batt_line = _parse_battery_stats_6byte(raw)
            if batt_line:
                extra = f"{ts} - [BATT] {batt_line}\n"
        _append(line + extra)
        count += 1
        if PPG_CHAR_UUID.lower() in cu:
            ppg_count += 1
            last_ppg_mono = mono
            if not ppg_started.is_set():
                ppg_started.set()
            if gate is not None:
                if clock_t0 is not None:
                    gate.set_session_elapsed(mono - clock_t0)
                gate.feed_ppg_payload(raw, mono)
                _maybe_set_couple_ready()
        elif ACCEL_CHAR_UUID.lower() in cu:
            accel_count += 1
            if not acc_started.is_set():
                acc_started.set()
            if gate is not None:
                gate.feed_acc_payload(raw, mono)
                _maybe_set_couple_ready()
        elif TEMP_CHAR_UUID.lower() in cu and gate is not None and len(raw) >= 6:
            # frame_counter u32 + int16 centidegrees
            centi = struct.unpack_from("<h", raw, 4)[0]
            gate.feed_temp_c(centi / 100.0)
        if verbose:
            print(line.rstrip(), flush=True)
        else:
            if mono - last_status_at >= 5.0:
                last_status_at = mono
                worn_bit = ""
                if last_dev_status is not None:
                    worn_bit = f" worn={last_dev_status['worn']}"
                couple_bit = ""
                if gate is not None:
                    couple_bit = f" {gate.status_snippet()}"
                    st = gate.state
                    if st == CoupleState.POOR and st != last_couple_warn and clock_started:
                        last_couple_warn = st
                        print(
                            "\nWARNING: couple=1 (on skin, weak pulse/vitals) — reseat if needed",
                            flush=True,
                        )
                print(
                    f"\rLogged {count} packets "
                    f"(PPG={ppg_count} ACC={accel_count}{worn_bit}{couple_bit})...",
                    end="",
                    flush=True,
                )

    print(f"Connecting to {address} ({USER_FACING_DEVICE_NAME}; advertises as {DEVICE_NAME})...")
    print(f"Logging to {out_path}")
    print(f"{format_last_known()}")
    print("Seat on temple, off the case 5+ s, disconnect iPhone, then wait for cues.\n")

    async with BleakClient(address) as client:
        firmware = await read_and_persist_firmware(client, address)
        _append(firmware_log_header(firmware, address))
        try:
            await write_worn_and_stream(client)
            print("  Wrote user mode = Worn + stream PPG|ACC")
        except Exception as e:
            print(f"  WARNING: worn/stream write failed: {e}", file=sys.stderr)

        await asyncio.sleep(0.3)

        for uuid, label in PROTOCOL_A_CCC_ORDER:

            def make_handler(char_uuid: str):
                def h(sender, data):
                    notification_handler(char_uuid, data)

                return h

            try:
                await client.start_notify(uuid, make_handler(uuid))
                print(f"  {label} subscribed.")
            except Exception as e:
                if label in ("BatteryStats", "Status", "FwLog"):
                    pass
                else:
                    print(f"  WARNING: {label} CCC failed: {e}", file=sys.stderr)
            await asyncio.sleep(CCC_STAGGER_S)

        if fw_log:
            uuid, label = PROTOCOL_A_FWLOG_CCC

            def make_fwlog_handler(char_uuid: str):
                def h(sender, data):
                    notification_handler(char_uuid, data)

                return h

            try:
                await client.start_notify(uuid, make_fwlog_handler(uuid))
                print(f"  {label} subscribed (opt-in).")
            except Exception as e:
                print(f"  WARNING: {label} CCC failed: {e}", file=sys.stderr)
            await asyncio.sleep(CCC_STAGGER_S)

        print("Subscribed. Waiting for PPG + ACC...")
        try:
            await asyncio.wait_for(ppg_started.wait(), timeout=45.0)
        except asyncio.TimeoutError:
            print(
                "ERROR: No PPG notifications in 45 s "
                f"(total packets={count}, PPG={ppg_count}). "
                "Lift off the case, seat on temple, disconnect iOS, keep this BLE link up.",
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

        try:
            await asyncio.wait_for(acc_started.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            print(
                "WARNING: No ACC yet — seat QC needs PPG+ACC; waiting briefly...",
                file=sys.stderr,
            )

        if last_dev_status and last_dev_status.get("on_dock") == 1:
            print(
                "WARNING: status on_dock=1 — lift off the case and wait 5 s.",
                file=sys.stderr,
            )

        if gate is not None:
            need = "couple=2" if not allow_couple_1 else "couple>=1"
            print(
                f"Waiting for seat QC ({need}; GATT worn≠couple; "
                f"PPG={ppg_count} ACC={accel_count})..."
            )
            deadline = loop.time() + COUPLE_PREFLIGHT_TIMEOUT_S
            preflight_desk_since: float | None = None
            while not couple_ready.is_set() and not stop_event.is_set():
                now = loop.time()
                gate.tick(now)
                st = gate.state
                if st == CoupleState.DESK:
                    if preflight_desk_since is None:
                        preflight_desk_since = now
                else:
                    preflight_desk_since = None
                _maybe_set_couple_ready()
                if now >= deadline:
                    break
                await asyncio.sleep(0.25)

            st = gate.state
            if not couple_ready.is_set():
                if st == CoupleState.DESK or st is None:
                    abort_reason = (
                        f"seat QC preflight failed: couple={couple_label(st)} "
                        f"(need {need}; desk/unknown — reseat on temple, off case)"
                    )
                else:
                    abort_reason = (
                        f"seat QC preflight timeout: couple={couple_label(st)} "
                        f"(need {need}; pass --allow-couple-1 to start on poor seat)"
                    )
                print(f"\nERROR: {abort_reason}", file=sys.stderr)
                stop_event.set()
                for uuid in chars:
                    try:
                        await client.stop_notify(uuid)
                    except Exception:
                        pass
                print(f"\nStopped early. Saved incomplete log to {out_path}")
                return
            print(
                f"Seat QC ready — {gate.status_snippet()} "
                f"(GATT worn may still be forced 1).\n"
            )

        clock_started = True
        clock_t0 = loop.time()
        desk_since = None
        print(
            f"Protocol A clock STARTS NOW "
            f"(PPG={ppg_count} ACC={accel_count}"
            f"{(' ' + gate.status_snippet()) if gate else ''}).\n"
        )
        cue_task = asyncio.create_task(_cue_loop(stop_event, duration_s))
        try:
            while not stop_event.is_set():
                mono = loop.time()
                if gate is not None and clock_t0 is not None:
                    gate.set_session_elapsed(mono - clock_t0)
                    gate.tick(mono)
                    st = gate.state
                    if st is None and watchdog:
                        abort_reason = (
                            "couple=unknown (PPG lost / cannot judge seat without BLE+PPG)"
                        )
                        print(f"\nERROR: {abort_reason}", file=sys.stderr)
                        stop_event.set()
                        break
                    if st == CoupleState.DESK:
                        if desk_since is None:
                            desk_since = mono
                        elif (mono - desk_since) >= COUPLE_DESK_ABORT_GRACE_S and watchdog:
                            abort_reason = (
                                f"couple=0 desk for {mono - desk_since:.0f}s "
                                f"[{gate.status_snippet()}]"
                            )
                            print(f"\nERROR: {abort_reason}", file=sys.stderr)
                            stop_event.set()
                            break
                    else:
                        desk_since = None
                if watchdog and clock_started:
                    silent = mono - last_ppg_mono
                    if last_ppg_mono > 0 and silent >= PPG_STALL_ABORT_S:
                        abort_reason = (
                            f"PPG stalled {silent:.1f}s "
                            f"(PPG={ppg_count} ACC={accel_count})"
                        )
                        if last_dev_status:
                            abort_reason += f" [{format_status_line(last_dev_status)}]"
                        if gate is not None:
                            gate.mark_unknown()
                            abort_reason += f" [{gate.status_snippet()}]"
                        print(f"\nERROR: {abort_reason}", file=sys.stderr)
                        stop_event.set()
                        break
                    if need_reworn:
                        need_reworn = False
                        if reworn_writes < MAX_REWORN_WRITES:
                            reworn_writes += 1
                            try:
                                await write_worn_and_stream(client)
                                print(
                                    f"  Re-wrote mode 3 ({reworn_writes}/{MAX_REWORN_WRITES})",
                                    flush=True,
                                )
                            except Exception as e:
                                print(f"  Re-worn write failed: {e}", file=sys.stderr)
                        else:
                            abort_reason = "worn dropped and mode 3 re-write exhausted"
                            if last_dev_status:
                                abort_reason += f" [{format_status_line(last_dev_status)}]"
                            print(f"\nERROR: {abort_reason}", file=sys.stderr)
                            stop_event.set()
                            break
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
    if gate is not None:
        print(f"Final seat QC: {gate.status_snippet()}")
    if abort_reason:
        print(f"Aborted: {abort_reason}", file=sys.stderr)
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
    ap.add_argument(
        "--no-watchdog",
        action="store_true",
        help="Do not abort on PPG stall or worn drop (cues run the full 6 min)",
    )
    ap.add_argument(
        "--fw-log",
        action="store_true",
        help="Subscribe firmware log 00A (can stall PPG on 1.0.82; off by default)",
    )
    ap.add_argument(
        "--no-coupling-gate",
        action="store_true",
        help="Skip Mac couple seat QC (legacy: start on first PPG)",
    )
    ap.add_argument(
        "--allow-couple-1",
        action="store_true",
        help="Start Protocol A clock when couple=1 (poor seat), not only couple=2",
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
        print(f"Scanning for {USER_FACING_DEVICE_NAME} (advertises as {DEVICE_NAME})...")
        print(format_last_known())
        address = asyncio.run(_scan_for_oralable())
        if not address:
            print(
                f"{USER_FACING_DEVICE_NAME} not found. Disconnect iOS, wake device, retry.",
                file=sys.stderr,
            )
            print(format_last_known(), file=sys.stderr)
            return 1
        print(f"Found {USER_FACING_DEVICE_NAME}: {address}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = RAW_DIR / f"TEMPORALIS_RAW_{ts}.txt"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")  # truncate / create

    try:
        asyncio.run(
            _run_session(
                address,
                out_path,
                args.duration,
                args.verbose,
                watchdog=not args.no_watchdog,
                fw_log=args.fw_log,
                coupling_gate=not args.no_coupling_gate,
                allow_couple_1=args.allow_couple_1,
            )
        )
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
