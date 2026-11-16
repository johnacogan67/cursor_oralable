#!/usr/bin/env python3
"""
Dual Protocol A — MAM + ANR M40 on shared wall-clock cues.

Runs MAM BLE logger (PPG/ACC; advertises as Oralable) and ANR Analog EMG logger
in parallel while printing the same Protocol A cue sequence (5-tap sync at 01:00).

Before Protocol A cues:
  1) ANR EMG preflight (rest → hard clench → rest; gate clench max, default ≥70)
  2) MAM IR optical preflight (same cue pattern; gate IR drop vs rest, default ≥8%)
  3) SpO₂ AC WARN (quiet ~12 s; never blocks — stamps spo2_preflight_ac_ok)

Seat MAM first (confirm IR trough on clench), then ANR vertical without shearing
the PPG window. EMG alone is not enough for Dual A. SpO₂ QC is post-hoc + AC warn.

FW 1.0.71+: MAM PPG/ACC run only while the Mac BLE link is up. Keep both
centrals connected for the whole session. Charge first with no phone (flash green
= still charging; brief green pulse ~every 10 s = hold ≥ ~4.20 V). While linked,
MAM status LED is off.

Usage (from cursor_oralable root):
  .venv/bin/python scripts/run_dual_protocol_a_session.py
  .venv/bin/python scripts/run_dual_protocol_a_session.py --dry-run
  .venv/bin/python scripts/run_dual_protocol_a_session.py \\
      --oralable-address UUID --anr-address UUID
  .venv/bin/python scripts/run_dual_protocol_a_session.py --skip-emg-preflight
  .venv/bin/python scripts/run_dual_protocol_a_session.py --skip-ir-preflight
  .venv/bin/python scripts/run_dual_protocol_a_session.py --emg-gate 100 --ir-drop-gate 0.08

See docs/ANR_M40_CONCORDANCE.md · docs/TEMPORALIS_COLLECTION_PROTOCOL.md (Dual A).
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

import importlib.util  # noqa: E402
import numpy as np  # noqa: E402


def _load_sibling(name: str, filename: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_anr = _load_sibling("run_anr_emg_session", "run_anr_emg_session.py")
_proto = _load_sibling("run_protocol_a_session", "run_protocol_a_session.py")

ANALOG_EMG_CHAR = _anr.ANALOG_EMG_CHAR
BATTERY_LEVEL_CHAR = _anr.BATTERY_LEVEL_CHAR
scan_for_anr = _anr.scan_for_anr
SESSION_DURATION_S = _proto.SESSION_DURATION_S
_cue_loop = _proto._cue_loop
_fmt_mmss = _proto._fmt_mmss
write_worn_and_stream = _proto.write_worn_and_stream
PPG_STALL_ABORT_S = _proto.PPG_STALL_ABORT_S

from src.parser.log_parser import PPG_CHANNEL_ORDER, _parse_ppg_sample  # noqa: E402
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
    PPG_CHAR_UUID,
    RAW_DIR,
    STATUS_CHAR_UUID,
    TEMP_CHAR_UUID,
    _format_log_line,
    _parse_battery_stats_6byte,
    _scan_for_oralable,
    format_status_line,
    parse_status_5byte,
)

# Placement preflight: rest → hard clench → rest (seconds)
PREFLIGHT_REST_S = 10.0
PREFLIGHT_CLENCH_S = 10.0
# Dual A stack often lands ~70–90; eng pack 20260812_085110 cleared at 70 (max 83).
# IR optical gate is the second QC — keep EMG as contact proof, not hero amplitude.
DEFAULT_EMG_GATE = 70  # clench-window max must clear this (0–1023 raw)
DEFAULT_IR_DROP_GATE = 0.08  # (rest_median - clench_min) / rest_median
# Reject optical coupling outside this rest-median band (raw IR ADC)
IR_REST_MEDIAN_MIN = 500.0
IR_REST_MEDIAN_MAX = 80_000_000.0
IR_ROLL_SAMPLES = 25  # ~0.5 s at 50 Hz for clench min
# Non-blocking SpO2 AC check after IR gate (never blocks Protocol A)
SPO2_AC_SAMPLE_S = 12.0
SPO2_AC_REL_MIN = 0.0005  # ac_proxy / median; below → WARN dead perfusion


async def _oralable_logger(
    address: str,
    out_path: Path,
    stop_event: asyncio.Event,
    verbose: bool,
) -> None:
    from bleak import BleakClient

    chars = [PPG_CHAR_UUID, ACCEL_CHAR_UUID, TEMP_CHAR_UUID, BATTERY_CHAR_UUID]
    count = 0
    ppg_count = 0
    accel_count = 0
    last_ppg_mono = 0.0
    last_dev_status: dict[str, int] | None = None
    loop = asyncio.get_event_loop()

    def make_handler(uuid: str):
        def _handler(_sender, data: bytearray) -> None:
            nonlocal count, ppg_count, accel_count, last_ppg_mono
            line = _format_log_line(uuid, bytes(data))
            with out_path.open("a", encoding="utf-8") as f:
                f.write(line)
            count += 1
            if uuid.lower() == PPG_CHAR_UUID.lower():
                ppg_count += 1
                last_ppg_mono = loop.time()
            elif uuid.lower() == ACCEL_CHAR_UUID.lower():
                accel_count += 1
            if verbose and count % 50 == 0:
                print(
                    f"  [{USER_FACING_DEVICE_NAME}] n={count} ppg={ppg_count} acc={accel_count}",
                    flush=True,
                )

        return _handler

    print(f"[{USER_FACING_DEVICE_NAME}] Connecting {address}...")
    print(format_last_known())
    async with BleakClient(address, timeout=30.0) as client:
        firmware = await read_and_persist_firmware(client, address)
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        with out_path.open("a", encoding="utf-8") as f:
            f.write(f"{ts} - {firmware_log_header(firmware, address).strip()}\n")
        try:
            await write_worn_and_stream(client)
            print(f"[{USER_FACING_DEVICE_NAME}] worn + stream mask written")
        except Exception as e:
            print(f"[{USER_FACING_DEVICE_NAME}] config write skipped/failed: {e}")

        for uuid in chars:
            await client.start_notify(uuid, make_handler(uuid))

        def batt_stats_handler(_sender, data: bytearray) -> None:
            parsed = _parse_battery_stats_6byte(bytes(data))
            if parsed:
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(f"{ts} - [BATT] {parsed}\n")

        def status_handler(_sender, data: bytearray) -> None:
            nonlocal last_dev_status
            parsed = parse_status_5byte(bytes(data))
            if not parsed:
                return
            last_dev_status = parsed
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
            with out_path.open("a", encoding="utf-8") as f:
                f.write(f"{ts} - [STATUS] {format_status_line(parsed)}\n")
            if parsed["worn"] == 0:
                print(
                    f"\n[Oralable] WARNING worn=0 ({format_status_line(parsed)})",
                    flush=True,
                )

        try:
            await client.start_notify(BATTERY_STATS_CHAR_UUID, batt_stats_handler)
        except Exception:
            pass
        try:
            await client.start_notify(STATUS_CHAR_UUID, status_handler)
            print("[Oralable] Status subscribed")
        except Exception:
            pass
        # FwLog CCC is off during Dual A capture — 00A notify starved PPG on 1.0.82.

        print(f"[Oralable] Logging → {out_path}")
        while not stop_event.is_set():
            if last_ppg_mono > 0 and (loop.time() - last_ppg_mono) >= PPG_STALL_ABORT_S:
                extra = ""
                if last_dev_status:
                    extra = f" [{format_status_line(last_dev_status)}]"
                print(
                    f"\n[Oralable] ERROR: PPG stalled {PPG_STALL_ABORT_S:.0f}s{extra}",
                    file=sys.stderr,
                )
                stop_event.set()
                break
            await asyncio.sleep(0.5)
        for uuid in chars:
            try:
                await client.stop_notify(uuid)
            except Exception:
                pass
    print(f"[Oralable] stopped samples≈{count}")


async def _anr_logger(
    address: str,
    out_path: Path,
    stop_event: asyncio.Event,
    verbose: bool,
) -> None:
    from bleak import BleakClient

    count = 0

    def on_emg(_sender, data: bytearray) -> None:
        nonlocal count
        if len(data) < 2:
            return
        emg = struct.unpack_from("<H", data, 0)[0]
        if emg > 1023:
            return
        ts = datetime.now(timezone.utc)
        line = f"{ts.isoformat()},{ts.timestamp():.6f},{emg}\n"
        with out_path.open("a", encoding="utf-8") as f:
            f.write(line)
        count += 1
        if verbose and count % 20 == 0:
            print(f"  [ANR] emg={emg} n={count}", flush=True)

    print(f"[ANR] Connecting {address}...")
    async with BleakClient(address, timeout=30.0) as client:
        if out_path.stat().st_size == 0:
            out_path.write_text(
                "# ANR_M40 EMG log — iso_utc,unix_s,emg_raw\n"
                "iso_utc,unix_s,emg_raw\n",
                encoding="utf-8",
            )
        try:
            batt = await client.read_gatt_char(BATTERY_LEVEL_CHAR)
            if batt:
                print(f"[ANR] Battery: {batt[0]}%")
        except Exception:
            pass
        await client.start_notify(ANALOG_EMG_CHAR, on_emg)
        print(f"[ANR] Logging → {out_path}")
        while not stop_event.is_set():
            await asyncio.sleep(0.5)
        try:
            await client.stop_notify(ANALOG_EMG_CHAR)
        except Exception:
            pass
    print(f"[ANR] stopped samples={count}")


def _print_anr_placement_reminder() -> None:
    print("\n" + "=" * 60)
    print("DUAL A SETUP — seat MAM alone first")
    print("=" * 60)
    print("  1) MAM ONLY on anterior temporalis peak (PPG on skin)")
    print("     — no ANR yet")
    print("  2) Hard clench — confirm IR-DC trough (live vitals or gate next)")
    print("     — if no trough: re-seat MAM; do not add ANR yet")
    print("  3) Then Kapton; ANR Red Dots VERTICAL on same belly")
    print("     — press stack without sliding / covering the optical window")
    print("  4) Fresh gel · snaps seated · optional headband")
    print("  Not a full MAM-only Protocol A — short optical check, then stack.")
    print("  EMG gate alone is not enough — IR must move on clench too.")
    print("=" * 60)


def _ppg_samples_from_notify(
    data: bytes, channel_order: str = PPG_CHANNEL_ORDER
) -> list[tuple[int, int, int]]:
    """Extract (green, red, ir) ADC triples from one PPG notify payload."""
    if len(data) < 16:
        return []
    n = (len(data) - 4) // 12
    out: list[tuple[int, int, int]] = []
    for i in range(n):
        j = 4 + i * 12
        if j + 12 > len(data):
            break
        v0, v1, v2 = struct.unpack_from("<III", data, j)
        sample = _parse_ppg_sample([v0, v1, v2], 0, bits=32, channel_order=channel_order)
        if sample is None:
            continue
        g, r, ir_val = sample
        out.append((int(g), int(r), int(ir_val)))
    return out


def _ir_samples_from_ppg_notify(data: bytes, channel_order: str = PPG_CHANNEL_ORDER) -> list[int]:
    """Extract IR ADC values from one PPG notify payload (frame counter + samples)."""
    return [ir for _g, _r, ir in _ppg_samples_from_notify(data, channel_order) if ir > 0]


def _ac_proxy(vals: list[float], win: int = 25) -> float:
    """Median |x - rolling_mean| — live AC amplitude without filtfilt."""
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=float)
    if len(arr) < win:
        return float(np.std(arr))
    # causal trailing mean
    means = np.convolve(arr, np.ones(win) / win, mode="valid")
    det = arr[win - 1 :] - means
    return float(np.median(np.abs(det)))


def _rolling_min(vals: list[float], win: int) -> float:
    if not vals:
        return float("nan")
    if len(vals) < win:
        return float(min(vals))
    arr = np.asarray(vals, dtype=float)
    # causal-ish: min of each trailing window; take global min of those
    mins = [float(arr[i : i + win].min()) for i in range(0, len(arr) - win + 1)]
    return float(min(mins)) if mins else float(arr.min())


async def _anr_emg_preflight(
    address: str,
    emg_gate: int,
    verbose: bool,
) -> dict[str, float | int | bool]:
    """Rest → hard clench → rest; return stats. Pass if clench max >= emg_gate."""
    from bleak import BleakClient

    samples: list[tuple[float, int]] = []  # (elapsed_s, emg)
    t0 = asyncio.get_event_loop().time()

    def on_emg(_sender, data: bytearray) -> None:
        if len(data) < 2:
            return
        emg = struct.unpack_from("<H", data, 0)[0]
        if emg > 1023:
            return
        elapsed = asyncio.get_event_loop().time() - t0
        samples.append((elapsed, emg))
        if verbose and len(samples) % 10 == 0:
            print(f"  [preflight] t={elapsed:4.1f}s emg={emg}", flush=True)

    total = PREFLIGHT_REST_S + PREFLIGHT_CLENCH_S + PREFLIGHT_REST_S
    print(f"\n[preflight] Connecting ANR {address}...")
    async with BleakClient(address, timeout=30.0) as client:
        try:
            batt = await client.read_gatt_char(BATTERY_LEVEL_CHAR)
            if batt:
                print(f"[preflight] ANR battery: {batt[0]}%")
        except Exception:
            pass
        await client.start_notify(ANALOG_EMG_CHAR, on_emg)

        print(f"\n>>> [0:00] REST — jaw relaxed ({PREFLIGHT_REST_S:.0f} s)")
        await asyncio.sleep(PREFLIGHT_REST_S)
        print(
            f"\n>>> [{_fmt_mmss(PREFLIGHT_REST_S)}] HARD CLENCH — hold max "
            f"({PREFLIGHT_CLENCH_S:.0f} s)"
        )
        await asyncio.sleep(PREFLIGHT_CLENCH_S)
        print(
            f"\n>>> [{_fmt_mmss(PREFLIGHT_REST_S + PREFLIGHT_CLENCH_S)}] "
            f"REST — release ({PREFLIGHT_REST_S:.0f} s)"
        )
        await asyncio.sleep(PREFLIGHT_REST_S)

        try:
            await client.stop_notify(ANALOG_EMG_CHAR)
        except Exception:
            pass

    def _window(t0_s: float, t1_s: float) -> list[int]:
        return [v for t, v in samples if t0_s <= t < t1_s]

    rest1 = _window(0.0, PREFLIGHT_REST_S)
    clench = _window(PREFLIGHT_REST_S, PREFLIGHT_REST_S + PREFLIGHT_CLENCH_S)
    rest2 = _window(PREFLIGHT_REST_S + PREFLIGHT_CLENCH_S, total)

    def _stats(vals: list[int]) -> tuple[float, float]:
        if not vals:
            return float("nan"), float("nan")
        return float(sum(vals) / len(vals)), float(max(vals))

    r1_mean, r1_max = _stats(rest1)
    c_mean, c_max = _stats(clench)
    r2_mean, r2_max = _stats(rest2)
    passed = bool(clench) and c_max >= float(emg_gate)

    print("\n[preflight] EMG summary (raw 0–1023)")
    print(f"  rest1   mean={r1_mean:6.1f}  max={r1_max:6.1f}  n={len(rest1)}")
    print(f"  clench  mean={c_mean:6.1f}  max={c_max:6.1f}  n={len(clench)}  gate>={emg_gate}")
    print(f"  rest2   mean={r2_mean:6.1f}  max={r2_max:6.1f}  n={len(rest2)}")
    if passed:
        print(f"[preflight] PASS — clench max {c_max:.0f} ≥ {emg_gate}")
    else:
        print(f"[preflight] FAIL — clench max {c_max if clench else 0:.0f} < {emg_gate}")
        print("  Re-seat: vertical Red Dots on peak bulge · gel · pressure · re-try")

    return {
        "passed": passed,
        "emg_gate": emg_gate,
        "n_samples": len(samples),
        "rest1_mean": r1_mean,
        "rest1_max": r1_max,
        "clench_mean": c_mean,
        "clench_max": c_max,
        "rest2_mean": r2_mean,
        "rest2_max": r2_max,
    }


async def _run_emg_preflight_until_pass(
    anr_addr: str,
    emg_gate: int,
    verbose: bool,
    force: bool,
) -> dict[str, float | int | bool]:
    """Retry preflight until gate passes, or force-continue after fail."""
    attempt = 0
    while True:
        attempt += 1
        _print_anr_placement_reminder()
        print(f"[preflight] Attempt {attempt} — ~{int(PREFLIGHT_REST_S * 2 + PREFLIGHT_CLENCH_S)} s")
        stats = await _anr_emg_preflight(anr_addr, emg_gate, verbose)
        if stats["passed"]:
            return stats
        if force:
            print("[preflight] --force: continuing to Protocol A despite FAIL")
            return stats
        print("\nPress Enter to re-seat and retry preflight, or Ctrl+C to abort.")
        print("(Or re-run with --force / --skip-emg-preflight)")
        try:
            await asyncio.to_thread(input)
        except EOFError:
            print("[preflight] No TTY — aborting. Use --force or --skip-emg-preflight.")
            raise SystemExit(2)


async def _oralable_ir_preflight(
    address: str,
    ir_drop_gate: float,
    verbose: bool,
) -> dict[str, float | int | bool]:
    """Rest → hard clench → rest; PASS if IR drops ≥ gate fraction vs rest median."""
    from bleak import BleakClient

    samples: list[tuple[float, float]] = []  # (elapsed_s, ir)
    t0 = asyncio.get_event_loop().time()

    def on_ppg(_sender, data: bytearray) -> None:
        elapsed = asyncio.get_event_loop().time() - t0
        for ir_val in _ir_samples_from_ppg_notify(bytes(data)):
            samples.append((elapsed, float(ir_val)))
            if verbose and len(samples) % 50 == 0:
                print(f"  [ir-preflight] t={elapsed:4.1f}s ir={ir_val}", flush=True)

    total = PREFLIGHT_REST_S + PREFLIGHT_CLENCH_S + PREFLIGHT_REST_S
    print(f"\n[ir-preflight] Connecting {USER_FACING_DEVICE_NAME} {address}...")
    async with BleakClient(address, timeout=30.0) as client:
        await read_and_persist_firmware(client, address)
        try:
            await write_worn_and_stream(client)
            print("[ir-preflight] worn + PPG|ACC stream enabled")
        except Exception as e:
            print(f"[ir-preflight] config write skipped/failed: {e}")

        await client.start_notify(PPG_CHAR_UUID, on_ppg)

        print(f"\n>>> [0:00] REST — jaw relaxed ({PREFLIGHT_REST_S:.0f} s) [IR optical]")
        await asyncio.sleep(PREFLIGHT_REST_S)
        print(
            f"\n>>> [{_fmt_mmss(PREFLIGHT_REST_S)}] HARD CLENCH — hold max "
            f"({PREFLIGHT_CLENCH_S:.0f} s) [watch IR-DC trough]"
        )
        await asyncio.sleep(PREFLIGHT_CLENCH_S)
        print(
            f"\n>>> [{_fmt_mmss(PREFLIGHT_REST_S + PREFLIGHT_CLENCH_S)}] "
            f"REST — release ({PREFLIGHT_REST_S:.0f} s)"
        )
        await asyncio.sleep(PREFLIGHT_REST_S)

        try:
            await client.stop_notify(PPG_CHAR_UUID)
        except Exception:
            pass

    def _window(t0_s: float, t1_s: float) -> list[float]:
        return [v for t, v in samples if t0_s <= t < t1_s]

    rest1 = _window(0.0, PREFLIGHT_REST_S)
    clench = _window(PREFLIGHT_REST_S, PREFLIGHT_REST_S + PREFLIGHT_CLENCH_S)
    rest2 = _window(PREFLIGHT_REST_S + PREFLIGHT_CLENCH_S, total)

    rest_med = float(np.median(rest1)) if rest1 else float("nan")
    clench_min = _rolling_min(clench, IR_ROLL_SAMPLES)
    if rest_med > 0 and np.isfinite(clench_min):
        drop_frac = float((rest_med - clench_min) / rest_med)
    else:
        drop_frac = float("nan")

    band_ok = bool(
        rest1
        and np.isfinite(rest_med)
        and IR_REST_MEDIAN_MIN <= rest_med <= IR_REST_MEDIAN_MAX
    )
    drop_ok = bool(np.isfinite(drop_frac) and drop_frac >= float(ir_drop_gate))
    passed = band_ok and drop_ok

    print("\n[ir-preflight] IR optical summary (raw ADC)")
    print(f"  rest1   median={rest_med:.0f}  n={len(rest1)}  band=[{IR_REST_MEDIAN_MIN:.0f},{IR_REST_MEDIAN_MAX:.0g}]")
    print(f"  clench  roll_min={clench_min:.0f}  n={len(clench)}  drop_frac={drop_frac:.3f}  gate>={ir_drop_gate:.3f}")
    print(f"  rest2   n={len(rest2)}")
    if passed:
        print(f"[ir-preflight] PASS — IR drop {drop_frac:.1%} ≥ {ir_drop_gate:.1%} (rest median OK)")
    else:
        print("[ir-preflight] FAIL — optical coupling / occlusion not clear on clench")
        if not band_ok:
            print("  Rest IR median outside sane band — re-seat PPG window on skin")
        if not drop_ok:
            print("  Clench did not drop IR enough — pressure on window, not shear; retry")
        print("  Tip: seat Oralable alone first; add ANR only after IR trough is visible")

    return {
        "passed": passed,
        "ir_drop_gate": float(ir_drop_gate),
        "n_samples": len(samples),
        "rest_median": rest_med,
        "clench_min": float(clench_min) if np.isfinite(clench_min) else float("nan"),
        "drop_frac": drop_frac,
        "band_ok": band_ok,
    }


async def _run_ir_preflight_until_pass(
    oral_addr: str,
    ir_drop_gate: float,
    verbose: bool,
    force: bool,
) -> dict[str, float | int | bool]:
    """Retry IR optical preflight until gate passes, or force-continue after fail."""
    attempt = 0
    while True:
        attempt += 1
        print("\n" + "=" * 60)
        print("IR OPTICAL PREFLIGHT (Oralable)")
        print("=" * 60)
        print(f"[ir-preflight] Attempt {attempt} — ~{int(PREFLIGHT_REST_S * 2 + PREFLIGHT_CLENCH_S)} s")
        stats = await _oralable_ir_preflight(oral_addr, ir_drop_gate, verbose)
        if stats["passed"]:
            return stats
        if force:
            print("[ir-preflight] --force: continuing to Protocol A despite FAIL")
            return stats
        print("\nPress Enter to re-seat Oralable/ANR and retry IR preflight, or Ctrl+C to abort.")
        print("(Or re-run with --force / --skip-ir-preflight)")
        try:
            await asyncio.to_thread(input)
        except EOFError:
            print("[ir-preflight] No TTY — aborting. Use --force or --skip-ir-preflight.")
            raise SystemExit(2)


async def _oralable_spo2_ac_warn(address: str, verbose: bool) -> dict[str, float | int | bool]:
    """
    Quiet ~12 s red/IR sample after IR gate. WARN if AC dead — never blocks Protocol A.
    """
    from bleak import BleakClient

    reds: list[float] = []
    irs: list[float] = []

    def on_ppg(_sender, data: bytearray) -> None:
        for _g, r, ir_val in _ppg_samples_from_notify(bytes(data)):
            if r > 0:
                reds.append(float(r))
            if ir_val > 0:
                irs.append(float(ir_val))
            if verbose and len(irs) % 50 == 0:
                print(f"  [spo2-ac] n_ir={len(irs)} ir={ir_val} red={r}", flush=True)

    print("\n" + "=" * 60)
    print("SpO₂ AC CHECK (non-blocking WARN)")
    print("=" * 60)
    print(f"[spo2-ac] Quiet jaw ~{SPO2_AC_SAMPLE_S:.0f} s — checking red/IR pulsatile AC")
    async with BleakClient(address, timeout=30.0) as client:
        await read_and_persist_firmware(client, address)
        try:
            await write_worn_and_stream(client)
        except Exception as e:
            print(f"[spo2-ac] config write skipped/failed: {e}")
        await client.start_notify(PPG_CHAR_UUID, on_ppg)
        await asyncio.sleep(SPO2_AC_SAMPLE_S)
        try:
            await client.stop_notify(PPG_CHAR_UUID)
        except Exception:
            pass

    red_med = float(np.median(reds)) if reds else float("nan")
    ir_med = float(np.median(irs)) if irs else float("nan")
    red_ac = _ac_proxy(reds)
    ir_ac = _ac_proxy(irs)
    red_rel = float(red_ac / red_med) if red_med > 0 and np.isfinite(red_ac) else float("nan")
    ir_rel = float(ir_ac / ir_med) if ir_med > 0 and np.isfinite(ir_ac) else float("nan")
    ac_ok = bool(
        np.isfinite(red_rel)
        and np.isfinite(ir_rel)
        and red_rel >= SPO2_AC_REL_MIN
        and ir_rel >= SPO2_AC_REL_MIN
    )

    print(f"[spo2-ac] red median={red_med:.0f} ac={red_ac:.1f} rel={red_rel:.5f}  n={len(reds)}")
    print(f"[spo2-ac] ir  median={ir_med:.0f} ac={ir_ac:.1f} rel={ir_rel:.5f}  n={len(irs)}")
    if ac_ok:
        print("[spo2-ac] OK — pulsatile AC present (engineering SpO2 context)")
    else:
        print("[spo2-ac] WARN — SpO₂ will be junk (red/IR AC dead or too weak)")
        print("  Continuing to Protocol A (IR/EMG gates already decided). Re-seat for oxygen arm.")

    return {
        "ac_ok": ac_ok,
        "red_ac": red_ac,
        "ir_ac": ir_ac,
        "red_rel": red_rel,
        "ir_rel": ir_rel,
        "n_red": len(reds),
        "n_ir": len(irs),
    }


async def _run_dual(
    oral_addr: str,
    anr_addr: str,
    oral_path: Path,
    anr_path: Path,
    duration_s: float,
    verbose: bool,
    pair_path: Path | None = None,
    preflight_stats: dict[str, float | int | bool] | None = None,
    ir_preflight_stats: dict[str, float | int | bool] | None = None,
    spo2_ac_stats: dict[str, float | int | bool] | None = None,
) -> None:
    stop_event = asyncio.Event()
    print("\n" + "=" * 60)
    print("DUAL PROTOCOL A — Oralable + ANR (shared cues)")
    print("ANR long-axis VERTICAL on anterior temporalis")
    print("At 01:00: FIVE taps on Oralable housing (EMG may also spike)")
    print("=" * 60)

    oral_task = asyncio.create_task(
        _oralable_logger(oral_addr, oral_path, stop_event, verbose)
    )
    anr_task = asyncio.create_task(
        _anr_logger(anr_addr, anr_path, stop_event, verbose)
    )
    # Brief settle so both CCC enables complete before cues
    await asyncio.sleep(2.0)
    # Cue T=0 after CCC settle; concordance defaults to first-sample align
    # (both streams start in the settle window). Pair file records cue wall-clock.
    t_cue = datetime.now(timezone.utc).timestamp()
    if pair_path is not None:
        with pair_path.open("a", encoding="utf-8") as f:
            f.write(f"cue_start_unix={t_cue:.6f}\n")
            f.write("# align: default first-sample; use --emg-offset-s if needed\n")
            if preflight_stats is not None:
                f.write(
                    f"emg_preflight_passed={int(bool(preflight_stats.get('passed')))}\n"
                    f"emg_preflight_gate={preflight_stats.get('emg_gate')}\n"
                    f"emg_preflight_clench_max={preflight_stats.get('clench_max')}\n"
                    f"emg_preflight_clench_mean={preflight_stats.get('clench_mean')}\n"
                )
            if ir_preflight_stats is not None:
                f.write(
                    f"ir_preflight_passed={int(bool(ir_preflight_stats.get('passed')))}\n"
                    f"ir_preflight_drop_frac={ir_preflight_stats.get('drop_frac')}\n"
                    f"ir_preflight_rest_median={ir_preflight_stats.get('rest_median')}\n"
                    f"ir_preflight_gate={ir_preflight_stats.get('ir_drop_gate')}\n"
                )
            if spo2_ac_stats is not None:
                f.write(
                    f"spo2_preflight_ac_ok={int(bool(spo2_ac_stats.get('ac_ok')))}\n"
                    f"spo2_preflight_red_ac={spo2_ac_stats.get('red_ac')}\n"
                    f"spo2_preflight_ir_ac={spo2_ac_stats.get('ir_ac')}\n"
                    f"spo2_preflight_red_rel={spo2_ac_stats.get('red_rel')}\n"
                    f"spo2_preflight_ir_rel={spo2_ac_stats.get('ir_rel')}\n"
                )
    cue_task = asyncio.create_task(_cue_loop(stop_event, duration_s))

    await cue_task
    stop_event.set()
    await asyncio.gather(oral_task, anr_task, return_exceptions=True)
    print(f"\n[{_fmt_mmss(duration_s)}] Dual session complete.")


async def _dry_run(duration_s: float) -> None:
    stop = asyncio.Event()
    print("DRY RUN — cues only (no BLE)")
    _print_anr_placement_reminder()
    print(
        f"[dry-run] Would run ANR EMG preflight: rest {PREFLIGHT_REST_S:.0f}s → "
        f"clench {PREFLIGHT_CLENCH_S:.0f}s → rest {PREFLIGHT_REST_S:.0f}s "
        f"(gate max>={DEFAULT_EMG_GATE})"
    )
    print(
        f"[dry-run] Would run Oralable IR preflight: same windows "
        f"(drop gate>={DEFAULT_IR_DROP_GATE:.0%})"
    )
    print(
        f"[dry-run] Would run SpO₂ AC WARN check (~{SPO2_AC_SAMPLE_S:.0f}s, non-blocking)"
    )
    await _cue_loop(stop, duration_s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Dual Protocol A: Oralable + ANR")
    ap.add_argument("--oralable-address", type=str)
    ap.add_argument("--anr-address", type=str)
    ap.add_argument("--duration", type=float, default=SESSION_DURATION_S)
    ap.add_argument("--oralable-out", type=Path)
    ap.add_argument("--anr-out", type=Path)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--emg-gate",
        type=int,
        default=DEFAULT_EMG_GATE,
        help=f"ANR clench-window max must reach this (default {DEFAULT_EMG_GATE})",
    )
    ap.add_argument(
        "--skip-emg-preflight",
        action="store_true",
        help="Skip rest/clench/rest ANR placement gate",
    )
    ap.add_argument(
        "--ir-drop-gate",
        type=float,
        default=DEFAULT_IR_DROP_GATE,
        help=f"IR drop fraction vs rest median (default {DEFAULT_IR_DROP_GATE})",
    )
    ap.add_argument(
        "--skip-ir-preflight",
        action="store_true",
        help="Skip Oralable IR optical clench gate",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Continue to Protocol A even if EMG or IR preflight fails",
    )
    args = ap.parse_args()

    if args.dry_run:
        try:
            asyncio.run(_dry_run(args.duration))
        except KeyboardInterrupt:
            print("\nInterrupted.")
        return 0

    try:
        from bleak import BleakClient  # noqa: F401
    except ImportError:
        print("Install bleak: pip install bleak", file=sys.stderr)
        return 1

    oral_addr = args.oralable_address
    anr_addr = args.anr_address
    if not oral_addr:
        print(f"Scanning for {USER_FACING_DEVICE_NAME} (advertises as Oralable)...")
        print(format_last_known())
        oral_addr = asyncio.run(_scan_for_oralable())
        if not oral_addr:
            print(
                f"{USER_FACING_DEVICE_NAME} not found. Disconnect iOS, wake device.",
                file=sys.stderr,
            )
            print(format_last_known(), file=sys.stderr)
            return 1
        print(f"{USER_FACING_DEVICE_NAME}: {oral_addr}")
    if not anr_addr:
        anr_addr = asyncio.run(scan_for_anr())
        if not anr_addr:
            print("ANR not found. Power on M40, disconnect phone apps.", file=sys.stderr)
            return 1
        print(f"ANR: {anr_addr}")

    preflight_stats: dict[str, float | int | bool] | None = None
    ir_preflight_stats: dict[str, float | int | bool] | None = None
    spo2_ac_stats: dict[str, float | int | bool] | None = None
    if not args.skip_emg_preflight:
        try:
            preflight_stats = asyncio.run(
                _run_emg_preflight_until_pass(
                    anr_addr,
                    args.emg_gate,
                    args.verbose,
                    args.force,
                )
            )
        except KeyboardInterrupt:
            print("\nInterrupted during EMG preflight.")
            return 130
        except SystemExit as e:
            return int(e.code) if isinstance(e.code, int) else 2
        print("[preflight] Settling 2 s before IR optical preflight...")
        asyncio.run(asyncio.sleep(2.0))

    if not args.skip_ir_preflight:
        try:
            ir_preflight_stats = asyncio.run(
                _run_ir_preflight_until_pass(
                    oral_addr,
                    args.ir_drop_gate,
                    args.verbose,
                    args.force,
                )
            )
        except KeyboardInterrupt:
            print("\nInterrupted during IR preflight.")
            return 130
        except SystemExit as e:
            return int(e.code) if isinstance(e.code, int) else 2
        # Non-blocking SpO2 AC WARN after IR gate (pass or --force continue)
        print("[preflight] Settling 2 s before SpO₂ AC check...")
        asyncio.run(asyncio.sleep(2.0))
        try:
            spo2_ac_stats = asyncio.run(
                _oralable_spo2_ac_warn(oral_addr, args.verbose)
            )
        except KeyboardInterrupt:
            print("\nInterrupted during SpO₂ AC check — continuing.")
            spo2_ac_stats = None
        except Exception as e:
            print(f"[spo2-ac] skipped ({e})")
            spo2_ac_stats = None
        print("[preflight] Settling 2 s before Dual A connect...")
        asyncio.run(asyncio.sleep(2.0))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    oral_path = args.oralable_out or (RAW_DIR / f"TEMPORALIS_RAW_{ts}.txt")
    anr_path = args.anr_out or (RAW_DIR / f"ANR_EMG_{ts}.txt")
    oral_path.parent.mkdir(parents=True, exist_ok=True)
    anr_path.parent.mkdir(parents=True, exist_ok=True)
    oral_path.write_text("", encoding="utf-8")
    anr_path.write_text("", encoding="utf-8")

    # Pairing sidecar for concordance
    pair_path = RAW_DIR / f"DUAL_PAIR_{ts}.txt"
    pair_path.write_text(
        f"session_ts={ts}\noralable={oral_path}\nanr={anr_path}\n"
        f"oralable_address={oral_addr}\nanr_address={anr_addr}\n",
        encoding="utf-8",
    )

    try:
        asyncio.run(
            _run_dual(
                oral_addr,
                anr_addr,
                oral_path,
                anr_path,
                args.duration,
                args.verbose,
                pair_path=pair_path,
                preflight_stats=preflight_stats,
                ir_preflight_stats=ir_preflight_stats,
                spo2_ac_stats=spo2_ac_stats,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("\nNext:")
    print("  .venv/bin/python scripts/align_anr_oralable_concordance.py \\")
    print(f"    --oralable {oral_path} \\")
    print(f"    --anr {anr_path}")
    print(f"  Pair file: {pair_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
