"""Regression checks for Python's TGM GATT contract-facing helpers."""

from pathlib import Path
import struct

from src.parser.log_parser import BATTERY_CHAR_UUIDS
from src.utils.ble_logger import (
    BATTERY_STATS_CHAR_UUID,
    BATTERY_USAGE_CHAR_UUID,
    _parse_battery_stats_6byte,
    _parse_battery_usage_8byte,
)


def test_battery_sources_use_firmware_defined_characteristics() -> None:
    assert BATTERY_USAGE_CHAR_UUID == "3A0FF00D-98C4-46B2-94AF-1AEE0FD4C48E"
    assert BATTERY_STATS_CHAR_UUID == BATTERY_USAGE_CHAR_UUID
    assert BATTERY_CHAR_UUIDS == (
        "3A0FF004-98C4-46B2-94AF-1AEE0FD4C48E",
        "3A0FF00D-98C4-46B2-94AF-1AEE0FD4C48E",
    )


def test_battery_usage_payload_is_little_endian() -> None:
    payload = bytes([0xD2, 0x04, 87, 0x05, 0x39, 0x30, 0x2A, 0x00])
    expected = "V: 1234mV | %: 87% | Flags: 0x05 | Used: 123.45mAh | Rem: 42min"

    assert _parse_battery_usage_8byte(payload) == expected
    assert _parse_battery_stats_6byte(payload) == expected
    assert _parse_battery_usage_8byte(payload[:7]) is None


def test_battery_usage_marks_unknown_remaining_budget() -> None:
    payload = bytes([0xE4, 0x0C, 100, 0x00, 0x00, 0x00, 0xFF, 0xFF])

    assert _parse_battery_usage_8byte(payload) == (
        "V: 3300mV | %: 100% | Flags: 0x00 | Used: 0.00mAh | Rem: n/a"
    )


def test_a2_golden_wire_vectors() -> None:
    ppg_pkt = struct.pack("<IIII", 0x01020304, 1, 0x12345, 0x7FFFF)
    assert ppg_pkt == bytes([
        0x04, 0x03, 0x02, 0x01,
        0x01, 0x00, 0x00, 0x00,
        0x45, 0x23, 0x01, 0x00,
        0xFF, 0xFF, 0x07, 0x00,
    ])
    assert 4 + 12 * 20 == 244

    acc_pkt = struct.pack("<Ihhh", 0x01020304, -1, 0x1234, -32768)
    assert acc_pkt == bytes([
        0x04, 0x03, 0x02, 0x01,
        0xFF, 0xFF, 0x34, 0x12, 0x00, 0x80,
    ])
    assert 4 + 6 * 25 == 154

    temp_pkt = struct.pack("<IhBB", 1, 2137, 0, 0)
    assert temp_pkt == bytes([0x01, 0x00, 0x00, 0x00, 0x59, 0x08, 0x00, 0x00])


def test_firmware_headers_lock_50hz_and_fifo_bounds() -> None:
    nrf_root = Path(__file__).resolve().parents[2] / "oralable_nrf"
    maxm_h = (nrf_root / "include/app/drivers/maxm86161.h").read_text(encoding="utf-8")
    lis_h = (nrf_root / "include/app/drivers/lis2dtw12.h").read_text(encoding="utf-8")
    tgm_h = (nrf_root / "app/src/tgm_service.h").read_text(encoding="utf-8")

    assert "MAXM86161_PPG_SR_50SPS 0x01" in maxm_h
    assert "MAXM86161_SMP_AVE_NONE 0x00" in maxm_h
    assert "MAXM86161_FIFO_MAX_BURST_WORDS 60" in maxm_h
    assert "MAXM86161_FIFO_MAX_BURST_BYTES 180" in maxm_h

    assert "LIS2DTW12_CTRL1_ODR_50HZ 0x40" in lis_h
    assert "LIS2DTW12_CTRL1_LP_MODE_4 0x03" in lis_h
    assert "LIS2DTW12_FIFO_MAX_BURST_SAMPLES 25" in lis_h
    assert "LIS2DTW12_FIFO_MAX_BURST_BYTES 150" in lis_h

    assert "TGM_PPG_SAMPLE_STRIDE 12" in tgm_h
    assert "TGM_ACC_SAMPLE_STRIDE 6" in tgm_h
    assert "TGM_PPG_FULL_FRAME_MIN_ATT_MTU (TGM_PPG_FULL_FRAME_BYTES + 3)" in tgm_h
    assert "TGM_ACC_FULL_FRAME_MIN_ATT_MTU (TGM_ACC_FULL_FRAME_BYTES + 3)" in tgm_h
