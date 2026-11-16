"""Parse 3A0FF009 status used by Mac Protocol A watchdog."""

import importlib.util
from pathlib import Path

from src.utils.ble_logger import (
    ACCEL_CHAR_UUID,
    BATTERY_CHAR_UUID,
    FW_LOG_CHAR_UUID,
    PPG_CHAR_UUID,
    STATUS_CHAR_UUID,
    format_status_line,
    parse_status_5byte,
)


def _load_protocol_a_session():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_protocol_a_session.py"
    spec = importlib.util.spec_from_file_location("run_protocol_a_session", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_status_5byte_worn_off_dock():
    parsed = parse_status_5byte(bytes([0, 1, 2, 54, 0]))
    assert parsed == {
        "on_dock": 0,
        "worn": 1,
        "device_state": 2,
        "battery_pct": 54,
        "charge_active": 0,
    }
    assert "worn=1" in format_status_line(parsed)
    assert "on_dock=0" in format_status_line(parsed)


def test_parse_status_5byte_pad_not_worn():
    parsed = parse_status_5byte(bytes([1, 0, 1, 11, 1]))
    assert parsed is not None
    assert parsed["on_dock"] == 1
    assert parsed["worn"] == 0
    assert parsed["charge_active"] == 1


def test_parse_status_rejects_short():
    assert parse_status_5byte(b"\x00\x01") is None


def test_protocol_a_ccc_order_ppg_before_optional_fwlog():
    uuids = [uuid for uuid, _label in _load_protocol_a_session().PROTOCOL_A_CCC_ORDER]
    assert uuids.index(BATTERY_CHAR_UUID) < uuids.index(STATUS_CHAR_UUID)
    assert uuids.index(STATUS_CHAR_UUID) < uuids.index(PPG_CHAR_UUID)
    assert uuids.index(PPG_CHAR_UUID) < uuids.index(ACCEL_CHAR_UUID)
    assert FW_LOG_CHAR_UUID not in uuids
