"""Persist last MAM BLE connection so firmware is known even after a brief link.

GATT firmware string is `3A0FF006`. BLE still advertises as Oralable; the
user-facing device name is MAM. TGM remains the legacy protocol id.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

FIRMWARE_VERSION_CHAR_UUID = "3A0FF006-98C4-46B2-94AF-1AEE0FD4C48E"
USER_FACING_DEVICE_NAME = "MAM"
STORE_PATH = Path(__file__).resolve().parents[2] / "data" / "last_mam_connection.json"


def load_last_connection() -> dict | None:
    if not STORE_PATH.exists():
        return None
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_last_connection(
    *,
    firmware: str | None,
    address: str | None = None,
    extra: dict | None = None,
) -> Path:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "device": USER_FACING_DEVICE_NAME,
        "firmware": (firmware or "").strip() or None,
        "address": address,
        "connected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        payload.update(extra)
    previous = load_last_connection() or {}
    if payload["firmware"] is None and previous.get("firmware"):
        payload["firmware"] = previous["firmware"]
        payload["firmware_source"] = "previous"
    STORE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return STORE_PATH


def format_last_known() -> str:
    data = load_last_connection()
    if not data:
        return f"{USER_FACING_DEVICE_NAME} firmware: unknown (no prior connection recorded)"
    fw = data.get("firmware") or "unknown"
    when = data.get("connected_at") or "unknown time"
    src = data.get("firmware_source")
    extra = " [from previous connection]" if src == "previous" else ""
    return f"{USER_FACING_DEVICE_NAME} firmware {fw}{extra} (last connected {when})"


def known_firmware() -> str | None:
    data = load_last_connection() or {}
    fw = data.get("firmware")
    return str(fw).strip() if fw else None


def firmware_log_header(firmware: str | None, address: str | None = None) -> str:
    """First line of a session log so firmware is known even if the run aborts."""
    data = load_last_connection() or {}
    fw = (firmware or data.get("firmware") or "unknown")
    addr = address or data.get("address") or ""
    return f"# {USER_FACING_DEVICE_NAME} firmware {fw} address={addr}\n"


async def read_and_persist_firmware(client, address: str | None = None) -> str | None:
    """Read GATT 3A0FF006 and persist. Safe on a brief connect."""
    firmware = None
    try:
        raw = await client.read_gatt_char(FIRMWARE_VERSION_CHAR_UUID)
        firmware = bytes(raw).decode("utf-8", errors="replace").rstrip("\x00").strip() or None
    except Exception as exc:  # bleak / ATT errors
        print(f"  WARNING: {USER_FACING_DEVICE_NAME} firmware read failed: {exc}")
    save_last_connection(firmware=firmware, address=address)
    if firmware:
        print(f"  {USER_FACING_DEVICE_NAME} firmware: {firmware}")
    else:
        print(f"  {format_last_known()}")
    return firmware
