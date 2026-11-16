#!/usr/bin/env python3
"""Verify the versioned TGM GATT contract against local sibling repositories.

The check is static and offline. It does not build, flash, connect to hardware,
or change any repository. Use --strict in CI once the four repositories are
provisioned as siblings of cursor_oralable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "contracts" / "tgm_gatt_v1.json"
FIRMWARE_HEADER = ROOT.parent / "oralable_nrf" / "app" / "src" / "tgm_service.h"
PYTHON_LOGGER = ROOT / "src" / "utils" / "ble_logger.py"
PYTHON_PARSER = ROOT / "src" / "parser" / "log_parser.py"
CORE_CONSTANTS = ROOT.parent / "OralableCore" / "Sources" / "OralableCore" / "BLE" / "BLEConstants.swift"
IOS_DEVICE = ROOT.parent / "oralable_swift" / "OralableApp" / "OralableApp" / "Devices" / "OralableDevice" / "OralableDevice.swift"


def text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def firmware_uuid_macros(source: str) -> dict[str, str]:
    pattern = re.compile(
        r"#define\s+(BT_UUID_TGM(?:_[A-Z_]+)?_VAL)\s+\\?\s*"
        r"BT_UUID_128_ENCODE\(0x([0-9a-fA-F]{8}),\s*0x98c4,\s*0x46b2,\s*0x94af,\s*0x1aee0fd4c48e\)",
        re.MULTILINE,
    )
    return {
        macro: f"{suffix.upper()}-98C4-46B2-94AF-1AEE0FD4C48E"
        for macro, suffix in pattern.findall(source)
    }


def check_contains(report: dict[str, list[str]], label: str, source: str, expected: str) -> None:
    if expected not in source:
        report["errors"].append(f"{label}: missing `{expected}`")


def check_contract(contract: dict[str, Any]) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {"passes": [], "errors": [], "warnings": []}
    characteristics = contract["characteristics"]

    try:
        firmware = text(FIRMWARE_HEADER)
        python_logger = text(PYTHON_LOGGER)
        python_parser = text(PYTHON_PARSER)
        core = text(CORE_CONSTANTS)
        ios = text(IOS_DEVICE)
    except FileNotFoundError as error:
        report["errors"].append(f"Required local source is missing: {error}")
        return report

    declared = firmware_uuid_macros(firmware)
    expected_firmware = {contract["service"]["firmware_macro"]: contract["service"]["uuid"]}
    expected_firmware.update({item["firmware_macro"]: item["uuid"] for item in characteristics})
    for macro, uuid in expected_firmware.items():
        actual = declared.get(macro)
        if actual != uuid:
            report["errors"].append(f"firmware: `{macro}` is `{actual}`, expected `{uuid}`")
    unexpected = sorted(set(declared) - set(expected_firmware))
    if unexpected:
        report["warnings"].append(f"firmware: unrecorded TGM UUID macros: {', '.join(unexpected)}")
    if not any(message.startswith("firmware:") for message in report["errors"]):
        report["passes"].append(f"firmware: {len(expected_firmware)} UUID macros match the contract")

    for item in characteristics:
        uuid = item["uuid"]
        swift_constant = item["swift_constant"]
        check_contains(report, "core", core, f'public static let {swift_constant} = "{uuid}"')
        check_contains(report, "ios", ios, f"BLEConstants.TGM.{swift_constant}")
    check_contains(report, "core", core, f'public static let serviceUUID = "{contract["service"]["uuid"]}"')
    check_contains(report, "ios", ios, "BLEConstants.TGM.serviceUUID")

    check_contains(report, "core", core, "public static let status = 5")
    check_contains(report, "core", core, "public static let temperatureCharUUID")
    if "3A0FFEF2-98C4-46B2-94AF-1AEE0FD4C48E" in python_logger:
        report["errors"].append("python logger: subscribes to unsupported current-firmware UUID 3A0FFEF2")
    if "3A0FF007-98C4-46B2-94AF-1AEE0FD4C48E" in python_parser or "3A0FF008-98C4-46B2-94AF-1AEE0FD4C48E" in python_parser:
        report["errors"].append("python parser: treats PPG register UUIDs 3A0FF007/008 as battery inputs")

    python_required = [
        item["uuid"]
        for item in characteristics
        if item["id"] in {"ppg", "accelerometer", "temperature", "battery_voltage", "device_status", "firmware_log", "battery_usage"}
    ]
    for uuid in python_required:
        check_contains(report, "python logger", python_logger, uuid)
    for item in characteristics:
        if item["id"] in {"ppg", "accelerometer", "temperature", "battery_voltage", "battery_usage"}:
            check_contains(report, "python parser", python_parser, item["uuid"])

    if not any(message.startswith("core:") for message in report["errors"]):
        report["passes"].append("core: all contract UUIDs, temperature name, and 5-byte status size are represented")
    if not any(message.startswith("ios:") for message in report["errors"]):
        report["passes"].append("ios: OralableDevice routes all TGM UUIDs through shared Core constants")
    if not any(message.startswith("python") for message in report["errors"]):
        report["passes"].append("python: logger/parser use canonical battery and streaming UUIDs")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true", help="Exit 2 on contract errors.")
    args = parser.parse_args()

    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        if contract.get("schema_version") != 1:
            raise ValueError("Unsupported contract schema_version")
        report = check_contract(contract)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"tgm-contract: {error}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"TGM GATT {contract['contract_version']} verification")
        for message in report["passes"]:
            print(f"PASS  {message}")
        for message in report["warnings"]:
            print(f"WARN  {message}")
        for message in report["errors"]:
            print(f"ERROR {message}")

    return 2 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
