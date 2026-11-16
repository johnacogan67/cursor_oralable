#!/usr/bin/env python3
"""Validate a raw Gen2 PPG/accelerometer BLE-capture manifest offline.

A successful validation establishes only that the supplied capture is internally
well-formed and traceable to declared candidate metadata. It never changes a
Gen2 program gate, authorizes hardware work, or establishes a release claim.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "contracts" / "tgm_gatt_v1.json"
SCHEMA_VERSION = 1
COUNTER_MODULUS = 2**32

# The frame ceilings are current firmware configuration facts, not a rate claim.
STREAM_RULES = {
    "ppg": {"contract_id": "ppg", "bytes_per_sample": 12, "frame_ceiling": 20},
    "accelerometer": {
        "contract_id": "accelerometer",
        "bytes_per_sample": 6,
        "frame_ceiling": 25,
    },
}

HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def add_issue(report: dict[str, Any], severity: str, message: str) -> None:
    report[severity].append(message)


def nonempty(mapping: dict[str, Any], key: str, path: str, report: dict[str, Any]) -> str | None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        add_issue(report, "errors", f"{path}.{key} must be a non-empty string")
        return None
    return value.strip()


def parse_utc(value: str, path: str, report: dict[str, Any]) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        add_issue(report, "errors", f"{path} must be an ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None:
        add_issue(report, "errors", f"{path} must include a timezone")
        return None
    return parsed


def contract_streams(contract: dict[str, Any]) -> dict[str, str]:
    found: dict[str, str] = {}
    for characteristic in contract.get("characteristics", []):
        identifier = characteristic.get("id")
        if identifier in {rule["contract_id"] for rule in STREAM_RULES.values()}:
            found[str(identifier)] = str(characteristic.get("uuid", "")).upper()
    return found


def check_metadata(capture: dict[str, Any], report: dict[str, Any]) -> None:
    if capture.get("schema_version") != SCHEMA_VERSION:
        add_issue(report, "errors", f"schema_version must be {SCHEMA_VERSION}")
    if capture.get("capture_type") != "gen2_ble_stream_capture":
        add_issue(report, "errors", "capture_type must be gen2_ble_stream_capture")
    nonempty(capture, "capture_id", "capture", report)

    candidate = capture.get("candidate")
    if not isinstance(candidate, dict):
        add_issue(report, "errors", "capture.candidate must be an object")
    else:
        revision = nonempty(candidate, "source_revision", "candidate", report)
        if revision and not SHA1_RE.fullmatch(revision):
            add_issue(report, "errors", "candidate.source_revision must be a 40-character Git commit SHA")
        image = nonempty(candidate, "firmware_image_sha256", "candidate", report)
        if image and not SHA256_RE.fullmatch(image):
            add_issue(report, "errors", "candidate.firmware_image_sha256 must be a SHA-256 digest")
        sdk = nonempty(candidate, "sdk_nrf_revision", "candidate", report)
        if sdk and not SHA1_RE.fullmatch(sdk):
            add_issue(report, "errors", "candidate.sdk_nrf_revision must be a 40-character Git commit SHA")
        nonempty(candidate, "zephyr_revision", "candidate", report)
        nonempty(candidate, "toolchain_id", "candidate", report)
        if candidate.get("source_tree") != "clean":
            add_issue(report, "errors", "candidate.source_tree must be clean; a dirty tree cannot be acceptance evidence")

    hardware = capture.get("hardware")
    if not isinstance(hardware, dict):
        add_issue(report, "errors", "capture.hardware must be an object")
    else:
        nonempty(hardware, "board_id", "hardware", report)
        nonempty(hardware, "hardware_revision", "hardware", report)

    collection = capture.get("collection")
    if not isinstance(collection, dict):
        add_issue(report, "errors", "capture.collection must be an object")
        return
    started = nonempty(collection, "started_at_utc", "collection", report)
    ended = nonempty(collection, "ended_at_utc", "collection", report)
    start_time = parse_utc(started, "collection.started_at_utc", report) if started else None
    end_time = parse_utc(ended, "collection.ended_at_utc", report) if ended else None
    if start_time and end_time and end_time <= start_time:
        add_issue(report, "errors", "collection.ended_at_utc must be later than collection.started_at_utc")
    nonempty(collection, "operator", "collection", report)
    nonempty(collection, "procedure_id", "collection", report)
    raw_hash = nonempty(collection, "raw_capture_sha256", "collection", report)
    if raw_hash and not SHA256_RE.fullmatch(raw_hash):
        add_issue(report, "errors", "collection.raw_capture_sha256 must be a SHA-256 digest")


def decode_payload(value: Any, path: str, report: dict[str, Any]) -> bytes | None:
    if not isinstance(value, str) or not value or len(value) % 2 or not HEX_RE.fullmatch(value):
        add_issue(report, "errors", f"{path}.payload_hex must be an even-length hexadecimal string")
        return None
    return bytes.fromhex(value)


def validate_stream(
    stream_id: str,
    stream: Any,
    expected_uuid: str | None,
    minimum_duration_s: float,
    minimum_delivery_fraction: float,
    report: dict[str, Any],
) -> None:
    rule = STREAM_RULES[stream_id]
    metrics: dict[str, Any] = {
        "bytes_per_sample": rule["bytes_per_sample"],
        "frame_ceiling": rule["frame_ceiling"],
    }
    report["streams"][stream_id] = metrics

    if not isinstance(stream, dict):
        add_issue(report, "errors", f"streams.{stream_id} must be an object")
        return

    actual_uuid = stream.get("characteristic_uuid")
    if not isinstance(actual_uuid, str) or actual_uuid.upper() != expected_uuid:
        add_issue(report, "errors", f"streams.{stream_id}.characteristic_uuid does not match the TGM contract")

    declared_rate = stream.get("declared_rate_hz")
    if not isinstance(declared_rate, (int, float)) or isinstance(declared_rate, bool) or declared_rate <= 0:
        add_issue(report, "errors", f"streams.{stream_id}.declared_rate_hz must be a positive number")
        return
    declared_rate = float(declared_rate)
    metrics["declared_rate_hz"] = declared_rate

    expected_samples = stream.get("expected_samples_per_packet")
    if not isinstance(expected_samples, int) or not 1 <= expected_samples <= rule["frame_ceiling"]:
        add_issue(
            report,
            "errors",
            f"streams.{stream_id}.expected_samples_per_packet must be 1..{rule['frame_ceiling']}",
        )
        return
    metrics["expected_samples_per_packet"] = expected_samples

    records = stream.get("records")
    if not isinstance(records, list) or len(records) < 2:
        add_issue(report, "errors", f"streams.{stream_id}.records must contain at least two notifications")
        return

    decoded: list[tuple[float, int, int, bytes]] = []
    identical_repeats = 0
    for index, record in enumerate(records):
        path = f"streams.{stream_id}.records[{index}]"
        if not isinstance(record, dict):
            add_issue(report, "errors", f"{path} must be an object")
            continue
        received_at_ms = record.get("received_at_ms")
        if not isinstance(received_at_ms, (int, float)) or isinstance(received_at_ms, bool) or received_at_ms < 0:
            add_issue(report, "errors", f"{path}.received_at_ms must be a non-negative number")
            continue
        payload = decode_payload(record.get("payload_hex"), path, report)
        if payload is None:
            continue
        payload_bytes = len(payload) - 4
        if len(payload) <= 4 or payload_bytes % rule["bytes_per_sample"]:
            add_issue(
                report,
                "errors",
                f"{path} payload is not 4 + N*{rule['bytes_per_sample']} bytes ({len(payload)} bytes)",
            )
            continue
        sample_count = payload_bytes // rule["bytes_per_sample"]
        if sample_count > expected_samples:
            add_issue(
                report,
                "errors",
                f"{path} has {sample_count} samples, above declared frame ceiling {expected_samples}",
            )
            continue
        frame_counter = int.from_bytes(payload[:4], "little")
        # nRF Connect sometimes writes the same notify twice. A second copy of the
        # same bytes is one frame, not a counter that went backwards.
        if decoded and decoded[-1][3] == payload:
            identical_repeats += 1
            continue
        decoded.append((float(received_at_ms), frame_counter, sample_count, payload))

    if len(decoded) < 2:
        add_issue(report, "errors", f"streams.{stream_id} has fewer than two valid notification records")
        return

    # nRF Connect often stamps several notifies with one millisecond. Equal times are
    # the host clock. A time that moves backwards is a broken log.
    backwards = 0
    for earlier, later in zip(decoded, decoded[1:]):
        if later[0] < earlier[0]:
            backwards += 1
    if backwards:
        add_issue(
            report,
            "errors",
            f"streams.{stream_id} has {backwards} receipt time(s) that move backwards",
        )
        return

    frame_gaps = 0
    duplicate_or_reverse = 0
    for previous, current in zip(decoded, decoded[1:]):
        increment = (current[1] - previous[1]) % COUNTER_MODULUS
        if increment == 0 or increment >= COUNTER_MODULUS // 2:
            duplicate_or_reverse += 1
        elif increment > 1:
            frame_gaps += increment - 1
    if duplicate_or_reverse:
        add_issue(report, "errors", f"streams.{stream_id} has {duplicate_or_reverse} duplicate or reverse frame counter transition(s)")
    if frame_gaps:
        add_issue(report, "errors", f"streams.{stream_id} has {frame_gaps} missing frame counter value(s)")

    times = [item[0] for item in decoded]
    counts = [item[2] for item in decoded]
    elapsed_s = (times[-1] - times[0]) / 1000.0
    if elapsed_s < minimum_duration_s:
        add_issue(
            report,
            "errors",
            f"streams.{stream_id} duration {elapsed_s:.3f}s is below required {minimum_duration_s:.3f}s",
        )

    samples_after_first = sum(counts[1:])
    delivery_rate = samples_after_first / elapsed_s if elapsed_s > 0 else 0.0
    threshold = declared_rate * minimum_delivery_fraction
    if delivery_rate < threshold:
        add_issue(
            report,
            "errors",
            f"streams.{stream_id} delivered {delivery_rate:.3f} Hz, below {threshold:.3f} Hz minimum",
        )

    intervals = [later - earlier for earlier, later in zip(times, times[1:])]
    expected_intervals = [count / declared_rate * 1000.0 for count in counts[:-1]]
    jitter = [actual - expected for actual, expected in zip(intervals, expected_intervals)]
    metrics.update(
        {
            "notification_count": len(decoded),
            "identical_consecutive_payloads": identical_repeats,
            "sample_count": sum(counts),
            "first_frame_counter": decoded[0][1],
            "last_frame_counter": decoded[-1][1],
            "frame_counter_gaps": frame_gaps,
            "duration_s": round(elapsed_s, 6),
            "delivery_rate_hz": round(delivery_rate, 6),
            "minimum_delivery_rate_hz": round(threshold, 6),
            "packet_interval_ms": {
                "minimum": round(min(intervals), 6),
                "maximum": round(max(intervals), 6),
                "mean": round(statistics.fmean(intervals), 6),
                "stdev": round(statistics.pstdev(intervals), 6),
            },
            "frame_cadence_error_ms": {
                "minimum": round(min(jitter), 6),
                "maximum": round(max(jitter), 6),
                "mean": round(statistics.fmean(jitter), 6),
                "stdev": round(statistics.pstdev(jitter), 6),
            },
        }
    )


def validate_capture(
    capture: dict[str, Any],
    contract: dict[str, Any],
    minimum_duration_s: float = 60.0,
    minimum_delivery_fraction: float = 0.90,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "errors": [],
        "warnings": [],
        "streams": {},
        "gate_decision": "not_a_gate_decision",
    }
    check_metadata(capture, report)
    expected_uuids = contract_streams(contract)
    for stream_id, rule in STREAM_RULES.items():
        if rule["contract_id"] not in expected_uuids:
            add_issue(report, "errors", f"TGM contract does not define stream {rule['contract_id']}")
            continue
        streams = capture.get("streams")
        stream = streams.get(stream_id) if isinstance(streams, dict) else None
        validate_stream(
            stream_id,
            stream,
            expected_uuids[rule["contract_id"]],
            minimum_duration_s,
            minimum_delivery_fraction,
            report,
        )
    report["capture_integrity_passed"] = not report["errors"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="JSON raw-capture manifest to validate")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--minimum-duration-s", type=float, default=60.0)
    parser.add_argument("--minimum-delivery-fraction", type=float, default=0.90)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true", help="Exit 2 if the capture has validation errors")
    args = parser.parse_args()

    if args.minimum_duration_s <= 0 or not 0 < args.minimum_delivery_fraction <= 1:
        parser.error("duration must be positive and delivery fraction must be in (0, 1]")
    try:
        capture = json.loads(args.capture.read_text(encoding="utf-8"))
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"gen2-stream-capture: {error}", file=sys.stderr)
        return 1

    report = validate_capture(capture, contract, args.minimum_duration_s, args.minimum_delivery_fraction)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Gen2 BLE stream-capture verification")
        for stream, metrics in report["streams"].items():
            if "sample_count" in metrics:
                print(
                    f"INFO  {stream}: {metrics['sample_count']} samples, "
                    f"{metrics['delivery_rate_hz']:.3f} Hz delivery, "
                    f"{metrics['frame_counter_gaps']} frame gap(s)"
                )
        for message in report["warnings"]:
            print(f"WARN  {message}")
        for message in report["errors"]:
            print(f"ERROR {message}")
        print("INFO  Result is capture-integrity only; it does not close G2-P3.")
    return 2 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
