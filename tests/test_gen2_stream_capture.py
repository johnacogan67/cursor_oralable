"""Regression coverage for Gen2 BLE raw-stream evidence validation."""

from __future__ import annotations

import copy
import struct

from scripts.verify_gen2_stream_capture import validate_capture

PPG_UUID = "3A0FF001-98C4-46B2-94AF-1AEE0FD4C48E"
ACC_UUID = "3A0FF002-98C4-46B2-94AF-1AEE0FD4C48E"


def packet(counter: int, samples: int, bytes_per_sample: int) -> str:
    """Make a deterministic protocol test packet; it is not hardware evidence."""
    return (struct.pack("<I", counter) + bytes(samples * bytes_per_sample)).hex()


def capture_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "capture_type": "gen2_ble_stream_capture",
        "capture_id": "unit-test-capture",
        "candidate": {
            "source_revision": "a" * 40,
            "firmware_image_sha256": "b" * 64,
            "sdk_nrf_revision": "c" * 40,
            "zephyr_revision": "v3.7.99-ncs2-3",
            "toolchain_id": "b8efef2ad5",
            "source_tree": "clean",
        },
        "hardware": {"board_id": "unit-test-board", "hardware_revision": "REV11"},
        "collection": {
            "started_at_utc": "2026-11-01T20:00:00Z",
            "ended_at_utc": "2026-11-01T20:01:01Z",
            "operator": "unit test",
            "procedure_id": "G2-P3-test",
            "raw_capture_sha256": "d" * 64,
        },
        "streams": {
            "ppg": {
                "characteristic_uuid": PPG_UUID,
                "declared_rate_hz": 50,
                "expected_samples_per_packet": 20,
                "records": [
                    {"received_at_ms": 0, "payload_hex": packet(100, 20, 12)},
                    {"received_at_ms": 400, "payload_hex": packet(101, 20, 12)},
                    {"received_at_ms": 800, "payload_hex": packet(102, 20, 12)},
                ],
            },
            "accelerometer": {
                "characteristic_uuid": ACC_UUID,
                "declared_rate_hz": 50,
                "expected_samples_per_packet": 25,
                "records": [
                    {"received_at_ms": 0, "payload_hex": packet(200, 25, 6)},
                    {"received_at_ms": 500, "payload_hex": packet(201, 25, 6)},
                    {"received_at_ms": 1000, "payload_hex": packet(202, 25, 6)},
                ],
            },
        },
    }


def contract() -> dict[str, object]:
    return {
        "characteristics": [
            {"id": "ppg", "uuid": PPG_UUID},
            {"id": "accelerometer", "uuid": ACC_UUID},
        ]
    }


def test_capture_manifest_reports_packet_counts_rates_and_no_gate_decision() -> None:
    report = validate_capture(capture_manifest(), contract(), minimum_duration_s=0.5)

    assert report["errors"] == []
    assert report["capture_integrity_passed"] is True
    assert report["gate_decision"] == "not_a_gate_decision"
    assert report["streams"]["ppg"]["sample_count"] == 60
    assert report["streams"]["ppg"]["delivery_rate_hz"] == 50.0
    assert report["streams"]["accelerometer"]["sample_count"] == 75
    assert report["streams"]["accelerometer"]["delivery_rate_hz"] == 50.0


def test_capture_manifest_rejects_malformed_payload_and_missing_frame() -> None:
    manifest = copy.deepcopy(capture_manifest())
    manifest["streams"]["ppg"]["records"][1]["payload_hex"] = "00010203ff"
    manifest["streams"]["accelerometer"]["records"][1]["payload_hex"] = packet(202, 25, 6)

    report = validate_capture(manifest, contract(), minimum_duration_s=0.5)

    assert report["capture_integrity_passed"] is False
    assert any("PPG" not in error and "4 + N*12" in error for error in report["errors"])
    assert any("accelerometer has 1 missing frame counter" in error for error in report["errors"])


def test_capture_manifest_allows_equal_receipt_times_when_counters_advance() -> None:
    manifest = copy.deepcopy(capture_manifest())
    manifest["streams"]["ppg"]["records"][1]["received_at_ms"] = 0
    manifest["streams"]["ppg"]["records"][2]["received_at_ms"] = 400

    report = validate_capture(manifest, contract(), minimum_duration_s=0.2)

    assert report["capture_integrity_passed"] is True
    assert report["streams"]["ppg"]["frame_counter_gaps"] == 0


def test_capture_manifest_collapses_a_repeated_identical_payload() -> None:
    manifest = copy.deepcopy(capture_manifest())
    repeated = copy.deepcopy(manifest["streams"]["ppg"]["records"][0])
    manifest["streams"]["ppg"]["records"].insert(1, repeated)

    report = validate_capture(manifest, contract(), minimum_duration_s=0.5)

    assert report["capture_integrity_passed"] is True
    assert report["streams"]["ppg"]["sample_count"] == 60
    assert report["streams"]["ppg"]["identical_consecutive_payloads"] == 1


def test_capture_manifest_rejects_a_repeated_counter_with_different_bytes() -> None:
    manifest = copy.deepcopy(capture_manifest())
    raw = bytes.fromhex(manifest["streams"]["ppg"]["records"][0]["payload_hex"])
    changed = raw[:4] + bytes([1]) + raw[5:]
    manifest["streams"]["ppg"]["records"][1]["payload_hex"] = changed.hex()
    manifest["streams"]["ppg"]["records"][2]["payload_hex"] = packet(101, 20, 12)

    report = validate_capture(manifest, contract(), minimum_duration_s=0.5)

    assert report["capture_integrity_passed"] is False
    assert any("duplicate or reverse frame counter" in error for error in report["errors"])


def test_capture_manifest_rejects_receipt_time_that_moves_backwards() -> None:
    manifest = copy.deepcopy(capture_manifest())
    manifest["streams"]["ppg"]["records"][2]["received_at_ms"] = 100

    report = validate_capture(manifest, contract(), minimum_duration_s=0.2)

    assert report["capture_integrity_passed"] is False
    assert any("move backwards" in error for error in report["errors"])


def test_capture_manifest_rejects_dirty_candidate_identity() -> None:
    manifest = capture_manifest()
    manifest["candidate"]["source_tree"] = "dirty"

    report = validate_capture(manifest, contract(), minimum_duration_s=0.5)

    assert report["capture_integrity_passed"] is False
    assert "candidate.source_tree must be clean" in "\n".join(report["errors"])
