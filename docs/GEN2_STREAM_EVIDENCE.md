# Gen2 PPG and accelerometer stream evidence

## Status and scope

This guide defines an **offline capture-integrity check** for Gen2 G2-P3 work. It does not authorize a device operation, change `gen2.program.json`, close a gate, or support a release, compatibility, clinical, or performance claim.

Use it only after the physical G2-P0 through G2-P2 prerequisites have been planned and only with an identified Gen2 patient-candidate board. Keep raw BLE captures and board/operator identifiers in the approved evidence store, not in this repository.

## Current Work source recheck — 2026-11-01

The present local `oralable_nrf` Work tree contains uncommitted source changes that plausibly address the earlier no-stream symptom:

- the accelerometer changed from a fixed interrupt pin to its DeviceTree `int-gpios` value and has a periodic FIFO-read fallback;
- PPG already uses DeviceTree interrupt metadata and has a periodic FIFO-read fallback;
- startup clears the nRF54 NFC pad configuration before using P1.03 as I²C SCL, uses DeviceTree power pins, and records sensor identity-read diagnostics;
- TGM reports PPG/accelerometer notification counters and last notification errors.

Those are source changes, not hardware evidence. At the time of this recheck, no new real capture, RTT result, part-ID read, or remote Git commit was available, while the preserved 30 October note still reports sensor I²C NACKs and no PPG/ACC samples. The current source is therefore **not an immutable candidate identity**.

The device rate used for a capture is:

- the MAXM86161 driver configures PPG at 50 Hz, with a 20-sample frame (nominal 400 ms); Swift assumes PPG at 50 Hz;
- the LIS2DTW12 driver configures the accelerometer at 50 Hz (`CTRL1 = 0x43`). `OralableCore` and `src/parser/log_parser.py` stamp those samples at 20 ms. The iOS `AlgorithmSpec` declares the same 50 Hz rate.

The device rate is 50 Hz for both sensors. A host log may stamp several notifies with the same millisecond, and nRF Connect sometimes writes the same notify twice. The capture check allows equal receipt times, counts one copy of a repeated identical payload, and rejects a time that moves backwards. A repeated counter with different bytes stays an error. Frame-counter gaps stay errors.

## Capture-manifest format

The verifier reads one JSON document. Store its raw packet records in the controlled evidence system; its `collection.raw_capture_sha256` records the digest of that immutable raw capture. The manifest requires:

- `schema_version: 1` and `capture_type: "gen2_ble_stream_capture"`;
- a capture identifier;
- a **clean** 40-hex Git commit, firmware image SHA-256, sdk-nrf commit, Zephyr revision, and toolchain identifier;
- board identifier and hardware revision;
- ISO-8601 UTC start/end times, operator, procedure identifier, and raw-capture SHA-256;
- both `ppg` and `accelerometer` stream objects, each with the canonical TGM UUID, a reviewed `declared_rate_hz`, a maximum samples-per-packet value, and ordered notification records;
- each notification record's local receipt time in milliseconds and raw `payload_hex`, including the four-byte little-endian frame counter.

The current shape checks are defined by `contracts/tgm_gatt_v1.json`:

- PPG (`3A0FF001`): `4 + 12*N` bytes, maximum 20 samples per current firmware frame;
- accelerometer (`3A0FF002`): `4 + 6*N` bytes, maximum 25 samples per current firmware frame.

The capture verifier checks UUIDs, metadata shape, packet structure, frame-counter continuity, strictly increasing receipt times, minimum duration, and delivered sample rate. Receipt times measure host-observed delivery cadence only; they do **not** prove on-device sample timing or radio latency.

## Run the check

From the `cursor_oralable` repository, after placing a real capture manifest in the approved evidence location:

```sh
.venv/bin/python scripts/verify_gen2_stream_capture.py \
  /approved-evidence/gen2-p3-capture.json \
  --minimum-duration-s 60 \
  --minimum-delivery-fraction 0.90 \
  --strict --format json
```

The command returns `2` for malformed, untraceable, too-short, low-delivery-rate, or frame-gap evidence. A zero exit code means only that the submitted data meets the declared capture-integrity rules. It is not a gate pass. The gate remains pending until independent review evaluates the raw evidence against the full G2-P3 procedure: sensor identity and readback, candidate hashes, stream-health counters, and the resolved rate decision.

## Required hardware evidence remains unchanged

For G2-P3, preserve sustained PPG and accelerometer notifications from the same identified physical Gen2 board and pinned patient-candidate image. Include raw payloads, decoded counters, packet/sample counts, delivery intervals, parser outcomes, device and firmware counters, and error logs. Link the reviewed evidence record—not raw health or device data—when a future authorized gate update is proposed.
