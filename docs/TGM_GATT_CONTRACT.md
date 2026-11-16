# TGM GATT contract

The TGM GATT service is the device-to-host compatibility boundary for Oralable/MAM.
The firmware declaration owns it:

- UUID definitions: `../oralable_nrf/app/src/tgm_service.h`
- GATT properties and permissions: `../oralable_nrf/app/src/tgm_service.c`

The versioned machine-readable representation is
[`contracts/tgm_gatt_v1.json`](../contracts/tgm_gatt_v1.json). It records the
service and all thirteen current characteristics (`3A0FF001` through
`3A0FF00D`), their intended properties, and the payload semantics that host
software may rely on.

## Consumer responsibilities

| Consumer | Responsibility |
|---|---|
| `oralable_nrf` | Own UUID allocation, GATT properties, and payload emission. |
| `cursor_oralable` | Collect/parse only the documented streaming and telemetry characteristics. |
| `OralableCore` | Expose canonical, platform-neutral names and packet-size/domain helpers. |
| `oralable_swift` | Use the shared Core constants for discovery, routing, reads, writes, and CCC setup. |

`3A0FF003` is **temperature**, despite historical client names containing
`command`. The old symbol is retained only as a compatibility alias. `3A0FF007`
and `3A0FF008` are PPG register control paths, not battery telemetry. The
supported battery-usage model is `3A0FF00D` on firmware 1.0.86+; the current
firmware does not declare `3A0FFEF2`.

## Verify before a compatibility change

From `cursor_oralable`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_tgm_contract.py --strict
```

The checker is static and offline. It checks the firmware header against the
versioned UUID set, then checks the Python logger/parser, Core constants, and
iOS `OralableDevice` for the canonical values.

## Change rules

1. Change firmware and `tgm_gatt_v1.json` in the same review.
2. Update Core, iOS, Python, fixtures, and contract verification in the same
   compatibility change.
3. Preserve old behavior with a documented compatibility alias only when it does
   not reinterpret a UUID or payload.
4. For an incompatible payload, use a new characteristic or an explicit
   versioned payload format; do not silently reuse a production UUID.
5. Capture an nRF Connect/device fixture and exercise the iOS path before marking
   a firmware/app pair compatible.

## Nominal 50 Hz sensor & config-state invariants

- **MAXM86161 PPG (`0x12` PPG Configuration 2):** Programmed and verified via readback as `0x08` (`PPG_SR = 0x01` in bits `7:3` for nominal 50 sps, `SMP_AVE = 0x00` in bits `2:0` for 1 pulse/sample without FIFO decimation). Bounded FIFO bursts read at most 60 words (180 bytes = 20 complete red/IR/green triplets), emitting 244-byte notifications (`4 + 12*20`, requiring `ATT_MTU >= 247`).
- **LIS2DTW12 Accelerometer (`0x20` CTRL1):** Programmed and verified via readback as `0x43` (`ODR = 0x40` for 50 Hz in bits `7:4`, `MODE = 0x00` low-power in bits `3:2`, `LP_MODE = 0x03` LP mode 4 in bits `1:0`). Note that `0x44` selects high-performance mode rather than LP4; `0x43` is the canonical low-power 50 Hz value. Bounded FIFO bursts read at most 25 samples (150 bytes), emitting 154-byte notifications (`4 + 6*25`, requiring `ATT_MTU >= 157`).
- **`3A0FF00C` v2 `battery_interval_s` (`u8` byte 5):** The v2 16-byte config-state layout stores `battery_interval_s` in a single `uint8_t` (defaulting to `10` or the last `00B` opcode `0x04` override `1..255`), whereas default ADC battery cadence `CONFIG_BATTERY_MEASUREMENT_INTERVAL` is `300` s. Preserve the 16-byte v2 wire layout; widen only in a coordinated v3 config-state revision.
