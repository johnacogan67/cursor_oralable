# Firmware 1.0.82 — flash guide (pcb00003 / REV10)

**App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

**Build date:** 27 Aug 2026 · **Target:** Stage A Phase 0 Vitals / Research Kit · patient app only  
**GATT string:** `1.0.82` (boot banner `v1.0.82-76bdb4ba748b`)

Includes 1.0.70 STAT blink, 1.0.72 green-only status LEDs, 1.0.80 sense-on-BLE, 1.0.81 5% sensor floor, **1.0.82 IR-pulse `worn`**.

---

## Files to flash

| Method | File |
|--------|------|
| **SWD / J-Link** | [`oralable_1.0.82_pcb00003_merged.hex`](./firmware/oralable_1.0.82_pcb00003_merged.hex) |
| **BLE OTA** | [`oralable_1.0.82_pcb00003_app_update.bin`](./firmware/oralable_1.0.82_pcb00003_app_update.bin) |
| **nRF Device Manager** | [`oralable_1.0.82_pcb00003_dfu_application.zip`](./firmware/oralable_1.0.82_pcb00003_dfu_application.zip) |

Also: `oralable_nrf/artifacts/oralable_1.0.82_pcb00003_*`

A device already on **1.0.70** (MCUboot) does **not** need J-Link. Prefer OTA.

---

## BLE OTA (Nordic Device Manager)

Do **not** use nRF Connect legacy DFU. The Oralable app has no DFU UI.

1. AirDrop [`oralable_1.0.82_pcb00003_dfu_application.zip`](./firmware/oralable_1.0.82_pcb00003_dfu_application.zip) to the iPhone.
2. Open **Nordic Device Manager** → scan → **Oralable** → Firmware Upgrade.
3. Pick the zip. Keep the phone close until reboot.
4. nRF Connect: read **`3A0FF006`** → **`1.0.82`**.

Mac CLI:

```bash
cd ~/work/oralable_nrf
./scripts/update_firmware.sh --ota --bin artifacts/oralable_1.0.82_pcb00003_app_update.bin
```

Full flow: [OTA_DEVICE_MANAGER.md](../../../oralable_nrf/docs/OTA_DEVICE_MANAGER.md).

---

## SWD flash (blank chip or OTA brick)

```bash
cd ~/work/oralable_nrf
./scripts/flash_and_rtt.sh --no-build --hex artifacts/oralable_1.0.82_pcb00003_merged.hex
```

---

## What’s new vs 1.0.70

| Area | Change |
|------|--------|
| **Sensors** | PPG/ACC follow BLE + notify (CCC). No `09 03` to start the chips. Disconnect → sensors off. |
| **`worn`** | Automatic = **IR pulse** (~2.5 s on, 20 s hold). Not die temperature. Mode 3 still forces worn. On-dock still `worn=0`. |
| **Below ~5% / 3.61 V** | PPG/ACC off even if BLE is up. MCU, advertising, and case charge stay on. Chemical protect remains **&lt; 2.8 V**. |
| **Status LED** | Green-only. Pad: flash green while charging, solid green at STAT taper. Off pad, no BLE: dark. Red/IR is PPG only. |
| **Compute** | MAM streams raw red/green/IR + ACC. HR/SpO₂/clench stay on phone/Mac. |
| **STAT** | Same as 1.0.70: blink = charging; taper = `charge_active=0` while `on_dock=1`. |

---

## nRF Connect check (after OTA)

1. Read `3A0FF006` → **1.0.82**.
2. CCC: battery → status → FwLog → PPG → ACC. Do **not** write `09 03`.
3. Bench, no skin: stream runs, `worn=0`.
4. Temple: `fw: ir_pulse worn=1` after ~2.5 s.
5. Disconnect: sensors off. Pad: green charge LEDs. Below 5%: sensors off, BLE stays up.

If temple never latches, IR AC/DC is below the on-threshold — say so; do not put die-temp worn back.

---

## iOS app (pair with 1.0.82)

| Area | Expect |
|------|--------|
| **FirmwareGate** | Hard min **1.0.63**; recommend **1.0.82** |
| **Automatic** | Preferred (STAT blink). Soft hint if device is still on 1.0.70 |
| **Device LED** | Flash green charging; solid green taper. LED off while linked |
| **TestFlight** | App **4.3.3+** |

---

## Rollback

Previous ship: [FIRMWARE_1.0.70_FLASH.md](./FIRMWARE_1.0.70_FLASH.md). Older: [FIRMWARE_1.0.66_FLASH.md](./FIRMWARE_1.0.66_FLASH.md).

---

## Related

- [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md)
- [VERSION_ALIGNMENT.md](./VERSION_ALIGNMENT.md)
- [VITALS_PILOT_TEST_PLAN.md](./VITALS_PILOT_TEST_PLAN.md)
