# Firmware 1.0.84 — flash guide (pcb00003 / REV10)

**App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

**Build date:** 29–30 Aug 2026 · **Role:** **Current Gen1 target** · Stage A Phase 0 / Research Kit · patient app only  
**GATT string:** `1.0.84` (boot banner `v1.0.84-5f144f9d93ac`)

Includes 1.0.82 IR-pulse `worn`, sense-on-BLE, 5% floor, green-only STAT LEDs, plus **pad/zombie recover** and **desk/bench abandon**.

---

## Files to flash

| Method | File |
|--------|------|
| **SWD / J-Link** | [`oralable_1.0.84_pcb00003_merged.hex`](./firmware/oralable_1.0.84_pcb00003_merged.hex) |
| **BLE OTA** | [`oralable_1.0.84_pcb00003_app_update.bin`](./firmware/oralable_1.0.84_pcb00003_app_update.bin) |
| **nRF Device Manager** | [`oralable_1.0.84_pcb00003_dfu_application.zip`](./firmware/oralable_1.0.84_pcb00003_dfu_application.zip) |

Also: `oralable_nrf/artifacts/oralable_1.0.84_pcb00003_*` · workspace build `oralable_nrf/build_pcb00003/merged.hex`

A device already on **1.0.70+** (MCUboot) does **not** need J-Link. Prefer OTA from 1.0.82 or earlier.

---

## BLE OTA (Nordic Device Manager)

Do **not** use nRF Connect legacy DFU. The Oralable app has no DFU UI.

1. AirDrop [`oralable_1.0.84_pcb00003_dfu_application.zip`](./firmware/oralable_1.0.84_pcb00003_dfu_application.zip) to the iPhone.
2. Open **Nordic Device Manager** → scan → **Oralable** → Firmware Upgrade.
3. Pick the zip. Keep the phone close until reboot.
4. nRF Connect: read **`3A0FF006`** → **`1.0.84`**.

Mac CLI:

```bash
cd ~/work/oralable_nrf
./scripts/update_firmware.sh --ota --bin artifacts/oralable_1.0.84_pcb00003_app_update.bin
```

Full flow: [OTA_DEVICE_MANAGER.md](../../../oralable_nrf/docs/OTA_DEVICE_MANAGER.md).

---

## SWD flash (blank chip or OTA brick)

```bash
cd ~/work/oralable_nrf
./scripts/flash_and_rtt.sh --no-build --hex artifacts/oralable_1.0.84_pcb00003_merged.hex
# or live workspace build:
# ./scripts/flash_and_rtt.sh --no-build --hex build_pcb00003/merged.hex
```

---

## What’s new vs 1.0.82

| Area | Change |
|------|--------|
| **Pad / zombie recover** | Idle PPG notify stall ~4 s on case → green + advertise (STAT pad wins leftover CCC) |
| **Desk / bench abandon** | `worn=0` + PPG sensing + ACC flat **10 min** → drop link + re-advertise |
| **FwLog** | Low priority vs PPG/ACC; Mac Protocol A leaves FwLog CCC **off** unless `--fw-log` |
| **GATT** | Unchanged (`3A0FF000` tree) |

Everything from 1.0.82 still applies: IR-pulse worn, sense-on-BLE, 5% sensor floor, green-only charge LEDs.

---

## nRF Connect check (after flash / OTA)

1. Read `3A0FF006` → **1.0.84**.
2. CCC: battery → status → PPG → ACC. Do **not** write `09 03`. FwLog optional.
3. Bench, no skin: stream runs, `worn=0`.
4. Temple: `fw: ir_pulse worn=1` after ~2.5 s.
5. Disconnect: sensors off. Pad: green charge LEDs.
6. Pad recover: after a zombie link, seat on Oralable case with no central → green + advertise within ~4 s.
7. Desk abandon (optional): leave off-case, not worn, ACC still, ~10 min → re-advertise.

---

## iOS app (pair with 1.0.84)

| Area | Expect |
|------|--------|
| **FirmwareGate** | Hard min **1.0.63**; recommend **1.0.84** |
| **Automatic** | Preferred (STAT blink). Soft hint if device is below 1.0.84 |
| **Device LED** | Flash green charging; solid green taper / hold. LED off while linked |
| **TestFlight** | App **4.3.3** build **5+** |

---

## Rollback

Prior ship: [FIRMWARE_1.0.82_FLASH.md](./FIRMWARE_1.0.82_FLASH.md). Older: [FIRMWARE_1.0.70_FLASH.md](./FIRMWARE_1.0.70_FLASH.md) · [FIRMWARE_1.0.66_FLASH.md](./FIRMWARE_1.0.66_FLASH.md).

---

## Related

- [VERSION_ALIGNMENT.md](./VERSION_ALIGNMENT.md)
- [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md)
- [PILOT_DRY_RUN_CHECKLIST.md](./PILOT_DRY_RUN_CHECKLIST.md)
- [VITALS_PILOT_TEST_PLAN.md](./VITALS_PILOT_TEST_PLAN.md)
