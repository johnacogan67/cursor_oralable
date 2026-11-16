# Firmware 1.0.85 — flash guide (pcb00003 / REV10)

**App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

**Build date:** 14 Sep 2026 · **Role:** **Legacy Gen 1.** Do not flash this on Gen 2 (REV11). Gen 2 bench: [GEN2_WOUT_KAGA_STATUS_2026-10-30.md](../hardware/GEN2_WOUT_KAGA_STATUS_2026-10-30.md).  
**GATT string:** `1.0.85` (boot banner `v1.0.85-3925d907a03b`)

Includes 1.0.84 pad/zombie recover and desk/bench abandon, plus **hold LED**: flash below ~4.20 V / 80%; rare green pulse in hold (not continuous solid green). STAT taper alone is not full.

---

## Files to flash

| Method | File |
|--------|------|
| **SWD / J-Link** | [`oralable_1.0.85_pcb00003_merged.hex`](./oralable_1.0.85_pcb00003_merged.hex) |
| **BLE OTA** | [`oralable_1.0.85_pcb00003_app_update.bin`](./oralable_1.0.85_pcb00003_app_update.bin) |
| **nRF Device Manager** | [`oralable_1.0.85_pcb00003_dfu_application.zip`](./oralable_1.0.85_pcb00003_dfu_application.zip) |

Also: `oralable_nrf/artifacts/oralable_1.0.85_pcb00003_*` · workspace build `oralable_nrf/build_pcb00003/dfu_application.zip`

A device already on **1.0.70+** (MCUboot) does **not** need J-Link. Prefer OTA from 1.0.84 or 1.0.82.

---

## BLE OTA (Nordic Device Manager)

Do **not** use nRF Connect legacy DFU. The Oralable app has no DFU UI.

1. AirDrop [`oralable_1.0.85_pcb00003_dfu_application.zip`](./oralable_1.0.85_pcb00003_dfu_application.zip) to the iPhone.
2. Open **Nordic Device Manager** → scan → **Oralable** → Firmware Upgrade.
3. Pick the zip. Keep the phone close until reboot.
4. nRF Connect: read **`3A0FF006`** → **`1.0.85`**.

Mac CLI:

```bash
cd ~/work/oralable_nrf
./scripts/update_firmware.sh --ota --bin artifacts/oralable_1.0.85_pcb00003_app_update.bin
```

Full flow: [OTA_DEVICE_MANAGER.md](../../../../oralable_nrf/docs/OTA_DEVICE_MANAGER.md).

---

## SWD note (5 Oct 2026)

The merged hex in this folder does not boot. The signed image is **0x39077** bytes. Swap-move keeps the last 4 KB of the slot, so the image must end at or before **0x39000**. MCUboot stops. The application never advertises, and the green LED stays off.

One MAM did boot that day from `oralable_nrf/build_pcb00003_fit/merged.hex` (signed length **0x31e03**, `app/slot_fit.conf`). That tree’s `app/VERSION` is **1.0.86**. It is not this kit artifact. Do not chip-erase this part. Program with J-Link Commander on the nRF5340-DK debug out. `scripts/flash_and_rtt.sh` uses nrfjprog, which did not open probe S/N **1050090445**.

## SWD flash (blank chip or OTA brick)

```bash
cd ~/work/oralable_nrf
./scripts/flash_and_rtt.sh --no-build --hex artifacts/oralable_1.0.85_pcb00003_merged.hex
# or live workspace build:
# ./scripts/flash_and_rtt.sh --no-build --hex build_pcb00003/merged.hex
```

---

## What’s new vs 1.0.84

| Area | Change |
|------|--------|
| **Hold LED** | Flash green while below ~4.20 V / 80% (charging or early STAT taper). Rare green pulse (~200 ms every 10 s) in hold. No continuous solid green. |
| **GATT** | Unchanged (`3A0FF000` tree). String `3A0FF006` = `1.0.85`. |

Everything from 1.0.84 still applies: pad/zombie recover, desk abandon, IR-pulse worn, sense-on-BLE, 5% sensor floor.

---

## nRF Connect check (after flash / OTA)

1. Read `3A0FF006` → **1.0.85**.
2. CCC: battery → status → PPG → ACC. Do **not** write `09 03`. FwLog optional.
3. Disconnect. Seat on Oralable case, no phone:
   - Below ~4.20 V: **flash green**.
   - In hold: **brief green pulse ~every 10 s**, not a steady light.
   - Watch `3A0FF004` mV — it should not sag the way continuous solid green did.
4. Connect → status LED off. Lift off case → LED off.
5. Pad recover / desk abandon: same as 1.0.84.

---

## iOS app (pair with 1.0.85)

| Area | Expect |
|------|--------|
| **FirmwareGate** | Hard min **1.0.63**; recommend **1.0.85** |
| **Automatic** | Preferred (STAT blink). Soft hint if device is below 1.0.85 |
| **Device LED** | Flash green below hold; rare pulse in hold. LED off while linked |
| **TestFlight** | App **4.3.3** build **5+** |

---

## Rollback

Prior ship: [FIRMWARE_1.0.84_FLASH.md](./FIRMWARE_1.0.84_FLASH.md). Older: [FIRMWARE_1.0.82_FLASH.md](./FIRMWARE_1.0.82_FLASH.md).

---

## Related

- [VERSION_ALIGNMENT.md](../VERSION_ALIGNMENT.md)
- [ED_PEDRO_QUICK_START.md](../clinical/ED_PEDRO_QUICK_START.md)
- [PILOT_DRY_RUN_CHECKLIST.md](../clinical/PILOT_DRY_RUN_CHECKLIST.md)
- [VITALS_PILOT_TEST_PLAN.md](../clinical/VITALS_PILOT_TEST_PLAN.md)
