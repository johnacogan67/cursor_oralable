# Firmware 1.0.65 — flash guide (pcb00003 / REV10)

**Build date:** July 2026 · **Target:** Phase 0 Vitals pilot (Ed and Pedro)

---

## Files to flash

| Method | File | Use when |
|--------|------|----------|
| **SWD / J-Link (first flash or recovery)** | [`oralable_1.0.65_pcb00003_merged.hex`](./firmware/oralable_1.0.65_pcb00003_merged.hex) | Blank chip, bricked OTA slot, or full reflash |
| **BLE OTA (mcumgr)** | [`oralable_1.0.65_pcb00003_app_update.bin`](./firmware/oralable_1.0.65_pcb00003_app_update.bin) | Device already runs MCUboot + signed app (≥1.0.36) |
| **nRF Connect Device Manager** | [`oralable_1.0.65_pcb00003_dfu_application.zip`](./firmware/oralable_1.0.65_pcb00003_dfu_application.zip) | Phone-based DFU |

**Canonical repo paths** (same binaries):

```
oralable_nrf/artifacts/oralable_1.0.65_pcb00003_merged.hex
oralable_nrf/artifacts/oralable_1.0.65_pcb00003_app_update.bin
oralable_nrf/artifacts/oralable_1.0.65_pcb00003_dfu_application.zip
```

**SHA-256 (merged.hex):** `1cb405314ba9e3479be2947a576ecf91c824db2323bae90bb9cbaab355af282b`

---

## SWD flash (recommended for bench)

```bash
cd oralable_nrf
nrfjprog --program artifacts/oralable_1.0.65_pcb00003_merged.hex --sectorerase --verify --reset
```

Or use the helper (builds if needed):

```bash
./scripts/flash_and_rtt.sh --no-build --hex artifacts/oralable_1.0.65_pcb00003_merged.hex
```

---

## Verify after flash

1. nRF Connect → connect → read **`3A0FF006`** → expect **`1.0.65-nrfconnect`**
2. Off pad: **green** status flash (dimmer than 1.0.64; asymmetric timing)
3. **Oralable magnetic charging case** + app **On wireless charger**: **red** flash; **Charge** chip should track within ~2 min on pilot units
4. iOS app **4.3.1+**: Dashboard **Device LED** row mirrors clip colour/pattern

---

## What’s new in 1.0.65 (vs 1.0.64)

| Area | Change |
|------|--------|
| **Status LED energy** | PA 56→28; asymmetric flash (~350 ms on, 1.2–2 s off) |
| **Charge detect** | Fixed `charge_active` on manual “On wireless charger”; 60 s battery samples on dock |
| **Robustness** | 1.0.64 LED re-arm (30 s), advertising recovery, disconnect LED restore — **unchanged** |
| **iOS** | Live **Device LED** mirror on vitals card; keepalive 8 s on dock / 5 s bench |

**Minimum app:** 4.3.1+ with Vitals phase and updated OralableCore.

---

## Related docs

- [ED_PEDRO_QUICK_START.md](../clinical/ED_PEDRO_QUICK_START.md)
- [VITALS_PILOT_TEST_PLAN.md](../clinical/VITALS_PILOT_TEST_PLAN.md)
- [VITALS_PHASE_GEN1_GEN2.md](../../VITALS_PHASE_GEN1_GEN2.md)
