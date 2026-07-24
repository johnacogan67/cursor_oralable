# Firmware 1.0.70 — flash guide (pcb00003 / REV10)

**Build date:** July 2026 · **Target:** Stage A Phase 0 Vitals pilot (**Ed/Pedro ship**) · patient app only

**Includes** 1.0.66 BLE RSSI/reconnect + 1.0.68 remapped battery gauge + **LTC4124 STAT activity** (blink = charging / on_dock).

> REV10 CHRSTS hardware is **OK**. Pre-1.0.70 firmware required a *stable* GPIO level, so STAT blink never latched → false “chrsts broken.” **1.0.70** treats blink as charging, steady assert as charge taper, steady inactive as undock.

---

## Files to flash

| Method | File |
|--------|------|
| **SWD / J-Link** | [`oralable_1.0.70_pcb00003_merged.hex`](./firmware/oralable_1.0.70_pcb00003_merged.hex) |
| **BLE OTA** | [`oralable_1.0.70_pcb00003_app_update.bin`](./firmware/oralable_1.0.70_pcb00003_app_update.bin) |
| **nRF Device Manager** | [`oralable_1.0.70_pcb00003_dfu_application.zip`](./firmware/oralable_1.0.70_pcb00003_dfu_application.zip) |

Also: `oralable_nrf/artifacts/oralable_1.0.70_pcb00003_*`

---

## SWD flash

```bash
cd ~/work/oralable_nrf
./scripts/flash_and_rtt.sh --no-build --hex artifacts/oralable_1.0.70_pcb00003_merged.hex
# or:
nrfjprog --program artifacts/oralable_1.0.70_pcb00003_merged.hex --sectorerase --verify --reset
```

Verify GATT **`3A0FF006`** → **`1.0.70`** (suffix may include `-nrfconnect` depending on build string).

---

## What’s new in 1.0.70

| Area | Change |
|------|--------|
| **CHRSTS / STAT** | `CONFIG_CHRSTS_STAT_ACTIVITY` — edge blink → `on_dock=1` + `charge_active=1`; steady assert → on pad, taper (`charge_active=0`); steady inactive → undock |
| **Battery LED** | On pad: **flash red** while charging; **solid red** on STAT taper (not necessarily 4.2 V) |
| **Battery %** | Remapped gauge (from 1.0.68): **0% ≈ 3.61 V**, **100% = 4.35 V** — rough voltage estimate |
| **Manual modes** | Placement 1/2/3 still override Automatic (pilot safety) |
| **Prior** | 1.0.66 TX +4 dBm / fast reconnect; 1.0.65 dim LEDs |

## nRF Connect / RTT gate (before handoff)

1. Seat on **Oralable case** → status `on_dock=1`, `charge_active=1`, flash red (RTT: `chrsts phase` / edges).
2. Leave until STAT steady → `charge_active=0`, still `on_dock=1`, solid red.
3. Lift off → `on_dock=0`, green bench LED.

## iOS app (pair with 1.0.70)

| Area | Expect |
|------|--------|
| **FirmwareGate** | Hard min **1.0.63**; recommend **1.0.70** |
| **Automatic** | Preferred on 1.0.70+ (STAT blink); soft warning if device still on 1.0.66 |
| **Device LED** | Mirror: flash when Charging chip on; solid on Taper while Dock on |
| **TestFlight** | App **4.3.3+** — vitals phase + Automatic + STAT LED mirror |

---

## Related

- [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md)
- [VITALS_PILOT_TEST_PLAN.md](./VITALS_PILOT_TEST_PLAN.md)
- [FIRMWARE_1.0.66_FLASH.md](./FIRMWARE_1.0.66_FLASH.md) (prior ship)
- [HW_ENGINEER_ALTIUM_BRIEF.md](./HW_ENGINEER_ALTIUM_BRIEF.md) (CHRSTS closed — no ECO)
