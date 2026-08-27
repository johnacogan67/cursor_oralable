# Firmware 1.0.66 — flash guide (pcb00003 / REV10)

**Build date:** July 2026 · **Role:** **Rollback only** (current ship = **1.0.82**) · Ed/Pedro = **patient app only**

Includes **1.0.65** energy, LED, and charge fixes **plus** BLE RSSI and reconnect improvements.

> **Superseded:** flash **[1.0.82](./FIRMWARE_1.0.82_FLASH.md)**. Prior ship [1.0.70](./FIRMWARE_1.0.70_FLASH.md). Keep 1.0.66 binaries for recovery only.

---

## Files to flash

| Method | File |
|--------|------|
| **SWD / J-Link** | [`oralable_1.0.66_pcb00003_merged.hex`](./firmware/oralable_1.0.66_pcb00003_merged.hex) |
| **BLE OTA** | [`oralable_1.0.66_pcb00003_app_update.bin`](./firmware/oralable_1.0.66_pcb00003_app_update.bin) |
| **nRF Device Manager** | [`oralable_1.0.66_pcb00003_dfu_application.zip`](./firmware/oralable_1.0.66_pcb00003_dfu_application.zip) |

Repo: `oralable_nrf/artifacts/oralable_1.0.66_pcb00003_*`

---

## SWD flash

```bash
cd oralable_nrf
nrfjprog --program artifacts/oralable_1.0.66_pcb00003_merged.hex --sectorerase --verify --reset
```

Verify GATT **`3A0FF006`** → **`1.0.66-nrfconnect`**

---

## What’s new in 1.0.66

| Area | Change |
|------|--------|
| **TX power** | +4 dBm (`CONFIG_BT_CTLR_TX_PWR_PLUS_4`) — stronger scan/connect RSSI |
| **Advertising** | ~100 ms interval for **90 s** after disconnect, then 500 ms (energy) |
| **Adv watchdog** | Force restart every **5 s** if stack stuck (was 10 s) |
| **1.0.65** | Dim LEDs, charge detect fix, app LED mirror — retained |

## iOS app (pair with 1.0.66)

| Area | Change |
|------|--------|
| **Reconnect attempts** | 8 default; **15** on charger off-body; **999** during recording |
| **connect fail** | Immediate retry scheduling (no 15 s hang) |
| **Stale link** | Auto-cancel zombie GATT → triggers reconnect |
| **Background** | Vitals phase keeps reconnect for remembered Oralable |
| **Foreground** | Auto-reconnect if not ready |
| **RSSI poll** | Every 3 s |

---

## Related

- [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md)
- [FIRMWARE_1.0.65_FLASH.md](./FIRMWARE_1.0.65_FLASH.md) (prior release notes)
- [GEN1_GEN2_TRACKING.md](../GEN1_GEN2_TRACKING.md) · current ship: [FIRMWARE_1.0.82_FLASH.md](./FIRMWARE_1.0.82_FLASH.md)
