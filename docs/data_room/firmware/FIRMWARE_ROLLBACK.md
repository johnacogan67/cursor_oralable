# Firmware rollback (Gen 1)

**Gen 1 frozen image:** [FIRMWARE_1.0.85_FLASH.md](./FIRMWARE_1.0.85_FLASH.md) · pack binaries in [`firmware/`](./)  
**Not the Gen 2 kit.** Bench status: [GEN2_WOUT_KAGA_STATUS_2026-10-30.md](../hardware/GEN2_WOUT_KAGA_STATUS_2026-10-30.md).  
**Primary rollback (N−1):** [FIRMWARE_1.0.84_FLASH.md](./FIRMWARE_1.0.84_FLASH.md) · same folder  
**Canonical stamps:** [VERSION_ALIGNMENT.md](../VERSION_ALIGNMENT.md)

## Policy (data room)

Keep **current + rollback** under `docs/data_room/firmware/`:

| Role | Version | Guide |
|------|---------|--------|
| Gen 1 frozen | **1.0.85** | [FIRMWARE_1.0.85_FLASH.md](./FIRMWARE_1.0.85_FLASH.md) |
| Rollback | **1.0.84** | [FIRMWARE_1.0.84_FLASH.md](./FIRMWARE_1.0.84_FLASH.md) |

**1.0.82** remains in the same folder as a second prior. Older Gen1 packs (**1.0.70**, **1.0.66**, …) live in `oralable_nrf/artifacts/` only. Archived flash notes: [archive/firmware_guides/](./archive/firmware_guides/).

## When to roll back

- OTA or SWD of **1.0.85** fails field check (GATT `3A0FF006`, pad flash / hold pulse, IR-pulse worn).  
- Prefer **1.0.84** first (pad recover · desk abandon · IR-pulse worn).  
- On a Gen 1 board only: deeper than 1.0.84 is recovery; re-flash **1.0.85** when that board is stable. Do not put these files on Gen 2.

## Older guides (archive)

| Version | Note |
|---------|------|
| 1.0.82 | IR-pulse worn — [FIRMWARE_1.0.82_FLASH.md](./FIRMWARE_1.0.82_FLASH.md) |
| 1.0.70 | STAT blink / taper — [archive/firmware_guides/FIRMWARE_1.0.70_FLASH.md](./archive/firmware_guides/FIRMWARE_1.0.70_FLASH.md) |
| 1.0.66 | Pre-STAT activity — [archive/firmware_guides/FIRMWARE_1.0.66_FLASH.md](./archive/firmware_guides/FIRMWARE_1.0.66_FLASH.md) |
| 1.0.65 | Early vitals — [archive/firmware_guides/FIRMWARE_1.0.65_FLASH.md](./archive/firmware_guides/FIRMWARE_1.0.65_FLASH.md) |

iOS hard min remains **1.0.63**; recommend **1.0.85**.
