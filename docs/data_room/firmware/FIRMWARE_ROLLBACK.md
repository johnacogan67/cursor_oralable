# Firmware rollback (Gen1)

**Current target:** [FIRMWARE_1.0.84_FLASH.md](./FIRMWARE_1.0.84_FLASH.md) · pack binaries in [`firmware/`](./firmware/)  
**Primary rollback (N−1):** [FIRMWARE_1.0.82_FLASH.md](./FIRMWARE_1.0.82_FLASH.md) · same `firmware/` folder  
**Canonical stamps:** [VERSION_ALIGNMENT.md](../VERSION_ALIGNMENT.md)

## Policy (data room)

Keep only **current + one rollback** under `docs/data_room/firmware/`:

| Role | Version | Guide |
|------|---------|--------|
| Current | **1.0.84** | [FIRMWARE_1.0.84_FLASH.md](./FIRMWARE_1.0.84_FLASH.md) |
| Rollback | **1.0.82** | [FIRMWARE_1.0.82_FLASH.md](./FIRMWARE_1.0.82_FLASH.md) |

Older Gen1 packs (**1.0.70**, **1.0.66**, **1.0.65**, …) live in `oralable_nrf/artifacts/` only. Archived flash notes: [archive/firmware_guides/](./archive/firmware_guides/).

## When to roll back

- OTA or SWD of **1.0.84** fails field check (GATT `3A0FF006`, pad green, IR-pulse worn).  
- Prefer **1.0.82** first (IR-pulse worn · sense-on-BLE · STAT blink).  
- Deeper than 1.0.82 only for recovery; re-flash **1.0.84** when stable.

## Older guides (archive)

| Version | Note |
|---------|------|
| 1.0.70 | STAT blink / taper — [archive/firmware_guides/FIRMWARE_1.0.70_FLASH.md](./archive/firmware_guides/FIRMWARE_1.0.70_FLASH.md) |
| 1.0.66 | Pre-STAT activity — [archive/firmware_guides/FIRMWARE_1.0.66_FLASH.md](./archive/firmware_guides/FIRMWARE_1.0.66_FLASH.md) |
| 1.0.65 | Early vitals — [archive/firmware_guides/FIRMWARE_1.0.65_FLASH.md](./archive/firmware_guides/FIRMWARE_1.0.65_FLASH.md) |

iOS hard min remains **1.0.63**; recommend **1.0.84**.
