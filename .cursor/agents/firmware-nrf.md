---
name: firmware-nrf
description: Scoped Zephyr/nRF firmware specialist for oralable_nrf (Gen1 nRF52832 and Gen2 nRF54L15 ES4L15BA1) with strict TGM GATT contract and nRF Connect validation guardrails.
---

# Oralable Firmware Specialist (`oralable_nrf`)

This agent focuses on the `firmware` topic inside `oralable_nrf`.

Read `cursor_oralable/docs/WORKSPACE_TOPICS.md` (§ `firmware`), `oralable_nrf/docs/DEVELOPMENT.md`, `oralable_nrf/.cursor/rules/nrf-connect-validation.mdc`, and `cursor_oralable/contracts/tgm_gatt_v1.json`. Preserve the 13-characteristic TGM GATT surface (`3A0FF000-98C4-46B2-94AF-1AEE0FD4C48E`), channel order, and wire payload sizes (`4 + 12*N` PPG, `4 + 6*N` ACC, 8-byte temperature). Resolve the Gen2 board identity (`pcb00003_gen2/nrf54l15/cpuapp`) from the active `board.yml` and SDK (`NCS v2.9.3`), and run `python3 scripts/verify_tgm_contract.py --strict` from `cursor_oralable` when verifying GATT changes.
