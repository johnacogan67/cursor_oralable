# Oralable research & validation documentation index

Python algorithm development, clinical protocols, and gold-standard validation (`cursor_oralable`). **Doc pack:** `docs/VERSION` → **1.3.11** (pilot FW **1.0.70** · app **4.3.3** — [VERSION_ALIGNMENT.md](./data_room/VERSION_ALIGNMENT.md)).

**Product roadmap (phases + BOM):** [PRODUCT_ROADMAP.md](./PRODUCT_ROADMAP.md)

**Product strategy:** [oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md](../../oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md) (competitors, regulatory, GTM, appendices A/B).

## Product phases & hardware

| Document | Description |
|----------|-------------|
| [PRODUCT_ROADMAP.md](./PRODUCT_ROADMAP.md) | **Canonical** Phase 0 / Phase 1+ / Gen2 features + Hardware ↔ BOM map |
| [IP_NORTH_STAR.md](./IP_NORTH_STAR.md) | **End goal** — Stage A wearable → Stage B medical; new US patent embodiment |
| [VITALS_PHASE_GEN1_GEN2.md](./VITALS_PHASE_GEN1_GEN2.md) | Pilot workarounds vs Gen2 hardware |
| [GEN1_GEN2_TRACKING.md](./GEN1_GEN2_TRACKING.md) | Living Gen1/Gen2 engineering timeline |
| [GEN1_GEN2_MIGRATION.md](./GEN1_GEN2_MIGRATION.md) | Capabilities, BOM delta, firmware map |
| [PCB00003_GEN2_REV11_HARDWARE.md](./PCB00003_GEN2_REV11_HARDWARE.md) | Gen2 REV11 nets / bring-up |

## Algorithms & signal processing

| Document | Description |
|----------|-------------|
| [ALGORITHM_ARCHITECTURE.md](./ALGORITHM_ARCHITECTURE.md) | Python ↔ Swift algorithm split, OralableCore integration |
| [IR_DC_ADC_FORMAT.md](./IR_DC_ADC_FORMAT.md) | Cheek IR-DC range (10M–70M), MAXM86161 ADC, `R_G_IR` order |
| [../README.md](../README.md) | Setup, `process_temporalis_gold.py`, clinical report scripts |

## Collection & protocol

| Document | Description |
|----------|-------------|
| [TEMPORALIS_COLLECTION_PROTOCOL.md](./TEMPORALIS_COLLECTION_PROTOCOL.md) | **Protocol A** (training, 5 taps) vs **Protocol B** (Ed/Pedro, 3-tap T=0) — read “do not mix” table |
| [data_room/PILOT_PROTOCOL_ED_PEDRO.md](./data_room/PILOT_PROTOCOL_ED_PEDRO.md) | Phase 1 pilot roles — phases canonical in Protocol B above |

## Investor / Point A data room

| Document | Description |
|----------|-------------|
| [data_room/README.md](./data_room/README.md) | Index (Ken 11 areas) · **v1.1.38** · pilot FW **1.0.70** · app **4.3.3** |
| [data_room/ED_PEDRO_QUICK_START.md](./data_room/ED_PEDRO_QUICK_START.md) | Phase 0 Vitals one-pager |
| [data_room/VERSION_ALIGNMENT.md](./data_room/VERSION_ALIGNMENT.md) | Canonical FW **1.0.70** · app **4.3.3** |
| [data_room/FIRMWARE_1.0.70_FLASH.md](./data_room/FIRMWARE_1.0.70_FLASH.md) | Pilot flash guide |
| [data_room/ORALABLE_FTS_36MO.md](./data_room/ORALABLE_FTS_36MO.md) | 36-month functional & technical specification |
| [data_room/REGULATORY_TIMELINE.md](./data_room/REGULATORY_TIMELINE.md) | Wellness → 510(k) / CE timeline |
| [data_room/GTM_ONE_PAGE.md](./data_room/GTM_ONE_PAGE.md) | GTM, pricing, CAC assumptions |
| [data_room/COST_AND_TIMELINE.md](./data_room/COST_AND_TIMELINE.md) | Stage A→B cost ranges + timeline (planning) |
| [data_room/PITCH_DECK_KEN.md](./data_room/PITCH_DECK_KEN.md) | Pitch distill for Ken / BalancePoints |
| [data_room/CURRENT_GOVERNANCE_STATUS.md](./data_room/CURRENT_GOVERNANCE_STATUS.md) | **Governance as-at 22 Jul** — cap table, Conor buyout, Ken/Nigel roles |
| [data_room/JAC_CORPORATE_STRUCTURE_AND_GOVERNANCE.md](./data_room/JAC_CORPORATE_STRUCTURE_AND_GOVERNANCE.md) | Canonical corporate / cap table / INTERNAL annex |
| [data_room/MEETING_BRIEF_KEN_NIGEL_2026-07-22.md](./data_room/MEETING_BRIEF_KEN_NIGEL_2026-07-22.md) | John/Ken/Nigel meeting prep (Conor deadline 23 Jul) |
| [data_room/FUNDING_POINT_B_AND_CAP_TABLE.md](./data_room/FUNDING_POINT_B_AND_CAP_TABLE.md) | Point B €180k + Register of Members distill |

## Clinical evaluation

| Document | Description |
|----------|-------------|
| [CLINICAL_VALIDATION.md](./CLINICAL_VALIDATION.md) | Oralable_7, Ed Owens package, self-validation gap analysis |

## Engineering (internal)

| Document | Description |
|----------|-------------|
| [ORALABLE_SYSTEM_ARCHITECTURE.md](./ORALABLE_SYSTEM_ARCHITECTURE.md) | Living system architecture, truth lockdown, §3 status matrix, **NotebookLM engineering bundle** |
| [internal/CLAUDE_IOS_REFACTOR_INSTRUCTIONS.md](./internal/CLAUDE_IOS_REFACTOR_INSTRUCTIONS.md) | iOS + OralableCore refactor (status table — many items done) |
| [upload/ORALABLE_COMBINED.md](./upload/ORALABLE_COMBINED.md) | **Deprecated** — PDF/partner export only; do not upload with architecture hub |

## Cross-repo

| Repo | Role |
|------|------|
| **oralable_nrf** | Firmware, GATT, OTA — `docs/README.md` |
| **oralable_swift** | Consumer + dentist apps, CloudKit, [MOBILE_APP_FLOWS.md](../../oralable_swift/docs/MOBILE_APP_FLOWS.md) |
| **OralableCore** | Shared BLE parsing, TFI/SASHB handshake export |

## Standard pipeline

1. Record per [TEMPORALIS_COLLECTION_PROTOCOL.md](./TEMPORALIS_COLLECTION_PROTOCOL.md)
2. `python scripts/process_temporalis_gold.py <ble_log.csv>`
3. `python scripts/generate_clinical_report.py --input data/validation/GOLD_STANDARD_VALIDATION.csv`
4. Optional: `oralable_nrf/scripts/check_ir_dc_scaling.py` on new logs

## NotebookLM

**Engineering set:** [ORALABLE_SYSTEM_ARCHITECTURE.md](./ORALABLE_SYSTEM_ARCHITECTURE.md) §17 (hub + IR_DC + clinical + DEVELOPMENT + CSVs). **Do not** add `upload/ORALABLE_COMBINED.md`.

**Investor set:** `data_room/*` + `oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md` (separate notebook).

*Last updated: July 2026*
