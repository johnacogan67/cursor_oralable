# Oralable research & validation documentation index

Python algorithm development, clinical protocols, and gold-standard validation (`cursor_oralable`). **Doc pack:** `docs/VERSION` → **1.3.16** (pilot FW **1.0.82** · app **4.3.3** — Gemini tech avenues + timeline — [VERSION_ALIGNMENT.md](./data_room/VERSION_ALIGNMENT.md)).

**Agent topic map:** [WORKSPACE_TOPICS.md](./WORKSPACE_TOPICS.md) · copy-paste starters [AGENTS.md](../AGENTS.md)

**Product roadmap (phases + BOM + canonical timeline §3 + tech avenues §2b):** [PRODUCT_ROADMAP.md](./PRODUCT_ROADMAP.md)

**Technology avenues distill:** [data_room/GEMINI_TEMPLE_PPG_AVENUES.md](./data_room/GEMINI_TEMPLE_PPG_AVENUES.md) (external exploration — not product claims)

**System map (visual / spreadsheet / mermaid):** [ORALABLE_SYSTEM_MAP.csv](./ORALABLE_SYSTEM_MAP.csv) · [ORALABLE_SYSTEM_MAP_DIAGRAMS.md](./ORALABLE_SYSTEM_MAP_DIAGRAMS.md) · Cursor canvas `oralable-system-map`

**Figures (master inventory):** [FIGURES.md](./FIGURES.md) · assets in [`figures/`](./figures/) · data-room copies [`data_room/figures/`](./data_room/figures/)

**Patient app working diagrams:** [oralable_swift/docs/MOBILE_APP_FLOWS.md §2](../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0) · Mermaid also in [ORALABLE_SYSTEM_MAP_DIAGRAMS.md §8](./ORALABLE_SYSTEM_MAP_DIAGRAMS.md)

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
| [TEMPORALIS_COLLECTION_PROTOCOL.md](./TEMPORALIS_COLLECTION_PROTOCOL.md) | **Protocol A** (training, 5 taps) vs **Protocol B** (Ed/Pedro, 3-tap T=0) — read “do not mix” table · Dual A (ANR) |
| [ANR_M40_CONCORDANCE.md](./ANR_M40_CONCORDANCE.md) | ANR M40 temporalis sEMG vs Oralable (Paper C) — Mac dual-BLE first |
| [OVERNIGHT_NIGHT_REPORT.md](./OVERNIGHT_NIGHT_REPORT.md) | **Canonical** overnight bands (BP-style), **state hypnogram = very useful primary** ([FIG-CO-025](./figures/FIG-CO-025-state-hypnogram-exemplar.png)); **in-app** + Mac/iOS PDF; ≥6 h gate |
| [CORE_ML_TRAINING_COHORT.md](./CORE_ML_TRAINING_COHORT.md) | **Canonical** Protocol A cohort sizes, users/sessions, demographics, leave-user-out |
| [data_room/PILOT_PROTOCOL_ED_PEDRO.md](./data_room/PILOT_PROTOCOL_ED_PEDRO.md) | Phase 1 pilot roles — phases canonical in Protocol B above |

## Investor / Point A data room

| Document | Description |
|----------|-------------|
| [data_room/README.md](./data_room/README.md) | Index (Ken 11 areas) · **v1.1.67** · Research Kit · Dual A measured + EDF · pilot FW **1.0.82** · app **4.3.3** |
| [data_room/COLLAB_NABAVI_MCGILL.md](./data_room/COLLAB_NABAVI_MCGILL.md) | Nabavi collab truth — McGill / RI-MUHC · not Dianyx · Ed/Pedro · Wout Altium |
| [data_room/LITERATURE_AND_PRIOR_ART.md](./data_room/LITERATURE_AND_PRIOR_ART.md) | TEC literature distill (Li ambulatory SB · BruxScreen · chewing PPG · encaps) |
| [data_room/PITCH_KOOROSH.pdf](./data_room/PITCH_KOOROSH.pdf) | Nabavi leave-behind (McGill IEEE + optional Altium) |
| [data_room/KOOROSH_OUTREACH.md](./data_room/KOOROSH_OUTREACH.md) | Email to seyed.nabavi@mcgill.ca |
| [data_room/ED_PEDRO_QUICK_START.md](./data_room/ED_PEDRO_QUICK_START.md) | Phase 0 Vitals one-pager (Ed Owens / Pedro Beacon) |
| [data_room/VERSION_ALIGNMENT.md](./data_room/VERSION_ALIGNMENT.md) | Canonical FW **1.0.82** · app **4.3.3** |
| [data_room/FIRMWARE_1.0.82_FLASH.md](./data_room/FIRMWARE_1.0.82_FLASH.md) | Pilot flash / OTA guide |
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
| **oralable_swift** | Consumer + dentist apps, CloudKit, [MOBILE_APP_FLOWS.md](../../oralable_swift/docs/MOBILE_APP_FLOWS.md) ([§2 working diagrams](../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)) |
| **OralableCore** | Shared BLE parsing, TFI/SASHB handshake export — [docs/README.md](../../OralableCore/docs/README.md) · [FIGURES.md](./FIGURES.md) |

## Standard pipeline

1. Record per [TEMPORALIS_COLLECTION_PROTOCOL.md](./TEMPORALIS_COLLECTION_PROTOCOL.md)  
   - Structured: Protocol A/B (minutes).  
   - **Evaluable overnight:** **≥ 6 h** worn (goal 8 h).
2. `python scripts/process_temporalis_gold.py <ble_log.csv>`
3. `python scripts/generate_clinical_report.py --input data/validation/GOLD_STANDARD_VALIDATION.csv`  
   (also writes overnight graphic pack under `plots/overnight_report/`)
4. Optional night pack only: `python scripts/generate_overnight_night_report.py --input data/validation/GOLD_STANDARD_VALIDATION.csv`
5. Optional: `oralable_nrf/scripts/check_ir_dc_scaling.py` on new logs  
6. **iOS:** Share → Clinical Temporalis PDF (same bout panels + event CSV; needs session samples / flushes)

## NotebookLM

**Engineering set:** [ORALABLE_SYSTEM_ARCHITECTURE.md](./ORALABLE_SYSTEM_ARCHITECTURE.md) §17 (hub + IR_DC + clinical + DEVELOPMENT + CSVs). **Do not** add `upload/ORALABLE_COMBINED.md`.

**Investor set:** `data_room/*` + `oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md` (separate notebook).

*Last updated: July 2026*
