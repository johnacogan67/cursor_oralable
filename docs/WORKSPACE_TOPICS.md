# Workspace topics — agent routing

Use this file when you start a Cursor agent and want it to stay on one topic.

**This is the topic map.** Other indexes exist; they do not replace this file.

| Existing map | What it is | Not for |
|--------------|------------|---------|
| [docs/README.md](./README.md) | Research / validation doc index | Agent prompts |
| [data_room/README.md](./data_room/README.md) | Ken / Balance Points diligence (11 areas) | Code routing |
| [ORALABLE_SYSTEM_ARCHITECTURE.md](./ORALABLE_SYSTEM_ARCHITECTURE.md) | Engineering truth (layers L0–L5) | Topic slugs |
| [ORALABLE_SYSTEM_MAP_DIAGRAMS.md](./ORALABLE_SYSTEM_MAP_DIAGRAMS.md) | Mermaid / pitch diagrams | Agent prompts |
| [upload/03_GITHUB_REPOS_OVERVIEW.txt](./upload/03_GITHUB_REPOS_OVERVIEW.txt) | **Deprecated** four-repo snapshot | Current truth |

**Versions:** [VERSION_ALIGNMENT.md](./data_room/VERSION_ALIGNMENT.md) · FW **1.0.82** · app **4.3.3** (do not invent numbers).

---

## How to address an agent

**Quick starters:** [AGENTS.md](../AGENTS.md) — copy-paste block per slug.

Paste this as the first line of a custom agent, or the first message of a chat:

```
Topic: <slug>
Read cursor_oralable/docs/WORKSPACE_TOPICS.md. Stay in that topic. Open only the listed repos and start-here files unless I ask to cross topics.
```

Replace `<slug>` with a row from the table below (`firmware`, `algorithms`, `clinical`, …).

If the work spans two topics, name both (`Topic: firmware + core`) and say which repo may change.

**Workspace file:** open all four repos via [oralable.code-workspace](../oralable.code-workspace).

---

## Four repos (this Cursor workspace)

Open via [oralable.code-workspace](../oralable.code-workspace).

| Repo | Role | Do not treat as |
|------|------|-----------------|
| **cursor_oralable** | Python gold, protocols, data room, this map | Shipping firmware or iOS UI |
| **oralable_nrf** | nRF firmware, GATT, OTA, boards | Algorithm numbers or app UX |
| **oralable_swift** | Patient + dentist iOS apps | Shared parse/math (that is Core) |
| **OralableCore** | Shared BLE parse, biometrics, CSV/EDF, Core ML | Screen layout or Zephyr |

Nearby but **not** in this workspace: `oralable-web`. Corporate / legal folders sit outside git.

**Mac → phone rule:** prove algorithms in `cursor_oralable`, then port to `OralableCore` + the app. Default: change the phone to match Mac. See [ALGORITHM_ARCHITECTURE.md](./ALGORITHM_ARCHITECTURE.md) and [MAC_PHONE_ALGORITHM_PARITY.md](./data_room/MAC_PHONE_ALGORITHM_PARITY.md).

---

## Topic slugs

| Slug | One line | Repos | Start here |
|------|----------|-------|------------|
| `firmware` | nRF firmware, GATT, OTA, nRF Connect | `oralable_nrf` | `oralable_nrf/docs/README.md` · `oralable_nrf/docs/DEVELOPMENT.md` |
| `hardware` | PCB, BOM, Gen1/Gen2, charge, Altium | `oralable_nrf` · `cursor_oralable` | [PRODUCT_ROADMAP.md](./PRODUCT_ROADMAP.md) · [HW_ENGINEER_ALTIUM_BRIEF.md](./data_room/HW_ENGINEER_ALTIUM_BRIEF.md) |
| `ios-patient` | Patient app, vitals UX, Dual A session | `oralable_swift` · `OralableCore` | `oralable_swift/docs/MOBILE_APP_FLOWS.md` |
| `ios-dentist` | Dentist app, CloudKit (dark / Phase 1+) | `oralable_swift` · `OralableCore` | `MOBILE_APP_FLOWS.md` §5–7 · [APPS_AND_REVENUE_EVAL.md](./data_room/APPS_AND_REVENUE_EVAL.md) |
| `core` | Shared parse, biometrics, export, Core ML | `OralableCore` | `OralableCore/docs/README.md` |
| `algorithms` | Mac Python gold, filters, TFI/SASHB, parity | `cursor_oralable` · `OralableCore` | [ALGORITHM_ARCHITECTURE.md](./ALGORITHM_ARCHITECTURE.md) |
| `overnight` | Night report, hypnogram, state machine | `cursor_oralable` · `oralable_swift` · `OralableCore` | [OVERNIGHT_NIGHT_REPORT.md](./OVERNIGHT_NIGHT_REPORT.md) |
| `clinical` | Protocol A/B, Dual A, ANR, Paper A | `cursor_oralable` | [TEMPORALIS_COLLECTION_PROTOCOL.md](./TEMPORALIS_COLLECTION_PROTOCOL.md) |
| `research-kit` | Ed/Pedro kits, flash, handoff | `cursor_oralable` | [ORALABLE_RESEARCH_KIT.md](./data_room/ORALABLE_RESEARCH_KIT.md) |
| `data-room` | Investor index, pitches, Ken areas | `cursor_oralable` | [data_room/README.md](./data_room/README.md) |
| `ip` | Patents, landscape, north star | `cursor_oralable` | [IP_NORTH_STAR.md](./IP_NORTH_STAR.md) · [IP_PORTFOLIO_STATUS.md](./data_room/IP_PORTFOLIO_STATUS.md) |
| `regulatory` | Wellness → 510(k) / CE | `cursor_oralable` | [REGULATORY_TIMELINE.md](./data_room/REGULATORY_TIMELINE.md) |
| `gtm` | Market, GTM, competitors | `cursor_oralable` · `oralable_nrf` | [GTM_ONE_PAGE.md](./data_room/GTM_ONE_PAGE.md) · `ORALABLE_MARKET_LANDSCAPE.md` |
| `governance` | Cap table, Ken/Nigel, funding | `cursor_oralable` | [CURRENT_GOVERNANCE_STATUS.md](./data_room/CURRENT_GOVERNANCE_STATUS.md) |
| `brand` | Figures, look, Hemingway/Orwell voice | `cursor_oralable` | [FIGURES.md](./FIGURES.md) · [VISUAL_AND_VOICE_DIRECTION.md](./data_room/VISUAL_AND_VOICE_DIRECTION.md) |

---

## Topic detail

### `firmware`

nRF52832 Gen1 ship (`pcb00003`). GATT TGM `3A0FF000`. OTA. nRF Connect is the BLE ground truth.

- **Code:** `oralable_nrf/app/src/` (`main.c`, `tgm_service.c`, `ble.c`, `ppg.c`, `charge_detector.c`)
- **Boards:** `oralable_nrf/boards/byteexplain/pcb00003/` · Gen2 stub `pcb00003_gen2/`
- **Docs:** `oralable_nrf/docs/DEVELOPMENT.md` · `OTA_DEVICE_MANAGER.md` · [FIRMWARE_1.0.82_FLASH.md](./data_room/FIRMWARE_1.0.82_FLASH.md)
- **Rule:** `oralable_nrf/.cursor/rules/nrf-connect-validation.mdc`
- **Do not:** change iOS parse to “fix” a firmware byte without an nRF Connect CSV.

### `hardware`

Pins, BOM, magnetic case, Gen1 vs Gen2. Lock from schematic / DTS / BOM, not chat memory.

- **Docs:** [PRODUCT_ROADMAP.md](./PRODUCT_ROADMAP.md) §1 · [GEN1_GEN2_MIGRATION.md](./GEN1_GEN2_MIGRATION.md) · [PCB00003_GEN2_REV11_HARDWARE.md](./PCB00003_GEN2_REV11_HARDWARE.md) · `oralable_nrf/docs/HARDWARE_ROADMAP_nRF54L15.md`
- **Quotes:** `docs/data_room/GEN1_COGS_KAGA_QUOTE.md` · `GEN2_COGS_KAGA_QUOTE.md` · `HW_ENGINEER_ALTIUM_BRIEF.md`
- **Architecture layers:** L1 hardware in [ORALABLE_SYSTEM_ARCHITECTURE.md](./ORALABLE_SYSTEM_ARCHITECTURE.md) §1–2

### `ios-patient`

Phase 0 temple vitals. Placement picker. Dual A session. Night PDF. Ed/Pedro ship the patient app only.

- **Code:** `oralable_swift/OralableApp/OralableApp/`
- **Docs:** `oralable_swift/docs/MOBILE_APP_FLOWS.md` · `OralableApp/LAUNCH_READINESS_CHECKLIST.md`
- **Dual A:** `Services/DualProtocolA/` · `Views/DualProtocolA/`
- **Do not:** turn on dentist app or CloudKit share for Ed/Pedro kits.

### `ios-dentist`

Professional app. CloudKit handshake. Dark until Phase 1+.

- **Code:** `oralable_swift/OralableApp/OralableForProfessionals/`
- **Shared:** OralableCore CloudKit + `AutomaticRecordingSession`
- **Docs:** `MOBILE_APP_FLOWS.md` §5–7 · [APPS_AND_REVENUE_EVAL.md](./data_room/APPS_AND_REVENUE_EVAL.md)

### `core`

Shared Swift package. BLE parse, buffers, HR/SpO2, MAM Core ML, CSV/EDF, design tokens.

- **Code:** `OralableCore/Sources/OralableCore/` (`BLE/`, `Calculations/`, `Export/`, `Signal/`)
- **Tests:** `OralableCore/Tests/OralableCoreTests/` · golden `GOLD_STANDARD_FILTER_PARITY.csv`
- **Docs:** `OralableCore/docs/README.md` · [ALGORITHM_ARCHITECTURE.md](./ALGORITHM_ARCHITECTURE.md)

### `algorithms`

Mac Python is the clinical reference. 50 Hz. Butterworth 0.5–8 Hz for HR. IR-DC low-pass for occlusion.

- **Python:** `cursor_oralable/src/analysis/features.py` · `overnight_states.py` · `src/processing/resampler.py`
- **Scripts:** `scripts/process_temporalis_gold.py` · `generate_clinical_report.py` · `generate_overnight_night_report.py`
- **Phone:** `OralableCore` + `oralable_swift` `UnifiedBiometricProcessor`
- **Docs:** [ALGORITHM_ARCHITECTURE.md](./ALGORITHM_ARCHITECTURE.md) · [IR_DC_ADC_FORMAT.md](./IR_DC_ADC_FORMAT.md) · [MAC_PHONE_ALGORITHM_PARITY.md](./data_room/MAC_PHONE_ALGORITHM_PARITY.md)
- **Do not:** invent phone defaults Mac would leave missing (e.g. fake SpO₂ 98%).

### `overnight`

State hypnogram is the primary night view. Evaluable overnight ≥6 h worn (goal 8 h). In-app unlock can be shorter — do not mix those gates.

- **Python:** `src/analysis/overnight_states.py` · `scripts/generate_overnight_night_report.py`
- **iOS:** `OvernightNightReportBuilder` · `OvernightStateClassifier` · `StateHypnogramView`
- **Docs:** [OVERNIGHT_NIGHT_REPORT.md](./OVERNIGHT_NIGHT_REPORT.md)

### `clinical`

Protocol A (training, 5 taps) vs Protocol B (Ed/Pedro, 3-tap T=0). Do not mix. Dual A = MAM + ANR M40. User-facing name is **MAM**; BLE still advertises **Oralable**; GATT/code stays **TGM**.

- **Docs:** [TEMPORALIS_COLLECTION_PROTOCOL.md](./TEMPORALIS_COLLECTION_PROTOCOL.md) · [ANR_M40_CONCORDANCE.md](./ANR_M40_CONCORDANCE.md) · [CLINICAL_VALIDATION.md](./CLINICAL_VALIDATION.md) · [CORE_ML_TRAINING_COHORT.md](./CORE_ML_TRAINING_COHORT.md)
- **Paper A:** `docs/data_room/PAPER_A_*.md`
- **Code:** `scripts/run_protocol_a_session.py` · `run_anr_emg_session.py` · `src/analysis/emg_spo2_nest.py`

### `research-kit`

Five kits to Pedro, gated on charge-to-temple. Patient app only.

- **Docs:** [ORALABLE_RESEARCH_KIT.md](./data_room/ORALABLE_RESEARCH_KIT.md) · [ED_PEDRO_QUICK_START.md](./data_room/ED_PEDRO_QUICK_START.md) · [PAPER_A_DATA_HANDOFF_SOP.md](./data_room/PAPER_A_DATA_HANDOFF_SOP.md) · [PEDRO_STATUS_UPDATE_2026-08.md](./data_room/PEDRO_STATUS_UPDATE_2026-08.md)
- **Flash:** [FIRMWARE_1.0.82_FLASH.md](./data_room/FIRMWARE_1.0.82_FLASH.md)

### `data-room`

Investor pack. Ken’s 11 areas live in [data_room/README.md](./data_room/README.md). Use that table for diligence; use this file for agent scope.

- **Pitches:** `PITCH_DECK_KEN` · `PITCH_CEO_CANDIDATE` · `PITCH_KOOROSH` · `PITCH_PEDRO_ED_FF` · `PITCH_TECH_OPERATORS`
- **Do not:** inflate kit counts, overnight *N*, AHI equivalence, or FDA/CE claims.

### `ip`

Stage A wearable → Stage B medical. Do not paste patent claim text or provisional specification wording into chats or new docs.

- **Docs:** [IP_NORTH_STAR.md](./IP_NORTH_STAR.md) · [IP_PORTFOLIO_STATUS.md](./data_room/IP_PORTFOLIO_STATUS.md) · [IP_EVAL_AND_LANDSCAPE.md](./data_room/IP_EVAL_AND_LANDSCAPE.md)

### `regulatory`

Wellness wording now. Stage B later. No FDA/CE claims in code comments or Core docs.

- **Docs:** [REGULATORY_TIMELINE.md](./data_room/REGULATORY_TIMELINE.md) · architecture §2 product truth

### `gtm`

Market sketch, GTM one-pager, competitor landscape.

- **Docs:** [GTM_ONE_PAGE.md](./data_room/GTM_ONE_PAGE.md) · [MARKET_SIZING.md](./data_room/MARKET_SIZING.md) · `oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md` · [DIANYX_FDA_AND_SMART_OAT_LANDSCAPE.md](./data_room/DIANYX_FDA_AND_SMART_OAT_LANDSCAPE.md)

### `governance`

Living snapshot first. Cap table and Point B sit in data room; statutory packs sit outside git.

- **Docs:** [CURRENT_GOVERNANCE_STATUS.md](./data_room/CURRENT_GOVERNANCE_STATUS.md) · [JAC_CORPORATE_STRUCTURE_AND_GOVERNANCE.md](./data_room/JAC_CORPORATE_STRUCTURE_AND_GOVERNANCE.md) · [FUNDING_POINT_B_AND_CAP_TABLE.md](./data_room/FUNDING_POINT_B_AND_CAP_TABLE.md)

### `brand`

Figure IDs stay locked. Prose is Hemingway/Orwell. Look is locked in the visual note.

- **Docs:** [FIGURES.md](./FIGURES.md) · [VISUAL_AND_VOICE_DIRECTION.md](./data_room/VISUAL_AND_VOICE_DIRECTION.md) · `docs/data_room/brand/`
- **Rule:** `.cursor/rules/prose-hemingway-orwell.mdc` (all four repos)

---

## Ken diligence areas (investor, not agent slugs)

These are Balance Points Point A weights. Map them to slugs if a diligence agent needs a code/doc home.

| Ken area | Use slug |
|----------|----------|
| Technology & Product | `firmware` + `ios-patient` + `core` + `algorithms` |
| Market & Positioning | `gtm` |
| GTM & Sales | `gtm` |
| User Traction & Revenue | `data-room` (gap) |
| Financials | `governance` |
| Team & Governance | `governance` |
| Legal, IP & Corp | `ip` |
| Risk, Regulation & Compliance | `regulatory` |
| International Readiness | `gtm` |
| Data Room & Investment Docs | `data-room` |
| Voice | `brand` |

---

## Architecture layers (when a change crosses topics)

From [ORALABLE_SYSTEM_ARCHITECTURE.md](./ORALABLE_SYSTEM_ARCHITECTURE.md) §1. Order matters.

| Layer | Topic | Truth |
|-------|-------|-------|
| L0 Product | `gtm` / `regulatory` | Wear site, phase, claims |
| L1 Hardware | `hardware` | Pins, BOM, polarity |
| L2 BLE contract | `firmware` + `core` | UUIDs, payload sizes |
| L3 Firmware policy | `firmware` | Worn, charger, LED, stream gates |
| L4 Algorithms | `algorithms` | 50 Hz, filters, TFI/SASHB |
| L5 Clinical | `clinical` | Protocol phases, gold CSVs |

---

## Cursor rules already on disk

Always-on (all four repos): plan-mode switch · bookmark sources · Hemingway/Orwell prose.

Topic-scoped:

| Rule | Topic |
|------|-------|
| `oralable_nrf/.cursor/rules/nrf-connect-validation.mdc` | `firmware` |
| `cursor_oralable/.cursorrules` | MAM / 50 Hz / cheek IR-DC defaults |

This file is the rest of the routing. Do not copy the whole map into every agent. Name a slug and point here.
