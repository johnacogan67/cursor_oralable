# Oralable figures inventory (master)

**Hub for all four repos.** Mermaid stays inline in Markdown for flows/architecture. Named SVG/PNG under `docs/figures/` for anatomy, product, hardware photos, and UI mocks. Status: `placeholder` | `draft` | `final`.

**Sibling indexes:** [oralable_nrf/docs/FIGURES.md](../../oralable_nrf/docs/FIGURES.md) · [oralable_swift/docs/FIGURES.md](../../oralable_swift/docs/FIGURES.md) · [OralableCore/docs/FIGURES.md](../../OralableCore/docs/FIGURES.md)

**Patient app working diagrams (Mermaid):** [oralable_swift/docs/MOBILE_APP_FLOWS.md §2](../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0) — session lifecycle, tabs, placement state, auto-record, BLE→UI (`FIG-IOS-006`…`008`). Also mirrored in [ORALABLE_SYSTEM_MAP_DIAGRAMS.md §8](./ORALABLE_SYSTEM_MAP_DIAGRAMS.md). Cross-linked from product, pilot, FTS, pitches, nrf landscape/DEVELOPMENT, and OralableCore docs (31 Jul 2026).

**Brand (not numbered FIG):** [data_room/brand/](./data_room/brand/) — logo lockups only.

**Pitch self-contained copies:** [data_room/figures/](./data_room/figures/) — same IDs as shared research figures when decks need local paths.

## Maintenance

1. New diagram → add a row here **first**, then embed.
2. Replace art by overwriting the same filename; flip status; keep caption ID stable.
3. Pitch HTML/PDF regen uses `data_room/figures/` + `data_room/brand/`; do not invent one-off slide-only names.
4. Do not put confidential claim text in SVG titles meant for external decks.
5. Clinical raw plots live under `data/plots/`; promote selected embeds here as `FIG-CO-xxx` (symlink path or copy later).

## Template

Copy [`figures/_placeholder.svg`](./figures/_placeholder.svg) → `FIG-CO-NNN-slug.svg`, update the three text lines inside the SVG, then embed:

```markdown
![FIG-CO-012 Gen1 device](./figures/FIG-CO-012-gen1-device-photo.svg)

*Figure FIG-CO-012 — Gen1 device photo (placeholder).*
```

## Inventory — cursor_oralable (`FIG-CO-*`)

| ID | Title | Path | Status | Owning doc(s) | Notes |
|----|-------|------|--------|---------------|-------|
| FIG-CO-001 | Stage A → Stage B pathway | [figures/FIG-CO-001-stage-ab-pathway.svg](./figures/FIG-CO-001-stage-ab-pathway.svg) | placeholder | IP_NORTH_STAR | Prefer Mermaid in-doc; SVG optional for decks |
| FIG-CO-002 | Patent embodiment stack | [figures/FIG-CO-002-patent-embodiment-stack.svg](./figures/FIG-CO-002-patent-embodiment-stack.svg) | placeholder | IP_NORTH_STAR | Block art for embodiment layers |
| FIG-CO-003 | Temporalis clip placement | [figures/FIG-CO-003-temple-placement.svg](./figures/FIG-CO-003-temple-placement.svg) | placeholder | TEMPORALIS_COLLECTION_PROTOCOL, ED_PEDRO_QUICK_START, data_room/figures | Replace with temple photo |
| FIG-CO-004 | Protocol B 3-tap sync | [figures/FIG-CO-004-three-tap-sync.svg](./figures/FIG-CO-004-three-tap-sync.svg) | placeholder | TEMPORALIS_COLLECTION_PROTOCOL | Accel spike sketch |
| FIG-CO-005 | Protocol A 5-tap sync | [figures/FIG-CO-005-protocol-a-five-tap.svg](./figures/FIG-CO-005-protocol-a-five-tap.svg) | placeholder | TEMPORALIS_COLLECTION_PROTOCOL | Training cohort |
| FIG-CO-006 | IR-DC occlusion trough | [figures/FIG-CO-006-ir-dc-occlusion-trough.svg](./figures/FIG-CO-006-ir-dc-occlusion-trough.svg) | placeholder | ALGORITHM_ARCHITECTURE, IR_DC_ADC_FORMAT | Cross-check clench vs DC trough |
| FIG-CO-007 | 50 Hz PPG pipeline | [figures/FIG-CO-007-ppg-50hz-pipeline.svg](./figures/FIG-CO-007-ppg-50hz-pipeline.svg) | placeholder | ALGORITHM_ARCHITECTURE | Mermaid preferred in-doc |
| FIG-CO-008 | Overnight night report layout | [figures/FIG-CO-008-night-report-layout.svg](./figures/FIG-CO-008-night-report-layout.svg) | placeholder | OVERNIGHT_NIGHT_REPORT | PDF/UI mock |
| FIG-CO-009 | Oralable_7 from sync1 | [figures/FIG-CO-009-oralable7-from-sync1.svg](./figures/FIG-CO-009-oralable7-from-sync1.svg) | placeholder | CLINICAL_VALIDATION | Promote `data/plots/.../oralable7_from_sync1.png` |
| FIG-CO-010 | Oralable_7 validation dashboard | [figures/FIG-CO-010-oralable7-validation-dashboard.svg](./figures/FIG-CO-010-oralable7-validation-dashboard.svg) | placeholder | CLINICAL_VALIDATION | Promote `.../oralable7_validation_dashboard.png` |
| FIG-CO-011 | Extraoral vs intraoral | [figures/FIG-CO-011-extraoral-vs-intraoral.svg](./figures/FIG-CO-011-extraoral-vs-intraoral.svg) | placeholder | LITERATURE_AND_PRIOR_ART, COLLAB_NABAVI_MCGILL, data_room/figures | Oralable temple vs Dianyx intraoral |
| FIG-CO-012 | Gen1 device photo | [figures/FIG-CO-012-gen1-device-photo.svg](./figures/FIG-CO-012-gen1-device-photo.svg) | placeholder | HW_ENGINEER_ALTIUM_BRIEF, pitches | BOM REV8 / PCB REV10 |
| FIG-CO-013 | Magnetic charge case | [figures/FIG-CO-013-magnetic-case.svg](./figures/FIG-CO-013-magnetic-case.svg) | placeholder | VITALS_PHASE_GEN1_GEN2, ED_PEDRO_QUICK_START | |
| FIG-CO-014 | PCB REV10 photo | [figures/FIG-CO-014-pcb-rev10-photo.svg](./figures/FIG-CO-014-pcb-rev10-photo.svg) | placeholder | PCB00003_GEN2_REV11_HARDWARE, HW brief | |
| FIG-CO-015 | Altium board overview | [figures/FIG-CO-015-altium-board-overview.svg](./figures/FIG-CO-015-altium-board-overview.svg) | placeholder | HW_ENGINEER_ALTIUM_BRIEF | From Wout / WeeGee |
| FIG-CO-016 | Ed/Pedro kit contents | [figures/FIG-CO-016-ed-pedro-kit-contents.svg](./figures/FIG-CO-016-ed-pedro-kit-contents.svg) | placeholder | ED_PEDRO_QUICK_START, data_room/figures | |
| FIG-CO-017 | Cheek vs temple sites | [figures/FIG-CO-017-cheek-vs-temple-sites.svg](./figures/FIG-CO-017-cheek-vs-temple-sites.svg) | placeholder | TEMPORALIS_COLLECTION_PROTOCOL, PRODUCT_ROADMAP | Pilot = temple |
| FIG-CO-018 | SASHB bout example | [figures/FIG-CO-018-sashb-bout-example.svg](./figures/FIG-CO-018-sashb-bout-example.svg) | placeholder | CLINICAL_VALIDATION, ALGORITHM_ARCHITECTURE | |
| FIG-CO-019 | Overnight hypnogram bands | [figures/FIG-CO-019-hypnogram-bands.svg](./figures/FIG-CO-019-hypnogram-bands.svg) | placeholder | OVERNIGHT_NIGHT_REPORT | Band chips stub; pair with FIG-CO-025 |
| FIG-CO-020 | Core ML MAM flow | [figures/FIG-CO-020-coreml-mam-flow.svg](./figures/FIG-CO-020-coreml-mam-flow.svg) | placeholder | ALGORITHM_ARCHITECTURE, CORE_ML_TRAINING_COHORT | Cross-link FIG-CORE-* |
| FIG-CO-021 | Temple lifestyle photo | [figures/FIG-CO-021-system-stack-photo.svg](./figures/FIG-CO-021-system-stack-photo.svg) | placeholder | pitches, GTM | External-safe |
| FIG-CO-022 | Charge-to-temple flow | [figures/FIG-CO-022-pilot-charge-to-temple.svg](./figures/FIG-CO-022-pilot-charge-to-temple.svg) | placeholder | ED_PEDRO_QUICK_START, PILOT_DRY_RUN | |
| FIG-CO-023 | Tape vs silicone potting | [figures/FIG-CO-023-silicone-vs-tape.svg](./figures/FIG-CO-023-silicone-vs-tape.svg) | placeholder | HW_ENGINEER_ALTIUM_BRIEF, LITERATURE | Pilot = tape |
| FIG-CO-024 | BruxScreen intake stub | [figures/FIG-CO-024-bruxscreen-intake.svg](./figures/FIG-CO-024-bruxscreen-intake.svg) | placeholder | CLINICAL_VALIDATION, TEMPORALIS_COLLECTION_PROTOCOL | Literature tool — not Oralable UI |
| FIG-CO-025 | State hypnogram exemplar | [figures/FIG-CO-025-state-hypnogram-exemplar.png](./figures/FIG-CO-025-state-hypnogram-exemplar.png) | final | OVERNIGHT_NIGHT_REPORT, TEMPORALIS_COLLECTION_PROTOCOL, PRODUCT_ROADMAP, MOBILE_APP_FLOWS | **Very useful overnight measure** — `TEMPORALIS_20260724/02_state_hypnogram.png`; **must ship in-app adaptation** (`StateHypnogramView` / FIG-IOS-003) |

## Mermaid hubs (not FIG assets)

| Doc | Role |
|-----|------|
| [ORALABLE_SYSTEM_MAP_DIAGRAMS.md](./ORALABLE_SYSTEM_MAP_DIAGRAMS.md) | Timeline, stack, GTM, evidence protocols |
| [ORALABLE_SYSTEM_ARCHITECTURE.md](./ORALABLE_SYSTEM_ARCHITECTURE.md) | Living architecture Mermaid |
| [GEN1_GEN2_TRACKING.md](./GEN1_GEN2_TRACKING.md) · [GEN1_GEN2_MIGRATION.md](./GEN1_GEN2_MIGRATION.md) | Gen migration Mermaid |
| [data_room/JAC_CORPORATE_STRUCTURE_AND_GOVERNANCE.md](./data_room/JAC_CORPORATE_STRUCTURE_AND_GOVERNANCE.md) | Org / cap Mermaid |
| [data_room/ORALABLE_FTS_36MO.md](./data_room/ORALABLE_FTS_36MO.md) | FTS Mermaid |
| [data_room/IP_EVAL_AND_LANDSCAPE.md](./data_room/IP_EVAL_AND_LANDSCAPE.md) | IP landscape Mermaid |

## Clinical plot promotion map

| FIG ID | Current plot path | Status |
|--------|-------------------|--------|
| FIG-CO-009 | `data/plots/ed_presentation/oralable7/oralable7_from_sync1.png` | placeholder |
| FIG-CO-010 | `data/plots/ed_presentation/oralable7/oralable7_validation_dashboard.png` | placeholder |
| **FIG-CO-025** | **`plots/overnight_report/TEMPORALIS_20260724/02_state_hypnogram.png`** | **final — primary / very useful overnight measure** |

When finals exist, replace the SVG stub with the PNG (same basename preferred) and set status `final`.

*Last updated: 31 Jul 2026*
