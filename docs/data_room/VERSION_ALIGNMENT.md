# Version alignment (canonical)

**As at:** 27 Jul 2026 · Data room pack **1.1.47** · Docs hub **1.3.16**

Use this table when updating flash guides, pilot handouts, architecture, website, or TestFlight notes. Prefer these strings over older “1.0.66 ship / 1.0.67 next” language.

**Canonical development timeline:** [PRODUCT_ROADMAP.md §3](../PRODUCT_ROADMAP.md#3-timeline-calendar--canonical) · planning mirror [COST_AND_TIMELINE.md](./COST_AND_TIMELINE.md) §1.  
**Technology avenues (Gemini distill):** [GEMINI_TEMPLE_PPG_AVENUES.md](./GEMINI_TEMPLE_PPG_AVENUES.md) · roadmap §2b · landscape §4b.

## Milestone — 24 Jul 2026 (Temporalis Mac Protocol A + overnight night report)

| Item | Status |
|------|--------|
| **Stack** | FW **1.0.70** · app **4.3.3** (build **4**) · Gen1 REV10 |
| **OralableCore** | `BruxismMAM_Temporalis.mlpackage` retrained from Mac BLE Protocol A log `data/raw/TEMPORALIS_RAW_20260724_084345.txt` |
| **Capture path** | `scripts/run_protocol_a_session.py` (bleak + worn-mode write + timed cues) → `process_temporalis_gold` / `run_temporalis_mam_pipeline.py` |
| **Night report (Mac)** | `scripts/generate_overnight_night_report.py` (+ hooked from `generate_clinical_report.py`) → `plots/overnight_report/<session>/` |
| **Night report (iOS)** | Share → Clinical Temporalis PDF: bout hypnogram, smoking-gun dual rail, event CSV (`OvernightStateClassifier` / `NightReportSampleLoader`) |
| **Overnight UX direction** | [OVERNIGHT_NIGHT_REPORT.md](../OVERNIGHT_NIGHT_REPORT.md) — **BP-style bands** (TFI / SASHB/h / rescue/h / tonic min/h); **state hypnogram primary**; no sleep-score-first |
| **Core ML cohort** | [CORE_ML_TRAINING_COHORT.md](../CORE_ML_TRAINING_COHORT.md) — Tier 1 ≈ 20–30 users × 3–5 Protocol A; leave-user-out; stratify sex/age/habitus/skin |
| **Evaluable overnight** | **≥ 6 h** worn (goal **8 h**); Protocol A/B minutes are not sleep sessions |
| **Ed/Pedro kits** | **Gated** — stack ready; not yet shipped (charge-to-temple) |

## Pilot ship status (ready ≠ delivered)

| Item | As at 26 Jul 2026 |
|------|------------------|
| **Ed/Pedro kits** | **Gated / not yet shipped** |
| **Stack** | FW **1.0.70** · app **4.3.3** · Gen1 REV10 — flash/TestFlight path ready |
| **Ship gate** | Case charge to **temple-ready SOC (≥50%)** + short worn HR/SpO₂ without brownout |
| **Status sense** | STAT blink policy in **1.0.70** (do not say “chrsts broken on REV10”) |
| **Open work** | Cell energy / coupling so voltage **rises** on the Oralable case |
| **Detail** | [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) § Pilot ship status |

| Layer | Current | Notes |
|-------|---------|--------|
| **Gen1 firmware (ship)** | **1.0.70** | `oralable_nrf` `app/VERSION` · STAT blink = charging · Oralable case only · hex in `data_room/firmware/` |
| **Gen1 FW GATT string** | `1.0.70` | Read `3A0FF006` after flash |
| **iOS FirmwareGate hard min** | **1.0.63** | Blocks older research builds |
| **iOS FirmwareGate recommend** | **1.0.70** | Automatic dock / STAT policy |
| **Oralable patient app** | **4.3.3** | Marketing version; vitals phase + STAT LED mirror + Temporalis MAM refresh |
| **Oralable app build** | **4** | `CURRENT_PROJECT_VERSION` (bump on each TestFlight) |
| **Gen2 firmware (target)** | **2.0.x** | Not on Ed/Pedro kits |
| **Rollback hex (optional)** | 1.0.66 | Keep in `firmware/` only for recovery |

## Feature milestones folded into 1.0.70

| Introduced | Feature | Still true in 1.0.70 |
|------------|---------|----------------------|
| 1.0.63 | Charger = red / bench = green; probe off | Yes |
| 1.0.65 | Dim LEDs; charge_active + mode 1 | Yes (superseded on-pad by STAT) |
| 1.0.66 | +4 dBm TX; fast reconnect adv | Yes |
| 1.0.67 | `.recycled` adv; worn-disconnect FIFO; PPG DT IRQ | Yes (in tree) |
| 1.0.68 | Remapped battery gauge 3.61–4.35 V | Yes |
| **1.0.70** | **CHRSTS STAT activity** (blink / taper / undock) | **Ship** |

## Flash / app pairing

| Role | Flash | App |
|------|-------|-----|
| Ed/Pedro Phase 0 | [FIRMWARE_1.0.70_FLASH.md](./FIRMWARE_1.0.70_FLASH.md) | TestFlight **4.3.3+** |
| Quick start | [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) | Same |
| Dry run | [PILOT_DRY_RUN_CHECKLIST.md](./PILOT_DRY_RUN_CHECKLIST.md) | Archive 4.3.3 |

## Do not say (outdated)

- “Pilot ship is 1.0.66” (unless describing rollback)
- “chrsts broken on REV10”
- “Automatic placement unreliable” without “pre-1.0.70”
- “Charge on Qi / MagSafe”
- “Kits already with Ed/Pedro” / “shipping” — until ship gate clears (say **gated** / **not yet shipped**)
