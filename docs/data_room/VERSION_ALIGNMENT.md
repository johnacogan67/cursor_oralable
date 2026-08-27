# Version alignment (canonical)

**As at:** 27 Aug 2026 · Data room pack **1.1.67** · Docs hub **1.3.16**  
**Nabavi collab:** [COLLAB_NABAVI_MCGILL.md](./COLLAB_NABAVI_MCGILL.md) (Phase 1 McGill / Koroosh · Phase 2 complementary Dianyx later)  
**Research Kit:** [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) — 5 kits → Pedro by **31 Aug 2026** · status [PEDRO_STATUS_UPDATE_2026-08.md](./PEDRO_STATUS_UPDATE_2026-08.md)  
**Paper A field:** [PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md) — Research Kit feasibility n≈5; Dual A + SpO₂∩EMG nest + Pedro 1–2 h oxygen arm  
**Sensor corroboration:** [SENSOR_CORROBORATION.md](./SENSOR_CORROBORATION.md) · parity [MAC_PHONE_ALGORITHM_PARITY.md](./MAC_PHONE_ALGORITHM_PARITY.md)  
**Literature distill:** [LITERATURE_AND_PRIOR_ART.md](./LITERATURE_AND_PRIOR_ART.md) · full audit [DATA_ROOM_VALIDATION_AND_FUTURE_WORK.md](./DATA_ROOM_VALIDATION_AND_FUTURE_WORK.md)

Use this table when you update flash guides, pilot handouts, architecture, the website, or TestFlight notes. Use these strings. Do not use older “1.0.70 ship” language except as rollback.

**Canonical development timeline:** [PRODUCT_ROADMAP.md §3](../PRODUCT_ROADMAP.md#3-timeline-calendar--canonical) · planning mirror [COST_AND_TIMELINE.md](./COST_AND_TIMELINE.md) §1. · **App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)  
**Technology avenues (Gemini distill):** [GEMINI_TEMPLE_PPG_AVENUES.md](./GEMINI_TEMPLE_PPG_AVENUES.md) · roadmap §2b · landscape §4b.

## Milestone — 24 Jul 2026 (Temporalis Mac Protocol A + overnight night report)

| Item | Status |
|------|--------|
| **Stack** | FW **1.0.70** (then-current) · app **4.3.3** (build **4**) · Gen1 REV10 |
| **OralableCore** | `BruxismMAM_Temporalis.mlpackage` retrained from Mac BLE Protocol A log `data/raw/TEMPORALIS_RAW_20260724_084345.txt` |
| **Capture path** | `scripts/run_protocol_a_session.py` (bleak + worn-mode write + timed cues) → `process_temporalis_gold` / `run_temporalis_mam_pipeline.py` |
| **Night report (Mac)** | `scripts/generate_overnight_night_report.py` (+ hooked from `generate_clinical_report.py`) → `plots/overnight_report/<session>/` |
| **Night report (iOS)** | Share → Clinical Temporalis PDF: bout hypnogram, smoking-gun dual rail, event CSV (`OvernightStateClassifier` / `NightReportSampleLoader`) |
| **Overnight UX direction** | [OVERNIGHT_NIGHT_REPORT.md](../OVERNIGHT_NIGHT_REPORT.md) — **BP-style bands**; **state hypnogram = very useful primary** ([FIG-CO-025](../figures/FIG-CO-025-state-hypnogram-exemplar.png)); **in-app** `StateHypnogramView` + morning card (flag `showOvernightHypnogram`); PDF for full pack; no sleep-score-first. **FIG-CO-025 pack wear ≈ 6 min** (layout only — not a ≥6 h overnight); see [DATA_ROOM_VALIDATION_AND_FUTURE_WORK.md](./DATA_ROOM_VALIDATION_AND_FUTURE_WORK.md) |
| **Core ML cohort** | [CORE_ML_TRAINING_COHORT.md](../CORE_ML_TRAINING_COHORT.md) — Tier 1 ≈ 20–30 users × 3–5 Protocol A; leave-user-out; stratify sex/age/habitus/skin |
| **Evaluable overnight** | **Ideal / Paper A Arm E/J:** **≥ 6 h** worn (goal **8 h**). **iOS band unlock:** **≥ 1 h** (`evaluableWearSeconds`). Protocol A/B minutes are not sleep sessions |
| **Ed/Pedro Research Kits** | **Gated** — stack ready; not yet shipped (charge-to-temple); target **5 by 31 Aug 2026** |

## Milestone — 8 Aug 2026 (Dual A SpO₂∩EMG nest + iOS Dual A slice 1)

| Item | Status |
|------|--------|
| **Mac concordance** | `align_anr_oralable_concordance.py` + `src/analysis/emg_spo2_nest.py` → `NEST.md` / `spo2_emg_nest` (AcuPebble-style burden nest; **not** AHI/ODI) |
| **iOS Dual Protocol A** | Patient app `showDualProtocolA` (default **OFF**) — Developer Settings opt-in; cues + EMG preflight + Share pack; Mac still primary until TestFlight pack aligns |
| **Claim discipline** | Nest ≠ Bruxoff/GrindCare equivalence; AcuPebble remains Pedro AHI reference — [ACUPEBBLE_VS_ORALABLE_ANR.md](./ACUPEBBLE_VS_ORALABLE_ANR.md) · [ANR_M40_CONCORDANCE.md](./ANR_M40_CONCORDANCE.md) |

## Milestone — 10 Aug 2026 (soft corroboration + iOS band unlock)

| Item | Status |
|------|--------|
| **Soft ACC + skin temp** | `SensorCorroboration` (32–38 °C) derates live quality / `isWorn`; overnight off-skin → quiet, no wear/SASHB — [SENSOR_CORROBORATION.md](./SENSOR_CORROBORATION.md) |
| **iOS overnight bands** | Unlock at **≥1 h** worn; ideal overnight / cohort recalibration still **≥6 h** |
| **Dual A default** | Still **OFF**; enable in Developer Settings for research Dual A (~6 min). Sleep is the normal path |
| **DUAL_PAIR meta** | Optional `skin_temp_mean_c` / `on_skin_fraction` when Oralable temp streams |

## Milestone — 12 Aug 2026 (measured Dual A + research EDF+)

| Item | Status |
|------|--------|
| **Mac Dual A session** | `TEMPORALIS_RAW_20260812_085110` + `ANR_EMG_20260812_085110` + `DUAL_PAIR_20260812_085110` — EMG preflight gate **70** (clench max **83**); Protocol A ~6 min complete |
| **Concordance pack** | `plots/concordance/20260812_085110/` — overlay, `NEST.md`, **`session.edf` (EMG inside)**, median EMG→IR-DC lag ≈ **4.9 s** |
| **SpO₂ / SASHB** | SpO₂ computed on Mac suite (aligned mean ≈ **89.5%**); engineering SASHB ≈ **929 %·s** (SpO₂&lt;90 AUC — **not** Azarbarzin HB; **not** AHI) |
| **Hypnogram layout** | Overnight toolchain on Dual A gold → `plots/overnight_report/TEMPORALIS_20260812_085110_dualA/02_state_hypnogram.png` — wear ≈ **6.0 min** (layout only) |
| **EDF+ (Mac + iOS)** | Align writes `session.edf` by default; iOS Dual A Share includes EDF with ANR EMG; Oralable-only EDF in Developer Settings — [ANR_M40_CONCORDANCE.md](../ANR_M40_CONCORDANCE.md) |
| **Claim discipline** | Hourly stacked burden ≠ hypoxic burden (SASHB line only); Dual A F1 vs labels this pack was **0** — QC / placement follow-up, not partner *N* |

## Milestone — 27 Aug 2026 (FW 1.0.82 ship)

| Item | Status |
|------|--------|
| **Stack** | FW **1.0.82** · app **4.3.3** · Gen1 REV10 |
| **OTA** | [FIRMWARE_1.0.82_FLASH.md](./FIRMWARE_1.0.82_FLASH.md) — Device Manager zip + signed bin + SWD merged.hex |
| **Sensors** | PPG/ACC on BLE + CCC; off when disconnected; off below 5% / 3.61 V with MCU up |
| **Worn** | Automatic = IR pulse, not die temperature. Mode 3 still forces worn. |
| **nRF Connect** | Confirm `3A0FF006` = `1.0.82` after OTA; temple IR-pulse latch is a field check |

## Pilot ship status (ready ≠ delivered)

| Item | As at 27 Aug 2026 |
|------|------------------|
| **Ed/Pedro Research Kits** | **Gated / not yet shipped** — target **5 kits to Pedro by 31 Aug 2026** |
| **Kit definition** | [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) · photos [RESEARCH_KIT_PHOTO_SELECTION.md](./RESEARCH_KIT_PHOTO_SELECTION.md) |
| **Stack** | FW **1.0.82** · app **4.3.3** · Gen1 REV10 — flash/OTA/TestFlight path ready |
| **Ship gate** | Case charge to **temple-ready SOC (≥50%)** + short worn HR/SpO₂ without brownout **on each unit** |
| **Status sense** | STAT blink policy from **1.0.70**; **1.0.82** IR-pulse worn + sense-on-BLE (do not say “chrsts broken on REV10”) |
| **Open work** | Cell energy / coupling so voltage **rises** on the Oralable case |
| **Detail** | [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) § Pilot ship status · dry-run §G [PILOT_DRY_RUN_CHECKLIST.md](./PILOT_DRY_RUN_CHECKLIST.md) |

| Layer | Current | Notes |
|-------|---------|--------|
| **Gen1 firmware (ship)** | **1.0.82** | `oralable_nrf` `app/VERSION` · sense-on-BLE · 5% floor · IR-pulse worn · STAT blink · Oralable case only · hex in `data_room/firmware/` |
| **Gen1 FW GATT string** | `1.0.82` | Read `3A0FF006` after flash / OTA |
| **iOS FirmwareGate hard min** | **1.0.63** | Blocks older research builds |
| **iOS FirmwareGate recommend** | **1.0.82** | Sense-on-BLE, green pad LEDs, IR-pulse worn |
| **Oralable patient app** | **4.3.3** | Marketing version; vitals phase + STAT LED mirror + Temporalis MAM refresh |
| **Oralable app build** | **4** | `CURRENT_PROJECT_VERSION` (bump on each TestFlight) |
| **Gen2 firmware (target)** | **2.0.x** | Not on Ed/Pedro kits |
| **Rollback hex (optional)** | 1.0.70 | Keep in `firmware/` · older 1.0.66 also there |

## Feature milestones folded into 1.0.82

| Introduced | Feature | Still true in 1.0.82 |
|------------|---------|----------------------|
| 1.0.63 | Probe off | Yes |
| 1.0.66 | +4 dBm TX; fast reconnect adv | Yes |
| 1.0.67 | `.recycled` adv; PPG DT IRQ | Yes |
| 1.0.68 | Remapped battery gauge 3.61–4.35 V | Yes |
| 1.0.70 | CHRSTS STAT activity (blink / taper / undock) | Yes |
| 1.0.72 | Status LEDs **green-only** (never red) | Yes |
| 1.0.80 | PPG/ACC follow BLE + CCC, not worn | Yes |
| 1.0.81 | Below 5% / 3.61 V: sensors off, MCU stays up | Yes |
| **1.0.82** | **Automatic worn = IR pulse** (not die temp) | **Ship** |

**Status LEDs (1.0.82):** green-only. On pad: flash green while charging; **solid green** at STAT taper **or** already ≥ ~70% / 4.05 V. Off pad, no BLE: dark. Red/IR is PPG sensing only while streaming. Do not tell operators to expect red on the pad.

## Flash / app pairing

| Role | Flash | App |
|------|-------|-----|
| Ed/Pedro Phase 0 | [FIRMWARE_1.0.82_FLASH.md](./FIRMWARE_1.0.82_FLASH.md) | TestFlight **4.3.3+** |
| Quick start | [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) | Same |
| Dry run | [PILOT_DRY_RUN_CHECKLIST.md](./PILOT_DRY_RUN_CHECKLIST.md) | Archive 4.3.3 |

## Do not say (outdated)

- “Pilot ship is 1.0.70” (unless describing rollback)
- “chrsts broken on REV10”
- “Automatic placement unreliable” without “pre-1.0.70”
- “Worn follows die temperature” (true only before 1.0.82)
- “Charge on Qi / MagSafe”
- “Kits already with Ed/Pedro” / “shipping” — until ship gate clears (say **gated** / **not yet shipped**)
