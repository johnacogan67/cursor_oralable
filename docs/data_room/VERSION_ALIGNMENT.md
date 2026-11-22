# Version alignment (canonical)

**As at:** 22 Nov 2026 · Data room pack **1.1.75** · Docs hub **1.3.21** · Protocol A seat QC = **`couple`** (Mac single + iOS Setup / Dual)  
**Nabavi collab:** [COLLAB_NABAVI_MCGILL.md](./clinical/COLLAB_NABAVI_MCGILL.md) (Phase 1 McGill / Koroosh · Phase 2 complementary Dianyx later)  
**Research Kit:** [ORALABLE_RESEARCH_KIT.md](./clinical/ORALABLE_RESEARCH_KIT.md) — 5 **Oralable-only** kits · **30 Sep 2026** handoff **missed** · still **gated** · no new date · meeting [clinical/ED_PEDRO_MEETING_2026-09-07.md](./clinical/ED_PEDRO_MEETING_2026-09-07.md) · sendable [clinical/PEDRO_STATUS_UPDATE_2026-09.md](./clinical/PEDRO_STATUS_UPDATE_2026-09.md)  
**Paper A field:** [PAPER_A_FEASIBILITY_PROTOCOL.md](./clinical/PAPER_A_FEASIBILITY_PROTOCOL.md) — n≈5 clinic loaner · ≥4 h nights · Dual A **not** a patient arm  
**Sensor corroboration:** [SENSOR_CORROBORATION.md](./bookmarks/SENSOR_CORROBORATION.md) · parity [MAC_PHONE_ALGORITHM_PARITY.md](./bookmarks/MAC_PHONE_ALGORITHM_PARITY.md)  
**Construct map:** [MEASUREMENT_CONSTRUCT_MAP.md](./clinical/MEASUREMENT_CONSTRUCT_MAP.md) — MAM vs ANR vs Dual A vs AcuPebble vs PSG (iterate there)  
**Literature distill:** [LITERATURE_AND_PRIOR_ART.md](./bookmarks/LITERATURE_AND_PRIOR_ART.md) · full audit [DATA_ROOM_VALIDATION_AND_FUTURE_WORK.md](./governance/DATA_ROOM_VALIDATION_AND_FUTURE_WORK.md)  
**Attachment (housing / skin):** [ATTACHMENT.md](../ATTACHMENT.md) · data-room [attachment/README.md](./attachment/README.md)

Use this table when you update flash guides, pilot handouts, architecture, the website, or TestFlight notes. Use these strings. Do not use older “1.0.70 ship” language except as rollback. Do not call **1.0.85** the current kit. That string is the frozen Gen 1 archive.

**Current stack (use these strings):** Gen 2 · BOM **REV9** · PCB **REV11** · Kaga **ES4L15BA1** (nRF54L15) · cell **LP270829 35 mAh** · sensors **MAXM86161** + **LIS2DTW12**. The reviewed image is **2.0.6-gen2-nrfconnect**. G2-P2, G2-P3, G2-P4, and G2-P6 were reviewed on 15–16 Nov 2026. Source is signed **2.0.8+0** and `prj.conf` broadcasts **MAM**. That sign version is not a claim that a board is running 2.0.8. No read of characteristic 006 in this room shows 2.0.7 or 2.0.8 on a module. Images through 2.0.7 were built to advertise **Oralable**. The phone accepts both names. `app/VERSION` is **1.0.87** and is not the Gen 2 string. App **4.3.3** (build **5**), shown as MAM Diagnostics. Kits **gated**. Gen 1 (REV10 / ES2832AA2 / FW **1.0.85**) is legacy. Do not flash it on Gen 2. The `pcb00003` tree stays buildable as a bench archive. The 20-unit PC53020 lot is a separate Kaga tracking thread. iOS FirmwareGate stays min **1.0.63** / recommend **1.0.85**. The iOS app is not production-compatible with Gen 2. Earlier bench notes, including 30 Oct and 2 Nov, stay as the record of those days.

**Canonical development timeline:** [PRODUCT_ROADMAP.md §3](../PRODUCT_ROADMAP.md#3-timeline-calendar--canonical) · planning mirror [COST_AND_TIMELINE.md](./governance/COST_AND_TIMELINE.md) §1. · **App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)  
**Technology avenues (Gemini distill):** [GEMINI_TEMPLE_PPG_AVENUES.md](./bookmarks/GEMINI_TEMPLE_PPG_AVENUES.md) · roadmap §2b · landscape §4b.

## Milestone — 24 Jul 2026 (Temporalis Mac Protocol A + overnight night report)

| Item | Status |
|------|--------|
| **Stack** | FW **1.0.70** (then-current) · app **4.3.3** (build **4**) · Gen1 REV10 |
| **OralableCore** | `BruxismMAM_Temporalis.mlpackage` retrained from Mac BLE Protocol A log `data/raw/TEMPORALIS_RAW_20260724_084345.txt` |
| **Capture path** | `scripts/run_protocol_a_session.py` (bleak + mode-3 + **couple** seat QC + timed cues) → `process_temporalis_gold` / `run_temporalis_mam_pipeline.py` — [TEMPORALIS_COLLECTION_PROTOCOL.md](../TEMPORALIS_COLLECTION_PROTOCOL.md) |
| **Night report (Mac)** | `scripts/generate_overnight_night_report.py` (+ hooked from `generate_clinical_report.py`) → `plots/overnight_report/<session>/` |
| **Night report (iOS)** | Share → Clinical Temporalis PDF: bout hypnogram, smoking-gun dual rail, event CSV (`OvernightStateClassifier` / `NightReportSampleLoader`) |
| **Overnight UX direction** | [OVERNIGHT_NIGHT_REPORT.md](../OVERNIGHT_NIGHT_REPORT.md) — **BP-style bands**; **state hypnogram = very useful primary** ([FIG-CO-025](../figures/FIG-CO-025-state-hypnogram-exemplar.png)); **in-app** `StateHypnogramView` + morning card (flag `showOvernightHypnogram`); PDF for full pack; no sleep-score-first. **FIG-CO-025 pack wear ≈ 6 min** (layout only — not a ≥6 h overnight); see [DATA_ROOM_VALIDATION_AND_FUTURE_WORK.md](./governance/DATA_ROOM_VALIDATION_AND_FUTURE_WORK.md) |
| **Core ML cohort** | [CORE_ML_TRAINING_COHORT.md](../CORE_ML_TRAINING_COHORT.md) — Tier 1 ≈ 20–30 users × 3–5 Protocol A; leave-user-out; stratify sex/age/habitus/skin |
| **Evaluable overnight** | **Beacon feasibility:** **≥ 4 h** worn (6 h better). **Hypnogram figure / band recalibration:** **≥ 6 h** (goal **8 h**). **iOS band unlock:** **≥ 1 h**. Protocol A/B minutes are not sleep sessions |
| **Ed/Pedro Research Kits** | **Gated** — Oralable-only (no ANR, no headband). **30 Sep 2026** handoff **missed**. No new date. |

## Milestone — 8 Aug 2026 (Dual A SpO₂∩EMG nest + iOS Dual A slice 1)

| Item | Status |
|------|--------|
| **Mac concordance** | `align_anr_oralable_concordance.py` + `src/analysis/emg_spo2_nest.py` → `NEST.md` / `spo2_emg_nest` (AcuPebble-style burden nest; **not** AHI/ODI) |
| **iOS Dual Protocol A** | Patient app `showDualProtocolA` (default **OFF**) — **lab/TestFlight**. Beacon n≈5 does **not** combine ANR + MAM. Mac Dual still methods precursor |
| **Claim discipline** | Nest ≠ Bruxoff/GrindCare equivalence; AcuPebble remains Pedro AHI reference — [ACUPEBBLE_VS_ORALABLE_ANR.md](./bookmarks/ACUPEBBLE_VS_ORALABLE_ANR.md) · [ANR_M40_CONCORDANCE.md](./bookmarks/ANR_M40_CONCORDANCE.md) |

## Milestone — 10 Aug 2026 (soft corroboration + iOS band unlock)

| Item | Status |
|------|--------|
| **Soft ACC + skin temp** | `SensorCorroboration` (32–38 °C) derates live quality / `isWorn`; overnight off-skin → quiet, no wear/SASHB — [SENSOR_CORROBORATION.md](./bookmarks/SENSOR_CORROBORATION.md) |
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
| **OTA** | [FIRMWARE_1.0.82_FLASH.md](./firmware/FIRMWARE_1.0.82_FLASH.md) — Device Manager zip + signed bin + SWD merged.hex |
| **Sensors** | PPG/ACC on BLE + CCC; off when disconnected; off below 5% / 3.61 V with MCU up |
| **Worn** | Automatic = IR pulse, not die temperature. Mode 3 still forces worn. |
| **nRF Connect** | Confirm `3A0FF006` = `1.0.82` after OTA; temple IR-pulse latch is a field check |

## Milestone — 30 Aug 2026 (FW 1.0.84 Gen1 target)

| Item | Status |
|------|--------|
| **Stack** | FW **1.0.84** · app **4.3.3** (build **5**) · Gen1 REV10 |
| **Build** | `oralable_nrf` `app/VERSION` PATCHLEVEL **84** · archived hex/zip in `data_room/firmware/` · [FIRMWARE_1.0.84_FLASH.md](./firmware/FIRMWARE_1.0.84_FLASH.md) |
| **Recover** | Pad/zombie: idle PPG notify ~4 s → green + advertise; desk/bench: `worn=0` + PPG sensing + ACC flat **10 min** → drop + re-advertise |
| **Mac Protocol A** | FwLog CCC **off** by default (`--fw-log` opt-in) · **`couple`** seat QC (`CouplingGate`; `--allow-couple-1` / `--no-coupling-gate`) |
| **iOS** | Hard min still **1.0.63**; recommend **1.0.84** (`FirmwareGate`) — no new GATT · Protocol A Setup = **couple** (default ON) · Dual A = EMG → **couple** · TestFlight build **5** |
| **OralableCore** | `CouplingGate` — Mac/phone parity for Protocol A seat QC (GATT `worn` stays separate) |
| **nRF Connect** | Confirm `3A0FF006` = `1.0.84` after flash · CCC order battery → status → PPG → ACC → temp |
| **Kits** | Still **gated / not yet shipped** — charge-to-temple unchanged · Beacon box = Oralable-only · **30 Sep 2026** handoff **missed** · no new date |
| **Seat QC lock (1 Sep)** | **Mac single** + **iOS Setup** + **iOS Dual (MAM)** = `couple`. **Mac Dual** still EMG ≥70 + **IR drop ≥8%** until `run_dual_protocol_a_session.py` ports couple. |
| **Beacon use (7 Sep)** | One night · clinic charge · no headband · pad/tape · no patient ANR · ≥4 h (6 better) — [ED_PEDRO_MEETING_2026-09-07.md](./clinical/ED_PEDRO_MEETING_2026-09-07.md) |
| **Living pack (12 Sep)** | Pack **1.1.70**. Clinical lock from 7 Sep unchanged. GTM Path **0** = clinic loaner. ANR is **lab Dual A**. **UP filing receipt** still **GAP**. August Pedro status **archived**. Kits still **gated**. |
| **Living pack (15 Sep)** | Pack **1.1.71** · docs hub **1.3.19**. Topic `attachment` (ANR-class Formlabs shell). Clinical lock and kit-ship **unchanged**. |
| **GTM Path 0 (19 Sep)** | n≈5 nights **unpaid**. Later RKaaS = company-owned kits + draft **€100**/processed report. Not Point B. [GTM_ONE_PAGE.md](./governance/GTM_ONE_PAGE.md) **1.2.2**. |
| **Living pack (19 Sep)** | Pack **1.1.72** · docs hub **1.3.20**. Living docs aligned to FW **1.0.85**. Frozen 7 Sep sendables unchanged. Clinical lock and kit-ship **unchanged**. |
| **Governance (19 Sep)** | Pack **1.1.73**. **Ken Kinsella is not a CEO candidate.** Point B / BalancePoints advisor path unchanged. CEO search = other candidates via Nigel. |
| **Living pack (3 Oct)** | Pack **1.1.74** · docs hub **1.3.21**. **30 Sep 2026** kit handoff **missed**. Kits still **gated**. No new date. Kit FW stays **1.0.85**. Local `app/VERSION` PATCHLEVEL **86** is **not** the kit target. |
| **Bench (5 Oct)** | One MAM left MCUboot. Phone showed **Oralable**. Green on the case. Archived 1.0.85 merged hex does **not** boot (119 bytes over the swap-move sector). Running image is the local size-opt build (**1.0.86**). GATT not read. Kits still **gated**. Pack stays **1.1.74**. |
| **Cutover (14 Oct)** | Gen 2 charger + MAM + **LP270829** cell in hand. Partners, Paper A, and the patient site name Gen 2. Firmware still a scaffold. Kits still **gated**. Gen 1 frozen. Pack stays **1.1.74**. |
| **Living pack (22 Nov)** | Pack **1.1.75**. Docs hub stays **1.3.21**. Reviewed image **2.0.6-gen2-nrfconnect**. Source signs **2.0.8+0** and broadcasts **MAM**. Board image after 2.0.6 is unconfirmed. `app/VERSION` **1.0.87** is not the Gen 2 string. App **4.3.3** (build **5**). Kits still **gated**. |

## Milestone — 14 Oct 2026 (Gen 2 is the partner hardware)

| Item | Status |
|------|--------|
| **Hardware in hand** | TGM charger (USB-C, coil) and small MAM, plus **LP270829 35 mAh**. Photos: [attachment_photo_source](../../figures/attachment_photo_source/). Bare boards and cell, not a worn housing. |
| **Identity** | Gen 2 · BOM **REV9** · PCB **REV11** · **ES4L15BA1** · fab job NCAB **PC53020** |
| **Firmware** | **2.0.x** target, as at this date. Not flashed yet. Board, config, and app are a **port of Gen 1** (`pcb00003`, same GATT, same `prj.conf` behavior). Later bench is the **30 Oct** milestone. |
| **Gen 1** | Frozen. REV10 / **1.0.85** stays a bench archive. No new 1.0.x work. No **1.0.86** kit target. |
| **Partners** | Paper A, Ed, Pedro, Sophie Warren, and the patient site describe this Gen 2 kit. Kits still **gated**. |
| **PC53020 lot** | The 20-unit re-route thread is separate. Do not mark that lot received from these two boards. |

## Milestone — 30 Oct 2026 (Gen 2 bench: advertises, sensors do not)

| Item | Status |
|------|--------|
| **Detail** | [GEN2_WOUT_KAGA_STATUS_2026-10-30.md](./hardware/GEN2_WOUT_KAGA_STATUS_2026-10-30.md) |
| **Wout** | 25 Oct: REV10 schematic is the REV11 design. SDA **P1.07**, SCL **P1.03**, SENS_EN **P1.08**, 10 kΩ pull-ups on the SENS_EN net. Open questions on U2 / U4 fit and the TPS79733 enable are written and **not sent**. |
| **Kaga** | Module **ES4L15BA1** answers on SWD. One blank board programmed **29 Oct** (`oralable_gen2_ota_merged.hex`, 299008 bytes, verify OK) from the nRF54L15 DK, J-Link **1057765387**. The 20-unit **PC53020** lot is still separate. Not a return-to-Kaga sensor fault unless Wout says the module pins are wrong. |
| **On air** | A board that was already running listed **Oralable** in nRF Connect and returned a firmware log (cell **3784 mV**). No PPG or accelerometer samples. Both chips NACKed. That board was not overwritten. |
| **Image** | Header on the already-running board read **1.0.87**. Not the **2.0.x** kit string. Pin scan still in that image. |
| **Kits** | Still **gated**. Not a ship. |

## Milestone — 2 Nov 2026 (Gen 2 streams until the cell floor)

| Item | Status |
|------|--------|
| **Detail** | [GEN2_BENCH_LEARNINGS_2026-11-02.md](./hardware/GEN2_BENCH_LEARNINGS_2026-11-02.md) |
| **On air** | One board, image **1.0.87**, streamed PPG (frames 0–1116) and accelerometer until the cell fell from 3410 mV to 3322 mV with the charger pin on. Link stayed up. Green PA 5 only. |
| **Candidate** | `merged.hex` rebuilt 3 Nov 04:30, SHA-256 `52a729e4ec436604f471613579f0961a806c97b776ac43f94e6b2af9e06acc80`. ELF string `2.0.0-gen2-nrfconnect`. Not programmed. GATT `006` not read. `app/VERSION` stays **1.0.87**. |
| **Gates** | G2-P0 through G2-P6 still **pending**. Next measurement is a charged cell (Stage B), not another pin scan. |
| **Kits** | Still **gated**. No new date. |

## Milestone — 4 Nov 2026 (scoped firmware implementation; result blocked)
## Milestone — 4 Nov 2026 (TGM GATT 50 Hz hardening + dual Gen1/Gen2 pristine builds)

Scoped firmware hardening in `oralable_nrf` locked the 13-characteristic TGM GATT wire serializers (`4 + 12*N` PPG, `4 + 6*N` ACC, 8-byte temperature), bounded FIFO bursts (180 B PPG / 150 B ACC), ATT MTU stream gating (`>=247` PPG, `>=157` ACC), nominal 50 Hz `MAXM86161` `0x12 = 0x08` with readback, and `LIS2DTW12` `CTRL1 = 0x43` (50 Hz low-power mode 4) with readback.

After guarding `nrf_nfct_pad_config_*` in `oralable_nrf/app/src/main.c` under `#if defined(NRF_NFCT_HAS_PAD_CONFIG_REG) && NRF_NFCT_HAS_PAD_CONFIG_REG` and updating `oralable_nrf/app/CMakeLists.txt` so any `gen2_` build directory or NCS v2.9 workspace avoids pinning NCS v2.7 modules, both pristine builds succeeded:

- **Gen2 pristine candidate (`pcb00003_gen2/nrf54l15/cpuapp`, NCS v2.9.3):** `oralable_nrf/build_gen2_nfct_compat/merged.hex` (exit 0).
- **Gen1 pristine archive (`pcb00003/nrf52832`, NCS v2.7.0):** `oralable_nrf/build_pcb00003_nfct_compat/merged.hex` (`sha256=1c7d5995fcd7106cf7413d7174e5c001c284c73693b7db12af18f5310012f37a`) and `dfu_application.zip` (exit 0).
- **Static & host checks:** `python3 scripts/verify_tgm_contract.py --strict` and the Python regression suite (`pytest`, including `A2` golden wire vectors and 50 Hz / FIFO header checks) pass.

`TGM_GATT_CONTRACT.md` now records `LIS2DTW12 CTRL1 = 0x43` as canonical for 50 Hz LP mode 4 (versus `0x44` high-performance mode) and documents that `3A0FF00C` v2 `battery_interval_s` (`u8`) remains a 1-byte field in v2 while default ADC cadence is 300 s. Physical device validation (`G2-P0` through `G2-P6`, connected `nRF Connect` 60-second 50 Hz sample count, and REV11 rail/GPIO checks) and release approval remain **gated**. No board was flashed and no kit-shipping or version targets changed.

## Milestone — 14 Sep 2026 (FW 1.0.85 hold LED)

| Item | Status |
|------|--------|
| **Stack** | FW **1.0.85** · app **4.3.3** (build **5**) · Gen1 REV10 |
| **Build** | `oralable_nrf` `app/VERSION` PATCHLEVEL **85** · hex/zip in `data_room/firmware/` · [FIRMWARE_1.0.85_FLASH.md](./firmware/FIRMWARE_1.0.85_FLASH.md) |
| **Hold LED** | Flash below ~4.20 V / 80%; rare green pulse (~200 ms / 10 s) in hold. STAT taper alone is not full. |
| **OTA** | Nordic Device Manager zip `oralable_1.0.85_pcb00003_dfu_application.zip` from 1.0.70+ MCUboot |
| **GATT** | Unchanged. Read `3A0FF006` = `1.0.85` after OTA |
| **iOS** | Hard min still **1.0.63**; recommend **1.0.85** |
| **Kits** | Still **gated / not yet shipped** · **30 Sep 2026** handoff **missed** · no new date |

## Milestone — 5 Oct 2026 (MAM application boots)

| Item | Status |
|------|--------|
| **Bench** | One pcb00003 MAM. nRF5340-DK debug out, J-Link S/N **1050090445**, Tag-Connect. The MAM runs on the charge case or its own cell. The DK does not power it. |
| **Seen** | Phone nRF Connect lists **Oralable**. Green flash on the charge case. John confirmed, 5 Oct 2026. |
| **Image that ran** | `oralable_nrf/build_pcb00003_fit/merged.hex` · `app/slot_fit.conf` (size optimization). Signed length **0x31e03**. Tree `app/VERSION` is **1.0.86**. GATT `3A0FF006` was **not** read. **Not** the kit target. |
| **14 Sep hex** | `oralable_1.0.85_pcb00003_merged.hex` stays in MCUboot. Signed image **0x39077**. Swap-move limit **0x39000** (last 4 KB sector). Over by **119** bytes. No advert and no green until the application runs. |
| **Flash rule** | Program the fitted hex. Do **not** chip-erase. UICR.APPROTECT stayed **0x5A**. A chip erase on this build code locks the CPU. |
| **Kits** | Still **gated**. One bench board. Not a kit ship. |

## Milestone — 22 Nov 2026 (reviewed 2.0.6; source signs 2.0.8)

| Item | Status |
|------|--------|
| **Reviewed image** | **2.0.6-gen2-nrfconnect**. G2-P2, G2-P3, G2-P4, and G2-P6 reviewed 15–16 Nov 2026. |
| **Source** | Sign version **2.0.8+0**. Broadcast name **MAM**. Not confirmed on a board. A 2.0.7 file was built. No 006 read here shows 2.0.7 or 2.0.8 on a module. |
| **`app/VERSION`** | **1.0.87**. Gen 1 file. Not the Gen 2 GATT string. |
| **App** | **4.3.3** (build **5**). Home screen MAM Diagnostics. Not production-compatible with Gen 2. |
| **Kits** | **Gated**. The 30 Sep 2026 handoff missed. No new date. |
| **Pack** | **1.1.75**. Docs hub stays **1.3.21**. |

## Pilot ship status (ready ≠ delivered)

The firmware lines in this table are the **14 Oct 2026** record. The 22 Nov milestone above is the current firmware status.

| Item | As at 14 Oct 2026 |
|------|------------------|
| **Ed/Pedro Research Kits** | **Gated / not yet shipped** — **30 Sep 2026** handoff **missed**. Five Oralable-only kits. No new date. Hardware is Gen 2. A bench image is on one blank board (29 Oct). It does not stream PPG or ACC. Not a kit version. |
| **Kit definition** | [ORALABLE_RESEARCH_KIT.md](./clinical/ORALABLE_RESEARCH_KIT.md) · no ANR · no headband · pads/tape |
| **Stack** | Gen 2 REV11 · **ES4L15BA1** · **LP270829** · FW **2.0.x** target (not a kit version; bench image on one board 29 Oct) · app **4.3.3** (build **5**) |
| **Ship gate** | Gen 2 advertises and streams, then case charge to **temple-ready SOC (≥50%)** + short worn HR/SpO₂ without brownout **on each unit** |
| **Use** | One night · clinic recharge · Ed packaging · Pedro Arm P |
| **Open work** | PPG and ACC do not answer. Questions for Wout not sent. Pin scan still in the running image. Printed shell [ATTACHMENT.md](../ATTACHMENT.md). Status [GEN2_WOUT_KAGA_STATUS_2026-10-30.md](./hardware/GEN2_WOUT_KAGA_STATUS_2026-10-30.md) |
| **Detail** | [ED_PEDRO_QUICK_START.md](./clinical/ED_PEDRO_QUICK_START.md) · dry-run §G [PILOT_DRY_RUN_CHECKLIST.md](./clinical/PILOT_DRY_RUN_CHECKLIST.md) · meeting [ED_PEDRO_MEETING_2026-09-07.md](./clinical/ED_PEDRO_MEETING_2026-09-07.md) (7 Sep record; current kit is Gen 2) |

| Layer | Current | Notes |
|-------|---------|--------|
| **Kit hardware** | Gen 2 REV11 · **ES4L15BA1** · **LP270829 35 mAh** | In hand 14 Oct 2026. Bring-up, not ship-ready. |
| **Gen2 firmware (reviewed)** | **2.0.6-gen2-nrfconnect** | G2-P2, G2-P3, G2-P4, G2-P6 reviewed 15–16 Nov. Not a kit release. |
| **Gen2 firmware (source)** | **2.0.8+0**, name **MAM** | Not confirmed on a board. Images through 2.0.7 advertised **Oralable**. |
| **Gen1 firmware (frozen)** | **1.0.85** | Bench archive. No new 1.0.x. `app/VERSION` **1.0.87** is not the Gen 2 string. |
| **iOS FirmwareGate hard min** | **1.0.63** | Gen 1 floor. The reviewed Gen 2 image is 2.0.6. The app is not production-compatible with Gen 2. |
| **iOS FirmwareGate recommend** | **1.0.85** | Soft hint for a Gen 1 bench board. |
| **Oralable patient app** | **4.3.3** | Home screen MAM Diagnostics. Protocol A Setup (**couple**) · Dual A EMG→couple |
| **Oralable app build** | **5** | `CURRENT_PROJECT_VERSION` (bump on each TestFlight) |
| **Prior ship / rollback** | **1.0.84** (N−1 in `data_room/firmware/`) · **1.0.82** also kept · **1.0.70** (deep recovery in `oralable_nrf/artifacts/` only) | See [FIRMWARE_ROLLBACK.md](./firmware/FIRMWARE_ROLLBACK.md) |

## Feature milestones folded into 1.0.85

| Introduced | Feature | Still true in 1.0.85 |
|------------|---------|----------------------|
| 1.0.63 | Probe off | Yes |
| 1.0.66 | +4 dBm TX; fast reconnect adv | Yes |
| 1.0.67 | `.recycled` adv; PPG DT IRQ | Yes |
| 1.0.68 | Remapped battery gauge 3.61–4.35 V | Yes |
| 1.0.70 | CHRSTS STAT activity (blink / taper / undock) | Yes |
| 1.0.72 | Status LEDs **green-only** (never red) | Yes |
| 1.0.80 | PPG/ACC follow BLE + CCC, not worn | Yes |
| 1.0.81 | Below 5% / 3.61 V: sensors off, MCU stays up | Yes |
| 1.0.82 | Automatic worn = IR pulse (not die temp) | Yes |
| **1.0.84** | **Pad/zombie recover + desk/bench abandon** | Yes |
| **1.0.85** | **Hold LED: rare pulse at ≥4.20 V / 80%** | Frozen Gen 1 archive. Not the current kit. |

**Status LEDs (1.0.85):** green-only. On pad, no BLE: **flash green** while below hold (~4.20 V / 80%, including early STAT taper); **rare green pulse** (~200 ms every 10 s) in hold. STAT taper alone is not full. Off pad, no BLE: dark. Red/IR is PPG sensing only while streaming. Archived **1.0.84** hex still uses continuous solid green at taper / ~70% / 4.05 V. Do not tell operators to expect red on the pad.

## Flash / app pairing

| Role | Flash | App |
|------|-------|-----|
| Ed/Pedro Phase 0 | Gen 2 **2.0.x** when it advertises. Frozen Gen 1 archive: [FIRMWARE_1.0.85_FLASH.md](./firmware/FIRMWARE_1.0.85_FLASH.md) | TestFlight **4.3.3** build **5+** |
| Quick start | [ED_PEDRO_QUICK_START.md](./clinical/ED_PEDRO_QUICK_START.md) | Same |
| Dry run | [PILOT_DRY_RUN_CHECKLIST.md](./clinical/PILOT_DRY_RUN_CHECKLIST.md) | Archive **4.3.3 (5)** |

## Do not say (outdated)

- “Pilot ship is 1.0.70” (unless describing rollback)
- “chrsts broken on REV10”
- “Automatic placement unreliable” without “pre-1.0.70”
- “Worn follows die temperature” (true only before 1.0.82)
- “Charge on Qi / MagSafe”
- “Magnetic case” as the name of **this build** — say **charge case**; line up the coils. A magnetic dock is a possible later case
- “Kits already with Ed/Pedro” / “shipping” — until ship gate clears (say **gated** / **not yet shipped**)
- “Recommend 1.0.84” as current — recommend is **1.0.85** (1.0.84 still connects)
- “1.0.86 is the kit / ship target” — local tree only, until a flash guide, nRF Connect check, and OralableCore parse exist
- “The 14 Sep 1.0.85 merged hex boots” — it does not. The signed image is 119 bytes over the swap-move sector and stays in MCUboot
- “Kits ship by 30 Sep 2026” as a live date — that handoff **missed**; kits stay **gated**; no new date in this pack
- “Ed/Pedro kits are Gen 1” or “kit FW is 1.0.85” as the current stack — current hardware is Gen 2 REV11; **1.0.85** is the frozen Gen 1 archive
- “Gen 2 kit firmware is on the air as 2.0.x” or “kits are ready to hand off” — the 2 Nov candidate is built and not programmed. The board that streamed still reports **1.0.87**. Kits stay **gated**. [GEN2_BENCH_LEARNINGS_2026-11-02.md](./hardware/GEN2_BENCH_LEARNINGS_2026-11-02.md)
- “The 20-unit PC53020 lot has arrived” — the 14 Oct photos are one charger and one MAM, not that lot

## GATT lockstep

TGM UUIDs must match between `oralable_nrf/app/src/tgm_service.h` (`BT_UUID_TGM_*`) and `OralableCore/Sources/OralableCore/BLE/BLEConstants.swift`. Change firmware first; update Core in the same PR wave; validate with nRF Connect.
