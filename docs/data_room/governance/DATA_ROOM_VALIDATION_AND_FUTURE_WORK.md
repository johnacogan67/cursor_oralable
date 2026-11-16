# Data room — alignment audit and open work

**As at:** 30 Oct 2026 · Pack **1.1.74** (canonical: [`VERSION`](../VERSION) · [`VERSION_ALIGNMENT.md`](../VERSION_ALIGNMENT.md))  
**Current bench:** Gen 2 REV11. A merged image was written to one blank board on 29 Oct. PPG and ACC do not stream. Gen 1 **1.0.85** is frozen, not the kit. [GEN2_WOUT_KAGA_STATUS_2026-10-30.md](../hardware/GEN2_WOUT_KAGA_STATUS_2026-10-30.md). The 3 Oct sentences below are the previous pass. Where they say kit FW is **1.0.85**, use this line instead.  
**Method:** Living pass after FW **1.0.85** landed in `app/VERSION` and FirmwareGate, but most living docs still said **1.0.84**. Clinical lock unchanged. Frozen 7 Sep sendables left as-is with a living pointer.  
**Related:** Paper A audit [PAPER_A_VALIDATION_AND_FUTURE_WORK.md](../clinical/PAPER_A_VALIDATION_AND_FUTURE_WORK.md) · Nabavi [COLLAB_NABAVI_MCGILL.md](../clinical/COLLAB_NABAVI_MCGILL.md) · kit [ORALABLE_RESEARCH_KIT.md](../clinical/ORALABLE_RESEARCH_KIT.md) · feasibility [PAPER_A_FEASIBILITY_PROTOCOL.md](../clinical/PAPER_A_FEASIBILITY_PROTOCOL.md) · ANR [ANR_M40_CONCORDANCE.md](../bookmarks/ANR_M40_CONCORDANCE.md) · corroboration [SENSOR_CORROBORATION.md](../bookmarks/SENSOR_CORROBORATION.md) · parity [MAC_PHONE_ALGORITHM_PARITY.md](../bookmarks/MAC_PHONE_ALGORITHM_PARITY.md) · Pedro [PEDRO_STATUS_UPDATE_2026-09.md](../clinical/PEDRO_STATUS_UPDATE_2026-09.md) · meeting [ED_PEDRO_MEETING_2026-09-07.md](../clinical/ED_PEDRO_MEETING_2026-09-07.md)

**Overall verdict:** Pack **1.1.74** is the living diligence set. **Ken Kinsella is not a CEO candidate** (closed 19 Sep). **Hard blocker for field *N*:** kits still **gated** (charge-to-temple). The **30 Sep 2026** handoff **missed**. No new date. Gen 2 is the hardware. **1.0.85** is the frozen Gen 1 image, not the kit. Local `oralable_nrf` `app/VERSION` PATCHLEVEL **86** is **not** the kit target. Dual A is **lab methods**, not a patient arm. GTM Path **0** (clinic loaner) is now, not Path A home charge. n≈5 nights **unpaid**; later RKaaS is a draft €100/report SKU, not Point B. **UP filing receipt** still **GAP** (8 Aug clock passed). August Pedro status is **archived**. Science watch: short Dual A / Protocol A packs (~6 min) are layout / methods only. SASHB ≠ Azarbarzin HB. Feasibility n≈5 ≠ registered clinical trial. **Ken / CEO / Pedro PDFs lag MD.**

---

## 0. Canonical truth table (use when docs disagree)

| Topic | Canonical value | Source of truth |
|-------|-----------------|-----------------|
| Pack | **1.1.74** | `VERSION` |
| Living as-at | **30 Oct 2026** | VERSION_ALIGNMENT · Gen 2 status note |
| FW kit target | **2.0.x** target. Not a kit version. Bench image on one board 29 Oct. PPG and ACC do not stream. Gen 1 **1.0.85** is frozen (rollback **1.0.84**). | VERSION_ALIGNMENT |
| App | **4.3.3** (build **5**) | VERSION_ALIGNMENT |
| Gen1 HW | BOM **REV8** · PCB **REV10** · **ES2832AA2** · MAXM86161 · LIS2DTW12 | SYSTEM_ARCHITECTURE / Altium brief |
| Gen2 HW | BOM **REV9** · PCB **REV11** · **ES4L15BA1** · FW 2.0.x | same |
| Charge | Oralable **charge case** — line up the RX and TX coils on this build; a magnetic dock is a possible later case; **not Qi / MagSafe** | VERSION_ALIGNMENT “Do not say” · PRODUCT_ROADMAP §1 |
| Kits | **Research Kit** · **Gated / not yet shipped** · Oralable-only · **30 Sep 2026** handoff **missed** · no new date | ORALABLE_RESEARCH_KIT · VERSION_ALIGNMENT |
| Pilot phase | **Phase 0** temple HR/SpO₂ Day-1; Dual A **lab only**; muscle claims Phase 1+ | PRODUCT_ROADMAP · Research Kit |
| Paper A field | **Research Kit feasibility n≈5** Beacon; one night · ≥4 h; Dual A lab precursor; landscape Bruxoff/AcuPebble/GrindCare; not AHI | PAPER_A_FEASIBILITY_PROTOCOL |
| Dual A | **Lab methods** (Mac scripts + SpO₂∩EMG nest + `session.edf`); measured eng `20260812_085110`; iOS Dual A = Developer Settings (`showDualProtocolA` **OFF**); **not** a Beacon patient arm | ANR_M40 · ORALABLE_RESEARCH_KIT · VERSION_ALIGNMENT |
| GTM now | **Path 0 clinic loaner** (one night, clinic charge, pads). n≈5 **unpaid**. RKaaS later. Path A consumer / home charge is later. | GTM_ONE_PAGE · landscape §11 |
| Unitary Patent | Instructed; **filing receipt GAP** (8 Aug 2026 clock passed as at 12 Sep) | IP_PORTFOLIO |
| Soft corroboration | Skin temp **32–38 °C** + ACC quality gate; SpO₂/HR numbers unchanged; missing temp → Mac overnight path | SENSOR_CORROBORATION · ALGORITHM_ARCHITECTURE §0 |
| SpO₂∩EMG nest | Descriptive desat∩EMG (`NEST.md`); **not** ODI/AHI; AcuPebble remains Pedro AHI ref | ACUPEBBLE_VS_ORALABLE_ANR · align script |
| SASHB | Engineering SpO₂&lt;90 AUC (%·s); **not** Azarbarzin HB; **not** AHI | Paper A §IV.D · OVERNIGHT |
| Research EDF+ | Dual A `session.edf` (EMG when Dual A); research handoff — **not** PSG | ANR_M40 · export path |
| Nabavi vehicle | **McGill / RI-MUHC** · Phase 2 complementary Dianyx later | COLLAB |
| Overnight bands (iOS) | Unlock **≥1 h** worn | OVERNIGHT · FeatureFlags / OvernightNightReportBuilder |
| Beacon evaluable night | Wear **≥4 h** (6 h better) | OVERNIGHT · PAPER_A_FEASIBILITY |
| Ideal overnight / Paper A figure | Wear **≥6 h** (goal 8 h) | OVERNIGHT · PAPER_A |
| FIG-CO-025 pack | Layout exemplar; wear ≈ **6 min** | kpi_summary · Paper A validation |
| SB gold standard | Lab **PSG** (expert); Bruxoff = conditional ambulatory EMG | BRUXOFF_PSG_GOLD_STANDARD |
| BLE validation refs | Oralable → **nRF Connect**; ANR → BLE Design Guide + ANR iPhone app | ANR_M40_CONCORDANCE · nrf rules |
| EP foundation | **EP 4 333 691 B1** granted/certificated | IP_PORTFOLIO |
| US provisional | **64/033,978** filed **9 Apr 2026** — pitch-safe: patent pending only | IP_PORTFOLIO |
| Point B | **€180k** by 31 Oct 2026 | FUNDING / CURRENT_GOVERNANCE |
| Ken ~28% CEO ask | **Closed 19 Sep** — Ken is **not** a CEO candidate | CURRENT_GOVERNANCE |

---

## 1. Inventory by cluster (12 Aug 2026)

| Cluster | Docs | Alignment |
|---------|------|-----------|
| **A. Index / versions** | README, VERSION, VERSION_ALIGNMENT | **Good** — pack **1.1.73** · as-at 19 Sep · FW **1.0.85** · Ken not CEO candidate |
| **B. Pilot ops** | ED_PEDRO_QUICK_START, ORALABLE_RESEARCH_KIT, PEDRO_STATUS (Sep), PILOT_DRY_RUN, FIRMWARE_* | **Good** — Oralable-only kits; Dual A lab; kits still gated |
| **C. Nabavi / Paper A / lit** | COLLAB, KOOROSH_*, PITCH_KOOROSH*, LITERATURE, PAPER_A_*, FEASIBILITY | **Good** — Dual A lab precursor; Beacon ≥4 h; hypnogram figures still need ≥6 h |
| **C2. Bookmarks** | DIANYX_*, BRUXOFF_*, ANR_M40_*, ACUPEBBLE_*, MAC_PHONE_*, **SENSOR_CORROBORATION**, landscape | **Good** — ANR not in-kit; Dianyx still pre-clearance (12 Sep recheck) |
| **D. Product / FTS / GTM / cost** | ORALABLE_FTS_36MO, COST_AND_TIMELINE, APPS_AND_REVENUE, GTM, GEMINI, MARKET_SIZING, REGULATORY | **Good** — Path 0 clinic loaner now; Path A later; extraoral ≠ no FDA |
| **E. Pitches** | PITCH_PEDRO_ED_FF*, PITCH_DECK_KEN*, PITCH_CEO*, PITCH_KOOROSH* | **MD updated 19 Sep** (FW 1.0.85). Pedro PDF still **7 Sep** (sendable frozen). **Ken / CEO PDFs lag MD** |
| **F. Governance / funding** | CURRENT_GOVERNANCE, JAC_CORPORATE, FUNDING, FINANCIALS, KEN_PRESEED, MEETING_BRIEF, CEO_JD | **Living 19 Sep** — Ken **not** a CEO candidate; 22 Jul ~28% ask **closed** |
| **G. IP** | IP_PORTFOLIO, IP_EVAL | **Aligned** — provisional 64/033,978 pitch-safe; **UP receipt GAP** (clock passed) |
| **H. Quotes / COGS** | GEN1/GEN2_COGS, BITTELE | **OK** dated archives |
| **I. People / agenda** | JOHN_COGAN_CV, ED_PEDRO_AGENDA, ED_PEDRO_MEETING_2026-09-07 | Agenda **parked**; live path = Sep status |
| **J. Assets** | brand/, figures/, RESEARCH_KIT_PHOTO_SELECTION | Kit photos curated; FIG-CO-016 Dual A stack — caption not Beacon box |

**README coverage:** Dual A `20260812_085110` + EDF called out on index. No orphans this pass.

---

## 2. High-severity findings

### H1 — FIG-CO-025 is not an overnight (science) — WATCH

`TEMPORALIS_20260724` wear ≈ **6 min**.  
**Status:** Captions are correct in Paper A, OVERNIGHT, FIGURES, and validation; this pass also flagged ED_PEDRO_QUICK_START, PILOT_PROTOCOL, and the README overnight row.  
**Still open:** Collect a consented ≥6 h night before Results figures.

### H2 — Ken equity narrative conflict (governance) — CLOSED 19 Sep

Ken is **not** a CEO candidate. The 22 Jul ~28% ask is closed. Older JAC 6% FD Ken-CEO sketch is archive only. Living snapshot: [CURRENT_GOVERNANCE_STATUS.md](./CURRENT_GOVERNANCE_STATUS.md).

### H3 — Conor cash consideration — CLOSED 5 Oct

Share transfer **signed** (John, 5 Oct 2026). No cash payable. Register entry is still open. July price talks are history. The living table in CURRENT/FUNDING wins.

### H4 — CEO / COST “kits shipped” forward language — FIXED this pass

- CEO 0–90 days → charge gate cleared → kits unblocked / first field evidence.  
- COST Now–Sep row → **target** kits shipped **after** gate (currently gated).

### H5–H6 — IP “IE/UK validated” / GTM CE — FIXED previously (31 Jul)

Re-check if those docs are re-edited.

### H7 — Field data collection blocked — OPEN (ops)

The Paper A feasibility protocol is written, but **no Beacon *N*** until charge-to-temple clears and Pedro locks ethics.

### H8 — Unitary Patent filing receipt (IP) — OPEN

8 Aug 2026 clock **passed** as at 12 Sep. Counsel instructed UP; **receipt still GAP** in git. Do not pitch “UP registered.” Ask PurdyLucey.

---

## 3. Medium findings

| ID | Finding | Docs | Owner / action |
|----|---------|------|----------------|
| M1 | Pack stamps **1.1.42** / as-at 22–26 Jul on governance/IP eval | CURRENT, KEN_PRESEED, IP_EVAL | Banner: content frozen at meeting date; stack → VERSION_ALIGNMENT |
| M2 | Phase 0 vs Phase 1+ muscle mixed in outbound decks | PITCH_*, FTS | Keep scope sentence on every send |
| M3 | COST “kits already partially tooled” | COST | OK if read as HW exists; not partner-delivered |
| M4 | ED_PEDRO_AGENDA parked (Jun) | AGENDA | Refresh after kits + Paper A feasibility lock with Pedro |
| M5 | Pitch **PDF** may lag MD/HTML | Pedro / Paper A PDFs **7 Sep**. Ken MD updated **12 Sep** — **Ken PDF lags** | Re-run Chrome print for Ken before send |
| M6 | KOOROSH email still mentioned Tier 1 20–30 without leading n≈5 | KOOROSH_OUTREACH | **Fixed** — feasibility n≈5 first |
| M7 | PITCH_KOOROSH.md still says Core ML Tier 1 20–30 (correct as later rung) | PITCH_KOOROSH | OK — not Paper A *N*; optional footnote to feasibility protocol |
| M8 | Li DOI fetch flaky | LITERATURE | Manual confirm before submit |
| M9 | Foundation IP (masseter/intraoral roots) vs Stage A temple | IP_EVAL | Keep explicit on investor IP slides |
| M10 | Dianyx still pre-clearance | DIANYX_* | Rechecked **12 Sep** — still no public K-number |
| M11 | Beacon ethics still TBD | FEASIBILITY_PROTOCOL · MAYORAL | Pedro lock required |
| M12 | Ed SB×FEP was photos only | ED_PEDRO_SB_FEP | **Closed 15 Aug 2026** — Owens & Mayoral *Front Behav Neurosci* 20:1920406 (published 14 Aug) |

---

## 4. Low / OK

- No “chrsts broken on REV10” as current truth.  
- MagSafe/Qi only as forbidden charge path.  
- Dianyx JV correctly staged (McGill first).  
- AHI claim discipline present in Mayoral / Bruxoff / Feasibility / SB×FEP notes.  
- New bookmarks linked from README + COLLAB/lit as appropriate.  
- Quote archives (Kaga/Bittele) correctly dated.

---

## 5. Per-doc scorecard (living markdown)

Legend: **A** aligned · **W** watch · **S** stale stamp · **F** fixed this pass · **P** parked · **N** new since 31 Jul

| Doc | Score | Notes |
|-----|-------|-------|
| README.md | A/F | As-at 19 Sep; pack 1.1.73; FW **1.0.85** current; Ken not CEO candidate |
| VERSION / VERSION_ALIGNMENT | A/F | Living pack 19 Sep · FW **1.0.85** |
| GTM_ONE_PAGE | A/F | Path 0 clinic loaner now; Path A later |
| ORALABLE_MARKET_LANDSCAPE | A/F | ANR not in-kit; Path 0 |
| MEASUREMENT_CONSTRUCT_MAP | A/F | Dual A lab; changelog 12 Sep |
| IP_PORTFOLIO_STATUS | A/W | UP receipt GAP (clock passed) |
| IP_EVAL_AND_LANDSCAPE | S/W | Body 26 Jul; living banner 12 Sep |
| PITCH_DECK_KEN.md | A/W | MD 12 Sep; **PDF lags** |
| CURRENT_GOVERNANCE_STATUS | A/W | Living 12 Sep |
| ED_PEDRO_QUICK_START | A/F | FIG-CO-025 ~6 min note |
| VITALS_PILOT_TEST_PLAN | archived | Gen 1 **1.0.85** plan. Stub in `clinical/`. Full text in `archive/`. |
| PILOT_DRY_RUN_CHECKLIST | A | |
| PILOT_PROTOCOL_ED_PEDRO | A/F | Phase 1+ deferred; FIG-CO-025 caveat |
| FIRMWARE_1.0.70_FLASH | A | |
| FIRMWARE_1.0.66/65 | A | Rollback / archive |
| COLLAB_NABAVI_MCGILL | A/F | n≈5 feasibility in IEEE line |
| KOOROSH_OUTREACH | A/F | n≈5 then Tier 1 |
| PITCH_KOOROSH.md/.html/.pdf | A/W | Framing OK; PDF logo fixed 31 Jul |
| LITERATURE_AND_PRIOR_ART | A | Bruxoff pointer |
| PAPER_A_IEEE_* / REVIEW / VALIDATION | A/W | Placeholders + exemplar |
| **PAPER_A_FEASIBILITY_PROTOCOL** | **N/A** | Field start checklist; ethics TBD |
| **MAYORAL_METHOD_*** | A | Pedro apnea vs Ed bruxing |
| **BRUXOFF_PSG_*** | **N/A** | Gold-standard ladder; ANR vs Bruxoff table |
| **ANR_M40_*** / **ORALABLE_RESEARCH_KIT** | **A** | Dual A lab methods; Mac scripts; Beacon BOM Oralable-only |
| **DIANYX_FDA_*** | **N/A** | Pre-clearance bookmarks |
| **ED_PEDRO_SB_FEP_*** | **A** | Published 14 Aug 2026 (doi:10.3389/fnbeh.2026.1920406); theory ≠ Paper A methods |
| HW_ENGINEER_ALTIUM_BRIEF | A | |
| ORALABLE_FTS_36MO | A/F | Clinic loaner ≥4 h; Gen4 concept |
| COST_AND_TIMELINE | A/F | Gen4 vs Gen2; 1.0.85 soak |
| APPS_AND_REVENUE_EVAL | A | |
| GEMINI_TEMPLE_PPG_AVENUES | A | Distill 27 Jul; Dual A lab banner 12 Sep |
| MARKET_SIZING | A | |
| REGULATORY_TIMELINE | A/F | Extraoral ≠ no FDA; Beacon evidence Q4 |
| PITCH_DECK_KEN* / PITCH_CEO* | A/W | Ken MD 12 Sep; **Ken PDF lags** |
| JAC_CORPORATE_* | W | Ken 6% vs 28% ask |
| FUNDING_POINT_B_* | A/W | |
| FINANCIALS_CASH_SNAPSHOT | A | Sensitive — no IBAN in git |
| KEN_PRESEED_STRUCTURE_EVAL | S | 1.1.42 |
| MEETING_BRIEF_KEN_NIGEL | W | Prep vs outcomes |
| CEO_JOB_DESCRIPTION | A/F | Kits language |
| JOHN_COGAN_CV | A | |
| ED_PEDRO_AGENDA | P | Slipped; reactivate after kits/feasibility |
| GEN*_COGS / BITTELE | A | Archives |

---

## 6. Cross-check matrix (critical claims)

| Claim | Expected | Result (12 Sep) |
|-------|----------|----------------|
| FW target string | 1.0.85 | **Pass** — living pilot docs |
| App | 4.3.3 (build 5) | **Pass** |
| Pack | 1.1.73 | **Pass** — VERSION + README |
| Kits status | Research Kit gated; Oralable-only; 30 Sep handoff missed; no new date | **Pass (3 Oct)** |
| Charge path | Oralable case only | **Pass** |
| GTM Path 0 | Clinic loaner now; Path A later | **Pass** — GTM_ONE_PAGE · landscape §11 |
| Nabavi Phase 1 | McGill not Dianyx JV | **Pass** |
| Paper A *N* | Research Kit feasibility ≈5 (not Tier 1 20–30 as first *N*; not registered trial) | **Pass** |
| Pedro Arm P | 1–2 h SpO₂/oxygen, not AHI | **Pass** |
| Dual A | Lab methods; Mac measured `20260812_085110`; SpO₂ nest + EDF; not a Beacon patient arm | **Pass** |
| SpO₂∩EMG nest | Not claimed as ODI/AHI | **Pass** — NEST.md / ACUPEBBLE bookmark |
| SASHB | SpO₂&lt;90 AUC — not Azarbarzin HB | **Pass** — Paper A §IV.D |
| Research EDF+ | Dual A `session.edf` research only | **Pass** — ANR_M40 |
| iOS Dual A flag | `showDualProtocolA` default OFF (Developer Settings) | **Pass** |
| iOS overnight bands | Unlock ≥1 h; Beacon evaluable ≥4 h; hypnogram figure / bands ≥6 h | **Pass** — OVERNIGHT + VERSION_ALIGNMENT |
| Soft corroboration | Temp 32–38 °C + ACC quality; no SpO₂ hard-zero | **Pass** — SENSOR_CORROBORATION |
| FIG-CO-025 | ~6 min, not overnight | **Pass** captions; need real ≥6 h data |
| Oralable vs Bruxoff/PSG | No SB concordance yet | **Pass** |
| ANR role | Lab Dual A comparator; not in Beacon box | **Pass** |
| BLE refs | Oralable = nRF Connect; ANR ≠ nRF Connect primary | **Pass** |
| Dianyx FDA | Pre-clearance | **Pass** — rechecked 12 Sep; no public K-number |
| Unitary Patent | Filing receipt in pack | **GAP** — 8 Aug clock passed |

---

## 7. Future investigation (prioritized)

### A. Unblocks Paper A + Pedro collection

1. Close **charge-to-temple** gate; ship kits.  
2. Send Sep use-rules note if not sent; Pedro/Ed lock Beacon ethics.  
3. Gen1 pad-attach soak without headband.  
4. First ≥4 h clinic night (PDF + CRF).  
5. True ≥6 h night → replace FIG-CO-025 as Results figure.  
6. Ask PurdyLucey for **UP filing receipt**.  
7. Koorosh post-Denmark: venue + methods ownership.

### B. Science / gold standard ladder

1. Keep Paper A on Research Kit feasibility (pulse-ox plus descriptive Dual A is OK); deeper Bruxoff/PSG-AV diagnostic concordance belongs in later Paper C.  
2. Run Mac `run_dual_protocol_a_session.py` and concordance on ≥1 kit; do not block Day-1 vitals ship.  
3. Optional: borrow Bruxoff only after Phase 0 and Arm P succeed.  
4. Python↔Swift overnight-state parity report for Table II QC.

### C. Governance / pack hygiene

1. Banner on CURRENT / KEN_PRESEED / IP_EVAL for frozen stamps.  
2. Board decision note on Ken 28% vs 6% FD models.  
3. Regenerate Ken/CEO PDFs if those decks go out.

### D. Do not do

- Claim Oralable AHI or Bruxoff-beating SB accuracy.  
- Treat Dual A SpO₂∩EMG nest `desat_events_per_hour` as ODI/AHI.  
- Treat ANR concordance as PSG-AV diagnosis or imply ANR partnership.  
- Use nRF Connect as ANR’s product reference (BLE Design Guide + ANR app).  
- Send provisional claim text with Nabavi/Pedro packs.  
- Imply kits already with partners.

---

## 8. Fixes applied

### 2 Aug 2026

1. KOOROSH_OUTREACH + COLLAB — Paper A ladder leads with **feasibility n≈5**.  
2. ED_PEDRO_QUICK_START + PILOT_PROTOCOL — FIG-CO-025 **~6 min** caveat.  
3. COST timeline + CEO 0–90 — kits **gated / after gate** language.  
4. README + VERSION_ALIGNMENT — as-at **2 Aug**; feasibility + audit pointers.  
5. This audit file rewritten for new doc cluster + scorecard.

### 3 Aug 2026 (ANR align)

1. [ANR_M40_CONCORDANCE.md](../bookmarks/ANR_M40_CONCORDANCE.md) — BLE Design Guide + ANR iPhone app bookmarks; nRF Connect optional for ANR only.  
2. Cross-links: BRUXOFF ladder, LITERATURE, PAPER_A_*, MAYORAL, GEMINI, REGULATORY, MARKET_SIZING, README.  
3. Truth table: Dual A in Paper A + BLE validation refs locked.  
4. [ACUPEBBLE_VS_ORALABLE_ANR.md](../bookmarks/ACUPEBBLE_VS_ORALABLE_ANR.md) — Pedro AcuPebble (HSAT/AHI) vs Oralable vs ANR; Oralable ≠ AHI clone.

### 8 Aug 2026 (SpO₂∩EMG nest + iOS Dual A slice 1)

1. Pack bump **1.1.61 → 1.1.62**; README / VERSION_ALIGNMENT / ANR + AcuPebble bookmarks.  
2. Mac `align_anr_oralable_concordance.py` + `emg_spo2_nest.py` → `NEST.md` documented in Research Kit + Paper A C2.  
3. iOS `showDualProtocolA` (default OFF) noted as TestFlight research; Mac still primary.  
4. Claim scorecard: nest ≠ ODI/AHI; AcuPebble remains Pedro AHI reference.

### 10 Aug 2026 (soft corroboration + band unlock + Dual A opt-in)

1. Pack bump **1.1.62 → 1.1.63**; [SENSOR_CORROBORATION.md](../bookmarks/SENSOR_CORROBORATION.md) bookmark.  
2. OVERNIGHT / VERSION_ALIGNMENT / Research Kit / Paper A: iOS bands **≥1 h**; ideal / Paper A Arm E/J still **≥6 h**.  
3. ALGORITHM_ARCHITECTURE + MAC_PHONE_ALGORITHM_PARITY: soft ACC/temp as standing exception (no SpO₂ hard-zero).  
4. ANR Dual A + PILOT_DRY_RUN: Developer Settings Dual A; `DUAL_PAIR` skin-temp meta.  
5. Claim scorecard updated; kits remain gated.

### 12 Aug 2026 (measured Dual A + research EDF+)

1. Pack bump **1.1.63 → 1.1.64**; VERSION_ALIGNMENT Milestone 12 Aug.  
2. Paper A draft / cover / validation: Dual A `20260812_085110` precursor; SASHB ≠ Azarbarzin; EDF+ methods.  
3. Pedro status, Research Kit, feasibility protocol, handoff SOP aligned.  
4. ANR + AcuPebble bookmarks: `session.edf`, measured pack facts, claim discipline.  
5. Kits remain gated; short packs ≠ overnight *N*.

### 7 Sep 2026 (Beacon meeting lock)

1. Pack stayed **1.1.68**. Paper A / kit / Pedro Sep / meeting note: Oralable-only, no headband, ≥4 h, clinic loaner, Dual A lab.

### 12 Sep 2026 (living diligence pass)

1. Pack bump **1.1.68 → 1.1.69**; docs hub **1.3.17 → 1.3.18**.  
2. GTM Path **0** clinic loaner; landscape ANR **not in-kit**; construct map Dual A lab.  
3. COST / FTS / regulatory: Gen4 concept vs Gen2; extraoral ≠ no FDA; Beacon evidence Q4.  
4. IP: UP filing receipt **GAP** (8 Aug clock passed). Dianyx still pre-clearance.  
5. Ken pitch **MD** updated; **Ken PDF lags**. Pedro sendable left at 7 Sep.  
6. Pack **1.1.70**: archive [PEDRO_STATUS_UPDATE_2026-08.md](../archive/PEDRO_STATUS_UPDATE_2026-08.md); drop it from the operating pack. Unique checklists / bookmarks kept. Broken pre-meeting PDF link removed.

### 19 Sep 2026 (FW 1.0.85 living align)

1. Pack bump **1.1.71 → 1.1.72**; docs hub **1.3.19 → 1.3.20**.
2. Living docs: current stack FW **1.0.85** (hold LED: flash below ~4.20 V; rare green pulse in hold). iOS FirmwareGate recommend **1.0.85**. 1.0.84 is N−1.
3. Operator LED tables (quick start, vitals test plan) no longer tell field kits to expect **red** or **solid green at taper**.
4. Frozen 7 Sep sendables (Pedro status, Pedro F&F MD, meeting note) kept; living pointer added.
5. **Ken / CEO / Pedro / Paper A PDFs still lag MD.** Re-run Chrome print before send.
6. Clinical lock unchanged: kits **gated**; Oralable-only; Dual A lab; Path 0 unpaid n≈5.

### 19 Sep 2026 (Ken not a CEO candidate)

1. Pack bump **1.1.72 → 1.1.73**.
2. Living lock: **Ken Kinsella is not a CEO candidate.** 22 Jul ~28% CEO ask **closed**. John remains acting founder-CEO. CEO search = other candidates via Nigel.
3. CURRENT / JAC / FUNDING / KEN_PRESEED / Ken pitch / CEO JD / CEO candidate pitch updated. 22 Jul meeting brief stays archive.

### 3 Oct 2026 (missed handoff + kit FW lock)

1. Pack bump **1.1.73 → 1.1.74**; docs hub **1.3.20 → 1.3.21**.
2. Living docs: **30 Sep 2026** kit handoff **missed**. Kits still **gated**. No replacement date.
3. Kit FW stays **1.0.85**. Local `oralable_nrf` `app/VERSION` PATCHLEVEL **86** (GATT `3A0FF00D`) is **not** the kit target until a flash guide, an nRF Connect check, and an OralableCore parse exist.
4. Dry-run A4 and C2 now match the **1.0.85** hold LED (flash below ~4.20 V / 80%; rare pulse in hold). They no longer ask for solid green.
5. Frozen 7 Sep sendables keep their bodies. Living pointers name the missed date. Ken / CEO / Pedro PDFs still lag markdown.

---

## 9. Suggested re-scan command

```bash
cd docs/data_room
rg -n 'chrsts broken|Pilot ship is 1\.0\.66|kits already with|about to ship|IE/UK validated|1\.1\.42|Oralable.*AHI|n=20–30' --glob '*.md'
rg -n 'FIG-CO-025|20260812_085110' --glob '*.md'   # short packs must note ~6 min
rg -n 'nRF Connect.*ANR|ANR.*nRF Connect primary|ANR partnership' --glob '*.md'
rg -n 'desat_events_per_hour|ODI|showDualProtocolA|Azarbarzin' --glob '*.md'   # nest/SASHB claim discipline
rg -n 'Need ≥6 hours worn for overnight bands|evaluableWearSeconds|1\.1\.63' --glob '*.md'
```

---

*Living audit. Re-run after kit ship, Beacon ethics number, UP receipt, or a pack bump past 1.1.74. Dual A *N* is lab-only — not a Beacon patient gate.*
