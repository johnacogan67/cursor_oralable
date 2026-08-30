# Data room — alignment audit and open work

**As at:** 30 Aug 2026 · Pack **1.1.68** (canonical: [`VERSION`](./VERSION) · [`VERSION_ALIGNMENT.md`](./VERSION_ALIGNMENT.md))  
**Method:** Re-align after measured Mac Dual A `20260812_085110`, research EDF+, and Paper A / Pedro / feasibility stamp bump. Prior soft corroboration + iOS band unlock (≥1 h) kept.  
**Related:** Paper A audit [PAPER_A_VALIDATION_AND_FUTURE_WORK.md](../clinical/PAPER_A_VALIDATION_AND_FUTURE_WORK.md) · Nabavi [COLLAB_NABAVI_MCGILL.md](../clinical/COLLAB_NABAVI_MCGILL.md) · kit [ORALABLE_RESEARCH_KIT.md](../clinical/ORALABLE_RESEARCH_KIT.md) · feasibility [PAPER_A_FEASIBILITY_PROTOCOL.md](../clinical/PAPER_A_FEASIBILITY_PROTOCOL.md) · ANR [ANR_M40_CONCORDANCE.md](../bookmarks/ANR_M40_CONCORDANCE.md) · corroboration [SENSOR_CORROBORATION.md](../bookmarks/SENSOR_CORROBORATION.md) · parity [MAC_PHONE_ALGORITHM_PARITY.md](../bookmarks/MAC_PHONE_ALGORITHM_PARITY.md) · Pedro [PEDRO_STATUS_UPDATE_2026-08.md](../clinical/PEDRO_STATUS_UPDATE_2026-08.md)

**Overall verdict:** Pack **1.1.68** aligns Paper A, Research Kit, Pedro status, and ANR/AcuPebble bookmarks with measured Dual A + EDF. **Hard blocker for field *N*:** Ed/Pedro Research Kits are still **gated** (charge-to-temple); target **5 by 31 Aug 2026**. Science watch: short Dual A / Protocol A packs (~6 min) are layout / methods only — not ≥6 h *N*. SASHB ≠ Azarbarzin HB. Feasibility n≈5 ≠ registered clinical trial. Ken equity narrative tension is unchanged.

---

## 0. Canonical truth table (use when docs disagree)

| Topic | Canonical value | Source of truth |
|-------|-----------------|-----------------|
| Pack | **1.1.68** | `VERSION` |
| Living as-at | **30 Aug 2026** | This audit · VERSION_ALIGNMENT |
| FW target | **1.0.84** (min gate 1.0.63; prior 1.0.82 / 1.0.70 = rollback) | VERSION_ALIGNMENT |
| App | **4.3.3** (build **5**) | VERSION_ALIGNMENT |
| Gen1 HW | BOM **REV8** · PCB **REV10** · **ES2832AA2** · MAXM86161 · LIS2DTW12 | SYSTEM_ARCHITECTURE / Altium brief |
| Gen2 HW | BOM **REV9** · PCB **REV11** · **ES4L15BA1** · FW 2.0.x | same |
| Charge | Oralable magnetic case only — **not Qi / MagSafe** | VERSION_ALIGNMENT “Do not say” |
| Kits | **Research Kit** · **Gated / not yet shipped** · target **5 → Pedro by 31 Aug 2026** | ORALABLE_RESEARCH_KIT · VERSION_ALIGNMENT |
| Pilot phase | **Phase 0** temple HR/SpO₂ Day-1; Dual A Mac optional; muscle claims Phase 1+ | PRODUCT_ROADMAP · Research Kit |
| Paper A field | **Research Kit feasibility n≈5** Beacon; Arm P 1–2 h oxygen; Dual A descriptive precursor; landscape Bruxoff/AcuPebble/GrindCare; not AHI | PAPER_A_FEASIBILITY_PROTOCOL |
| Dual A | **In Paper A methods** (Mac scripts + SpO₂∩EMG nest + `session.edf`); measured eng `20260812_085110`; iOS Dual A = Developer Settings (`showDualProtocolA` **OFF** by default); overnight Dual A = later | ANR_M40 · ORALABLE_RESEARCH_KIT · VERSION_ALIGNMENT |
| Soft corroboration | Skin temp **32–38 °C** + ACC quality gate; SpO₂/HR numbers unchanged; missing temp → Mac overnight path | SENSOR_CORROBORATION · ALGORITHM_ARCHITECTURE §0 |
| SpO₂∩EMG nest | Descriptive desat∩EMG (`NEST.md`); **not** ODI/AHI; AcuPebble remains Pedro AHI ref | ACUPEBBLE_VS_ORALABLE_ANR · align script |
| SASHB | Engineering SpO₂&lt;90 AUC (%·s); **not** Azarbarzin HB; **not** AHI | Paper A §IV.D · OVERNIGHT |
| Research EDF+ | Dual A `session.edf` (EMG when Dual A); research handoff — **not** PSG | ANR_M40 · export path |
| Nabavi vehicle | **McGill / RI-MUHC** · Phase 2 complementary Dianyx later | COLLAB |
| Overnight bands (iOS) | Unlock **≥1 h** worn | OVERNIGHT · FeatureFlags / OvernightNightReportBuilder |
| Ideal overnight / Paper A Arm E/J | Wear **≥6 h** (goal 8 h) | OVERNIGHT · PAPER_A |
| FIG-CO-025 pack | Layout exemplar; wear ≈ **6 min** | kpi_summary · Paper A validation |
| SB gold standard | Lab **PSG** (expert); Bruxoff = conditional ambulatory EMG | BRUXOFF_PSG_GOLD_STANDARD |
| BLE validation refs | Oralable → **nRF Connect**; ANR → BLE Design Guide + ANR iPhone app | ANR_M40_CONCORDANCE · nrf rules |
| EP foundation | **EP 4 333 691 B1** granted/certificated | IP_PORTFOLIO |
| US provisional | **64/033,978** filed **9 Apr 2026** — pitch-safe: patent pending only | IP_PORTFOLIO |
| Point B | **€180k** by 31 Oct 2026 | FUNDING / CURRENT_GOVERNANCE |
| Ken ~28% | **Asked, not agreed** | CURRENT_GOVERNANCE |

---

## 1. Inventory by cluster (12 Aug 2026)

| Cluster | Docs | Alignment |
|---------|------|-----------|
| **A. Index / versions** | README, VERSION, VERSION_ALIGNMENT | **Good** — pack **1.1.68** · as-at 30 Aug |
| **B. Pilot ops** | ED_PEDRO_QUICK_START, ORALABLE_RESEARCH_KIT, PEDRO_STATUS, PILOT_DRY_RUN, PILOT_PROTOCOL, FIRMWARE_* | **Good** — Dual A measured + EDF noted; kits still gated |
| **C. Nabavi / Paper A / lit** | COLLAB, KOOROSH_*, PITCH_KOOROSH*, LITERATURE, PAPER_A_*, FEASIBILITY | **Good** — Paper A Dual A precursor filled; overnight Results still need ≥6 h |
| **C2. Bookmarks** | DIANYX_*, BRUXOFF_*, ANR_M40_*, ACUPEBBLE_*, MAC_PHONE_*, **SENSOR_CORROBORATION** | **Good** — EDF + SASHB ≠ Azarbarzin; nest ≠ AHI |
| **D. Product / FTS / GTM / cost** | ORALABLE_FTS_36MO, COST_AND_TIMELINE, APPS_AND_REVENUE, GTM, GEMINI, MARKET_SIZING, REGULATORY | **OK** — FTS still says ≥6 h for Phase 1+ overnight (ideal; not wrong) |
| **E. Pitches** | PITCH_PEDRO_ED_FF*, PITCH_DECK_KEN*, PITCH_CEO*, PITCH_KOOROSH* | **OK** — no claim inflation this pass; regenerate PDFs only if sending |
| **F. Governance / funding** | CURRENT_GOVERNANCE, JAC_CORPORATE, FUNDING, FINANCIALS, KEN_PRESEED, MEETING_BRIEF, CEO_JD | **Living 7 Aug** on CURRENT_GOVERNANCE; Ken ~28% ask still open |
| **G. IP** | IP_PORTFOLIO, IP_EVAL | **Aligned** — provisional 64/033,978 pitch-safe |
| **H. Quotes / COGS** | GEN1/GEN2_COGS, BITTELE | **OK** dated archives |
| **I. People / agenda** | JOHN_COGAN_CV, ED_PEDRO_AGENDA | Agenda **parked**; reactivate via Pedro status |
| **J. Assets** | brand/, figures/, RESEARCH_KIT_PHOTO_SELECTION | Kit photos curated; FIG-CO-025 ~6 min |

**README coverage:** Dual A `20260812_085110` + EDF called out on index. No orphans this pass.

---

## 2. High-severity findings

### H1 — FIG-CO-025 is not an overnight (science) — WATCH

`TEMPORALIS_20260724` wear ≈ **6 min**.  
**Status:** Captions are correct in Paper A, OVERNIGHT, FIGURES, and validation; this pass also flagged ED_PEDRO_QUICK_START, PILOT_PROTOCOL, and the README overnight row.  
**Still open:** Collect a consented ≥6 h night before Results figures.

### H2 — Ken equity narrative conflict (governance) — OPEN

CURRENT (~28% ask, not agreed) vs older JAC_CORPORATE (~6% FD model). Board-sensitive — left as-is.

### H3 — Conor €10k cash-at-transfer vs deferred — OPEN

The outcomes table in CURRENT/FUNDING wins; older prep language may stay as history.

### H4 — CEO / COST “kits shipped” forward language — FIXED this pass

- CEO 0–90 days → charge gate cleared → kits unblocked / first field evidence.  
- COST Now–Sep row → **target** kits shipped **after** gate (currently gated).

### H5–H6 — IP “IE/UK validated” / GTM CE — FIXED previously (31 Jul)

Re-check if those docs are re-edited.

### H7 — Field data collection blocked — OPEN (ops)

The Paper A feasibility protocol is written, but **no Beacon *N*** until charge-to-temple clears and Pedro locks ethics.

---

## 3. Medium findings

| ID | Finding | Docs | Owner / action |
|----|---------|------|----------------|
| M1 | Pack stamps **1.1.42** / as-at 22–26 Jul on governance/IP eval | CURRENT, KEN_PRESEED, IP_EVAL | Banner: content frozen at meeting date; stack → VERSION_ALIGNMENT |
| M2 | Phase 0 vs Phase 1+ muscle mixed in outbound decks | PITCH_*, FTS | Keep scope sentence on every send |
| M3 | COST “kits already partially tooled” | COST | OK if read as HW exists; not partner-delivered |
| M4 | ED_PEDRO_AGENDA parked (Jun) | AGENDA | Refresh after kits + Paper A feasibility lock with Pedro |
| M5 | Pitch **PDF** may lag MD/HTML | Regenerated **7 Aug** from `_print.html` (Pedro 10p · Koorosh 10p · CEO 10p · Ken 15p) | Re-run Chrome print if MD changes |
| M6 | KOOROSH email still mentioned Tier 1 20–30 without leading n≈5 | KOOROSH_OUTREACH | **Fixed this pass** — feasibility n≈5 first |
| M7 | PITCH_KOOROSH.md still says Core ML Tier 1 20–30 (correct as later rung) | PITCH_KOOROSH | OK — not Paper A *N*; optional footnote to feasibility protocol |
| M8 | Li DOI fetch flaky | LITERATURE | Manual confirm before submit |
| M9 | Foundation IP (masseter/intraoral roots) vs Stage A temple | IP_EVAL | Keep explicit on investor IP slides |
| M10 | DIANYX landscape as-at still 31 Jul | DIANYX_* | Refresh when K-number appears |
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
| README.md | A/F | As-at 2 Aug; audit + overnight caveat |
| VERSION / VERSION_ALIGNMENT | A/F | Feasibility protocol pointer |
| ED_PEDRO_QUICK_START | A/F | FIG-CO-025 ~6 min note |
| VITALS_PILOT_TEST_PLAN | A | |
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
| **ANR_M40_*** / **ORALABLE_RESEARCH_KIT** | **A** | Dual A in Paper A methods; Mac scripts; kit BOM |
| **DIANYX_FDA_*** | **N/A** | Pre-clearance bookmarks |
| **ED_PEDRO_SB_FEP_*** | **A** | Published 14 Aug 2026 (doi:10.3389/fnbeh.2026.1920406); theory ≠ Paper A methods |
| HW_ENGINEER_ALTIUM_BRIEF | A | |
| ORALABLE_FTS_36MO | A/W | |
| COST_AND_TIMELINE | A/F | Timeline gated wording |
| APPS_AND_REVENUE_EVAL | A | |
| GTM_ONE_PAGE | A | |
| GEMINI_TEMPLE_PPG_AVENUES | A | As-at 27 Jul OK as distill |
| MARKET_SIZING | A | |
| REGULATORY_TIMELINE | A/W | |
| PITCH_DECK_KEN* / PITCH_CEO* | A/W | PDF sync |
| CURRENT_GOVERNANCE_STATUS | S/W | Best snapshot; stamp old |
| JAC_CORPORATE_* | W | Ken 6% vs 28% ask |
| FUNDING_POINT_B_* | A/W | |
| FINANCIALS_CASH_SNAPSHOT | A | Sensitive — no IBAN in git |
| KEN_PRESEED_STRUCTURE_EVAL | S | 1.1.42 |
| MEETING_BRIEF_KEN_NIGEL | W | Prep vs outcomes |
| CEO_JOB_DESCRIPTION | A/F | Kits language |
| IP_PORTFOLIO_STATUS | A/W | |
| IP_EVAL_AND_LANDSCAPE | S/W | 1.1.42 |
| JOHN_COGAN_CV | A | |
| ED_PEDRO_AGENDA | P | Slipped; reactivate after kits/feasibility |
| GEN*_COGS / BITTELE | A | Archives |

---

## 6. Cross-check matrix (critical claims)

| Claim | Expected | Result (12 Aug) |
|-------|----------|----------------|
| FW target string | 1.0.84 | **Pass** — living pilot docs |
| App | 4.3.3 (build 5) | **Pass** |
| Pack | 1.1.68 | **Pass** — VERSION + README |
| Kits status | Research Kit gated; target 5 by 31 Aug | **Pass** |
| Charge path | Oralable case only | **Pass** |
| Nabavi Phase 1 | McGill not Dianyx JV | **Pass** |
| Paper A *N* | Research Kit feasibility ≈5 (not Tier 1 20–30 as first *N*; not registered trial) | **Pass** |
| Pedro Arm P | 1–2 h SpO₂/oxygen, not AHI | **Pass** |
| Dual A | In Paper A methods; Mac measured `20260812_085110`; SpO₂ nest + EDF | **Pass** |
| SpO₂∩EMG nest | Not claimed as ODI/AHI | **Pass** — NEST.md / ACUPEBBLE bookmark |
| SASHB | SpO₂&lt;90 AUC — not Azarbarzin HB | **Pass** — Paper A §IV.D |
| Research EDF+ | Dual A `session.edf` research only | **Pass** — ANR_M40 |
| iOS Dual A flag | `showDualProtocolA` default OFF (Developer Settings) | **Pass** |
| iOS overnight bands | Unlock ≥1 h; ideal / Paper A ≥6 h | **Pass** — OVERNIGHT + VERSION_ALIGNMENT |
| Soft corroboration | Temp 32–38 °C + ACC quality; no SpO₂ hard-zero | **Pass** — SENSOR_CORROBORATION |
| FIG-CO-025 | ~6 min, not overnight | **Pass** captions; need real ≥6 h data |
| Oralable vs Bruxoff/PSG | No SB concordance yet | **Pass** |
| ANR role | Research Kit comparator; not consumer product alone | **Pass** |
| BLE refs | Oralable = nRF Connect; ANR ≠ nRF Connect primary | **Pass** |
| Dianyx FDA | Pre-clearance | **Pass** |

---

## 7. Future investigation (prioritized)

### A. Unblocks Paper A + Pedro collection

1. Close **charge-to-temple** gate; ship kits.  
2. Pedro locks Beacon ethics + Arm P CRF ([PAPER_A_FEASIBILITY_PROTOCOL](../clinical/PAPER_A_FEASIBILITY_PROTOCOL.md) §7).  
3. First dry-run CSV + optional 1–2 h MAD window.  
4. True ≥6 h night → replace FIG-CO-025 as Results figure.  
5. Koorosh post-Denmark: venue + methods ownership.

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

*Living audit. Re-run after kit ship, Beacon ethics number, partner Dual A *N*, or a pack bump past 1.1.68.*
