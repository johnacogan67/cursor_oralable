# Paper A — document alignment, source validation, future investigation

**As at:** 30 Aug 2026 · Pack **1.1.68** · Dual A measured pack + research EDF+ + SASHB claim clarity (Paper A overnight still ≥6 h)  
**Scope:** Deep-dive check of [PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md](./PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md) against canonical docs, literature distill, code, and figure assets.  
**Pair with:** [PAPER_A_REVIEW_COVER.md](./PAPER_A_REVIEW_COVER.md) · [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) · [PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md) · [MEASUREMENT_CONSTRUCT_MAP.md](./MEASUREMENT_CONSTRUCT_MAP.md) · [COLLAB_NABAVI_MCGILL.md](./COLLAB_NABAVI_MCGILL.md) · [LITERATURE_AND_PRIOR_ART.md](./LITERATURE_AND_PRIOR_ART.md) · [VERSION_ALIGNMENT.md](./VERSION_ALIGNMENT.md)

**Verdict (12 Aug):** Paper A still centers the **Oralable Research Kit** with Beacon **n≈5**. **High-severity exemplar rule unchanged:** short Dual A / Protocol A packs (~6 min) are layout / methods precursors, not ≥6 h overnight *N*. Mac Dual A **`20260812_085110`** is the first measured concordance + SpO₂/SASHB + `session.edf` pack. Kits remain **gated**. SASHB in docs = engineering SpO₂&lt;90 AUC — **not** Azarbarzin HB.

---

## 1. Alignment matrix (docs ↔ Paper A)

| Claim / topic in Paper A | Canonical source | Status | Notes |
|--------------------------|------------------|--------|-------|
| Extraoral temporalis (not intraoral) | COLLAB · lit distill · pitch | **Aligned** | Locked differentiator |
| McGill / RI-MUHC · not Dianyx JV | COLLAB · outreach · cover | **Aligned** | COI disclosure still placeholder |
| Stage A wellness · no FDA/CE | COLLAB · IP north star · cover | **Aligned** | Keep in Abstract/Discussion |
| Patent: pending only, no claim text | IP_PORTFOLIO · IP_EVAL | **Aligned** | Provisional **64/033,978** filed 9 Apr 2026 — OK to say “filed/pending”; do not paste claims |
| FW **1.0.84** · app **4.3.3** (build **5**) · Gen1 BOM REV8 / PCB REV10 / ES2832AA2 | VERSION_ALIGNMENT · Ed/Pedro QS | **Aligned** | Matches Paper §III.B (stack current) |
| MAXM86161 + LIS2DTW12 | SYSTEM_ARCHITECTURE · HW Altium brief · GEN1_GEN2 | **Aligned** | Locked MPNs |
| 50 Hz research grid, BP 0.5–8 Hz, IR-DC &lt;1 Hz | ALGORITHM_ARCHITECTURE · .cursorrules | **Aligned** | Exact SOS/order still placeholder |
| TFI = IR-DC + green AC slope → 0–100 | `features.calculate_tfi` | **Aligned** | Paper wording OK; add methods detail later |
| SASHB = SpO₂&lt;90% AUC (%·s); rate ÷ wear_h | `features` / `self_validate` · OVERNIGHT | **Aligned** | Not AHI; **not** Azarbarzin HB — paper §IV.D clarified 12 Aug |
| Dual A measured pack `20260812_085110` | VERSION_ALIGNMENT · concordance plots | **Aligned** | SpO₂ yes; SASHB ≈929 %·s; EDF with EMG; lag ≈4.9 s; F1=0 this pack |
| Research EDF+ Dual A | `src/export/edf_writer.py` · OralableCore `EDFWriter` | **Aligned** | Mac + iOS Dual A Share; not PSG |
| States quiet/tonic/phasic/rescue/recovery | `overnight_states.py` · OVERNIGHT | **Aligned** | Thresholds still unpublished |
| ≥6 h evaluable night (Paper A / ideal) | OVERNIGHT · VERSION_ALIGNMENT | **Aligned** — ideal ≥6 h; iOS band unlock ≥1 h | **Misused exemplar** — see §2 |
| Hypnogram primary UX | OVERNIGHT · PRODUCT_ROADMAP · in-app Swift | **Aligned** | In-app ships; PDF secondary |
| Protocol A 5-tap / B 3-tap / overnight | TEMPORALIS_COLLECTION_PROTOCOL | **Aligned** | Paper table correct |
| Tier 1 ≈ 20–30 × 3–5 Protocol A | CORE_ML_TRAINING_COHORT · pitch | **Aligned** | |
| BruxScreen optional intake → Paper B | lit · protocol | **Aligned** | |
| Ed/Pedro kits with partners | VERSION_ALIGNMENT · Ed/Pedro QS | **Watch** | Kits **gated / not yet shipped** — Paper A must not imply field *N* |
| Phase 0 vitals vs Phase 1+ muscle | PRODUCT_ROADMAP · Research Kit protocol | **Resolved in rewrite** | Scope sentence + Dual A in Paper A methods; patient-facing TFI still Phase 1+ |
| Research Kit / Dual A / landscape | ORALABLE_RESEARCH_KIT · FEASIBILITY | **Aligned (8 Aug)** | Bruxoff ref · AcuPebble nest · SpO₂∩EMG Dual A nest · GrindCare peer |
| Dual A SpO₂∩EMG nest | ANR_M40 · ACUPEBBLE · Paper A C2 | **Aligned (8 Aug)** | `NEST.md`; not ODI/AHI |
| Anterior temporalis seat | TEMPORALIS_ANATOMY · Paper A §III.A2 · FIG-CO-056/057 | **Aligned (8 Aug)** | Elevate vs retract; MAM optical vs ANR electrical |
| 5 kits by 31 Aug 2026 | Research Kit · Pedro status | **Ops target** | Charge-to-temple still hard gate |
| Pack stamp | VERSION = **1.1.68** | **Aligned (30 Aug)** | Dual A measured + EDF · FW **1.0.84** |
| Venue list | Pitch: EMBC / Sensors / JBHI; Paper: JBHI / EMBC | **Minor drift** | Add **Sensors** to Paper placeholder |
| FIG-CO-003/006/007/011/012/019/025 present | `docs/figures/` | **Aligned** (assets exist) | Most SVGs still placeholders |

---

## 2. High-severity finding — short packs are not overnights

| Pack | Wear | Notes |
|------|------|-------|
| `TEMPORALIS_20260724` | ~**6.0 min** | FIG-CO-025 layout exemplar |
| `TEMPORALIS_20260812_085110_dualA` | ~**6.0 min** | Dual A gold → night-report layout; SpO₂/SASHB present |

**Implication:** Hypnogram PNGs show the **pipeline**, not evaluable ≥6 h nights. Paper A §VII.D lists both as non-inferential.

**Also (12 Aug):** Dual A concordance F1 vs Protocol A labels = **0** on `20260812_085110` despite lag ≈ 4.9 s and live SpO₂ — treat as QC / placement follow-up before filling Paper A agreement tables.

---

## 3. Phase 0 vs Paper A OMG tension (resolve in scope sentence)

| Layer | Truth today |
|-------|-------------|
| **Pilot ship (Ed/Pedro)** | Phase 0 Day-1 — temple **HR / SpO₂**; Research Kits gated (5 by 31 Aug); Dual A Mac optional |
| **Engineering / Paper A methods** | Full OMG path (IR-DC, TFI, states, hypnogram) exists in Python + Swift |
| **Product Phase 1+** | Muscle UX / clench-grind unlock on same Gen1 HW |

**Recommended Paper A scope sentence (for co-authors):**  
*“Paper A describes the measurement system and offline/online processing methods. Beacon pilot data may start with temple vitals and wear feasibility (Phase 0); IR-DC jaw-load endpoints follow as software and protocol mature on the same hardware.”*

Without that sentence, readers may assume muscle endpoints are already in the Beacon field protocol.

---

## 4. Source validation (external literature)

| Ref | Citation in draft | Validation | Action |
|-----|-------------------|------------|--------|
| **[1] Li et al.** | *Aust Dent J* 2024;69(1 Suppl):S53–S62 · doi:10.1111/adj.13057 | Matches lit distill; Wiley DOI form OK (page returned 406 to automated fetch — treat as **cite OK, PDF in Seed A TEC**) | Keep; note “accepted Jan 2025 / 2025 branding” in lit distill |
| **[2] Nabavi / Cogan / Roy intraoral PPG** | Was placeholder | **Filled:** Nabavi S, Cogan J, Roy A, Canfield B, Kibler R, Emerick C. Sleep Monitoring with Intraorally Measured Photoplethysmography (PPG) Signals. *2022 IEEE Sensors.* doi:[10.1109/SENSORS52175.2022.9967075](https://doi.org/10.1109/sensors52175.2022.9967075) | Confirm author order with Koorosh |
| **[3] Papapanagiotou 2017** | *IEEE JBHI* · doi:10.1109/JBHI.2016.2625271 | Matches lit distill | Keep |
| **[4] Lobbezoo BruxScreen** | *J Oral Rehabil* 2024;51:59–66 · doi:10.1111/joor.13442 | Matches lit distill (automated fetch timed out; Seed A PDF present) | Keep |
| **[5]–[9]** | Internal docs | Correct as **working refs**; must become appendix / citeable methods before submission | Replace before camera-ready |
| **[10] Charlton** | Placeholder | Not yet specific | Koorosh + John pick 1–2 PWA papers actually used |
| **[11] ICAB/STAB/RMMA** | Placeholder | Li cites ICAB 2018 / STAB | Add Lobbezoo consortium norms |
| **[17] Owens & Mayoral 2026** | *Front Behav Neurosci* 20:1920406 · doi:10.3389/fnbeh.2026.1920406 | Published 14 Aug 2026; PDF in Drive `notebook_lm/Sources/fnbeh-20-1920406.pdf` | Keep as Related Work / Paper B only — not Methods |

**Related Nabavi papers (optional Related Work, not Oralable product):**

- Flexible Hybrid Intraoral Sleep Monitoring System — doi:[10.1109/SENSORS56945.2023.10325122](https://doi.org/10.1109/sensors56945.2023.10325122)  
- Use only to **contrast** intraoral vs extraoral — do not merge product claims.

**Li gap claim:** “No temple optical / PPG–OMG ambulatory device in Li’s commercial table” — **supported by lit distill** as of Dec 2024 review cutoff. Re-run a short literature search before submission (devices published 2025–2026).

---

## 5. Internal consistency fixes applied in this pass

1. COLLAB pack stamp **1.1.58 → 1.1.59** (later **1.1.61** with Research Kit)  
2. Paper A ref **[2]** filled with IEEE Sensors 2022 DOI  
3. Paper A Fig. 5 / §VII.D: exemplar labeled **~6 min illustrative session**, not overnight  
4. Paper A §IX expanded with **Future investigation** checklist (below mirrored)  
5. Venue placeholder adds **Sensors** (matches pitch)  
6. Links: this validation doc from README + COLLAB  

---

## 6. Remaining doc drift (not all fixed here)

| Item | Where | Severity | Owner |
|------|-------|----------|-------|
| OVERNIGHT / FIGURES may still call FIG-CO-025 “overnight exemplar” without wear duration | OVERNIGHT_NIGHT_REPORT · FIGURES | High (partner confusion) | John |
| Pitch slide “overnight phenotype” vs Phase 0 kit reality | PITCH_KOOROSH | Medium | John |
| ACC native rate: firmware often 50 Hz; some docs say “up to 100 Hz” | .cursorrules vs upload docs | Low | John — state “resampled to 50 Hz analysis grid” |
| Python↔Swift numeric parity not CI-complete | ALGORITHM_ARCHITECTURE | Medium for Paper A methods rigor | Koorosh + John |
| Ed/Pedro agenda still slipped / pre–Stage A refresh | ED_PEDRO_AGENDA_2026-06-07 | Medium | John after Nabavi yes |
| Literature distill “Where incorporated” lacked Paper A row | LITERATURE | Low | Fixed below |

---

## 7. Directions for future investigation

### A. Measurement science (Paper A → camera-ready)

1. **True ≥6 h nights** — collect consented temple overnights; regenerate hypnogram + bands; retire short-session FIG-CO-025 as sole Fig. 5.  
2. **Publishable algorithm table** — filter orders/SOS, IR-DC LP, state machine windows/thresholds/hysteresis, SpO₂ curve (empirical 110–25R per `self_validate`).  
3. **SNR / coupling QC** — define IR-DC in-range (10M–70M raw), green SNR after BP, SpO₂ quality-gate pass rate; fill Table II.  
4. **Parity study** — same CSV through Python `overnight_states` vs Swift `OvernightStateClassifier`; report max abs epoch disagreement.  
5. **Motion robustness** — ACC rejection / false tonic during head turn; compare to Papapanagiotou-style ACC gating (awake prior art).  
6. **Updated ambulatory SB search** — confirm Li gap still holds post-2024 (optical temple devices).

### B. Clinical / Beacon (Paper B path)

1. Ethics / consent classification (wellness feasibility vs clinical investigation).  
2. BruxScreen-Q (± C) pilot as labels — correlation with TFI / tonic·h / hypnogram fractions (not diagnosis).  
3. Chairside usability study: does hypnogram-first change dentist review time vs CSV-only?  
4. Recalibrate Low/Moderate/High bands from ≥6 h distributions (current cutoffs provisional).  
5. Inclusion/exclusion + skin tone / habitus stratification (Tier 1 plan).

### C. Concordance (Paper C path)

1. Simultaneous **anterior temporalis sEMG** vs IR-DC bout timing (event-level F1 / lag) — founder Mac path: **ANR M40** Dual Protocol A ([ANR_M40_CONCORDANCE.md](./ANR_M40_CONCORDANCE.md); `run_dual_protocol_a_session.py` → `align_anr_oralable_concordance.py`). Expect ~1–5 s hemodynamic lag.  
2. Optional masseter channel (Bruxoff-class); swallow/speech FP gates from Protocol B.  
3. Later: PSG-AV RMMA subset (resource-heavy; not Paper A).  
4. BLE refs: Oralable → **nRF Connect**; ANR → BLE Design Guide + [ANR iPhone app](https://www.anrcorp.com/iphoneapp/) (nRF Connect optional inspect only).

### D. Collaboration / governance

1. Authorship order + CRediT; McGill affiliation line; REB if McGill analyzes.  
2. COI text (McGill vs Dianyx officer role) signed off.  
3. Data-sharing DPA before raw logs leave JAC.  
4. Venue sequence: conference abstract (EMBC/Sensors) vs JBHI full methods.  
5. Counsel: academic collab ≠ inventorship transfer (already in outreach checklist).

### E. Hardware / enablement (supports methods reproducibility)

1. Publication photos (device + placement) replacing FIG-CO-012/003 stubs.  
2. Block diagram (Fig. 2c) from Altium — non-confidential.  
3. Encapsulation path (tape → potting) only if claiming optical window stability.  
4. Ship gate: charge-to-temple ≥50% SOC so Beacon *N* can start.

---

## 8. Pre-send checklist (Koorosh / Ed / Pedro pack)

- [ ] Paper A + review cover + this validation note  
- [ ] FIG-CO-025 only with **~6 min illustrative** caption (or swap for real overnight)  
- [ ] No claim charts / Ken decks / raw CSVs  
- [ ] Scope sentence on Phase 0 vs OMG methods included  
- [ ] Ref [2] DOI confirmed by Koorosh  

---

*Audit trail for Paper A alignment. Update when exemplar night, ethics, or venue lock change.*
