# Paper A — Feasibility protocol (n≈5 · Research Kit · Beacon)

**As at:** 30 Aug 2026 · Pack **1.1.68** · FW **1.0.84** · app **4.3.3** (build **5**)  
**Status:** Working protocol for Pedro / Ed / John — **not** ethics-approved until Beacon path locked · **not** a registered clinical trial  
**Kit:** [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) · status [PEDRO_STATUS_UPDATE_2026-08.md](./PEDRO_STATUS_UPDATE_2026-08.md)  
**Paper:** [PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md](./PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md) · cover [PAPER_A_REVIEW_COVER.md](./PAPER_A_REVIEW_COVER.md)  
**Ops handout:** [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) · dry run [PILOT_DRY_RUN_CHECKLIST.md](./PILOT_DRY_RUN_CHECKLIST.md) · handoff [PAPER_A_DATA_HANDOFF_SOP.md](./PAPER_A_DATA_HANDOFF_SOP.md)  
**Related:** [MAYORAL_METHOD_ORALABLE_VALIDATION.md](./MAYORAL_METHOD_ORALABLE_VALIDATION.md) · [BRUXOFF_PSG_GOLD_STANDARD.md](./BRUXOFF_PSG_GOLD_STANDARD.md) · [ANR_M40_CONCORDANCE.md](./ANR_M40_CONCORDANCE.md) · [ACUPEBBLE_VS_ORALABLE_ANR.md](./ACUPEBBLE_VS_ORALABLE_ANR.md) · [COLLAB_NABAVI_MCGILL.md](./COLLAB_NABAVI_MCGILL.md)

**One-liner:** Beacon **feasibility** (n≈5) for the **Oralable Research Kit** (temple MAM + ANR Dual A + iOS long wear): usability, wear, QC, Pedro **1–2 h** oxygen/MAD arm, stretch ≥6 h, and a **descriptive** landscape vs Bruxoff / AcuPebble / GrindCare. Not AHI. Not SB diagnosis. Dual A concordance stays descriptive. Eng Dual A `20260812_085110` is methods precursor only.

---

## 1. Objectives (Paper A)

| # | Objective | Success (descriptive) |
|---|-----------|------------------------|
| O1 | Show Research Kit Oralable can be set up and worn at home / clinic | ≥80% of attempted sessions produce exportable CSV |
| O2 | Confirm temple **HR + SpO₂** signal quality under Phase 0 stack | “Vitals ready” / good-signal fraction logged; QC notes |
| O3 | Capture usability (comfort, setup time, adverse events) | CRF complete for each participant |
| O4 | **Arm P (Pedro):** 1–2 h oxygen-burden check ± MAD setting | ≥1 window per willing MAD user with setting logged |
| O5 | Stretch: ≥6 h night for hypnogram figure replacement | Optional for first close; target before submission |
| O6 | **Dual A (Mac):** labeled Oralable + ANR Protocol A on ≥1 kit session | Paired logs + concordance pack + `NEST.md` + **`session.edf`** (SpO₂∩EMG; not AHI; SASHB ≠ Azarbarzin HB) |
| O7 | **Landscape:** document output classes vs Bruxoff / AcuPebble / GrindCare | Methods table + figure placeholders (photos when available) |

**Out of scope (claim language)**

- Diagnostic **AHI** claims (SpO₂ / desat / engineering SASHB = SpO₂&lt;90 AUC only; AcuPebble remains Pedro’s AHI reference)  
- Azarbarzin **hypoxic burden** claims from Oralable alone (needs scored respiratory events)  
- “Superior to Bruxoff” or SB diagnosis vs lab **PSG-AV**  
- FDA/CE efficacy language  
- Calling this protocol a registered **clinical trial** (it is Beacon feasibility n≈5 until ethics path says otherwise)  
- Full product TFI claims in patient-facing copy (methods may describe pipeline)

**Scope sentence for paper / partners:**  
*Paper A describes the Oralable Research Kit measurement system (extraoral temporalis optical MAM + optional temporalis sEMG Dual A), processing methods, and a small Beacon feasibility cohort (n≈5) emphasizing wear, usability, temple vitals, longer-wear arms, and descriptive comparator landscape. IR-DC jaw-load and Dual A concordance are engineering phenotypes, not clinical SB diagnoses.*

---

## 2. Design

```text
Paper A feasibility — n≈5 (Beacon) · Research Kit
├── Core — setup, comfort, QC, export (HR/SpO₂)
├── Arm P (Pedro subset) — 1–2 h SpO₂ / oxygen check ± MAD
├── Arm E/J (stretch) — ≥6 h nights → hypnogram
├── Arm Dual A (Mac) — Protocol A Oralable + ANR M40
├── Landscape — Bruxoff (ref) · AcuPebble (AHI nest) · GrindCare (peer class)
└── Optional — finger/earlobe pulse ox overlap (short window)
```

| Item | Spec |
|------|------|
| **N** | ≈**5** (Pedro ± Ed self-nights first, then +3 consented recruits) |
| **Site** | Beacon Consultants Sleep Health Clinic / home wear under partner oversight |
| **Kit** | [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) — Gen1 Oralable + ANR M40 + iOS **4.3.3** (build **5**) + FW **1.0.84** |
| **Placement** | **Anterior temporalis** elevating belly (extraoral) — [TEMPORALIS_ANATOMY_AND_PLACEMENT.md](./TEMPORALIS_ANATOMY_AND_PLACEMENT.md) · FIG-CO-056/057 |
| **Ship gate** | **5 kits to Pedro by 31 Aug 2026** — each unit charge-to-temple proven before field *N* |

---

## 3. Arms

### Core (all participants)

1. Charge on Oralable case → temple mount → Connect (per [ED_PEDRO_QUICK_START](./ED_PEDRO_QUICK_START.md)).  
2. Session ≥5–10 min with HR + SpO₂ visible (“Good signal” when stable).  
3. Share → export CSV (+ optional Clinical Temporalis PDF).  
4. Complete CRF (below).

### Arm P — Pedro subset (apnea / MAD titration interest)

| Item | Spec |
|------|------|
| **Duration** | **1–2 hours** continuous temple wear (early sleep or agreed window) |
| **MAD** | Pedro logs device type, **VDO (mm)**, **advancement (mm or % max)**, symptoms |
| **Oralable metrics** | SpO₂ mean / min; time or burden below threshold (e.g. &lt;90%); HR summary; wear continuity |
| **AcuPebble nest** | Same-night or sequential AHI/ODI from Pedro’s AcuPebble when available — **compare, do not replace** |
| **Language** | “Oxygen burden / desaturation pattern for titration feedback” — **not** “AHI” |
| **Hypothesis (exploratory)** | Therapeutic MAD setting → lower oxygen burden vs baseline / prior setting (within-subject; descriptive) |

### Arm E/J — stretch (Ed + John / consented ≥6 h)

| Item | Spec |
|------|------|
| **Duration** | **≥6 h** worn (evaluable overnight) |
| **Outputs** | State hypnogram PDF/PNG; provisional bands if QC allows |
| **Bruxing endpoints** | Engineering phenotypes (states / TFI when unlocked) — not SB diagnosis |

### Arm Dual A — Mac labeled concordance

| Item | Spec |
|------|------|
| **Script** | `scripts/run_dual_protocol_a_session.py` (~6 min Protocol A cues) |
| **Devices** | Oralable temple + ANR M40 temporalis sEMG |
| **Post** | `scripts/align_anr_oralable_concordance.py` @ 50 Hz → IR-DC/EMG F1 + SpO₂∩EMG nest (`NEST.md`) + research **`session.edf`** (EMG inside; not PSG) |
| **iOS (optional)** | Developer Settings `showDualProtocolA` (default OFF) Share **4 files** incl. `session.edf` → same Mac align; Mac remains primary until proven |
| **Eng precursor** | Mac Dual A `20260812_085110` measured (SpO₂/SASHB/EDF; ~6 min; F1 vs labels = 0 this pack — QC follow-up) |
| **Claim** | Descriptive bout timing / amplitude / SpO₂ nest — not PSG-AV SB diagnosis; not AcuPebble AHI; SASHB ≠ Azarbarzin HB |

### Landscape / reference (descriptive)

| Device | Role in Paper A |
|--------|-----------------|
| **Bruxoff** | Ambulatory EMG-class **reference** when available (not required to start Core/Arm P) |
| **AcuPebble** | Pedro’s OSA HSAT / AHI nest for Arm P context |
| **GrindCare** | Temporalis sEMG peer class in related-work / landscape table |
| **Pulse ox** | Optional ≥20–30 min finger/earlobe overlap — descriptive SpO₂ agreement |

---

## 4. CRF (minimum fields)

| Field | Example |
|-------|---------|
| Participant ID (coded) | P01 |
| Date / start–end time | … |
| Operator | Pedro / Ed / self |
| FW string / app version | 1.0.84 / 4.3.3 (build 5) |
| Kit ID | RK01…RK05 |
| Placement confirmed temple | Y/N |
| Session type | Core / Arm P 1–2 h / ≥6 h / Dual A |
| MAD in use | Y/N · brand · VDO mm · advancement |
| AcuPebble used? | Y/N · SKU · AHI/ODI notes (if shared) |
| Bruxoff used? | Y/N · notes |
| Dual A session ID | … |
| Comfort 0–10 | … |
| Setup time (min) | … |
| Adverse events | skin / sleep disruption / none |
| Signal notes | drops, phone distance, “Vitals ready” |
| Export filename | `Oralable_VITALS_Pedro_YYYYMMDD.csv` |
| Pulse ox used? | Y/N · notes |

---

## 5. Ethics & claim discipline

| Item | Action |
|------|--------|
| Beacon ethics / consent | **Lock with Pedro (+ Ed)** — wellness feasibility vs clinical investigation; protocol number TBD |
| McGill REB | Only if analysis under McGill affiliation (Koorosh) — separate from Beacon start |
| Patient language | Stage A wellness measurement — not diagnosis of apnea or bruxism |
| Do not claim | AHI equivalence to PSG; superiority to Bruxoff; FDA clearance |

See [BRUXOFF_PSG_GOLD_STANDARD.md](./BRUXOFF_PSG_GOLD_STANDARD.md): portable EMG SB vs PSG fails in moderate/severe OSA. The oxygen arm is not EMG brux counts.

---

## 6. Data handoff → Paper A Results

**One-page SOP (which Share button / which file):** [PAPER_A_DATA_HANDOFF_SOP.md](./PAPER_A_DATA_HANDOFF_SOP.md) · **PDF** [PAPER_A_DATA_HANDOFF_SOP.pdf](./PAPER_A_DATA_HANDOFF_SOP.pdf).

Per participant or kit session, John receives:

1. Files per arm (Core CSV; Arm P / overnight → **Clinical Temporalis PDF**; Dual A → Mac pack or iOS **four-file** Share incl. `session.edf`) — see SOP  
2. Completed CRF row  
3. Optional screenshots (vitals card / Device LED)  
4. Dual A Mac pair logs + concordance pack (`NEST.md`, overlay, `session.edf`) when run  
5. AcuPebble / Bruxoff notes if nested  

**Do not** use Share → continuous CSV alone for 1–2 h or overnight (that export is ~3 min RAM). John builds descriptive tables: wear success rate, usability, SpO₂ QC, Arm P within-subject sketches, Dual A precursor plots. Replace FIG-CO-025 with a consented ≥6 h hypnogram when available. Eng packs `TEMPORALIS_20260724` and Dual A `20260812_085110` (~6 min each) are **layout / methods only**, not study *N*.

---

## 7. Checklist — start collecting with Pedro

### JAC / John (hard gate)

- [ ] Charge-to-temple gate closed on pilot unit (1.0.70)  
- [ ] **5 Research Kits** prepared (Oralable + ANR + case + TestFlight + cue card) — target handoff **31 Aug 2026**  
- [ ] Pedro dry-run: charge → temple → HR/SpO₂ → Share CSV  
- [x] Mac Dual A engineering precursor (`20260812_085110` + EDF) — repeat on kit hardware after ship  
- [ ] Optional Mac Dual A dry-run on one **shipped** kit  

### Pedro (+ Ed) — lock before / at handoff

- [ ] Agree this protocol = Paper A Research Kit **feasibility** (n≈5), not diagnostic AHI study / not registered trial language unless Beacon files that way  
- [ ] Confirm Arm P 1–2 h + MAD log fields + AcuPebble nest rules  
- [ ] Beacon ethics / consent path named (owner + next filing step)  
- [ ] Cohort plan: self-nights first? +3 recruits inclusion?  
- [ ] Authorship / Beacon affiliation line for Paper A  
- [ ] First calendar slot for dry-run + first Arm P window  
- [ ] Bruxoff access? (Y/N) · AcuPebble SKU for methods  

### After first logs

- [ ] Send Koorosh anonymized QC summary + venue/methods ask  
- [ ] Decide when Ed ≥6 h stretch starts  
- [ ] Schedule Dual A Mac sessions for concordance figures  

---

## 8. Authorship sketch (TBD)

| Role | Person | Likely CRediT |
|------|--------|----------------|
| Device / analysis | John A. Cogan (JAC) | Conceptualization, software, data curation |
| Methods / McGill | Koroosh Nabavi | Methodology, formal analysis (when engaged) |
| Clinical site / Arm P | Pedro Mayoral Sanz | Investigation, resources, joint clinical framing |
| Clinical site / overnight | Edward Owens | Investigation, clinical framing |

Order TBD on call.

---

*Draft for partner markup. Replaces informal WhatsApp bullets after Pedro signs off on ethics path + Arm P. Kit definition: [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md).*
