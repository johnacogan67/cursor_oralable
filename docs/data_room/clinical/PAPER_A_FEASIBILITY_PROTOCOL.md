# Paper A — Feasibility protocol (n≈5 · Research Kit · Beacon)

**As at:** 30 Oct 2026 · Pack **1.1.74** · **Gen 2** REV11 · **ES4L15BA1** · **LP270829** · app **4.3.3** (build **5**) · bench image is not kit firmware **2.0.x** · sensors do not stream · Gen 1 is legacy  
**Status:** Working protocol for Pedro / Ed / John — **not** ethics-approved until Beacon path locked · **not** a registered clinical trial  
**Meeting lock:** [ED_PEDRO_MEETING_2026-09-07.md](./ED_PEDRO_MEETING_2026-09-07.md) · sendable [PEDRO_STATUS_UPDATE_2026-09.md](./PEDRO_STATUS_UPDATE_2026-09.md)  
**Kit:** [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md)  
**Paper:** [PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md](./PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md) · cover [PAPER_A_REVIEW_COVER.md](./PAPER_A_REVIEW_COVER.md)  
**Ops handout:** [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) · dry run [PILOT_DRY_RUN_CHECKLIST.md](./PILOT_DRY_RUN_CHECKLIST.md) · handoff [PAPER_A_DATA_HANDOFF_SOP.md](./PAPER_A_DATA_HANDOFF_SOP.md)  
**Related:** [MAYORAL_METHOD_ORALABLE_VALIDATION.md](./MAYORAL_METHOD_ORALABLE_VALIDATION.md) · [BRUXOFF_PSG_GOLD_STANDARD.md](../bookmarks/BRUXOFF_PSG_GOLD_STANDARD.md) · [ANR_M40_CONCORDANCE.md](./ANR_M40_CONCORDANCE.md) · [ACUPEBBLE_VS_ORALABLE_ANR.md](../bookmarks/ACUPEBBLE_VS_ORALABLE_ANR.md) · [COLLAB_NABAVI_MCGILL.md](./COLLAB_NABAVI_MCGILL.md)

**One-liner:** Beacon **feasibility** (n≈5) for the **Oralable Research Kit** (temple MAM + iOS; **no ANR on the patient**): clinic-charged **one night**, pad/tape attach (**no headband**), evaluable wear **≥4 h** (6 h better), Pedro oxygen/MAD nest with AcuPebble, and a **descriptive** landscape vs Bruxoff / AcuPebble / GrindCare. Not AHI. Not SB diagnosis. Dual A stays Mac/eng methods precursor only.

---

## 1. Objectives (Paper A)

| # | Objective | Success (descriptive) |
|---|-----------|------------------------|
| O1 | Show Research Kit Oralable can be set up and worn (clinic loaner) | ≥80% of attempted sessions produce exportable PDF/CSV |
| O2 | Confirm temple **HR + SpO₂** signal quality under Phase 0 stack | “Vitals ready” / good-signal fraction logged; QC notes |
| O3 | Capture usability (comfort, setup time, adverse events) | CRF complete for each participant |
| O4 | **Arm P (Pedro):** oxygen-burden check ± MAD setting | ≥1 window per willing MAD user with setting logged (may sit inside a ≥4 h night) |
| O5 | **Beacon night:** ≥4 h worn (6 h better) for hypnogram / wear QC | ≥1 evaluable ≥4 h night per willing participant |
| O6 | **Hypnogram figure (stretch):** ≥6 h night to replace FIG-CO-025 | Optional for first close; target before overnight inferential claims |
| O7 | **Landscape:** document output classes vs Bruxoff / AcuPebble / GrindCare | Methods table + figure placeholders (photos when available) |
| O8 | **Dual A (optional, not patients):** Mac labeled Oralable + ANR | Eng precursor `20260812_085110` already exists; more packs John-run only |

**Out of scope (claim language)**

- Diagnostic **AHI** claims (SpO₂ / desat / engineering SASHB = SpO₂&lt;90 AUC only; AcuPebble remains Pedro’s AHI reference)  
- Azarbarzin **hypoxic burden** claims from Oralable alone (needs scored respiratory events)  
- “Superior to Bruxoff” or SB diagnosis vs lab **PSG-AV**  
- FDA/CE efficacy language  
- Calling this protocol a registered **clinical trial** (it is Beacon feasibility n≈5 until ethics path says otherwise)  
- Full product TFI claims in patient-facing copy (methods may describe pipeline)  
- Requiring patients to wear **ANR + MAM** together  
- Invoicing n≈5 nights (Paper A is unpaid collaboration; Path 0 RKaaS is a later GTM SKU — [GTM_ONE_PAGE.md](../governance/GTM_ONE_PAGE.md))  

**Scope sentence for paper / partners:**  
*Paper A describes the Oralable Research Kit measurement system (extraoral temporalis optical MAM; optional temporalis sEMG Dual A as a methods precursor, not a patient requirement), processing methods, and a small Beacon feasibility cohort (n≈5) emphasizing clinic-loaner wear, usability, temple vitals, ≥4 h nights, and a descriptive comparator landscape. IR-DC jaw-load and Dual A concordance are engineering phenotypes, not clinical SB diagnoses.*

---

## 2. Design

```text
Paper A feasibility — n≈5 (Beacon) · Research Kit (Oralable-only)
├── Core — setup, comfort, QC, export (HR/SpO₂)
├── Beacon night — one night · clinic charge · ≥4 h (6 h better)
├── Arm P (Pedro subset) — oxygen / MAD window (often inside the same night)
├── Landscape — Bruxoff (ref) · AcuPebble (AHI nest) · GrindCare (peer class)
├── Optional stretch — ≥6 h night → Paper A hypnogram figure / band recalibration
└── Dual A (Mac, not patients) — Protocol A Oralable + ANR M40
```

| Item | Spec |
|------|------|
| **N** | ≈**5** (Pedro ± Ed self-nights first, then +3 consented recruits) |
| **Site** | Beacon Consultants Sleep Health Clinic · **one night** · return next day |
| **Charge** | **Clinic** on the Oralable charge case (coils aligned now; magnetic dock later) — patient does not take a charger home |
| **Attach** | Anterior temporalis · **ECG / Red Dot pads or tape** · **no headband** |
| **Kit** | [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) — Gen 2 Oralable + charger + pads + iOS **4.3.3** (build **5**) · sensors do not stream yet · **ANR not in Beacon box** |
| **Placement** | **Anterior temporalis** elevating belly (extraoral) — [TEMPORALIS_ANATOMY_AND_PLACEMENT.md](./TEMPORALIS_ANATOMY_AND_PLACEMENT.md) · FIG-CO-056/057 |
| **Use owner** | **Ed** — packaging, attach, one-night return |
| **Arm P / AcuPebble owner** | **Pedro** |
| **Ship gate** | Charge-to-temple on each unit · **30 Sep 2026** handoff **missed** (31 Aug also missed) · kits still gated · no new date |

---

## 3. Arms

### Core (all participants)

1. Clinic charges clip on Oralable case → pad/tape on anterior temporalis (no headband) → Connect (per [ED_PEDRO_QUICK_START](./ED_PEDRO_QUICK_START.md)).  
2. Short QC: HR + SpO₂ visible (“Good signal” when stable).  
3. Patient takes the seated device for **one night**; returns next day.  
4. Share → **Clinical Temporalis PDF** (primary for ≥4 h) + CRF.

### Beacon night — Ed use rule (all n≈5)

| Item | Spec |
|------|------|
| **Duration** | **≥4 h** worn is evaluable for feasibility wear/QC. **6 h better.** |
| **Nights** | **One night** per loan, then return and recharge in clinic |
| **Attach** | Pads or tape on temple — AccuPebble/Sunrise class, not a headband |
| **Outputs** | State hypnogram PDF; wear continuity; HR/SpO₂ QC |
| **Bands** | iOS may unlock provisional bands from ≥1 h; **do not** fail a 4 h night as insufficient for feasibility. Recalibrate overnight **bands** from ≥6 h nights when they exist |
| **Bruxing endpoints** | Engineering phenotypes (states / TFI when unlocked) — not SB diagnosis |

### Arm P — Pedro subset (apnea / MAD titration interest)

| Item | Spec |
|------|------|
| **Duration** | Oxygen/MAD window **1–2 h** if Pedro wants a short check; otherwise the same **≥4 h** night with MAD logged |
| **MAD** | Pedro logs device type, **VDO (mm)**, **advancement (mm or % max)**, symptoms |
| **Oralable metrics** | SpO₂ mean / min; time or burden below threshold (e.g. &lt;90%); HR summary; wear continuity |
| **AcuPebble nest** | Same-night or sequential AHI/ODI from Pedro’s AcuPebble when available — **compare, do not replace** |
| **Language** | “Oxygen burden / desaturation pattern for titration feedback” — **not** “AHI” |
| **Hypothesis (exploratory)** | Therapeutic MAD setting → lower oxygen burden vs baseline / prior setting (within-subject; descriptive) |

### Dual A — Mac labeled concordance (**not** a patient arm)

| Item | Spec |
|------|------|
| **Who** | John (lab). **Ed does not require patients to combine ANR and MAM.** |
| **Script** | `scripts/run_dual_protocol_a_session.py` (~6 min Protocol A cues) |
| **Devices** | Oralable temple + ANR M40 temporalis sEMG |
| **Post** | `scripts/align_anr_oralable_concordance.py` @ 50 Hz → IR-DC/EMG F1 + SpO₂∩EMG nest (`NEST.md`) + research **`session.edf`** (EMG inside; not PSG) |
| **iOS (optional)** | Developer Settings `showDualProtocolA` (default OFF) — not Day-1 Beacon |
| **Eng precursor** | Mac Dual A `20260812_085110` measured (SpO₂/SASHB/EDF; ~6 min; F1 vs labels = 0 this pack — QC follow-up) |
| **Claim** | Descriptive bout timing / amplitude / SpO₂ nest — not PSG-AV SB diagnosis; not AcuPebble AHI; SASHB ≠ Azarbarzin HB |

### Landscape / reference (descriptive)

| Device | Role in Paper A |
|--------|-----------------|
| **Bruxoff** | Ambulatory EMG-class **reference** when available (not required to start Core / Beacon night) |
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
| FW string / app version | 2.0.x when it advertises / 4.3.3 (build 5). Write the GATT string that was read. Do not write 1.0.85 for a Gen 2 unit. |
| Kit ID | RK01…RK05 |
| Placement confirmed temple | Y/N |
| Attach | Pad / tape / other (no headband) |
| Session type | Core / Beacon ≥4 h / ≥6 h stretch / Arm P window |
| MAD in use | Y/N · brand · VDO mm · advancement |
| AcuPebble used? | Y/N · SKU · AHI/ODI notes (if shared) |
| Bruxoff used? | Y/N · notes |
| Dual A session ID | John-run only — leave blank for patients |
| Comfort 0–10 | … |
| Setup time (min) | … |
| Adverse events | skin / sleep disruption / none |
| Signal notes | drops, phone distance, “Vitals ready” |
| Export filename | `Oralable_NIGHT_Pedro_YYYYMMDD.pdf` |
| Pulse ox used? | Y/N · notes |

---

## 5. Ethics & claim discipline

| Item | Action |
|------|--------|
| Beacon ethics / consent | **Lock with Pedro (+ Ed)** — wellness feasibility vs clinical investigation; protocol number TBD |
| McGill REB | Only if analysis under McGill affiliation (Koorosh) — separate from Beacon start |
| Patient language | Stage A wellness measurement — not diagnosis of apnea or bruxism |
| Do not claim | AHI equivalence to PSG; superiority to Bruxoff; FDA clearance; extraoral = “no medical-device law” |

See [BRUXOFF_PSG_GOLD_STANDARD.md](../bookmarks/BRUXOFF_PSG_GOLD_STANDARD.md): portable EMG SB vs PSG fails in moderate/severe OSA. The oxygen arm is not EMG brux counts.

---

## 6. Data handoff → Paper A Results

**One-page SOP (which Share button / which file):** [PAPER_A_DATA_HANDOFF_SOP.md](./PAPER_A_DATA_HANDOFF_SOP.md) · **PDF** [PAPER_A_DATA_HANDOFF_SOP.pdf](./PAPER_A_DATA_HANDOFF_SOP.pdf).

Per participant or kit session, John receives:

1. Files per arm (Core CSV; Beacon night / Arm P → **Clinical Temporalis PDF**) — see SOP  
2. Completed CRF row  
3. Optional screenshots (vitals card / Device LED)  
4. Dual A Mac pair logs only if John ran Dual A (not expected from Beacon patients)  
5. AcuPebble / Bruxoff notes if nested  

**Do not** use Share → continuous CSV alone for multi-hour arms (that export is ~3 min RAM). John builds descriptive tables: wear success rate, usability, SpO₂ QC, Arm P within-subject sketches. Replace FIG-CO-025 with a consented **≥6 h** hypnogram when available; **≥4 h** nights still count for feasibility *N*. Eng packs `TEMPORALIS_20260724` and Dual A `20260812_085110` (~6 min each) are **layout / methods only**, not study *N*.

---

## 7. Checklist — start collecting with Pedro / Ed

### JAC / John (hard gate)

- [ ] Gen2 G2-P0 through G2-P6 evidence is independently reviewed before any unit is used for this protocol; a local build or bench advertisement is not acceptance evidence
- [ ] **5 Oralable-only kits** prepared (MAM + charger + pads/tape + TestFlight + quick start) — **30 Sep 2026** handoff **missed**; no new date; no hex until the port runs  
- [ ] Pad-attach soak: Gen 2 on temple **without headband**, `couple` logged  
- [ ] Beacon dry-run: clinic charge → pad seat → HR/SpO₂ → one night → PDF  
- [x] Mac Dual A engineering precursor (`20260812_085110` + EDF) — **not** a patient requirement  

### Pedro (+ Ed) — lock before / at handoff

- [ ] Agree this protocol = Paper A Research Kit **feasibility** (n≈5), not diagnostic AHI study / not registered trial language unless Beacon files that way  
- [ ] Confirm **Ed use rules:** one night, clinic charge, no headband, pads/tape, no ANR on the patient  
- [ ] Confirm **Pedro:** ≥4 h enough (6 h better); Arm P MAD log + AcuPebble nest  
- [ ] Beacon ethics / consent path named (owner + next filing step)  
- [ ] Cohort plan: self-nights first? +3 recruits inclusion?  
- [ ] Authorship / Beacon affiliation line for Paper A  
- [ ] First calendar slot for dry-run + first ≥4 h night  
- [ ] Clinic pad brand they already use (ECG)  
- [ ] Bruxoff access? (Y/N) · AcuPebble SKU for methods  

### After first logs

- [ ] Send Koorosh anonymized QC summary + venue/methods ask  
- [ ] Prefer a ≥6 h night for Fig. 5 replacement when one exists  

---

## 8. Authorship sketch (TBD)

| Role | Person | Likely CRediT |
|------|--------|----------------|
| Device / analysis | John A. Cogan (JAC) | Conceptualization, software, data curation |
| Methods / McGill | Koroosh Nabavi | Methodology, formal analysis (when engaged) |
| Clinical site / Arm P | Pedro Mayoral Sanz | Investigation, resources, joint clinical framing |
| Clinical site / wear and packaging | Edward Owens | Investigation, clinical framing |

Order TBD on call. Ed directed patient-use rules in the 7 Sep meeting.

---

*Draft for partner markup. Kit definition: [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md). Meeting source: [ED_PEDRO_MEETING_2026-09-07.md](./ED_PEDRO_MEETING_2026-09-07.md).*
