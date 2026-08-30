# Literature & prior-art distill (Seed A TEC pack)

**App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

**As at:** 7 Aug 2026 · Pack **1.1.61**  
**Sources:** Google Drive `Oralable Seed A Data Room/04_TEC/` (local paths below — not in git)  
**Use:** Market and clinical framing, Paper A related work, HW enclosure notes, founder CV.  
**Claim tone:** Stage A wellness. Cite as literature, not as Oralable clearance evidence.

**Canonical product differentiator:** Oralable = **extraoral temporalis** optical (OMG / IR-DC + ACC) — not sEMG, not intraoral appliance sensing. See [COLLAB_NABAVI_MCGILL.md](../clinical/COLLAB_NABAVI_MCGILL.md). **Figures:** [../FIGURES.md](../../FIGURES.md).

![FIG-CO-011 Extraoral vs intraoral](../figures/FIG-CO-011-extraoral-vs-intraoral.svg)

*Figure FIG-CO-011 — Extraoral temporalis (Oralable) vs intraoral sensing (placeholder).*

---

## 1. Li et al. 2025 — Ambulatory devices to detect sleep bruxism (narrative review)

| | |
|--|--|
| **Cite** | Li C, Yap S, Loh A, Yap YJ, Kujan O, Balasubramaniam R. Ambulatory devices to detect sleep bruxism: a narrative review. *Aust Dent J.* 2024;69(1 Suppl):S53–S62. doi:[10.1111/adj.13057](https://doi.org/10.1111/adj.13057) (accepted Jan 2025) |
| **File** | `04_TEC_260609_paper_references_collatated_review_paper_Australian Dental Journal - 2025 - Li - Ambulatory devices…pdf` |
| **Incorporate into** | Market landscape · Paper A related work · regulatory/clinical screening narrative |

### Useful facts

- Gold standard for SB diagnosis remains **PSG with audio-visual recording (PSG-AV)** — resource-intensive / poorly accessible.
- SB prevalence cited ~**8–16% adults** (higher in children); RMMA on **masseter and temporalis**.
- Assessment frameworks: **ICAB** (2018) · **STAB** (instrument + clinical + patient tools).
- Commercial ambulatory landscape reviewed (to Dec 2024) is dominated by **sEMG**, Type II PSG, mandibular motion, or **intra-splint force** — Table 3 examples:

| Device | Modality (Li) | Notes vs PSG-AV (as reported) |
|--------|---------------|-------------------------------|
| BiteStrip | masseter sEMG | Episodic sens. ~72–84%; specificity vs PSG-AV often missing |
| dia-BRUXO | masseter sEMG | Limited PSG-AV comparison |
| GrindCare | **anterior temporalis** sEMG (+ biofeedback impulse) | Diagnostic sens/spec ~60% in cited adult work |
| Bruxoff | masseter sEMG + ECG | Better than EMG-only in some reports; still overestimates RMMA — see [BRUXOFF_PSG_GOLD_STANDARD.md](./BRUXOFF_PSG_GOLD_STANDARD.md) (Cid-Verdejo 2023 vs PSG in OSA) |
| NOX T3 / Sleep Profiler | Type II PSG multi-channel | Mixed; more channels needed for fair SB assessment |
| Sunrise | mandibular movement | Episodic sens. high; tonic clench / non-OSA gaps |
| ISFDD | pressure in maxillary splint | Intraoral force — different class |

- **Conclusion (authors):** select ambulatory devices are **promising as screening tools**, but need more PSG-AV validation before widespread clinical/domestic adoption.
- **EMG-only limitation (authors):** without AV, hard to exclude orofacial / other muscular activities → **overestimate** SB.

### Oralable implication

Li’s commercial table lists **no temple optical / PPG–OMG ambulatory device**. That is the gap Oralable targets (extraoral temporalis IR-DC + ACC). Keep sEMG for Stage B and research concordance — **ANR M40** Dual A in the [Oralable Research Kit](../clinical/ORALABLE_RESEARCH_KIT.md) (Paper A descriptive precursor; deeper PSG-AV later) — not as the consumer modality.

---

## 2. Lobbezoo et al. 2023/2024 — BruxScreen

| | |
|--|--|
| **Cite** | Lobbezoo F, Ahlberg J, Verhoeff MC, et al. The bruxism screener (BruxScreen): Development, pilot testing and face validity. *J Oral Rehabil.* 2024;51:59–66. doi:[10.1111/joor.13442](https://doi.org/10.1111/joor.13442) |
| **File** | `04_TEC_260609_paper_BruxScreen_…pdf` |
| **Incorporate into** | Ed/Pedro protocol · clinical validation · dentist app later |

### Useful facts

- **STAB** = comprehensive assessment (too heavy for everyday practice / “A4” Accurate–Applicable–Affordable–Accessible).
- **BruxScreen** = lighter screener: **BruxScreen-Q** (patient questionnaire) + **BruxScreen-C** (dentist clinical form).
- Face validity / pilot done; authors: ready for deeper psychometric testing in general dentistry.
- Prevalence framing: awake bruxism ~8–31%; sleep bruxism ~12.8±3.1% (as cited).

### Oralable implication

Use BruxScreen (or STAB subsets) as **clinical label / intake** next to device CSV — not as a stand-in for overnight instrumented phenotype. Fits Pedro/Ed workflows and Paper B clinical labels.

---

## 3. Papapanagiotou et al. 2017 — Chewing detection (PPG + audio + ACC)

| | |
|--|--|
| **Cite** | Papapanagiotou V, Diou C, Zhou L, et al. A novel chewing detection system based on PPG, audio, and accelerometry. *IEEE J Biomed Health Inform.* 2017;21(3):607–618. doi:[10.1109/JBHI.2016.2625271](https://doi.org/10.1109/JBHI.2016.2625271) |
| **File** | `04_TEC_260609_paper_A_Novel_Chewing_Detection_System_Based_on_PPG_Audio_and_Accelerometry.pdf` |
| **Incorporate into** | Algorithm / Paper A related work (awake chewing ≠ sleep bruxism) |

### Useful facts

- Ear-hook **PPG** + in-ear **audio** + belt **ACC**; SVM / late fusion.
- Eating/chewing detection accuracy up to ~0.94 (class-weighted ~0.89) on semi-free-living data.
- ACC used to suppress false chewing during high activity.
- Mentions strain / temporalis–masseter muscle sensing in related work — different goal (nutrition / snacking).

### Oralable implication

Prior art shows **PPG near the ear/jaw can carry masticatory information**, but: (1) **awake chewing / eating**, not overnight SB; (2) **ear** placement plus audio; (3) not IR-DC occlusion / TFI. Cite as adjacent optical–mastication literature; keep Oralable claims on **sleep temporalis OMG**.

---

## 4. Silicone encapsulation — potting vs tape (internal HW note, Mar 2026)

| | |
|--|--|
| **Source** | Internal technical note (Gmail export) · 26 Mar 2026 · John |
| **File** | `04_TEC_260306_2026-03-06_hardware_Gmail - Silicone Encapsulation_ Potting vs. Tape.pdf` |
| **Incorporate into** | HW engineer brief · Gen1/Gen2 enclosure path |

### Useful facts (for ~20×7×4 mm skin-contact wearable)

| | Potting (liquid cast) | Tape lamination (“sandwich”) |
|--|----------------------|------------------------------|
| Best for | Mass production / finished look | Prototyping / rapid iteration |
| Finish | Monolithic / polished | Visible seam / handmade |
| Optical | Uniform (mix-dependent) | High via thin membrane |

**Recommendation in note:** Tape for **immediate testing**; transition to **potting** for commercial finish once sensor placement/dimensions validated.

### Oralable implication

Pilot and Ed–Pedro kits can stay on tape or the current housing. **Gen2 / volume** should budget potting tooling. The optical window over MAXM86161 must stay clear in either path.

---

## 5. Cogan 1999 — Trinity PhD thesis

| | |
|--|--|
| **Cite** | Cogan JA. *Computer approaches to total hip replacement evaluation just prior to operation.* PhD thesis, Trinity College Dublin, Mechanical Engineering, 1999. |
| **File** | `04_TEC_260609_paper_Cogan TCD THESIS 5247 Computer approaches.pdf` |
| **Incorporate into** | [JOHN_COGAN_CV.md](../pitches/JOHN_COGAN_CV.md) (already Trinity PhD — enrich cite) |

### Useful facts

- Computational evaluation of **THR** (fit-and-fill, FEA, clinical hip scores, rule-based / fuzzy / NN methods).
- Shows the founder’s long **medical engineering and decision-support** background — not bruxism IP, but credibility for clinical–engineering work (McGill / Beacon).

---

## 6. Cogan & Prendergast — hip prosthesis knowledge engineering (1993 era)

| | |
|--|--|
| **Cite** | Cogan JA, Prendergast PJ. Preliminary investigation of knowledge engineering to optimise hip prosthesis design. Bioengineering Research Centre, Dept. of Mechanical Engineering, Trinity College Dublin. *(Seed A file: `04_TEC_260609_paper_your_paper_with_prendercast_1993.pdf` — scanned; conference/proceedings venue not on extract — confirm page/volume if citing formally)* |
| **Funding** | EOLAS ST/92/106 “Computer-Aided Design of Orthopaedic Implants” |
| **Incorporate into** | [JOHN_COGAN_CV.md](../pitches/JOHN_COGAN_CV.md) early publications · founder clinical–engineering narrative |

### Useful facts

- Prototype **intelligent system** for hip prosthesis design: combines **rule-based ES**, **model-based ES** (e.g. FEA / fatigue), and a **hypertext knowledge-base**.
- Knowledge drawn from **engineering sciences**, **clinical sciences**, and **manufacturing technology**; design stages: specification → conceptual → technical → manufacture → follow-up.
- Stack (prototype): Visual Basic + Windows DDE → Excel, FEA, expert shell **M4**; 486 PC era.
- Parallel goal: accumulate interdisciplinary knowledge (engineers + clinicians) so design is less “intuitive-only.”
- Complements the 1999 TCD PhD (*Computer approaches to total hip replacement evaluation…*).

### Oralable implication

Not bruxism IP. It shows a long founder track record in **clinical–engineering knowledge systems** — useful for CV and McGill–Beacon credibility, not Paper A related work.

---

## Where incorporated

| Doc | What landed |
|-----|-------------|
| [ORALABLE_MARKET_LANDSCAPE.md](./ORALABLE_MARKET_LANDSCAPE.md) (§4c) | Li ambulatory device table + optical gap |
| [CLINICAL_VALIDATION.md](../../CLINICAL_VALIDATION.md) | BruxScreen / STAB screening note |
| [TEMPORALIS_COLLECTION_PROTOCOL.md](../../TEMPORALIS_COLLECTION_PROTOCOL.md) | Optional BruxScreen-Q intake |
| [HW_ENGINEER_ALTIUM_BRIEF.md](../hardware/HW_ENGINEER_ALTIUM_BRIEF.md) | Encapsulation potting vs tape |
| [JOHN_COGAN_CV.md](../pitches/JOHN_COGAN_CV.md) | Thesis full cite |
| [PITCH_KOOROSH.md](../pitches/PITCH_KOOROSH.md) / Paper A framing | Related-work gap (no optical ambulatory in Li) |
| [PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md](../clinical/PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md) | Related work §§ + refs [1]–[4]; intraoral contrast [2]; Owens & Mayoral 2026 [17] (theory; Paper B) |
| [ED_PEDRO_SB_FEP_DRAFT_PAPER.md](../clinical/ED_PEDRO_SB_FEP_DRAFT_PAPER.md) | Owens & Mayoral 2026 SB × FEP (*Front Behav Neurosci*); cite Paper B / Arm P, not Paper A methods |
| [PAPER_A_VALIDATION_AND_FUTURE_WORK.md](../clinical/PAPER_A_VALIDATION_AND_FUTURE_WORK.md) | DOI/source audit + future investigation |
| [BRUXOFF_PSG_GOLD_STANDARD.md](./BRUXOFF_PSG_GOLD_STANDARD.md) | Cid-Verdejo 2023 Bruxoff vs PSG; Oralable ladder vs gold standard |
| [ANR_M40_CONCORDANCE.md](../bookmarks/ANR_M40_CONCORDANCE.md) | ANR M40 temporalis sEMG Research Kit / Paper A Dual A precursor; BLE ref ≠ nRF Connect |
| [HAPPY_RING.md](./HAPPY_RING.md) | Happy Ring Oura-like finger HSAT (K240236 / K242224 hAHI) |
| [ORALABLE_RESEARCH_KIT.md](../clinical/ORALABLE_RESEARCH_KIT.md) | Canonical kit BOM · Dual A wear stack · competitor landscape |
| [MARKET_SIZING.md](../governance/MARKET_SIZING.md) | Prevalence cites (literature) |

### Nabavi / Cogan intraoral PPG (full cite for Paper A ref [2])

Nabavi S, Cogan J, Roy A, Canfield B, Kibler R, Emerick C. Sleep Monitoring with Intraorally Measured Photoplethysmography (PPG) Signals. *2022 IEEE Sensors.* doi:[10.1109/SENSORS52175.2022.9967075](https://doi.org/10.1109/sensors52175.2022.9967075). **Contrast only** with Oralable extraoral temporalis — not product continuity.

---

*Do not attach full PDFs to Nabavi cold outreach. Use citations + this distill.*
