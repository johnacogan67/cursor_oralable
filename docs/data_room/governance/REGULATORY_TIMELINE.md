# Regulatory timeline — wellness wearable first, medical device later

**One-page diligence summary** · **Version:** 1.2.0 · **July 2026**  
**Not legal advice** — confirm with regulatory counsel before submission.

**Related:** [ORALABLE_FTS_36MO.md](./ORALABLE_FTS_36MO.md) · [PRODUCT_ROADMAP.md](../PRODUCT_ROADMAP.md) · [IP_NORTH_STAR.md](../IP_NORTH_STAR.md) · [COST_AND_TIMELINE.md](./COST_AND_TIMELINE.md) · `oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md` §8–9 · **App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

**Strategy:** **Stage A = wellness wearable** (ship now). **Stage B = medical device** (later). Build and patent through Stage A. Clearance is a later gate.

Stage A wellness first · Stage B medical later · new US patent embodiment · Ed/Pedro = patient app only.

**Cost & timeline (planning):** [COST_AND_TIMELINE.md](./COST_AND_TIMELINE.md) — Phase 0 now–Sep 2026; Phase 1+ Q4’26–Q1’27; Gen2 parallel; Stage B H2’27–2028. Mid Stage A ~€200–250k; Stage A+Gen2 ~€350–450k; through Stage B ~€0.8–1.0M (ranges — not a budget).

---

## Current posture (today) — Stage A

| Item | Status |
|------|--------|
| **US product class** | General **wellness wearable** — no FDA clearance claimed |
| **EU** | Wellness / general health positioning — **no CE medical mark** on consumer SKU |
| **App claims** | Metadata compliance tests block "diagnose", "FDA cleared", "medical device" |
| **Software infrastructure** | `RegulatoryPackageBuilder`, ISO 14971-style risk entries, clinical PDF export |
| **Pilot evidence path** | **Phase 0** temple vitals on Gen1 → **Phase 1+** muscle IR-DC / TFI → Gen2 → **then** Stage B 510(k) target |
| **Ed/Pedro** | Patient app only; wellness wearable validation |

---

## Staged pathway

```
2026         STAGE A — WELLNESS WEARABLE (ship first)
             Phase 0 temple vitals · Phase 1+ TFI/SASHB embodiment
             iOS App Store · wellness disclaimers · patient app
             New US patent submission supported by working product
                    │
2026–2027    CLINICAL EVIDENCE (still Stage A claims)
             Ed/Pedro → expanded pilots · EMG concordance · clinical reports
                    │
2027 H1      STAGE B PRE-SUBMISSION
             Predicate analysis · study report · SaMD boundary locked
                    │
2027 H2+     STAGE B — MEDICAL DEVICE (target)
             US 510(k) monitoring indication (sleep bruxism / jaw activity)
             CE MDR Class IIa exploration (parallel)
                    │
2028+        CLEARED PRODUCT TIER
             Locked algorithms · IFU · professional CDS (label-dependent)
```

---

## Target US indication (draft — counsel review)

> The Oralable Oral Activity Monitor is intended for use by adults in the home environment to **monitor and record episodes of nocturnal jaw muscle activity consistent with sleep bruxism**. Data are reviewed in a mobile application by the user and, optionally, their dental care provider. **Not intended** to diagnose sleep disorders, replace polysomnography, or guide acute treatment without professional interpretation.

**Predicate strategy:** 510(k) substantial equivalence to **home bruxism / sleep activity monitors** (often EMG-based). Oralable differs on **optical IR-DC at extraoral temporalis** with concordance data (research comparator: ANR M40 / Paper C — [ANR_M40_CONCORDANCE.md](../clinical/ANR_M40_CONCORDANCE.md); not a consumer EMG product).

---

## CE Mark (EU) — indicative timeline

| Milestone | Target | Notes |
|-----------|--------|-------|
| MDR classification confirmation | Q1 2027 | Likely Class IIa (monitoring) — confirm with NB |
| Technical documentation | Q2–Q3 2027 | Leverage 510(k) clinical package |
| Notified body engagement | Q3 2027 | Parallel to or after FDA pre-sub |
| CE marking (goal) | 2028 | Depends on NB queue |

---

## Evidence plan (maps to pilot)

| Study | Purpose | Timing |
|-------|---------|--------|
| **Ed/Pedro pilot** | Operating evidence; protocol fidelity; false-positive gates | Q2–Q3 2026 |
| **5-user pilot** | Usage retention, export workflow, dentist handshake | Q3 2026 |
| **20-device field** | Accuracy vs structured protocol + EMG subset | Q4 2026 |
| **Pivotal / concordance** | Sensitivity/specificity vs EMG gold standard | 2027 |

Protocol: [PILOT_PROTOCOL_ED_PEDRO.md](../clinical/PILOT_PROTOCOL_ED_PEDRO.md)

---

## Lifestyle vs medical risk (Ken flag)

| Risk | Mitigation |
|------|------------|
| Wellness category insufficient for traction | Parallel **dentist channel** (handshake exports); plan clearance for monitoring claim |
| Over-claiming in marketing | Compliance tests + metadata review; separate professional copy review |
| SaMD scope creep | Lock algorithm version at submission; open research mode excluded from cleared build |

---

## Insurance & compliance gaps (open)

| Item | Status | Target |
|------|--------|--------|
| Product liability insurance | ⏳ Confirm certificate | Before scale pilot |
| HIPAA BAA | Not required for wellness CloudKit opt-in | HCP tier if needed post-clearance |
| Cybersecurity summary | Auth + encryption documented | FDA cybersecurity guidance alignment Q1 2027 |

---

## Key codebase references

- `oralable_swift/.../Regulatory/RegulatoryPackageBuilder.swift`
- `oralable_swift/.../Models/Regulatory/RegulatoryModels.swift`
- `OralableCore/.../CloudKit/ProfessionalHandshakeExport.swift`
- `cursor_oralable/docs/CLINICAL_VALIDATION.md`

*Bump version when milestones or indication language changes.*
