# Pilot protocol — Ed & Pedro (Phase 1+ muscle)

> **DEFERRED (July 2026).** Active Ed/Pedro work is **Phase 0 Vitals** — temple HR/SpO₂.  
> Printable handout: [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) · Test plan: [VITALS_PILOT_TEST_PLAN.md](./VITALS_PILOT_TEST_PLAN.md) · Roadmap: [PRODUCT_ROADMAP.md](../PRODUCT_ROADMAP.md)

**Do not** run this protocol as the current pilot until Phase 0 vitals gates pass.

**Program:** Oralable Point A operating evidence · **Version:** 1.1.0 · **July 2026**  
**Scope:** Phase 1+ muscle / IR-DC / Protocol B on **same Gen1 hardware** (BOM REV8 · PCB REV10 · ES2832AA2 · FW ≥ **1.0.70**)  
**Protocol leads:** **[Dr Edward Owens](https://beaconconsultantssleephealthclinic.ie/team-member/dr-edward-owens/)** & **[Dr Pedro Mayoral Sanz](https://beaconconsultantssleephealthclinic.ie/team-member/dr-pedro/)** (Beacon · structured validation, protocol fidelity)  
**Sponsor:** JAC Dental Ltd

**Related:** [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) (**Phase 0 one-pager**) · [PILOT_DRY_RUN_CHECKLIST.md](./PILOT_DRY_RUN_CHECKLIST.md) · [../TEMPORALIS_COLLECTION_PROTOCOL.md](../TEMPORALIS_COLLECTION_PROTOCOL.md) (**Protocol B**) · [../CLINICAL_VALIDATION.md](../CLINICAL_VALIDATION.md) · [ORALABLE_FTS_36MO.md](./ORALABLE_FTS_36MO.md) · **App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

**Phase timings:** Same as Protocol B in `TEMPORALIS_COLLECTION_PROTOCOL.md` (do not use Protocol A training timings).

---

**Strategy stack:** Stage A wellness wearable → Stage B medical (later) · new US patent embodiment · Ed/Pedro = patient app only.  
**Cost & timeline (planning):** [COST_AND_TIMELINE.md](./COST_AND_TIMELINE.md) — Phase 0 now–Sep 2026; Phase 1+ Q4’26–Q1’27; Gen2 parallel; Stage B H2’27–2028. Mid Stage A ~€200–250k; Stage A+Gen2 ~€350–450k; through Stage B ~€0.8–1.0M (ranges — not a budget).

## 1. Objectives

| Objective | Ken / investor metric |
|-----------|----------------------|
| Demonstrate **working prototype** with real user/professional workflow | Technology & Product |
| Produce **operating evidence** (logs, exports, pass/fail gates) | All sub-dimensions |
| Repeat **Ed/Pedro structured protocol** on **shipping Gen1 hardware** (BOM REV8 · REV10 · ES2832AA2 · FW ≥ **1.0.70**) | Clinical accuracy path (after Phase 0) |
| Validate **iOS app** path: pair → auto-record → export → optional dentist handshake | FTS workflow |
| Path to **5 users** (Ed, Pedro + 3 recruits) for traction narrative | User Traction |

**Not in Phase 1:** Pivotal EMG concordance study (Phase 2 / 20-device program).

---

## 2. Roles

| Role | Person | Responsibility |
|------|--------|----------------|
| **Protocol lead / clinical validator** | Dr Edward Owens (Beacon) | Session oversight, protocol timing, sign-off on fidelity reports |
| **Protocol lead / clinical validator** | Dr Pedro Mayoral Sanz (Beacon) | Same; joint sign-off on pass/fail matrix |
| **Technical operator** | John Cogan (or delegate) | Device prep, flash FW **≥ 1.0.36** (bench build → [architecture §3](../ORALABLE_SYSTEM_ARCHITECTURE.md#3-validation-status-matrix-where-we-are)), app build, log export |
| **Participants (Phase 1A)** | Ed, Pedro | Each completes structured sessions on themselves |
| **Participants (Phase 1B)** | +3 recruits | After Ed/Pedro gate passes — dentist-referred or research volunteers |

---

## 3. Hardware & software baseline

| Item | Requirement |
|------|-------------|
| Device | Oralable REV10 / pcb00003 cheek clip |
| Firmware | **≥ 1.0.70** (STAT blink dock; `FirmwareGate` min 1.0.63 / recommend 1.0.70) |
| iOS app | Oralable patient TestFlight (vitals phase; Automatic placement OK on 1.0.70) |
| Coupling | Cheek **R_G_IR**; IR-DC raw **10M–70M** (`IR_DC_ADC_FORMAT.md`) |
| Recording | **Automatic** on connect; worn-gated streaming |
| Dentist app | Optional smoke test: share code → participant visible |

**Pre-session checklist:** `oralable_nrf/docs/DEVELOPMENT.md` manual smoke (on-body notifies, off-body silent).

---

## 4. Session types

### 4.1 Structured lab session (~45 min)

**Purpose:** Repeat gold-standard phases for algorithm gates (sync, tonic, phasic, swallow, apnea, speech).

**Mounting:** Cheek / masseter per REV10 fit guide (`TemporalisFitGuideView` / calibration flow).  
**Anchor:** **T=0 = 1st 3-tap sync** (Protocol B — not Protocol A training timings).

**Phase table (elapsed seconds, pass criteria):** [TEMPORALIS_COLLECTION_PROTOCOL.md § Protocol B](../TEMPORALIS_COLLECTION_PROTOCOL.md#protocol-b--edpedro-structured-validation).

**Minimum per protocol lead:** **2 structured sessions** each (Ed, Pedro) on separate days.

### 4.2 Overnight session (optional but recommended)

**Purpose:** Wellness metrics — TFI, SASHB, hourly rollups, battery life, night-report graphics.

| Item | Target |
|------|--------|
| Duration | **≥ 6 h worn** (goal **8 h**) — minimum for an *evaluable* overnight; under 6 h = debug only |
| Hourly correlation | Prefer **≥ 3 h** continuous so smoking-gun r (hourly SASHB vs rescue) can compute |
| Coupling | IR-DC stable in range; worn gate OK |
| Export (iOS) | Share → research CSV + **Clinical Temporalis PDF** (bout hypnogram, dual-rail, event CSV) |
| Export (Mac) | `generate_clinical_report.py` → also `plots/overnight_report/<session>/night_report.pdf` |

**Minimum:** **1 overnight (≥6 h)** each for Ed and Pedro before expanding to Phase 1B.

**Not overnight:** Protocol A (~6 min) and Protocol B (~4.5 min) are structured locks only — see [TEMPORALIS_COLLECTION_PROTOCOL.md](../TEMPORALIS_COLLECTION_PROTOCOL.md) § Overnight sleep session.

**Review UX:** lead with **state hypnogram** (in-app morning card / Share preview + PDF) + provisional **Low / Moderate / High** bands (TFI, SASHB/h, rescue/h) — [OVERNIGHT_NIGHT_REPORT.md](../OVERNIGHT_NIGHT_REPORT.md) · FIG-CO-025. Recalibrate cutoffs from these pilot nights.

### 4.3 Professional workflow smoke (once)

1. Patient app: complete recording → generate **share code** → upload to CloudKit (when prod ready).  
2. Dentist app: enter code → participant appears → view dashboard / historical.  
3. Export **ProfessionalHandshakeExport** JSON or CSV import fallback.

---

## 5. Data capture

| Artifact | Path / tool |
|----------|-------------|
| BLE log (reference) | App **Share → Export Protocol B validation log** (nRF format CSV) **or** nRF Connect export |
| App export | ShareView research CSV |
| Validation | `python -m src.validation.self_validate <log> --segment-from 1` |
| Dashboard plot | `validation_dashboard` with `segment_from_sync=1` |
| Clinical summary | `clinical_summary` PDF |
| Investor bundle | Update `docs/CLINICAL_VALIDATION.md` with new run section |

**Naming convention:**

```
Oralable_PILOT_Ed_YYYYMMDD_sessionN.txt
Oralable_PILOT_Pedro_YYYYMMDD_sessionN.txt
```

Store under `cursor_oralable/data/raw/pilot_ed_pedro/` (create on first run).

---

## 6. Pass / fail gates (session level)

| Gate | Threshold |
|------|-----------|
| Sync anchor | 1st 3-tap detected |
| Sensor coupling | IR-DC median in cheek OK band |
| Green SNR | ≥ 10 dB (warn if lower) |
| Swallow FP | 0 |
| Speech FP | 0 |
| Phasic | Jitter RMS present |
| Apnea rescue | Detected per cheek tier config |
| App BLE | Connect ≥ 2 min on-body without drop (structured session) |

**Phase 1A complete when:** Ed and Pedro each pass **2 structured sessions** + **1 overnight** (or documented waiver with reason).

---

## 7. Phase 1B — expand to 5 users

| Step | Action |
|------|--------|
| 1 | Ed/Pedro sign Phase 1A summary |
| 2 | Recruit **3** additional participants (consent + wellness disclaimer) |
| 3 | Each: app onboarding + **1 overnight** minimum |
| 4 | Aggregate: nights recorded, export success rate, share connections |

**Ken alignment:** First **customer validation** evidence for traction slide.

---

## 8. Consent & wellness disclaimer

- Participants acknowledge device is **not a medical device** at this stage.  
- Data used for **product validation and investor diligence**.  
- Optional: `PilotDataManager` + `Anonymizer` if IDs are published.  
- Ed/Pedro as protocol leads document consent for recruits.

---

## 9. Timeline

| Week | Activity |
|------|----------|
| **W1** | FW flash + app RC on 2 clips; Ed/Pedro structured session #1 each |
| **W2** | Structured session #2; overnight #1 each; validation reports |
| **W3** | Professional workflow smoke; Phase 1A sign-off |
| **W4–6** | Recruit 3 users; Phase 1B overnights; update data room |

---

## 10. Deliverables (data room)

| Deliverable | Location |
|-------------|----------|
| Validator one-pager | [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) |
| Phase 1A fidelity reports | `CLINICAL_VALIDATION.md` § Pilot Ed/Pedro 2026 |
| Plots | `data/plots/pilot_ed_pedro/` — promote embeds via [../FIGURES.md](../FIGURES.md) |
| Session log index | `data/raw/pilot_ed_pedro/README.md` |
| Sign-off sheet | `data_room/PILOT_SIGNOFF_ED_PEDRO.pdf` (external) |

---

## 11. Validation commands (reference)

```bash
cd cursor_oralable
python -m src.validation.self_validate \
  data/raw/pilot_ed_pedro/Oralable_PILOT_Ed_YYYYMMDD_1.txt \
  --segment-from 1 -o data/plots/pilot_ed_pedro/ed_from_sync1.png

python -c "
from pathlib import Path
from src.validation_dashboard import run_validation_dashboard
run_validation_dashboard(
    log_path=Path('data/raw/pilot_ed_pedro/Oralable_PILOT_Ed_YYYYMMDD_1.txt'),
    segment_from_sync=1,
    output_path=Path('data/plots/pilot_ed_pedro/ed_dashboard.png'),
)
"
```

---

## 12. Sign-off

| Role | Name | Phase 1A | Date |
|------|------|----------|------|
| Protocol lead | Dr Edward Owens | ☐ | |
| Protocol lead | Dr Pedro Mayoral Sanz | ☐ | |
| Technical | John Cogan | ☐ | |

*After sign-off, notify Balance Points / Ken with updated data room index.*
