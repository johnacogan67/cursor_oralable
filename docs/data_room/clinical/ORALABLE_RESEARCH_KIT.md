# Oralable Research Kit — canonical definition

**As at:** 30 Oct 2026 · Pack **1.1.74** · **Gen 2 only** · BOM REV9 · PCB REV11 · **ES4L15BA1** · **LP270829** · app **4.3.3** (build **5**)  
**Status:** Beacon / Paper A definition · **5 Oralable-only kits** · **30 Sep 2026** handoff **missed** · still **gated** · no new date. Gen 1 is legacy. A bench image was written to one blank board on 29 Oct. It is not kit firmware **2.0.x**. PPG and ACC do not stream. [GEN2_WOUT_KAGA_STATUS_2026-10-30.md](../hardware/GEN2_WOUT_KAGA_STATUS_2026-10-30.md)  
**Ship gate:** Charge-to-temple on each Oralable unit (see [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md))  
**Beacon use (7 Sep meeting):** one night · clinic recharge · **no headband** · pad/tape attach · **no ANR on the patient** · evaluable **≥4 h** (6 h better) — [ED_PEDRO_MEETING_2026-09-07.md](./ED_PEDRO_MEETING_2026-09-07.md)  
**Seat QC:** Mac / iOS single Protocol A and iOS Dual (MAM) use **`couple`** — [TEMPORALIS_COLLECTION_PROTOCOL.md](../../TEMPORALIS_COLLECTION_PROTOCOL.md) · Mac Dual still IR drop ≥8% until script ports couple.  
**Sendables:** [PEDRO_STATUS_UPDATE_2026-09.md](./PEDRO_STATUS_UPDATE_2026-09.md) · [PITCH_PEDRO_ED_FF.md](../pitches/PITCH_PEDRO_ED_FF.md) · protocol [PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md) · data handoff [PAPER_A_DATA_HANDOFF_SOP.md](./PAPER_A_DATA_HANDOFF_SOP.md) · construct map [MEASUREMENT_CONSTRUCT_MAP.md](./MEASUREMENT_CONSTRUCT_MAP.md) · Pedro construct note [PEDRO_CONSTRUCT_MAP_NOTE.md](./PEDRO_CONSTRUCT_MAP_NOTE.md) / [PDF](./PEDRO_CONSTRUCT_MAP_NOTE.pdf)  
**Visual direction (locked):** [VISUAL_AND_VOICE_DIRECTION.md](../brand/VISUAL_AND_VOICE_DIRECTION.md) · hero [FIG-CO-054](../figures/FIG-CO-054-matisse-photo-dual-a-stack.png) · catalog [RESEARCH_KIT_PHOTO_SELECTION.md](./RESEARCH_KIT_PHOTO_SELECTION.md)  
**IP (pitch-safe):** US provisional **64/033,978** filed **9 Apr 2026** — *Apparatus and Method for Muscle Activity Monitoring* · Temporalis OMG / IR-DC path · **patent pending** — [IP_PORTFOLIO_STATUS.md](../governance/IP_PORTFOLIO_STATUS.md) · do **not** attach provisional PDF / claim text

**One-liner:** Beacon box = Gen 2 temple **Oralable MAM** + **iOS app** + pads/tape + charger. One-night clinic loaner. Dual A (ANR) stays Mac/eng methods — **not** a patient requirement. IEEE feasibility n≈5 (Koorosh + Pedro + Ed). Measured eng Dual A: `20260812_085110` (layout only — kits still gated).

**Sophie Warren.** Two hats, one device. She is the first patient-discovery participant (chronic TMJ) and the MBM Commercial contact for later commercial paper. Anything she is shown — her own night, a dentist report, or a later lease description — is this Gen 2 temple MAM. n≈5 Beacon nights stay unpaid.

**Commercial lock:** n≈5 nights are **unpaid**. Do not put a processing fee in Beacon sendables. Later Path 0 RKaaS (company-owned kits, draft €100/report): [GTM_ONE_PAGE.md](../governance/GTM_ONE_PAGE.md).

---

## 1. BOM

### Beacon box (n≈5 — what ships)

| Item | Spec / role |
|------|-------------|
| **Oralable Gen 2 MAM** | BOM REV9 · PCB REV11 · ES4L15BA1 · cell **LP270829** 35 mAh · temple PPG (R/G/IR) + ACC. Bench image 29 Oct is not the kit. Sensors do not stream yet. |
| **Charge case** (or research dock) | USB-C · **clinic** charge before each night · line the clip coil up with the case coil · magnetic dock is a possible later case · **not** Qi / MagSafe · patient does not take the charger home |
| **ECG / Red Dot pads** (or equivalent) + **tape** | Temple attach — Ed: AccuPebble/Sunrise class or ECG pads · **no headband**. Printed ANR-class shell (Formlabs): [ATTACHMENT.md](../../ATTACHMENT.md) — charge bare MAM → seat in housing → one night → return → remove → recharge (§0b). Not a Beacon BOM change until charge-to-temple |
| **iPhone + Oralable TestFlight** | App **4.3.3** build **5+** · Protocol A Setup (**couple** seat QC) · 4 h+ night |
| **Quick start** | [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) — first worn session = Phase 0 vitals |

### Not in the Beacon box

| Item | Where it lives |
|------|----------------|
| **ANR M40** | John lab Dual A only — Ed does not require patients to combine ANR + MAM |
| **Headband** (Ridge etc.) | Out. Ed rejected it |
| **Dual A cue card** | John lab only |
| Oralable for Dentists, CloudKit share, FDA/CE claims | Product — out of kit |

![FIG-CO-016 Research Kit flat-lay](../figures/FIG-CO-016-research-kit-flatlay.png)

*Figure FIG-CO-016 — Engineering flat-lay (includes headband and ANR from the Dual A photo set). Beacon ship is Oralable + case + pads — not this full Dual A stack. Photo selection: [RESEARCH_KIT_PHOTO_SELECTION.md](./RESEARCH_KIT_PHOTO_SELECTION.md).*

![FIG-CO-013 Charge dock](../figures/FIG-CO-013-research-charge-dock.png)

*Figure FIG-CO-013 — Oralable research module in charge dock (pre-deploy). Clinic charges here; patient does not.*

---

## 2. Collection modes

| Mode | Where | Duration | Purpose |
|------|-------|----------|---------|
| **Core vitals** | iOS Oralable | ≥5–10 min | Setup, comfort, HR/SpO₂ QC |
| **Protocol A Setup** | iOS Oralable (default gate ON) | ~6 min cues | One-time training; MAM **`couple`** seat QC (not GATT worn) |
| **Beacon night** | iOS Oralable · clinic loaner | **≥4 h** (6 h better) · **one night** | Feasibility wear/QC + hypnogram; clinic recharge after return |
| **Arm P (Pedro)** | iOS Oralable | Same night or 1–2 h window | Oxygen burden ± MAD (not AHI) |
| **Hypnogram figure (stretch)** | iOS Oralable | **≥6 h** | Replace FIG-CO-025; band recalibration |
| **Dual Protocol A** | **Mac** (John) | ~6 min cues | Labeled Oralable + ANR — **not patients**; preflight = EMG ≥70 + **IR drop ≥8%** |
| **Dual Protocol A (iOS)** | `showDualProtocolA` (**default OFF**) | ~6 min | Lab/TestFlight only — not Beacon Day-1 |
| **Concordance post** | Mac | offline | `align_anr_oralable_concordance.py` → nest + **`session.edf`**; SASHB ≠ Azarbarzin HB; nest ≠ AHI |

**Seat QC split (do not mix):** Mac/iOS **single** Protocol A and **iOS Dual (MAM)** use **`couple`**. **Mac Dual** still uses IR optical clench-drop until the Mac dual script ports `CouplingGate`. Detail: [TEMPORALIS_COLLECTION_PROTOCOL.md](../../TEMPORALIS_COLLECTION_PROTOCOL.md) · [VERSION_ALIGNMENT.md](../VERSION_ALIGNMENT.md).

---

## 2b. Dual A wear stack (methods only — not Beacon patients)

US provisional **64/033,978** (filed 9 Apr 2026) describes a **temporalis optical sensor** for **OMG / IR-DC** hemodynamic occlusion (not EMG). Coupling, pressure, and alignment matter for the optical data. **ANR is the electrical comparator, not the invention.** Beacon feasibility does **not** stack ANR.

**Locate point (Beacon and Dual A):** Clench. Find the **anterior temporalis** bulge between the eyebrow tail and the ear. Seat the Oralable optical window on that peak. Placement: [TEMPORALIS_ANATOMY_AND_PLACEMENT.md](./TEMPORALIS_ANATOMY_AND_PLACEMENT.md).

**Beacon attach:**

```text
Clinic charge on charge case (coils aligned)
  → Skin / temporalis
  → Oralable PPG window (optical in)
  → ECG / Red Dot pads or tape (no headband)
  → One night → return → clinic recharge
```

**Lab Dual A layer cake (John only):**

```text
Skin / temporalis
  → Oralable PPG window
  → Kapton (when ANR is added)
  → ANR M40 on Red Dots — long axis VERTICAL
```

| Step | Figure |
|------|--------|
| Seat Oralable + silicone | ![FIG-CO-032](../figures/FIG-CO-032-oralable-silicone-temple.png) |
| ANR vertical (lab Dual A) | ![FIG-CO-031](../figures/FIG-CO-031-dual-a-anr-vertical-temple.png) |
| Headband | **Do not use for Beacon** (FIG-CO-033 is Dual A photo history only) |

---

## 3. Competitor / reference landscape (outputs compared, not replaced)

| Rank | Device | Job | Oralable relationship |
|------|--------|-----|------------------------|
| 1 | **AcuPebble** | Pedro’s OSA HSAT / **AHI** | Nest alongside Arm P — Oralable ≠ AHI clone ([ACUPEBBLE_VS_ORALABLE_ANR.md](../bookmarks/ACUPEBBLE_VS_ORALABLE_ANR.md)). Attach class Ed wants for MAM. |
| 2 | **Sunrise** | Mandibular motion (chin) | Form-factor peer (adhesive, one night) — not IR-DC PPG ([ORALABLE_MARKET_LANDSCAPE.md](../bookmarks/ORALABLE_MARKET_LANDSCAPE.md) §4c) |
| 3 | **ANR M40** | Research temporalis sEMG | **Lab Dual A comparator** — not in Beacon box ([ANR_M40_CONCORDANCE.md](./ANR_M40_CONCORDANCE.md)) |
| 4 | **GrindCare** | Anterior temporalis **sEMG** | Peer class at same muscle site; Oralable is **optical** OMG |
| 5 | **Bruxoff** | Masseter sEMG + ECG | Ambulatory EMG **reference** when available — not PSG gold |
| 6 | **Happy Ring** | Finger **hAHI** HSAT | Same AHI-class shelf as AcuPebble — not jaw |

**Claim discipline:** No AHI equivalence; no “superior to Bruxoff”; Dual A concordance = descriptive; PSG-AV remains diagnostic gold standard for SB.

---

## 4. Paper A / IEEE path

- Protocol: [PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md)  
- Draft: [PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md](./PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md)  
- Collab: [COLLAB_NABAVI_MCGILL.md](./COLLAB_NABAVI_MCGILL.md) · n≈5 Beacon · Koorosh methods · Pedro Arm P · Ed wear/packaging  

---

## 5. Ship plan — 5 Oralable-only kits (30 Sep 2026 missed)

See § Research Kit ship in [PILOT_DRY_RUN_CHECKLIST.md](./PILOT_DRY_RUN_CHECKLIST.md).

| Gate | Owner |
|------|--------|
| Charge-to-temple closed on pilot, then each of 5 units | John |
| Flash/verify 5× FW **1.0.85** · pack cases + clips + pads | John |
| TestFlight invite Pedro (+ Ed) | John |
| Print quick start (no Dual A cue card in Beacon box) | John |
| Dry-run one kit: clinic charge → pad seat → HR/SpO₂ | John |
| Hand off / ship. **30 Sep 2026** **missed**. No new date. | John → Ed/Pedro |
| Ethics + first ≥4 h night calendar | Ed + Pedro |

ANR allocation is **not** a ship gate.

---

## 6. Related scripts (Mac)

| Script | Role |
|--------|------|
| `scripts/run_protocol_a_session.py` | Oralable-only Protocol A |
| `scripts/run_anr_emg_session.py` | ANR-only EMG log (lab) |
| `scripts/run_dual_protocol_a_session.py` | Dual A — John lab, not Beacon patients |
| `scripts/align_anr_oralable_concordance.py` | Align @ 50 Hz + nest + research EDF+ |
| `scripts/generate_overnight_night_report.py` | Night report pack |

**Engineering precursor (not kit *N*):** Mac Dual A `20260812_085110` — concordance + SpO₂/SASHB + EDF; hypnogram layout at `plots/overnight_report/TEMPORALIS_20260812_085110_dualA/` (~6 min).

---

*Canonical kit definition. Day-1 vitals handout: [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md).*
