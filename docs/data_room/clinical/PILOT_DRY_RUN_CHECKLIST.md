# Pilot dry-run checklist (John — before Ed/Pedro handoff)

> **Active path (3 Oct 2026): Phase 0 Vitals** on each **Oralable Research Kit** — temple HR/SpO₂.  
> Kit program: [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) · **5 Oralable-only kits**. **30 Sep 2026** handoff **missed**. Still **gated**. No new date.  
> Protocol B / cheek muscle steps below are **Phase 1+ only** (do after Phase 0 gates pass).

**Goal (Phase 0):** No Gen2 kit may enter a temple-vitals or Beacon handoff until the separately reviewed G2-P0…G2-P6 evidence gate is complete. Kits stay **gated**. The historical bench note that one blank board was programmed while PPG and ACC did not stream is not acceptance evidence. [GEN2_WOUT_KAGA_STATUS_2026-10-30.md](../hardware/GEN2_WOUT_KAGA_STATUS_2026-10-30.md).
**Goal (Research Kit):** 5× (Oralable + charger + pads/tape + TestFlight + quick start) after charge-to-temple. **No ANR. No headband.**  
**Goal (Phase 1+):** Full Protocol B and `self_validate.py` only after the same Gen2 hardware satisfies its evidence gate (BOM REV9 / REV11).

**Future kit gate:** After the Gen2 program closes, the candidate must separately demonstrate Oralable-case charge to temple-ready SOC (≥50%) plus short worn HR/SpO₂ without brownout. See [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) § Pilot ship status.

**Hardware identity:** Gen2 · **BOM REV9** · PCB **REV11** · Kaga **ES4L15BA1** · cell **LP270829** · FW **2.0.x** candidate · Oralable charger (coil align now; magnetic dock later; not Qi). Controlled hardware work follows `oralable_nrf/docs/GEN2_PROGRAM_GATE.md`; the build helpers intentionally do not program hardware.
See [PRODUCT_ROADMAP.md](../../PRODUCT_ROADMAP.md) · [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) · [FIRMWARE_1.0.85_FLASH.md](../firmware/FIRMWARE_1.0.85_FLASH.md) · prior [FIRMWARE_1.0.84_FLASH.md](../firmware/FIRMWARE_1.0.84_FLASH.md) · [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md)

**Time (Phase 0):** ~45 min · **Time (Phase 1+ Protocol B):** ~90 min

Related: [PEDRO_STATUS_UPDATE_2026-09.md](./PEDRO_STATUS_UPDATE_2026-09.md) · [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) · [VITALS_PILOT_TEST_PLAN.md](./VITALS_PILOT_TEST_PLAN.md) · [PILOT_PROTOCOL_ED_PEDRO.md](../archive/PILOT_PROTOCOL_ED_PEDRO.md) *(deferred)* · [../FIGURES.md](../../FIGURES.md) · **App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

![FIG-CO-022 Charge to temple](../figures/FIG-CO-022-pilot-charge-to-temple.svg)

*Figure FIG-CO-022 — Charge-to-temple pilot flow (placeholder).*

---

**Strategy stack:** Stage A wellness wearable first; Stage B medical later. New US patent embodiment. Ed/Pedro use the patient app only.  
**Cost & timeline (planning):** [COST_AND_TIMELINE.md](../governance/COST_AND_TIMELINE.md) — Phase 0 continues past Sep 2026 (30 Sep handoff missed); Phase 1+ Q4’26–Q1’27; Gen2 parallel; Stage B H2’27–2028. Mid Stage A ~€200–250k; Stage A+Gen2 ~€350–450k; through Stage B ~€0.8–1.0M (ranges — not a budget).

## A. Hardware & firmware

| Step | Action | Pass |
|------|--------|------|
| A1 | Connect the nRF54L15 DK (J-Link **1057765387**) to the Gen 2 MAM. Tag-Connect on J1. One power source. | ☐ |
| A2 | Pins in the board file are SDA P1.07, SCL P1.03, SENS_EN P1.08, BATEN P1.04. A merged hex was written on 29 Oct. Do not treat that image as the kit. | ☐ |
| A3 | nRF Connect: name **Oralable**. Read `006`. Do not call it **2.0.x** until that read says so. PPG and ACC notifies are not a pass until samples arrive. | ☐ |
| A4 | On the Oralable charger: `on_dock=1` + `charge_active=1` (blink); later taper → `charge_active=0`. LED behavior to port from Gen 1: **flash green** below hold; **rare green pulse** in hold. Not solid green. Not red. | ☐ |
| A5 | **Oralable case** + charged clip (>30% SOC) | ☐ |

Programmer for the 29 Oct write was the nRF54L15 DK, J-Link **1057765387**, not **1050090445**. Image: `artifacts/oralable_gen2_ota_merged.hex`. Unplug Tag-Connect after reset. The nRF5340 DK serial is not this programmer.

---

## B. iOS build (TestFlight / Ad Hoc)

| Step | Action | Pass |
|------|--------|------|
| B1 | Confirm OralableApp **4.3.3** build **5+** (`CURRENT_PROJECT_VERSION`) | ☐ |
| B2 | Archive **OralableApp** scheme, **Release**, device **Any iOS** | ☐ |
| B3 | Distribute → TestFlight (internal) or Ad Hoc for Ed/Pedro devices | ☐ |
| B4 | Install on your iPhone; confirm app opens past sign-in + fit gate | ☐ |

**Pilot build must include:** FW gate min **1.0.63** · recommend **1.0.85** · Protocol A Setup (**couple** seat QC) · Vitals phase · **Device LED** mirror (STAT flash / hold pulse) · placement picker with **Automatic** · nRF CSV export.

```bash
cd /Users/johnacogan67/work/oralable_swift/OralableApp
# Release compile check (no signing):
xcodebuild -scheme OralableApp -destination 'generic/platform=iOS' \
  -configuration Release build
```

---

## C. Connect smoke (~5 min) — Phase 0 Vitals

| Step | Action | Pass |
|------|--------|------|
| C1 | Placement **Automatic** (or **On wireless charger**) → seat in **Oralable case** → flash green (FW ≥ 1.0.72) / Charging chip | ☐ |
| C2 | Still on case after taper: Dock on, Charging off. **1.0.85** shows a **rare green pulse** in hold, not solid green. (Archived **1.0.84** used solid green at taper.) | ☐ |
| C3 | Settings → placement **Worn** (manual mode 3) | ☐ |
| C4 | Mount on **temple** → Connect | ☐ |
| C5 | Dashboard HR and/or SpO₂ live ≥ 2 min, no disconnect in first 5 min | ☐ |
| C6 | Settings → placement shows **Worn** (temple vitals — not cheek muscle UI) | ☐ |

**Fail actions:**
- Disconnect in first 5 s → read GATT `006` on this Gen 2 board. Do not load Gen 1 firmware. Do not enable `00A` in Developer Settings.
- Flat dashboard → PPG and ACC are not streaming on the bench image. Placement will not create samples.
- Dock stays 0 on case → confirm the Oralable case (not MagSafe). Do not reflash **1.0.85**.

---

## C′. Protocol B prepare — Phase 1+ only (skip for Phase 0 ship)

| Step | Action | Pass |
|------|--------|------|
| C′1 | Share → **Prepare Protocol B session** | ☐ |
| C′2 | Mount on cheek / temporalis per Protocol B | ☐ |

---

## D. Protocol B mini-run — **Phase 1+ only** (~10 min smoke; full ~45 min)

> Skip this section for Phase 0. Phase 0 is section A plus the [30 Oct bench note](../hardware/GEN2_WOUT_KAGA_STATUS_2026-10-30.md). The old Gen 1 vitals plan is archived.

Use full phase table in [TEMPORALIS_COLLECTION_PROTOCOL.md § Protocol B](../../TEMPORALIS_COLLECTION_PROTOCOL.md#protocol-b--edpedro-structured-validation).

| Step | Action | Pass |
|------|--------|------|
| D1 | **T=0:** 3-tap sync in first 5 s after stream starts | ☐ |
| D2 | Max tonic clench ~30–45 s | ☐ |
| D3 | Phasic grinding ~60–105 s | ☐ |
| D4 | Swallow control ~120–135 s | ☐ |
| D5 | Stay connected ≥ 2 min continuous on-body | ☐ |

**Do not** tap **Scan** on Devices during the session.

---

## E. Export & validation

**Phase 0:** export session CSV / vitals log; confirm HR or SpO₂ present.  
**Phase 1+:** Share → **Export Protocol B validation log** → `self_validate.py` (IR-DC cheek band).

| Step | Action | Pass |
|------|--------|------|
| E1 | Export session / Protocol B log → save to Mac (AirDrop/Files) | ☐ |
| E2 | Rename: `Oralable_PILOT_John_YYYYMMDD_dryrun1.csv` | ☐ |
| E3 | Copy to `cursor_oralable/data/raw/pilot_ed_pedro/` | ☐ |
| E4 | Run validation (below) — sync detected, no hard failures | ☐ |
| E5 | Optional: Share → Clinical Temporalis PDF (bout hypnogram / dual-rail / event CSV) | ☐ |
| E5b | iOS overnight **bands** unlock at **≥ 1 h**; Beacon evaluable **≥ 4 h** (6 h better); hypnogram figure / Paper A Arm E/J still **≥ 6 h** (goal 8 h); Protocol A/B minutes ≠ sleep | ☐ |
| E5c | Optional research: Developer Settings → Dual Protocol A ON (default OFF); EMG → MAM **couple** → cues; otherwise sleep path | ☐ |

```bash
cd /Users/johnacogan67/work/cursor_oralable
mkdir -p data/raw/pilot_ed_pedro data/plots/pilot_ed_pedro

python -m src.validation.self_validate \
  data/raw/pilot_ed_pedro/Oralable_PILOT_John_YYYYMMDD_dryrun1.csv \
  --segment-from 1 \
  -o data/plots/pilot_ed_pedro/john_dryrun1.png
```

**Dry-run pass criteria (minimum):**
- 1st 3-tap sync detected
- IR-DC in cheek OK band (10M–70M raw)
- Swallow FP = 0, Speech FP = 0 (if you ran those phases)
- BLE log line count > 5,000 for a ≥10 min session

---

## F. Ship gate (legacy 2-kit smoke)

| Item | Ready |
|------|-------|
| Dry-run validation plot archived | ☐ |
| Gen 2 unit streams PPG and ACC, then charge-to-temple closes. Do not flash Gen 1 **1.0.85** onto REV11. | ☐ |
| TestFlight invite (**4.3.3** build **5**, 1.0.85-aware) sent to Ed + Pedro | ☐ |
| Quick start printed / emailed | ☐ |
| First kit packed for smoke ship | ☐ |

**After pass:** notify Pedro/Ed with TestFlight link + [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) · status [PEDRO_STATUS_UPDATE_2026-09.md](./PEDRO_STATUS_UPDATE_2026-09.md).

---

## G. Research Kit ship — 5 Oralable-only kits (30 Sep 2026 missed)

Canonical BOM: [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md). Hard gate: charge-to-temple on **each** Oralable unit before field *N*. ANR is **not** a ship gate.

| Step | Action | Pass |
|------|--------|------|
| G1 | Close charge-to-temple on a Gen 2 unit after it streams | ☐ |
| G2 | Install/verify **5×** Oralable → GATT `006` = **2.0.x** | ☐ |
| G3 | Pack **5×** charge cases + clips + pads/tape (kit IDs RK01–RK05) | ☐ |
| G4 | Pad-attach soak: Gen 2 on temple **without headband** (`couple` logged) | ☐ |
| G5 | Print quick start ×5 (no Dual A cue card in Beacon box) | ☐ |
| G6 | TestFlight invite Pedro (+ Ed); one-night path confirmed | ☐ |
| G7 | Dry-run **one** kit: clinic charge → pad seat → HR/SpO₂ | ☐ |
| G8 | Optional: Mac Dual A dry-run (John lab only — not a ship blocker) | ☐ |
| G9 | Hand off / ship **5 kits**. **30 Sep 2026** date **missed**. No replacement date in this checklist. | ☐ |
| G10 | Calendar first ≥4 h night + ethics lock with Ed + Pedro | ☐ |

| Kit ID | Oralable serial / `005` | Charge gate | Pads packed | Packed |
|--------|-------------------------|-------------|-------------|--------|
| RK01 | | ☐ | ☐ | ☐ |
| RK02 | | ☐ | ☐ | ☐ |
| RK03 | | ☐ | ☐ | ☐ |
| RK04 | | ☐ | ☐ | ☐ |
| RK05 | | ☐ | ☐ | ☐ |

**After handoff:** Beacon dry-run, then one ≥4 h night, then Paper A CRF rows. Dual A Mac sessions stay John-run.

---

## Record results

| Field | Value |
|-------|--------|
| Date | |
| Clip ID / device `005` | |
| FW `006` | |
| iOS build number | |
| Session duration | |
| `self_validate` outcome | |
| Notes | |
