# Pilot dry-run checklist (John — before Ed/Pedro handoff)

> **Active path (Aug 2026): Phase 0 Vitals** on each **Oralable Research Kit** — temple HR/SpO₂.  
> Kit program: [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) · **5 kits → Pedro by 31 Aug 2026**.  
> Protocol B / cheek muscle steps below are **Phase 1+ only** (do after Phase 0 gates pass).

**Goal (Phase 0):** Flash Gen1 kit, temple vitals smoke, honest device state — **ship gate before Pedro handoff** (kits **gated** as at 7 Aug 2026).  
**Goal (Research Kit):** 5× (Oralable + case + ANR M40 + TestFlight + cue card) after charge-to-temple on each Oralable unit.  
**Goal (Phase 1+):** Full Protocol B and `self_validate.py` on the same Gen1 hardware (BOM REV8 / REV10).

**Ship gate:** Oralable-case charge to temple-ready SOC (≥50%) + short worn HR/SpO₂ without brownout. See [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) § Pilot ship status.

**Hardware identity:** Gen1 · **BOM REV8** · PCB **REV10** · Kaga **ES2832AA2** · FW **1.0.84** · Oralable magnetic case (not Qi).  
See [PRODUCT_ROADMAP.md](../PRODUCT_ROADMAP.md) · [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) · [FIRMWARE_1.0.84_FLASH.md](../firmware/FIRMWARE_1.0.84_FLASH.md) · prior [FIRMWARE_1.0.82_FLASH.md](../firmware/FIRMWARE_1.0.82_FLASH.md) · [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md)

**Time (Phase 0):** ~45 min · **Time (Phase 1+ Protocol B):** ~90 min

Related: [PEDRO_STATUS_UPDATE_2026-08.md](./PEDRO_STATUS_UPDATE_2026-08.md) · [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) · [VITALS_PILOT_TEST_PLAN.md](./VITALS_PILOT_TEST_PLAN.md) · [PILOT_PROTOCOL_ED_PEDRO.md](./PILOT_PROTOCOL_ED_PEDRO.md) *(deferred)* · [../FIGURES.md](../FIGURES.md) · **App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

![FIG-CO-022 Charge to temple](../figures/FIG-CO-022-pilot-charge-to-temple.svg)

*Figure FIG-CO-022 — Charge-to-temple pilot flow (placeholder).*

---

**Strategy stack:** Stage A wellness wearable first; Stage B medical later. New US patent embodiment. Ed/Pedro use the patient app only.  
**Cost & timeline (planning):** [COST_AND_TIMELINE.md](../governance/COST_AND_TIMELINE.md) — Phase 0 now–Sep 2026; Phase 1+ Q4’26–Q1’27; Gen2 parallel; Stage B H2’27–2028. Mid Stage A ~€200–250k; Stage A+Gen2 ~€350–450k; through Stage B ~€0.8–1.0M (ranges — not a budget).

## A. Hardware & firmware

| Step | Action | Pass |
|------|--------|------|
| A1 | Connect J-Link to pcb00003 | ☐ |
| A2 | Flash **1.0.84** per [FIRMWARE_1.0.84_FLASH.md](../firmware/FIRMWARE_1.0.84_FLASH.md) (or prior [1.0.82](../firmware/FIRMWARE_1.0.82_FLASH.md)) | ☐ |
| A3 | nRF Connect: read `006` → **1.0.84** | ☐ |
| A4 | On Oralable case: `on_dock=1` + `charge_active=1` (blink); later taper → `charge_active=0`. LED: **solid green** | ☐ |
| A5 | **Oralable case** + charged clip (>30% SOC) | ☐ |

```bash
cd /Users/johnacogan67/work/oralable_nrf
./scripts/flash_and_rtt.sh --no-build --hex build_pcb00003/merged.hex   # 1.0.84
# J-Link SNR default: 1050090445 — export JLINK_SNR=... if different
```

---

## B. iOS build (TestFlight / Ad Hoc)

| Step | Action | Pass |
|------|--------|------|
| B1 | Confirm OralableApp **4.3.3** build **5+** (`CURRENT_PROJECT_VERSION`) | ☐ |
| B2 | Archive **OralableApp** scheme, **Release**, device **Any iOS** | ☐ |
| B3 | Distribute → TestFlight (internal) or Ad Hoc for Ed/Pedro devices | ☐ |
| B4 | Install on your iPhone; confirm app opens past sign-in + fit gate | ☐ |

**Pilot build must include:** FW gate min **1.0.63** · recommend **1.0.84** · Protocol A Setup gate · Vitals phase · **Device LED** mirror (STAT flash/taper) · placement picker with **Automatic** · nRF CSV export.

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
| C2 | Wait for optional solid green (STAT taper, FW ≥ 1.0.72) while still on case — Dock on, Charging off | ☐ |
| C3 | Settings → placement **Worn** (manual mode 3) | ☐ |
| C4 | Mount on **temple** → Connect | ☐ |
| C5 | Dashboard HR and/or SpO₂ live ≥ 2 min, no disconnect in first 5 min | ☐ |
| C6 | Settings → placement shows **Worn** (temple vitals — not cheek muscle UI) | ☐ |

**Fail actions:**
- Disconnect in first 5 s → confirm FW **1.0.84**; do not enable `00A` in Developer Settings.
- Flat dashboard → placement **Worn**; reconnect; check temple contact / quality.
- Dock stays 0 on case → confirm Oralable case (not MagSafe); reflash **1.0.84**.

---

## C′. Protocol B prepare — Phase 1+ only (skip for Phase 0 ship)

| Step | Action | Pass |
|------|--------|------|
| C′1 | Share → **Prepare Protocol B session** | ☐ |
| C′2 | Mount on cheek / temporalis per Protocol B | ☐ |

---

## D. Protocol B mini-run — **Phase 1+ only** (~10 min smoke; full ~45 min)

> Skip this section for Phase 0 Ed/Pedro vitals kits. Use [VITALS_PILOT_TEST_PLAN.md](./VITALS_PILOT_TEST_PLAN.md) instead.

Use full phase table in [TEMPORALIS_COLLECTION_PROTOCOL.md § Protocol B](../TEMPORALIS_COLLECTION_PROTOCOL.md#protocol-b--edpedro-structured-validation).

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
| E5b | iOS overnight **bands** unlock at **≥ 1 h**; ideal overnight / Paper A Arm E/J still **≥ 6 h** (goal 8 h); Protocol A/B minutes ≠ sleep | ☐ |
| E5c | Optional research: Developer Settings → Dual Protocol A ON (default OFF); otherwise sleep path | ☐ |

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
| Pilot clip flashed **1.0.84** + charge-to-temple closed | ☐ |
| TestFlight invite (**4.3.3** build **5**, 1.0.84-aware) sent to Ed + Pedro | ☐ |
| Quick start printed / emailed | ☐ |
| First kit packed for smoke ship | ☐ |

**After pass:** notify Pedro/Ed with TestFlight link + [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) · status [PEDRO_STATUS_UPDATE_2026-08.md](./PEDRO_STATUS_UPDATE_2026-08.md).

---

## G. Research Kit ship — 5 kits to Pedro by 31 Aug 2026

Canonical BOM: [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md). Hard gate: charge-to-temple on **each** Oralable unit before field *N*.

| Step | Action | Pass |
|------|--------|------|
| G1 | Close charge-to-temple on pilot unit (FW **1.0.84**) | ☐ |
| G2 | Flash/verify **5×** Oralable → GATT `006` = **1.0.84** | ☐ |
| G3 | Pack **5×** magnetic cases + clips (kit IDs RK01–RK05) | ☐ |
| G4 | Allocate **5× ANR M40** (if shortfall: document count + Oralable-first ship) | ☐ |
| G5 | Print Dual A cue card + quick start ×5 | ☐ |
| G6 | TestFlight invite Pedro (+ Ed); research/long-wear path confirmed | ☐ |
| G7 | Dry-run **one** full kit: charge → temple → HR/SpO₂ → Share CSV | ☐ |
| G8 | Optional: Mac Dual A dry-run (`run_dual_protocol_a_session.py`) on one kit | ☐ |
| G9 | Hand off / ship **5 kits by 31 Aug 2026** | ☐ |
| G10 | Calendar Arm P + ethics lock with Pedro | ☐ |

| Kit ID | Oralable serial / `005` | ANR serial | Charge gate | Packed |
|--------|-------------------------|------------|-------------|--------|
| RK01 | | | ☐ | ☐ |
| RK02 | | | ☐ | ☐ |
| RK03 | | | ☐ | ☐ |
| RK04 | | | ☐ | ☐ |
| RK05 | | | ☐ | ☐ |

**ANR count available:** ______ / 5 · **Notes:** ______

**After handoff:** Pedro dry-run, then Arm P window, then Paper A CRF rows, then Dual A Mac sessions for concordance figures. iOS Dual A overnight hardening is follow-on work (not a ship blocker).

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
