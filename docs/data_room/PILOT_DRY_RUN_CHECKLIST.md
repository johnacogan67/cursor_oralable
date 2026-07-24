# Pilot dry-run checklist (John — before Ed/Pedro handoff)

> **Active path (July 2026): Phase 0 Vitals** — temple HR/SpO₂.  
> Protocol B / cheek muscle steps below are **Phase 1+ only** (do after Phase 0 gates pass).

**Goal (Phase 0):** Flash Gen1 kit, temple vitals smoke, honest device state — **ship gate before Ed/Pedro handoff** (kits not yet shipped as at 22 Jul 2026).  
**Goal (Phase 1+):** Full Protocol B + `self_validate.py` on the same Gen1 hardware (BOM REV8 / REV10).

**Ship gate:** Oralable-case charge to temple-ready SOC (≥50%) + short worn HR/SpO₂ without brownout. See [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) § Pilot ship status.

**Hardware identity:** Gen1 · **BOM REV8** · PCB **REV10** · Kaga **ES2832AA2** · FW **1.0.70** · Oralable magnetic case (not Qi).  
See [PRODUCT_ROADMAP.md](../PRODUCT_ROADMAP.md) · [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) · [FIRMWARE_1.0.70_FLASH.md](./FIRMWARE_1.0.70_FLASH.md)

**Time (Phase 0):** ~45 min · **Time (Phase 1+ Protocol B):** ~90 min

Related: [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) · [VITALS_PILOT_TEST_PLAN.md](./VITALS_PILOT_TEST_PLAN.md) · [PILOT_PROTOCOL_ED_PEDRO.md](./PILOT_PROTOCOL_ED_PEDRO.md) *(deferred)*

---

**Strategy stack:** Stage A wellness wearable → Stage B medical (later) · new US patent embodiment · Ed/Pedro = patient app only.  
**Cost & timeline (planning):** [COST_AND_TIMELINE.md](./COST_AND_TIMELINE.md) — Phase 0 now–Sep 2026; Phase 1+ Q4’26–Q1’27; Gen2 parallel; Stage B H2’27–2028. Mid Stage A ~€200–250k; Stage A+Gen2 ~€350–450k; through Stage B ~€0.8–1.0M (ranges — not a budget).

## A. Hardware & firmware

| Step | Action | Pass |
|------|--------|------|
| A1 | Connect J-Link to pcb00003 | ☐ |
| A2 | Flash [`firmware/oralable_1.0.70_pcb00003_merged.hex`](./firmware/oralable_1.0.70_pcb00003_merged.hex) | ☐ |
| A3 | nRF Connect: read `006` → **1.0.70** | ☐ |
| A4 | On Oralable case: `on_dock=1` + `charge_active=1` (blink); later taper → solid red, `charge_active=0` | ☐ |
| A5 | **Oralable case** + charged clip (>30% SOC) | ☐ |

```bash
cd /Users/johnacogan67/work/oralable_nrf
./scripts/flash_and_rtt.sh --no-build --hex artifacts/oralable_1.0.70_pcb00003_merged.hex
# J-Link SNR default: 1050090445 — export JLINK_SNR=... if different
```

---

## B. iOS build (TestFlight / Ad Hoc)

| Step | Action | Pass |
|------|--------|------|
| B1 | Confirm OralableApp **4.3.3** / bump build if needed (`CURRENT_PROJECT_VERSION`) | ☐ |
| B2 | Archive **OralableApp** scheme, **Release**, device **Any iOS** | ☐ |
| B3 | Distribute → TestFlight (internal) or Ad Hoc for Ed/Pedro devices | ☐ |
| B4 | Install on your iPhone; confirm app opens past sign-in + fit gate | ☐ |

**Pilot build must include:** FW gate min **1.0.63** · recommend **1.0.70** · Vitals phase · **Device LED** mirror (STAT flash/taper) · placement picker with **Automatic** · nRF CSV export.

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
| C1 | Placement **Automatic** (or **On wireless charger**) → seat in **Oralable case** → red flash / Charging chip | ☐ |
| C2 | Wait for optional solid red (STAT taper) while still on case — Dock on, Charging off | ☐ |
| C3 | Settings → placement **Worn** (manual mode 3) | ☐ |
| C4 | Mount on **temple** → Connect | ☐ |
| C5 | Dashboard HR and/or SpO₂ live ≥ 2 min, no disconnect in first 5 min | ☐ |
| C6 | Settings → placement shows **Worn** (temple vitals — not cheek muscle UI) | ☐ |

**Fail actions:**
- Disconnect in first 5 s → confirm FW **1.0.70**; do not enable `00A` in Developer Settings.
- Flat dashboard → placement **Worn**; reconnect; check temple contact / quality.
- Dock stays 0 on case → confirm Oralable case (not MagSafe); reflash 1.0.70.

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
| E5 | Optional: Share → Clinical Temporalis PDF | ☐ |

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

## F. Ship gate

| Item | Ready |
|------|-------|
| Dry-run validation plot archived | ☐ |
| 2 clips flashed **1.0.70** | ☐ |
| TestFlight invite (1.0.70-aware app) sent to Ed + Pedro | ☐ |
| Quick start PDF printed / emailed | ☐ |
| **Oralable cases** + clips packed | ☐ |

**After pass:** notify Ed/Pedro with TestFlight link + [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md).

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
