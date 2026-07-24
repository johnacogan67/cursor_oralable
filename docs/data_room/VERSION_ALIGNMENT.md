# Version alignment (canonical)

**As at:** 24 Jul 2026 · Data room pack **1.1.38** · Docs hub **1.3.11**

Use this table when updating flash guides, pilot handouts, architecture, website, or TestFlight notes. Prefer these strings over older “1.0.66 ship / 1.0.67 next” language.

## Milestone — 24 Jul 2026 (Temporalis Mac Protocol A)

| Item | Status |
|------|--------|
| **Stack** | FW **1.0.70** · app **4.3.3** (build **4**) · Gen1 REV10 |
| **OralableCore** | `BruxismMAM_Temporalis.mlpackage` retrained from Mac BLE Protocol A log `data/raw/TEMPORALIS_RAW_20260724_084345.txt` |
| **Capture path** | `scripts/run_protocol_a_session.py` (bleak + worn-mode write + timed cues) → `process_temporalis_gold` / `run_temporalis_mam_pipeline.py` |
| **Ed/Pedro kits** | **Not yet shipped** (charge-to-temple gate unchanged) |

## Pilot ship status (ready ≠ delivered)

| Item | As at 24 Jul 2026 |
|------|------------------|
| **Ed/Pedro kits** | **Not yet shipped** |
| **Stack** | FW **1.0.70** · app **4.3.3** · Gen1 REV10 — flash/TestFlight path ready |
| **Ship gate** | Case charge to **temple-ready SOC (≥50%)** + short worn HR/SpO₂ without brownout |
| **Status sense** | STAT blink policy in **1.0.70** (do not say “chrsts broken on REV10”) |
| **Open work** | Cell energy / coupling so voltage **rises** on the Oralable case |
| **Detail** | [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) § Pilot ship status |

| Layer | Current | Notes |
|-------|---------|--------|
| **Gen1 firmware (ship)** | **1.0.70** | `oralable_nrf` `app/VERSION` · STAT blink = charging · Oralable case only · hex in `data_room/firmware/` |
| **Gen1 FW GATT string** | `1.0.70` | Read `3A0FF006` after flash |
| **iOS FirmwareGate hard min** | **1.0.63** | Blocks older research builds |
| **iOS FirmwareGate recommend** | **1.0.70** | Automatic dock / STAT policy |
| **Oralable patient app** | **4.3.3** | Marketing version; vitals phase + STAT LED mirror + Temporalis MAM refresh |
| **Oralable app build** | **4** | `CURRENT_PROJECT_VERSION` (bump on each TestFlight) |
| **Gen2 firmware (target)** | **2.0.x** | Not on Ed/Pedro kits |
| **Rollback hex (optional)** | 1.0.66 | Keep in `firmware/` only for recovery |

## Feature milestones folded into 1.0.70

| Introduced | Feature | Still true in 1.0.70 |
|------------|---------|----------------------|
| 1.0.63 | Charger = red / bench = green; probe off | Yes |
| 1.0.65 | Dim LEDs; charge_active + mode 1 | Yes (superseded on-pad by STAT) |
| 1.0.66 | +4 dBm TX; fast reconnect adv | Yes |
| 1.0.67 | `.recycled` adv; worn-disconnect FIFO; PPG DT IRQ | Yes (in tree) |
| 1.0.68 | Remapped battery gauge 3.61–4.35 V | Yes |
| **1.0.70** | **CHRSTS STAT activity** (blink / taper / undock) | **Ship** |

## Flash / app pairing

| Role | Flash | App |
|------|-------|-----|
| Ed/Pedro Phase 0 | [FIRMWARE_1.0.70_FLASH.md](./FIRMWARE_1.0.70_FLASH.md) | TestFlight **4.3.3+** |
| Quick start | [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md) | Same |
| Dry run | [PILOT_DRY_RUN_CHECKLIST.md](./PILOT_DRY_RUN_CHECKLIST.md) | Archive 4.3.3 |

## Do not say (outdated)

- “Pilot ship is 1.0.66” (unless describing rollback)
- “chrsts broken on REV10”
- “Automatic placement unreliable” without “pre-1.0.70”
- “Charge on Qi / MagSafe”
- “Kits already with Ed/Pedro” / “shipping” — until ship gate clears (say **gated** / **not yet shipped**)
