# Paper A — data handoff SOP (one page)

**PDF:** [PAPER_A_DATA_HANDOFF_SOP.pdf](./PAPER_A_DATA_HANDOFF_SOP.pdf) *(regenerated 7 Sep 2026)* · [HTML](./PAPER_A_DATA_HANDOFF_SOP.html)

**As at:** 7 Sep 2026 · Pack **1.1.68**  
**Living stack (30 Oct):** **Gen 2** REV11 · pack **1.1.74** · kits **gated**. Write the firmware string you read on the CRF. Do not write **1.0.85**. That string is Gen 1. [GEN2_WOUT_KAGA_STATUS_2026-10-30.md](../hardware/GEN2_WOUT_KAGA_STATUS_2026-10-30.md).  
**For:** Pedro / Ed / John · Beacon feasibility **n≈5** (not a registered clinical trial)  
**Protocol:** [PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md) · kit [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) · Day-1 [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md)

**One-liner:** After each arm, send John the files below (AirDrop / Files / Mail). Do **not** rely on Share → continuous CSV alone for multi-hour arms — that export is only ~3 min of RAM. Dual A is **John lab only** — Beacon patients do not send ANR files.

---

## Naming

```text
Oralable_<ARM>_<Who>_<YYYYMMDD>_<kitID>
```

Examples: `Oralable_CORE_Pedro_20260820_RK01` · `Oralable_ARMP_Pedro_20260821_RK01` · `Oralable_DUALA_Mac_20260822_RK02`

Send: files + one CRF row ([feasibility §4](./PAPER_A_FEASIBILITY_PROTOCOL.md#4-crf-minimum-fields)). Note FW string (**1.0.85**) and app (**4.3.3**).

---

## Which button / which file (by arm)

| Arm | Duration | In the Oralable app — do this | Send John | Do not use as sole handoff |
|-----|----------|-------------------------------|-----------|----------------------------|
| **Core** | ≥5–10 min | Connect → temple pad → HR/SpO₂ stable → **Share → Save CSV to Files** (or Share sheet). Optional: screenshot vitals card | CSV + CRF | — |
| **Beacon night** | **≥4 h** (6 h better) · one night | Clinic charge. Wear continuous. Keep phone nearby. Morning return → **Share → Export PDF — Clinical Temporalis Report** | **PDF** (primary) + CRF | Continuous CSV alone (~3 min RAM) |
| **Arm P** | 1–2 h window or same night | Same as Beacon night if MAD logged. PDF + MAD fields + AcuPebble notes if any | **PDF** + CRF + MAD + AcuPebble notes | Continuous CSV alone |
| **Hypnogram figure** | **≥6 h** | Same PDF path. Prefer wear ≥6 h before calling it a figure night | **PDF** + CRF | Treating a 4 h night as the Fig. 5 replacement |
| **Dual A (John lab)** | ~6 min | **Mac:** `run_dual_protocol_a_session.py` (EMG ≥70 + IR drop ≥8%) → then `align_anr_oralable_concordance.py` | Pair logs + `plots/concordance/<session>/` (`NEST.md`, overlay, **`session.edf`**) | Asking Beacon patients for Dual A files |

**Sleep / one-night loaner is the default path.** Dual A stays **OFF**. Patients do not combine ANR and MAM.

---

## What each file is for

| File | Use in Paper A |
|------|----------------|
| Research / Share CSV (`device_type,iso8601_timestamp,red,ir,…,spo2_percent,…`) | Wear success, SpO₂ QC, Arm P sketches |
| Clinical Temporalis **PDF** | Hypnogram + bands + bout tables (Arm P / overnight) |
| Memory-flush CSVs (app Application Support; hourly) | Rebuild long nights if PDF sample stream thin — John can pull from phone Files if needed |
| Dual A Mac pack (John lab) | Methods Dual A + SpO₂∩EMG nest (**not** AHI); EDF = research, not PSG · **not** from Beacon patients |
| CRF | Comfort, setup time, MAD, AcuPebble, adverse events |

---

## Partner-only (not from Oralable app)

| Item | Owner |
|------|--------|
| CRF complete | Pedro / Ed / self |
| MAD type · VDO · advancement | Pedro |
| AcuPebble AHI/ODI (nest, do not replace) | Pedro |
| Bruxoff notes (if used) | Pedro |
| Beacon ethics / consent ID | Pedro (+ Ed) |

---

## Quick checks before you send

- [ ] Kit ID + date in filename  
- [ ] FW **1.0.85** / app **4.3.3** on CRF  
- [ ] Arm named (Core / Beacon ≥4 h / Arm P / ≥6 h figure)  
- [ ] Beacon night or Arm P → **PDF** attached (not only short continuous CSV)  
- [ ] No AHI language in filenames or notes for Oralable metrics  
- [ ] No ANR / Dual A files expected from Beacon patients  

---

## John receives → builds

Wear % · usability · SpO₂ QC · Arm P within-subject oxygen sketches · Dual A lab precursor plots · ≥4 h feasibility nights · ≥6 h hypnogram when available (replaces FIG-CO-025 layout exemplar).

*Ops companion to [PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md) §6. Not ethics approval.*
