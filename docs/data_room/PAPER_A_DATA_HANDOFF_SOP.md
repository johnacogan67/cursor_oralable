# Paper A — data handoff SOP (one page)

**PDF:** [PAPER_A_DATA_HANDOFF_SOP.pdf](./PAPER_A_DATA_HANDOFF_SOP.pdf) · [HTML](./PAPER_A_DATA_HANDOFF_SOP.html)

**As at:** 30 Aug 2026 · Pack **1.1.68**  
**For:** Pedro / Ed / John · Beacon feasibility **n≈5** (not a registered clinical trial)  
**Protocol:** [PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md) · kit [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) · Day-1 [ED_PEDRO_QUICK_START.md](./ED_PEDRO_QUICK_START.md)

**One-liner:** After each arm, send John the files below (AirDrop / Files / Mail). Do **not** rely on Share → continuous CSV alone for multi-hour arms — that export is only ~3 min of RAM.

---

## Naming

```text
Oralable_<ARM>_<Who>_<YYYYMMDD>_<kitID>
```

Examples: `Oralable_CORE_Pedro_20260820_RK01` · `Oralable_ARMP_Pedro_20260821_RK01` · `Oralable_DUALA_Mac_20260822_RK02`

Send: files + one CRF row ([feasibility §4](./PAPER_A_FEASIBILITY_PROTOCOL.md#4-crf-minimum-fields)). Note FW string (**1.0.84**) and app (**4.3.3**).

---

## Which button / which file (by arm)

| Arm | Duration | In the Oralable app — do this | Send John | Do not use as sole handoff |
|-----|----------|-------------------------------|-----------|----------------------------|
| **Core** | ≥5–10 min | Connect → temple → HR/SpO₂ stable → **Share → Save CSV to Files** (or Share sheet). Optional: screenshot vitals card | CSV + CRF | — |
| **Arm P** | **1–2 h** | Wear continuous. Keep phone nearby. End → **Share → Export PDF — Clinical Temporalis Report**. Also save any session CSV if offered | **PDF** (primary) + CRF + MAD fields + AcuPebble notes if any | Continuous CSV alone (~3 min RAM) |
| **Arm E/J** | **≥6 h** (goal 8 h) | Overnight / long wear. Morning → **Clinical Temporalis PDF** + hypnogram on Dashboard/Share | **PDF** + CRF. Prefer wear ≥6 h before calling it overnight *N* | Continuous CSV alone; 1–2 h Arm P is **not** Arm E/J |
| **Dual A (preferred)** | ~6 min | **Mac:** `run_dual_protocol_a_session.py` → then `align_anr_oralable_concordance.py` | Pair logs + `plots/concordance/<session>/` (`NEST.md`, overlay, **`session.edf` with EMG**) + CRF Dual A ID | Claiming nest as AHI/ODI |
| **Dual A (iOS optional)** | ~6 min | Settings → tap About **7×** → Developer Settings → **Dual Protocol A ON** → Dashboard → Dual Protocol A → run → **Share export pack** | `TEMPORALIS_RAW_*` + `ANR_EMG_*` + `DUAL_PAIR_*` + **`session.edf` (ANR EMG inside)**; John may re-run Mac align | Leaving Dual A ON for App Store / Day-1 vitals |

**Sleep is the default path.** Dual A stays **OFF** unless you are running a Dual A session.

---

## What each file is for

| File | Use in Paper A |
|------|----------------|
| Research / Share CSV (`device_type,iso8601_timestamp,red,ir,…,spo2_percent,…`) | Wear success, SpO₂ QC, Arm P sketches |
| Clinical Temporalis **PDF** | Hypnogram + bands + bout tables (Arm P / overnight) |
| Memory-flush CSVs (app Application Support; hourly) | Rebuild long nights if PDF sample stream thin — John can pull from phone Files if needed |
| Dual A Mac pack / iOS Share (4 files incl. `session.edf` with EMG) | Methods Dual A + SpO₂∩EMG nest (**not** AHI); EDF = research, not PSG |
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
- [ ] FW **1.0.84** / app **4.3.3** on CRF  
- [ ] Arm named (Core / Arm P / ≥6 h / Dual A)  
- [ ] Arm P or overnight → **PDF** attached (not only short continuous CSV)  
- [ ] Dual A → four files (incl. `session.edf`) or Mac concordance folder  
- [ ] No AHI language in filenames or notes for Oralable metrics  

---

## John receives → builds

Wear % · usability · SpO₂ QC · Arm P within-subject oxygen sketches · Dual A precursor plots · ≥6 h hypnogram when available (replaces FIG-CO-025 layout exemplar).

*Ops companion to [PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md) §6. Not ethics approval.*
