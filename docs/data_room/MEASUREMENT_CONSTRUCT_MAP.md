# Measurement construct map — MAM · ANR · Dual A · AcuPebble · PSG

**As at:** 30 Aug 2026 · Pack **1.1.68**  
**Status:** Living table — iterate here; other docs **point**, they do not copy the matrix  
**Agent slug:** `clinical` · start this file when the question is “what does X measure vs Y”

**One-liner:** Four instrument families, different physics. Dual A already scores **EMG burst → LP IR-DC trough**. Other MAM names stay **labels** until F1 (or an event clock) says otherwise. **AHI**, **HB**, and **SASHB** are three different oxygen numbers. **AcuPebble always uses finger ox** (AHI + finger SpO₂ / ODI). PSG-AV = AHI **and** RMMA. **MAM and ANR do not use finger ox.** MAM SpO₂ / SASHB is **temple PPG**. ANR is EMG only. MAM never becomes AHI or Azarbarzin HB.

**Pedro sendable:** [PEDRO_CONSTRUCT_MAP_NOTE.md](./PEDRO_CONSTRUCT_MAP_NOTE.md) · [PDF](./PEDRO_CONSTRUCT_MAP_NOTE.pdf) — tables + why they matter for AcuPebble / Arm P / Dual A / FEP. Do not paste Table 1 into Paper A.

**Related (detail, not duplicates):** [ANR_M40_CONCORDANCE.md](../ANR_M40_CONCORDANCE.md) · [ACUPEBBLE_VS_ORALABLE_ANR.md](./ACUPEBBLE_VS_ORALABLE_ANR.md) · [BRUXOFF_PSG_GOLD_STANDARD.md](./BRUXOFF_PSG_GOLD_STANDARD.md) · [ED_PEDRO_SB_FEP_DRAFT_PAPER.md](./ED_PEDRO_SB_FEP_DRAFT_PAPER.md) · [OVERNIGHT_NIGHT_REPORT.md](../OVERNIGHT_NIGHT_REPORT.md) · [ALGORITHM_ARCHITECTURE.md](../ALGORITHM_ARCHITECTURE.md) · [TEMPORALIS_COLLECTION_PROTOCOL.md](../TEMPORALIS_COLLECTION_PROTOCOL.md) · [MAYORAL_METHOD_ORALABLE_VALIDATION.md](./MAYORAL_METHOD_ORALABLE_VALIDATION.md)

---

## How to iterate

Edit **this file**. Bump **As at**. Add a line under [Changelog](#changelog). Do not paste a second copy of Table 1 into pitches or Paper A — link here.

| Kind | Change freely | Lock unless evidence moves |
|------|---------------|----------------------------|
| Open | “Could” / AcuPebble export event-tied HB / new Dual A F1 | — |
| Locked | Eng pack `20260812_085110` numbers · gates EMG ≥70 and IR ≥8% · SASHB ≠ AHI ≠ Azarbarzin HB · **MAM/ANR/Dual A: no finger ox** · **AcuPebble: always finger ox** · Paper A does not test FEP | New measured pack or a written protocol change |

**F1** (this map): event-match score. Pair bouts if midpoints fall inside a time window; then precision × recall harmonic mean. Script: `scripts/align_anr_oralable_concordance.py` (`_event_f1`). **1** = all matched. **0** = none. Not a sleep index.

**Oxygen trio (do not merge):**

| Name | What it counts | Needs scored apneas/hypopneas? |
|------|----------------|--------------------------------|
| **AHI** | Event **count** per hour of sleep | Yes |
| **HB** (Azarbarzin hypoxic burden) | SpO₂ **area tied to those events** | **Yes** |
| **SASHB** | Oralable area: Σ(90 − SpO₂)·dt when SpO₂ &lt; 90% (%·s) | **No** — continuous **temple** SpO₂ only (**not** finger ox) |

---

## Instruments (jobs)

| | **MAM** (Oralable) | **ANR M40** | **Dual A** | **AcuPebble** | **PSG-AV** |
|--|--------------------|-------------|------------|---------------|------------|
| **Job** | Temple optical + motion + **temple** SpO₂ | Temporalis sEMG | Same-site optical vs electrical | Home OSA HSAT (neck + **finger SpO₂**) | Lab gold |
| **Site** | Anterior temporalis | Same belly, electrodes vertical | Stack: seat MAM first | Neck + finger ox | Full montage + masticatory EMG + **finger ox** |
| **Ox source** | Temple PPG — **no finger ox** | **None** (EMG only) | Temple PPG from MAM — **no finger ox** | **Finger** SpO₂ — **always** | Finger / PSG ox |
| **Now** | Vitals; IR-DC / states as engineering | Bout onset + amplitude 0–1023 | EMG→IR lag + IR↔EMG F1 + temple SpO₂∩EMG | **AHI + ODI** (finger SpO₂ series) | **AHI and** scored RMMA |
| **Not** | Finger ox; AHI; HB; EMG µV; FEP latency | Finger ox; AHI; IR-DC; Core ML | Finger ox; AHI; HB; Core ML vs EMG | RMMA; IR-DC; Azarbarzin HB unless the **export** has event-tied SpO₂ area | Oralable classes unless MAM is worn |

Acurable product pages list SA100 and Ox100. **This map: AcuPebble always uses finger ox.** Do not write Pedro methods as acoustic-only. MAM/ANR/Dual A never borrow that probe. [ACUPEBBLE_VS_ORALABLE_ANR.md](./ACUPEBBLE_VS_ORALABLE_ANR.md).

---

## Table 1 — constructs (canonical)

Columns: **now** = what exists in code / Dual A today. **MAM if verified** = the quantity you may report once it is no longer a class name. **Still not** = even after a pass.

| # | Construct | Now | PSG-AV | AcuPebble | ANR | Dual A now | **MAM if verified** | Still not | Verify with |
|---|-----------|-----|--------|-----------|-----|------------|---------------------|-----------|-------------|
| 1 | **AHI** (apnea–hypopnea events / sleep hour) | Impossible from MAM/ANR | **Gold** (airflow + effort + sleep hours) | Home AHI | No | No | **None** | Temple PPG/ACC/IR cannot become AHI | Keep AcuPebble or PSG. Event **timestamps** needed for pairing, not nightly AHI alone |
| 2 | **HB** (Azarbarzin hypoxic burden) | Impossible from MAM/ANR (no finger ox, no scored events) | **Yes** if events are scored (usually **finger / PSG** SpO₂ area **linked to** apnea/hypopnea) | AHI/ODI **do not** equal HB. HB only if the report includes **event-tied** SpO₂ area — **not claimed** | No | No | **None** as Azarbarzin HB | SASHB; Dual A pairing; AHI; ODI | PSG, or AcuPebble **only if** Pedro’s export has event-linked area |
| 3 | **SASHB** | Engineering SpO₂&lt;90 area from **temple PPG** — **not finger ox** | Not this formula (their HB is event-tied, usually finger) | Not AcuPebble’s formula | No (no SpO₂) | SASHB is computed from **MAM temple** SpO₂. It is shown **beside** ANR EMG. Descriptive — not AHI, not HB, not finger ox | Stay “temple SpO₂&lt;90 area (%·s / wear h)”. Do not rename to HB | AHI; Azarbarzin HB; ODI; finger SpO₂ | [ACUPEBBLE_VS_ORALABLE_ANR.md](./ACUPEBBLE_VS_ORALABLE_ANR.md) |
| 4 | **LP IR-DC trough** (Butterworth &lt;~0.8–1 Hz) | Scored Dual A | No optical OMG | No | No | **Primary pair:** EMG first, IR later. EDF inverts IR so troughs go up like EMG | **Occlusion onset / trough time** (hemodynamic lag ~1–5 s) | EMG onset; µV | Dual A lag + IR↔EMG F1 |
| 5 | **EMG burst onset** | ANR / PSG | **Yes** — RMMA clock | No | **Yes** ~10 Hz notify | Clock for row 4 | MAM cannot get electrical onset; verified IR is a **lagged optical clock** | Millisecond lockstep | ANR vs PSG RMMA later |
| 6 | **IR-DC % drop / occlusion** (tonic hold ~2.5–8% of rest) | Gate **≥8%** of rest median | No | No | No | Same event as EMG, **not** same units. Gate: EMG max **≥70 and** IR ≥8% | **Occlusion depth** (% of rest) | EMG amplitude; %MVC | Dual A preflight + depth vs bout (no unit conversion) |
| 7 | **EMG amplitude** | ANR 0–1023; z-score bouts | µV / often ≥10% MVC | No | **Yes** | Contact proof, not MVC | — | Depth ≠ µV | Do not map 1023 ↔ % drop |
| 8 | **Overnight tonic** (IR drop + stable ACC) | MAM **label** | Tonic RMMA — different definition | No | Sustained EMG — **not scored vs IR** | Not F1’d | **Tonic occlusion minutes** | Tonic RMMA until F1 vs ANR/PSG EMG | Overnight Dual A |
| 9 | **Overnight phasic** (high ACC power) | MAM **label** | Phasic RMMA on EMG | No | Phasic EMG bursts | **Not F1’d** — ACC ≠ EMG | At best **high-motion bouts**. This row may **fail** | Phasic RMMA; grinding EMG | Overnight Dual A — expect weak |
| 10 | **Core ML MAM** (quiet / tonic / phasic / rescue, 1 s × 6 ch) | Product classes. 10% IR-DC shift **gates** inference only | No | No | No | **Never run vs ANR** | **1 s optical–motion class** if F1 vs EMG (or PSG RMMA) holds | EMG class by itself; apnea class | Concordance + Core ML; not Paper A |
| 11 | **Rescue** (IR drop + **temple** SpO₂ &lt; **92%** overnight) | MAM **state**. Pairing = EMG ∩ temple desat, not this class. **No finger ox** | Apnea/hypopnea **events** + RMMA (finger ox on PSG) | AHI **events** + **finger** desat if exported | No | Pairing ≠ rescue = EMG. Temple SpO₂ only | **Load + temple-desat bout** if timestamps match AcuPebble/PSG **event ends** | AHI; finger-ox desat; Owens & Mayoral airway RMMA until that pairing is scored | AcuPebble or PSG event list + Dual A. Paper A does **not** test FEP |
| 12 | **Recovery** (`recovery_median_s`) | IR-DC + ACC settle | Could define EMG + autonomic return | No | Inter-bout unused for this | Not FEP latency | **Optical–motion return time** after a bout | **Homeostatic latency** (their paper: EMG + HR/HRV) | Do not rename. Operationalize FEP on PSG or Dual A + HR later |
| 13 | **TFI** | Optical load index | No | No | No | No | **Night occlusion burden** if vs EMG episode index (units still differ) | MVC%; AASM SB index | Overnight Dual A / PSG EMG index |

**Eng pack (locked):** `20260812_085110` (~6 min, not ≥6 h). Median EMG→IR lag ≈ **4.9 s**. IR↔EMG F1 ≈ **0.61**. F1 vs Protocol A labels = **0**. `spo2_qc=warn`. Seat MAM alone first.

---

## Table 2 — stacks (what you get)

| Stack | You get now | You still lack |
|-------|-------------|----------------|
| **MAM** | IR trough, % drop, SASHB, tonic/phasic/rescue/recovery **labels**, Core ML, **temple** SpO₂ | Finger ox; AHI; HB; electrical onset; FEP latency |
| **ANR** | EMG onset + amplitude bouts | Finger ox; AHI; HB; IR; Core ML; rescue-as-airway |
| **Dual A** | Table 1 rows **4–6** + **temple** SASHB shown beside EMG | Finger ox; AHI; HB; Core ML vs EMG; tonic/phasic F1; FEP latency |
| **AcuPebble** | AHI + ODI / **finger** SpO₂ (**always**) | HB unless export is event-tied; all jaw rows |
| **AcuPebble + Dual A** | AHI **plus** EMG→IR. Finger SpO₂ is **AcuPebble only**; Dual A still temple + EMG | Coupling **only if** event timestamps align — **not built**. Still not HB unless event-tied area. Do not treat Dual A SpO₂ as the finger channel |
| **PSG-AV** | AHI **and** HB (if scored) **and** RMMA | Oralable IR-DC / Core ML / SASHB unless MAM is worn |

Hypothesis 3.5.4 (MAD/CPAP cuts RMMA **timed to apnea/hypopnea end**) needs **event timestamps** plus a bout clock. Nightly AHI is not enough. Closest home path: **AcuPebble events** (finger ox) + ANR (or Dual A). AHI/ODI still ≠ Azarbarzin HB unless the export has event-tied area. PSG-AV already has both. Cite FEP for Paper B / Arm P, not Paper A methods.

---

## Claim discipline

| Do | Do not |
|----|--------|
| Point here for “what matches what” | Equate SASHB or the Dual A pairing to AHI / ODI / Azarbarzin HB |
| Say MAM/ANR/Dual A have **no finger ox** (temple PPG + EMG) | Call Dual A SpO₂ a finger-ox or HSAT channel |
| Say **AcuPebble always uses finger ox** | Write Pedro AcuPebble as acoustic-only / no finger ox |
| Call Dual A primary pair EMG→IR trough | Call IR trough EMG; call ACC phasic RMMA |
| Call Core ML / tonic / phasic / rescue **labels** until F1 | Run Core ML vs ANR in prose as if scored |
| Call recovery optical–motion settle | Call it homeostatic latency |
| Call AcuPebble Pedro’s AHI/ODI tool (finger ox) | Call AcuPebble AHI or ODI **Azarbarzin HB** unless the export has event-tied SpO₂ area |
| Keep AcuPebble for AHI | Claim a temple microphone would score AHI (Gen1 has no mic; wrong site vs neck HSAT) |
| Paper A = feasibility / Dual A precursor | Paper A tests FEP endotypes |

---

## Changelog

| Date | Change |
|------|--------|
| 29 Aug 2026 | First lock from Dual A / AcuPebble / FEP construct tables. Includes **MAM if verified** column. |
| 29 Aug 2026 | Rows **1–3** = AHI · Azarbarzin **HB** · **SASHB**. Split AcuPebble **SA100** vs **Ox100 (SpO₂-tied)**. Plain Dual A SASHB wording (computed; shown beside EMG; not AHI/HB). |
| 29 Aug 2026 | Locked: **MAM and ANR do not use finger ox.** Dual A SASHB/rescue = temple PPG. Finger ox = Ox100 or PSG only. |
| 29 Aug 2026 | Locked: **AcuPebble always uses finger ox.** Collapsed SA100 vs Ox100 columns. Finger ox = AcuPebble or PSG. Dual A still temple PPG. |
| 29 Aug 2026 | Table 2: dropped **PSG + Dual A** stack. |
| 29 Aug 2026 | Pedro sendable [PEDRO_CONSTRUCT_MAP_NOTE.md](./PEDRO_CONSTRUCT_MAP_NOTE.md) / PDF — pointer only; tables stay here. |
