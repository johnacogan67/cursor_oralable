# What AcuPebble, MAM, Dual A, and PSG actually measure

**PDF:** [PEDRO_CONSTRUCT_MAP_NOTE.pdf](./PEDRO_CONSTRUCT_MAP_NOTE.pdf) · [HTML](./PEDRO_CONSTRUCT_MAP_NOTE.html)  
**To:** Dr Pedro Mayoral Sanz · cc Dr Edward Owens  
**From:** John A. Cogan (JAC / Oralable)  
**As at:** 30 Aug 2026 · Pack **1.1.68**  
**Status:** Working note for methods / Paper A vs Paper B. Not a product claim. Not a diagnosis.

Internal lock (iterate there, do not copy Table 1 into other docs): [MEASUREMENT_CONSTRUCT_MAP.md](./MEASUREMENT_CONSTRUCT_MAP.md).

---

## Email (copy-paste)

Attach [PEDRO_CONSTRUCT_MAP_NOTE.pdf](./PEDRO_CONSTRUCT_MAP_NOTE.pdf). Cc Ed if this is a methods lock.

Subject: What AcuPebble, Oralable, Dual A, and PSG measure

Pedro,

I have written a short note so that we keep each instrument in its own box when we write methods. The first page defines the words we use: instrument, methods, Dual A, muscle versus infrared, Paper A, and the FEP paper. The tables follow.

AcuPebble remains your home AHI and ODI, from neck sound and finger oxygen. Oralable sits on the temple. It does not use a finger probe. Dual A is muscle activity (ANR) and the Oralable infrared trough on the same temporalis. AHI, Azarbarzin hypoxic burden, and our SASHB are not the same number.

Table 2 shows what each stack gives you now. Table 1 is the lock on what matches what.

Please tell me which path you want first at Beacon: AcuPebble with Oralable for the 1–2 hour arm; Dual A (muscle versus infrared); or AcuPebble event times lined up with muscle bouts. That last path is for the FEP paper. It is not built, and it is not Paper A.

John

---

## Words we use

| Word | Meaning |
|------|---------|
| **Instrument** | The device and what it senses. AcuPebble: neck sound + **finger** oxygen (AHI / ODI). Oralable (MAM): temple light and motion; oxygen is **temple**, not finger. ANR: electrical muscle activity on the same temporalis. PSG-AV: the lab study. |
| **Methods** | How we write the measurement: site, sample rate, what we scored, what we do not claim. Not your MAD titration method. |
| **Dual A** | Oralable and ANR on the same anterior temporalis at the same time. Seat Oralable first. Confirm an infrared trough on a hard clench. Then add ANR, electrodes vertical. What we score today: an EMG burst, then an infrared trough about 1–5 seconds later. Not AHI. Not a night. One engineering pack is about six minutes. |
| **Muscle versus infrared** | Dual A’s main pair. **Muscle** = ANR EMG (the clock). **Infrared** = Oralable IR-DC trough (blood squeezed from the tissue; it lags). Same event, two physics. Not the same units. |
| **Paper A** | The Oralable kit / feasibility paper (IEEE draft): wear, temple vitals, Dual A as a descriptive precursor. About five people at Beacon. Not a registered trial. Not an AHI claim. It does not test the FEP theory. |
| **FEP paper** | Owens and Mayoral, *Frontiers in Behavioral Neuroscience*, 2026. Theory: sleep bruxism as predictive regulation. No Oralable cohort. Hypothesis 3.5.4 needs apnea/hypopnea **times** plus a muscle clock. That coupling is not built. Cite for Paper B / Arm P, not as Paper A methods. |

Instruments measure. Dual A pairs muscle with infrared. Methods is how we write that. Paper A is the kit paper. The FEP paper is the theory paper.

---

## Why this note

You already run **AcuPebble** for home OSA (AHI + ODI, **finger ox**). Oralable (**MAM**) sits on the **temple**. ANR is **EMG** on the same temporalis belly. Dual A is MAM + ANR on that site.

These are different physics. Mixing them in methods copy makes AHI look like temple SpO₂, or EMG look like an IR trough. The two tables below are the lock we are using so Paper A, Dual A, and your FEP paper stay in the right boxes.

**Three oxygen numbers (do not merge):**

| Name | What it counts | Needs scored apneas/hypopneas? |
|------|----------------|--------------------------------|
| **AHI** | Event **count** per hour of sleep | Yes |
| **HB** (Azarbarzin hypoxic burden) | SpO₂ **area tied to those events** | **Yes** |
| **SASHB** | Oralable area when temple SpO₂ &lt; 90% | **No** — temple PPG only, **not** finger ox |

Your AcuPebble headline is **AHI / ODI**. That is not Azarbarzin HB unless your export has event-tied SpO₂ area. We do not claim that.

**Ox lock:** AcuPebble **always** uses finger ox. MAM, ANR, and Dual A **do not**. Dual A SpO₂ is still temple PPG.

---

## Relevance to you

| Your work | What the tables say |
|-----------|---------------------|
| **AcuPebble (today)** | Home AHI + ODI from neck + **finger** SpO₂. Keep this as the apnea reference. Oralable does not replace it. |
| **Arm P / MAD nights** | MAM can sit beside AcuPebble for temple vitals and later jaw-load. Temple SpO₂ ≠ finger ox. SASHB ≠ AHI ≠ HB. |
| **Dual A (research)** | Same-site **muscle (EMG)** then **infrared (IR-DC)** trough (~1–5 s lag). One engineering pack: IR↔EMG F1 ≈ 0.61 vs EMG; F1 vs Protocol A labels = 0; ~6 min, not a night. |
| **FEP / hypothesis 3.5.4** | Needs **event timestamps** plus a bout clock. Nightly AHI is not enough. Closest home path: AcuPebble events + Dual A. **Not built.** Cite FEP for Paper B / Arm P, not Paper A methods. |
| **PSG-AV** | Gold AHI, RMMA, and HB if scored. Lab, not the home stack we ship. |

Paper A is feasibility / Dual A precursor. It does **not** test FEP endotypes.

---

## Table 2 — what each stack gives you now

| Stack | You get now | You still lack |
|-------|-------------|----------------|
| **MAM** (Oralable temple) | IR trough, % drop, SASHB, tonic/phasic/rescue/recovery **labels**, Core ML, **temple** SpO₂ | Finger ox; AHI; HB; electrical onset; FEP latency |
| **ANR** | EMG onset + amplitude bouts | Finger ox; AHI; HB; IR; Core ML; rescue-as-airway |
| **Dual A** | **Muscle (EMG)** versus **infrared (IR-DC)** trough + **temple** SASHB shown beside EMG | Finger ox; AHI; HB; Core ML vs EMG; overnight tonic/phasic F1; FEP latency |
| **AcuPebble** | AHI + ODI / **finger** SpO₂ (**always**) | HB unless export is event-tied; all jaw rows |
| **AcuPebble + Dual A** | AHI **plus** EMG→IR. Finger SpO₂ is **AcuPebble only**; Dual A still temple + EMG | Coupling only if event timestamps align — **not built**. Dual A SpO₂ is not the finger channel |
| **PSG-AV** | AHI **and** HB (if scored) **and** RMMA | Oralable IR-DC / Core ML / SASHB unless MAM is worn |

---

## Table 1 — constructs (what matches what)

**Now** = what exists today. **MAM if verified** = the name we may use only after Dual A (or PSG) F1, or an event clock. **Still not** = even after a pass.

| # | Construct | Now | PSG-AV | AcuPebble | ANR | Dual A now | **MAM if verified** | Still not |
|---|-----------|-----|--------|-----------|-----|------------|---------------------|-----------|
| 1 | **AHI** | Impossible from MAM/ANR | **Gold** | Home AHI | No | No | **None** | Temple PPG/ACC/IR cannot become AHI |
| 2 | **HB** (Azarbarzin) | Impossible (no finger ox, no scored events) | **Yes** if event-tied finger/PSG area | AHI/ODI ≠ HB unless export is event-tied — **not claimed** | No | No | **None** as HB | SASHB; Dual A pairing; AHI; ODI |
| 3 | **SASHB** | Temple PPG SpO₂&lt;90 area — **not finger ox** | Not this formula | Not AcuPebble’s formula | No (no SpO₂) | Temple SASHB shown beside EMG. Not AHI, not HB, not finger ox | Stay “temple SpO₂&lt;90 area” | AHI; HB; ODI; finger SpO₂ |
| 4 | **LP IR-DC trough** | Scored Dual A | No optical OMG | No | No | **Primary pair:** EMG first, IR later | Occlusion onset / trough (~1–5 s lag) | EMG onset; µV |
| 5 | **EMG burst onset** | ANR / PSG | **Yes** — RMMA clock | No | **Yes** | Clock for row 4 | MAM cannot get electrical onset | Millisecond lockstep |
| 6 | **IR-DC % drop** | Gate ≥8% of rest | No | No | No | EMG ≥70 **and** IR ≥8% | Occlusion depth (% of rest) | EMG amplitude; %MVC |
| 7 | **EMG amplitude** | ANR 0–1023 | µV / often ≥10% MVC | No | **Yes** | Contact proof, not MVC | — | Depth ≠ µV |
| 8 | **Overnight tonic** | MAM **label** | Tonic RMMA (different def) | No | Sustained EMG, not scored vs IR | Not F1’d | Tonic occlusion minutes | Tonic RMMA until F1 |
| 9 | **Overnight phasic** | MAM **label** | Phasic RMMA | No | Phasic EMG | **Not F1’d** — ACC ≠ EMG | High-motion bouts; this row may fail | Phasic RMMA; grinding EMG |
| 10 | **Core ML MAM** | Product classes; 10% IR-DC **gates** only | No | No | No | Never run vs ANR | 1 s optical–motion class if F1 vs EMG holds | EMG class; apnea class |
| 11 | **Rescue** | MAM state. EMG ∩ **temple** desat. **No finger ox** | Events + RMMA (finger ox) | AHI events + **finger** desat if exported | No | Pairing ≠ rescue = EMG | Load + temple-desat bout if timestamps match AcuPebble/PSG event ends | AHI; finger-ox desat; FEP airway RMMA until scored |
| 12 | **Recovery** | IR-DC + ACC settle | Could define EMG + autonomic return | No | Unused for this | Not FEP latency | Optical–motion return time | **Homeostatic latency** (your paper: EMG + HR/HRV) |
| 13 | **TFI** | Optical load index | No | No | No | No | Night occlusion burden if vs EMG episode index | MVC%; AASM SB index |

**Engineering Dual A pack (locked, not a night):** session `20260812_085110`, ~6 min. Median EMG→IR lag ≈ **4.9 s**. IR↔EMG F1 ≈ **0.61**. F1 vs Protocol A labels = **0**.

---

## Do / do not (for methods)

| Do | Do not |
|----|--------|
| Keep AcuPebble for AHI / ODI (finger ox) | Call Oralable or Dual A an HSAT or finger-ox channel |
| Call Dual A primary pair EMG → IR trough (muscle versus infrared) | Call the IR trough EMG; call ACC phasic RMMA |
| Call tonic / phasic / rescue / Core ML **labels** until F1 | Treat Core ML as scored vs ANR |
| Call recovery optical–motion settle | Call it homeostatic latency |
| Call AcuPebble AHI/ODI | Call AHI or ODI Azarbarzin HB unless the export has event-tied SpO₂ area |

---

*Device-inferred wellness / research signals — not a medical diagnosis. Paper A does not test FEP.*
