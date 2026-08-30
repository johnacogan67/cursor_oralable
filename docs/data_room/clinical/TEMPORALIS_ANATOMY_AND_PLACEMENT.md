# Temporalis anatomy → Oralable PPG + ANR placement

**As at:** 30 Aug 2026 · Pack **1.1.68**  
**Anatomy source:** Kenhub “Muscles of mastication” tutorial (local NotebookLM PDF: `…/notebook_lm/Sources/description from ken hub of muscels of mastication.pdf`)  
**Related:** [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) · [ANR_M40_CONCORDANCE.md](./ANR_M40_CONCORDANCE.md) · [../TEMPORALIS_COLLECTION_PROTOCOL.md](../../TEMPORALIS_COLLECTION_PROTOCOL.md) · [ACUPEBBLE_VS_ORALABLE_ANR.md](../bookmarks/ACUPEBBLE_VS_ORALABLE_ANR.md)

**One-liner:** Seat Oralable PPG on the **anterior temporalis** elevating belly (vertical fibers). Seat ANR Red Dots **along** those fibers (long axis vertical). MAM measures **optical** load + vitals in that belly; ANR measures **electrical** motor-unit activity of the same belly. Not TMD diagnosis. Not AHI.

---

## 1. Anatomy that drives placement (Kenhub)

Four muscles of mastication: masseter, temporalis, medial pterygoid, lateral pterygoid. Mandible inserts for all four; TMJ is a modified hinge (translate + pivot).

| Fact | Detail |
|------|--------|
| Shape / site | Flat fan on lateral skull; covers much of the temporal bone; easy to palpate on open/close |
| Origin | Temporal fossa up to inferior temporal line + temporal fascia |
| Insertion | Thick tendon under zygomatic arch → apex and medial **coronoid process** |
| Innervation | Anterior + posterior deep temporal nerves (mandibular / V3) |
| Blood | Anterior + posterior deep temporal arteries (maxillary) |

**Functional split (fiber direction) — Kenhub anatomy [cite]; FIG-CO-056 / 057 Oralable schematics:**

| Part | Fiber direction | Action (Kenhub) | Research Kit seat? |
|------|-----------------|-----------------|--------------------|
| **Anterior** | Near **vertical** | Elevates mandible (with posterior) → occlusion / bite | **Yes — primary** (Oralable + ANR Dual A) |
| **Posterior** | Near **horizontal** | Elevates + **retracts** mandible (only masticator that retracts) | **No** as primary Dual A site |

![FIG-CO-056 — Anterior fibers elevate](../figures/FIG-CO-056-temporalis-anterior-elevate.png)

*FIG-CO-056. Oralable schematic: anterior (vertical) fibers elevate the mandible. Seat Oralable/ANR on this **anterior** belly. Anatomy facts after Kenhub mastication tutorial; teaching stills stay in `figures/research_kit_photo_source/kenhub_temporalis_*.png`.*

![FIG-CO-057 — Posterior fibers retract](../figures/FIG-CO-057-temporalis-posterior-retract.png)

*FIG-CO-057. Oralable schematic: posterior (near-horizontal) fibers retract the mandible and still help elevate. Do **not** use this posterior belly as the primary Oralable/ANR site for occlusion / bruxism-class Dual A.*

Masseter is the strongest elevator but sits over the ramus/cheek (Bruxoff-class masseter EMG is a different site). Oralable Research Kit targets **extraoral anterior temporalis**, same site class as GrindCare.

**Clinical note (context only):** Kenhub lists TMD as muscle- and/or joint-related; muscle-related pain includes tenderness on palpation or contraction of the masticators. Oralable / ANR outputs are **engineering phenotypes**, not TMD or SB diagnosis. PSG-AV remains SB gold standard.

---

## 2. Where to put the Oralable PPG (optical window)

**Target:** peak of the **anterior temporalis** belly — the vertical elevating fibers.

**How to find it**

1. Subject sits still; fingers on the temple between the **eyebrow tail** and the **ear**.
2. Hard clench 2–3 s → feel the firm bulge (anterior belly).
3. Release → mark that peak.
4. Seat the Gen1 clip so the **PPG / coil face is on skin** over that mark (optical in). Firm but comfortable coupling (protocol: ~5–15 mmHg strap-equivalent when headband used).

**Do**

- Centre the optical window on the anterior bulge, not over hairline alone or over the ear.
- Keep coil + PPG toward skin (see Research Kit wear stack).
- Prefer the same side for a Dual A session (photograph placement).

**Do not**

- Sit primarily on the **posterior** (horizontal) belly near the ear — that region is more retraction than occlusion load.
- Confuse masseter (cheek / ramus) with temporalis (temple).
- Claim finger SpO₂ or HSAT equivalence from temple PPG.

**Figures (practice):** FIG-CO-056 (anatomy target) · FIG-CO-032 / FIG-CO-055 (Oralable on temporalis) · FIG-CO-003 placement.

---

## 3. Where to put the ANR M40 electrodes

**Target:** same **anterior temporalis** region, electrodes on skin (Red Dot gel on skin).

**Orientation (locked for Dual A)**

- ANR module long axis **VERTICAL** — parallel to **anterior** temporalis fibers.
- Bridge the peak bulge so the Analog EMG sense path spans the elevating belly (same site class as GrindCare literature placement).
- Stack: skin → Oralable optical window → Kapton lock (when used) → silicone if used → **ANR on Red Dots** → optional headband. ANR also presses the optical stack into the temple.

**Do**

- **Setup — seat Oralable alone first:** Oralable only on the peak → hard clench → see IR-DC trough → then Kapton + ANR (not a full Oralable-only Protocol A).
- Add ANR without sliding the optical window; press the stack onto skin.
- Run Mac Dual A preflight: EMG clench max ≥**70** raw (default — Dual A stack rarely hits 100) **and** IR drop ≥**8%**. SpO₂ AC WARN is non-blocking.
- Keep Red Dot gel contact; if EMG or IR fails, re-seat from Oralable alone.

**Do not**

- Stack ANR before the Oralable-alone IR trough is clear.
- Place ANR horizontal across the posterior belly as the primary Dual A site.
- Treat strong EMG as proof of good IR-DC or finger SpO₂.
- Use ANR alone as OSA grading or SB diagnosis.

**Why (short):** Setup + gate rationale — [ANR_M40_CONCORDANCE.md § Setup / Why these gates](../../ANR_M40_CONCORDANCE.md#setup--seat-oralable-alone-first).

**Figures:** FIG-CO-056/057 (fiber directions) · FIG-CO-031 (ANR vertical) · FIG-CO-026 (device) · FIG-CO-036 layer cake.

---

## 4. What Oralable MAM vs ANR measure (same muscle, different physics)

Both devices sit on the **anterior elevating belly** (FIG-CO-056). They do **not** measure the same quantity.

| | **Oralable MAM** (optical + ACC) | **ANR M40** (sEMG) |
|--|----------------------------------|---------------------|
| **Physics** | Photoplethysmography through skin over the muscle + tissue perfusion | Surface electromyography of motor-unit / membrane activity |
| **When anterior temporalis elevates / clenches** | Muscle stiffens and compresses local vessels → **IR-DC trough** (hemodynamic occlusion / OMG); green AC may change; ACC may show vibration | EMG raw amplitude rises → **bout** onset/offset |
| **What it is good for** | Continuous **vitals** (HR, SpO₂), overnight **states / SASHB / TFI**, multi-hour wear without electrodes | Fast **electrical** timing of clench/grind-class activity; Dual A concordance reference |
| **Lag / pairing** | Optical trough often **lags** EMG by ~1–5 s | Electrical lead of the mechanical / perfusion event |
| **Not measuring** | Not motor-unit spikes; not masseter; not AHI | Not SpO₂; not IR-DC; not OSA grade |

```text
Anterior temporalis elevates (FIG-CO-056)
        │
        ├─► ANR EMG spike ──────────── electrical "when the muscle fires"
        │
        └─► Oralable IR-DC trough ──── optical "when perfusion is occluded"
                    │
                    └─► + HR / SpO₂ / SASHB from same PPG window
```

**Posterior retraction (FIG-CO-057):** real anatomy, wrong primary Dual A target. A sensor parked on the horizontal belly would mix retraction with elevation and weaken occlusion / bruxism-class concordance.

---

## 5. Side-by-side layout (research Dual A)

```text
        eyebrow tail
             │
    ┌────────▼────────┐
    │  ANR (vertical) │  ← long axis ‖ anterior fibers
    │   ●────● Red    │
    │      Dots       │
    │  ┌──────────┐   │
    │  │ Oralable │   │  ← PPG window on anterior bulge peak
    │  │ PPG/coil │   │
    │  └──────────┘   │
    └─────────────────┘
             │
            ear
```

Photograph both sensors for the session folder. Prefer one temple per Dual A run.

---

## 6. Information derived from each path

| Path | Sensor | Primary derived signals | Use |
|------|--------|-------------------------|-----|
| **Oralable PPG** | Red / Green / IR @ 50 Hz research grid | **IR-DC** occlusion / OMG troughs; green AC pulse morphology; **HR**; **SpO₂** (temple empirical curve); **SASHB** | Phase 0 vitals; overnight state hypnogram; TFI / rescue / burden engineering metrics |
| **Oralable ACC** | LIS2DTW12 | Motion, five-tap sync, jaw vibration context | Sync + motion gates |
| **ANR M40** | Analog EMG `0x2A58` (~10 Hz, 0–1023) | Raw EMG amplitude; **bout onset/offset**; Dual A concordance vs IR-DC / labels | Research comparator (Paper A precursor; deeper PSG-AV later) |
| **SpO₂ ∩ EMG nest** | Oralable SpO₂ + ANR bouts (Mac align) | Desat events (≥10 s @ SpO₂ &lt; 90%); EMG∩desat fractions (`NEST.md`) | AcuPebble-style **oxygen-burden context** — **not** AHI/ODI |

**Typical Dual A timing expectation:** EMG bout leads IR-DC trough by ~1–5 s (hemodynamic lag). Labels from Protocol A cues are the training / concordance reference.

---

## 7. Claim discipline

| Do | Do not |
|----|--------|
| Say anterior temporalis = elevating / occluding belly (FIG-CO-056) | Seat Dual A primarily on posterior retracting belly (FIG-CO-057) |
| Say ANR vertical ‖ anterior fibers; MAM = optical occlusion + vitals | Claim MAM “is EMG” or ANR “is SpO₂” |
| Nest SpO₂ with EMG as descriptive burden context | Call nest ODI or AHI; replace AcuPebble |
| Cite Kenhub for anatomy facts; use Oralable FIG-CO-056/057 in papers | Imply Kenhub partnership or clinical validation of Oralable |

---

## 8. Bookmark

| Source | Location | Use |
|--------|----------|-----|
| Kenhub muscles of mastication tutorial | Local NotebookLM PDF | Full mastication set + TMD note |
| Publication schematics | FIG-CO-056 / FIG-CO-057 (`…-temporalis-*.png`) | Anterior elevate vs posterior retract |
| Kenhub stills (teaching archive) | `figures/research_kit_photo_source/kenhub_temporalis_*.png` | Internal reference only — not paper assets |
| Canonical Dual A procedure | [../ANR_M40_CONCORDANCE.md](../../ANR_M40_CONCORDANCE.md) | Preflight + Mac scripts |
| Kit wear stack | [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) §2b | Layer cake + wear photos |
