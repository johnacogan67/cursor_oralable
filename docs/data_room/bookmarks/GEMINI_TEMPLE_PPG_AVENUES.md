# Temple PPG avenues — Gemini exploration distill

**Status:** External exploration distill · **not** Oralable clinical evidence or product claims  
**As at:** 27 Jul 2026  
**Source:** Gemini conversation (share [UiCw1EsJguV6](https://share.gemini.google/UiCw1EsJguV6) · app `b1033d03cf397b38`)  
**Local capture:** `~/Downloads/Measuring Brain Signals Down Spinal Cord.md`

**Canonical product posture:** [PRODUCT_ROADMAP.md](../PRODUCT_ROADMAP.md) §2b · [IP_NORTH_STAR.md](../IP_NORTH_STAR.md) · landscape [ORALABLE_MARKET_LANDSCAPE.md](../../../oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md) §4b · **App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

---

## Banner — do not misuse

- Stage A language stays **wellness awareness / pattern context** only.
- Gemini “diagnose / treat” wording is **rephrased** here as research or Stage B exploration.
- The main Oralable path is still **night temple clip on Gen1** (Phase 0 → Phase 1+).
- Day glasses, haptic therapy, and stroke/seizure/TBI screens are **not** Phase 0 commitments.

---

## 1. Modality ladder (where Oralable sits)

Measuring descending motor and cortico-spinal activity runs from invasive electrical gold standards to optical hemodynamic proxies:

| Class | Examples | What it measures | Oralable relation |
|-------|----------|------------------|-------------------|
| Direct neural / IOM | MEP, D-wave, ESG, BCI microarrays | Electrical descending tracts | **Orthogonal** — clinic / surgery |
| Brain electrical | EEG, CMC (+ EMG) | Cortical / corticomuscular sync | **Orthogonal** — PPG may use ANS/HRV proxy only |
| Muscle electrical | sEMG (ANR M40, Cometa) | Action potentials | **Adjacent** gold standard — Research Kit Dual A / Paper A descriptive precursor ([ANR_M40_CONCORDANCE.md](../clinical/ANR_M40_CONCORDANCE.md) · [ORALABLE_RESEARCH_KIT.md](../clinical/ORALABLE_RESEARCH_KIT.md)); not consumer Oralable alone |
| Optical muscle (OMG) | Temple / cheek IR-DC PPG | Hemodynamic occlusion | **Oralable core** (Phase 1+) |
| Scalp fNIRS | Motor cortex blood | Neurovascular coupling | Related physics; different site / claim |
| Overnight SpO₂ / BP peers | Wellue, Aktiia/Hilo | SpO₂ / BP time series | Same report grammar; different biomarker |

**Temporal lag:** EEG, EMG, and MEP are millisecond-scale. PPG and optical myography follow blood-volume change with a hemodynamic delay of about **~1–5 s**. That fits ambulatory wearables and overnight phenotype mapping. It does not replace intraoperative or millisecond nerve-conduction diagnostics.

**Physics Oralable uses:** IR PPG over temporalis and the superficial temporal artery — capillary compression on clench (optical myography), arterial pulse (HR/SpO₂/HRV), ACC jaw vibration, and temp (worn / context). Same sensor suite; product claims stay phase-gated.

---

## 2. Avenue matrix (Near / Mid / Far / Out)

| Horizon | Avenue | Sensors / needs | Oralable posture |
|---------|--------|-----------------|------------------|
| **Near** | Temple vitals HR / SpO₂ | Green / R / IR PPG | **Phase 0 (now)** |
| **Near→Mid** | Optical myography bruxism / TMJ load + ACC | IR-DC + jaw vib | **Phase 1+ Stage A** |
| **Near→Mid** | Overnight SpO₂ burden (SASHB) + state hypnogram | SpO₂ + overnight states | Eng PDF + **in-app hypnogram** shipping; Phase 1+ muscle UX polish |
| **Mid** | Daytime clench / stress (chew vs clench filter) | Same sensors; day UX | **Avenue** — not Ed/Pedro |
| **Far** | Migraine prodrome / vascular amplitude | Temporal artery PPG + temp | Research / Stage B explore |
| **Far** | Sleep apnea **screening claims** | SpO₂ + HR + ACC | Stage A = pattern context only; diagnostic = Stage B |
| **Far** | Haptic / audio biofeedback therapy | Actuator not on Gen1 BOM | Gen2+ / accessory hardware avenue |
| **Out (Stage A)** | Bilateral carotid / PTT BP / GCA / TBI / seizure | 2nd site, ECG, clinical gold | Deferred — not roadmap |

**Day vs night:** The night temple clip is the product of record. Day form factor (glasses / frames) shares the sensor thesis but is a **separate UX and noise-filtering product** — explore after Phase 1+; do not slip into Phase 0.

---

## 3. Sensor-fusion ideas kept as research notes

| Target (Gemini language) | Distill for Oralable docs |
|--------------------------|---------------------------|
| Bruxism / TMJ | Aligns with Phase 1+ IR-DC + ACC — Stage A awareness |
| Migraine / vascular | Research avenue only |
| Concussion / TBI | Out of Stage A — needs impact protocol + clinical gold |
| Sleep apnea architecture | SpO₂/HR context in Stage A; no OSA diagnosis claim |
| Epileptic seizure | Out of Stage A |
| Core temp / circadian | Temp today is worn/context; core-temp claim not Gen1 |
| Bilateral carotid / stroke | Needs dual temple — out |
| PTT blood pressure | Needs ECG/chest or ring sync — out |
| GCA / temporal arteritis | Clinical triage only — out |
| Active biofeedback therapy | Actuator + therapy claims — Far / Gen2+ |

---

## 4. Pointers

- Roadmap avenues table: [PRODUCT_ROADMAP.md §2b](../PRODUCT_ROADMAP.md#2b-technology-avenues)
- Landscape modality ladder: `oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md` §4b
- Overnight bands / hypnogram: [OVERNIGHT_NIGHT_REPORT.md](../OVERNIGHT_NIGHT_REPORT.md)
- System map CSV sheet filters: `tech_avenue` rows in [ORALABLE_SYSTEM_MAP.csv](../ORALABLE_SYSTEM_MAP.csv)
