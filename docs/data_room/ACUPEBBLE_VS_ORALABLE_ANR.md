# Bookmark — AcuPebble vs Oralable vs ANR M40

**As at:** 30 Aug 2026 · Pack **1.1.68**  
**Context:** Pedro Mayoral uses **AcuPebble** (Acurable) for home OSA today. **AcuPebble always uses finger ox.** Oralable is the temple optical layer. ANR M40 is research temporalis sEMG.  
**Related:** [MEASUREMENT_CONSTRUCT_MAP.md](./MEASUREMENT_CONSTRUCT_MAP.md) (living construct table — iterate there) · Pedro note [PEDRO_CONSTRUCT_MAP_NOTE.md](./PEDRO_CONSTRUCT_MAP_NOTE.md) / [PDF](./PEDRO_CONSTRUCT_MAP_NOTE.pdf) · [MAYORAL_METHOD_ORALABLE_VALIDATION.md](./MAYORAL_METHOD_ORALABLE_VALIDATION.md) · [PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md) · [ANR_M40_CONCORDANCE.md](./ANR_M40_CONCORDANCE.md) · [BRUXOFF_PSG_GOLD_STANDARD.md](./BRUXOFF_PSG_GOLD_STANDARD.md) · [DIANYX_FDA_AND_SMART_OAT_LANDSCAPE.md](./DIANYX_FDA_AND_SMART_OAT_LANDSCAPE.md) · [HAPPY_RING.md](./HAPPY_RING.md) (Oura-like finger HSAT — same AHI-class shelf, not Pedro’s current tool) · [OVERNIGHT_NIGHT_REPORT.md](../OVERNIGHT_NIGHT_REPORT.md)

**One-liner:** Three jobs, not one. AcuPebble = Pedro’s **OSA HSAT / AHI + finger SpO₂** (always finger ox). Oralable = temple **vitals and later jaw-load** with any MAD. ANR = **sEMG** bout timing. Dual A SpO₂∩EMG pairing + engineering SASHB are oxygen-burden context only (**not** Azarbarzin HB). Oralable does **not** replace AcuPebble AHI.

---

## Primary sources

| Source | URL / note |
|--------|------------|
| AcuPebble Ox100 (neck + finger SpO₂ HSAT) | [acurable.com — Ox100](https://acurable.com/en-us/products/acupebble-Ox100/) |
| AcuPebble SA100 (neck acoustic HSAT) | [acurable.com — SA100](https://acurable.com/en-us/products/acupebble-SA100/) |
| FDA 510(k) Ox100 | [K222950](https://www.accessdata.fda.gov/cdrh_docs/pdf22/K222950.pdf) — Class II ventilatory effort recorder / HSAT class |
| Azarbarzin hypoxic burden (literature) | Azarbarzin et al., *Eur Respir J* — event-linked SpO₂ area (needs scored respiratory events) |
| Oralable SASHB (implementation) | `ClinicalBiometricSuite` / [ALGORITHM_ARCHITECTURE.md](../ALGORITHM_ARCHITECTURE.md) — Σ(90 − SpO₂)·dt when SpO₂ &lt; 90% |
| ANR M40 path | [ANR_M40_CONCORDANCE.md](./ANR_M40_CONCORDANCE.md) |
| Oralable Arm P | [PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md) |

Acurable lists SA100 and Ox100. **This workspace: AcuPebble always uses finger ox.** Do not write Pedro methods as acoustic-only. Construct lock: [MEASUREMENT_CONSTRUCT_MAP.md](./MEASUREMENT_CONSTRUCT_MAP.md).

---

## SASHB vs Azarbarzin hypoxic burden

**Judgment:** Same *idea* (oxygen load under the curve), different definition. Oralable **SASHB** is an engineering SpO₂&lt;90 AUC. **Azarbarzin HB** needs scored apneas/hypopneas. Do not equate them. AcuPebble’s headline for Pedro is **AHI/ODI**, not either HB formula.

| | **Azarbarzin hypoxic burden (HB)** | **Oralable SASHB** | **AcuPebble (Pedro use)** |
|--|------------------------------------|--------------------|---------------------------|
| **What it is** | Sleep-medicine SpO₂ area **linked to scored respiratory events** | Engineering area: Σ(90 − SpO₂)·dt when SpO₂ &lt; 90% (%·s; often / wear hour) | Cleared home **OSA** test outputs |
| **Needs scored events?** | **Yes** (PSG / HSAT event list) | **No** — continuous SpO₂ only | Uses its own HSAT scoring for AHI/ODI |
| **SpO₂ source** | Finger / PSG ox typically | Temple PPG (empirical curve) | **Finger** SpO₂ (always) |
| **Oralable uses it?** | **No** — do not label SASHB as Azarbarzin HB | **Yes** — overnight report, Dual A nest, EDF `SASHB_cum` | Nest beside AcuPebble AHI; do not replace it |
| **AcuPebble uses it?** | Not what we claim from Pedro’s reports | No | **AHI / ODI** are the headline |
| **Say** | Literature / research HB when events exist | “Engineering SpO₂&lt;90 AUC (SASHB)” | “Pedro’s AHI/ODI reference” |
| **Do not say** | That Oralable SASHB = HB | That SASHB = AHI/ODI or Azarbarzin HB | That Oralable SpO₂/SASHB = AcuPebble AHI |

**Dual A nest:** Oralable SpO₂ / SASHB next to ANR EMG bout timing = AcuPebble-style *oxygen-burden context* (descriptive). Nest `desat_events_per_hour` is **not** ODI and **not** Azarbarzin HB.

---

## Comparison

| | **AcuPebble** (Pedro today) | **Oralable** Gen1 | **ANR M40** |
|--|----------------------------|-------------------|-------------|
| **Primary job** | Home **OSA** test / grade | Temple **HR/SpO₂** + later IR-DC jaw-load | **sEMG** muscle activity |
| **Site** | Neck base + **finger ox** | Extraoral **temporalis** | Temporalis electrodes |
| **Signals** | Neck acoustics (airflow/resp/snore/cardiac) + SpO₂/PPG/movement | PPG (R/G/IR) + ACC + temp | Analog EMG 0–1023 @ ~10 Hz |
| **Headline outputs** | **AHI, ODI**, severity report | HR, SpO₂, (later) HOI / TFI / state hypnogram | EMG amplitude / bouts |
| **Regulatory** | FDA Class II HSAT (e.g. Ox100 K222950) | Stage A wellness / research — **not** HSAT-cleared | Research / vendor EMG tool |
| **SB / bruxism** | Not its claim | Optical jaw-load path (Phase 1+) | Direct muscle electrical |
| **Fits Pedro** | Current **apnea truth** for MAD titration | **Arm P** companion: oxygen burden + wear **with any MAD** | Optional **Paper C** vs Oralable IR-DC |

---

## Nesting for Pedro (do not skip)

1. **AcuPebble** — airway / AHI reference he already trusts.  
2. **Oralable** — same-night or 1–2 h temple vitals + later brux/jaw-load alongside MAD ([MAYORAL_METHOD_ORALABLE_VALIDATION.md](./MAYORAL_METHOD_ORALABLE_VALIDATION.md)).  
3. **ANR** — research clench/grind timing vs IR-DC; orthogonal to OSA diagnosis.

```text
AcuPebble (AHI / ODI)     ← Pedro apnea reference
        │
Oralable Arm P (SpO₂ burden, wear)  ← complementary, not AHI clone
        │
ANR Dual A (EMG bouts)    ← Research Kit / Paper A descriptive precursor
                              (deeper PSG-AV / Bruxoff concordance → later)
```

**Dual A SpO₂∩EMG nest:** Mac `align_anr_oralable_concordance.py` joins Oralable SpO₂ / SASHB with ANR EMG bout timing (`NEST.md`, `metrics.json` → `spo2_emg_nest`). This is AcuPebble-style **oxygen-burden nesting**, not an AHI/ODI clone and not Bruxoff bout equivalence. See [ANR_M40_CONCORDANCE.md](../ANR_M40_CONCORDANCE.md).

---

## Claim discipline

| Do | Do not |
|----|--------|
| Call AcuPebble Pedro’s HSAT / AHI tool | Say Oralable SpO₂ / SASHB **=** AcuPebble AHI |
| Nest Oralable as overnight / Arm P **companion** to MAD + AcuPebble | Claim Oralable replaces AcuPebble or PSG |
| Report Dual A SpO₂∩EMG nest as descriptive burden + EMG timing | Treat nest `desat_events_per_hour` as ODI or AHI |
| Call SASHB engineering SpO₂&lt;90 AUC (%·s) | Call SASHB Azarbarzin hypoxic burden (needs scored events) |
| Use research `session.edf` for Dual A handoff | Call EDF a PSG or AcuPebble substitute |
| Use ANR for temporalis sEMG concordance research | Treat ANR concordance as OSA grading or SB diagnosis |
| Cite Acurable / FDA pages | Imply Acurable or ANR partnership unless contracted |

---

*Bookmark from founder comparison, 3 Aug 2026. Update if Pedro confirms SKU or a new AcuPebble clearance.*
