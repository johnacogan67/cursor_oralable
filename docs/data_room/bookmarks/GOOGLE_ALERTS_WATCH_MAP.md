# Google Alerts watch map — Tier 1 core · Tier 2 ROW

**As at:** 12 Sep 2026 · Pack **1.1.69** (watch list unchanged)  
**Agent slug:** `gtm` · create alerts at [google.com/alerts](https://www.google.com/alerts)  
**Tone:** operating watch list — **not** a TAM study · **not** Oralable clearance evidence · Stage A wellness  
**Related:** [GTM_ONE_PAGE.md](../governance/GTM_ONE_PAGE.md) · [MARKET_SIZING.md](../governance/MARKET_SIZING.md) · [ORALABLE_MARKET_LANDSCAPE.md](./ORALABLE_MARKET_LANDSCAPE.md) · [MEASUREMENT_CONSTRUCT_MAP.md](../clinical/MEASUREMENT_CONSTRUCT_MAP.md) · [REGULATORY_TIMELINE.md](../governance/REGULATORY_TIMELINE.md) · [IP_NORTH_STAR.md](../../IP_NORTH_STAR.md) · [DIANYX_FDA_AND_SMART_OAT_LANDSCAPE.md](./DIANYX_FDA_AND_SMART_OAT_LANDSCAPE.md)

**One-liner:** Watch **US, Canada, UK, and Europe as equals**. Watch **ROW weekly**, Australia first. Ireland is HQ (JAC, kits, Beacon), not a fifth core market. This is the inbox map; the GTM one-pager may still **ship** Ireland/UK first.

---

## 0. How to run

Paste each query into Google Alerts. Use quotes, `OR`, minus, and `site:`. Set **as-it-happens** on Tier 1 regulators and named competitors; **weekly** on societies, payors, and all ROW.

Gmail labels: `watch/shared` · `watch/us` · `watch/ca` · `watch/uk` · `watch/eu` · `watch/row` · `hardware` (China fab / clones only).

| Tier | Markets | Cadence |
|------|---------|---------|
| **1 — core** | US · Canada · UK · Europe (EU/EEA) | Equal alert **count**. Equal attention when a hit lands. US volume will be higher — do not add extra US alerts to “catch up”. |
| **2 — ROW** | Australia/NZ first · then Japan/Korea · then GCC · then LatAm/India | Weekly. Do not staff like a launch. |
| **Not a market** | China fab (Kaga / NCAB / Bittele) | `hardware` — clone and NMPA risk, not GTM |

**Do not alert on** `MAM`, `TGM`, `OMG`, `SASHB`, `Dual A`, `Protocol A`, or `rescue` alone. Too noisy, or no web hits yet.

**Do not** put provisional filing numbers or claim text in an alert.

Ship order in [GTM_ONE_PAGE.md](../governance/GTM_ONE_PAGE.md) (Ireland/UK → EU → US) is unchanged. This file is **what to learn in parallel**.

Ken **International Readiness** remains a GAP: [MARKET_SIZING.md](../governance/MARKET_SIZING.md) is a US nightguard sketch only. Alerts do not replace TAM cuts for CA / UK / EU / ROW.

---

## 1. Shared queries (no country in the query)

Maps to landscape §7 and the construct map. Peers sell into all Tier 1 regions.

| Alert name | Query | Why |
|------------|--------|-----|
| Brand | `"Oralable" OR "JAC Dental Solutions"` | Own name; over-claims |
| Brand medical wording | `"Oralable" (bruxism OR "sleep apnea" OR "FDA cleared" OR "medical device")` | Third-party medical language |
| GrindCare | `"GrindCare" (bruxism OR temporalis OR "sleep bruxism")` | Same **site class** (anterior temporalis), electrical not optical |
| Bruxoff / BiteStrip | `Bruxoff OR BiteStrip OR "dia-BRUXO" (bruxism OR RMMA)` | Ambulatory SB EMG class (Li 2025) |
| ANR / Cometa | `"ANR M40" OR "ANR Muscle Sense" OR "Cometa EMG" (bruxism OR temporalis OR masseter)` | Research EMG comparator |
| AcuPebble | `AcuPebble OR Acurable (AHI OR HSAT OR Ox100 OR SA100)` | Pedro’s AHI tool — **nest, do not clone** |
| Happy Ring / Belun | `"Happy Ring" OR "Happy Health" OR Belun OR NightOwl (HSAT OR hAHI OR 510(k) OR "Class II")` | Finger HSAT shelf — not jaw |
| Smart OAT | `Dianyx OR "t.e.s.a." OR SomnoMed OR ProSomnus ("Rest Assure" OR RPMO2 OR 510(k) OR "CE mark" OR "Health Canada" OR MHRA)` | Intraoral / MAD sensing — complementary anatomy |
| Optical myography | `"optical myography" OR (temporalis PPG) (bruxism OR masseter)` | MAM physics class |
| Ambulatory SB science | `"sleep bruxism" (ambulatory OR wearable OR RMMA) (PSG OR EMG)` | Li / PSG-AV trail |
| Hypoxic burden lit | `"hypoxic burden" Azarbarzin OR "event-related" (SpO2 OR "sleep apnea")` | **HB ≠ SASHB** — keep the names apart |
| Foundation IP | `"EP 4333691" OR "EP 4 333 691" OR WO2022234145 OR AESYRA` | Granted EP + cited competitor art |
| Silicon | `MAXM86161 OR nRF54L15 wearable` | Gen1 PPG / Gen2 MCU design-ins |

Leave Oura / WHOOP / Ultrahuman off unless you want a weekly dump. They sit **orthogonal** in the landscape (finger/wrist wellness).

---

## 2. Tier 1 — same four jobs, four regions

Copy the **job**, change the **geography**. UK is not Europe (MHRA / UKCA). Canada is not a US add-on (Health Canada; McGill / RI-MUHC).

### 2.1 Regulator (Stage B watch — not Stage A claims)

| Region | Query |
|--------|--------|
| **US** | `site:accessdata.fda.gov (bruxism OR "jaw muscle" OR masticatory OR HSAT OR "oral appliance")` |
| **Canada** | `("Health Canada" OR "Medical Device Licence" OR MDL) (bruxism OR "sleep apnea" OR wearable OR "oral appliance")` |
| **UK** | `(MHRA OR UKCA OR "UKCA mark") (bruxism OR "sleep monitor" OR HSAT OR "oral appliance" OR wearable)` |
| **Europe** | `(MDR OR "Class IIa" OR "notified body" OR "CE mark") (bruxism OR "sleep monitor" OR HSAT OR "oral appliance" OR wearable)` |

Optional US product-code watch: `(MNR OR MWI OR PLC) (510(k) OR "Class II") (sleep OR bruxism OR "oral appliance" OR PPG)`.

### 2.2 Dental-sleep / sleep-medicine societies

| Region | Query |
|--------|--------|
| **US** | `AADSM OR ABDSM (bruxism OR "oral appliance" OR wearable)` |
| **Canada** | `("Canadian Sleep Society" OR CSS OR "Canadian Academy" "dental sleep" OR McGill OR "RI-MUHC") (bruxism OR sleep OR wearable OR PPG)` |
| **UK** | `("British Society of Dental Sleep Medicine" OR BSDSM) (bruxism OR OAT OR "home sleep")` |
| **Europe** | `(ESRS OR "European Sleep" OR EADSM OR "European Academy" "dental sleep") (bruxism OR OAT OR wearable)` |

Ed/Pedro already sit on AADSM / ABDSM. Keep the Canadian Sleep Society / McGill line so Canada is not only a US-society leak.

### 2.3 Trade press

| Region | Query |
|--------|--------|
| **US** | `site:sleepreviewmag.com (bruxism OR "oral appliance" OR wearable OR PPG)` |
| **Canada** | `("JCDA" OR "CDA Oasis" OR "Canadian Journal of Dental") (bruxism OR sleep OR "oral appliance")` |
| **UK** | `(BDJ OR "British Dental Journal" OR "Dentistry.co.uk") (bruxism OR "dental sleep" OR "oral appliance")` |
| **Europe** | `("Sleep Medicine" OR "European Respiratory") (bruxism OR "oral appliance" OR wearable) (Germany OR France OR Netherlands OR EU)` |

### 2.4 Payor / system (market structure, not a billing plan)

| Region | Query |
|--------|--------|
| **US** | `(CMS OR Medicare OR RPM OR CPT) ("oral appliance" OR E0486 OR bruxism OR "dental sleep")` |
| **Canada** | `(OHIP OR RAMQ OR provincial) (bruxism OR "oral appliance" OR "sleep apnea") (dental OR wearable)` |
| **UK** | `(NICE OR NHS OR "NHS England") (bruxism OR "mandibular advancement" OR "home sleep" OR wearable)` |
| **Europe** | `(reimbursement OR "health technology assessment" OR HTA) (bruxism OR "oral appliance" OR HSAT) (Germany OR France OR Netherlands)` |

RPM / CPT hits are intel. Stage A is wellness. Do not treat them as an Oralable billing plan.

### 2.5 App Store / consumer (Path A)

| Region | Query |
|--------|--------|
| **US** | `"App Store" (bruxism OR "teeth grinding") (US OR "United States")` |
| **Canada** | `"App Store" (bruxism OR bruxisme OR "teeth grinding") (Canada OR Canadian)` |
| **UK** | `"App Store" (bruxism OR "teeth grinding") (UK OR "United Kingdom")` |
| **Europe** | `(bruxism OR "Zähneknirschen" OR "bruxisme") (app OR wearable) (Germany OR France OR "App Store")` |

### 2.6 Privacy (professional app later)

| Region | Query |
|--------|--------|
| **US** | `(HIPAA OR FTC) (wearable OR "sleep app") (health OR "medical claim")` |
| **Canada** | `(PIPEDA OR "Law 25" OR Quebec privacy) (health OR wearable) (sleep OR dental)` |
| **UK** | `("UK GDPR" OR ICO) (wearable OR "health data") (sleep OR dental)` |
| **Europe** | `GDPR (wearable OR "health app" OR "medical device") (sleep OR dental)` |

Quebec Law 25 matters because RI-MUHC / McGill sits there.

### 2.7 IP (portfolio already four-legged; Canada CIPO is a GAP)

| Region | Query |
|--------|--------|
| **US** | `site:patents.google.com (masseter OR temporalis) (PPG OR photoplethysmography OR "optical myography")` |
| **Canada** | `(CIPO OR "Canadian patent") (bruxism OR masseter OR temporalis) (PPG OR wearable)` |
| **UK** | `(UKIPO OR "European patent" UK) (masseter OR bruxism) (optical OR PPG)` |
| **Europe** | `("unitary patent" OR EPO) (bruxism OR masseter) (optical OR PPG)` |

Counsel mail (Peacock / PurdyLucey) stays with counsel. Alerts are for **other people’s** filings and press.

---

## 3. Tier 1 starter (create these first — 20)

**Shared (8)**

1. `"Oralable" OR "JAC Dental Solutions"`
2. `"GrindCare" (bruxism OR temporalis)`
3. `Bruxoff OR BiteStrip (bruxism OR RMMA)`
4. `AcuPebble OR Acurable (AHI OR HSAT)`
5. `Dianyx OR SomnoMed OR ProSomnus (510(k) OR "CE mark" OR MHRA OR "Health Canada")`
6. `"optical myography" OR (temporalis PPG) (bruxism OR masseter)`
7. `"EP 4333691" OR WO2022234145 OR AESYRA`
8. `MAXM86161 OR nRF54L15 wearable`

**US (3)** · **Canada (3)** · **UK (3)** · **Europe (3)** — regulator + society + payor from §2.1, §2.2, §2.4.

That is equal **count**. A single Health Canada or MHRA hit counts the same as a week of 510(k) summaries.

---

## 4. Tier 2 — ROW (add after the twenty; all weekly)

ROW is not one query. Split by **why you care**. Australia is the only ROW cluster already in the pack (SomnoMed ASX; Li 2025 in *Aust Dent J*).

### 4.1 Australia / NZ — first ROW watch

| Alert name | Query |
|------------|--------|
| TGA | `TGA ("sleep" OR bruxism OR "oral appliance" OR "home sleep") (wearable OR monitor OR ARTG)` |
| SomnoMed ASX | `SomnoMed (ASX OR "Rest Assure" OR TGA)` — skip if already in shared §1 |
| AU dental sleep | `("Australian Dental Journal" OR "dental sleep") (bruxism OR "oral appliance" OR wearable) (Australia OR NZ)` |

If TGA + SomnoMed starts to look like a real English-language path, promote Australia to **Tier 1.5** (still behind US/CA/UK/EU).

### 4.2 Japan / Korea

| Alert name | Query |
|------------|--------|
| PMDA / MFDS | `(PMDA OR MFDS) (bruxism OR "sleep apnea" OR wearable OR PPG)` |

### 4.3 Gulf private-pay (one alert)

| Alert name | Query |
|------------|--------|
| GCC dental sleep | `("dental sleep" OR bruxism OR "oral appliance") (UAE OR Dubai OR Saudi OR Qatar) (wearable OR clinic)` |

### 4.4 LatAm / India (one alert)

| Alert name | Query |
|------------|--------|
| ANVISA / CDSCO | `(ANVISA OR CDSCO) (bruxism OR "sleep apnea" OR "oral appliance" OR wearable)` |

### 4.5 China — hardware, not GTM

| Alert name | Query | Label |
|------------|--------|--------|
| Clone / NMPA | `(NMPA OR "Made in China") (bruxism OR "night guard" OR temporalis OR "cheek PPG") (wearable OR sensor)` | `hardware` |

---

## 5. ROW starter (6 weekly)

1. `TGA ("sleep" OR bruxism OR "oral appliance") (wearable OR ARTG)`
2. `SomnoMed (ASX OR TGA OR "Rest Assure")` — omit if already in shared
3. `("Australian Dental Journal" OR "dental sleep" Australia) (bruxism OR wearable)`
4. `(PMDA OR MFDS) (bruxism OR "sleep apnea" OR wearable OR PPG)`
5. `("dental sleep" OR bruxism) (UAE OR Dubai OR Saudi) (wearable OR "oral appliance")`
6. `(ANVISA OR CDSCO) (bruxism OR "sleep apnea" OR wearable)`

---

## 6. How to read a hit

| Source | Treat as |
|--------|----------|
| FDA · Health Canada · MHRA · MDR | Tier 1 — same importance |
| AADSM · CSS · BSDSM · ESRS | Tier 1 channel |
| TGA · PMDA · SomnoMed ASX | ROW competitor / future path |
| NMPA · Made-in-China PPG clips | Hardware / IP risk |
| India / Brazil device news | File; do not plan |

**Construct lock** ([MEASUREMENT_CONSTRUCT_MAP.md](../clinical/MEASUREMENT_CONSTRUCT_MAP.md)) — all geographies:

- MAM = extraoral temporalis optical. **No finger ox.** Not AHI. Not Azarbarzin HB.
- AcuPebble / Happy Ring = HSAT nest. Pedro’s AHI tool stays **AcuPebble**.
- Dual A / ANR = research EMG clock.
- SASHB = engineering temple SpO₂&lt;90 area. Do not rename to HB or ODI.

**Stage A copy** in every store: no “FDA cleared,” no “CE medical,” no “Health Canada licensed,” no “UKCA medical” until that gate is real.

---

## 7. Noise

| Region | Typical junk | Tighten with |
|--------|----------------|--------------|
| US | Wellness rings, OSA pharma | `bruxism OR "oral appliance" OR masticatory` |
| Europe | Generic MDR consultancies | Named device class + bruxism/HSAT |
| UK | NHS IT programmes | `bruxism OR mandibular OR MHRA` |
| Canada | Provincial dental-fee schedules | `sleep OR bruxism OR wearable OR PPG` |
| ROW | Tourism and generic “digital health” | Named regulator (TGA, PMDA) + bruxism/OAT |

PubMed saved searches catch paywalled papers Google Alerts miss: `temporalis AND (PPG OR photoplethysmography)` and `sleep bruxism AND (ambulatory OR wearable)`.

---

## 8. Claim discipline

- Do **not** treat a 510(k), CE, UKCA, or Health Canada **hit about someone else** as Oralable status.
- Do **not** equate AHI, ODI, or Azarbarzin HB with SASHB or Dual A pairing.
- Do **not** put provisional numbers or claim text in an alert.
- Do **not** use this file to fill Ken Market TAM. US nightguard math stays in [MARKET_SIZING.md](../governance/MARKET_SIZING.md); CA/UK/EU/ROW cuts are still GAP.
- Re-check the construct map before pasting a press claim into a pitch.

---

*Watch map from workspace landscape + construct map. Rechecked 12 Sep 2026 (no new K-number in Dianyx public pages). Update when a named competitor clears a new K-number / CE / UKCA / MDL, or when Australia is promoted to Tier 1.5.*
