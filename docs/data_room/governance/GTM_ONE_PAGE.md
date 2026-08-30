# Go-to-market — one page

**Version:** 1.2.0 · **July 2026**

**Related:** [data_room/README.md](./README.md) · [PRODUCT_ROADMAP.md](../PRODUCT_ROADMAP.md) · [IP_NORTH_STAR.md](../IP_NORTH_STAR.md) · [REGULATORY_TIMELINE.md](./REGULATORY_TIMELINE.md) · [COST_AND_TIMELINE.md](./COST_AND_TIMELINE.md) · `oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md` §11 · [../FIGURES.md](../FIGURES.md) · **App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

---

**Strategy stack:** Stage A wellness wearable → Stage B medical (later) · new US patent embodiment · Ed/Pedro = patient app only.  
**Cost & timeline (planning):** [COST_AND_TIMELINE.md](./COST_AND_TIMELINE.md) — Phase 0 now–Sep 2026; Phase 1+ Q4’26–Q1’27; Gen2 parallel; Stage B H2’27–2028. Mid Stage A ~€200–250k; Stage A+Gen2 ~€350–450k; through Stage B ~€0.8–1.0M (ranges — not a budget).

![FIG-CO-021 Temple lifestyle](../figures/FIG-CO-021-system-stack-photo.svg)

*Figure FIG-CO-021 — Temple lifestyle photo (placeholder; external-safe).*

## Positioning (Phase 0 · Stage A wellness wearable)

**For:** Adults validating overnight temple vitals (HR/SpO₂) with Oralable Gen1 kits; later jaw-load awareness (Phase 1+).  
**Product:** Gen1 clip (**BOM REV8** / **REV10** / **ES2832AA2** / FW **1.0.84**) + **Oralable magnetic charging case** + **Oralable** patient iOS app (**4.3.3** build **5**).  
**Promise (Phase 0):** Reliable temple heart rate and SpO₂ with honest device state — **not** a diagnosis · **not** a medical device.  
**Promise (Phase 1+):** See overnight jaw activity patterns (wellness wearable on the new US patent path).  
**Later (Stage B):** Medical-device pathway — separate regulatory gate; see [REGULATORY_TIMELINE.md](./REGULATORY_TIMELINE.md) · [IP_NORTH_STAR.md](../IP_NORTH_STAR.md).

**Ed/Pedro iteration:** **Patient app only** — do **not** ship or activate **Oralable for Dentists** until Phase 0 gates pass and Phase 1+ shareable rollups exist. See [APPS_AND_REVENUE_EVAL.md](./APPS_AND_REVENUE_EVAL.md).

**Voice (target):** Lead with user outcomes — sleep quality, morning jaw comfort — not sensor specs; no medical claims in Stage A.

**Hardware note:** Charge only in the Oralable case (not a phone Qi pad). Gen2 (BOM REV9 / REV11) is upcoming — see PRODUCT_ROADMAP.

---

## Two paths (simultaneous)

| Path | Buyer | Motion | App | Ed/Pedro Phase 0 |
|------|-------|--------|-----|------------------|
| **A — Consumer** | Individual | App Store, website, Amazon/dental retail | **Oralable** (patient) | **In scope** |
| **B — Professional** | Dental practice | KOL referral, conferences, pilot sites | Patient + **Oralable for Dentists** | **Deferred** — activate post Phase 1+ |

Path B does not need FDA clearance at launch if copy stays **wellness monitoring**; keep Path B dark for Ed/Pedro. Match professional metadata to consumer disclaimers until Path C.

---

## Pricing (App Store — configured in code)

| Product | Price (EUR) | Notes |
|---------|-------------|-------|
| Oralable Premium monthly | €9.99/mo | Consumer analytics / export tier |
| Oralable Premium yearly | €99.99/yr | |
| Dentist Professional monthly | €29.99/mo | Up to ~50 participants |
| Dentist Professional yearly | €299.99/yr | |
| Dentist Practice monthly | €99.99/mo | Higher participant cap |
| Dentist Practice yearly | €999.99/yr | |

**Hardware:** Clip sold separately (margin + bundle with first-month promo). **Elasticity study:** post-launch A/B — not yet modelled (Ken gap).

---

## Customer acquisition (assumptions — modelled, not proven)

| Channel | Role | CAC assumption (draft) |
|---------|------|------------------------|
| **Dentist referral** | Primary B2B2C — leaflet + share-code setup in practice | Low direct CAC; practice subscription drives B side |
| **App Store organic** | Search: bruxism, teeth grinding, jaw pain | €15–40 modeled CPI (category benchmark) |
| **Website / content** | oralable.com, researcher pages | Content + SEO |
| **Dental conferences** | Demo + pilot recruitment | Event cost allocated per lead |

**Year-1 target (draft):** 500 consumer downloads · 20 dentist accounts · 50 active share connections.  
**Ken gap:** Replace assumptions with measured CAC after 90 days live.

---

## Sales model

| Segment | Model |
|---------|--------|
| Consumer | **Direct** digital — App Store, web store for hardware |
| Dentist | **Direct** practice subscription + **patient-mediated** data (share code) |
| Future | Dental distributor / OEM white-label (not in Year-1 plan) |

---

## International (outline)

| Phase | Geography | Requirement |
|-------|-----------|-------------|
| Launch | **Ireland / UK** | English App Store; **wellness positioning** (no CE medical mark on consumer SKU — [REGULATORY_TIMELINE.md](./REGULATORY_TIMELINE.md)) |
| +6 mo | **EU core** (DE, FR) | Localization v1.1; GDPR already addressed |
| +12 mo | **US** primary if not simultaneous | FDA wellness; clearance path separate |
| Scale | **Android** | Required for EU/Android-heavy referrals — H2 2026 MVP |

Formal expansion plan: detail after UK/Ireland traction (Ken: HIGH priority, not CRITICAL).

---

## Launch dependencies (next 30 days)

1. Ed/Pedro **Phase 0 vitals** on **patient app only** (no professional app)  
2. App Store metadata + screenshots (simplified vitals dashboard)  
3. App Store Connect IAP live *(consumer; Path B later)*  
4. CloudKit production schema *(needed before activating Oralable for Dentists)*  

---

## Metrics to report investors (post-launch)

| Metric | Source | Ed/Pedro Phase 0 |
|--------|--------|------------------|
| MAU / nights recorded | App analytics | Patient app |
| Pilot protocol pass rate | Vitals test plan / clinical docs | Patient app evidence |
| Churn / 3-night retention | Pilot + App Store | Patient app |
| Share connections created | CloudKit / share codes | **N/A until Path B** |
| Dentist paid accounts | StoreKit | **N/A until Path B** |

*Expand to full GTM spreadsheet · pitch distill: [PITCH_DECK_KEN.md](../pitches/PITCH_DECK_KEN.md) · Ken CRITICAL follow-on: Voice + Financials.*
