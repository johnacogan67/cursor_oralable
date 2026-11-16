# Go-to-market — one page

**Version:** 1.2.3 · **3 Oct 2026** · Pack **1.1.74**

**Related:** [data_room/README.md](./README.md) · [PRODUCT_ROADMAP.md](../../PRODUCT_ROADMAP.md) · [IP_NORTH_STAR.md](../../IP_NORTH_STAR.md) · [REGULATORY_TIMELINE.md](./REGULATORY_TIMELINE.md) · [COST_AND_TIMELINE.md](./COST_AND_TIMELINE.md) · [ORALABLE_MARKET_LANDSCAPE.md](../bookmarks/ORALABLE_MARKET_LANDSCAPE.md) §11 · [GOOGLE_ALERTS_WATCH_MAP.md](../bookmarks/GOOGLE_ALERTS_WATCH_MAP.md) (US/CA/UK/EU equal watch · ROW weekly) · [../FIGURES.md](../../FIGURES.md) · **App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

---

**Strategy stack:** Stage A wellness wearable → Stage B medical (later) · new US patent embodiment · Beacon = **patient app only** · **clinic loaner** (Path 0), not App Store home charge. Paper A n≈5 nights are **unpaid**. Path 0 **RKaaS** (company-owned kits + per-report fee) is a later clinic SKU, not n≈5 cash and not Point B.  
**Cost & timeline (planning):** [COST_AND_TIMELINE.md](./COST_AND_TIMELINE.md) — Phase 0 continues past Sep 2026 (**30 Sep** handoff missed); Phase 1+ Q4’26–Q1’27 after Phase 0 gates; Gen2 parallel **or skip if Gen4 pod is next**; Stage B H2’27–2028. Mid Stage A ~€200–250k; Stage A+Gen2 ~€350–450k; through Stage B ~€0.8–1.0M (ranges — not a budget).

![FIG-CO-021 Temple lifestyle](../figures/FIG-CO-021-system-stack-photo.svg)

*Figure FIG-CO-021 — Temple lifestyle photo (placeholder; external-safe).*

## Positioning (Phase 0 · Stage A wellness wearable)

**For:** Beacon clinic-loaner nights first (HR/SpO₂, ≥4 h); later home jaw-load awareness (Phase 1+).  
**Product:** Gen1 clip (**BOM REV8** / **REV10** / **ES2832AA2** / FW **1.0.85**) + **Oralable charge case (clinic; coil align now; magnetic dock later)** + pads/tape + **Oralable** patient iOS app (**4.3.3** build **5**).  
**Promise (Phase 0):** Reliable temple heart rate and SpO₂ with honest device state — **not** a diagnosis · **not** a medical device.  
**Promise (Phase 1+):** See overnight jaw activity patterns (wellness wearable on the new US patent path).  
**Later (Stage B):** Medical-device pathway — separate regulatory gate; see [REGULATORY_TIMELINE.md](./REGULATORY_TIMELINE.md) · [IP_NORTH_STAR.md](../../IP_NORTH_STAR.md).

**Beacon iteration:** **Patient app only** — clinic charges the case; patient does not take a charger home. Do **not** ship or activate **Oralable for Dentists** until Phase 0 gates pass. See [APPS_AND_REVENUE_EVAL.md](./APPS_AND_REVENUE_EVAL.md) · meeting [ED_PEDRO_MEETING_2026-09-07.md](../clinical/ED_PEDRO_MEETING_2026-09-07.md).

**Voice (target):** Lead with user outcomes — sleep quality, morning jaw comfort — not sensor specs; no medical claims in Stage A.

**Hardware note:** Charge only in the Oralable case (not a phone Qi pad). Gen2 (BOM REV9 / REV11) can run in parallel **or skip** if Gen4 combined pod is the next enclosure — see PRODUCT_ROADMAP.

---

## Three paths (do not mix Beacon with Path A)

| Path | Buyer | Motion | App | Beacon Phase 0 |
|------|-------|--------|-----|----------------|
| **0 — Clinic loaner** | Sleep clinic (Beacon) | One night · clinic charge · pad/tape · ≥4 h · Oralable-only | **Oralable** (patient) | **Now** |
| **A — Consumer** | Individual | App Store, website, home charge case | **Oralable** (patient) | **Later** — different SKU |
| **B — Professional** | Dental practice | KOL referral, conferences | Patient + **Oralable for Dentists** | **Deferred** — activate post Phase 1+ |

Path **0** is the 7 Sep Ed use rule. Path A (consumer App Store + home charge) is a later product. Do not write Paper A methods as Path A. Path B stays dark for Beacon. Wellness copy only until Path C / Stage B.

### Path 0 commercial — Research Kit as a Service (later SKU)

Company: **JAC Dental Solutions Limited** (not a separate “MAM Diagnostics” entity). Hardware stays company inventory. The clinic loans the kit for one night; the patient does not take the charger home. Cash, when billed, is a **processing fee per evaluable overnight report** (Clinical Temporalis PDF) — not a kit sale and not an App Store IAP.

| Window | Stance |
|--------|--------|
| **Paper A n≈5** | **€0.** Collaboration + operating evidence. Do not invoice Ed, Pedro, or Beacon for feasibility nights. Five nights × €100 is not runway. |
| **After first evaluable nights** | Optional clinic pilot only if the site asks. Lease terms, who pays (clinic vs patient), and wellness invoice copy still **open**. Ethics / consent named first. |
| **Later Path 0 SKU** | Sleep-clinic product: lease + per-report fee. Distinct from Path A home-charge. |

**Draft rate (not a signed rate card):** **€100** technical operator fee per processed overnight report. Wellness measurement only — not HSAT, not AHI, not a bruxism diagnosis.

**Scale (illustration, not a 2026 plan):** five kits rotating at ~20 nights/month would be ~€10k/month **if** a clinic later ran that motion. Do not put that number in a Point B use-of-funds or a Beacon term sheet.

**Do not:** treat RKaaS as a substitute for Point B (€180k); mix this fee with Path A / Path B IAP; bill n≈5 as a paid service; send Ed a term sheet until charge-to-temple clears and Beacon names ethics / consent.

---

## Pricing

### Path 0 — clinic processing (draft)

| Product | Price (EUR) | Notes |
|---------|-------------|-------|
| Technical operator fee | **€100** / processed overnight report | RKaaS. Company keeps hardware. **n≈5 = €0.** Later Path 0 SKU only. |

### Path A / B — App Store (configured in code)

| Product | Price (EUR) | Notes |
|---------|-------------|-------|
| Oralable Premium monthly | €9.99/mo | Consumer analytics / export tier |
| Oralable Premium yearly | €99.99/yr | |
| Dentist Professional monthly | €29.99/mo | Up to ~50 participants |
| Dentist Professional yearly | €299.99/yr | |
| Dentist Practice monthly | €99.99/mo | Higher participant cap |
| Dentist Practice yearly | €999.99/yr | |

**Hardware (Path A later):** Clip sold separately (margin + bundle with first-month promo). Path 0 does **not** sell the clip to the clinic. **Elasticity study:** post-launch A/B — not yet modelled (Ken gap).

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
| **Sleep clinic (Path 0)** | **Lease** (company-owned kits) + **per-report** processing fee **after** n≈5. Not a kit sale. Not n≈5 cash. |
| Consumer (Path A, later) | **Direct** digital — App Store, web store for hardware |
| Dentist (Path B, later) | **Direct** practice subscription + **patient-mediated** data (share code) |
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

**Watch (not ship):** US, Canada, UK, and Europe as equals; ROW weekly, Australia first — [GOOGLE_ALERTS_WATCH_MAP.md](../bookmarks/GOOGLE_ALERTS_WATCH_MAP.md). Ireland is HQ, not a fifth core market.

---

## Launch dependencies (as at 3 Oct 2026)

The “next 30 days” list from 19 Sep is still the list. The **30 Sep** handoff passed with the gate open.

1. **Charge-to-temple** on five Oralable-only kits (still **gated**; **30 Sep** missed; no new date)  
2. Gen1 **pad-attach soak** without headband (`couple` logged)  
3. Send [PEDRO_STATUS_UPDATE_2026-09.md](../clinical/PEDRO_STATUS_UPDATE_2026-09.md) if not already sent (use rules only)  
4. Beacon **ethics / consent** name even if this stays wellness feasibility  
5. **Do not invoice** n≈5 and **do not** send an RKaaS term sheet until charge-to-temple clears  

App Store IAP, screenshots, and CloudKit production schema are **Path A / Path B** work — not Beacon Path 0.  

---

## Metrics to report investors (post-launch)

| Metric | Source | Ed/Pedro Phase 0 |
|--------|--------|------------------|
| MAU / nights recorded | App analytics | Patient app |
| Pilot protocol pass rate | Vitals test plan / clinical docs | Patient app evidence |
| Churn / 3-night retention | Pilot + App Store | Patient app |
| Path 0 processing invoices | Clinic billing (RKaaS) | **N/A — n≈5 unpaid** |
| Share connections created | CloudKit / share codes | **N/A until Path B** |
| Dentist paid accounts | StoreKit | **N/A until Path B** |

*Expand to full GTM spreadsheet · pitch distill: [PITCH_DECK_KEN.md](../pitches/PITCH_DECK_KEN.md) · Ken CRITICAL follow-on: Voice + Financials.*
