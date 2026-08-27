# Oralable® — Pitch deck (distilled for Ken / BalancePoints)

**App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

**Audience:** Ken Kinsella · [BalancePoints](https://balancepoints.co.uk) (Point A → Point B framing)  
**Status:** Working draft · July 2026 · strong for **Ken working session**; **not** F&F-close ready until valuation / terms / use-of-funds / runway / Voice  

**Present this:** open [`PITCH_DECK_KEN.html`](./PITCH_DECK_KEN.html) in a browser (← → keys) · PDF [`PITCH_DECK_KEN.pdf`](./PITCH_DECK_KEN.pdf).  
This `.md` file is the source distill for edits.

**Brand (HTML/PDF):** [`brand/oralable_logo_lockup.png`](./brand/oralable_logo_lockup.png) · **Oralable®** · **Word of Mouth™** · Open Sans · trademarks of JAC Dental Solutions Limited.

**Point A baseline (Ken 9 June 2026):** **1.5 / 5.0** Essential. Materially improved since: cash, COGS, Point B ask, cap table, market sketch, IP filings confirmed.

---

## Slide 1 — Title

**[Oralable® lockup]**  
**Word of Mouth™**

Temple wearable for overnight vitals → jaw-load awareness → (later) medical device  

**Stage A now:** patent-implementing **wellness wearable**  
**Stage B later:** **medical device** pathway (not claimed today)

Speaker: One sentence — we ship a working wearable that matches a new US patent filing first; clearance is a later gate.

---

## Slide 2 — Point A in one page (where we actually are)

| Dimension | Point A (July 2026) |
|-----------|---------------------|
| **Legal entity** | **JAC DENTAL SOLUTIONS LIMITED** (Ireland) |
| **Product** | Gen1 Phase 0 (temple HR/SpO₂) — **Research Kits built, gated** (not yet shipped); target **5 to Pedro by 31 Aug 2026**; charge-to-temple SOC ≥50% |
| **Hardware** | BOM **REV8** · PCB **REV10** · Kaga **ES2832AA2** · FW **1.0.82** · Oralable magnetic case (**not Qi**) |
| **App** | **Patient app only** for this iteration — Oralable for Dentists **off** |
| **Claims** | Wellness only — **not** a medical device |
| **IP** | **EP 4 333 691 B1** granted · **IE/UK** validating · **UP** due 8 Aug · US RCE · provisional **64/033,978** — [IP_PORTFOLIO_STATUS.md](./IP_PORTFOLIO_STATUS.md) · [IP_EVAL_AND_LANDSCAPE.md](./IP_EVAL_AND_LANDSCAPE.md) |
| **Revenue** | Near-zero live — hardware + IAP coded, soft launch gated |
| **Cash (17 Jul 2026)** | **€1,293** closing · period in €65k / out €73k · **~€63.5k director loans** (founder-funded) |
| **Cash plan (mid)** | Stage A eng ~€200–250k · **Ken Point B raise €180k by Oct 2026** — [FUNDING_POINT_B_AND_CAP_TABLE.md](./FUNDING_POINT_B_AND_CAP_TABLE.md) |
| **Point B ask stack** | **€50k F&F** (priced equity) + **€100k EI PSSF CLN** + **€30k HPSU grant** |
| **Cap table (JAC)** | John ~**63%** · Conor ~**21%** · angels ~**16%** (11,901 ords @ €0.01) — **Amstrow Cosec buyout of Conor in negotiation** (Jul 2026) |
| **Gen1 / Gen2 sample COGS** | Gen1 Kaga ~€3.9k · Gen2 Kaga **€3,735** · Bittele ~$3.46k — see COGS docs |
| **Still GAP for Ken** | F&F **valuation/terms** · CLN terms · use-of-funds · runway monthlies · Voice · volume EMS |

Speaker: Investors form Point A whether we write it down or not. This is the honest version.

---

## Slide 3 — Problem

Sleep bruxism (clench / grind) damages teeth, causes jaw pain, and ties to overnight oxygen burden — but:

- **Rings / watches** measure general wellness at finger/wrist — wrong site for jaw load  
- **Clinic EMG** is accurate but episodic, expensive, not overnight consumer  
- **Patients and dentists** lack a continuous, at-home optical + motion record of jaw hemodynamics

Speaker: A condition-specific problem at an unusual site — not “another sleep tracker.”

---

## Slide 4 — Solution

**Oralable** = Nordic BLE temple clip + magnetic charging case + iOS patient app  

| Now (Phase 0) | Next (Phase 1+) | Later (Stage B) |
|---------------|-----------------|-----------------|
| Temple **HR + SpO₂** with honest device state | **IR-DC** occlusion, **TFI**, **SASHB** in patient app | Cleared / CE **medical device** (separate gate) |
| Pilot evidence on Gen1 | Patent embodiment in product | Locked IFU / QMS |

**Physics:** PPG + IR-DC hemodynamic occlusion + jaw accelerometry — **not** sEMG, **not** a ring.

Speaker: Phase 0 proves the substrate; Phase 1+ puts the invention in software on the same Gen1 hardware.

---

## Slide 5 — Why we win (positioning)

```
General wellness          Condition-specific
        │                         │
Finger/wrist ── Oura, WHOOP, Happy Ring (hAHI)
        │                         │
Jaw / temple ──────────── Oralable (PPG / IR-DC)
                          ANR / Cometa (EMG clinic)
```

- **Orthogonal to rings** — same sensor classes, different physics  
- **Adjacent to dental / sleep** — consumer form factor, open 50 Hz pipeline (Swift + Python)  
- **Two-sided design** — patient app first; dentist share **after** Phase 1+ rollups (not Ed/Pedro)

**US market sketch (founder worksheet):** ~**4.1M** custom nightguard users → ~**1.4M** wearable-adopting “smart nightguard” prospects — [MARKET_SIZING.md](./MARKET_SIZING.md). Phase 0 does **not** claim that SOM yet.

Speaker: Do not sell Phase 0 as “bruxism solved.” Sell reliable temple vitals, then embodiment.

---

## Slide 6 — Product today (shipping truth)

| Item | Spec |
|------|------|
| Clip | Gen1 · REV10 · ES2832AA2 (nRF52832) |
| Charge | **Oralable magnetic case** (LTC4124 / LTC6990) — not phone Qi |
| Firmware | **1.0.82** (sense-on-BLE · IR-pulse worn · STAT blink = charging) |
| Placement | **Temple** · Automatic dock on 1.0.70+ (manual modes still available) |
| App | Oralable patient **4.3.3** · TestFlight vitals phase |
| Out of scope | Oralable for Dentists · Protocol B muscle · medical claims |
| **Handoff** | **Not yet shipped** — ship gate = charge to temple-ready SOC + short worn vitals |

**Pilot:** Ed & Pedro — patient app only · temple sessions · CSV evidence  
Pass Phase 0 gates → unlock Phase 1+ muscle / patent metrics. Gen2 (parallel) hardens charge/status for the next hardware era.

---

## Slide 7 — Strategy: Stage A → Stage B

```
Stage B — Medical device (H2 2027–2028+)
        ▲  evidence + locked claims + QMS
Stage A — Wellness wearable (ship first)
        ▲  patent-implementing product
US patent submission  ←  foundation grants (US & EU)
        ▲
Phase 1+ (TFI / SASHB / IR-DC) → Phase 0 (temple vitals) → Gen1 → Gen2
```

| Stage | Class | App | Money |
|-------|-------|-----|-------|
| **A** | Wellness wearable | Patient; pro deferred | Hardware + consumer IAP |
| **B** | Medical (later) | Patient + clinical workflows | Higher ASP; regulatory gate |

**Rule:** No FDA/CE claims in App Store, website, or Ed/Pedro materials during Stage A.

---

## Slide 8 — Intellectual property

| Track | Status (diligence) | Pitch-safe line |
|-------|--------------------|-----------------|
| **EP / WO foundation** | **WO2022234145** · **EP 4 333 691 B1** · IE/UK steps taken · UP due **8 Aug** | Sign IE AoA; UP receipt pending |
| **US utility 18/289,827** | Final OA → **RCE + response filed 12 Jun 2026** (Peacock) | Pending US monitoring apparatus (continued exam) |
| **New US provisional** | **64/033,978** filed **9 Apr 2026** — “Apparatus and Method for Muscle Activity Monitoring” | Provisional on file (Peacock) |
| **Case / enclosure** | Assignment to JAC in process | Hardware ecosystem protected |
| **Embodiment** | Phase 0 temple → Phase 1+ IR-DC / TFI / SASHB | Product practices the invention |

Detail: [IP_PORTFOLIO_STATUS.md](./IP_PORTFOLIO_STATUS.md) · [IP_EVAL_AND_LANDSCAPE.md](./IP_EVAL_AND_LANDSCAPE.md).  
**Do not** put claim text or unpublished provisional wording on the public website.

---

## Slide 9 — Go-to-market (two paths)

| Path | Buyer | Motion | Ed/Pedro |
|------|-------|--------|----------|
| **A — Consumer** | Individual | App Store + web hardware | **In scope** |
| **B — Professional** | Dental practice | KOL / pilots + share codes | **Deferred** until Phase 1+ |

**Pricing (coded, not all live):** Premium €9.99/mo · Dentist Professional €29.99/mo · Practice €99.99/mo (+ yearly)

**Market sketch (US):** SAM ~**4.1M** custom nightguards · SOM ~**1.4M** wearable × smart nightguard interest — [MARKET_SIZING.md](./MARKET_SIZING.md) *(sources still to harden)*.

**Year-1 draft targets (unproven):** 500 downloads · 20 dentist accounts · 50 share connections  
**Ken gap:** measured CAC · cite primary market sources · Ireland/UK cut

Geography: Ireland / UK first → EU → US; Android still **GAP**.

---

## Slide 10 — Traction (honest)

| Evidence | Status |
|----------|--------|
| Gen1 kits + FW 1.0.82 + app 4.3.3 | **Stack ready** (flash + TestFlight) — **not delivered** to Ed/Pedro yet |
| Ship gate (charge → worn vitals) | **In progress** — STAT from 1.0.70; 1.0.82 IR-pulse worn; closing energy/coupling on case |
| Ed/Pedro Phase 0 protocol + quick start | **Docs ready** — handoff after ship gate |
| Prior lab / temporalis validation runs | **Partial** (science; not consumer MAU) |
| App Store live MAU / MRR | **GAP** |
| Dentist paid seats | **N/A** until Path B |
| Pilot pass-rate metrics for investors | **Pending** Ed/Pedro field completion |

Speaker: Ready ≠ delivered. Traction after Ed/Pedro sessions = first customer-validation evidence Ken asked for.

---

## Slide 11 — Technology (diligence one-liner)

| Layer | Stack |
|-------|-------|
| Firmware | Zephyr / nRF · TGM GATT · MCUboot OTA · `oralable_nrf` |
| Sensors | MAXM86161 PPG · LIS2DTW12 ACC · 50 Hz stream |
| Apps | iOS patient + dentist (dentist gated) · `oralable_swift` |
| Shared | OralableCore — BLE parse, algorithms, CloudKit handshake |
| Science | Python gold standard · `cursor_oralable` · clinical protocols |
| Gen2 (parallel) | nRF54L15 · BOM REV9 / REV11 · ~30 mAh · FW 2.0.x scaffold |

Open pipeline (phone-side) vs a closed ring — helps IP embodiment and Stage B evidence.

---

## Slide 12 — Roadmap & cash (mid-case planning)

| Window | Focus | Mid cash (EUR) |
|--------|-------|----------------|
| **Now – Sep 2026** | Phase 0 Ed/Pedro | ~30–50k (P0) |
| **Q4 2026 – Q1 2027** | Phase 1+ embodiment + US patent file | Toward Stage A ~200–250k |
| **2026–2027** | Gen2 bring-up (parallel) | Stage A+Gen2 ~350–450k |
| **H2 2027 – 2028** | Stage B 510(k) / CE exploration | Through Stage B ~0.8–1.0M |

**Spend now:** pilot, Phase 1+ patient app, patent file, Gen2 EVT only when it helps Stage A quality.  
**Defer:** dentist app launch, full ISO audit, volume EMS PO before COGS is known.

*Ranges = order-of-magnitude planning — replace with EMS / Peacock / CRO / accountant quotes.*

---

## Slide 13 — Business model (what can actually bill)

```
Hardware (clip + case)  ──►  now / pilot (low volume)
Consumer Premium IAP    ──►  after credible Phase 0–1+ dashboard
Dentist Practice IAP    ──►  after Phase 1+ share + CloudKit
Cleared device (Stage B)──►  18–36+ months after wearable proof
```

**Risk if we sell too early:** marketing “bruxism” while shipping temple SpO₂ only.  
**Policy:** Ed/Pedro = patient evidence only — no professional IAP show before Phase 1+.

---

## Slide 14 — Cash & ask (for Ken session)

**Cash Point A:** closing **€1,293** (17 Jul) · ~**€63.5k** director loans · ~€0 product revenue — [FINANCIALS_CASH_SNAPSHOT.md](./FINANCIALS_CASH_SNAPSHOT.md).

**Point B ask (BalancePoints, Jun 2026):** **€180k by 31 Oct 2026** = **€50k F&F** priced equity + **€100k EI PSSF CLN** + **€30k HPSU grant** — [FUNDING_POINT_B_AND_CAP_TABLE.md](./FUNDING_POINT_B_AND_CAP_TABLE.md).

**Cap table:** John ~63% · Conor ~21% · angels ~16% (11,901 ords) — Conor purchase via Amstrow Cosec **in negotiation** (counters €10k/€15k; detail INTERNAL in [JAC governance](./JAC_CORPORATE_STRUCTURE_AND_GOVERNANCE.md)).

| Ready for discussion | Still needs fill |
|----------------------|------------------|
| Point B **structure** (€50 / €100 / €30) | F&F **pre-money valuation** + term summary |
| Cap table snapshot | PSSF **CLN terms** / dilution |
| Cash + sample COGS (Gen1/Gen2/Bittele) | **Use of funds** for €50k and for €180k |
| Product / Stage A→B story | 12-month **runway** monthlies |
| Ken Point A scores (1.5 Essential) | Founder **Voice** · statutory accounts into pack |

**Suggested next:** (1) F&F one-pager (ask + cap + terms + runway) · (2) Voice session · (3) confirm HPSU/PSSF eligibility with DA · (4) Kaga volume ladder.

---

## Slide 15 — Closing line

**Point A:** Working Gen1 temple wearable in pilot; wellness-only; patent path live; **cash thin and founder-loan funded**.  
**Point B:** Soft-launch Stage A embodiment → Gen2 nights → Stage B medical when evidence allows — **needs capital beyond director loans**.  

We do not claim to be a medical device today.  
We are building the wearable that makes that path possible.

---

## Appendix A — Speaker cheat sheet (30 seconds each)

1. **What:** Temple clip for overnight vitals → jaw-load metrics.  
2. **Who:** Adults + (later) dentists via patient share.  
3. **Why now:** Gen1 stack ready; Ed/Pedro handoff gated on charge-to-temple; IP filing in flight.  
4. **Moat:** Site + IR-DC physics + patent embodiment + open clinical pipeline.  
5. **Money:** Hardware first; SaaS after Phase 1+; medical ASP later.  
6. **Need:** Size ask vs Stage A ~€200–250k mid (cash ~€1.3k); Voice session; execute pilot.

---

## Appendix B — Map to Ken / BalancePoints areas

| Ken area | Deck slides | Status |
|----------|-------------|--------|
| Technology & Product | 4–6, 11–12 | PARTIAL → improving |
| Market & Positioning | 3–5 | PARTIAL ([MARKET_SIZING.md](./MARKET_SIZING.md) sketch; cites GAP) |
| GTM & Sales | 9, 13 | PARTIAL |
| User Traction & Revenue | 10, 13 | GAP until pilot + live |
| Financials | 12, 14 | PARTIAL (cash snapshot; ask/accounts still GAP) |
| Team & Governance | 14 + [current status](./CURRENT_GOVERNANCE_STATUS.md) + [JAC governance](./JAC_CORPORATE_STRUCTURE_AND_GOVERNANCE.md) | PARTIAL (Conor buyout live to 23 Jul; Nigel Independent Director; **proposed** Ken CEO; CRO/CEO contract GAP) |
| Legal, IP & Corp | 8 | PARTIAL ([IP_PORTFOLIO_STATUS.md](./IP_PORTFOLIO_STATUS.md) · [IP_EVAL_AND_LANDSCAPE.md](./IP_EVAL_AND_LANDSCAPE.md)) |
| Risk / Regulation | 7 | PARTIAL |
| International | 9 | GAP (Android) |
| Data Room & Investment Docs | This file + index | PARTIAL |
| Voice | 14 | GAP — session pending |

---

## Appendix C — Source documents (do not invent beyond these)

| Topic | Doc |
|-------|-----|
| Founder CV / pitch bio | [`JOHN_COGAN_CV.md`](./JOHN_COGAN_CV.md) |
| End goal | [`../IP_NORTH_STAR.md`](../IP_NORTH_STAR.md) |
| Phases / BOM | [`../PRODUCT_ROADMAP.md`](../PRODUCT_ROADMAP.md) |
| Cost / timeline | [`COST_AND_TIMELINE.md`](./COST_AND_TIMELINE.md) |
| Cash snapshot | [`FINANCIALS_CASH_SNAPSHOT.md`](./FINANCIALS_CASH_SNAPSHOT.md) |
| Point B ask + cap table | [`FUNDING_POINT_B_AND_CAP_TABLE.md`](./FUNDING_POINT_B_AND_CAP_TABLE.md) |
| Current governance snapshot | [`CURRENT_GOVERNANCE_STATUS.md`](./CURRENT_GOVERNANCE_STATUS.md) |
| Corporate structure / Ken & Nigel | [`JAC_CORPORATE_STRUCTURE_AND_GOVERNANCE.md`](./JAC_CORPORATE_STRUCTURE_AND_GOVERNANCE.md) |
| 22 Jul Ken/Nigel meeting brief | [`MEETING_BRIEF_KEN_NIGEL_2026-07-22.md`](./MEETING_BRIEF_KEN_NIGEL_2026-07-22.md) |
| Market sizing sketch | [`MARKET_SIZING.md`](./MARKET_SIZING.md) |
| IP portfolio status | [`IP_PORTFOLIO_STATUS.md`](./IP_PORTFOLIO_STATUS.md) |
| IP eval + landscape | [`IP_EVAL_AND_LANDSCAPE.md`](./IP_EVAL_AND_LANDSCAPE.md) |
| Gen1 / Gen2 / Bittele COGS | [`GEN1_COGS_KAGA_QUOTE.md`](./GEN1_COGS_KAGA_QUOTE.md) · [`GEN2_COGS_KAGA_QUOTE.md`](./GEN2_COGS_KAGA_QUOTE.md) · [`BITTELE_Q100918A1_PCB_QUOTE.md`](./BITTELE_Q100918A1_PCB_QUOTE.md) |
| GTM | [`GTM_ONE_PAGE.md`](./GTM_ONE_PAGE.md) |
| Apps / revenue | [`APPS_AND_REVENUE_EVAL.md`](./APPS_AND_REVENUE_EVAL.md) |
| Regulatory | [`REGULATORY_TIMELINE.md`](./REGULATORY_TIMELINE.md) |
| Pilot handout | [`ED_PEDRO_QUICK_START.md`](./ED_PEDRO_QUICK_START.md) |
| Data room index | [`README.md`](./README.md) |
| Market landscape | `oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md` |
| FTS | [`ORALABLE_FTS_36MO.md`](./ORALABLE_FTS_36MO.md) |

---

*Draft for Ken / BalancePoints · July 2026 · bump `docs/data_room/VERSION` when this becomes investor-final.*
