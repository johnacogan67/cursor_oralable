# Cost & timeline map — Stage A wearable → Stage B medical

**Status:** Planning estimate · July 2026 · **not a budget or fundraising model**  
**Currency:** EUR (aligns with GTM IAP). Ranges are **order-of-magnitude** industry / SMED-wearable norms — replace with Peacock, EMS, CRO, and accountant quotes.

**Related:** [IP_NORTH_STAR.md](../../IP_NORTH_STAR.md) · [PRODUCT_ROADMAP.md](../../PRODUCT_ROADMAP.md) · [REGULATORY_TIMELINE.md](./REGULATORY_TIMELINE.md) · [GTM_ONE_PAGE.md](./GTM_ONE_PAGE.md) · [APPS_AND_REVENUE_EVAL.md](./APPS_AND_REVENUE_EVAL.md) · **App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

**Ken gaps (still open):** F&F valuation/terms · PSSF CLN terms · use-of-funds · runway monthlies · volume kit COGS · Voice.  
**Near-term ask (Point B):** [FUNDING_POINT_B_AND_CAP_TABLE.md](./FUNDING_POINT_B_AND_CAP_TABLE.md) — €50k F&F + €100k PSSF + €30k HPSU = **€180k by Oct 2026**.  
**Pitch / cash / COGS:** [PITCH_DECK_KEN.md](../pitches/PITCH_DECK_KEN.md) · [FINANCIALS_CASH_SNAPSHOT.md](./FINANCIALS_CASH_SNAPSHOT.md) · Gen1/Gen2/Bittele COGS docs.

**Propagated to:** product roadmap, IP north star, Gen1/Gen2 docs, data-room pilots, GTM, regulatory, FTS, system architecture, nRF DEVELOPMENT/README/landscape, iOS MOBILE_APP_FLOWS, website WEBSITE.md (internal only — no € on public pages).

---

## 1. Timeline (likely)

**Canonical calendar:** [PRODUCT_ROADMAP.md §3](../../PRODUCT_ROADMAP.md#3-timeline-calendar--canonical). Do not invent alternate phase dates here.

```
2026 Jul 24    ENG MILESTONE (shipped)
               Protocol A Mac · Core ML Tier 0 · night-report PDF + in-app hypnogram
                      │
2026 Jul–Sep    PHASE 0 / Ed–Pedro (Stage A) — YOU ARE HERE
               Gen1 stack ready · Research Kits gated (5 by 31 Aug) · temple vitals · Dual A Mac optional · patient app only
                      │
2026 Q4–2027 Q1  PHASE 1+ embodiment (Stage A)
               TFI / SASHB / IR-DC live in patient UX · Protocol B · ≥6 h overnight eval
               Core ML Tier 1 cohort · soft Premium optional · professional app still gated
                      │
2026 Q4–2027 H2  GEN2 bring-up (parallel)
               REV11 EVT · FW 2.0.x · vitals parity · longer nights
                      │
2026–2027        IP — new US patent submission / prosecution (counsel)
                      │
2027 H1–H2       STAGE B pre-sub + clinical package
               Predicate · study report · QMS climb
                      │
2027 H2–2028     STAGE B 510(k) / CE exploration
               Cleared tier only after gate — not App Store wellness claims
```

| Window | Focus | Exit criteria |
|--------|-------|---------------|
| **24 Jul 2026** | Eng overnight + Core ML Tier 0 | PDF/Mac pack + mlpackage retrain (done) |
| **Now – Sep 2026** | Phase 0 Ed/Pedro Research Kit | Temple HR/SpO₂ gates; CSV evidence; **target 5 Research Kits to Pedro by 31 Aug 2026** after charge-to-temple (currently **gated**); Dual A Mac optional; no pro app |
| **Q4 2026 – Q1 2027** | Phase 1+ Stage A | TFI/SASHB/IR-DC live in patient UX; Protocol B; overnight ≥6 h eval; patent-table exports |
| **Q4 2026 – H2 2027** | Gen2 HW/FW | G2-P0…P6; temple parity; CHRSTS/SOC better than Gen1 |
| **2026–2027** | US patent | Non-provisional submitted; prosecution as needed |
| **H1 2027+** | Stage B prep | Pre-sub package; clinical concordance plan funded |
| **H2 2027–2028** | Stage B filing | 510(k) target; CE MDR exploration parallel |

**Calendar risk:** Stage B slips if Phase 0 fails, Gen2 EVT is late, or clinical recruitment is slow. Stage A wearable can still ship on Gen1.

---

## 2. Cost map by workstream (EUR ranges)

Assumptions: lean team (founder plus part-time contractors), Ireland/EU base, Gen1 kits already partially tooled, ~10–30 pilot units then ~100 Stage A soft-launch, Gen2 one EVT spin before pilot parity.

### A. Stage A — wellness wearable (next ~12–18 months)

| Workstream | Low | Mid | High | Notes |
|------------|-----|-----|------|-------|
| **Ed/Pedro Phase 0 ops** | 5k | 15k | 35k | Gen1 units exist; partner handoff still **gated** — travel, phones/TestFlight, time, Beacon liaison, replacements |
| **Gen1 build / scrap / spares** | 10k | 25k | 60k | Sample floors: Kaga Gen1 [GEN1_COGS](../hardware/GEN1_COGS_KAGA_QUOTE.md); Bittele PCB00003 [BITTELE](../hardware/BITTELE_Q100918A1_PCB_QUOTE.md) (~$173/set); **volume finished-kit COGS still TBD** |
| **Patient app (Phase 0→1+)** | 20k | 50k | 100k | Contractor iOS / OralableCore; overnight report; StoreKit live; **not** pro app yet |
| **Firmware 1.0.84 soak / field** (was 1.0.70 line) | 5k | 15k | 30k | STAT + IR-pulse + pad/desk recover; TestFlight 4.3.3 build 5 |
| **Phase 1+ science / validation** | 15k | 40k | 80k | Protocol B deferral → muscle gates; Python reports; small concordance |
| **Cloud / Apple / tools** | 1k | 3k | 8k | Apple Developer, TestFlight, CloudKit (low until scale), CI |
| **Website / GTM content** | 2k | 8k | 20k | oralable.com updates, screenshots, soft launch ads (optional) |
| **Insurance / legal ops (Stage A)** | 3k | 10k | 25k | Product liability wellness tier, contracts, privacy |
| **Contingency (15%)** | — | — | — | Apply to subtotal |
| **Stage A subtotal (ex-Gen2, ex-IP)** | **~60k** | **~165k** | **~360k** | Mid ≈ lean Stage A year |

### B. Gen2 hardware (parallel · ~12–18 months)

| Workstream | Low | Mid | High | Notes |
|------------|-----|-----|------|-------|
| **REV11 EVT / fab / assembly** | 25k | 60k | 150k | PCB spins, stencil, EMS NPI lot (tens of units) |
| **Modules / BOM (ES4L15, cell)** | 10k | 30k | 80k | Sample lot [GEN2_COGS](../hardware/GEN2_COGS_KAGA_QUOTE.md): PCB €145×20 + stencil €525 + cells **€13×20** + ship = **€3,735**; volume 1k–10k TBD |
| **HW eng / Altium / RF** | 15k | 40k | 90k | Pin map, CHRSTS fix, antenna, charge ISET for 30 mAh |
| **Gen2 FW bring-up** | 20k | 50k | 100k | `pcb00003_gen2`, NCS, OTA, soak |
| **Gen2 subtotal** | **~70k** | **~180k** | **~420k** | Can slip right without killing Stage A on Gen1 |

### C. IP — new US patent (counsel-led)

| Workstream | Low | Mid | High | Notes |
|------------|-----|-----|------|-------|
| **US non-provisional / utility prep + file** | 12k | 25k | 45k | Peacock / US agent; drawings; claims |
| **Prosecution (2–3 yrs, partial accrual)** | 8k | 20k | 50k | Office actions; budget annually |
| **EU / PCT options (optional)** | 5k | 20k | 60k | Only if strategy requires |
| **FTO / software IP memo (Ken gap)** | 5k | 15k | 35k | Recommended before Stage B |
| **IP subtotal (near-term file + year 1)** | **~25k** | **~60k** | **~140k** | Portfolio admin external |

### D. Stage B — medical device (later · ~18–36 months after Phase 1+)

| Workstream | Low | Mid | High | Notes |
|------------|-----|-----|------|-------|
| **Regulatory counsel + 510(k) package** | 40k | 100k | 250k | Predicate, file, FDA fees, responses |
| **Clinical / concordance study** | 50k | 150k | 400k | Site(s), EMG gold, n≈20–50+; biggest swing |
| **ISO 13485 / QMS / IEC 62304 climb** | 30k | 80k | 200k | Audit, SOPs, SaMD lifecycle |
| **CE MDR Class IIa (parallel)** | 40k | 100k | 250k | NB fees + technical file |
| **Locked build / labeling / IFU** | 15k | 40k | 80k | Pro app clinical mode, claim-safe UX |
| **Stage B subtotal** | **~175k** | **~470k** | **~1.2M** | Do **not** fund from Phase 0 kit cash alone |

### E. People (often the real burn)

If founder time is unpaid, cash burn is mostly contractors (tables above).  
If you pay 1–2 FTE eng plus fractional regulatory:

| Model | Mid burn / month | 12-month |
|-------|------------------|----------|
| Founder + light contractors | 5–15k | 60–180k |
| Founder + 1 iOS/FW contractor + HW retainer | 15–35k | 180–420k |
| Small team (2–3 FTE + counsel) | 40–80k | 480–960k |

**Add people burn on top of A–D** if not already embedded in those line items.

---

## 3. Rolled-up scenarios (cash out · EUR)

| Scenario | Scope | Mid estimate | High (stress) |
|----------|--------|--------------|---------------|
| **P0 only** | Ed/Pedro + FW polish + light app | **~30–50k** | ~100k |
| **Stage A lean** | P0 + Phase 1+ + IP file + soft launch (Gen1) | **~200–250k** | ~500k |
| **Stage A + Gen2** | Above + REV11 bring-up | **~350–450k** | ~900k |
| **Through Stage B mid** | Stage A+Gen2 + 510(k) mid clinical | **~800k–1.0M** | ~2M+ |

**Revenue offsets (optimistic GTM draft — not proven):**  
Hardware margin on kits plus Premium €9.99/mo; dentist seats **after** Phase 1+. Year-1 draft targets (500 downloads / 20 dentists) **do not** pay for Stage B on their own.

---

## 4. What to spend now vs later

| Spend now (unlocks Stage A + patent embodiment) | Defer |
|--------------------------------------------------|--------|
| Ed/Pedro Phase 0 (patient app) | Oralable for Dentists activation |
| 1.0.70 field soak + tag | Full ISO 13485 audit |
| Phase 1+ TFI/SASHB in patient app | Large n clinical pivotal |
| US patent file (counsel) | CE NB engagement |
| Gen2 EVT only if CHRSTS/battery block Stage A quality | Volume EMS PO before COGS known |
| Minimal CloudKit when share needed | Path B IAP marketing |

---

## 5. Timeline × cost (mid-case cash)

| Period | Likely cash (mid) | Main uses |
|--------|-------------------|-----------|
| **Jul–Sep 2026** | 20–40k | Ed/Pedro, spares, FW soak, app polish |
| **Q4 2026** | 40–80k | Phase 1+ app, IP filing push, Gen2 kickoff |
| **H1 2027** | 80–150k | Gen2 EVT, Phase 1+ evidence, soft launch |
| **H2 2027** | 100–250k | Stage B pre-sub start **or** Gen2 pilot parity |
| **2028** | 150–400k+ | 510(k)/CE if Stage B funded |

---

## 6. Decision gates (spend only if pass)

| Gate | Pass → unlock spend |
|------|---------------------|
| **G0** Phase 0 vitals | Phase 1+ eng + more kits |
| **G1** Phase 1+ embodiment in app | Soft Premium; consider pro app |
| **G2** Gen2 parity (optional parallel) | Gen2 pilot kits |
| **G3** US patent submitted | Prosecution budget |
| **G4** Concordance protocol funded | Stage B clinical + QMS |

---

## 7. Next actions to harden numbers

1. EMS / Bittele / Hosiden: **Gen1 COGS** and **Gen2 EVT quote** (N=10 / N=100).  
2. Peacock: **US filing + 12-mo prosecution** fixed quote.  
3. Regulatory counsel: **510(k) ballpark** for optical bruxism monitor predicate.  
4. Beacon / CRO: **pilot expansion** cost for n=20 concordance.  
5. Accountant: fold into runway / ask (Financials still Ken **GAP**).

---

*Estimates only. Update when quotes land; bump `docs/data_room/VERSION`.*
