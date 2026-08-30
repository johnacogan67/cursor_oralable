# Gen2 COGS evidence — KAGA FEI Europe (quotation 6848)

**Status:** Sample-lot quote for Ken / BalancePoints · May–Jun 2026  
**Buyer:** JAC Dental Solutions · Ireland  
**Supplier:** KAGA FEI Europe GmbH · Elvis Prunkl / Dominic Kohlhöfer  

**Sources (external PDFs — do not commit):**  
- `6848-20260512.pdf` — **complete PCB + stencil + extra batteries** (Our Ref **6848-20260511** · dated **28 May 2026** on face)  
- `KFEU_6848-20260511.pdf` — earlier **module-only** revision (11 May 2026)  
- `Gmail - RE_ KAGA_ new design for Oralable - Quotation 6848-20260511.pdf` — Ken + Elvis PO / volume thread (Jun 2026)  
- Design / battery thread: `Gmail - Fwd_ KAGA_ new design for Oralable.pdf` · `Gmail - RE_ KAGA_ new design for Oralable.pdf`  

---

## 1. Canonical sample quote — complete PCB lot (use this)

**Your reference:** PCB00003-TGM · Valid to **30 Jun 2026** · Payment **30 days net** · **EXW Langen**

| Item | Qty | Unit EUR | Ext. EUR | Notes |
|------|-----|----------|----------|-------|
| **PCB acc. PCB00003-TGM-PROD_DATA** | **20** | **145** | **2,900** | Lead **6–8 weeks** |
| **One-off stencil cost** | 1 | **525** | **525** | NRE |
| **Extra batteries** | **20** | **13** | **260** | **LP270829** path — Bittele could not source full cell lot |
| **Shipping** | 1 | **50** | **50** | |
| **Total** | | | **€3,735** | As printed on quote |

**Loaded cost:** €3,735 / 20 ≈ **€186.75 / set** (incl. amortised stencil + 1 extra cell/set + ship).

**Why “extra batteries”:** Bittele ([BITTELE_Q100918A1_PCB_QUOTE.md](./BITTELE_Q100918A1_PCB_QUOTE.md)) populated only **5/20** with cells. The Kaga line adds **20×** cells for Gen2 sample bring-up and spares.

---

## 2. Earlier revision — module only (superseded for kit COGS)

`KFEU_6848-20260511.pdf` · 11 May 2026 · Your ref Samples_ES4L15BA1:

| Item | Qty | Unit EUR | Ext. |
|------|-----|----------|------|
| ES4L15BA1-SAMPLE | 20 | 8.06 | 161.20 |
| Shipping | 1 | 50.00 | 50.00 |
| **Lot** | | | **€211.20** |

Keep for module unit price only. **Do not** use as Gen2 finished-PCB COGS.

---

## 3. Commercial / design context

| Topic | Decision / status |
|-------|-------------------|
| Scope John wants | Complete Gen2: sensor + charger + battery (MAM/TGM) |
| Sample cell | **LP270829 35 mAh** (LP260820 MOQ 5k) — approved 18 May |
| Assembly geography | **Ireland** final assembly; components OK from **China** |
| Class (near term) | **Wellness** — not FDA/medical at this stage (Stage A) |
| FCC Gen2 | John manages after units (Gen1 via IA) |
| Parallel China PCB | Bittele Q100918A1 — **$172.99×20** paid (cells short) |

---

## 4. Ken / BalancePoints thread (Jun 2026)

From quotation email (Elvis ↔ Kenneth Kinsella `ken@balancepoints.co.uk`, cc John):

- Ken: Ireland VAT reverse-charge · quote in **EUR** · research-batch reprice ask on **€3,735** · outline pricing for **500–5,000** commercial · Kaga family rollout alignment  
- Elvis (8 Jun): sample **20 pcs pricing unchanged**; volume feedback in flight for **1k / 5k / 10k**; proposed intro call  

**Volume ladder (first indication — Elvis 10 Aug 2026):** **€72/pc @ MOQ 5k** · **€71/pc @ MOQ 10k**. Other bands (500 / 50k / 500k) still open.

**Fab EQ (gates sample schedule):** NCAB **PC53020** / **282000-ITN** on REV11 — replies and email draft in [KAGA_NCAB_PC53020_EQ_2026-08.md](./KAGA_NCAB_PC53020_EQ_2026-08.md).

---

## 5. Compare sample paths

| Path | What | Lot signal |
|------|------|------------|
| **Kaga 6848 complete** | 20 PCB + stencil + 20 cells + ship | **€3,735** |
| Bittele Q100918A1 | 20 turn-key sets (5 with battery) | **$3,459.80** (~€2,968 paid) |
| Kaga module-only (old) | 20× ES4L15BA1 | €211 |

---

## 6. Deck one-liner

> Gen2 Kaga sample lot (6848, May 2026): **20× PCB @ €145** + **€525 stencil** + **20× batteries @ €13** + **€50 ship** = **€3,735** (~€187/set). Volume indication **€72 @ 5k / €71 @ 10k** (Aug 2026); sample schedule gated on NCAB EQ.

---

*Related:* [KAGA_NCAB_PC53020_EQ_2026-08.md](./KAGA_NCAB_PC53020_EQ_2026-08.md) · [BITTELE_Q100918A1_PCB_QUOTE.md](./BITTELE_Q100918A1_PCB_QUOTE.md) · [GEN1_COGS_KAGA_QUOTE.md](./GEN1_COGS_KAGA_QUOTE.md) · [COST_AND_TIMELINE.md](../governance/COST_AND_TIMELINE.md) · [PITCH_DECK_KEN.md](../pitches/PITCH_DECK_KEN.md)
