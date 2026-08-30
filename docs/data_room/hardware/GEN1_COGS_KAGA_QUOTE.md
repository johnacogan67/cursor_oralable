# Gen1 COGS evidence — KAGA FEI Europe sample quote

**Status:** Historical quote skim for Ken / BalancePoints · **not** volume finished-goods COGS  
**Sources (external PDFs — do not commit):**  
- `KFEU_6297-20240517.pdf` — Quotation **6297-20240517** · **17 May 2024**  
- `Gmail - RE_ Byteexplain + KAGA Regular Meeting.pdf` — Elvis Prunkl thread covering the quote  

**Buyer on quote:** Byteexplain Limited · 18 Eaton Square, Blackrock, Ireland  
**Supplier:** KAGA FEI Europe GmbH · Langen, Germany · Contact: Elvis Prunkl  

---

## 1. Quoted lines (Gen1 / “NEU” sample PCB)

| Item | Qty | Unit EUR | Ext. EUR | Notes |
|------|-----|----------|----------|-------|
| Sample-PCB for NEU according BOM provided by Byteexplain | **20** | **95.00** | **1,900** | MOQ 20 · lead **6–8 weeks** · depends on **MAXM86161** availability |
| NRE (stencil, fixture, set-up…) | 1 | **1,995.00** | **1,995** | One-time |
| **Sample lot total** | | | **€3,895** | EXW Langen · cash in advance |

**Validity on quote:** to **16 Jun 2024** (expired). Use as **sample PCB cost order-of-magnitude**, not a live 2026 PO price.

**Incoterms:** EXW Langen — packaging / VAT / freight / duty **extra**.

---

## 2. What this is / is not

| Is | Is not |
|----|--------|
| Assembled **sample PCB** unit cost at N=20 | Full **clip + magnetic case** retail COGS |
| Early Gen1 / NEU EMS path via Kaga | Serial / volume price ladder (50 / 100 / 500) |
| NRE for first sample build | Enclosure / plastics / battery pack / final test / packaging |
| Evidence that PCB assembly was ~**€95** at sample | 2026 Bittele / other EMS superseding quotes |

**Loaded sample PCB cost (amortise NRE over 20):**  
(€1,900 + €1,995) / 20 ≈ **€194.75 / board** for that lot only.

---

## 3. Meeting context (5–17 May 2024) — COGS-relevant

From Kaga regular-meeting notes in the Gmail PDF:

1. **20 NEU samples** — sample production **China**; quote issued 17 May.  
2. **Charger** — Byteexplain to send update → Kaga to quote **serial parts + 20 charger samples** (not in this PDF).  
3. Design DFM optimisation check (Kaga).  
4. Test definition — link Hagen ↔ Kaga engineer.  
5. Battery charge level at factory / shelf-life (Byteexplain). Clip+charger concept likened to **AirPods**.  
6. **Production plant Vietnam** — OK for US and EU ship.  
7. Kaga can quote **complete finished product** (assembled PCB + housing) once **charger housing data** provided.  
8. Standards for US/EU sale — Byteexplain cited **IEC 60601** (medical safety roadmap). Stage A wellness today is a separate path from that production talk.

---

## 4. How to use for Ken / Stage A plan

| Use | Caution |
|-----|---------|
| Fill “we have real EMS quotes” | Quote **stale** (May 2024); re-quote before volume PO |
| Floor for **bare PCB** sample economics | Case, coil, plastics, battery, test, scrap, logistics missing |
| Bridge to Bittele ~€3k line on Revolut (May 2026) | Different vendor / lot — do not double-count as same build |
| Ask sizing | Still need **volume COGS** for clip+case kit ASP |

**Still GAP:** serial quote for PCB; charger enclosure assembly; finished kit BOM roll-up at 50/100/500; 2026 refresh from Kaga or alternate EMS.

---

## 5. Deck one-liner

> Gen1 sample PCB (Kaga FEI EU, May 2024): **€95/unit × 20** + **€1,995 NRE** ≈ **€3.9k** lot · EXW · not finished-goods or volume COGS.

---

*Related:* [GEN2_COGS_KAGA_QUOTE.md](./GEN2_COGS_KAGA_QUOTE.md) · [FINANCIALS_CASH_SNAPSHOT.md](../governance/FINANCIALS_CASH_SNAPSHOT.md) · [COST_AND_TIMELINE.md](../governance/COST_AND_TIMELINE.md) · [PITCH_DECK_KEN.md](../pitches/PITCH_DECK_KEN.md)
