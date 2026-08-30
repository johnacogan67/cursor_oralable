# Kaga / NCAB PC53020 — Gen2 PCB EQ replies (Aug 2026)

**As at:** 11 Aug 2026  
**Quote:** KAGA FEI Europe **6848-20260511** · Gen2 sample lot [GEN2_COGS_KAGA_QUOTE.md](./GEN2_COGS_KAGA_QUOTE.md)  
**Board:** PCB00003-TGM-PCB-REV11 · NCAB job **PC53020 Rev 01** · tooling **282000-ITN**  
**Chain:** Kaga (Elvis Prunkl) → Hosiden Besson (EMS) → NCAB UK (EQ) → China fab CAM  

**Sources (external — do not commit binaries):**  
- `Gmail - RE_ KAGA_ new design for Oralable - Quotation 6848-20260511.pdf` (Elvis 10 Aug 2026)  
- `QUERIES - PC53020 Rev 01 (...PCB-REV11) - 282000-ITN.xlsx` — NCAB Engineering Questions  
- Filled replies copy: `QUERIES - PC53020 Rev 01 (...PCB-REV11) - 282000-ITN-REPLIES.xlsx` (same Drive Sources folder)

**One-liner:** Three fab DFM asks on REV11. Accept stackup + copper shave; accept NPTH clearance moves only with redline sign-off; reject Cu exposure in NPTH. CAM tool is **Frontline Genesis 2000**.

---

## 1. Claim discipline

| Do say | Do not say |
|--------|------------|
| Sample lot still **€3,735** (20 PCB + stencil + cells) pending EQ close | Volume COGS locked |
| First volume indication: **€72 @ MOQ 5k** · **€71 @ MOQ 10k** (Elvis 10 Aug) | Investor unit cost proven at 50k / 500k |
| EQ is **fab stackup / NPTH clearance**, not BOM or module change | ES4L15 / MAXM86161 redesign |
| Designer (Wout) should confirm items 2–3 redlines before fab | Blank “move any trace” approval |

---

## 2. What program they use

| Layer | Tool |
|-------|------|
| Your design stackup | **Altium Designer** Layer Stack Manager (WeeGee / Wout) |
| EQ form | **NCAB Group Engineering Questions (EQ)** — Microsoft Excel template on NCAB UK SharePoint |
| Fab DFM screenshots (fig7 / fig8) | **Frontline Genesis 2000** (Job Matrix, Popview, job `282000-itn`) |

---

## 3. EQ table — recommended G / H

| Item | Question (summary) | Solution they offer | **G Reply** | **H Remark** |
|------|--------------------|---------------------|-------------|--------------|
| 1 | Laser needs 0.33 oz outer foil; stackup adjusted for board thickness | **A:** Outer base 0.33 oz + IPC III plating, finish ≥34.3 µm; FIG A OK | **Accepted (A)** | Proceed with FIG A. Keep finished board **0.8 ± 0.1 mm** and outer finished Cu **≥ 35 µm**. Material **S1000-2M / S1000-2MB** (TG180) or equivalent TG≥150 OK. |
| 2 | NPTH close to tracks/copper; need **≥ 8 mil** after compensation | **A:** Move traces and shave copper | **Accepted with condition (A)** | 8 mil min after compensation OK. Copper shave OK. Trace moves only with redline / updated Gerbers for our designer approval before manufacture. No netlist change. |
| 3 | NPTH close to pads; same 8 mil rule | **A:** Shave copper (no exposure) · **B:** Follow Gerber, accept Cu exposure | **Accepted (A). Rejected (B).** | Shave copper so none is exposed in NPTH. Do not accept copper exposure. |

---

## 4. Volume price (first indication)

From Elvis 10 Aug 2026 (same thread as EQ):

| MOQ | Unit EUR (indication) |
|-----|------------------------|
| 5,000 | **€72** |
| 10,000 | **€71** |

Other qty bands (500 / 50k / 500k) still open. Sample lot pricing unchanged until EQ closes and schedule returns.

---

## 5. Email draft — Elvis + Wout

**To:** Elvis.Prunkl@eu.kagafei.com  
**Cc:** wout.geeurickx@weegee.be · Dominic.Kohlhoefer@eu.kagafei.com  
**Subject:** RE: KAGA Quotation 6848-20260511 — PC53020 EQ replies

```
Hello Elvis,

Thank you for the NCAB Engineering Questions on PC53020 / PCB00003-TGM-PCB-REV11.

Please find our replies in column G (with remarks in column H) of the attached spreadsheet:

1. Stackup (FIG A) — Accepted. Outer base 0.33 oz + plating to ≥35 µm finished Cu. Keep finished thickness 0.8 ± 0.1 mm. S1000-2M / S1000-2MB (or equivalent TG≥150) OK.

2. NPTH to tracks — Accepted with condition. 8 mil after compensation and copper shave OK. Any trace moves need redline / updated Gerbers for our designer (Wout, cc) to approve before manufacture. No netlist change.

3. NPTH to pads — Accept option A (shave copper). Reject option B (copper exposure in NPTH).

Wout — please confirm items 2 and 3 from your side if anything in the popviews sits on a critical net. Once you are happy, Elvis can treat this as customer sign-off.

Elvis — after these points are closed, please send the delivery schedule for the 20 prototype assemblies (quote 6848-20260511). I also note the first volume indication of €72 @ 5k and €71 @ 10k; please share the other quantity bands when you have them.

Best regards,
John A. Cogan, PhD
CTO, JAC Dental Solutions Limited
```

---

## 6. Bookmark table

| # | Source | Why it matters |
|---|--------|----------------|
| 1 | Elvis email 10 Aug 2026 (thread RE: 6848-20260511) | EQ attached; schedule gated on G/H; volume €72/€71 |
| 2 | NCAB EQ xlsx PC53020 / 282000-ITN | Three fab asks + FIG A stackup + Genesis screenshots |
| 3 | [GEN2_COGS_KAGA_QUOTE.md](./GEN2_COGS_KAGA_QUOTE.md) | Canonical sample-lot €3,735 |
| 4 | [HW_ENGINEER_ALTIUM_BRIEF.md](./HW_ENGINEER_ALTIUM_BRIEF.md) | Wout / Altium ownership for redline sign-off |
| 5 | [PCB00003_GEN2_REV11_HARDWARE.md](../PCB00003_GEN2_REV11_HARDWARE.md) | Gen2 REV11 hardware reference |

---

*Related:* [GEN2_COGS_KAGA_QUOTE.md](./GEN2_COGS_KAGA_QUOTE.md) · [COST_AND_TIMELINE.md](../governance/COST_AND_TIMELINE.md) · [HW_ENGINEER_ALTIUM_BRIEF.md](./HW_ENGINEER_ALTIUM_BRIEF.md)
