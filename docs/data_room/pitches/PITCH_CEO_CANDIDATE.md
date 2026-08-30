# Oralable — Pitch for prospective CEO candidates

**App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

**Audience:** CEO candidates (via Independent Director / Amstrow intros)  
**Status:** Sendable leave-behind · **27 Jul 2026** · Stage A truth  
**Not:** Ken / BalancePoints Pre-Seed memo (€1.25m · 28% · intraoral · Chief Scientist)  
**Pair with:** [CEO_JOB_DESCRIPTION.md](./CEO_JOB_DESCRIPTION.md)

**Present:** open [`PITCH_CEO_CANDIDATE.html`](./PITCH_CEO_CANDIDATE.html) (← →) · PDF [`PITCH_CEO_CANDIDATE.pdf`](./PITCH_CEO_CANDIDATE.pdf)

**Brand (use in HTML/PDF):**
- Lockup: [`brand/oralable_logo_lockup.png`](./brand/oralable_logo_lockup.png) (icon + **Oralable** wordmark)
- Tagline trademark: **Word of Mouth™**
- Type: Open Sans (site brand stack); wordmark rendered from lockup artwork
- Footer line on slides: *Oralable® and Word of Mouth™ are trademarks of JAC Dental Solutions Limited*

---

## Slide 1 — Title

**[Oralable lockup]**  
**Word of Mouth™**

Temple wearable for overnight vitals → jaw-load awareness → (later) medical device  

**Stage A now:** patent-implementing **wellness wearable**  
**Stage B later:** **medical device** pathway — not claimed today  

**Entity:** JAC Dental Solutions Limited (Ireland)

---

## Slide 2 — Problem

Sleep bruxism (clench / grind) damages teeth, causes jaw pain, and ties to overnight oxygen burden — but:

- **Rings / watches** measure general wellness at finger/wrist — wrong site for jaw load  
- **Clinic EMG** is accurate but episodic, expensive, not overnight consumer  
- **Patients and dentists** lack a continuous, at-home optical + motion record of jaw hemodynamics  

This is a **condition-specific** problem at an unusual site — not “another sleep tracker.”

---

## Slide 3 — Solution

**Oralable** = Nordic BLE **temple clip** + **magnetic charging case** + **iOS patient app**

| Now (Phase 0) | Next (Phase 1+) | Later (Stage B) |
|---------------|-----------------|-----------------|
| Temple **HR + SpO₂** with honest device state | **IR-DC** occlusion, **TFI**, **SASHB** in patient app | Cleared / CE **medical device** (separate gate) |
| Pilot evidence on Gen1 | Patent embodiment in product | Locked IFU / QMS |

**Physics:** PPG + IR-DC hemodynamic occlusion + jaw accelerometry — **not** sEMG, **not** a ring, **not** intraoral for this Gen1 path.

---

## Slide 4 — Why now / moat

- **Working Gen1 stack** — hardware, firmware, and patient app in the field path  
- **IP path (high level):** European foundation patent **EP 4 333 691 B1** granted · US utility in continued prosecution · US provisional on muscle-activity / Temporalis refinements  
- **Open 50 Hz pipeline** (Swift + Python) — evidence and embodiment, not a closed ring black box  
- **Orthogonal to rings** — same sensor classes, different physics (jaw / temple)  

---

## Slide 5 — Product truth (July 2026)

| Item | Spec |
|------|------|
| Clip | Gen1 · BOM REV8 · PCB REV10 · Kaga ES2832AA2 (nRF52832) |
| Charge | **Oralable magnetic case** — not phone Qi / MagSafe |
| Firmware | **1.0.84** (IR-pulse worn · STAT blink · pad/desk recover) |
| App | Oralable patient **4.3.3** (TestFlight) — dentist app **off** for pilot |
| Placement | **Temple** |
| Claims | **Wellness only** — not a medical device today |
| Handoff | **Research Kits gated** — target **5 to Pedro by 31 Aug 2026** when charge reaches temple-ready SOC + short worn vitals hold |

---

## Slide 6 — Traction (honest)

| Evidence | Status |
|----------|--------|
| Gen1 kits + FW 1.0.84 + app 4.3.3 (build 5) | **Stack ready** — flash / TestFlight path live |
| Ed/Pedro Research Kits | **Pending ship** (charge-to-temple) — 5 kits target **31 Aug 2026** |
| App Store live MAU / MRR | **Near zero** — soft launch gated |
| Funding to date | **Founder-loan funded** (~€63.5k director loans in-period; cash thin) |
| Board | Founder + **Independent Director** (Amstrow) in place |

Ready ≠ delivered. First customer-validation evidence comes from the patient pilot.

---

## Slide 7 — Business model

```
Hardware (clip + case)   ──►  now / pilot (low volume)
Consumer Premium IAP     ──►  after credible Phase 0–1+ dashboard
Dentist / practice IAP   ──►  after Phase 1+ share (later)
Cleared device (Stage B) ──►  separate regulatory gate
```

**Policy:** Ed/Pedro = **patient app only**. No medical claims in Stage A marketing.

---

## Slide 8 — The CEO opportunity

| | Founder (Exec Chair) | CEO (this role) |
|--|----------------------|-----------------|
| Owns | Product, clinical/science, IP with counsel | Capital, commercial, ops, day-to-day |
| Near term | Pilot quality · Phase 1+ embodiment | **Point B ~€180k** (F&F €50k + EI PSSF €100k + HPSU €30k) |
| Next | Patent embodiment evidence | Soft launch · next equity round story |

**Mission (12–18 months):** Point B → pilot evidence → credible Stage A soft launch → board-ready next round — **without** overselling medical claims.

Cap table: founder-majority path; ordinary cleanup **in progress**; named angels ~16%. Detail on request under NDA / board process.

---

## Slide 9 — Package frame

| Element | Discussion band |
|---------|-----------------|
| Title | Chief Executive Officer |
| Cash | Modest / deferred until Point B; then lean early-stage salary |
| Equity | **~5–10% FD options** (centre ~**6–8%**) |
| Vesting | **4 years** · **1-year cliff** |
| Milestones | Appointment · F&F · Point B · tenure (see JD §7a) |

**Not the frame:** ~28% day-one **issued** founder share transfer.  
Full brief: [CEO_JOB_DESCRIPTION.md](./CEO_JOB_DESCRIPTION.md).

---

## Slide 10 — Ask of the candidate

We want a conversation if you:

1. Have raised early capital (Ireland / UK / EU angels, EI, pre-seed)  
2. Can operate a small **hardware + software** company  
3. Will keep **Stage A wellness** claims honest  
4. Want to partner with a technical founder who stays **Executive Chairman**  

**Next step:** short intro with John (+ Independent Director as useful) · product walkthrough · then board/counsel on terms.

---

## Appendix — Cover email (copy/paste)

**Subject:** Oralable — Word of Mouth™ · company overview + CEO role

Hi [Name],

Nigel Woods suggested I send a short overview of Oralable and the CEO role we’re hiring for.

**Oralable — Word of Mouth™** is an Irish company building a **temple-worn wellness wearable**: overnight vitals first, then jaw-load / bruxism-related metrics — with a medical-device path as a **later** gate, not today’s claim. We have a working Gen1 stack (hardware, firmware, iOS patient app); the first patient pilot waits on a final charge-to-temple check.

I need a **commercial CEO** to own fundraising, go-to-market, and day-to-day leadership. I stay **Executive Chairman** on product and IP. Near-term capital is Point B, about **€180k** (F&F + Enterprise Ireland instruments).

Attached:
1. Company pitch (PDF)  
2. CEO job description (includes package discussion band: options with vesting and milestones)

Happy to set up a short call if this looks interesting.

Best regards,  
John Cogan  
Founder · JAC Dental Solutions Limited / Oralable

---

*Related:* [CEO_JOB_DESCRIPTION.md](./CEO_JOB_DESCRIPTION.md) · [PITCH_DECK_KEN.md](./PITCH_DECK_KEN.md) (BalancePoints working session — separate) · [CURRENT_GOVERNANCE_STATUS.md](../governance/CURRENT_GOVERNANCE_STATUS.md)
