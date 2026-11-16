# Ed / Pedro meeting — 7 Sep 2026: evaluation

**As at:** 7 Sep 2026 · Pack **1.1.68** · stack FW **1.0.84** · app **4.3.3** (build **5**) · kits still **gated**  
**Current kit (14 Oct):** Gen 2 REV11. This note is what was said on 7 Sep, when the kit on the table was Gen 1. Do not read the body as the hardware we hand over now. Living stack: [VERSION_ALIGNMENT.md](../VERSION_ALIGNMENT.md). Kits still **gated**.  
**Attendees (this meeting):** John A. Cogan · Dr Edward Owens · Dr Pedro Mayoral Sanz  
**Status:** Internal evaluation after the 7 Sep meeting. Protocol / kit / Paper A scope **updated 7 Sep** to match. Not ethics-approved. Sendable: [PEDRO_STATUS_UPDATE_2026-09.md](./PEDRO_STATUS_UPDATE_2026-09.md).  
**Locked sources:** [PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md) · [PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md](./PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md) · [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) · [VERSION_ALIGNMENT.md](../VERSION_ALIGNMENT.md) · [IP_PORTFOLIO_STATUS.md](../governance/IP_PORTFOLIO_STATUS.md) · [IP_EVAL_AND_LANDSCAPE.md](../governance/IP_EVAL_AND_LANDSCAPE.md) · [REGULATORY_TIMELINE.md](../governance/REGULATORY_TIMELINE.md) · [PRODUCT_ROADMAP.md](../../PRODUCT_ROADMAP.md) · [COST_AND_TIMELINE.md](../governance/COST_AND_TIMELINE.md)

**One-liner:** Ed and Pedro like what the MAM *shows* (sleep-apnea pattern + bruxing events). Ed, not Pedro, set the use rules: no headband, one night, clinic recharge, AccuPebble/Sunrise-like stick-on or ECG pads, rubbery feel. Dual A is **not** required for the n≈5 feasibility. Paper A and kits can start on **Gen1** under those rules. Combined EMG+PPG “Gen4”, USB-C, and iPhone placement-from-ear are a **later product + patent** track — they must not block Beacon *N*.

---

## Verdict

The meeting unblocked the **clinical story** and blocked the **current packaging story**.

| Layer | Outcome |
|-------|---------|
| **Signal / information** | Both happy. Keep Paper A as measurement + feasibility. Do not upgrade wording to AHI or SB diagnosis. |
| **Who directs Beacon use** | Ed. Pedro remains the named clinical contact and Arm P / MAD owner. Treat Ed as packaging and patient-use owner. |
| **n≈5 feasibility** | Oralable-only. No ANR stack. No headband. One night. Clinic charges. Pedro wants **≥4 h** (6 h better). |
| **Paper A Dual A (O6)** | Demote to optional methods / eng precursor. Do not make patients combine ANR + MAM. |
| **Ship this month** | Gen1 kits without ANR and without headband. 31 Aug target is already late. |
| **“Gen4” combined pod** | New form-factor concept (not Gen2 nRF54L15). Design in parallel. Do not wait for it to start n≈5. |
| **FDA** | Extraoral + USB-C does **not** cancel medical-device law. Stage A wellness claims stay. Counsel, not this note, classifies the device. |
| **Patent** | Broaden **64/033,978** conversion (due **~9 Apr 2027**) with placement-from-ear + combined optical/EMG embodiments. Do not file claim text in git. |

---

## 1. What they said (mapped to current docs)

### 1.1 Ed directed; Pedro is the named contact

Docs tell agents to “deal with Pedro.” The meeting showed **Ed directing packaging and patient use**. That matches Paper A’s split more than the kit ops notes: Pedro = Arm P / MAD / AcuPebble; Ed = overnight / wear / how the patient actually uses it ([PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md) §8).

**Change in practice:** copy Ed on use, attachment, and clinic-return rules. Keep Pedro as ethics / Arm P / MAD / AcuPebble owner. Do not wait for Pedro to override Ed on packaging.

### 1.2 They like the information (apnea pattern + bruxing events)

This is the strongest result. It supports Paper A’s hypnogram + oxygen-burden + jaw-load map as **chairside-useful**.

**Claim discipline (unchanged):**

- Temple SpO₂ / SASHB = engineering oxygen burden. **Not** AcuPebble AHI. **Not** Azarbarzin hypoxic burden. Pedro’s apnea reference stays **AcuPebble** ([MEASUREMENT_CONSTRUCT_MAP.md](./MEASUREMENT_CONSTRUCT_MAP.md)).
- IR-DC / states / “bruxing events” = engineering phenotypes. **Not** PSG-AV sleep-bruxism diagnosis.
- Paper A Abstract already says we do not claim AHI equivalence. Keep that sentence.

### 1.3 No headband; AccuPebble / Sunrise attach, or ECG pads; rubbery texture

Ed rejected the Ridge-style headband in the Research Kit BOM ([ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) §1). He wants attachment in the same *class* as:

- **AcuPebble** (Acurable) — small neck sensor, adhesive, one-night HSAT loaner ([ACUPEBBLE_VS_ORALABLE_ANR.md](../bookmarks/ACUPEBBLE_VS_ORALABLE_ANR.md)).
- **Sunrise** — chin mandibular-movement sensor, literature peer in Li 2025 ([ORALABLE_MARKET_LANDSCAPE.md](../bookmarks/ORALABLE_MARKET_LANDSCAPE.md) §4c).
- **ECG / Red Dot pads** — already in the kit for ANR. Ed accepts this for the MAM.

Gen1 today is a **clip** plus optional headband / Kapton / silicone. For n≈5, seat Gen1 with **3M Red Dot (or equivalent) pads + tape**, no headband. Do not invent a new clip overnight.

“Rubbery like AccuPebble / Sunrise” is a **Gen4 enclosure** spec (overmold / pot), not a Gen1 paint job.

### 1.4 Pedro: 4 hours enough; 6 better

Locked docs disagree with each other on duration:

| Doc | Duration |
|-----|----------|
| Paper A Arm P | **1–2 h** oxygen / MAD |
| Paper A Arm E/J + overnight bands | **≥6 h** worn (goal 8 h) |
| iOS overnight band unlock | **≥1 h** |
| This meeting (Pedro) | **≥4 h** (6 h better) |
| This meeting (Ed) | **One night**, then return |

**Recommendation for Beacon feasibility (n≈5):** evaluable session = **≥4 h** worn. Prefer 6 h when the patient sleeps that long. Keep **≥6 h** as the Paper A *figure* target and as the overnight-band recalibration rule. Do not fail a 4 h night as “insufficient” for feasibility wear/QC.

Arm P 1–2 h can stay as a **short MAD window** inside a 4 h+ night, or as a separate daytime/early-sleep check if Pedro still wants it.

### 1.5 Ed: one night, return, clinic recharge

This removes **patient charging** as a Beacon problem. It does **not** remove John’s **charge-to-temple** gate before a kit leaves the bench ([VERSION_ALIGNMENT.md](../VERSION_ALIGNMENT.md)).

Ops model becomes **clinic loaner** (AcuPebble-like): charge in clinic → seat → one night → return → download → recharge. Matches Ed’s packaging concern better than a consumer multi-night wearable.

GTM Path A (consumer App Store + home charge case) is a **different** product later. Do not mix the two in Paper A methods.

### 1.6 Extraoral wearable and “no FDA”; USB-C instead of sealed wireless charge

**What is already true:** Stage A is an extraoral temple **wellness wearable**. No FDA/CE medical claims. Ed/Pedro = patient app only ([REGULATORY_TIMELINE.md](../governance/REGULATORY_TIMELINE.md), [IP_NORTH_STAR.md](../../IP_NORTH_STAR.md)).

**What is not true:** putting the sensor on skin, or adding USB-C, does not by itself remove FDA or MDR. Classification follows **intended use and claims**. If copy says diagnose apnea or bruxism, it is a medical-device path regardless of charge connector.

**What the meeting does change (product, not law):**

- Fully enclosed + inductive charge is **not** required for Beacon or for a clinic-return SKU.
- Gen1 already charges from the **charge case** (USB-C on the case). Seat the clip so the coils line up. A magnetic dock is a possible later case, not this kit. The link is clip↔case, not a phone Qi pad and not WPC Qi.
- A **USB-C on the MAM pod** (ANR Muscle Sense style) is a fair Gen4 clinic SKU. It drops Track-4 case complexity for that SKU. Keep the charge case for Gen1 kits now.

Ask regulatory counsel before any public line that the extraoral MAM “has no FDA / medical-device requirements.”

### 1.7 Ed does not need ANR + MAM in the n≈5 feasibility (or later clinic use)

This is a real protocol cut. Paper A objective **O6** (Dual A on ≥1 kit session) and the kit BOM (ANR in every box) assumed patients would stack both.

**Do:** ship Oralable-only kits to Beacon. Dual A stays Mac/eng (session `20260812_085110`) and optional John-run methods. Paper A related-work table can still *describe* ANR as the electrical comparator without making patients wear it.

**Do not:** hold kits for ANR stock, Dual A cue cards, or headband compression.

### 1.8–1.9 John’s Gen4: EMG + PPG in one pod; pad attach; potting; recessed electrodes

Workspace hardware generations are only **Gen1** (nRF52832 / BOM REV8 / REV10) and **Gen2** (nRF54L15 / BOM REV9 / REV11). There is no Gen3/Gen4 BOM. Call this concept **MAM Gen4 (combined pod)** so it is not confused with Gen2.

| Gen4 idea | Evaluation |
|-----------|------------|
| PPG + ANR-class EMG in one housing | Matches Ed’s pad attach. Makes Dual A a **single device** later. Analog EMG next to PPG LEDs needs layout isolation (optical window vs gel, LED EMI, shared ground). |
| EMG snaps as the attach | Acceptable to Ed. Same site class as GrindCare (anterior temporalis sEMG) plus optical window. |
| Recessed electrode wells | Sound: pads stick, housing is pulled onto skin, PPG window sits proud of the gel plane. Prototype on a dummy shell before PCB. |
| Transparent silicone over PPG, opaque white on top | Optical path is the risk (thickness, bubbles, yellowing, index vs MAXM86161 window). White top is good against stray light. USB-C through pot is the other risk (strain, ingress). ANR avoids full pot for that reason. |
| USB-C on the pod | Fits clinic recharge. Drops sealed-wireless requirement for this SKU. |

Gen2 (bigger cell, nRF54L15) can still run in parallel **or** be skipped if Gen4 is the next spin. Do not run three hardware programs on Phase 0 cash ([COST_AND_TIMELINE.md](../governance/COST_AND_TIMELINE.md)).

### 1.10 iPhone finds the temporalis bulge from the ear; broaden the patent

Current clinical method is **palpation**: eyebrow tail to ear, hard clench, mark the anterior bulge ([TEMPORALIS_ANATOMY_AND_PLACEMENT.md](./TEMPORALIS_ANATOMY_AND_PLACEMENT.md)). Provisional **64/033,978** already lists an **ear-anchor / pouch matrix** theme ([IP_EVAL_AND_LANDSCAPE.md](../governance/IP_EVAL_AND_LANDSCAPE.md) §2). FIG-CO-039 / 042 are ear-anchor drawings, not Vision placement.

**Broaden (counsel, not git claim text):**

- Phone camera / TrueDepth / face-mesh landmarks: **ear**, lateral canthus, zygomatic arch → estimate anterior temporalis seat.
- Optionally overlay a clench cue and confirm bulge vs the estimate.
- Write this as **craniofacial landmarks and facial morphology**, not “race.” Race is a poor proxy, weak enablement, and a bad ethics line for Beacon / IEEE.

This is **new matter** relative to a purely mechanical ear-anchor. Put it in the conversion spec or in a **second provisional** filed before **~9 Apr 2027**, then combine. Peacock owns the vehicle. Assignment John → JAC for Track 1 is still **open**.

---

## 2. IEEE Paper A — what to change, what to keep

**Keep:** extraoral anterior temporalis OMG @ 50 Hz; n≈5 Beacon feasibility; no AHI; no “beats Bruxoff”; Dual A as *methods physics* (optical vs electrical); McGill / Koorosh methods; Pedro Arm P nested with AcuPebble; wellness / measurement tone.

**Change after you accept this note:**

| Paper A item (today) | After this meeting |
|----------------------|--------------------|
| Research Kit = MAM + **ANR in every kit** | Beacon kit = **MAM + iOS**. ANR optional, John-run |
| O6 Dual A on ≥1 kit session | **Optional.** Eng pack `20260812_085110` stays methods precursor |
| Arm E/J ≥6 h required for stretch | Feasibility evaluable = **≥4 h**; 6 h preferred for hypnogram figure |
| Headband in wear stack | **Out** for Beacon. Pad / tape / later Gen4 snaps |
| Multi-night home charge | **One night**, clinic return and recharge |
| “5 kits to Pedro by 31 Aug 2026” | **Missed.** New handoff date needed (see §5) |

Empty Results tables stay empty until first 4 h nights exist. Short Dual A / Protocol A packs (~6 min) remain layout only ([PAPER_A_VALIDATION_AND_FUTURE_WORK.md](./PAPER_A_VALIDATION_AND_FUTURE_WORK.md)).

**Authorship:** Ed’s role in the meeting matches “clinical protocol / wear” more than a courtesy byline. Lock order on a short follow-up (still TBD).

**Venue:** still `[PLACEHOLDER — OWNER: Koorosh]`. Do not pick JBHI vs EMBC vs Sensors from this meeting.

**Ethics:** still unnamed. One-night clinic loaner + wellness copy may be the easier Beacon path than a stacked Dual A “clinical investigation.” Pedro + Ed must name it.

---

## 3. US patent — clocks and how to broaden

Do **not** paste claim or specification wording here.

| Clock | Date | Status |
|-------|------|--------|
| Track 1 provisional **64/033,978** filed | 9 Apr 2026 | Temporalis / OMG / IR-DC / TFI / SASHB / ear-anchor / pouch |
| Convert / continue | **~9 Apr 2027** | **~7 months** from this meeting |
| Track 2 US **18/289,827** RCE | 12 Jun 2026 | Awaiting examiner (last check in-repo: 22 Jul) |
| EP **4 333 691 B1** opposition | 8 Apr 2027 | Docketed |
| UP request | was 8 Aug 2026 | Receipt still a **GAP** in IP status (22 Jul) |
| Assignment Track 1 John → JAC | — | **Open** |

**Meeting-driven embodiments to add (themes only):**

1. Phone-assisted temporalis **seat from ear and face landmarks** (and optional clench confirm). Morphology, not race.
2. Combined **optical PPG + surface EMG** housing using electrode pads as the attach, PPG window proud / electrode wells recessed.
3. Clinic-return overnight wearable: USB-C on the pod, silicone overmold (clear optical face, opaque dorsal).
4. One-night dental-sleep loaner workflow (charge in clinic, wear, return).

**Vehicle (ask Peacock):**

- Prefer a **second provisional now** if the combined-pod and Vision-placement matter is substantial and not clearly in the Apr 2026 filing. Cheap priority. Merge at convert.
- Or fold into the **conversion** of 64/033,978 if counsel says the Apr spec already supports ear-anchor + phone guidance.
- Track 4 (charge case / enclosure) is **less** central for a USB-C clinic pod. Do not drop it until counsel says the Gen1 case SKU is abandoned.

**Enablement risk:** “iPhone recognises the bulge by race” will not carry a claim. Landmarks + measured anthropometry + a worked example (ear tragus → anterior belly offset) will. Product can still use a lookup by facial type **if** the spec teaches the geometric method.

Public copy stays “US provisional filed / patent pending.” No numbers on oralable.com until counsel says so ([IP_NORTH_STAR.md](../../IP_NORTH_STAR.md)).

---

## 4. Hardware and ops — two tracks, do not serialise

```text
NOW (Beacon n≈5)                    PARALLEL (product + IP)
Gen1 clip · FW 1.0.84               MAM Gen4 combined pod (concept)
Pad / tape attach · no headband     EMG snaps + recessed wells
Clinic charge on charge case (coils aligned)      USB-C on pod · silicone pot
One night · ≥4 h                    Rubbery AccuPebble/Sunrise feel
Oralable-only kits                  PPG + EMG in one housing
iOS 4.3.3 patient app               iPhone landmark placement aid
                                    Convert 64/033,978 by Apr 2027
```

Gen2 nRF54L15 is a **PCB/SoC** upgrade, not Ed’s packaging answer. Decide with Wout / cash whether Gen2 EVT is still worth it if Gen4 is the next enclosure.

**Gen1 attach experiment (this week, before ship):** Red Dot or equivalent on the Gen1 optical face vs tape-only vs leftover silicone. Measure `couple` (unknown / 0 / 1 / 2) over 30–60 min. If pads hold a 4 h night without headband, Beacon can start.

---

## 5. Timelines and milestones (revised)

Canonical calendar: Phase 0 **now–Sep 2026**; kit handoff **30 Sep 2026** (31 Aug missed). Point B funding target remains **€180k by Oct 2026** ([COST_AND_TIMELINE.md](../governance/COST_AND_TIMELINE.md)).

| When | Milestone | Owner | Gate |
|------|-----------|-------|------|
| **This week** | Write meeting minutes to Ed/Pedro (use rules only; no patent claims) | John | Confirm 4 h, one night, no ANR, no headband, pads OK |
| **This week** | Gen1 pad-attach soak (`couple` + 4 h desk/temple) | John | Wear without headband |
| **By 21 Sep 2026** | Charge-to-temple on **Oralable-only** kits (ANR optional, not blocking) | John | FW **1.0.84** · TestFlight **4.3.3** build **5** |
| **By 30 Sep 2026** | Handoff kits to Beacon; dry-run one night in clinic | John → Ed/Pedro | Ethics path named (even if “wellness feasibility”) |
| **Oct 2026** | First **≥4 h** nights (n=1–2: Ed/Pedro self) | Ed/Pedro | Share Clinical Temporalis PDF ([PAPER_A_DATA_HANDOFF_SOP.md](./PAPER_A_DATA_HANDOFF_SOP.md)) |
| **Oct 2026** | Point B cash | John / Ken | Do not fund Gen4 from kit delay excuses |
| **Q4 2026** | n≈5 Core + ≥4 h; Arm P MAD log if Pedro wants | Beacon | Paper A Results skeleton |
| **Q4 2026** | Peacock: second provisional **or** conversion outline (ear landmarks + combined pod + USB-C clinic SKU) | John + Peacock | Assignment John → JAC |
| **Q4 2026 – Q1 2027** | Phase 1+ muscle UX on **same Gen1** (IR-DC / TFI / SASHB) | App | Independent of Gen4 |
| **~9 Apr 2027** | Convert **64/033,978** | Peacock | Hard clock |
| **2027** | Gen4 EVT only if pad-pod is the ship SKU | HW | After n≈5 is running, not before |

**Do not:** slip n≈5 until Gen4 exists. That would miss Paper A, Point B evidence, and Ed’s actual ask (use the information, fix the packaging later).

---

## 6. Open asks (still on Pedro/Ed)

From the August status note, still open, now sharper:

1. Beacon ethics / consent name.  
2. First clinic night (one unit, pad attach, ≥4 h).  
3. Authorship / Beacon line.  
4. Bruxoff access? (nice; not blocking).  
5. AcuPebble SKU / photo if they will share (methods figure).  
6. **New:** confirm pad brand they already use in clinic (ECG) so Gen1 attach matches.

---

## 7. Recommended doc updates (done 7 Sep)

Accepted and applied:

1. This file stays the meeting source of truth.  
2. [PAPER_A_FEASIBILITY_PROTOCOL.md](./PAPER_A_FEASIBILITY_PROTOCOL.md) — Dual A lab only; ≥4 h evaluable; one-night clinic return; no headband.  
3. [ORALABLE_RESEARCH_KIT.md](./ORALABLE_RESEARCH_KIT.md) — ANR and headband out of Beacon BOM; pads in.  
4. [PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md](./PAPER_A_IEEE_TEMPORALIS_OMG_DRAFT.md) + [PAPER_A_REVIEW_COVER.md](./PAPER_A_REVIEW_COVER.md).  
5. [PEDRO_STATUS_UPDATE_2026-09.md](./PEDRO_STATUS_UPDATE_2026-09.md) sendable.  
6. [IP_EVAL_AND_LANDSCAPE.md](../governance/IP_EVAL_AND_LANDSCAPE.md) — conversion themes (no claim text).  
7. [PRODUCT_ROADMAP.md](../../PRODUCT_ROADMAP.md) — Gen4 concept vs Gen2; clinic-loaner SKU.

---

*Internal evaluation. Versions from [VERSION_ALIGNMENT.md](../VERSION_ALIGNMENT.md). Not legal or regulatory advice.*
