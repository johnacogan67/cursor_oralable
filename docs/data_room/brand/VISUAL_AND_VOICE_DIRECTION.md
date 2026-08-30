# Oralable — visual direction & voice

**As at:** 7 Aug 2026 · Pack **1.1.61**  
**Status:** Canonical style truth for pitches, website, partner PDFs, and iOS look-and-feel guidance  
**Owner:** John / JAC · brand assets [brand/README.md](./brand/README.md)  
**Figures index:** [../FIGURES.md](../FIGURES.md) · photo / line catalog [RESEARCH_KIT_PHOTO_SELECTION.md](../clinical/RESEARCH_KIT_PHOTO_SELECTION.md)

**One-liner:** Clean black-on-white brand; Matisse contour for people; **photo** for real hardware; plain Orwell English. Hero hybrid = [FIG-CO-054](../figures/FIG-CO-054-matisse-photo-dual-a-stack.png).

---

## 0. Locked north star

| Layer | Direction | Canonical example |
|-------|-----------|-------------------|
| **Hybrid hero** | Line-drawn person + **photoreal** Oralable / ANR on temple | **[FIG-CO-054](../figures/FIG-CO-054-matisse-photo-dual-a-stack.png)** — locked 7 Aug 2026 |
| **Line craft** | Matisse **contour economy** (sparse black continuous line, white ground) | FIG-CO-043 · FIG-CO-051 |
| **Hardware** | Prefer real photos / photo cutouts when recognition matters | FIG-CO-016 · 026 · 030 · 052–054 |
| **Prose** | Short words, concrete, active — no marketing fog. Orwell, not Hemingway understatement | [.cursor/rules/prose-orwell.mdc](../../.cursor/rules/prose-orwell.mdc) |
| **Brand marks** | Official lockup only; Oralable® · Word of Mouth™ | [brand/oralable_logo_lockup.png](./brand/oralable_logo_lockup.png) |

Do **not** treat Great Wave / Matisse cut-outs / Rembrandt costume / Dürer rhinoceros as style. Borrow **craft**, not famous **subjects** ([RESEARCH_KIT_PHOTO_SELECTION.md](../clinical/RESEARCH_KIT_PHOTO_SELECTION.md)).

---

## 1. Key images (use these)

### 1.1 Hero / partner first look

| ID | File | Role |
|----|------|------|
| **FIG-CO-054** | [FIG-CO-054-matisse-photo-dual-a-stack.png](../figures/FIG-CO-054-matisse-photo-dual-a-stack.png) | **Locked visual direction** — Dual A story |
| FIG-CO-055 | [FIG-CO-055-matisse-photo-oralable-finger-press.png](../figures/FIG-CO-055-matisse-photo-oralable-finger-press.png) | Oralable finger-press hybrid (046 pose + wa15 photo) |
| FIG-CO-052 | [FIG-CO-052-matisse-photo-oralable-temple.png](../figures/FIG-CO-052-matisse-photo-oralable-temple.png) | Oralable-only hybrid |
| FIG-CO-053 | [FIG-CO-053-matisse-photo-anr-temple.png](../figures/FIG-CO-053-matisse-photo-anr-temple.png) | ANR-only hybrid |
| FIG-CO-051 | [FIG-CO-051-matisse-portrait-temple-device.png](../figures/FIG-CO-051-matisse-portrait-temple-device.png) | Line-only portrait craft reference |

### 1.2 Product / kit truth (photos)

| ID | Role |
|----|------|
| FIG-CO-016 | Research Kit flat-lay |
| FIG-CO-013 | Charge dock |
| FIG-CO-026 | ANR M40 + Red Dots |
| FIG-CO-030 | PPG / research module close-up |
| FIG-CO-031–033 | Dual A wear / headband crops |

### 1.3 Line-only (cue cards, methods stubs)

| ID | Role |
|----|------|
| FIG-CO-043 | Dual A stack line (**craft** reference) |
| FIG-CO-041–050 | Placement, kit, PCB, pads — same Matisse contour set |

### 1.4 Avoid for external / Pedro

- Patent Figs 6–8 NotebookLM claim slides  
- Full provisional PDF / claim text  
- Pure Cubist / caricature (Daumier) for clinical partners  
- Inflated overnight *N* figures (FIG-CO-025 = ~6 min layout only)

---

## 2. Visual principles

1. **White ground first.** Black line and black type. Accent sparingly (app tokens — see §5).  
2. **Person = line. Device = photo** when telling the product story (054 pattern).  
3. **One job per frame.** Hero: brand or hybrid wear, one line of truth, one CTA. No stat strips on the first viewport (website).  
4. **Temple is the site.** Extraoral temporalis — show placement clearly; do not imply intraoral product.  
5. **Honest readiness.** Kits gated until charge-to-temple; no “already shipped” imagery.  
6. **Claim-safe IP art.** Patent-pending wording only; embodiment drawings OK; no claim text.

---

## 3. Pitch decks

| Do | Do not |
|----|--------|
| Open Sans / clean sans; black text; white slides | Purple gradients, glow, pill-stat clusters |
| Logo lockup in brand bar | Redraw “Oralable” in a random display font |
| Lead Dual A / kit with **054** or kit photo 016 | Stock “wellness watch” imagery |
| Short speaker notes in plain English | Fog: leverage, seamless, unlock, journey |
| Landscape HTML → `_print.html` → PDF pipeline | One-page Chrome print of interactive deck |

Canonical decks: [PITCH_PEDRO_ED_FF](../pitches/PITCH_PEDRO_ED_FF.md) · [PITCH_KOOROSH](../pitches/PITCH_KOOROSH.md) · [PITCH_DECK_KEN](../pitches/PITCH_DECK_KEN.md) · [PITCH_CEO_CANDIDATE](../pitches/PITCH_CEO_CANDIDATE.md) · [PITCH_TECH_OPERATORS](../pitches/PITCH_TECH_OPERATORS.md).

---

## 4. Website

Follow the same north star as pitches:

- Brand (lockup + Oralable® / Word of Mouth™) is a **hero-level** signal.  
- First viewport: brand, one headline, one short sentence, one CTA group, one dominant image — prefer **054** or temple wear photo.  
- No card clutter in the hero; no floating promo chips on media.  
- Concrete product / temple placement imagery over abstract gradients.  
- Prose: Orwell (see §6).  

Implementation detail lives with the site repo / `WEBSITE.md` when present; this file is the **look and voice** source of truth.

---

## 5. Mobile app (iOS)

| Layer | Guidance |
|-------|----------|
| **Tokens / components** | `OralableCore` DesignSystem · app `DesignSystem.swift` · `Assets.xcassets` (PrimaryBlack / PrimaryWhite / Gray* / Accent) |
| **Tone** | Calm clinical wellness — black/white/gray; accent for status, not decoration |
| **Charts** | Clear IR / vitals; hypnogram-first overnight UX ([OVERNIGHT_NIGHT_REPORT](../OVERNIGHT_NIGHT_REPORT.md)); FIG-CO-025 = layout exemplar only |
| **Onboarding / empty states** | Prefer Matisse contour or hybrid 052–054 over cartoon stock |
| **Copy** | Short labels; honest device state; no diagnosis language in Phase 0 |

Flows: [MOBILE_APP_FLOWS.md](../../../oralable_swift/docs/MOBILE_APP_FLOWS.md).

---

## 6. Text / voice

Permanent rule: [.cursor/rules/prose-orwell.mdc](../../.cursor/rules/prose-orwell.mdc).

| Prefer | Avoid |
|--------|--------|
| Short words; cut what you can spare | Stock marketing fog |
| Concrete facts (FW 1.0.70, gated kits) | Inflated readiness |
| Active voice | Passive-only claim language |
| Tables for versions / BOM | Rewriting locked tables into essay |

Partner email / WhatsApp: same voice as [PEDRO_STATUS_UPDATE_2026-08.md](../clinical/PEDRO_STATUS_UPDATE_2026-08.md).

---

## 7. How to use this file

| Audience | Start here |
|----------|------------|
| Pitch / F&F / Pedro PDF | §0–§3 · drop **054** |
| Website / landing | §0 · §2 · §4 · §6 |
| iOS UI / empty states | §0 · §5 · DesignSystem tokens |
| New figure generation | Match **054** hybrid or **043** line-only · catalog in RESEARCH_KIT_PHOTO_SELECTION |
| Agent / contractor | This file + brand README + prose rule |

---

## 8. Related

| Doc | Role |
|-----|------|
| [ORALABLE_RESEARCH_KIT.md](../clinical/ORALABLE_RESEARCH_KIT.md) | Kit BOM · Dual A · ship |
| [RESEARCH_KIT_PHOTO_SELECTION.md](../clinical/RESEARCH_KIT_PHOTO_SELECTION.md) | Photo / line / hybrid catalog |
| [brand/README.md](./brand/README.md) | Lockup + trademarks |
| [../FIGURES.md](../FIGURES.md) | Full FIG-CO index |
| [VERSION_ALIGNMENT.md](./VERSION_ALIGNMENT.md) | Stack versions for any “current truth” copy |

---

*Founder locked hybrid direction on FIG-CO-054 · 7 Aug 2026. Update this file when the north-star image or brand lockup changes.*
