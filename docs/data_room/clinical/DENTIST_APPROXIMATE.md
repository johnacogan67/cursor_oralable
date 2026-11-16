# Dentist approximate — aim and shortcomings

**As at:** 24 Sep 2026 · Pack **1.1.73**  
**Status:** Living note. Update this file when a named result moves a row. Do not copy [MEASUREMENT_CONSTRUCT_MAP.md](./MEASUREMENT_CONSTRUCT_MAP.md) Table 1.  
**Agent rule:** `.cursor/rules/dentist-approximate.mdc`

**One-liner:** The aim is one temple wearable a dentist can use for jaw load and overnight oxygen, and in time for the evaluation AcuPebble, Sunrise, and a sleep study cover today. The current bar is a useful approximate with these shortcomings written down. Equivalence is not the claim.

---

## Aim

Replace, for a dentist’s evaluation of sleep apnea and bruxism, the separate jobs of AcuPebble (apnea count), Sunrise (chin effort), polysomnography with audio-video, and the other peers in [ORALABLE_MARKET_LANDSCAPE.md](../bookmarks/ORALABLE_MARKET_LANDSCAPE.md).

This file records the aim. It does not clear a home sleep test, and it does not equate SASHB with Azarbarzin hypoxic burden or with an apnea–hypopnea index.

## What a dentist can use now

| Picture | Channel | How to say it |
|---------|---------|----------------|
| Sustained clench | Infrared drop on the anterior temporalis | The jaw was loaded, and for how long. Muscle electricity leads this drop by a few seconds. |
| Grind-like motion | Accelerometer | Motion. Tooth contact is unconfirmed. |
| Oxygen load | SASHB: Σ(90 − SpO₂) × 0.02 s while temple SpO₂ is under 90% | How far oxygen sat below 90%, and the low. |
| Coincidence | Jaw-load bout and an oxygen drop on the same night | A hint the clench and the breathing trouble landed together. |

Paired nights against AcuPebble, Sunrise, ANR, or a sleep study are the study that can move a row. They are not a claim that the pod already stands in for those devices.

## Shortcomings

Status is **open**, **narrowed**, or **closed**. Close a row only when the “Moves when” column is met by a named pack, letter, or written protocol change. Cite that result in the changelog.

| # | Shortcoming | Moves when |
|---|-------------|------------|
| 1 | No apnea count. Events that stay above 90% oxygen add nothing to SASHB. | A paired night with AcuPebble or sleep-study event times, plus a written error the dentist accepts. |
| 2 | SASHB adds every second under 90%, including a badly seated pod. Azarbarzin hypoxic burden adds only the dip tied to a scored apnea, from a finger or lab oximeter, per hour asleep. | A same-night comparison of SASHB with event-tied area. |
| 3 | No split between an obstructed airway and a central pause. | A chin or effort channel, or a temple proxy checked against Sunrise or belts. |
| 4 | Hours worn stand in for hours asleep. | Sleep–wake agreement against a scored night. |
| 5 | Temple oxygen is a noisier site than a finger probe. A low night can be coupling. | Paired finger-ox bias on worn nights, with the SpO₂ quality flag reported. |
| 6 | Accelerometer grind includes turns and swallows. | Event match against temporalis EMG or against tooth-sound audio. The construct map already expects this row to be weak. |
| 7 | The optical clench follows the muscle burst. One lab pack (`20260812_085110`, ~6 min) put the median lag at about 4.9 s. | An overnight match, not that six-minute pack alone. |
| 8 | This is not a diagnosis and not a cleared home sleep test. | A regulatory filing. A better algorithm does not close this row. |

## How to update

1. Edit this file in the same session that produces the new result.
2. Bump **As at**. Leave the pack number unless [VERSION](../VERSION) has moved.
3. Add a changelog row: date, what changed, pack or file that justifies it.
4. Change a shortcoming’s status. Leave the aim paragraph as the aim.
5. Do not paste this table into a pitch, a Pedro sendable, or Paper A. Link here.
6. Do not close row 8 because a night “looked like” apnea.

| Status | Meaning |
|--------|---------|
| open | Still true. No named result against it. |
| narrowed | A named result bounds the miss. The row stays. |
| closed | The “Moves when” test is met and the wording above the table can change. |

---

## Changelog

| Date | Change |
|------|--------|
| 24 Sep 2026 | First lock. Aim = dentist evaluation of apnea and bruxism on one temple wearable. Current bar = four pictures above. Rows 1–8 open. Pack **1.1.73**. |
