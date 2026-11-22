# OmniRing and later Penn State rings

**As at:** 18 Nov 2026 · Pack **1.1.74**  
**Tone:** open-web literature bookmark · **not** Oralable clearance evidence · Stage A wellness for Oralable  
**Related:** [HAPPY_RING.md](./HAPPY_RING.md) · [LITERATURE_AND_PRIOR_ART.md](./LITERATURE_AND_PRIOR_ART.md) · [GEMINI_TEMPLE_PPG_AVENUES.md](./GEMINI_TEMPLE_PPG_AVENUES.md) · landscape [ORALABLE_MARKET_LANDSCAPE.md](./ORALABLE_MARKET_LANDSCAPE.md)

**One-liner:** OmniRing (IoTDI 2023) is an open **finger-ring** build (nRF52832, IMU, MAX30101) with a real **finger-pose** result and a thin **resting heart-rate** check. The same lab’s 2025 ring, FuSenseRing, shows finger PPG alone misses clinical blood pressure once skin temperature and contact force move. Neither paper measures temple occlusion, bruxism, or a full night.

---

## 1. Bookmarks (primary)

| # | Source | Why it matters |
|---|--------|----------------|
| 1 | [Zhou, Lu, et al., IoTDI 2023](https://doi.org/10.1145/3576842.3582382) · local read `omniring_opt.pdf` | Platform paper. Flex PCB ring, 2.5–2.8 g, ~$25 wholesale, 15 mAh arc cell |
| 2 | [Project page](https://www.cse.psu.edu/~mkg31/projects/omniring/) · [github.com/hzhou3/omniring](https://github.com/hzhou3/omniring) | Cited release. Public repo listing seen 18 Nov 2026 is Python and a citation, not a full PCB and firmware tree |
| 3 | [ASLRing, IoTDI 2024](https://doi.org/10.1109/IoTDI61053.2024.00022) | Motion follow-on. 14 native signers, 26.9% word error. Video-made training data fails under occlusion and bad light |
| 4 | [VisRing, UIST 2025](https://dl.acm.org/doi/10.1145/3746059.3747806) | Same ring class plus a 160×32 bendable display, IMU, PPG, and temperature. 6.6 g, about $35. Comfort study, n=12 |
| 5 | [FuSenseRing, IMWUT 2025](https://doi.org/10.1145/3770689) | Health follow-on. PPG + ECG + skin temperature + contact force. 60 people. The PPG-alone warning |
| 6 | [Gowda lab page](https://www.cse.psu.edu/~mkg31/) · [Taiting Lu](https://www.taitinglu.com/) | Later titles and the NSF I-Corps grant **2333583** (smart ring for finger motion and healthcare) |

### Secondary

| Source | Note |
|--------|------|
| SmartDampener, IMWUT 2024 (listed on the lab page) | Tennis analytics. Same people. Not a health ring |
| *PPG as a Bridge*, IMWUT 2026 (listed on Lu’s page) | PPG used to pair wearables and check identity. No DOI checked here |

---

## 2. What the 2023 paper shows

Hardware: nRF52832, ICM-20948 on SPI, MAX30101 on I2C, dual LDO (3.3 V MCU, 1.8 V sensors), clear resin shell, TPU inner liner, Android app, raw sensors over BLE.

Power, from their numbers: about **12 mA** while streaming and about **76 µA** when the MCU still samples and sends only now and then. On a **15 mAh** cell, continuous stream is about **1.25 hours**. The “week” is the 76 µA mode (about eight days if that current holds). A night of continuous raw PPG does not fit that cell.

Motion: transformer on a few rings, trained from public sign-language video plus augmentation. Against a Leap camera, 12 people (ages 20–47): **6.57 mm** joint position and **8.68°** joint angle. Error stayed flat over about six hours because they reset drift. They do not integrate the IMU for long.

Heart rate: they call it preliminary and leave a full PPG study for later. Users stood, sat, and ran. Rate was taken **after** the motion, while still, from the **green** channel after a bandpass, against a **smartwatch**. The text gives no beats-per-minute error, no SpO2 number, and no night. Twelve people, mixed skin tones, and a line that they saw no large skin-tone gap. That sample is too small to settle skin tone. The reference is another optical watch.

Blood pressure, sleep, emotion, and on-ring processing are named and left open. A user survey (n=12, at home) rated the ring above Tap Strap, Myo, and a glove for comfort, weight, and look. That is wear comfort, not a sleep study.

---

## 3. What they published after

ASLRing (2024) stays on finger motion. VisRing (2025) adds a display and a temperature sensor and still does not report a night of oxygen or jaw load.

FuSenseRing is the health result. The abstract says finger PPG alone misses clinical blood pressure: skin temperature moved the error by **more than 10 mmHg**, and contact force warped the pulse wave. With ECG, temperature, and force, on 60 people, they report systolic **−0.09 ± 7.67 mmHg** and diastolic **0.42 ± 5.74 mmHg** across skin temperatures 17–40 °C, and say this meets AAMI/ISO 81060-2. Those numbers are theirs, on the finger, with extra sensors. They are not an Oralable result.

---

## 4. Oralable implication

The form-factor split is worth keeping: state stream current and sleep current apart. Do not treat a week-long claim as a night of 50 Hz raw PPG.

Their PPG section is a caution even on the finger, the easy site. They trusted heart rate only once the hand was still. FuSenseRing then showed temperature and contact pressure move a finger pulse wave enough to break blood pressure. Pressing harder for a cleaner trace also changes the trace.

That site is not the temple. Oralable’s infrared DC drop is muscle occlusion at the temporalis, with the accelerometer for grind, and green for rate. OmniRing never measured occlusion, bruxism, or an evaluable night. The later result still applies to the vitals half: one PPG channel, without temperature and without a contact check, is a weak base for blood pressure or for an oxygen number shown to a dentist.

Happy Ring remains the cleared finger HSAT in this shelf ([HAPPY_RING.md](./HAPPY_RING.md)). OmniRing is a research platform, not that clearance.

---

## 5. Claim discipline

- Do **not** cite OmniRing heart rate, SpO2, sleep, or blood pressure as validated.
- Do **not** treat FuSenseRing’s AAMI/ISO line as Oralable evidence, or as proof that a temple PPG can do the same.
- Do **not** say the group open-sourced a complete ring kit. The 2023 repo listing is code and a citation. VisRing’s paper says its own hardware and software are shared.
- Do **not** use this file to imply Oralable medical-device status.

---

*Bookmarks from the IoTDI 2023 PDF and open web, 18 Nov 2026. Update if the lab posts a night study, a SpO2 result, or a full hardware release.*
