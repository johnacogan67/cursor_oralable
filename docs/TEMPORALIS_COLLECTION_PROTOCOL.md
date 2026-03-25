# Temporalis Data Collection Protocol (REV10 Single-Node)

**Target:** Temporalis Anterior (Temple)

**Sampling Rate:** 50 Hz (standardized via computer BLE logger)

**Objective:** Create labeled training set for Core ML MAM Net

---

## Phase 1: Physical Setup

1. **Locate Target:** Place fingers on the temple. Clench teeth firmly. Locate the peak bulge.
2. **Mounting:** Secure the headband so the sensor window is directly over the peak bulge.
3. **Tension Check:** Ensure the strap feels firm but comfortable (target: 5–15 mmHg).

---

## Phase 2: The 10-Minute Lock Sequence

| Time offset | Action | Clinical target |
|-------------|--------|-----------------|
| 00:00 – 01:00 | Rest (quiet) | Establish IR-DC baseline voltage (target **1.5 V–2.5 V**). |
| 01:00 – 01:10 | Sync-Taps | Five firm, rhythmic taps on the sensor housing for alignment. |
| 01:10 – 02:00 | Rest | Allow signal to settle after movement. |
| 02:00 – 02:10 | Max Tonic Clench | Clench teeth as hard as possible for 10 s (HOI anchor). |
| 02:10 – 03:00 | Rest | Observe HOI recovery (blood volume return). |
| 03:00 – 03:20 | Phasic Grinding | Rhythmic side-to-side jaw movement for 20 s. |
| 03:20 – 04:00 | Rest | Clear accelerometer jitter baseline. |
| 04:00 – 04:20 | Simulated Apnea | Hold breath for 20 s (verify SpO₂ dip). |
| 04:20 – 04:30 | Tonic Rescue | Perform a 10 s clench at the end of breath-hold. |
| 04:30 – 06:00 | Final Recovery | Absolute stillness for 90 s. |

---

## Phase 3: Post-Collection Validation

- **Baseline audit:** IR-DC **< 2.8 V** (strap tension within calibrated range; see Phase 2 baseline target 1.5 V–2.5 V during rest).
- **HOI crash:** Visible **~15%** drop during clench.
- **File format:** CSV saved as **`TEMPORALIS_RAW_01.csv`**.

---

## Ground truth usage

This document is the reference for temporal offsets used in **`label_enum`** (e.g. Phase 5 alignment) when processing recordings that follow this protocol. Offsets are wall-clock relative to recording start; accuracy within **1–2 seconds** is required for CNN-LSTM training alignment.

**Logger command (reference):**

```bash
python src/utils/ble_logger.py --out TEMPORALIS_RAW_01.csv
```

---

## Three design anchors

1. **Temporal precision:** Table offsets (e.g. 01:00, 02:00) define segment boundaries for labeling.
2. **Clinical targets:** Each action (Sync-Tap, Tonic, Phasic, Rescue) maps to expected IR and accelerometer signatures (HOI, motion, recovery).
3. **Hardware guardrails:** 1.5 V–2.5 V baseline audit ensures strap tension stays in the calibrated range before trusting labels.
