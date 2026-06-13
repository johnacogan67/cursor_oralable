# Temporalis Data Collection Protocol (REV10 Single-Node)

**Related:** [docs/README.md](./README.md) · [IR_DC_ADC_FORMAT.md](./IR_DC_ADC_FORMAT.md) · [CLINICAL_VALIDATION.md](./CLINICAL_VALIDATION.md) · Firmware **1.0.36+** worn gating (`oralable_nrf`)

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

---

## Validation anchoring (T=0 = 1st 3-tap sync)

All Python validation runs anchor **T=0 at the first 3-tap sync**, not recording start.

| Component | T=0 definition | How set |
|-----------|----------------|--------|
| **self_validate.py** | 1st sync | `--segment-from 1` → `t0_s = start_s` |
| **validation_dashboard** | 1st sync | `segment_from_sync=1` → `anchors[0]` |
| **clinical_summary** | 1st sync | Segments from 1st anchor via `find_all_three_tap_anchors` |
| **Protocol CSV** | `JOHN_COGAN_1ST_SYNC_PROTOCOL.csv` | Header: "T=0 = 1st 3-Tap Sync" |

### Run commands (1st sync)

```bash
python -m src.validation.self_validate data/raw/Oralable_20260304_090927.txt \
  --segment-from 1 -o data/plots/self_validation_from_sync1.png

python -c "
from pathlib import Path
from src.validation_dashboard import run_validation_dashboard
run_validation_dashboard(
    log_path=Path('data/raw/Oralable_20260304_090927.txt'),
    segment_from_sync=1,
    output_path=Path('data/plots/validation_dashboard_sync1.png'),
)
"
```

### Ed/Pedro protocol phases (elapsed from 1st sync)

| Phase | Elapsed (s) | Action |
|-------|-------------|--------|
| 0 | 0–5 | 3-Tap Sync |
| 1 | 30–45 | Max Tonic Clench |
| 2 | 45–60 | Rest |
| 3 | 60–105 | Phasic Grinding |
| 4 | 105–120 | Rest |
| 5 | 120–135 | Swallow/Control |
| 6 | 150–195 | Simulated Apnea |
| 7 | 210–270 | Natural Speech |

**Files:** `data/validation_logs/JOHN_COGAN_1ST_SYNC_PROTOCOL.csv` · plots in `data/plots/ed_presentation/`
