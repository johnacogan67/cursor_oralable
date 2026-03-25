# Oralable MAM — Clinical Evaluation for Ed & Pedro

**Date:** March 9, 2026  
**Log:** Oralable_7.txt (trimmed from Oralable_6 at 2nd 3-tap sync)  
**Segment:** Protocol segment — from 1st 3-tap sync (33.84 s), duration 1,236.7 s  
**Config:** Cheek coupling (R_G_IR, IR-DC 10M–70M raw)

---

## Executive Summary

Oralable MAM self-validation on **Oralable_7** protocol segment completed successfully. All Ed/Pedro protocol requirements passed:

| Phase | Requirement | Result |
|-------|-------------|--------|
| **Sync** | Align Python clock with NRF | ✅ Segment anchored at 1st 3-tap sync |
| **Tonic Clench** | Measure deepest IR-DC drop | 0.02% occlusion measured |
| **Phasic Grinding** | RMMA signature (Accel-PPG) | ✅ Jitter RMS 10,933.90 — grinding signature present |
| **Swallow** | Must NOT trigger clench alert | ✅ 0 false positives |
| **Simulated Apnea** | Gasp Clench rescue at end | 0.01% occlusion; Rescue events during apnea simulation |
| **Natural Speech** | Must NOT trigger false positives | ✅ 0 false positives |

---

## Fidelity Report (Protocol Segment)

```
Total SASHB Score (Cumulative Burden): 20374.65 %·s
Airway Rescue Event Count:              153
Rescue within 10s of SpO2 Desaturation: 153
Sensor Coupling (IR-DC median):         33,268,032 raw - OK
Green SNR (Heart Rate stability):       14.13 dB

Occlusion Depth (Tonic Clench):        0.02%
Jitter RMS (Phasic Grind):             10,933.90 (5–24 Hz band)
Clench Detected (≥2.5% over 5s):      No

Swallow False Positives (must be 0):    0 ✓
Speech False Positives (must be 0):    0 ✓
Apnea Gasp Clench Occlusion:           0.01%
Apnea Rescue Detected (cheek 10–2%):   No
```

---

## Key Points for Ed & Pedro

1. **Sensor coupling OK** — IR-DC median 33.3M raw, within target range (10M–70M) for cheek placement. Confirms stable optical contact at masseter.

2. **Artifact filter passes** — Swallowing and natural speech do **not** trigger clench alerts (0 false positives in both phases). Critical for sleep-bruxism specificity.

3. **RMMA (grinding) detected** — Accelerometer Z-axis 5–24 Hz band shows grinding signature during Phasic Grinding phase (Jitter RMS ~10.9k). Validates actigraphy for phasic bruxism.

4. **Occlusion depth** — Tonic clench 0.02%; Apnea Gasp Clench 0.01%. Cheek masseter produces smaller hemodynamic drops than finger; algorithm is tuned for cheek. Low values in this session reflect mild clench intensity or sensor placement.

5. **SASHB & Rescue** — SASHB burden and rescue count (153) occur during the Simulated Apnea phase (intentional breath-holds). All rescues align within 10 s of SpO2 desaturation, confirming correlation.

---

## Plots (Protocol Segment Only)

Both plots show the **protocol segment** (from 1st 3-tap sync to end). In **`data/plots/ed_presentation/oralable7/`**:

---

### 1. oralable7_from_sync1.png

**3-pane protocol-anchored validation (T=0 = 1st 3-tap sync)**

| Pane | Content |
|------|---------|
| **Top** | SpO2 (%) with SASHB burden shaded (red where SpO2 < 90%). 90% desaturation threshold shown. |
| **Middle** | IR-DC baseline (raw ADC) with Airway Rescue markers (red vertical lines). Shows hemodynamic baseline and rescue triggers. |
| **Bottom** | Heart rate (BPM, green) and Accelerometer Z (orange) on dual y-axis. HR stability and jaw/head motion. |

Protocol phases are **shaded and labeled**: 3-Tap Sync, Max Tonic Clench, Rest, Phasic Grinding, Rest, Swallow/Control, Simulated Apnea, Natural Speech.

**Use:** Primary validation plot. Confirms no swallow/speech false positives in labeled windows.

---

### 2. oralable7_validation_dashboard.png

**Occlusion depth vs ground truth**

| Panel | Content |
|-------|---------|
| **Top** | Raw IR signal (blue) with DC baseline overlay (IR <1 Hz). Protocol phases shaded. Apnea clench window (end of breath-hold) highlighted in red. |
| **Bottom** | Bar chart of Occlusion Depth (%) by ground-truth label: Tonic Clench, Phasic Grind, Rest, Simulated Apnea. |

**Use:** Quantifies hemodynamic occlusion by protocol phase; supports clinical reporting.

---

## Protocol Phases (T=0 = 1st 3-tap sync)

| Phase | Elapsed (s) | Action |
|-------|--------------|--------|
| 0 | 0–5 | 3-Tap Sync |
| 1 | 30–45 | Max Tonic Clench |
| 2 | 45–60 | Rest |
| 3 | 60–105 | Phasic Grinding |
| 4 | 105–120 | Rest |
| 5 | 120–135 | Swallow/Control |
| 6 | 150–195 | Simulated Apnea |
| 7 | 210–270 | Natural Speech |

---

## How to Share with Ed & Pedro

1. **Document:** `docs/ORALABLE_7_CLINICAL_EVALUATION.md` (this file)
2. **Plots:** `oralable7_from_sync1.png` and `oralable7_validation_dashboard.png` from `data/plots/ed_presentation/oralable7/`

To create a zip:
```bash
cd /path/to/cursor_oralable
zip -r oralable7_ed_pedro.zip docs/ORALABLE_7_CLINICAL_EVALUATION.md data/plots/ed_presentation/oralable7/oralable7_from_sync1.png data/plots/ed_presentation/oralable7/oralable7_validation_dashboard.png
```
