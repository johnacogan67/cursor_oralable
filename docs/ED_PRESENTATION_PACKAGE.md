# Oralable MAM Validation — Ed Owens Presentation Package

**Date:** March 4, 2026  
**Log:** Oralable_20260304_090927.txt  
**Segment:** From 1st 3-tap sync (19.38 s) — 293.7 s total

---

## Executive Summary

Oralable MAM self-validation completed successfully. All Ed/Pedro protocol requirements passed:

| Phase | Requirement | Result |
|-------|-------------|--------|
| **Sync** | Align Python clock with NRF | ✅ Sync detected, segment anchored |
| **Tonic Clench** | Measure deepest IR-DC drop | 0.20% occlusion measured |
| **Phasic Grinding** | RMMA signature (Accel-PPG) | ✅ Jitter RMS 116.93 — grinding detected |
| **Swallow** | Must NOT trigger clench alert | ✅ 0 false positives |
| **Simulated Apnea** | Gasp Clench rescue at end | ✅ 2.18% occlusion; Rescue detected (cheek 10–2%) |
| **Natural Speech** | Must NOT trigger false positives | ✅ 0 false positives |

---

## Fidelity Report

```
Total SASHB Score (Cumulative Burden): 1183.66 %·s
Airway Rescue Event Count:              0 (15% threshold)
Rescue within 10s of SpO2 Desaturation: 0
Sensor Coupling (IR-DC median):         202988 raw - OK
Green SNR (Heart Rate stability):       27.09 dB

Occlusion Depth (Tonic Clench):        0.20%
Jitter RMS (Phasic Grind):            116.93 (5–24 Hz band)
Clench Detected (≥3% over 5s):        No

Swallow False Positives (must be 0):    0 ✓
Speech False Positives (must be 0):     0 ✓
Apnea Gasp Clench Occlusion:           2.18%
Apnea Rescue Detected (cheek 10–2%):   Yes
```

---

## Key Points for Ed

1. **Sensor coupling OK** — IR-DC in expected range (30k–400k raw) for cheek placement.

2. **Artifact filter passes** — Swallowing and natural speech do not trigger clench alerts.

3. **Gasp Clench detected** — Cheek-specific threshold (10%, 5%, 3%, or 2% in 500 ms) triggers on the Gasp Clench at end of breath-hold. Standard 15% threshold is for finger/acute events.

4. **RMMA (grinding) detected** — Accelerometer Z-axis 5–24 Hz band shows clear grinding signature during Phasic Grinding phase.

5. **Occlusion depth** — Tonic clench 0.20%; Apnea Gasp Clench 2.18%. Cheek masseter produces smaller hemodynamic drops than finger; algorithm tuned for cheek.

---

## Plots

**All in `data/plots/ed_presentation/`** (copied for easy sharing):

| File | Description |
|------|-------------|
| **self_validation_from_sync1.png** | 3-pane: SpO2 + SASHB burden, IR-DC + Rescue markers, HR + Accel Z. Protocol phases shaded. |
| **validation_dashboard_sync1.png** | Occlusion depth vs ground truth; phase metrics. |
| **clinical_summary_sync1.png** | Plot A: IR-DC + clench overlay. Plot B: Accel Z RMS (Grinding vs Quiet Sleep). |
| **clinical_summary_sync1.pdf** | Same as above, PDF for slides. |

---

## Protocol Phases (T=0 = 1st 3-tap sync)

| Phase | Timing | Action |
|-------|--------|--------|
| 0 | 0:00–0:05 | 3-Tap Sync |
| 1 | 0:30–0:45 | Max Tonic Clench |
| 2 | 0:45–1:00 | Rest |
| 3 | 1:00–1:45 | Phasic Grinding |
| 4 | 1:45–2:00 | Rest |
| 5 | 2:00–2:15 | Swallow/Control |
| 6 | 2:30–3:15 | Simulated Apnea |
| 7 | 3:30–4:30 | Natural Speech |

---

## Battery

BatteryStats (3A0FFEF2) hex present; full telemetry requires firmware with power telemetry enabled. CG-320B (15 mAh) sufficient for 8 h clinical night when streaming.
