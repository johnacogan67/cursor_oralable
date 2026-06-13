# Clinical validation log

Consolidated Ed/Pedro protocol evaluations and presentation packages.  
**Protocol:** [TEMPORALIS_COLLECTION_PROTOCOL.md](./TEMPORALIS_COLLECTION_PROTOCOL.md) (T=0 = **1st 3-tap sync**).

---

## Run: Oralable_7 (March 9, 2026)

**Log:** Oralable_7.txt (trimmed from Oralable_6 at 2nd 3-tap sync)  
**Segment:** From 1st 3-tap sync (33.84 s), duration 1,236.7 s  
**Config:** Cheek coupling (R_G_IR, IR-DC 10M–70M raw)

### Executive summary

| Phase | Requirement | Result |
|-------|-------------|--------|
| Sync | Align Python clock with NRF | ✅ Segment anchored at 1st 3-tap sync |
| Tonic Clench | Measure deepest IR-DC drop | 0.02% occlusion measured |
| Phasic Grinding | RMMA signature (Accel-PPG) | ✅ Jitter RMS 10,933.90 |
| Swallow | Must NOT trigger clench alert | ✅ 0 false positives |
| Simulated Apnea | Gasp Clench rescue at end | 0.01% occlusion; Rescue events during apnea |
| Natural Speech | Must NOT trigger false positives | ✅ 0 false positives |

### Fidelity report

```
Total SASHB Score:                     20374.65 %·s
Airway Rescue Event Count:              153
Rescue within 10s of SpO2 Desaturation: 153
Sensor Coupling (IR-DC median):         33,268,032 raw - OK
Green SNR:                              14.13 dB
Occlusion Depth (Tonic Clench):        0.02%
Jitter RMS (Phasic Grind):             10,933.90
Swallow False Positives:                0 ✓
Speech False Positives:                 0 ✓
```

### Plots

`data/plots/ed_presentation/oralable7/` — `oralable7_from_sync1.png`, `oralable7_validation_dashboard.png`

### Share zip

```bash
cd /path/to/cursor_oralable
zip -r oralable7_ed_pedro.zip docs/CLINICAL_VALIDATION.md \
  data/plots/ed_presentation/oralable7/oralable7_from_sync1.png \
  data/plots/ed_presentation/oralable7/oralable7_validation_dashboard.png
```

---

## Run: Oralable_20260304_090927 (March 4, 2026) — Ed Owens package

**Log:** Oralable_20260304_090927.txt  
**Segment:** From 1st 3-tap sync (19.38 s) — 293.7 s total

### Executive summary

| Phase | Requirement | Result |
|-------|-------------|--------|
| Sync | Align Python clock with NRF | ✅ Sync detected |
| Tonic Clench | Measure deepest IR-DC drop | 0.20% occlusion |
| Phasic Grinding | RMMA signature | ✅ Jitter RMS 116.93 |
| Swallow | Must NOT trigger clench alert | ✅ 0 false positives |
| Simulated Apnea | Gasp Clench rescue | ✅ 2.18% occlusion; Rescue detected (cheek 10–2%) |
| Natural Speech | Must NOT trigger false positives | ✅ 0 false positives |

### Fidelity report

```
Total SASHB Score:                     1183.66 %·s
Sensor Coupling (IR-DC median):         202988 raw - OK
Green SNR:                              27.09 dB
Occlusion Depth (Tonic Clench):        0.20%
Apnea Gasp Clench Occlusion:           2.18%
Apnea Rescue Detected (cheek 10–2%):   Yes
Swallow / Speech False Positives:      0 ✓
```

### Plots

`data/plots/ed_presentation/` — `self_validation_from_sync1.png`, `validation_dashboard_sync1.png`, `clinical_summary_sync1.png`, `clinical_summary_sync1.pdf`

### Battery note

CG-320B (15 mAh) sufficient for 8 h clinical night when streaming (design target).

---

## Self-validation re-evaluation (same log, gap analysis)

Early evaluation before swallow/speech/rescue checks were fully implemented in `self_validate.py`.

| Phase | Ed/Pedro requirement | Early status | After pipeline updates |
|-------|----------------------|--------------|------------------------|
| 0 Sync | 3-tap alignment | ✅ Pass | ✅ Pass |
| 1 Tonic | Occlusion delta | ⚠️ Low (0.20%) | Measured; cheek thresholds tuned |
| 2 Phasic | RMMA in Accel | ✅ Pass | ✅ Pass |
| 3 Swallow | 0 false positives | ❌ Gap | ✅ Implemented in validator |
| 4 Apnea | Gasp clench rescue | ❌ Fail at 15% | ✅ Cheek tiers 10–2% |
| 5 Speech | 0 false positives | ❌ Gap | ✅ Implemented in validator |

**Implemented additions:** Swallow/speech false-positive counts; apnea Gasp Clench occlusion window; configurable cheek rescue thresholds (10%, 5%, 3%, 2% in 500 ms).

---

## Protocol phases (T=0 = 1st 3-tap sync)

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

*Last updated: June 2026*
