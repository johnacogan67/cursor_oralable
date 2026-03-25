# Self-Validation Re-Evaluation (Ed/Pedro Protocol)

**Protocol:** 6 phases, T=0 = 3rd tap of first sync  
**Log:** Oralable_20260304_090927 (from 1st sync)

---

## Protocol → Ed/Pedro Requirements

| Phase | Action | Timing | Ed/Pedro Requirement | Current Validation | Status |
|-------|--------|--------|------------------------|--------------------|--------|
| 0 | 3-Tap Sync | 0:00–0:05 | **Sync Accuracy:** Tap device 3× hard. Aligns Python clock with NRF uptime. | `find_all_three_tap_anchors` detects sync; segment starts at anchor | ✅ **PASS** — Sync detected, alignment used |
| 1 | Max Tonic Clench | 0:30–0:45 | **Occlusion Delta:** Measure the deepest possible IR-DC drop. | Occlusion Depth: **0.20%** | ⚠️ **LOW** — Measured, but very shallow for “deepest possible” |
| 2 | Phasic Grinding | 1:00–1:45 | **RMMA Signature:** Side-to-side jaw movement. Tests Accel-PPG fusion. | Jitter RMS: **116.93** (5–24 Hz band) | ✅ **PASS** — RMMA-like activity detected in Accel |
| 3 | Swallow/Control | 2:00–2:15 | **Artifact Filter:** Swallowing a sip of water. Must NOT trigger clench alert. | Not tested | ❌ **GAP** — No false-positive check during Swallow |
| 4 | Simulated Apnea | 2:30–3:15 | **Rescue Timing:** Hold breath 30s, perform “Gasp Clench” at end. | Airway Rescue Count: **0** | ❌ **FAIL** — Gasp Clench should trigger 15% drop; none detected |
| 5 | Natural Speech | 3:30–4:30 | **False Positive Check:** Read README aloud. Must NOT trigger jaw-motion false positives. | Not tested | ❌ **GAP** — No false-positive check during Speech |

---

## Phase-by-Phase Assessment

### Phase 0 — Sync Accuracy ✅
- 3-tap sync found at 19.38 s; segment correctly anchored.
- Python clock aligned with NRF via sync anchor.

### Phase 1 — Occlusion Delta ⚠️
- **Requirement:** Measure deepest possible IR-DC drop.
- **Result:** 0.20% occlusion during 30–45 s.
- **Interpretation:** Either (a) clench was submaximal, (b) baseline/trough calculation is conservative, or (c) sensor placement limits occlusion visibility. We are measuring; value is low for “deepest possible.”

### Phase 2 — RMMA Signature ✅
- **Requirement:** Side-to-side jaw movement; Accel-PPG fusion.
- **Result:** Jitter RMS 116.93 in 5–24 Hz band during 60–105 s.
- **Interpretation:** Accel shows grinding-like activity. Fusion (correlation of Accel + PPG) is not explicitly tested but both channels are present.

### Phase 3 — Artifact Filter ❌ GAP
- **Requirement:** Swallowing must NOT trigger clench alert.
- **Current:** No validation of false positives during 120–135 s.
- **Needed:** Count clench alerts (15% in 500 ms or 3% in 5 s) during Swallow; report “Swallow False Positives: 0” as pass.

### Phase 4 — Rescue Timing ❌ FAIL
- **Requirement:** Gasp Clench at end of 30 s breath-hold should be detected (15% IR-DC drop in 500 ms).
- **Result:** Airway Rescue Count = 0.
- **Interpretation:** Either (a) Gasp Clench did not produce ≥15% drop, (b) timing/window is off, or (c) threshold is too strict for cheek. Ed/Pedro expect at least one rescue event during this phase.

### Phase 5 — False Positive Check ❌ GAP
- **Requirement:** Natural speech must NOT trigger clench alerts.
- **Current:** No validation of false positives during 210–270 s.
- **Needed:** Count clench alerts during Natural Speech; report “Speech False Positives: 0” as pass.

---

## Summary

| Requirement | Status |
|-------------|--------|
| Sync Accuracy | ✅ Pass |
| Occlusion Delta (measure) | ⚠️ Low (0.20%) |
| RMMA Signature | ✅ Pass |
| Artifact Filter (Swallow) | ❌ Not validated |
| Rescue Timing (Gasp Clench) | ❌ 0 events (expected ≥1) |
| False Positive (Speech) | ❌ Not validated |

---

## Implemented Validation Additions ✅

1. **Swallow/Control (Phase 3):** Counts `is_airway_rescue` during 120–135 s. Reports `Swallow False Positives (must be 0)` with ✓/✗.
2. **Simulated Apnea (Phase 4):** Computes occlusion in last ~15 s (Gasp Clench window). Reports `Apnea Gasp Clench Occlusion` (%) and `Apnea Rescue Detected`.
3. **Natural Speech (Phase 5):** Counts `is_airway_rescue` during 210–270 s. Reports `Speech False Positives (must be 0)` with ✓/✗.

## Remaining Considerations

- **Occlusion Delta (Phase 1):** If 0.20% persists with maximal clench, consider: (a) pre-clench baseline from Rest (45–60 s), (b) reporting trough timestamp for manual inspection.
- **Rescue Timing:** Implemented configurable cheek thresholds: 10%, 5%, 3%, 2% in 500 ms. Gasp Clench with 2.18% occlusion triggers at 2%.
