# Mac → Phone algorithm parity (bookmark)

**Canonical detail:** [../ALGORITHM_ARCHITECTURE.md](../../ALGORITHM_ARCHITECTURE.md) §0.

**One-liner:** Mac Python is the clinical reference. The phone matches unless the registry lists an exception.

| Item | Path |
|------|------|
| Mac clinical suite | `src/analysis/features.py` |
| Overnight states | `src/analysis/overnight_states.py` |
| Phone live owner | `oralable_swift/.../UnifiedBiometricProcessor.swift` |
| Shared filters / SpO2 / MAM | `OralableCore` |
| Golden filter + SpO2 CI | `OralableCore/Tests/.../ParityTests.swift` |
| Spec YAML | `src/config/algorithm_spec.yaml` |

**Standing exceptions:** live causal filters; live trailing SpO2 window; MAM 10% shift gate (classification only); **soft ACC + skin-temp corroboration** (quality / `isWorn` / overnight wear+SASHB when temp present — SpO₂ numbers unchanged). MAM tensor inputs are raw-scale with stride 25 (train-aligned). See architecture §0 table · [SENSOR_CORROBORATION.md](./SENSOR_CORROBORATION.md).

**Dual A SpO₂∩EMG nest:** Mac-only offline (`scripts/align_anr_oralable_concordance.py` + `src/analysis/emg_spo2_nest.py`). iOS Dual Protocol A shows live SpO₂ and exports PPG for Mac nest — phone does not re-implement nest metrics yet. Claim: nest ≠ AHI/ODI. `DUAL_PAIR` may include `skin_temp_mean_c` / `on_skin_fraction`.

**iOS overnight band unlock:** ≥**1 h** worn (`OvernightNightReportBuilder.evaluableWearSeconds`). Ideal / Paper A Arm E/J / cohort recalibration still **≥6 h** (goal 8 h).

**As at:** 30 Aug 2026 · Pack **1.1.68** (parity rules unchanged; Dual A EDF is research export)
