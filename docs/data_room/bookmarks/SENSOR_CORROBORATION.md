# Bookmark — Soft ACC + skin-temperature corroboration

**As at:** 1 Sep 2026 · Pack **1.1.68**  
**Code:** `OralableCore/SensorCorroboration.swift` · live `BiometricProcessor` / `UnifiedBiometricProcessor` · overnight `OvernightStateClassifier`  
**Related seat QC:** Protocol A **`couple`** (`CouplingGate`) is separate — temple IR/ACC gate for Setup / Dual; not this soft temp gate.  
**Parity bookmark:** [MAC_PHONE_ALGORITHM_PARITY.md](./MAC_PHONE_ALGORITHM_PARITY.md) · architecture [../ALGORITHM_ARCHITECTURE.md](../../ALGORITHM_ARCHITECTURE.md) §0

**One-liner:** Skin temperature (32–38 °C) and ACC motion **soft-gate** vitals quality, `isWorn`, and overnight wear / SASHB credit. SpO₂ and HR **numbers** still compute for Mac Protocol A parity. Missing temperature does not block (Mac overnight path unchanged).

---

## Rules

| Signal | Role | Soft gate |
|--------|------|-----------|
| **Skin temp** | On-body / coupling | Finite and outside **32–38 °C** → off-skin |
| **ACC** | Motion / phasic vs tonic (overnight already); live quality | High motion → quality × **0.3** |
| **Missing temp** | Unknown | Treat as allow (no gate) |

**Live vitals:** multiply `heartRateQuality` and `spo2Quality` by corroboration multiplier; `isWorn` requires on-skin when temp present. Do **not** hard-zero SpO₂/HR.

**Overnight (iOS):** off-skin samples → force **quiet**; exclude from wear seconds and SASHB. Missing temp → IR + ACC + SpO₂ only (Mac `overnight_states.py` parity).

**Protocol A `couple` (not this bookmark):** Mac single + iOS Setup + iOS Dual (MAM) use `CouplingGate` for seat QC (unknown / 0 desk / 1 poor / 2 good). GATT `worn` stays forced for streaming. Soft temp/ACC here does **not** replace `couple`.

**Dual Protocol A:** cue math unchanged. `DUAL_PAIR` may stamp `skin_temp_mean_c` / `on_skin_fraction` and **`couple_*`** when iOS Dual ran MAM seat QC.

---

## Claim discipline

- Soft gate ≠ medical worn-detection claim.  
- iOS overnight **band unlock** is **≥1 h** worn ([OVERNIGHT_NIGHT_REPORT.md](../../OVERNIGHT_NIGHT_REPORT.md)); Beacon feasibility **≥4 h**; hypnogram figure / cohort recalibration still prefer **≥6 h** (goal 8 h).  
- Python Mac overnight states are **not** rewritten in this pass.

---

## Primary paths

| Layer | Path |
|-------|------|
| Shared gate | `OralableCore/.../SensorCorroboration.swift` |
| Live Core | `BiometricProcessor.process(..., temperatureC:)` |
| Live app | `UnifiedBiometricProcessor` + `DeviceManagerAdapter` |
| Overnight | `NightReportSample.temperature` · `OvernightStateClassifier` |
| Dual A meta | `DualProtocolAExport` `DUAL_PAIR` (`skin_temp_*` + optional `couple_*`) |
| Protocol A couple | `OralableCore/CouplingGate.swift` · Mac `src/utils/coupling_gate.py` |
