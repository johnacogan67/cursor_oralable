# Oralable Algorithm Architecture: Python ↔ iOS Swift

This document describes how to separate algorithm design (developed in Python in this repo) from the iOS app and use the same algorithms in both.

---

## 1. Current State

### Python (cursor_oralable)

| Module | Purpose |
|--------|---------|
| `src/analysis/features.py` | Butterworth filters (0.5–8 Hz bandpass, <1 Hz lowpass), beat detection, IR DC baseline, 5s window biomarkers (ir_dc_shift, rise_fall_symmetry, HRV SVD) |
| `src/analysis/visualize_test.py` | Bandpass 0.5–8 Hz, median filter for accel, rolling mean for IR DC trend |
| `src/utils/sync_align.py` | 3-tap sync detection on accel Z (2s window, 3σ threshold) |
| `src/parser/log_parser.py` | TDM parsing, 50 Hz resampling |
| `src/processing/resampler.py` | 50 Hz linear interpolation |

### Swift (oralable_swift)

| Component | Purpose |
|-----------|---------|
| `UnifiedBiometricProcessor` | HR (peak detection), SpO2 (R-value), motion compensation, perfusion index |
| `SignalProcessingPipeline` | Placeholder HR/SpO2, motion compensation, activity classification |
| `OralableCore` | Shared package (MotionCompensator, ActivityClassifier, CSV parsing, etc.) |

**Gap:** Swift uses different algorithms (peak detection, empirical SpO2 curve) and does not yet implement the Python-derived filters, beat morphology, or bruxism biomarkers.

---

## 2. Architecture: Single Source of Truth

```
┌─────────────────────────────────────────────────────────────────┐
│  ALGORITHM SPEC (YAML/JSON)                                      │
│  - Filter params (0.5–8 Hz, <1 Hz, order 4)                      │
│  - Window sizes (50 samples, 5s, 100 samples)                    │
│  - Sync tap params (2s window, 3σ, min 80ms between taps)        │
│  - SpO2 calibration coefficients                                │
└─────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────────┐            ┌─────────────────────────────┐
│  Python (Research)   │            │  Swift (Production)          │
│  - features.py       │            │  - OralableCore/Algorithms   │
│  - visualize_test.py │            │  - Accelerate (vDSP) filters  │
│  - sync_align.py     │            │  - Core ML (if ML models)    │
│  - scipy.signal      │            │  - CircularBuffer(100)      │
└─────────────────────┘            └─────────────────────────────┘
```

---

## 3. Implementation Plan

### Phase 1: Algorithm Spec (Shared Parameters)

Create `src/config/algorithm_spec.yaml`:

```yaml
# Oralable MAM Algorithm Specification
# Single source of truth for Python and Swift

sampling:
  ppg_hz: 50.0
  accel_hz: 100.0

filters:
  ppg_bandpass:
    lowcut_hz: 0.5
    highcut_hz: 8.0
    order: 4
  ir_dc_lowpass:
    cutoff_hz: 0.8
    order: 4
  accel_median:
    window: 5

buffers:
  ppg_circular_size: 100   # 2s at 50 Hz
  hr_window_samples: 150  # 3s
  spo2_window_samples: 150

sync_taps:
  window_seconds: 2.0
  sigma_threshold: 3.0
  min_distance_ms: 80

beat_detection:
  min_distance_samples: 20   # ~0.4s, max 150 bpm
  prominence_factor: 0.5    # std * factor

spo2_calibration:
  # Empirical: SpO2 = a*R² + b*R + c
  a: -45.060
  b: 30.354
  c: 94.845
```

Python loads this via `pyyaml`; Swift loads it from a bundled JSON (or hardcoded constants generated from the spec).

---

### Phase 2: Swift Algorithm Module

Add an **Algorithms** layer inside `OralableCore` (or a new `OralableAlgorithms` package):

```
OralableCore/
  Sources/
    OralableCore/
      Algorithms/
        ButterworthFilter.swift    # vDSP-based IIR
        PPGProcessor.swift         # Bandpass + beat detection
        IRDCProcessor.swift        # Lowpass + rolling mean
        SyncTapDetector.swift      # 3-tap on accel Z
        SpO2Calculator.swift       # R-value + calibration
```

**Key Swift implementations:**

1. **ButterworthFilter** – Use `Accelerate` (vDSP) for IIR filtering, or `DSPFilters`-style biquad cascade. Coefficients can be generated in Python and exported:

   ```python
   # In Python: export filter coeffs for Swift
   from scipy.signal import butter
   b, a = butter(4, [0.5/25, 8/25], btype='band')
   # Export b, a as [Double] for Swift
   ```

2. **PPGProcessor** – Circular buffer of 100 samples → bandpass filter → peak detection (same logic as `detect_beats_from_green_bp`).

3. **IRDCProcessor** – Lowpass <1 Hz for occlusion dip; rolling mean for trend.

4. **SyncTapDetector** – Port `_find_three_taps_in_signal` from `sync_align.py`.

---

### Phase 3: Core ML for ML Models

For classifiers (e.g., bruxism vs arousal from window biomarkers):

1. **Train in Python** (scikit-learn, PyTorch, etc.) on `features_windows_5s.csv`.
2. **Export to Core ML**:

   ```python
   import coremltools as ct
   # ... train model ...
   mlmodel = ct.convert(model, inputs=[...])
   mlmodel.save("BruxismClassifier.mlmodel")
   ```

3. **Use in Swift**:

   ```swift
   let model = BruxismClassifier()
   let input = BruxismClassifierInput(ir_dc_shift: ..., rise_fall_symmetry: ..., ...)
   let output = try model.prediction(input: input)
   ```

The `.cursorrules` already specify: *"Prepare models for Core ML"*.

---

### Phase 4: Data Flow in iOS App

```
BLE (50 Hz PPG, ~100 Hz Accel)
    │
    ▼
┌─────────────────────────────────────┐
│  OralableCore.Algorithms             │
│  - Resample accel → 50 Hz (optional) │
│  - Butterworth bandpass (Green)      │
│  - Butterworth lowpass (IR)          │
│  - Median filter (Accel Z)           │
│  - Beat detection → HR, HRV          │
│  - IR DC trend → occlusion indicator │
│  - Sync tap detection (if needed)    │
└─────────────────────────────────────┘
    │
    ▼
UnifiedBiometricProcessor / DashboardViewModel
```

---

## 4. File Layout After Rearchitecture

### Python (cursor_oralable)

```
src/
  config/
    algorithm_spec.yaml      # NEW: shared spec
  analysis/
    features.py              # Load spec, run algorithms
    visualize_test.py
  utils/
    sync_align.py
  processing/
    resampler.py
  parser/
    log_parser.py
```

### Swift (oralable_swift / OralableCore)

```
OralableCore/
  Sources/OralableCore/
    Algorithms/              # NEW
      FilterCoefficients.swift   # From Python export
      ButterworthFilter.swift
      PPGProcessor.swift
      IRDCProcessor.swift
      SyncTapDetector.swift
    Services/
      MotionCompensator.swift   # Existing
      ActivityClassifier.swift  # Existing
```

### Shared

```
cursor_oralable/
  models/
    coreml/
      BruxismClassifier.mlmodel   # Exported from Python
```

---

## 5. Validation Strategy

1. **Unit tests:** Python and Swift produce identical outputs for the same input (use exported test vectors).
2. **Golden files:** Run Python on `session_50hz.csv` → save `features_labeled.csv`, `features_windows_5s.csv`. Swift processes the same CSV (or equivalent) and diff results.
3. **Cross-verify:** Per `.cursorrules`, *"Every clench detection algorithm must be cross-verified against the DC-trough depth in the IR channel."*

---

## 6. Quick Start

1. **Create the spec:** Add `src/config/algorithm_spec.yaml` with the parameters above.
2. **Add a Python export script:** `scripts/export_filter_coeffs.py` to generate Swift-ready coefficients.
3. **Implement `ButterworthFilter.swift`** in OralableCore using vDSP or a biquad implementation.
4. **Replace placeholder logic** in `SignalProcessingPipeline` and `UnifiedBiometricProcessor` with calls to the new Algorithms module.
5. **Train and export** any ML model to Core ML when ready.

---

## 7. References

- [Peter Charlton PWA](https://peterhcharlton.github.io/) – Pulse waveform analysis
- [Zhang et al. 2023 PPG-Net](https://doi.org/10.1016/j.bspc.2023.104567) – Blood flow morphology
- [Apple Accelerate vDSP](https://developer.apple.com/documentation/accelerate/vdsp) – Signal processing on iOS
- [Core ML Tools](https://coremltools.readthedocs.io/) – Python → Core ML export
