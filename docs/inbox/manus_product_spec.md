# Muscle Activity Monitoring — Patent Disclosure to Software Specification

**Status:** engineering analysis and proposed update plan; no source-code or model changes performed.  
**Prepared:** 4 November 2026.  
**Staged project inbox:** `/home/ubuntu/MAM_Context/cursor_oralable/docs/inbox/`.

## 1. Authority, provenance, and limits

Primary source: `notebook_lm/Sources/us_patent_2.zip` → `2026-04-08 Oralable MAM Provisional Specification.pdf`, **16 pages**, paragraphs **[0001]–[0088]**, title **APPARATUS AND METHOD FOR MUSCLE ACTIVITY MONITORING**.

Verified extracted PDF: `/home/ubuntu/MAM_Context/_index/nested_sources/us_patent_2/2026-04-08 Oralable MAM Provisional Specification.pdf`. Its bytes match the member in the original ZIP.

```text
SHA-256: f64dbd5a091211165a352988d2319ad69ae9c5e19b9904dfb10c010c24a743f5
```

The accompanying `efilingAck75212743.pdf` identifies provisional application **64/033,978** and receipt on **9 April 2026**. This analysis uses the specification, not an inferred granted claim set. **No appended numbered claims are present in this 16-page file.** References to “appended claims” in [0007] and [0088] do not supply those claims. Therefore, this document establishes disclosure-to-code traceability, not a certification that software exactly matches enforceable legal claims. Patent counsel must supply/review the operative claims and selected embodiments before any claim-conformance assertion.

Classification used below:

- **Definition:** an expressly defined term or qualifier.
- **Disclosed method/embodiment:** a described method or optional “may” alternative, not a universal instruction to implement every alternative together.
- **Non-limiting example:** Example 1 or Example 2, expressly introduced as non-limiting in [0082].
- **Engineering formalization/proposal:** a reproducible implementation expression or safeguard added here, not an equation or numerical constraint quoted from the patent.

All page numbers below are the PDF’s printed pages. Repeated disclosures are consolidated with all relevant paragraph references. Figure numbers, component labels, paragraph numbers, and page numbers are not detection thresholds. Referenced drawings are not imported as additional authority for this specification-only analysis.

## 2. Complete requirement register

### 2.1 Detection predicates, features, and event timing

| ID | Extracted rule and mathematical/timing content | Source / status | Implementation consequence |
|---|---|---|---|
| R01 | “About” or “approximately” means **within 20%** of the cited numerical amount. “A,” “an,” “the,” and “said” mean **one or more** unless context dictates otherwise. | [0087], p.16; definition | Preserve approximation metadata. Do not apply ±20% to unqualified inequalities, ≥4 hours, or example numbers without the qualifier. Endpoint tolerance handling needs review. |
| R02 | Optical myography uses IR **baseline DC-offset shifts** associated with contraction-related blood-volume occlusion; an IR-DC baseline crash is a proxy for muscle force. Hemodynamic occlusion means temporary displacement of blood volume due to contraction. | [0006], p.2; [0032], p.5 | Preserve the low-frequency/DC signal and baseline-relative direction. No equation converts the signal to newtons or absolute bite force. |
| R03 | First SpO₂ measurement is **less than 90% of its SpO₂ baseline**. | [0024], pp.4–5; [0081], p.14; disclosed method/embodiment | A strict relative predicate, not a fixed saturation cutoff. |
| R04 | First IR-DC measurement is **less than 85% of its IR-DC baseline**. Example 2 describes **>15% signal drop** as an example of the crash. | [0024], pp.4–5; [0081], p.14; [0085], p.16 | These formulations are equivalent only when using the same positive baseline and compatible units/polarity. Exactly 15% drop is excluded. Example 2 is not a universal clench rule. |
| R05 | IR-DC drop occurs **after** the first SpO₂ measurement. Second SpO₂ is **greater than 90% of its baseline**, **after** the first IR-DC measurement. | [0024], pp.4–5; [0081], p.14 | Preserve strict ordering of measurement timestamps and the distinct recovery comparator. No maximum separation, minimum persistence, hysteresis, refractory period, or event-merging rule is specified. |
| R06 | An “oxygen desaturation event” is peripheral oxygen saturation **less than 90%**. Fig.3’s description also uses an absolute **<90%** dip followed by an IR-DC dip and SpO₂ recovery. | [0030], p.5; [0035], p.6; definition/figure description | Compute an absolute desaturation flag separately from R03’s baseline-relative predicate. Never substitute one for the other. |
| R07 | Example 2 says declining SpO₂ is followed by an **“immediate”** IR-DC crash. | [0085], p.16; non-limiting example | No numerical latency is given. Do not invent a 4.9-second delay, a maximum correlation window, or a zero-lag requirement. |
| R08 | “Occlusion Depth” is a drop in an established IR-DC baseline. Example 1 quantifies it as a **percentage drop in IR-DC** in response to muscle movement. | [0028], p.5; [0084], pp.15–16 | Expose signed raw/relative shift and percentage drop with explicit baseline provenance. The patent does not prescribe the estimator, denominator convention, aggregation, or handling of invalid baselines. |
| R09 | **Temporalis Fatigue Index (TFI)** assesses exhaustion of masseter and/or temporalis. Calculate from **spectral shift and/or cumulative slope change of the IR baseline** over **continuous monitoring of at least four hours**. Progressive decline in **DC-baseline recovery rate after tonic clenching** indicates reduced perfusion/recovery capacity. | [0025], p.5; definition; [0078]–[0079], p.14 | Enforce ≥4 hours of qualified continuous monitoring for a metric represented as the defined TFI. No spectral bands, transform, recovery estimator, combination equation, normalization, exhaustion threshold, or gap allowance is supplied. |
| R10 | **Sleep Apnea-Specific Hypoxic Burden (SASHB)** means **total area under an SpO₂ saturation curve**. | [0027], p.5; definition | Literal definition is not a supplied desaturation-deficit-area algorithm. Integration domain, reference level, event segmentation, quadrature, units, and normalization are unspecified. Do not silently substitute a conventional clinical hypoxic-burden formula. |
| R11 | Optical monitoring may use **two or more wavelengths** to isolate muscle activity from circulatory noise. The method may detect more than one wavelength **at the same time**, specifically **660 nm and 880 nm simultaneously** as an example. | [0046], p.7; [0076], p.13; optional embodiment/example | Preserve channel identity and timing. No numerical permitted skew or spectral isolation equation is provided. The 660/880 values are not qualified by “about,” but remain an optional example. |
| R12 | Emission and reflected/transmitted/scattered detection wavelength alternatives: **about 500 nm; about 500–1100 nm; about 600–1000 nm; about 700–900 nm; or about 1100 nm**. | [0046], p.7; [0075]–[0076], p.13; optional embodiments | Hardware/profile metadata, not software frequency bands. Preserve all alternatives and qualifiers; do not confuse nm with Hz. |
| R13 | Hemodynamic occlusion and/or kinetic data may be **resampled to a normalization standard**: **about 10 Hz; about 10–50 Hz; about 15–45 Hz; about 20–40 Hz; about 25–35 Hz; or about 50 Hz**. | [0054], p.8; optional embodiments | No uniquely mandated 50 Hz rate, native acquisition rate, interpolation, antialiasing method, clock, packet rule, or missing-data rule. A selected 50 Hz implementation is an engineering profile supported by one disclosed alternative. |
| R14 | Establish reflected-light baseline while the user is **not contracting** temporalis and/or masseter. Enumerated approximate acquisition periods are in §2.2. | [0077], p.13; optional method | Separate noncontracting baseline acquisition from positive-control contraction. No mean/median/window quality formula is prescribed. |
| R15 | Baseline may be **dynamically updated** over multi-hour monitoring to account for longitudinal hemodynamic shifts and calculate exhaustion metrics. Approximate durations are in §2.2. | [0078], p.14; optional method | No update cadence or adaptive-filter coefficient. Retain a historical baseline/trend sufficient for R09; adaptation must not silently erase the fatigue features. |
| R16 | Compare contracting reflected-light measurements with baseline to establish a **positive control** and assess SNR. Compare repeated measurements with baseline and/or positive control to determine **offset-shift → hemodynamic occlusion pattern → diagnosis/TFI**. | [0079], p.14; optional method | No fixed number of contractions, positive-control acceptance threshold, SNR equation, or pattern classifier is supplied. |
| R17 | On sensor-pocket placement, measure signal strength, calculate SNR, and select the pocket with **highest SNR**. Approximate check durations are in §2.2. | [0068], p.11; [0074], pp.12–13; optional method | Selection is comparative maximization, not an absolute dB cutoff. Define the SNR estimator and tie/missing-data policy before implementation. |
| R18 | Accelerometer measures kinetic/rhythmic muscle movement, associated with **phasic grinding**. PPG is associated with **tonic clenching**; combined SpO₂/IR-DC with **airway rescue**. Accelerometer magnitude, motion power, and IR-DC patterns aid differentiation. | [0012], [0017], p.3; [0036], [0040]–[0041], p.6; [0048], pp.7–8; [0084], pp.15–16 | No magnitude equation, FFT band, rhythm threshold, motion-power threshold, event-duration classification rule, or ML confidence cutoff is specified. The Example 1 activity lengths must not become detector-duration thresholds. |
| R19 | Temporalis and masseter contact-pressure alternatives are **about 3 mmHg; about 3–20 mmHg; about 5–17 mmHg; about 7–15 mmHg; about 9–13 mmHg; or about 17 mmHg**. Pressure is described as essential for optical coupling. | [0055], p.9; [0065], p.10; optional settings | Record fit/coupling metadata; do not claim software measures or enforces mmHg without a calibrated physical measurement. |
| R20 | Example 1 target pressure is **5–15 mmHg**, without “about.” | [0083], p.15; non-limiting example | Keep distinct from R19’s approximate alternative ranges. It is a validation setup target, not a universal operational or clench threshold. |
| R21 | Example 1 resting IR-DC baseline is **1.5–2.5 V**, **ideally <2.8 V**. | [0083], Table 1, p.15; non-limiting example | “Ideally” is a preference, not a universal exclusion rule. The printed range does not define open/closed endpoint logic. Voltage cannot be equated with ADC counts without a calibrated conversion and firmware/gain provenance. |
| R22 | Example 1’s complete timed validation sequence runs from **00:00 to 06:00**, including specified rest, clench, grinding, simulated apnea, rescue, and recovery intervals. | [0083], Table 1, p.15; non-limiting example | Exact schedule reproduced in §2.3. Interpret the interval notation as minute:second, consistent with stated 10/20-second activities. Half-open software labels are an engineering convention. |
| R23 | Perform **five sync taps** in the **01:00–01:10** interval. | [0083], Table 1, p.15; non-limiting example | No tap spacing, amplitude, axis, matching tolerance, or five-tap detection algorithm is prescribed. Do not mistake the count for five consecutive samples. |
| R24 | Example 2 reports temporalis reliability **ICC 0.70** versus traditional masseter **ICC 0.31**. | [0085], p.16; reported validation results | Not classifier thresholds, probabilities, confidence gates, or guaranteed product performance. ICC form, cohort, sample size, and uncertainty are not given. |
| R25 | Values listed include **SpO₂, IR-DC, accelerometer magnitude, kinetic movement, heart rate, SASHB, and MFI**. | [0080], p.14 | **MFI is not defined or expressly equated to TFI.** Do not alias them without review. HR/SpO₂ formulas are not supplied in this file. |
| R26 | Sensors may operate **independently, sequentially and/or simultaneously**. Geometry may use a discrete pocket matrix at defined angular/linear offsets from the ear anchor; tragus-based distance estimation, photogrammetry/3D scanning, and highest-SNR selection guide placement. | [0047], p.7; [0066], p.11; [0070], pp.11–12; [0072], p.12; [0074], pp.12–13 | No sensor sequencing delays, angle values, distances, or placement tolerances are supplied. Metadata must distinguish placement estimates from measured geometry. |
| R27 | “Apnea window” spans **cessation of breathing → oxygen desaturation**. “Airway rescue event” spans **cessation of breathing → restoration of saturation**. | [0029], [0031], p.5; definitions | No numerical duration or direct breathing-cessation measurement algorithm. A SpO₂/IR-DC sequence alone does not provide a measured cessation timestamp. |
| R28 | Haptics/stimulus may follow tonic clenching, phasic grinding, or rescue events and may shift sleep position/induce micro-arousal. | [0067], p.11; [0073], p.12; optional embodiments | No stimulus amplitude, latency, duration, repetition limit, or safety policy. No autonomous intervention is authorized or specified by this update plan. |
| R29 | Asymmetrical “lock-and-key” pockets may permit insertion in **only one orientation**. | [0058], p.9; [0063], p.10; optional embodiments | Physical orientation property, not an accelerometer angular cutoff. Record orientation/fit metadata only where supported; no orientation angle is specified. |

### 2.2 Every enumerated setup/monitoring timing alternative

Every nominal value and endpoint in this table is qualified by **“about”** in the source. R01 preserves the ±20% definition; these alternatives are not all simultaneous requirements.

| Requirement | All disclosed alternatives | Nominal normalized values in seconds |
|---|---|---|
| **R17: pocket signal check** | 30 s; 30 s–10 min; 1–9 min; 2–8 min; 3–7 min; 4–6 min; or 10 min | Point 30; ranges [30,600], [60,540], [120,480], [180,420], [240,360]; point 600 |
| **R14: noncontracting baseline acquisition** | 10 s; 10 s–20 min; 30 s–18 min; 1–16 min; 2–14 min; 4–12 min; 6–10 min; or 20 min | Point 10; ranges [10,1200], [30,1080], [60,960], [120,840], [240,720], [360,600]; point 1200 |
| **R15: dynamic-baseline monitoring** | 1 h; 1–12 h; 2–11 h; 3–10 h; 4–9 h; 5–8 h; 6–7 h; or 12 h | Point 3600; ranges [3600,43200], [7200,39600], [10800,36000], [14400,32400], [18000,28800], [21600,25200]; point 43200 |
| **R09: defined TFI qualification** | **At least four hours continuous monitoring**, not qualified by “about” | **Duration ≥14,400 s**; no patented gap tolerance |

A nominal one-hour monitoring embodiment can support other measurements; it does not satisfy the separately defined ≥4-hour TFI. A dynamic-monitoring period of “about four hours” must not relax the TFI minimum to 3.2 hours. The patent’s description of a multi-hour duration is not a baseline refresh interval.

For a positive nominal scalar `x`, the engineering representation of “within 20%” is `[0.8x, 1.2x]`. Thus “about 50 Hz” has scalar tolerance [40,60] Hz, while the selected model profile may still use exactly 50 Hz. How approximation applies across a range’s endpoints is a review decision; do not automatically widen all operational validator ranges or detector thresholds.

### 2.3 Example 1 — complete six-minute validation schedule

**Authority:** [0083], Table 1, p.15. This is a **non-limiting validation example**, not a patient instruction to perform breath holding and not an operational rule requiring all clenches to last ten seconds.

| Printed interval | Proposed elapsed interval `[start,end)` seconds | Derived duration | Activity / explicit instruction |
|---|---|---:|---|
| 00:00–01:00 | [0,60) | 60 s | Quiet rest; establish IR-DC baseline 1.5–2.5 V, ideally <2.8 V |
| 01:00–01:10 | [60,70) | 10 s | Five sync taps on the housing |
| 01:10–2:00 | [70,120) | 50 s | Rest |
| 02:00–02:10 | [120,130) | 10 s | Max tonic clench |
| 02:10–03:00 | [130,180) | 50 s | Rest / IR-DC recovery |
| 03:00–03:20 | [180,200) | 20 s | Phasic grinding |
| 03:20–04:00 | [200,240) | 40 s | Rest |
| 04:00–04:20 | [240,260) | 20 s | Simulated apnea / breath hold |
| 04:20–04:30 | [260,270) | 10 s | Tonic rescue clench |
| 04:30–06:00 | [270,360) | 90 s | Final recovery / stillness |

The schedule spans 360 seconds. At the **proposed exact 50 Hz engineering profile**, that is **18,000 grid samples** under the half-open convention. Ten- and twenty-second activities correspond to 500 and 1,000 grid samples. The patent supplies neither half-open interval labeling nor sample counts. For now, use simulated/replayed fixtures; any human breath-hold validation requires an approved supervised protocol.

## 3. Engineering formalization — explicitly not patent-supplied equations

Use timestamps and units explicitly. Let `B_IR > 0` be an established, calibrated IR-DC baseline; `I(t)` a measurement in compatible units; `B_SpO2` a valid saturation baseline in percentage points; and `S(t)` a valid saturation measurement in the same convention.

```text
signed_ir_shift(t) = I(t) - B_IR
ir_fraction_of_baseline(t) = I(t) / B_IR
occlusion_depth_percent(t) = 100 * (B_IR - I(t)) / B_IR

relative_desaturation(t1) := S(t1) < 0.90 * B_SpO2
relative_ir_crash(t2)     := I(t2) < 0.85 * B_IR
relative_recovery(t3)     := S(t3) > 0.90 * B_SpO2
relative_rescue_candidate := predicates above AND t1 < t2 < t3

absolute_desaturation(t)  := S(t) < 90.0  # percentage-point representation
```

The percentage-drop equation is a proposed conventional formalization of R08, consistent with R04 under its stated assumptions. It is not printed in the patent. Avoid absolute-value conversion, silent clipping, baseline replacement between comparison stages, and silently treating fractional SpO₂ (0–1) as percent-point SpO₂ (0–100). Baseline strategy, event snapshot/version behavior, and sensor calibration require a documented decision.

`absolute_desaturation` and `relative_desaturation` must remain separate outputs/profiles. For baseline SpO₂ **98%**, the relative boundary is **88.2%**: a measurement of **89%** is absolutely desaturated but does **not** satisfy `<90% of baseline`. Equality at 88.2% satisfies neither the strict relative drop nor strict relative recovery. This deliberately preserves the disclosure’s inequalities rather than adding unapproved hysteresis.

For baseline IR-DC 100 compatible units:

- 85.0 units → 15.0% drop → **not** the strict crash predicate.
- 84.999 units → approximately 15.001% drop → crash predicate **true**.
- 85.001 units → approximately 14.999% drop → crash predicate **false**.

Invalid/nonpositive IR baselines, NaN/Inf, missing channels, timestamp inversions, or unqualified data must return explicit invalid/unknown states rather than fabricate a normal or rescue classification. These are engineering safeguards, not additional patent thresholds.

A stream-derived flag is an **observed pattern/candidate**, not proof of apnea, causal airway rescue, diagnostic accuracy, or force in physical units. The patent’s examples do not establish clinical performance of a new implementation.

## 4. Python and Swift/Core ML change map

### 4.1 Actual code availability

**No actual Python or Swift checkout was supplied or found in this sandbox.** The inbox is a user-approved staging directory, not a newly cloned repository. No live source files, model descriptors, tensor shapes, or current behavior were inspected. All existing-file targets below are **documented but unverified** candidates.

Secondary mapping evidence is the archived `notebook_lm/Oralable Seed A Data Room/05_DEV/05_DEV_260609_ORALABLE_PLATFORM_DOCUMENTATION.pdf`; the reviewed extracted text is `_index/text/d0246d65a972eb20fe31.txt`. Its platform snapshot is dated June 2026. It is used only for implementation location hints, never as a patent threshold source.

| Area | Documented target candidates | Proposed work / requirements |
|---|---|---|
| Parsing and units | `cursor_oralable/src/parser/log_parser.py`; `OralableCore/Sources/OralableCore/BLE/BLEDataParser.swift` | Preserve channel identities, source timestamps, scales, calibration and native rates; avoid voltage/ADC or channel-order assumptions. R02, R11–R13, R21, R25–R26. |
| Resampling | `cursor_oralable/src/processing/resampler.py`; shared Swift processing pipeline | Selected model profile of exactly 50 Hz is consistent with a disclosed alternative, not a universal patent requirement. Preserve missingness and synchronized multi-channel timing. R11–R13. |
| IR/DC baseline and features | `cursor_oralable/src/analysis/features.py`; `OralableCore/Sources/OralableCore/Algorithms/IRDCProcessor.swift` | Baseline/control provenance, signed shift, percent depth, longitudinal recovery/trends. Estimators and adaptive-baseline policy require decisions. R02, R08–R09, R14–R16. |
| Placement/calibration | Proposed Python calibration component under `src/`; actual Swift calibration owner to locate | Highest-SNR pocket selection, complete disclosed duration profiles, fit metadata, explicit ADC-to-V calibration. R14, R16–R17, R19–R21, R26, R29. |
| Detection / labels | `cursor_oralable/src/analysis/label_generator.py`; `OralableCore` EventDetector/StateTransitionDetector classes, exact files to verify | Separate absolute/relative saturation predicates; strict ordered sequence; do not reuse protocol labels as proven clinical events. R03–R07, R18, R22–R23, R27. |
| Protocol sync | `cursor_oralable/src/utils/sync_align.py`; `docs/TEMPORALIS_COLLECTION_PROTOCOL.md`; actual Swift alignment owner to locate | Add versioned five-tap/six-minute Example 1 fixture without silently removing other protocol profiles. R22–R23. |
| Long-session metrics | `cursor_oralable/src/analysis/features.py` (`calculate_tfi()` documented); `scripts/generate_clinical_report.py`; actual Swift metrics owner to locate | Separate continuous coverage qualification from rolling inference. TFI ≥4 h; undefined TFI/SASHB/MFI algorithms stay blocked from authoritative reporting. R09–R10, R15–R16, R25, R27. |
| Swift buffers and inference | `OralableCore/Sources/OralableCore/BLE/SensorDataBuffer.swift`; `OralableCore` MAMInferenceManager class; `Signal/AlgorithmSpec.swift`; `Resources/BruxismMAM_Temporalis.mlpackage` (locations to verify) | Derive tensors from the actual model descriptor/feature manifest; preserve feature order, units, timestamp range, validity and warm-up; reject incompatible input. No model tensor size is stated in the patent. |
| Gold pipeline | `scripts/process_temporalis_gold.py`; Python and Swift parity tests | Golden shared replay fixtures, strict boundary cases, protocol boundaries, continuous coverage, waveform and feature parity. Every requirement R01–R29 mapped in `tasks.json`. |

### 4.2 Historical drift signals — not verified current defects

The June 2026 platform document reports three-tap synchronization and a commit description **“Lower airway-rescue label threshold from 15% to 10%.”** It also documents filter bands/orders, five-second biomarkers, buffer/window sizes, a SpO₂ polynomial, and raw-ADC coupling ranges. These are historical implementation choices, **not numerical content of this provisional specification**.

Investigate them in the real checkout. Do not blindly replace all 10% thresholds with 15%: R04 concerns the specified IR-baseline-relative rescue sequence and an example, not every research classifier. Keep old experimental profiles versioned and label which embodiment each implements. Likewise, add an Example 1 **five-tap** profile rather than silently changing all three-tap recordings.

**Not specified by this patent:** Butterworth cutoff/order, 5-second feature window, 2-second circular buffer, 3-second HR/SpO₂ window, 3σ sync threshold, 80 ms tap spacing, SpO₂ calibration polynomial, raw-ADC fit range, model tensor shape, class logits/thresholds, 4.9-second lag, BLE packet dimensions, and a native MCU-specific sampling rate. Do not relabel these as patent claims.

### 4.3 Core ML buffer contract and fatigue-history separation

Proposed design, pending actual model inspection:

1. Decode raw timestamped samples with explicit physical/channel units and validity.
2. Resample onto the **selected** profile grid; preserve source rate, original timestamps, dropped data, channel skew and calibration metadata.
3. Build features according to a **versioned feature manifest** containing names, order, units, window definitions, rate, stride, normalization, tensor dtype/shape and validity policy. The patent does not supply these fields’ numerical values.
4. Validate the real Core ML model’s input description against that manifest. Reject mismatches, incomplete windows and unknown schema versions; do not zero-pad missing data without an explicit approved model policy.
5. Keep bounded online inference buffers **separate** from persisted/streaming baseline recovery and fatigue-history state. Never require a four-hour tensor solely because TFI needs four-hour observation coverage.
6. Use acquisition/sample time, not BLE arrival time, to evaluate event ordering. Do not impose an invented numeric cross-channel latency while representing the file’s “immediate” example as fully implemented.

Illustrative **derived** counts at exactly 50 Hz: 100 samples = 2 s; 150 = 3 s; 250 = 5 s; 500 = 10 s; 1,000 = 20 s; 3,000 = 60 s; 18,000 = 6 min; 720,000 = 4 h; 2,160,000 = 12 h. Four hours of one raw Float32 channel uses 2,880,000 bytes before timestamps/metadata. These are engineering sizing examples, not patented buffer capacities.

A sample count alone does not establish continuity: 720,000 duplicated, irregular, interpolated-over-gap, or concatenated samples cannot automatically qualify TFI. Specify gap/jitter tolerance and coverage endpoints. A profile grid of N samples at frequency f spans N/f seconds under half-open coverage `[start,start+N/f)`; the final sample timestamp alone is `(N−1)/f` after the first. This convention is proposed and must be consistent across Python and Swift.

## 5. Decisions that must remain explicit

| Decision | Required resolution before affected tasks can pass |
|---|---|
| D01 — operative legal scope | Supply/review actual claims and choose disclosed embodiments; no automatic statement of legal claim conformance. |
| D02 — live source ownership | Inspect real `cursor_oralable`, `OralableCore`, and `oralable_swift` checkouts; confirm exact files, task import schema and model artifacts. |
| D03 — profile and approximation | Choose nominal rates/wavelengths/pressure/durations; decide endpoint tolerance semantics. Preserve source “about” separately from operational settings. |
| D04 — baselines / calibration | Specify baseline estimator, SNR estimator, positive-control acceptance, voltage conversion, gain handling, tie policy, and event baseline snapshot/update semantics. |
| D05 — event timing | Define “immediate,” correlation window, persistence, hysteresis, event merging/refractory policy, and measured-vs-inferred cessation/restoration. Values must remain unset until selected, not silently assigned defaults. |
| D06 — features / model | Define accelerometer magnitude/motion power, rhythm criteria, spectral processing, feature windows, classifier confidence and the actual Core ML tensor contract. |
| D07 — TFI | Choose spectral and/or slope/recovery algorithm, scale, normalization, continuous-data qualification/gap tolerance and any clinical threshold. Do not report a shorter metric as the defined TFI. |
| D08 — SASHB / MFI | Resolve literal saturation AUC versus deficit-area intent, integration domain/units/normalization, and whether MFI is distinct or a drafting error. |
| D09 — physical fit / human validation | Determine measurement provenance for pressure/placement, example protocol safety, empirical performance evidence and authorized use. Human breath holding and haptic interventions are not activated by this plan. |
| D10 — parity tolerances | Select numerical tolerances for Python Float64, Swift Float32/vDSP and model conversion; use identical strict comparison logic. No legal approximation clause is a substitute for numerical parity tolerance. |

## 6. Acceptance and traceability tests

`tasks.json` is a **custom, proposal-only planning payload**, not proof of compatibility with an unseen task runner. It includes requirement and decision references, documented/proposed target paths, dependencies, acceptance criteria, and non-execution safeguards.

Minimum release evidence:

- Every R01–R29 maps to at least one task. Approximate embodiments and non-limiting examples retain their source classification.
- Relative SpO₂ 98% baseline/89% reading versus absolute <90% is tested as different behavior.
- Strict relative IR boundary at 85% is tested immediately below, equal and above; exactly 15% depth is excluded.
- Relative recovery at exactly 90% of baseline is not counted as `>` recovery. Timestamp order is tested independently of packet arrival order.
- Missing/invalid baseline, unit mismatch, duplicate/nonmonotonic timestamps, packet loss, channel skew, unqualified windows, and configuration/model mismatch produce documented invalid/unknown states.
- All ten Table 1 segments are contiguous, nonoverlapping, and cover 360 seconds; exactly five taps are the example’s expected count. Ten-/twenty-second scripted activities do not become universal classifier thresholds.
- TFI is unqualified before 14,400 seconds of continuous coverage and can only become qualified at/after that duration under a documented continuity policy. Qualification does not substitute for an approved TFI formula.
- Python and Swift produce matching channel order, baseline snapshots, feature tensors, units, time ranges, event predicates and validity masks on shared fixtures under approved numeric tolerances.
- ML label/feature changes trigger model/data manifest versioning, revalidation and, where necessary, retraining/export; do not change runtime buffers independently of model input expectations.
- ICC 0.70/0.31 remains source provenance, not a runtime acceptance cutoff. No medical benefit, mmHg measurement, absolute force, causal rescue or operative-claim conformance is asserted without corresponding evidence/review.

**Delivery boundary:** these files propose changes; they do not apply them. No code execution against the project, model retraining, Git commit, deployment, lead generation, external submission, or user-facing intervention was performed.
