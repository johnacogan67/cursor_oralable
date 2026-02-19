# Claude Instructions: Oralable iOS Swift Refactor & OralableCore

**Purpose:** Give these instructions to Claude or Claude Code to refactor the oralable_swift iOS app, extend the OralableCore package with BLE parsing and algorithm modules, and align with the Python algorithm design in cursor_oralable.

---

## Part 1: Architecture Deep Dive (Read First)

### 1.1 Repository Layout

```
cursor_oralable/                    # Python research repo (this repo)
├── src/
│   ├── parser/log_parser.py       # BLE log parsing (reference)
│   ├── analysis/features.py        # Algorithm design (reference)
│   ├── utils/ble_logger.py        # BLE connection (Python)
│   └── utils/sync_align.py         # Sync tap detection
├── data/raw/                       # BLE log files
└── docs/

oralable_swift_ref/                 # iOS app (clone from github.com/johnacogan67/oralable_swift)
├── OralableApp/
│   ├── OralableApp/                # Main consumer app
│   │   ├── Devices/OralableDevice.swift      # BLE device + parsing
│   │   ├── Managers/
│   │   │   ├── BLECentralManager.swift       # CoreBluetooth wrapper
│   │   │   ├── DeviceManager.swift           # Connection state machine
│   │   │   └── DeviceManagerAdapter.swift   # Binds to ViewModels
│   │   ├── Services/
│   │   │   ├── UnifiedBiometricProcessor.swift  # HR, SpO2, motion
│   │   │   └── SignalProcessingPipeline.swift   # Placeholder pipeline
│   │   └── Managers/BLE/
│   │       ├── SensorDataProcessor.swift
│   │       └── BioMetricCalculator.swift
│   ├── OralableForProfessionals/   # Dentist app target
│   └── OralableWorkspace.xcworkspace
└── OralableCore/                   # Swift Package (separate repo - clone alongside OralableApp)
    └── Sources/OralableCore/
```

**OralableCore:** Separate repository at **https://github.com/johnacogan67/OralableCore.git**

- Clone as sibling of `OralableApp`: `oralable_swift_ref/OralableCore/`
- Workspace references `../../OralableCore` (local path)
- Already contains: `BLE/BLEConstants.swift`, `BLE/BLEDataParser.swift`, `Calculations/` (HeartRateCalculator, BiometricProcessor, etc.)
- **Gap:** `BLEDataParser` does not skip the 4-byte frame counter in PPG/accel packets; `OralableDevice` still uses inline parsing. Algorithm modules (ButterworthFilter, PPGProcessor, IRDCProcessor) are not yet in OralableCore.

### 1.2 BLE Protocol (Oralable MAM)

**Service UUID:** `3A0FF000-98C4-46B2-94AF-1AEE0FD4C48E` (TGM Service)

| Characteristic UUID | Purpose | Packet Format |
|---------------------|---------|---------------|
| `3A0FF001` | PPG (Red, IR, Green) | 4-byte frame_counter + N×12 bytes (3×uint32 per sample) |
| `3A0FF002` | Accelerometer | 4-byte frame_counter + N×6 bytes (3×int16 X,Y,Z per sample) |
| `3A0FF003` | Temperature / Command | 8 bytes: frame_counter + int16 centidegrees |
| `3A0FF004` | Battery (TGM) | 4 bytes: int32 millivolts |

**PPG Packet (3A0FF001):**
- Bytes 0–3: `frame_counter` (uint32 LE)
- Bytes 4+: Samples. Each sample = 12 bytes (3×uint32 LE): **Red @ offset 0, IR @ offset 4, Green @ offset 8**
- Firmware may send 20 or 50 samples per packet. iOS `OralableDevice` assumes 20.
- Sample rate: 50 Hz (20 ms between samples)

**Accelerometer Packet (3A0FF002):**
- Bytes 0–3: `frame_counter` (uint32 LE)
- Bytes 4+: Samples. Each sample = 6 bytes (3×int16 LE): X, Y, Z
- Typically 25 samples per packet at ~100 Hz

**TDM Mode (Python only):** If BLE log shows cyclic zeros (one channel zero per sample), use TDM parsing: combine slots 3k, 3k+1, 3k+2 to reconstruct R, IR, G. iOS direct connection typically does NOT need TDM.

### 1.3 Data Flow (Current)

```
BLE Notification (CoreBluetooth)
    → OralableDevice.didUpdateValueFor (CBPeripheralDelegate)
    → parseSensorData() / parseAccelerometerData() / parseTemperature() / parseBatteryData()
    → readingsBatchSubject.send([SensorReading])
    → DeviceManager.handleReadingsBatch()
    → DeviceManagerAdapter.updateSensorValues()
    → SensorDataProcessor.appendToHistory()
    → BioMetricCalculator / HeartRateCalculator (HR, SpO2)
    → DashboardViewModel (UI)
```

### 1.4 OralableCore Dependencies (Current)

The app imports `OralableCore` for:
- `SensorReading`, `SensorType`, `SensorData`, `PPGData`, `AccelerometerData`, etc.
- `BiometricResult`, `ActivityType`, `MotionCompensator`, `ActivityClassifier`
- `BioMetricCalculator`, `HeartRateCalculator`
- `Logger`, `DeviceInfo`, `BLEDeviceProtocol`, `BLEManagerProtocol`
- `CSVParser`, `CSVExporter`, `DesignSystem`, `BatteryConversion`

**OralableCore** exists at https://github.com/johnacogan67/OralableCore.git and already provides these types. Clone it as a sibling of OralableApp for local development.

---

## Part 2: Instructions for Claude

### Task 1: OralableCore Setup

**OralableCore exists at https://github.com/johnacogan67/OralableCore.git**

1. Clone as sibling of OralableApp: `cd oralable_swift_ref && git clone https://github.com/johnacogan67/OralableCore.git`
2. Open `OralableWorkspace.xcworkspace` — it references `../../OralableCore`
3. Add new **Algorithms/** folder and files: `AlgorithmSpec.swift`, `ButterworthFilter.swift`, `PPGProcessor.swift`, `IRDCProcessor.swift`
4. **Package.swift**: Ensure `Accelerate` framework is available for vDSP (iOS/macOS targets include it by default)

### Task 2: BLEConstants (OralableCore)

**OralableCore already has `BLE/BLEConstants.swift`** with TGM UUIDs and packet sizes. Add to `BLEConstants.TGM` if missing:
- `frameCounterBytes = 4`
- `ppgSampleRateHz: Double = 50.0`
- `accelSampleRateHz: Double = 100.0`
- `ppgSamplesPerPacket = 20`, `accelSamplesPerPacket = 25`

### Task 3: Fix BLEDataParser for Oralable MAM Packet Format

**OralableCore has `BLE/BLEDataParser.swift`**, but it does **not** skip the 4-byte frame counter. The TGM packet format is:
- Bytes 0–3: `frame_counter` (uint32 LE) — **must be skipped**
- Bytes 4+: samples (12 bytes each for PPG, 6 bytes each for accel)

Update `BLEDataParser.parsePPGData` and `parseAccelerometerData` to:

- **PPG:** Start at byte 4 (skip frame_counter). For each sample i: offset = 4 + i×12. Red@offset+0, IR@offset+4, Green@offset+8.
- **Accelerometer:** Start at byte 4. For each sample: offset = 4 + i×6. X, Y, Z as int16.
- Add overloads that return `[SensorReading]` with per-sample timestamps for `OralableDevice` integration, or extend existing methods to accept `sampleDataStart: Int = 4`.

`BLEDataParser.parseTGMBatteryData` and temperature parsing already exist; ensure temperature uses centidegrees format (bytes 4–5 = int16) per firmware.

### Task 4: Create AlgorithmSpec.swift

Port parameters from `cursor_oralable/docs/ALGORITHM_ARCHITECTURE.md` and `src/config/algorithm_spec.yaml` (if it exists):

```swift
public struct AlgorithmSpec {
    public static let ppgBandpassLowHz: Double = 0.5
    public static let ppgBandpassHighHz: Double = 8.0
    public static let ppgFilterOrder: Int = 4
    public static let irDCLowpassCutoffHz: Double = 0.8
    public static let accelMedianWindow: Int = 5
    public static let circularBufferSize: Int = 100
    public static let sampleRateHz: Double = 50.0
}
```

### Task 5: Create ButterworthFilter.swift (OralableCore/Algorithms)

Implement a Butterworth IIR filter using Accelerate (vDSP) or a biquad cascade. The Python reference uses:
- `scipy.signal.butter(order=4, [low/nyq, high/nyq], btype='band')` for bandpass
- `scipy.signal.filtfilt(b, a, data)` for zero-phase filtering

For real-time Swift, use forward filtering (not filtfilt). Export filter coefficients from Python:

```python
# In cursor_oralable: scripts/export_filter_coeffs.py
from scipy.signal import butter
b, a = butter(4, [0.5/25, 8/25], btype='band')
# Output b, a as Swift array literals
```

Implement `ButterworthFilter` with:
- `init(lowcut: Double, highcut: Double, fs: Double, order: Int)` for bandpass
- `init(cutoff: Double, fs: Double, order: Int)` for lowpass
- `filter(_ input: [Double]) -> [Double]`

### Task 6: Refactor OralableDevice to Use OralableCore Parser

In `OralableDevice.swift`:
1. Import OralableCore (already done)
2. Replace inline `parseSensorData`, `parseAccelerometerData`, etc. with calls to `OralableBLEParser.parsePPGPacket`, `parseAccelerometerPacket`, etc.
3. Keep the same delegate flow: `didUpdateValueFor` → route by characteristic UUID → call parser → `readingsBatchSubject.send(readings)`

### Task 7: Add TDM Parsing Support (Optional)

If BLE logs show cyclic zeros, add `OralableBLEParser.parsePPGPacketTDM` that implements the logic from `log_parser.py` lines 386–421:
- Slot 3k: (IR, G) → use IR, G
- Slot 3k+1: (R, G) → use R
- Slot 3k+2: (R, IR) → discard or use for validation
- Combine into one (R, IR, G) per logical sample.

### Task 8: Connect to Oralable MAM Over BLE (Verification)

The existing flow already connects:
1. `BLECentralManager.startScanning()` — optionally filter by `[tgmServiceUUID]`
2. On discovery: `BLECentralManager.connect(to: peripheral)`
3. `OralableDevice.discoverServices()` → `discoverCharacteristics()` → `enableNotifications()`
4. `peripheral.setNotifyValue(true, for: characteristic)` for 3A0FF001, 3A0FF002, 3A0FF003, 3A0FF004

Ensure `OralableDevice` discovers and enables notifications for all four characteristics. The current implementation does this; verify no regressions.

### Task 9: Replace Placeholder Algorithms in UnifiedBiometricProcessor

In `UnifiedBiometricProcessor` (or equivalent):
1. Use `ButterworthFilter` for PPG bandpass (0.5–8 Hz) before peak detection
2. Use lowpass (<1 Hz) for IR DC trend (occlusion)
3. Use `CircularBuffer(100)` for 50 Hz stream as per .cursorrules
4. Align beat detection with `features.py` `detect_beats_from_green_bp` (min_distance ~0.4s, prominence = std*0.5)

---

## Part 3: File Creation Checklist

| File | Location | Status |
|------|----------|--------|
| `Package.swift` | OralableCore/ | Exists |
| `BLEConstants.swift` | OralableCore/Sources/.../BLE/ | Exists — extend with frame counter, sample rates |
| `BLEDataParser.swift` | OralableCore/Sources/.../BLE/ | Exists — fix to skip 4-byte frame counter |
| `AlgorithmSpec.swift` | OralableCore/Sources/.../Algorithms/ | **Create** — shared params |
| `ButterworthFilter.swift` | OralableCore/Sources/.../Algorithms/ | **Create** — IIR filter |
| `PPGProcessor.swift` | OralableCore/Sources/.../Algorithms/ | **Create** — bandpass + beats |
| `IRDCProcessor.swift` | OralableCore/Sources/.../Algorithms/ | **Create** — lowpass + trend |

---

## Part 4: Reference Files to Read

Before implementing, read these files in cursor_oralable:

1. **`src/parser/log_parser.py`** — Full BLE log parsing, TDM logic, hex parsing
2. **`src/utils/ble_logger.py`** — BLE characteristic UUIDs, log format
3. **`src/analysis/features.py`** — `_butter_bandpass`, `_butter_lowpass`, `compute_filters`, `detect_beats_from_green_bp`
4. **`src/analysis/visualize_test.py`** — `butter_bandpass_filter`, `median_filter_1d`
5. **`src/utils/sync_align.py`** — `_find_three_taps_in_signal` (for future SyncTapDetector)

And in oralable_swift_ref:

1. **`OralableApp/Devices/OralableDevice.swift`** — Current parsing, delegate flow
2. **`OralableApp/Managers/BLECentralManager.swift`** — BLE scan/connect
3. **`OralableApp/Managers/DeviceManager.swift`** — Connection state machine
4. **`OralableApp/Services/UnifiedBiometricProcessor.swift`** — HR, SpO2, buffers

---

## Part 5: Prompt to Paste into Claude

Copy and paste this block when starting a new conversation:

---

**System context:** I am refactoring the Oralable iOS Swift app (oralable_swift) to:
1. Create the OralableCore Swift Package with BLE parsing and signal processing algorithms
2. Align BLE parsing with the Python reference in cursor_oralable (log_parser.py)
3. Connect to the Oralable MAM device over BLE and parse PPG, accelerometer, temperature, and battery data
4. Use the algorithm design from cursor_oralable (Butterworth filters, beat detection, IR DC trend)

**Instructions:**
1. Read the architecture document at `docs/CLAUDE_IOS_REFACTOR_INSTRUCTIONS.md` (this file)
2. Read the reference Python files: `src/parser/log_parser.py`, `src/analysis/features.py`, `src/utils/ble_logger.py`
3. Read the Swift files: `OralableApp/Devices/OralableDevice.swift`, `OralableApp/Managers/DeviceManager.swift`
4. **OralableCore** is at https://github.com/johnacogan67/OralableCore.git — clone as sibling of OralableApp
5. Fix `BLEDataParser` to skip 4-byte frame counter; add AlgorithmSpec, ButterworthFilter, PPGProcessor, IRDCProcessor to OralableCore
6. Refactor OralableDevice to use OralableCore's BLEDataParser for parsing (with frame-counter-aware overloads)
7. Ensure BLE connection flow works for 3A0FF001, 3A0FF002, 3A0FF003, 3A0FF004

**Constraints:**
- PPG packet: 4-byte frame_counter + N×12 bytes (Red, IR, Green per sample at offsets 0, 4, 8)
- Accelerometer: 4-byte frame_counter + N×6 bytes (X, Y, Z int16 per sample)
- Sample rates: PPG 50 Hz, Accel 100 Hz
- Filter params: bandpass 0.5–8 Hz for HR, lowpass <1 Hz for IR DC

---
