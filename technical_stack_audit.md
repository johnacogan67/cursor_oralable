# Technical stack audit

**As at:** 4 Nov 2026  
**Audience:** an external agent (Manus) that must brief work without opening the trees  
**Written from:** files on disk in the Cursor multi-root workspace  
**This file lives in:** `cursor_oralable/technical_stack_audit.md` (the folder that holds `oralable.code-workspace`)

Read this file first. Then read the path it names. Do not treat chat memory, a Drive copy, or an older GitHub snapshot as newer than the file on disk.

## How to read this

The product is not one repository and not a web app. It is a temple wearable (PPG + accelerometer), a phone app, a shared Swift library, and a Mac Python reference. Business notes sit beside the code in `cursor_oralable/docs/`.

There is no application server, no SQL database, and no `.env` file in the four workspace repos.

## Claim lock

Use these strings. Do not invent a newer kit, a ship date, an AHI match, or an FDA/CE clearance.

| Item | String to use | Source |
|---|---|---|
| Docs hub | **1.3.21** | `cursor_oralable/docs/VERSION` |
| Data-room pack | **1.1.74** | `cursor_oralable/docs/data_room/VERSION_ALIGNMENT.md` |
| Patient app | **4.3.3** (build **5**) | same file |
| Gen 1 kit firmware | **1.0.85** (legacy archive) | same file |
| Gen 2 kit firmware | **2.0.x target**. Not a kit version | same file |
| On-disk firmware `app/VERSION` | **1.0.87** (`PATCHLEVEL = 87`) | `oralable_nrf/app/VERSION` |
| Kits | **Gated.** 30 Sep 2026 handoff missed. No new date | `VERSION_ALIGNMENT.md` |
| Hardware in the partner story | Gen 2 · BOM **REV9** · PCB **REV11** · Kaga **ES4L15BA1** (nRF54L15) · cell **LP270829 35 mAh** · sensors **MAXM86161** + **LIS2DTW12** | `VERSION_ALIGNMENT.md` |

`VERSION_ALIGNMENT.md` is the version file. Its opening paragraphs and the “Pilot ship status” table still say a 29 Oct bench image does not stream PPG or accelerometer. A later heading in the same file, **2 Nov 2026**, says one board on image **1.0.87** streamed PPG and accelerometer until the cell fell. A **3 Nov** `merged.hex` candidate carries ELF string `2.0.0-gen2-nrfconnect` and was **not programmed**. Gates G2-P0 through G2-P6 are **pending**. Kits stay gated. Quote the dated row you mean. Do not collapse those rows into one status line.

`oralable_nrf/README.md` still prints firmware **1.0.85**. The file `app/VERSION` prints **1.0.87**. Trust `app/VERSION` for the tree, and `VERSION_ALIGNMENT.md` for what may be called a kit.

## Workspace roots

Open the four folders together with `cursor_oralable/oralable.code-workspace`.

| Id | Path | Git remote | Branch | HEAD (4 Nov 2026) | Uncommitted paths |
|---|---|---|---|---|---|
| research | `/Users/johnacogan67/work/cursor_oralable` | `github.com/johnacogan67/cursor_oralable` | `main` | `67239b0` | 225 |
| firmware | `/Users/johnacogan67/work/oralable_nrf` | `github.com/johnacogan67/oralable_nrf` | `main` | `7f4db62` | 96 |
| ios | `/Users/johnacogan67/work/oralable_swift` | `github.com/johnacogan67/oralable_swift` | `main` | `43c8f6c` | 115 |
| core | `/Users/johnacogan67/work/OralableCore` | `github.com/johnacogan67/OralableCore` | `main` | `c3e7617` | 28 |

The four product trees are dirty. This audit describes the working trees, not a clean `main` snapshot. `workspace.manifest.json` is the map of roles. It also names two sites that are **not** folders in `oralable.code-workspace`:

| Manifest id | Expected path | On this machine |
|---|---|---|
| `public_web` | `../oralable-web` | **Directory absent** |
| `mam_web` | `../mam-website` | Present. Git `main` at `151e6a1`. Working tree clean. **Not** opened in the Cursor workspace |

Large Nordic trees also sit under `/Users/johnacogan67/work/` (`nrf`, `zephyr`, `nrfxlib`, `modules`, `bootloader`, `tools`, `test`). They are SDK checkouts. They are not Oralable product repos. Do not edit them from a business brief.

---

## 1. Core tech stack

### Languages

| Layer | Language | Where |
|---|---|---|
| Reference algorithms, log parse, reports | Python **3.13** only (`requires-python >=3.13,<3.14`) | `cursor_oralable` |
| Device firmware | C, Kconfig, Devicetree, CMake | `oralable_nrf` |
| Shared library | Swift tools **5.9**. iOS 16, macOS 13 | `OralableCore` |
| Phone apps | Swift, SwiftUI | `oralable_swift` |
| Clinic site | HTML, CSS, small JS, Google Apps Script | `mam-website` (outside the Cursor workspace) |
| Contracts and algorithm constants | JSON, YAML | `cursor_oralable/contracts`, `cursor_oralable/src/config` |

### Frameworks and runtimes

| Layer | Stack |
|---|---|
| Python package | `oralable-research` **0.1.0**. Setuptools. Pytest. Entry: `pyproject.toml` |
| Firmware OS | Zephyr via the nRF Connect SDK. West manifest `oralable_nrf/west.yml` pins `sdk-nrf` **v2.7.0** (Gen 1 release path) |
| Gen 2 build | Separate NCS **v2.9.3** workspace and toolchain `b8efef2ad5`. Recorded in `oralable_nrf/firmware.toolchains.json`. Status: evidence-gated, non-release |
| Boot / update | Sysbuild + MCUboot. OTA notes in `oralable_nrf/docs/OTA_DEVICE_MANAGER.md`. Local scripts, not a cloud flasher |
| Phone UI | SwiftUI. Two schemes: `OralableApp` (patient) and `OralableForProfessionals` |
| BLE on the phone | CoreBluetooth through **AsyncBluetooth 6.1.0** |
| On-device model | Core ML package `BruxismMAM_Temporalis.mlpackage` inside OralableCore. Provenance sidecar beside it. Training stays on the Mac |
| Shared math | OralableCore. **No** Swift Package dependencies |

### Third-party pins that matter

Python, from `cursor_oralable/pyproject.toml` (lockfile `requirements.lock`):

| Always installed | Pinned |
|---|---|
| numpy | 2.4.2 |
| pandas | 3.0.1 |
| PyYAML | 6.0.3 |
| scipy | 1.17.1 |

| Extra (`pip install` group) | Pinned |
|---|---|
| `bluetooth` | bleak 2.1.1 |
| `reporting` | matplotlib 3.10.8, plotly 6.6.0, pyEDFlib 0.1.42 |
| `model` | coremltools 9.0, scikit-learn 1.8.0, tensorflow 2.21.0 |
| `test` | pytest 9.0.2, pyEDFlib 0.1.42 |

iOS Swift packages, from `OralableApp.xcodeproj/.../Package.resolved`:

| Package | Version |
|---|---|
| AsyncBluetooth | 6.1.0 |
| swift-algorithms | 1.2.1 |
| swift-async-algorithms | 1.1.1 |
| swift-collections | 1.3.0 |
| swift-numerics | 1.1.1 |

Firmware parts named in board and driver code: Nordic nRF52832 (Gen 1 module ES2832AA2) and nRF54L15 (Gen 2 module ES4L15BA1), Analog Devices MAXM86161, ST LIS2DTW12. Custom Zephyr drivers live under `oralable_nrf/drivers/sensor/`.

### Data stores

| Store | Role | Status |
|---|---|---|
| Files on disk (CSV, nRF Connect logs, EDF, plots) | Research evidence | `cursor_oralable/data/`, `plots/`. Both are gitignored |
| UserDefaults + app Documents | Phone settings, flags, exports | `oralable_swift` |
| CloudKit public database | Patient/professional share records | Container `iCloud.com.jacdental.oralable.shared`. Contract says production schema is **not verified or deployed** |
| Apple Health | Optional sync | `AppleHealthManager.swift` |
| StoreKit config | Subscription products in the Xcode project | `Configuration.storekit`, `ProfessionalConfiguration.storekit`. UI flag defaults off |
| Google Sheets | Clinic-site lead intake | `mam-website` Apps Script. Outside this workspace. Do not copy the sheet into git |

No Postgres, SQLite product database, Redis, or Firebase project is in these repos.

### Signal constants (do not restate loosely)

The Mac/phone algorithm file is `cursor_oralable/src/config/algorithm_spec.yaml` (`oralable-algorithm-v1`, version 1.0.0).

- PPG analysis rate: **50 Hz**
- Accelerometer rate in that YAML: **100 Hz** (firmware still batches a 50 Hz stream; do not “fix” one side from a brief)
- PPG heart-rate band: Butterworth **0.5–8.0 Hz**, order 4
- IR-DC low-pass in the YAML: **0.8 Hz**, order 4
- Circular PPG buffer: **100** samples
- BLE service: `3A0FF000-98C4-46B2-94AF-1AEE0FD4C48E`, advertised name `Oralable`
- Wire contract: `cursor_oralable/contracts/tgm_gatt_v1.json` (`oralable.tgm-gatt` **1.0.0**). Firmware declaration owns the bytes

---

## 2. Architecture overview

### What talks to what

```text
MAXM86161 + LIS2DTW12
        │  I2C
        ▼
oralable_nrf  (Zephyr, TGM GATT, MCUboot)
        │  BLE notify / read
        ├──────────────────────────────► oralable_swift
        │                                 OralableCore parses bytes
        │                                 SwiftUI shows Phase 0 vitals
        │
        └─ nRF Connect CSV / text log ─► cursor_oralable
                                          parser → 50 Hz resample
                                          → filters, reports, Core ML export
                                          → OralableCore model package
```

Contract owners, from `workspace.manifest.json`:

| Contract | Owner | Consumers |
|---|---|---|
| TGM GATT bytes | firmware | research, core, ios |
| 50 Hz algorithms and model provenance | research | core, ios |
| Swift parse, domain types, Core ML, export | core | ios |
| Connection life, storage, screens | ios | none |

Change the owner first. Then the consumers. A business brief does not skip that order.

### `cursor_oralable` — research, docs, contracts

| Path | What lives there |
|---|---|
| `src/parser/` | nRF Connect / HEX logs → CSV |
| `src/processing/` | Resample to 50 Hz |
| `src/analysis/` | Features, labels, overnight states, EMG/SpO2 nest |
| `src/config/` | `algorithm_spec.yaml` and its loader |
| `src/export/` | EDF writer |
| `src/validation/` | Self-check helpers |
| `src/utils/` | BLE log helper, coupling gate, sync align |
| `scripts/` | Session runners, gold CSV, clinical report, contract checkers |
| `tests/` | Pytest for spec, GATT contract, app-service contract, coupling, EDF |
| `contracts/` | `tgm_gatt_v1.json`, `oralable_app_services_v1.json` |
| `docs/` | Protocols, architecture, roadmap |
| `docs/data_room/` | Partner, clinical, governance, firmware flash notes |
| `docs/inbox/` | **Manus drop zone** (ephemeral) |
| `data/`, `plots/`, `models/`, `notebooks/` | Local evidence. Mostly gitignored |
| `Makefile` | `bootstrap`, `test`, `verify` |
| `.github/workflows/python.yml` | Checkout check on GitHub |

There is no HTTP router. “Routing” here is which script you run. The usual chain is: log in `data/raw/` → `src/parser/log_parser.py` → `src/processing/resampler.py` → `scripts/process_temporalis_gold.py` → `scripts/generate_clinical_report.py`.

### `oralable_nrf` — firmware

| Path | What lives there |
|---|---|
| `app/src/main.c` | Bring-up, policy, streaming gates |
| `app/src/tgm_service.c` | GATT service. Source of UUID and payload layout |
| `app/src/ble.c` | Advertising, MTU, connection |
| `app/src/ppg.c`, `acc.c` | Sensor acquire |
| `app/src/battery.c`, `charge_detector.c`, `battery_led_indicator.c` | Cell, charger pin, LED |
| `app/prj.conf`, `app/Kconfig`, `app/VERSION` | Build config and version |
| `boards/byteexplain/pcb00003/` | Gen 1 board (nRF52832) |
| `boards/byteexplain/pcb00003_gen2/` | Gen 2 board (nRF54L15) |
| `drivers/sensor/maxm86161/`, `lis2dtw12/` | Sensor drivers |
| `bringup/`, `bringup_ble/` | Bench images. Not the patient-app release |
| `firmware.toolchains.json`, `gen2.program.json` | Which SDK and board a track uses |
| `scripts/` | Bootstrap, flash, OTA helper, workspace checks |
| `west.yml` | NCS v2.7.0 |

No GitHub Actions workflow in this repo. A firmware change is not done until an nRF Connect log agrees with the contract. Rule file: `oralable_nrf/.cursor/rules/nrf-connect-validation.mdc`.

Gen 1 release board string: `pcb00003/nrf52832`, application `app/`.  
Gen 2 board string: `pcb00003_gen2/nrf54l15/cpuapp`. Tracks `gen2_bringup` (`bringup_ble`) and `gen2_patient_candidate` (`app`) are non-release.

### `OralableCore` — shared Swift

One library target. No external packages. Folders under `Sources/OralableCore/`:

| Folder | Job |
|---|---|
| `Algorithms/`, `Filters/`, `Signal/` | Filters, PPG, IR-DC. Must stay free of SwiftUI and CloudKit |
| `BLE/` | Parse TGM payloads, constants, buffers |
| `Calculations/` | Heart rate, SpO2, coupling, Core ML features |
| `Calibration/`, `Events/`, `Models/` | Session and device values |
| `CSV/`, `Export/` | CSV and EDF out |
| `CloudKit/` | Record schema adapters, including `OralableAppServiceContract` |
| `DesignSystem/`, `SharedComponents/` | Shared visuals. Must not own signal math |
| `Resources/` | `BruxismMAM_Temporalis.mlpackage` and provenance JSON |

Boundary rules: `OralableCore/docs/PACKAGE_BOUNDARIES.md`.  
Tests: `Tests/OralableCoreTests/`. No GitHub Actions file in this repo. The iOS workflow checks out this package.

### `oralable_swift` — two iOS apps

Xcode workspace: `oralable_swift/OralableApp/OralableWorkspace.xcworkspace`.  
The app project references OralableCore at `../../OralableCore` (sibling of `oralable_swift`).

| Path | Job |
|---|---|
| `OralableApp/OralableApp/` | Patient app. Bundle `com.jacdental.oralable` |
| `OralableApp/OralableForProfessionals/` | Dentist app. Bundle `com.jacdental.oralable.professionals`. Contract status: deferred, Phase 1+. Not authorized for App Store submission |
| `Views/` | SwiftUI screens: Dashboard, Overnight, Protocol A, Dual Protocol A, Settings, Share |
| `Services/` | Night report, clinical PDF, Protocol A |
| `Devices/OralableDevice/` | BLE device session |
| `Managers/FeatureFlags.swift` | On/off switches. Stored in UserDefaults. Developer Settings (tap version 7 times) |
| `Managers/SharedDataManager.swift` | CloudKit share |
| `Utilities/FirmwareGate.swift` | Minimum firmware the app will use |
| `OralableAppTests/`, UI test targets | App tests |
| `.github/workflows/ios.yml` | Simulator build and test |

Screen flow is SwiftUI navigation, not URL routes. Phase 0 (Ed/Pedro) is the patient app and temple vitals. The dentist app and CloudKit share are later. Read `oralable_swift/docs/MOBILE_APP_FLOWS.md` before changing a screen.

### `mam-website` — static clinic site (outside the Cursor workspace)

Netlify publishes the repo root. Build command is empty. Forms post to a private Google Apps Script, which writes a private Google Sheet. Procedure: `mam-website/CRM_WORKFLOW.md`. Public host in `netlify.toml`: `mamdiagnostics.com`.

A static site copy also sits at `oralable_swift/OralableApp/oralable-website/`. The manifest’s `oralable-web` repo is not on this machine. Do not assume those three website locations are the same tree.

### Docs an agent should open for a topic

| Topic | Start here |
|---|---|
| System picture | `cursor_oralable/docs/ORALABLE_SYSTEM_ARCHITECTURE.md` |
| Algorithms | `cursor_oralable/docs/ALGORITHM_ARCHITECTURE.md` |
| Overnight report | `cursor_oralable/docs/OVERNIGHT_NIGHT_REPORT.md` |
| Phone flows | `oralable_swift/docs/MOBILE_APP_FLOWS.md` |
| Firmware day-to-day | `oralable_nrf/docs/DEVELOPMENT.md` and `docs/CLEAN_CHECKOUT.md` |
| Gen 2 program gate | `oralable_nrf/docs/GEN2_PROGRAM_GATE.md` |
| Versions | `cursor_oralable/docs/data_room/VERSION_ALIGNMENT.md` |
| Data room index | `cursor_oralable/docs/data_room/README.md` |

`docs/ORALABLE_SYSTEM_ARCHITECTURE.md` was last stamped **2026-09-19** and still centres Gen 1 firmware **1.0.85**. Use it for layer order (hardware → GATT → firmware policy → algorithms → clinical). Use `VERSION_ALIGNMENT.md` for the date and the version string.

---

## 3. Environment and deployment

### Environment variables

| Check | Result |
|---|---|
| `.env`, `.env.example`, `dotenv` in the four workspace repos | **None found** |
| Python config | `pyproject.toml`, `requirements.txt` → `requirements.lock`, local `.venv` via `./scripts/bootstrap_python.sh` |
| Firmware config | `prj.conf`, Kconfig, board DTS, `firmware.toolchains.json`. No secrets file |
| iOS config | `Info.plist`, entitlements, StoreKit files, `FeatureFlags` in UserDefaults. CloudKit container id is in `contracts/oralable_app_services_v1.json` |
| Codex / Cursor MCP | Nordic docs only: `https://nordicsemi.mcp.kapa.ai` in `cursor_oralable/.cursor/mcp.json` and `.codex/config.toml` |

Do not invent an env file to “complete” the loop. Add one only when a real secret or host URL must stay out of git, and ship a `.env.example` with empty values.

### Pipelines that exist

| Pipeline | File | What it does |
|---|---|---|
| Python checkout | `cursor_oralable/.github/workflows/python.yml` | On push to `main`, pull request, or manual run. `macos-14`. Python 3.13. Installs `requirements.lock`, editable package, runs `scripts/verify_research_checkout.py`. Timeout 20 min |
| iOS | `oralable_swift/.github/workflows/ios.yml` | On push and pull request to `main`. `macos-latest`. Schemes `OralableApp` and `OralableForProfessionals`. Checks out `johnacogan67/OralableCore` at ref **`c3e7617`** (this matches local OralableCore HEAD; it does not include the 28 dirty Core files). `xcodebuild test` on an iPhone simulator. `CODE_SIGNING_ALLOWED=NO`. Uploads `.xcresult` for 14 days |
| Firmware | — | **No GitHub Actions workflow** in `oralable_nrf` |
| OralableCore alone | — | **No workflow.** Covered only by the iOS job’s pinned checkout |
| Clinic site | `mam-website/netlify.toml` | Static publish of `.`. No build command. Redirects to `https://mamdiagnostics.com` |

### Pipelines that are absent

No Dockerfile, docker-compose, Vercel config, Render config, Fly config, or Procfile in the four workspace repos. Firmware is built on a Mac with the nRF Connect SDK and flashed over SWD or Device Manager. The patient app is an Xcode archive, and the app-service contract marks App Store submission **blocked pending release gates**.

Local commands that already stand in for CI:

| Repo | Command |
|---|---|
| research | `make bootstrap` then `make verify` (pytest plus checkout script) |
| research contracts | `python3 scripts/verify_tgm_contract.py`, `scripts/verify_app_service_contract.py`, `scripts/verify_algorithm_spec.py` |
| firmware | `oralable_nrf/docs/CLEAN_CHECKOUT.md` and `scripts/verify_firmware_workspace.py`. Flash only when a human asks |
| core | `swift test` in `OralableCore` |
| ios | Xcode schemes above, or the GitHub workflow |

---

## 4. Integration readiness

The loop already has a drop zone. Use it. Do not scatter briefs into firmware or Swift.

### Where a brief goes

| Kind of input | Put it here | Lifetime |
|---|---|---|
| One engineering task for Cursor | `cursor_oralable/docs/inbox/tasks.json` | Ephemeral. `docs/inbox/cleanup.sh` deletes it |
| Prose intent beside that task | `cursor_oralable/docs/inbox/manus_product_spec.md` | Same. The script deletes it too |
| A decision that must survive the task | `cursor_oralable/docs/data_room/` (clinical, governance, pitches) or the matching `docs/*.md` | Durable. Plain English. Do not raise kit, overnight N, AHI, or regulatory claims |
| A byte layout, UUID, or app-service field | `cursor_oralable/contracts/*.json`, then the owner repo | Version the contract. Do not hide a wire change inside a task essay |
| A filter, rate, or buffer length | `cursor_oralable/src/config/algorithm_spec.yaml` | Mac is the reference. Phone follows |
| A new offline chart or table over logs | `cursor_oralable/scripts/` plus `src/` if the math is shared | Read `data/raw/`. Write `data/validation/` or `plots/` |
| A phone on/off switch | `oralable_swift/.../Managers/FeatureFlags.swift` | UserDefaults. There is no remote-config service |
| A clinic survey or lead row | `mam-website` (`join.html`, `apps-script/Code.gs`, `CRM_WORKFLOW.md`) | Google Sheet. Not this workspace |

`tasks.json` is not in `.gitignore`. A commit will take the live brief with it unless `cleanup.sh` has run. Run that script only after the task is implemented and the verifiers named in `.cursorrules` have passed (`pytest`, `make verify`, contract scripts, nRF Connect check when BLE changed).

### Shape Cursor already expects

`.cursorrules` (Manus bridge, repeated four times — same text) says:

1. Watch `docs/inbox/`.
2. If `tasks.json` or `manus_product_spec.md` appears, treat it as the structural instruction before other edits.
3. Apply the change in the repo the task names. Keep 50 Hz resampling and the Butterworth bands.
4. Do not commit inbox-driven code until local pytest, `make verify`, and the nRF Connect schema check pass.
5. Do not edit `contracts/` without the verify scripts.
6. Then run `./docs/inbox/cleanup.sh`.

A useful `tasks.json` has at least:

| Field | Why |
|---|---|
| `spec_version` | So Cursor can reject an old shape |
| `topic` | Must match a token both sides agree (see the mismatch below) |
| `engineering_tasks[].id` | Stable id |
| `engineering_tasks[].target_repo` | One of `cursor_oralable`, `oralable_nrf`, `oralable_swift`, `OralableCore` |
| `engineering_tasks[].scope.implementation_paths` | Files to touch |
| `engineering_tasks[].scope.do_not_modify` | Files and contracts to leave |
| `engineering_tasks[].acceptance_tests` | Commands and pass lines that already exist in-repo |
| `completion_policy` | Say whether git commit, flash, or a release is allowed. Default is no |

`manus_product_spec.md` is still the empty template (topic token pending). Fill it or leave it. Do not leave a second, conflicting spec beside a finished `tasks.json`.

### Inbox contents on 4 Nov 2026

`docs/inbox/tasks.json` holds one task:

| Field | Value |
|---|---|
| Id | `FW-TGM-REV11-GATT-50HZ-20261104` |
| `topic` | `firmware` |
| `target_repo` | `oralable_nrf` |
| `status` | `ready_for_implementation_with_validation_gates` |
| Created | `2026-11-04T09:33:13Z` |
| Asks | Keep the current TGM GATT surface, bound FIFO reads, nominal 50 Hz telemetry, on the REV11 / ES4L15BA1 tree |
| Does not authorize | Flash, OTA, git push, or a silent change to the iOS/Core/Python contract |

That file is large and specific. This audit does not repeat it. Cursor should execute it only under the completion policy inside the task.

### Where analytics scripts belong

Drop new Mac analysis next to the scripts that already do this job:

| Job | Existing script |
|---|---|
| Protocol A capture | `scripts/run_protocol_a_session.py` |
| Gold 50 Hz CSV | `scripts/process_temporalis_gold.py` |
| Clinical text report | `scripts/generate_clinical_report.py` |
| Overnight pack | `scripts/generate_overnight_night_report.py` |
| ANR Dual A align | `scripts/align_anr_oralable_concordance.py` |
| Core ML export | `scripts/generate_mam_model.py`, `scripts/convert_temporalis_mam.py` |
| IR-DC scale check | `scripts/check_ir_dc_scaling.py` |

Outputs stay under `data/validation/` and `plots/`. Do not commit raw nights or model weights unless a human asks. `data/` and `plots/` are gitignored.

Phone code that must match the Mac lives in OralableCore, with parity tests (`Tests/OralableCoreTests/ParityTests.swift` and `cursor_oralable/tests/`). Change the Mac first.

### Topic tokens disagree — fix this before the loop is trusted

Three files describe “which topic is this?” and they do not match.

| File | What it says |
|---|---|
| `cursor_oralable/AGENTS.md` | Slugs: `firmware`, `hardware`, `attachment`, `ios-patient`, `ios-dentist`, `core`, `algorithms`, `overnight`, `clinical`, `research-kit`, `data-room`, `ip`, `regulatory`, `gtm`, `governance`, `brand` |
| `cursor_oralable/docs/WORKSPACE_TOPICS.md` | Rewritten as a Manus registry: `TOPIC-001` `PATENT_MAM_64033978`, `TOPIC-002` `HW_KAGA_PCB00003`, `TOPIC-003` `BIZ_RKAAS_FLEET`, `TOPIC-004` `BIZ_EI_NEW_FRONTIERS`. It says every `tasks.json` must carry one of those tokens |
| Live `docs/inbox/tasks.json` | `"topic": "firmware"` — an `AGENTS.md` slug, not a `TOPIC-00N` token |

Until one file is the registry, Cursor and Manus will validate different tokens. Recommended single rule, if a human confirms it:

- `tasks.json` `topic` uses the `AGENTS.md` slug (`firmware`, `clinical`, …).
- `WORKSPACE_TOPICS.md` either returns to that slug map or gains a column that maps `TOPIC-00N` onto a slug.
- A task with an unknown topic is rejected, not guessed.

`AGENTS.md` still tells agents to read `WORKSPACE_TOPICS.md` for repo roles. That file no longer lists those roles. The role table in this audit and in `workspace.manifest.json` is the one that matches the trees.

### What Manus should not do

- Do not put billing, survey, or pitch text into `tgm_service.c`, `algorithm_spec.yaml`, or OralableCore filters.
- Do not mark a shortcoming in `docs/data_room/clinical/DENTIST_APPROXIMATE.md` closed without the test that file names.
- Do not paste patent claim wording into git.
- Do not call the dentist app or CloudKit share ready. The contract says the professional app is deferred and the CloudKit production schema is not deployed.
- Do not flash, OTA, or push because a task file exists.
- Do not treat `mam-website` or the missing `oralable-web` folder as inside the Cursor workspace.

### Small loop that will work

1. Manus writes `docs/inbox/manus_product_spec.md` (intent, topic slug, repos, constants touched) and `docs/inbox/tasks.json` (paths, do-not-touch, acceptance commands).
2. Cursor implements only `target_repo` and the listed paths.
3. Cursor runs the acceptance commands in the task, plus `make verify` when Python changed, and an nRF Connect log when GATT changed.
4. Durable facts, if any, go into `docs/data_room/` or `VERSION_ALIGNMENT.md` in the same session.
5. `./docs/inbox/cleanup.sh` removes the two inbox files.
6. A human commits. The brief is not the commit.

---

## 5. Drift to leave visible

| Item | Fact |
|---|---|
| Manus block in `.cursorrules` | The same bridge is pasted **four** times. One copy is the rule |
| `WORKSPACE_TOPICS.md` vs `AGENTS.md` vs live `tasks.json` | Three topic schemes. See section 4 |
| `oralable_nrf/README.md` vs `app/VERSION` | README says 1.0.85. `VERSION` file says 1.0.87 |
| `VERSION_ALIGNMENT.md` internal dates | 14 Oct / 29 Oct “does not stream” and 2 Nov “streamed until the cell floor” are both in the file |
| `ORALABLE_SYSTEM_ARCHITECTURE.md` | Stamped 2026-09-19, Gen 1 centred |
| Gen 1 battery in that architecture doc | Says LP260820 ~30 mAh for Gen 2. `VERSION_ALIGNMENT.md` says the sample cell is **LP270829 35 mAh**. Use the version file |
| iOS docs firmware recommend | `oralable_swift/docs/README.md` says recommend **1.0.84**. `VERSION_ALIGNMENT.md` kit string is **1.0.85**, hard min **1.0.63**. Use the version file |
| `oralable-web` | Named in `workspace.manifest.json`. Folder not on disk |
| SDK trees beside the repos | Present, not part of the product workspace |
| Backup sources in firmware | `app/src/*.c.backup` files exist. Do not treat them as the build |

---

## 6. Files this audit used

- `oralable.code-workspace`, `workspace.manifest.json`, `AGENTS.md`, `.cursorrules`
- `pyproject.toml`, `requirements.txt`, `Makefile`, `.github/workflows/python.yml`, `.gitignore`
- `docs/inbox/tasks.json`, `docs/inbox/manus_product_spec.md`, `docs/inbox/cleanup.sh`
- `docs/WORKSPACE_TOPICS.md`, `docs/VERSION`, `docs/data_room/VERSION_ALIGNMENT.md`
- `docs/ORALABLE_SYSTEM_ARCHITECTURE.md`, `src/config/algorithm_spec.yaml`
- `contracts/tgm_gatt_v1.json`, `contracts/oralable_app_services_v1.json`
- `oralable_nrf/west.yml`, `app/VERSION`, `firmware.toolchains.json`, `README.md`
- `OralableCore/Package.swift`, `docs/PACKAGE_BOUNDARIES.md`
- `oralable_swift/.github/workflows/ios.yml`, `Package.resolved`, `FeatureFlags.swift`
- `mam-website/netlify.toml`, `mam-website/README.md`

Not done in this pass: builds, pytest, `swift test`, flashing, or a file-by-file diff of the dirty trees.
