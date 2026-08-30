# Oralable agent starters

Copy one block into a **Cursor custom agent** or the **first chat message**. Full detail: [docs/WORKSPACE_TOPICS.md](docs/WORKSPACE_TOPICS.md).

Open the workspace via [oralable.code-workspace](oralable.code-workspace) (all four repos).

---

## Default (no topic)

```
Read cursor_oralable/docs/WORKSPACE_TOPICS.md for repo roles and topic slugs.
Infer write repo from open files; ask if unclear.
Versions: cursor_oralable/docs/data_room/VERSION_ALIGNMENT.md — do not invent FW/app numbers.
```

---

## `firmware`

```
Topic: firmware
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § firmware.
Repos: oralable_nrf only unless I ask to cross topics.
Start: oralable_nrf/docs/DEVELOPMENT.md
Validate every BLE change with nRF Connect (oralable_nrf/.cursor/rules/nrf-connect-validation.mdc).
```

## `hardware`

```
Topic: hardware
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § hardware.
Repos: oralable_nrf + cursor_oralable docs; no app/firmware logic changes unless I ask.
Start: cursor_oralable/docs/PRODUCT_ROADMAP.md §1
Lock facts from schematic/DTS/BOM, not chat memory.
```

## `ios-patient`

```
Topic: ios-patient
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § ios-patient.
Repos: oralable_swift + OralableCore.
Start: oralable_swift/docs/MOBILE_APP_FLOWS.md
Phase 0 temple vitals only; Ed/Pedro = patient app, no dentist app or CloudKit share.
```

## `ios-dentist`

```
Topic: ios-dentist
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § ios-dentist.
Repos: oralable_swift + OralableCore.
Start: oralable_swift/docs/MOBILE_APP_FLOWS.md §5–7
CloudKit / professional app is Phase 1+ unless I say otherwise.
```

## `core`

```
Topic: core
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § core.
Repos: OralableCore only unless I ask to cross topics.
Start: OralableCore/docs/README.md
No FDA / Stage B claims in this package.
```

## `algorithms`

```
Topic: algorithms
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § algorithms.
Repos: cursor_oralable + OralableCore (+ oralable_swift if porting).
Start: cursor_oralable/docs/ALGORITHM_ARCHITECTURE.md
Construct map: cursor_oralable/docs/data_room/clinical/MEASUREMENT_CONSTRUCT_MAP.md (labels vs Dual A / AHI).
Mac Python is the reference; change phone to match Mac. Run parity tests before claiming phone support.
```

## `overnight`

```
Topic: overnight
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § overnight.
Repos: cursor_oralable + oralable_swift + OralableCore.
Start: cursor_oralable/docs/OVERNIGHT_NIGHT_REPORT.md
Construct map: cursor_oralable/docs/data_room/clinical/MEASUREMENT_CONSTRUCT_MAP.md (tonic/phasic/rescue/recovery = labels until F1).
Evaluable overnight ≥6 h worn; do not mix with shorter in-app unlock gates.
```

## `clinical`

```
Topic: clinical
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § clinical.
Repos: cursor_oralable only unless Dual A needs app/firmware.
Start: cursor_oralable/docs/TEMPORALIS_COLLECTION_PROTOCOL.md
Construct map (MAM vs ANR vs AcuPebble vs PSG): cursor_oralable/docs/data_room/clinical/MEASUREMENT_CONSTRUCT_MAP.md — iterate there; do not copy Table 1.
ANR Dual A procedure: edit cursor_oralable/docs/ANR_M40_CONCORDANCE.md (full); data_room clinical bookmark is stub only.
Do not mix Protocol A (5 taps) and Protocol B (3-tap T=0).
```

## `research-kit`

```
Topic: research-kit
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § research-kit.
Repos: cursor_oralable docs + oralable_nrf flash + oralable_swift patient app as needed.
Start: cursor_oralable/docs/data_room/clinical/ORALABLE_RESEARCH_KIT.md
Five kits to Pedro; patient app only; gated on charge-to-temple.
```

## `data-room`

```
Topic: data-room
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § data-room.
Repos: cursor_oralable docs only — no code changes unless I ask.
Start: cursor_oralable/docs/data_room/README.md
Construct map: cursor_oralable/docs/data_room/clinical/MEASUREMENT_CONSTRUCT_MAP.md (MAM / ANR / AcuPebble / PSG — iterate there).
Claim discipline: do not inflate kits shipped, overnight N, AHI equivalence, or FDA/CE status.
```

## `ip`

```
Topic: ip
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § ip.
Repos: cursor_oralable docs only.
Start: cursor_oralable/docs/IP_NORTH_STAR.md
Do not paste patent claim text or provisional specification wording.
```

## `regulatory`

```
Topic: regulatory
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § regulatory.
Repos: cursor_oralable docs (+ app RegulatoryPackageBuilder only if I ask).
Start: cursor_oralable/docs/data_room/governance/REGULATORY_TIMELINE.md
Wellness wording now; Stage B later.
```

## `gtm`

```
Topic: gtm
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § gtm.
Repos: cursor_oralable + oralable_nrf market docs.
Start: cursor_oralable/docs/data_room/governance/GTM_ONE_PAGE.md
Competitor detail: cursor_oralable/docs/data_room/bookmarks/ORALABLE_MARKET_LANDSCAPE.md
```

## `governance`

```
Topic: governance
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § governance.
Repos: cursor_oralable docs only.
Start: cursor_oralable/docs/data_room/governance/CURRENT_GOVERNANCE_STATUS.md
Statutory / legal folders sit outside git.
```

## `brand`

```
Topic: brand
Read cursor_oralable/docs/WORKSPACE_TOPICS.md § brand.
Repos: cursor_oralable docs/figures + data_room.
Start: cursor_oralable/docs/FIGURES.md
Prose: Orwell (.cursor/rules/prose-orwell.mdc). Keep figure IDs locked.
```
