# Workspace release baseline

This workspace is four Git repositories opened together by
[`oralable.code-workspace`](../oralable.code-workspace):

- `cursor_oralable` — Python reference pipeline and documentation hub
- `oralable_nrf` — firmware and custom boards
- `OralableCore` — shared Swift package
- `oralable_swift` — patient and professional iOS apps

The nearby public sites (`oralable-web`, `mam-website`) are not firmware or app
build dependencies. Include them only when a public release changes their copy,
forms, or deployment configuration.

## Why this exists

A release needs more than one repository at a particular commit. It needs a
reviewed set of source changes, compatible revisions of each layer, a defined
BLE contract, and evidence from the build/test and device-to-app path.

`workspace.manifest.json` names the repositories, release inputs, generated
patterns, and contract owners. `scripts/workspace_baseline.py` reads that
manifest and reports local Git state without changing files.

## Run the checker

From `cursor_oralable`:

```bash
python3 scripts/workspace_baseline.py
```

Write a Markdown record:

```bash
python3 scripts/workspace_baseline.py \
  --output ../outputs/oralable-release-baseline.md
```

Write a machine-readable record, including the two public sites:

```bash
python3 scripts/workspace_baseline.py \
  --include-external \
  --format json \
  --output ../outputs/oralable-release-baseline.json
```

For CI or a pre-tag check, use `--strict`. It exits with status 2 when tracked
changes or untracked paths outside the generated-pattern list remain:

```bash
python3 scripts/workspace_baseline.py --strict
```

## How to use the report

1. Review every **tracked change**.
2. Review every **reviewable untracked** path. Commit it, deliberately remove it,
   or document why it belongs outside Git.
3. Do not treat **generated/local untracked** paths as release evidence. Build
   folders, caches, virtual environments, DerivedData, and local research output
   must not substitute for source, manifests, tests, or signed artifacts.
4. Record the four canonical Git revisions together before tagging a compatible
   firmware/Core/app/reference-algorithm baseline.
5. Attach build/test evidence and a device-to-app BLE validation artifact before
   claiming a release baseline.

## Contract ownership

| Contract | Owner | Consumers |
|---|---|---|
| TGM GATT UUIDs, payloads, endianness, status semantics | `oralable_nrf` | Python, Core, iOS |
| 50 Hz reference algorithms and model-training provenance | `cursor_oralable` | Core, iOS |
| Shared Swift parsers, domain APIs, Core ML package, exports | `OralableCore` | iOS |
| CoreBluetooth lifecycle, platform services, storage, UX | `oralable_swift` | — |

The checker is intentionally non-destructive. It does not commit, push, delete,
change ignore rules, build, test, flash, run OTA, or contact external services.
