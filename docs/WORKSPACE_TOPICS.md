# Oralable workspace topics and handoff routing

**Integration status:** local file-based handoff controls installed; automatic dispatch is not enabled. A queued request is not evidence of implementation or hardware readiness.

Use [MANUS_CURSOR_HANDOFF.md](MANUS_CURSOR_HANDOFF.md) for approval, ownership, baselines, receipts and archival. Product priorities/timeline remain in [PRODUCT_ROADMAP.md](PRODUCT_ROADMAP.md); technical ownership remains in `../workspace.manifest.json`; partner-facing readiness/version wording remains in `data_room/VERSION_ALIGNMENT.md`. This page is a routing registry, not another roadmap or release record.

## Canonical topic slugs

The request envelope's `topic` is exactly one of the following slugs. [AGENTS.md](../AGENTS.md) provides the corresponding starters and start-here documents. A slug names the subject; the approved `target_repo` and exact path allowlist determine write authority.

- `firmware`: sensor, BLE, board-target and firmware implementation in `oralable_nrf`.
- `hardware`: hardware evidence and documentation; board/firmware changes require their own approved scope.
- `attachment`: mechanical attachment/housing documentation in the research/docs hub.
- `ios-patient`: patient application work in `oralable_swift`; Core changes require a separate approved task.
- `ios-dentist`: professional application work in `oralable_swift`, subject to current product-phase gates.
- `core`: shared package work in `OralableCore`.
- `algorithms`: Python-reference work in `cursor_oralable`; consumer changes need their own approved scope.
- `overnight`: overnight analysis/reporting; name one approved write repository per task.
- `clinical`: evidence/protocol documentation in the hub; do not infer clinical equivalence or clearance.
- `research-kit`: kit coordination with independently gated firmware/app/device tasks.
- `data-room`: reviewed data-room documents, not blanket permission over all data-room assets.
- `ip`: IP planning documents, with existing confidentiality restrictions.
- `regulatory`: reviewed regulatory documents; no clearance/status promotion without evidence and authorization.
- `gtm`: go-to-market research and approved business documents.
- `governance`: approved governance documents; legal/statutory material remains outside engineering context.
- `brand`: approved prose/figure changes, preserving locked identifiers.

Cross-repository work requires separately approved task scopes. Do not interpret two slugs, a legacy token or a subject description as permission to write every listed component. The unconditional handoff rule applies even when topic-specific rules do not load.

## Legacy business aliases

Retain these aliases for identifying older planning conversations, not as executable technical contracts. New envelopes use canonical slugs. If an alias appears in source material, resolve it to the subject below and then use explicit task scope; the validator does not accept aliases as the `topic` value.

`[TOPIC-001]` / `PATENT_MAM_64033978` refers to the `algorithms` subject. `[TOPIC-002]` / `HW_KAGA_PCB00003` refers to `firmware` bring-up. `[TOPIC-003]` / `BIZ_RKAAS_FLEET` refers to `core` export/billing integration; iOS work requires a separate scope. `[TOPIC-004]` / `BIZ_EI_NEW_FRONTIERS` refers to `gtm` survey/business validation.

### Historical topic notes — not current execution authority

The previous registry recorded patent-related infrared/DC research, 50 Hz reference processing and filtering; Kaga REV11 bring-up on a named NCS revision; fleet lease/report pricing proposals; and New Frontiers survey economics. These are historical planning inputs. Resolve the active board/SDK, wire channel order, sampling policy, algorithm reference and permitted claims from current owner-controlled contracts and source evidence. Do not enforce a historical NCS pin, price, cheek configuration or patent interpretation simply because it appeared in an earlier topic note.

The previous milestone tracker listed Patent Code-Lock as in progress and TGM GATT Stable, RKaaS Log Audit and Practice Survey Hook as awaiting work. Those labels were not independently verified during handoff setup and are not promoted here. In particular, a successful firmware build is not the complete device-to-app validation or hardware-release gate. Use the canonical roadmap and reviewed evidence records for current status.

## Usage protocol

Manus drafts a request in `docs/inbox/tasks.json` with a canonical slug, stable task ID, exact implementation paths, explicit prohibited actions, linked canonical requirements and a recorded scoped baseline. The request remains `awaiting_approval` until the user approves the bound fingerprint. Never auto-approve from an inbox change or from this topic registry.

Cursor confirms actual authorization, validates the approved request and live baseline, and works only inside the accepted scope. It returns a separate retained receipt in `docs/outbox/<task-id>.json`. An owner reviews results before Manus proposes changes to canonical business/status documents. Archival copies reviewed records and retains the live originals; no watcher, scheduler or chat synchronization is implied.
