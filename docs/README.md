# Oralable research & validation documentation index

Python algorithm development, clinical protocols, and gold-standard validation (`cursor_oralable`). **Doc pack:** `docs/VERSION` → **1.2.0** (aligned with `oralable_nrf` landscape v1.2).

**Product strategy:** [oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md](../../oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md) (competitors, regulatory, GTM, appendices A/B).

## Algorithms & signal processing

| Document | Description |
|----------|-------------|
| [ALGORITHM_ARCHITECTURE.md](./ALGORITHM_ARCHITECTURE.md) | Python ↔ Swift algorithm split, OralableCore integration |
| [IR_DC_ADC_FORMAT.md](./IR_DC_ADC_FORMAT.md) | Cheek IR-DC range (10M–70M), MAXM86161 ADC, `R_G_IR` order |
| [../README.md](../README.md) | Setup, `process_temporalis_gold.py`, clinical report scripts |

## Collection & protocol

| Document | Description |
|----------|-------------|
| [TEMPORALIS_COLLECTION_PROTOCOL.md](./TEMPORALIS_COLLECTION_PROTOCOL.md) | Sync-tap phases, 50 Hz, **T=0 = 1st 3-tap sync** anchoring |

## Clinical evaluation

| Document | Description |
|----------|-------------|
| [CLINICAL_VALIDATION.md](./CLINICAL_VALIDATION.md) | Oralable_7, Ed Owens package, self-validation gap analysis |

## Engineering (internal)

| Document | Description |
|----------|-------------|
| [internal/CLAUDE_IOS_REFACTOR_INSTRUCTIONS.md](./internal/CLAUDE_IOS_REFACTOR_INSTRUCTIONS.md) | iOS + OralableCore refactor guidance |
| [upload/ORALABLE_COMBINED.md](./upload/ORALABLE_COMBINED.md) | Legacy combined pack (see README hubs first) |

## Cross-repo

| Repo | Role |
|------|------|
| **oralable_nrf** | Firmware, GATT, OTA — `docs/README.md` |
| **oralable_swift** | Consumer + dentist apps, CloudKit |
| **OralableCore** | Shared BLE parsing, TFI/SASHB handshake export |

## Standard pipeline

1. Record per [TEMPORALIS_COLLECTION_PROTOCOL.md](./TEMPORALIS_COLLECTION_PROTOCOL.md)
2. `python scripts/process_temporalis_gold.py <ble_log.csv>`
3. `python scripts/generate_clinical_report.py --input data/validation/GOLD_STANDARD_VALIDATION.csv`
4. Optional: `oralable_nrf/scripts/check_ir_dc_scaling.py` on new logs

*Last updated: June 2026*
