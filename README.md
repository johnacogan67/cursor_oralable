# Oralable MAM Signal Processing

**Documentation:** [docs/README.md](docs/README.md) · Product strategy in `oralable_nrf/docs/ORALABLE_MARKET_LANDSCAPE.md`

## Reproducible Python setup

The reference code is an installable package (`oralable-research`) with a tested Python 3.13 lock snapshot. The complete clean-checkout workflow is in [docs/PYTHON_DEVELOPMENT.md](docs/PYTHON_DEVELOPMENT.md).

From a fresh clone:

```bash
./scripts/bootstrap_python.sh
```

Then use the package and run the offline validation suite:

```bash
.venv/bin/python src/parser/log_parser.py "data/raw/text.txt"
make verify
```

## Data Flow
1. Place nrfConnect logs in `data/raw/`.
2. Run `src/parser/log_parser.py` to convert HEX to initial CSV.
3. Run `src/processing/resampler.py` to normalize to 50Hz.

## Researcher Instructions (Temporalis gold chain)

Use the locked Python 3.13 environment (see **Reproducible Python setup** above).

1. **Record** a session following `docs/TEMPORALIS_COLLECTION_PROTOCOL.md` and save the BLE log (e.g. `TEMPORALIS_RAW_01.csv` or under `data/raw/`).

2. **Build the gold validation CSV** (50 Hz, Butterworth filters, automatic Temporalis labels, SpO2 / SASHB):
   ```bash
   python scripts/process_temporalis_gold.py path/to/your_ble_log.csv
   ```
   Output: `data/validation/GOLD_STANDARD_VALIDATION.csv`, plots under `plots/temporalis_validation/`.

3. **Optional — protocol segment table only** (no recording):
   ```bash
   python -m src.analysis.label_generator --temporalis-auto
   ```

4. **Clinical report** (TFI, SASHB, event counts, SpO2–clench timing) for IEEE / patent summaries:
   ```bash
   python scripts/generate_clinical_report.py --input data/validation/GOLD_STANDARD_VALIDATION.csv
   ```
   Console output and `data/validation/clinical_report.txt` (override with `--out`).

**Note:** `src/analysis/features.py` defines `calculate_tfi()` (Temporalis Fatigue Index) used by the report script.
