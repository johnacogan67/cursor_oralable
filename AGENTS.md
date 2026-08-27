# AGENTS.md

## Cursor Cloud specific instructions

### Repository overview

This is a multi-repo workspace for the **Oralable MAM** wearable (PPG + accelerometer on the cheek for sleep bruxism detection). Four repos live under `/agent/repos/`:

| Repo | Language | Buildable on Linux |
|---|---|---|
| `cursor_oralable` | Python 3 | **Yes** — signal processing, BLE log parsing, model training, tests |
| `OralableCore` | Swift (SPM) | **No** — requires Apple `Accelerate` + `CoreML` frameworks |
| `oralable_swift` | Swift (Xcode) | **No** — requires macOS / Xcode |
| `oralable_nrf` | C (Zephyr) | **No** — requires nRF Connect SDK + ARM toolchain |

On a Linux Cloud Agent VM, only `cursor_oralable` can be developed and tested.

### Python environment (`cursor_oralable`)

- **Venv location:** `.venv` inside the repo root.
- **Activate:** `source .venv/bin/activate`
- **Run tests:** `python -m pytest tests/ -v`
  - 1 pre-existing failure: `test_no_rescue_when_drop_small` — the test expects a 10% IR-DC drop to **not** trigger airway rescue, but the code threshold is `<= -10%` (inclusive). This is a known test/code mismatch, not an environment issue.
- **No linter is configured.** Use `python -m py_compile <file>` for syntax checks.
- **Key scripts** (run from repo root with venv active):
  - `python src/parser/log_parser.py <log_file>` — parse BLE hex logs
  - `python src/analysis/features.py [session_csv]` — extract beat features + clinical metrics
  - `python scripts/check_ir_dc_scaling.py <log_file>` — IR-DC scaling diagnostic
- **Data files** in the repo root: `TEMPORALIS_GOLD_STANDARD.csv`, `TEMPORALIS_RAW_01.csv` — raw BLE logs for testing the pipeline.
- The full pipeline flow is: parse BLE log → merge PPG + accel → resample to 50Hz → apply Butterworth filters → clinical biometrics (SpO2, airway rescue, SASHB) → beat detection → feature extraction.
