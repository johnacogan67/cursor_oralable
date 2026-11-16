# Python Clean Checkout and Test Workflow

The `cursor_oralable` repository is an installable Python research package named `oralable-research`. The source import namespace remains `src` for compatibility with existing collection scripts.

## Tested environment

- **Python:** 3.13 only for this lock snapshot (`>=3.13,<3.14` in `pyproject.toml`)
- **Lock:** `requirements.lock`, a fully pinned package-resolution snapshot generated from the validated macOS Python 3.13 environment
- **Package:** install this repository in editable mode after installing the lock

The lock is a reproducibility snapshot, not an assertion that every dependency is required by every workflow. Package extras in `pyproject.toml` identify the Bluetooth, reporting, model, and test dependencies used by each domain.

## Fresh clone

```bash
cd cursor_oralable
./scripts/bootstrap_python.sh
```

The bootstrap creates `.venv` if necessary, installs `requirements.lock`, installs the local package with `pip install --no-build-isolation -e .` so it uses the pinned build tooling already in that environment, and runs the standard verification suite. It does not access BLE hardware, train a model, or promote an artifact.

If a particular Python 3.13 executable or virtual environment location is required:

```bash
PYTHON_BIN=python3.13 VENV_DIR="$PWD/.venv" ./scripts/bootstrap_python.sh
```

## Daily commands

```bash
make test          # pytest only
make verify        # install check + spec parity + GATT contract + model provenance + pytest
make verify-fast   # the same checks without pytest
```

The equivalent direct command is:

```bash
.venv/bin/python scripts/verify_research_checkout.py
```

## What verification means

`verify_research_checkout.py` confirms that the editable distribution is installed and that repository metadata exists; then it runs:

1. strict YAML-to-Swift algorithm-spec parity;
2. strict firmware/Python/Core/iOS TGM GATT contract parity;
3. bundled Core ML fingerprint/provenance validation in non-release mode; and
4. all offline pytest tests.

A legacy model may report a warning in normal provenance mode. A release gate remains intentionally closed unless the model is explicitly non-stub, user-holdout validated, and approved; see [MODEL_PROVENANCE.md](./MODEL_PROVENANCE.md).

## CI

`.github/workflows/python.yml` runs the same locked installation and checkout verifier on pushes and pull requests. Any change to dependencies must update both `pyproject.toml` and `requirements.lock`, then pass the clean verification workflow before it is merged.
