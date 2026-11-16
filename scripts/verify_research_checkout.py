#!/usr/bin/env python3
"""Verify a cleanly installed Oralable research checkout without Bluetooth hardware."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, *command: str) -> bool:
    print(f"==> {label}")
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode:
        print(f"ERROR {label} failed with exit code {completed.returncode}")
        return False
    print(f"PASS  {label}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="Run metadata and contract checks without pytest")
    args = parser.parse_args()

    errors: list[str] = []
    for relative in ("pyproject.toml", "requirements.lock", "src/config/algorithm_spec.yaml", "contracts/tgm_gatt_v1.json"):
        if not (ROOT / relative).is_file():
            errors.append(f"missing required repository file: {relative}")
    try:
        installed = importlib.metadata.version("oralable-research")
    except importlib.metadata.PackageNotFoundError:
        errors.append("oralable-research is not installed; run scripts/bootstrap_python.sh or pip install -e .")
    else:
        print(f"PASS  installed oralable-research {installed}")

    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1

    checks = [
        ("algorithm specification parity", sys.executable, "scripts/verify_algorithm_spec.py", "--strict"),
        ("TGM GATT contract", sys.executable, "scripts/verify_tgm_contract.py", "--strict"),
        ("bundled model provenance (non-release)", sys.executable, "scripts/verify_model_provenance.py"),
    ]
    if not args.skip_tests:
        checks.append(("pytest", sys.executable, "-m", "pytest", "-q"))

    passed = all(run(label, *command) for label, *command in checks)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
