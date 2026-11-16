#!/usr/bin/env python3
"""Verify a bundled Core ML artifact has an intact, auditable provenance record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT.parent / "OralableCore" / "Sources" / "OralableCore" / "Resources" / "BruxismMAM_Temporalis.mlpackage"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_hash(child).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Oralable Core ML provenance")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--release", action="store_true", help="require a release-eligible non-stub artifact")
    args = parser.parse_args()

    model = args.model
    provenance_path = model.with_suffix(".provenance.json")
    errors: list[str] = []
    warnings: list[str] = []
    if not model.is_dir():
        errors.append(f"model package missing: {model}")
    if not provenance_path.is_file():
        errors.append(f"provenance missing: {provenance_path}")
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1

    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        print(f"ERROR invalid JSON: {error}")
        return 1
    lifecycle = provenance.get("lifecycle", {})
    artifact = provenance.get("artifact", {})
    observed_hash = tree_hash(model)
    if artifact.get("coreml_tree_sha256") != observed_hash:
        errors.append("Core ML package fingerprint does not match provenance")
    else:
        print(f"PASS  Core ML package fingerprint: {observed_hash}")
    if lifecycle.get("stub_model") is not False:
        warnings.append("artifact is not affirmatively recorded as non-stub")
    if lifecycle.get("release_eligible") is not True:
        warnings.append(f"release gate is closed (status: {lifecycle.get('status', 'unknown')})")
    if args.release:
        if lifecycle.get("stub_model") is not False:
            errors.append("release requires lifecycle.stub_model: false")
        if lifecycle.get("release_eligible") is not True:
            errors.append("release requires lifecycle.release_eligible: true")
        validation = provenance.get("validation", {})
        if validation.get("strategy") != "user_holdout":
            errors.append("release requires user-holdout validation")
    for warning in warnings:
        print(f"WARN  {warning}")
    for error in errors:
        print(f"ERROR {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
