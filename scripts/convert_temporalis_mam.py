#!/usr/bin/env python3
"""Convert a reviewed Temporalis MAM Keras model to Core ML.

A Keras artifact and its JSON provenance record are always required. The former
placeholder MIL model path is intentionally removed: generated stub models must
never be written to the bundled OralableCore production resource location.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _default_output(root: Path) -> Path:
    candidate = (
        root.parent
        / "OralableCore"
        / "Sources"
        / "OralableCore"
        / "Resources"
        / "BruxismMAM_Temporalis.mlpackage"
    )
    return candidate if candidate.parent.is_dir() else root / "BruxismMAM_Temporalis.mlpackage"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_provenance(path: Path, keras_path: Path) -> dict[str, Any]:
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid provenance record {path}: {error}") from error

    artifact = provenance.get("artifact")
    lifecycle = provenance.get("lifecycle")
    validation = provenance.get("validation")
    if not isinstance(artifact, dict) or not isinstance(lifecycle, dict) or not isinstance(validation, dict):
        raise ValueError("provenance requires artifact, lifecycle, and validation objects")
    if artifact.get("keras_sha256") != _sha256_file(keras_path):
        raise ValueError("provenance Keras SHA-256 does not match --keras")
    if lifecycle.get("stub_model") is not False:
        raise ValueError("provenance must explicitly declare stub_model: false")
    return provenance


def _require_release_approval(provenance: dict[str, Any]) -> None:
    lifecycle = provenance["lifecycle"]
    validation = provenance["validation"]
    if lifecycle.get("release_eligible") is not True:
        raise ValueError("production Core ML export requires lifecycle.release_eligible: true")
    if validation.get("strategy") != "user_holdout":
        raise ValueError("production Core ML export requires validation.strategy: user_holdout")
    if not validation.get("metrics"):
        raise ValueError("production Core ML export requires recorded validation metrics")


def _convert_keras(keras_path: Path, out_path: Path) -> None:
    import coremltools as ct

    mlmodel = ct.convert(
        str(keras_path),
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        inputs=[ct.TensorType(shape=(1, 50, 6), name="input")],
    )
    mlmodel.author = "Oralable"
    mlmodel.short_description = "Temporalis MAM (reviewed Keras export)"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))


def _write_coreml_provenance(source: dict[str, Any], out_path: Path) -> Path:
    record = dict(source)
    artifact = dict(record["artifact"])
    artifact.update(
        {
            "coreml_path": str(out_path),
            "coreml_tree_sha256": _sha256_tree(out_path),
            "input_shape": [1, 50, 6],
            "class_order": ["quiet", "phasic", "tonic", "rescue"],
        }
    )
    record["artifact"] = artifact
    target = out_path.with_suffix(".provenance.json")
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a provenance-backed Keras Temporalis MAM model to Core ML")
    parser.add_argument("--keras", required=True, type=Path, help="Reviewed Keras model path (.h5 or SavedModel)")
    parser.add_argument("--provenance", required=True, type=Path, help="Training provenance JSON emitted by generate_mam_model.py")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output .mlpackage directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_path = args.output if args.output else _default_output(root)
    production_path = _default_output(root).resolve()

    if not args.keras.exists():
        print(f"Keras path not found: {args.keras}")
        return 1
    try:
        provenance = _read_provenance(args.provenance, args.keras)
        if out_path.resolve() == production_path:
            _require_release_approval(provenance)
        import coremltools  # noqa: F401
        _convert_keras(args.keras, out_path)
        target = _write_coreml_provenance(provenance, out_path)
    except (ImportError, ValueError, OSError) as error:
        print(f"Core ML export blocked: {error}")
        return 1

    print(f"Converted reviewed Keras model → {out_path}")
    print(f"Wrote Core ML provenance → {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
