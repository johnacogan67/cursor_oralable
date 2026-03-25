#!/usr/bin/env python3
"""
Convert Temporalis MAM classifier to BruxismMAM_Temporalis.mlpackage (Core ML).

Priority:
  1. If --keras points to a saved Keras model (*.h5, SavedModel dir, etc.), convert via coremltools.
  2. Else build a placeholder MIL program with input shape [1, 50, 6] (1 s @ 50 Hz, six channels).

Typical channel order for inference (must match iOS tensor layout):
  0: Green AC — Butterworth bandpass 0.5–4 Hz (Temporalis research)
  1: IR DC — lowpass 0.8 Hz
  2: Red AC — bandpass 0.5–4 Hz
  3–5: Accelerometer x, y, z (g)

Usage:
  python scripts/convert_temporalis_mam.py [--keras PATH] [--output PATH]

Default output: ../OralableCore/Sources/OralableCore/Resources/BruxismMAM_Temporalis.mlpackage
  (relative to cursor_oralable repo root)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _default_output(root: Path) -> Path:
    cand = root.parent / "OralableCore" / "Sources" / "OralableCore" / "Resources" / "BruxismMAM_Temporalis.mlpackage"
    if cand.parent.is_dir():
        return cand
    return root / "BruxismMAM_Temporalis.mlpackage"


def _mil_stub(out_path: Path) -> None:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    import numpy as np

    np.random.seed(42)
    d_in = 50 * 6
    w1 = np.random.randn(96, d_in).astype(np.float32) * 0.05
    b1 = np.zeros(96, dtype=np.float32)
    w2 = np.random.randn(48, 96).astype(np.float32) * 0.05
    b2 = np.zeros(48, dtype=np.float32)
    w3 = np.random.randn(4, 48).astype(np.float32) * 0.05
    b3 = np.zeros(4, dtype=np.float32)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 50, 6))])
    def prog(input):
        x = mb.reshape(x=input, shape=(1, d_in), name="flatten")
        x = mb.linear(x=x, weight=w1, bias=b1, name="fc1")
        x = mb.relu(x=x, name="relu1")
        x = mb.linear(x=x, weight=w2, bias=b2, name="fc2")
        x = mb.relu(x=x, name="relu2")
        x = mb.linear(x=x, weight=w3, bias=b3, name="fc3")
        x = mb.softmax(x=x, axis=-1, name="probabilities")
        return x

    mlmodel = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16)
    mlmodel.author = "Oralable"
    mlmodel.short_description = "Temporalis MAM: Quiet, Phasic, Tonic, Rescue (@ 50x6)"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))


def _convert_keras(keras_path: Path, out_path: Path) -> None:
    import coremltools as ct

    mlmodel = ct.convert(
        str(keras_path),
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        inputs=[ct.TensorType(shape=(1, 50, 6), name="input")],
    )
    mlmodel.author = "Oralable"
    mlmodel.short_description = "Temporalis MAM (from Keras)"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BruxismMAM_Temporalis.mlpackage")
    parser.add_argument("--keras", default=None, help="Keras model path (.h5 or SavedModel)")
    parser.add_argument("--output", "-o", default=None, help="Output .mlpackage directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_path = Path(args.output) if args.output else _default_output(root)

    keras_path = Path(args.keras) if args.keras else None

    try:
        import coremltools  # noqa: F401
    except ImportError:
        print("Install: pip install coremltools (and tensorflow if using --keras)")
        return 1

    if keras_path is not None:
        if not keras_path.exists():
            print(f"Keras path not found: {keras_path}")
            return 1
        try:
            _convert_keras(keras_path, out_path)
        except Exception as e:
            print(f"Keras conversion failed: {e}")
            return 1
        print(f"Converted Keras → {out_path}")
        return 0

    _mil_stub(out_path)
    print(f"Created stub MIL model → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
