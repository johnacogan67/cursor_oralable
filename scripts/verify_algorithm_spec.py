#!/usr/bin/env python3
"""Check the executable Python algorithm specification against the Swift mirror.

Swift deliberately stores compile-time constants so the production package does
not need a YAML parser. This verifier is the compatibility gate between the
runtime YAML reference and that Swift mirror.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.algorithm_spec import SPEC, spec_fingerprint  # noqa: E402

SWIFT_SPEC = ROOT.parent / "OralableCore" / "Sources" / "OralableCore" / "Signal" / "AlgorithmSpec.swift"
PYTHON_FEATURES = ROOT / "src" / "analysis" / "features.py"
PYTHON_TRAINING = ROOT / "scripts" / "generate_mam_model.py"


def value(path: str) -> Any:
    current: Any = SPEC
    for part in path.split("."):
        current = current[part]
    return current


def swift_number(source: str, name: str) -> float | None:
    match = re.search(
        rf"public\s+static\s+let\s+{re.escape(name)}\s*:\s*(?:Double|Int)\s*=\s*(-?[0-9.]+)",
        source,
    )
    return float(match.group(1)) if match else None


def check(report: dict[str, list[str]], label: str, actual: float | None, expected: float) -> None:
    if actual is None:
        report["errors"].append(f"{label}: constant not found")
    elif abs(actual - expected) > 1e-9:
        report["errors"].append(f"{label}: expected {expected}, found {actual}")
    else:
        report["passes"].append(f"{label}: {actual}")


def verify() -> dict[str, list[str]]:
    report: dict[str, list[str]] = {"passes": [], "errors": [], "warnings": []}
    try:
        swift = SWIFT_SPEC.read_text(encoding="utf-8")
        features = PYTHON_FEATURES.read_text(encoding="utf-8")
        training = PYTHON_TRAINING.read_text(encoding="utf-8")
    except OSError as error:
        report["errors"].append(f"required source unavailable: {error}")
        return report

    if "from src.config.algorithm_spec import SPEC" not in features:
        report["errors"].append("Python features do not import the algorithm specification")
    else:
        report["passes"].append("Python features import the algorithm specification")
    if "from src.config.algorithm_spec import SPEC" not in training:
        report["errors"].append("MAM training does not import the algorithm specification")
    else:
        report["passes"].append("MAM training imports the algorithm specification")

    mapping = {
        "ppgSampleRate": "sampling.ppg_hz",
        "accelSampleRate": "sampling.accel_hz",
        "hrBandpassLow": "filters.ppg_bandpass.lowcut_hz",
        "hrBandpassHigh": "filters.ppg_bandpass.highcut_hz",
        "temporalisBandpassHigh": "filters.temporalis_ac_bandpass.highcut_hz",
        "irDCLowpassCutoff": "filters.ir_dc_lowpass.cutoff_hz",
        "filterOrder": "filters.ppg_bandpass.order",
        "minHeartRate": "beat_detection.min_bpm",
        "maxHeartRate": "beat_detection.max_bpm",
        "minPeakDistanceSeconds": "beat_detection.min_distance_seconds",
        "peakProminenceMultiplier": "beat_detection.prominence_factor",
        "spo2Intercept": "spo2.intercept",
        "spo2RatioCoefficient": "spo2.r_coefficient",
        "spo2ClampMin": "spo2.clamp_min",
        "spo2ClampMax": "spo2.clamp_max",
        "spo2WindowSeconds": "spo2.window_seconds",
        "airwayRescueWindowMs": "airway_rescue.window_ms",
        "airwayRescueDropThreshold": "airway_rescue.ir_dc_drop_threshold",
        "sashbSpO2Threshold": "sashb.spo2_threshold",
        "sashbDtSeconds": "sashb.dt_s",
        "mamWindowSamples": "mam.window_samples",
        "mamStrideSamples": "mam.stride_samples",
        "mamActivityGateShiftPercent": "mam.activity_gate_shift_percent",
    }
    for swift_name, spec_path in mapping.items():
        check(report, f"Swift {swift_name}", swift_number(swift, swift_name), float(value(spec_path)))

    report["passes"].append(f"YAML SHA-256: {spec_fingerprint()}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify AlgorithmSpec.swift against algorithm_spec.yaml")
    parser.add_argument("--strict", action="store_true", help="return non-zero when any check fails")
    args = parser.parse_args()
    report = verify()
    print(f"Algorithm specification {SPEC['metadata']['id']} {SPEC['metadata']['version']}")
    for message in report["passes"]:
        print(f"PASS  {message}")
    for message in report["warnings"]:
        print(f"WARN  {message}")
    for message in report["errors"]:
        print(f"ERROR {message}")
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
