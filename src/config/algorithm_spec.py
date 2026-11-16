"""Runtime loader for the versioned Oralable signal-processing specification.

Python research code consumes ``algorithm_spec.yaml`` directly. Swift keeps an
intentional compile-time mirror; ``scripts/verify_algorithm_spec.py`` checks that
mirror so mobile builds do not need a YAML parser at runtime.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).with_suffix(".yaml")


class AlgorithmSpecError(ValueError):
    """Raised when the checked-in algorithm specification is malformed."""


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise AlgorithmSpecError(f"algorithm spec requires a mapping at {key!r}")
    return value


def _required_number(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlgorithmSpecError(f"algorithm spec requires a numeric value at {key!r}")
    return float(value)


def load_algorithm_spec(path: str | Path = SPEC_PATH) -> dict[str, Any]:
    """Load and validate the documented processing and model-shape parameters."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AlgorithmSpecError(f"unable to read algorithm spec {path}: {error}") from error
    except yaml.YAMLError as error:
        raise AlgorithmSpecError(f"invalid YAML in algorithm spec {path}: {error}") from error

    if not isinstance(raw, dict):
        raise AlgorithmSpecError("algorithm spec root must be a mapping")

    metadata = _required_mapping(raw, "metadata")
    if not isinstance(metadata.get("id"), str) or not metadata["id"]:
        raise AlgorithmSpecError("algorithm spec metadata.id must be a non-empty string")
    if not isinstance(metadata.get("version"), str) or not metadata["version"]:
        raise AlgorithmSpecError("algorithm spec metadata.version must be a non-empty string")

    sampling = _required_mapping(raw, "sampling")
    filters = _required_mapping(raw, "filters")
    ppg_bandpass = _required_mapping(filters, "ppg_bandpass")
    ir_dc_lowpass = _required_mapping(filters, "ir_dc_lowpass")
    temporalis_bandpass = _required_mapping(filters, "temporalis_ac_bandpass")
    beat_detection = _required_mapping(raw, "beat_detection")
    spo2 = _required_mapping(raw, "spo2")
    rescue = _required_mapping(raw, "airway_rescue")
    sashb = _required_mapping(raw, "sashb")
    mam = _required_mapping(raw, "mam")

    ppg_hz = _required_number(sampling, "ppg_hz")
    accel_hz = _required_number(sampling, "accel_hz")
    if ppg_hz <= 0 or accel_hz <= 0:
        raise AlgorithmSpecError("sampling frequencies must be positive")

    for section, keys in (
        (ppg_bandpass, ("lowcut_hz", "highcut_hz", "order")),
        (ir_dc_lowpass, ("cutoff_hz", "order")),
        (temporalis_bandpass, ("lowcut_hz", "highcut_hz", "order")),
        (beat_detection, ("min_distance_seconds", "prominence_factor", "min_bpm", "max_bpm")),
        (spo2, ("intercept", "r_coefficient", "clamp_min", "clamp_max", "window_seconds")),
        (rescue, ("window_ms", "ir_dc_drop_threshold")),
        (sashb, ("spo2_threshold", "dt_s")),
        (mam, ("window_samples", "stride_samples", "activity_gate_shift_percent")),
    ):
        for key in keys:
            _required_number(section, key)

    if ppg_bandpass["lowcut_hz"] >= ppg_bandpass["highcut_hz"]:
        raise AlgorithmSpecError("filters.ppg_bandpass lowcut_hz must be below highcut_hz")
    if temporalis_bandpass["lowcut_hz"] >= temporalis_bandpass["highcut_hz"]:
        raise AlgorithmSpecError("filters.temporalis_ac_bandpass lowcut_hz must be below highcut_hz")
    if spo2["clamp_min"] >= spo2["clamp_max"]:
        raise AlgorithmSpecError("spo2 clamp_min must be below clamp_max")
    if mam["window_samples"] <= 0 or mam["stride_samples"] <= 0:
        raise AlgorithmSpecError("mam window_samples and stride_samples must be positive")

    return raw


def spec_fingerprint(path: str | Path = SPEC_PATH) -> str:
    """Return the SHA-256 fingerprint of the exact spec bytes used by a run."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SPEC = load_algorithm_spec()
