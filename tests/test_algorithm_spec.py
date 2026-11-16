from src.config.algorithm_spec import SPEC, load_algorithm_spec, spec_fingerprint


def test_algorithm_spec_loads_and_has_a_stable_fingerprint() -> None:
    loaded = load_algorithm_spec()

    assert loaded["metadata"]["id"] == "oralable-algorithm-v1"
    assert loaded["metadata"]["version"] == "1.0.0"
    assert len(spec_fingerprint()) == 64


def test_reference_values_match_declared_protocol_a_and_mam_contract() -> None:
    assert SPEC["sampling"]["ppg_hz"] == 50.0
    assert SPEC["filters"]["ppg_bandpass"] == {
        "lowcut_hz": 0.5,
        "highcut_hz": 8.0,
        "order": 4,
    }
    assert SPEC["spo2"]["intercept"] == 110.0
    assert SPEC["spo2"]["r_coefficient"] == 25.0
    assert SPEC["airway_rescue"]["ir_dc_drop_threshold"] == -0.10
    assert SPEC["mam"]["window_samples"] == 50
    assert SPEC["mam"]["stride_samples"] == 25
