"""Unit tests for SpO2 ∩ ANR EMG nest helpers."""

from __future__ import annotations

import numpy as np

from src.analysis.emg_spo2_nest import (
    count_overlaps,
    desat_bouts,
    nest_markdown,
    nest_metrics,
)


def test_desat_bouts_min_duration():
    fs = 50.0
    n = int(60 * fs)
    spo2 = np.full(n, 97.0)
    # 5 s dip — too short
    spo2[int(10 * fs) : int(15 * fs)] = 85.0
    # 12 s dip — counts
    spo2[int(30 * fs) : int(42 * fs)] = 88.0
    bouts = desat_bouts(spo2, fs=fs, min_duration_s=10.0)
    assert len(bouts) == 1
    assert bouts[0][0] == int(30 * fs)


def test_emg_overlap_increments_with_desat():
    fs = 50.0
    n = int(120 * fs)
    spo2 = np.full(n, 97.0)
    # Desat 20–40 s
    spo2[int(20 * fs) : int(40 * fs)] = 86.0
    emg_bouts = [
        (int(5 * fs), int(8 * fs)),  # no overlap
        (int(25 * fs), int(28 * fs)),  # overlaps desat
    ]
    m = nest_metrics(spo2, emg_bouts, sashb_total=1.2, fs=fs, overlap_tol_s=5.0)
    assert m["desat_event_count"] == 1
    assert m["emg_bout_count"] == 2
    assert m["emg_bouts_with_desat"] == 1
    assert m["desat_events_with_emg"] == 1
    assert abs(m["frac_emg_bouts_with_desat"] - 0.5) < 1e-9
    assert "not AHI" in m["note"].lower() or "not claimed ODI" in m["note"]
    assert "Bruxoff" in m["note"]


def test_overlap_tol_connects_near_miss():
    # EMG ends at sample 100; desat starts at 120; tol=30 → overlap
    left_hit, right_hit = count_overlaps([(50, 100)], [(120, 200)], tol_samples=30)
    assert left_hit == 1
    assert right_hit == 1
    left_hit2, _ = count_overlaps([(50, 100)], [(120, 200)], tol_samples=5)
    assert left_hit2 == 0


def test_nest_markdown_claim_discipline():
    m = nest_metrics(np.full(500, 96.0), [], sashb_total=0.0)
    body = nest_markdown(m, "test_session")
    assert "not AHI" in body.lower() or "Not** AHI" in body or "Not AHI" in body or "AHI/ODI" in body
    assert "Bruxoff" in body
    assert "AcuPebble" in body
    assert m["spo2_qc"] == "ok"
    assert "spo2_qc" in body


def test_spo2_qc_warn_on_low_mean():
    n = 500
    spo2 = np.full(n, 89.0)
    red = np.full(n, 100.0)  # AC present
    ir = np.full(n, 120.0)
    m = nest_metrics(spo2, [], red_ac=red, ir_ac=ir)
    assert m["spo2_qc"] == "warn"
    assert m["spo2_finite_frac"] == 1.0
    assert m["spo2_frac_below_90"] == 1.0


def test_spo2_qc_fail_on_flat_ac():
    n = 500
    spo2 = np.full(n, 98.0)
    m = nest_metrics(spo2, [], red_ac=np.zeros(n), ir_ac=np.zeros(n))
    assert m["spo2_qc"] == "fail"
