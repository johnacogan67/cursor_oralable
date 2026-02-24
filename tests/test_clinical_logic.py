#!/usr/bin/env python3
"""
Unit tests for ClinicalBiometricSuite: SpO2, Airway Rescue, SASHB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.features import (
    ClinicalBiometricSuite,
    compute_filters,
    FS,
    DT_50HZ,
    AIRWAY_RESCUE_WINDOW_SAMPLES,
    SPO2_DIP_THRESHOLD,
)


def _make_df(n: int, red_dc: float, red_ac: float, ir_dc: float, ir_ac: float) -> pd.DataFrame:
    """Create 50 Hz dataframe with deterministic Red/IR signals."""
    t = pd.date_range("2026-02-01 00:00:00", periods=n, freq="20ms")
    # AC components: sinusoidal at ~1 Hz (pulsatile)
    t_s = np.arange(n) * DT_50HZ
    red_ac_sig = red_ac * np.sin(2 * np.pi * 1.0 * t_s)
    ir_ac_sig = ir_ac * np.sin(2 * np.pi * 1.0 * t_s)
    df = pd.DataFrame(
        {
            "red": red_dc + red_ac_sig,
            "ir": ir_dc + ir_ac_sig,
            "green": 1000.0 + 50 * np.sin(2 * np.pi * 1.0 * t_s),
        },
        index=t,
    )
    return df


class TestSpO2:
    """SpO2 empirical curve: R = (Red_AC/Red_DC)/(IR_AC/IR_DC), SpO2 = 110 - 25*R."""

    def test_spo2_requires_red_channel(self):
        """ClinicalBiometricSuite returns NaN spo2 when red_dc/red_ac missing."""
        df = pd.DataFrame(
            {"ir": np.ones(200), "green": np.ones(200), "ir_dc": np.ones(200)},
            index=pd.date_range("2026-02-01", periods=200, freq="20ms"),
        )
        suite = ClinicalBiometricSuite()
        result = suite.process(df)
        assert result["spo2_pct"].isna().all()

    def test_spo2_clamped_60_100(self):
        """SpO2 values must be clamped between 60% and 100%."""
        # R=0 -> SpO2=110, but we clamp to 100
        # R=2 -> SpO2=60
        # R=3 -> SpO2=35, clamp to 60
        n = 500
        t = pd.date_range("2026-02-01", periods=n, freq="20ms")
        # Create signals that yield R values we can control via amplitude ratios
        red_dc, ir_dc = 1000.0, 1000.0
        red_ac, ir_ac = 10.0, 10.0
        df = pd.DataFrame(
            {"red": red_dc + red_ac * np.sin(np.arange(n) * 0.1), "ir": ir_dc + ir_ac * np.sin(np.arange(n) * 0.1), "green": np.ones(n)},
            index=t,
        )
        df = compute_filters(df)
        suite = ClinicalBiometricSuite()
        result = suite.process(df)
        spo2 = result["spo2_pct"].dropna()
        assert (spo2 >= 60).all()
        assert (spo2 <= 100).all()

    def test_spo2_handles_nan(self):
        """SpO2 calculation handles NaN in input signals."""
        n = 300
        t = pd.date_range("2026-02-01", periods=n, freq="20ms")
        red = 1000.0 + 10 * np.sin(np.arange(n) * 0.1).astype(float)
        ir = 1000.0 + 10 * np.sin(np.arange(n) * 0.1 + 0.1).astype(float)
        red[50:60] = np.nan
        ir[100:110] = np.nan
        df = pd.DataFrame({"red": red, "ir": ir, "green": np.ones(n)}, index=t)
        df = compute_filters(df)
        suite = ClinicalBiometricSuite()
        result = suite.process(df)
        assert result["spo2_pct"].notna().any()

    def test_spo2_formula_sanity(self):
        """R ratio and SpO2 calibration produce plausible values."""
        n = 500
        df = _make_df(n, red_dc=1000, red_ac=8, ir_dc=1000, ir_ac=8)
        df = compute_filters(df)
        suite = ClinicalBiometricSuite()
        result = suite.process(df)
        mean_spo2 = result["spo2_pct"].median()
        assert 85 <= mean_spo2 <= 100


class TestAirwayRescue:
    """Airway Rescue: IR DC drop >= 15% within 500 ms."""

    def test_rescue_detects_15_percent_drop(self):
        """Rescue event when IR DC drops by 15% in 500 ms window."""
        n = 100
        w = AIRWAY_RESCUE_WINDOW_SAMPLES
        t = pd.date_range("2026-02-01", periods=n, freq="20ms")
        ir_dc = np.ones(n) * 1000.0
        # At sample w, drop from 1000 to 800 (20% drop)
        ir_dc[w:] = 800.0
        df = pd.DataFrame({"red": np.ones(n), "ir": ir_dc, "green": np.ones(n), "ir_dc": ir_dc}, index=t)
        suite = ClinicalBiometricSuite()
        result = suite.process(df)
        assert result["is_airway_rescue"].iloc[w] == 1

    def test_no_rescue_when_drop_small(self):
        """No rescue when drop is less than 15%."""
        n = 100
        w = AIRWAY_RESCUE_WINDOW_SAMPLES
        t = pd.date_range("2026-02-01", periods=n, freq="20ms")
        ir_dc = np.ones(n) * 1000.0
        ir_dc[w:] = 900.0  # 10% drop
        df = pd.DataFrame({"red": np.ones(n), "ir": ir_dc, "green": np.ones(n), "ir_dc": ir_dc}, index=t)
        suite = ClinicalBiometricSuite()
        result = suite.process(df)
        assert result["is_airway_rescue"].sum() == 0

    def test_rescue_handles_nan(self):
        """Rescue logic skips NaN IR DC without crashing."""
        n = 80
        w = AIRWAY_RESCUE_WINDOW_SAMPLES
        t = pd.date_range("2026-02-01", periods=n, freq="20ms")
        ir_dc = 1000.0 * np.ones(n)
        ir_dc[10:20] = np.nan
        ir_dc[w:] = 800.0
        df = pd.DataFrame({"red": np.ones(n), "ir": ir_dc, "green": np.ones(n), "ir_dc": ir_dc}, index=t)
        suite = ClinicalBiometricSuite()
        result = suite.process(df)
        assert "is_airway_rescue" in result.columns


class TestSASHB:
    """SASHB: cumulative area for SpO2 < 90%."""

    def test_sashb_cumulative_increases_when_dip(self):
        """Cumulative SASHB increases only when SpO2 < 90%."""
        n = 200
        t = pd.date_range("2026-02-01", periods=n, freq="20ms")
        spo2 = np.ones(n) * 95.0
        spo2[50:70] = 85.0
        clinical_df = pd.DataFrame(
            {"spo2_pct": spo2, "is_airway_rescue": np.zeros(n), "cumulative_sashb": np.zeros(n)},
            index=t,
        )
        suite = ClinicalBiometricSuite()
        result = suite._compute_sashb(clinical_df["spo2_pct"])
        cum = result.values
        assert cum[-1] > cum[40]
        assert cum[50] < cum[65]
        # 20 samples at 0.02s each, 5 units per sample below 90 -> 20 * 5 * 0.02 = 2.0
        expected_approx = 20 * (90 - 85) * DT_50HZ
        assert abs(cum[69] - expected_approx) < 0.1

    def test_sashb_zero_when_all_above_threshold(self):
        """Cumulative SASHB stays 0 when SpO2 always >= 90%."""
        n = 100
        t = pd.date_range("2026-02-01", periods=n, freq="20ms")
        spo2 = pd.Series(np.ones(n) * 95.0, index=t)
        suite = ClinicalBiometricSuite()
        result = suite._compute_sashb(spo2)
        assert result.iloc[-1] == 0.0

    def test_sashb_handles_nan(self):
        """SASHB treats NaN SpO2 as no contribution (no increment)."""
        n = 50
        t = pd.date_range("2026-02-01", periods=n, freq="20ms")
        spo2 = pd.Series(np.ones(n) * 85.0, index=t)
        spo2.iloc[20:25] = np.nan
        suite = ClinicalBiometricSuite()
        result = suite._compute_sashb(spo2)
        assert np.isfinite(result).all()


class TestClinicalBiometricSuiteIntegration:
    """Full pipeline integration."""

    def test_process_returns_three_columns(self):
        """process() returns spo2_pct, is_airway_rescue, cumulative_sashb."""
        df = _make_df(400, 1000, 10, 1000, 10)
        df = compute_filters(df)
        suite = ClinicalBiometricSuite()
        result = suite.process(df)
        assert "spo2_pct" in result.columns
        assert "is_airway_rescue" in result.columns
        assert "cumulative_sashb" in result.columns
        assert len(result) == len(df)

    def test_no_red_graceful(self):
        """When red missing, spo2 is NaN but rescue and sashb still run (rescue uses ir_dc)."""
        n = 100
        t = pd.date_range("2026-02-01", periods=n, freq="20ms")
        df = pd.DataFrame({"ir": np.ones(n) * 1000, "green": np.ones(n), "ir_dc": np.ones(n) * 1000}, index=t)
        suite = ClinicalBiometricSuite()
        result = suite.process(df)
        assert result["spo2_pct"].isna().all()
        assert "is_airway_rescue" in result.columns
        assert "cumulative_sashb" in result.columns
        assert result["cumulative_sashb"].iloc[-1] == 0.0  # NaN spo2 -> no SASHB contribution
