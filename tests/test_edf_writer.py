"""Dual A EDF+ writer — EMG required; research claim discipline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib

from src.export.edf_spec import ANN_EMG_BOUT, CH_EMG, CH_IR, CH_IR_DC, CH_RED, CH_SPO2
from src.export.edf_writer import write_dual_a_edf


def _frame(n: int = 250) -> pd.DataFrame:
    t = np.arange(n, dtype=float) / 50.0
    spo2 = np.full(n, 97.0)
    spo2[100:160] = 86.0  # 1.2 s @ 50 Hz — short; annotations from caller
    return pd.DataFrame(
        {
            "elapsed_s": t,
            "ir_dc": 1e6 + 1000 * np.sin(t),
            "emg_raw": np.linspace(100, 400, n),
            "spo2_pct": spo2,
            "cumulative_sashb": np.cumsum(np.maximum(0, 90 - spo2) * 0.02),
            "label_enum": np.where(t < 2.0, 0, 2),
            "label_name": np.where(t < 2.0, "quiet", "tonic"),
            "red": np.full(n, 2000.0),
            "green": np.full(n, 2100.0),
            "accel_x": np.zeros(n),
        }
    )


def test_write_dual_a_edf_includes_emg(tmp_path: Path):
    df = _frame(300)
    out = tmp_path / "session.edf"
    write_dual_a_edf(
        df,
        out,
        emg_bouts=[(50, 80)],
        spo2_desat_bouts=[(100, 200)],
        session_id="test_dual_a",
        require_emg=True,
    )
    assert out.is_file()
    with pyedflib.EdfReader(str(out)) as r:
        labels = r.getSignalLabels()
        assert CH_EMG in labels
        assert CH_SPO2 in labels
        assert r.getFileDuration() >= 5.0
        anns = r.readAnnotations()
        texts = list(anns[2]) if len(anns) >= 3 else []
        assert any(ANN_EMG_BOUT in t for t in texts)


def test_require_emg_raises_without_column(tmp_path: Path):
    df = _frame(100).drop(columns=["emg_raw"])
    try:
        write_dual_a_edf(df, tmp_path / "x.edf", require_emg=True)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "emg" in str(e).lower()


def test_default_omits_raw_ppg(tmp_path: Path):
    df = _frame(300)
    out = tmp_path / "default.edf"
    write_dual_a_edf(df, out, require_emg=True)
    with pyedflib.EdfReader(str(out)) as r:
        labels = set(r.getSignalLabels())
        ir_dc = r.readSignal(r.getSignalLabels().index(CH_IR_DC))
    assert CH_EMG in labels
    assert CH_SPO2 in labels
    assert CH_RED not in labels
    assert CH_IR not in labels
    # IR_DC fills EDFbrowser ±100 pane (not flat, not barcode)
    assert 40.0 <= float(np.max(np.abs(ir_dc))) <= 100.0


def test_include_raw_ppg_is_viewer_scaled(tmp_path: Path):
    df = _frame(300)
    df["ir"] = 1e5 + 5000 * np.sin(np.arange(300) / 50.0 * 2 * np.pi)
    out = tmp_path / "raw.edf"
    write_dual_a_edf(df, out, require_emg=True, include_raw_ppg=True)
    with pyedflib.EdfReader(str(out)) as r:
        labels = r.getSignalLabels()
        assert CH_IR in labels
        assert CH_RED in labels
        ir = r.readSignal(labels.index(CH_IR))
        spo2 = r.readSignal(labels.index(CH_SPO2))
    assert float(np.max(np.abs(ir))) <= 100.0
    # SpO2 viewer: desat below 90 becomes clearly negative %dev
    assert float(np.min(spo2)) < -20.0
