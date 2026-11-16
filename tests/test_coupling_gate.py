"""Unit tests for Mac Protocol A CouplingGate."""

from __future__ import annotations

import math
import struct

from src.utils.coupling_gate import (
    CoupleState,
    CouplingGate,
    couple_label,
    parse_acc_magnitudes,
    parse_ppg_ir_samples,
)


def _ppg_payload(irs: list[int], reds: list[int] | None = None) -> bytes:
    reds = reds or [0] * len(irs)
    body = bytearray(struct.pack("<I", 1))
    for r, ir in zip(reds, irs):
        body += struct.pack("<III", r, ir, 0)
    return bytes(body)


def _acc_payload(mags_xyz: list[tuple[int, int, int]]) -> bytes:
    body = bytearray(struct.pack("<I", 1))
    for x, y, z in mags_xyz:
        body += struct.pack("<hhh", x, y, z)
    return bytes(body)


def test_couple_label():
    assert couple_label(None) == "unknown"
    assert couple_label(CoupleState.DESK) == "desk"
    assert couple_label(CoupleState.GOOD) == "ok"


def test_parse_ppg_ir():
    payload = _ppg_payload([10_000_000, 11_000_000])
    assert parse_ppg_ir_samples(payload) == [10_000_000, 11_000_000]


def test_parse_acc_mag():
    payload = _acc_payload([(1000, 0, 0), (0, 1000, 0)])
    mags = parse_acc_magnitudes(payload)
    assert len(mags) == 2
    assert abs(mags[0] - 1000) < 1e-6


def test_unknown_without_samples():
    g = CouplingGate()
    assert g.state is None
    assert not g.streams_ready


def test_unknown_acc_only():
    g = CouplingGate(require_acc=True)
    t = 0.0
    for _ in range(20):
        g.feed_acc_magnitudes([1000.0] * 10, t)
        t += 0.1
    assert g.state is None
    assert not g.streams_ready


def test_desk_flat_ir_still_acc():
    g = CouplingGate(require_acc=True)
    t = 0.0
    # Still ACC
    for _ in range(10):
        g.feed_acc_magnitudes([16000.0] * 10, t)
        t += 0.05
    # Flat / near-zero IR (desk ambient)
    for _ in range(200):
        g.feed_ppg_samples([500] * 5, t)
        t += 0.02
    assert g.streams_ready
    assert g.state == CoupleState.DESK


def test_poor_dc_no_pulse():
    g = CouplingGate(require_acc=True, allow_ppg_only=False)
    t = 0.0
    for _ in range(10):
        g.feed_acc_magnitudes([16000.0] * 5, t)
        t += 0.05
    # Steady DC in temple band, no AC
    for _ in range(300):
        g.feed_ppg_samples([30_000_000] * 5, t)
        t += 0.02
    assert g.state == CoupleState.POOR


def test_good_pulse_and_hr():
    g = CouplingGate(require_acc=True)
    t = 0.0
    for _ in range(10):
        g.feed_acc_magnitudes([16000.0] * 5, t)
        t += 0.05
    # ~60 BPM IR sine on temple DC (50 Hz → 0.02 s)
    fs = 50.0
    for i in range(400):
        phase = 2 * math.pi * 1.0 * (i / fs)  # 60 BPM
        ir = int(30_000_000 + 80_000 * math.sin(phase))
        red = int(28_000_000 + 60_000 * math.sin(phase))
        g.feed_ppg_samples([ir], t, reds=[red])
        t += 1.0 / fs
    assert g.pulse_latched
    assert g.state == CoupleState.GOOD


def test_ppg_stall_marks_unknown():
    g = CouplingGate(require_acc=True)
    t = 1.0
    g.feed_acc_magnitudes([16000.0] * 10, t)
    g.feed_ppg_samples([30_000_000] * 5, t)
    assert g.streams_ready
    g.tick(t + 9.0)
    assert g.state is None


def test_on_dock_forces_unknown():
    g = CouplingGate(require_acc=True, allow_ppg_only=True)
    t = 0.0
    g.set_on_dock(True)
    for _ in range(100):
        g.feed_ppg_samples([30_000_000] * 5, t)
        t += 0.02
    assert g.state is None
