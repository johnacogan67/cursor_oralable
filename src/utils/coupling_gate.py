"""
Mac Protocol A seat QC (`couple`).

States (only while BLE + PPG are live; ACC corroborates):
  None / unknown — no PPG (or Protocol A waiting for ACC)
  0 — desk / not on body
  1 — on skin, poor seat / no plausible vitals yet
  2 — on skin + IR pulse + plausible HR and/or SpO2 AC

GATT status `worn` is separate (Protocol A often forces mode 3).
No BLE ⇒ unknown — never map disconnect to couple=0.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence


class CoupleState(IntEnum):
    DESK = 0
    POOR = 1
    GOOD = 2


COUPLE_LABELS = {
    None: "unknown",
    CoupleState.DESK: "desk",
    CoupleState.POOR: "poor",
    CoupleState.GOOD: "ok",
}

# Match FW ir_pulse_* in tgm_service.c
IR_PULSE_DC_MIN = 2000
IR_PULSE_ON_S = 2.5
IR_PULSE_HOLD_S = 20.0
PI_ON = 0.0005  # ac/dc
PI_OFF = 0.0002

# 32-bit temple coupling band (docs/IR_DC_ADC_FORMAT.md)
IR_DC_TEMPLE_MIN_32 = 10_000_000
IR_DC_TEMPLE_MAX_32 = 70_000_000

HR_MIN_BPM = 40.0
HR_MAX_BPM = 180.0
HR_BUFFER_S = 4.0
VITALS_GAP_GRACE_S = 20.0
DC_CONTACT_S = 1.5
ACC_STILL_VAR = 2.5e4  # |g| variance of int16 magnitude; desk stillness
ACC_HIGH_MOTION_VAR = 2.5e6
ACC_WINDOW = 50
PPG_STALL_UNKNOWN_S = 8.0

# Protocol A cue windows where demotion 2→1 should freeze (elapsed from T=0)
ARTIFACT_CUE_WINDOWS: tuple[tuple[float, float], ...] = (
    (120.0, 130.0),  # max tonic
    (180.0, 200.0),  # phasic
    (240.0, 270.0),  # breath-hold + rescue
)


def couple_label(state: CoupleState | None) -> str:
    return COUPLE_LABELS.get(state, "unknown")


def parse_ppg_ir_samples(payload: bytes) -> list[int]:
    """Extract IR samples from a PPG notify (frame_counter + red,ir,green × N)."""
    if len(payload) < 16:
        return []
    out: list[int] = []
    off = 4
    while off + 12 <= len(payload):
        _red, ir, _green = struct.unpack_from("<III", payload, off)
        out.append(int(ir))
        off += 12
    return out


def parse_acc_magnitudes(payload: bytes) -> list[float]:
    """Extract |g| from ACC notify (frame_counter + x,y,z int16 × N)."""
    if len(payload) < 10:
        return []
    out: list[float] = []
    off = 4
    while off + 6 <= len(payload):
        x, y, z = struct.unpack_from("<hhh", payload, off)
        out.append(math.sqrt(float(x * x + y * y + z * z)))
        off += 6
    return out


def _in_artifact_window(elapsed_s: float | None) -> bool:
    if elapsed_s is None:
        return False
    for start, end in ARTIFACT_CUE_WINDOWS:
        if start <= elapsed_s < end:
            return True
    return False


@dataclass
class CouplingGate:
    """Online Mac seat QC from PPG IR + ACC (+ optional temp)."""

    require_acc: bool = True
    allow_ppg_only: bool = False

    _state: CoupleState | None = None
    _ppg_seen: bool = False
    _acc_seen: bool = False
    _last_ppg_mono: float = 0.0
    _last_acc_mono: float = 0.0

    _ir_dc: float = 0.0
    _ir_ac_lp: float = 0.0
    _filt_init: bool = False
    _pulse_latched: bool = False
    _pulse_good_since: float | None = None
    _pulse_bad_since: float | None = None
    _dc_ok_since: float | None = None
    _dc_bad_since: float | None = None

    _acc_mags: list[float] = field(default_factory=list)
    _acc_var: float = 0.0
    _high_motion: bool = False
    _still: bool = False

    _ir_resid_buf: list[tuple[float, float]] = field(default_factory=list)
    _hr_bpm: float | None = None
    _spo2_ac_ok: bool = False
    _vitals_ok_since: float | None = None
    _vitals_bad_since: float | None = None

    _temp_c: float | None = None
    _on_dock: bool = False
    _freeze_demote: bool = False

    @property
    def state(self) -> CoupleState | None:
        return self._state

    @property
    def pulse_latched(self) -> bool:
        return self._pulse_latched

    @property
    def hr_bpm(self) -> float | None:
        return self._hr_bpm

    @property
    def streams_ready(self) -> bool:
        if not self._ppg_seen:
            return False
        if self.require_acc and not self.allow_ppg_only and not self._acc_seen:
            return False
        return True

    def mark_unknown(self) -> None:
        self._state = None
        self._ppg_seen = False
        self._acc_seen = False
        self._filt_init = False
        self._pulse_latched = False
        self._pulse_good_since = None
        self._pulse_bad_since = None
        self._dc_ok_since = None
        self._dc_bad_since = None
        self._ir_resid_buf.clear()
        self._hr_bpm = None
        self._spo2_ac_ok = False
        self._vitals_ok_since = None
        self._vitals_bad_since = None

    def set_on_dock(self, on_dock: bool) -> None:
        self._on_dock = bool(on_dock)

    def set_freeze_demote(self, freeze: bool) -> None:
        self._freeze_demote = bool(freeze)

    def set_session_elapsed(self, elapsed_s: float | None) -> None:
        self._freeze_demote = _in_artifact_window(elapsed_s)

    def feed_temp_c(self, temp_c: float) -> None:
        if math.isfinite(temp_c):
            self._temp_c = float(temp_c)

    def feed_ppg_payload(self, payload: bytes, mono_s: float) -> CoupleState | None:
        irs = parse_ppg_ir_samples(payload)
        if not irs:
            return self._state
        # Also peek red for SpO2 AC proxy from same packet
        reds = self._parse_ppg_channel(payload, channel=0)
        return self.feed_ppg_samples(irs, mono_s, reds=reds)

    @staticmethod
    def _parse_ppg_channel(payload: bytes, channel: int) -> list[int]:
        """channel 0=red, 1=ir, 2=green."""
        if len(payload) < 16:
            return []
        out: list[int] = []
        off = 4
        while off + 12 <= len(payload):
            vals = struct.unpack_from("<III", payload, off)
            out.append(int(vals[channel]))
            off += 12
        return out

    def feed_ppg_samples(
        self,
        ir_samples: Sequence[int],
        mono_s: float,
        reds: Sequence[int] | None = None,
    ) -> CoupleState | None:
        if not ir_samples:
            return self._state
        self._ppg_seen = True
        self._last_ppg_mono = mono_s

        red_list = list(reds) if reds is not None else []
        for i, ir in enumerate(ir_samples):
            red = red_list[i] if i < len(red_list) else None
            self._update_ir_filters(int(ir), mono_s, red=red)

        self._update_hr(mono_s)
        self._recompute_state(mono_s)
        return self._state

    def feed_acc_payload(self, payload: bytes, mono_s: float) -> CoupleState | None:
        mags = parse_acc_magnitudes(payload)
        if not mags:
            return self._state
        return self.feed_acc_magnitudes(mags, mono_s)

    def feed_acc_magnitudes(
        self, magnitudes: Sequence[float], mono_s: float
    ) -> CoupleState | None:
        if not magnitudes:
            return self._state
        self._acc_seen = True
        self._last_acc_mono = mono_s
        self._acc_mags.extend(float(m) for m in magnitudes)
        if len(self._acc_mags) > ACC_WINDOW * 4:
            self._acc_mags = self._acc_mags[-ACC_WINDOW * 4 :]
        window = self._acc_mags[-ACC_WINDOW:]
        if len(window) >= 8:
            mean = sum(window) / len(window)
            self._acc_var = sum((x - mean) ** 2 for x in window) / len(window)
            self._still = self._acc_var < ACC_STILL_VAR
            self._high_motion = self._acc_var > ACC_HIGH_MOTION_VAR
        self._recompute_state(mono_s)
        return self._state

    def tick(self, mono_s: float) -> CoupleState | None:
        """Call from session loop — PPG stall → unknown."""
        if self._ppg_seen and (mono_s - self._last_ppg_mono) >= PPG_STALL_UNKNOWN_S:
            self.mark_unknown()
            return None
        self._recompute_state(mono_s)
        return self._state

    def status_snippet(self) -> str:
        label = couple_label(self._state)
        hr = f"{self._hr_bpm:.0f}" if self._hr_bpm is not None else "-"
        return (
            f"couple={label} pulse={'1' if self._pulse_latched else '0'} "
            f"hr={hr} still={'1' if self._still else '0'}"
        )

    def _dc_in_band(self) -> bool:
        dc = self._ir_dc
        if dc < IR_PULSE_DC_MIN:
            return False
        if dc >= 1_000_000:
            return IR_DC_TEMPLE_MIN_32 <= dc <= IR_DC_TEMPLE_MAX_32
        return True

    def _update_ir_filters(self, ir: int, mono_s: float, red: int | None) -> None:
        if not self._filt_init:
            self._ir_dc = float(ir)
            self._ir_ac_lp = 0.0
            self._filt_init = True
            return

        dc = self._ir_dc
        resid = abs(float(ir) - dc)
        dc += (float(ir) - dc) / 32.0
        if dc < 0:
            dc = 0.0
        self._ir_dc = dc
        self._ir_ac_lp += (resid - self._ir_ac_lp) / 8.0
        self._ir_resid_buf.append((mono_s, float(ir) - dc))
        cutoff = mono_s - HR_BUFFER_S
        self._ir_resid_buf = [(t, v) for t, v in self._ir_resid_buf if t >= cutoff]

        # SpO2 AC proxy: both red and IR have meaningful residual vs DC
        if red is not None and dc > 0:
            red_resid = abs(float(red) - dc)  # rough; red has own DC but OK for gate
            pi_ir = self._ir_ac_lp / max(dc, 1.0)
            self._spo2_ac_ok = pi_ir >= PI_OFF and red_resid > 0

        if not self._dc_in_band():
            self._dc_ok_since = None
            if self._dc_bad_since is None:
                self._dc_bad_since = mono_s
            self._pulse_good_since = None
            if self._pulse_bad_since is None:
                self._pulse_bad_since = mono_s
            if self._pulse_bad_since is not None and (
                mono_s - self._pulse_bad_since
            ) >= IR_PULSE_HOLD_S:
                self._pulse_latched = False
            return

        if self._dc_ok_since is None:
            self._dc_ok_since = mono_s
        self._dc_bad_since = None

        pi = self._ir_ac_lp / max(self._ir_dc, 1.0)
        pulse_on = pi >= PI_ON
        pulse_off = pi < PI_OFF

        if pulse_on:
            self._pulse_bad_since = None
            if self._pulse_good_since is None:
                self._pulse_good_since = mono_s
            if (
                self._pulse_good_since is not None
                and (mono_s - self._pulse_good_since) >= IR_PULSE_ON_S
            ):
                self._pulse_latched = True
        elif pulse_off:
            self._pulse_good_since = None
            if self._pulse_bad_since is None:
                self._pulse_bad_since = mono_s
            if (
                self._pulse_bad_since is not None
                and (mono_s - self._pulse_bad_since) >= IR_PULSE_HOLD_S
            ):
                self._pulse_latched = False

    def _update_hr(self, mono_s: float) -> None:
        buf = self._ir_resid_buf
        if len(buf) < 20:
            self._hr_bpm = None
            return
        # Simple peak pick on positive residual crossings of adaptive threshold
        vals = [v for _, v in buf]
        thr = max(0.0, 0.4 * max(abs(v) for v in vals))
        times = [t for t, _ in buf]
        peaks: list[float] = []
        for i in range(1, len(vals) - 1):
            if vals[i] >= thr and vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
                if not peaks or (times[i] - peaks[-1]) >= 0.30:
                    peaks.append(times[i])
        if len(peaks) < 2:
            self._hr_bpm = None
            return
        intervals = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
        mean_i = sum(intervals) / len(intervals)
        if mean_i <= 0:
            self._hr_bpm = None
            return
        bpm = 60.0 / mean_i
        if HR_MIN_BPM <= bpm <= HR_MAX_BPM:
            self._hr_bpm = bpm
        else:
            self._hr_bpm = None

    def _vitals_plausible(self) -> bool:
        hr_ok = self._hr_bpm is not None and HR_MIN_BPM <= self._hr_bpm <= HR_MAX_BPM
        return hr_ok or self._spo2_ac_ok

    def _recompute_state(self, mono_s: float) -> None:
        if self._on_dock:
            # Charger path — not a couple value; leave unknown for Protocol A policy
            self._state = None
            return

        if not self.streams_ready:
            self._state = None
            return

        if not self._filt_init:
            self._state = None
            return

        dc_contact = (
            self._dc_ok_since is not None
            and (mono_s - self._dc_ok_since) >= DC_CONTACT_S
            and self._dc_in_band()
        )

        if not dc_contact:
            # Desk: flat / out-of-band IR; ACC stillness reinforces but is not required
            lost = (
                self._dc_bad_since is not None
                and (mono_s - self._dc_bad_since) >= min(IR_PULSE_HOLD_S, 3.0)
            )
            if lost or not self._dc_in_band():
                if self._still or not self._acc_seen:
                    self._state = CoupleState.DESK
                else:
                    self._state = CoupleState.DESK
            else:
                self._state = CoupleState.DESK if self._state is None else self._state
            return

        # On-skin optical path
        vitals = self._vitals_plausible()
        if vitals:
            self._vitals_bad_since = None
            if self._vitals_ok_since is None:
                self._vitals_ok_since = mono_s
        else:
            self._vitals_ok_since = None
            if self._vitals_bad_since is None:
                self._vitals_bad_since = mono_s

        want_good = (
            self._pulse_latched
            and vitals
            and not self._high_motion
        )

        if want_good:
            self._state = CoupleState.GOOD
            return

        # Demote from GOOD only after grace (unless artifact freeze)
        if self._state == CoupleState.GOOD:
            if self._freeze_demote:
                return
            vitals_gap = (
                self._vitals_bad_since is not None
                and (mono_s - self._vitals_bad_since) >= VITALS_GAP_GRACE_S
            )
            pulse_lost = not self._pulse_latched
            if vitals_gap or pulse_lost:
                self._state = CoupleState.POOR
            return

        self._state = CoupleState.POOR
