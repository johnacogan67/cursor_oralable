# Oralable — Ed & Pedro quick start (Phase 0 Vitals)

**One page · 24 Jul 2026 · Heart rate + SpO₂ on temple**

Full test plan: [VITALS_PILOT_TEST_PLAN.md](./VITALS_PILOT_TEST_PLAN.md) · Flash: [FIRMWARE_1.0.70_FLASH.md](./FIRMWARE_1.0.70_FLASH.md) · Hardware: [VITALS_PHASE_GEN1_GEN2.md](../VITALS_PHASE_GEN1_GEN2.md) · Cost/timeline: [COST_AND_TIMELINE.md](./COST_AND_TIMELINE.md)

**Support:** John Cogan

---

## Pilot ship status (as at 24 Jul 2026)

| Item | Status |
|------|--------|
| **Kits with Ed/Pedro** | **Not yet shipped** |
| **Build / flash / app** | Gen1 kits + FW **1.0.70** + patient app **4.3.3** — **ready to hand off** once gate clears |
| **Ship gate** | Charge on **Oralable case** to a **temple-ready SOC** (target **≥50%** remapped gauge) and hold a short worn **HR + SpO₂** session without brownout |
| **Charge status (firmware)** | **1.0.70** STAT blink = charging / on_dock — software status path closed; do **not** say “chrsts broken on REV10” |
| **Still closing** | Energy / case coupling so the cell **actually rises** to the worn-session floor (hardware + validation, not a missing app) |
| **After Phase 0 gates** | Reinstate **muscle / occlusion** metrics for patent embodiment; **professional app** later |
| **Gen2 (parallel)** | Larger cell, re-verified STAT/dock, better SOC / status LED path — removes this class of issue longer term; **not** on these pilot kits |

**One-liner for partners:** *Ready ≠ delivered — two kits ship when charge-to-temple is proven under 1.0.70.*

---

## What you are running

| Item | Version / note |
|------|----------------|
| **Firmware** | **1.0.70** ([flash guide](./FIRMWARE_1.0.70_FLASH.md)) — LTC4124 STAT blink = charging |
| **iOS app** | **Oralable 4.3.3** (TestFlight) — vitals phase; recommends FW **1.0.70**; hard min **1.0.63** |
| **Hardware** | Gen1 · BOM **REV8** · PCB **REV10** · Kaga **ES2832AA2** · Oralable magnetic case (**not Qi / MagSafe**) |

---

## What changed (vs Protocol B / older kits)

| Before | Now (Phase 0 · 1.0.70) |
|--------|-------------------------|
| Cheek + muscle / Protocol B | **Temple** + **HR / SpO₂ only** |
| Fit calibration required | **No user calibration** |
| “chrsts broken” / forced manual only | **STAT blink = on case**; **Automatic OK** on 1.0.70 |
| Green LED on charger | **Red flash** while charging; **solid red** on charge taper |
| MagSafe / Qi pads | **Oralable case + USB-C only** |
| Battery % as fuel gauge | **Rough voltage estimate** (0% ≈ 3.61 V, 100% = 4.35 V) |

**Legacy kits on 1.0.66:** still work with the new app using **manual** placement. Prefer flash to **1.0.70** before Day 1.

---

## What’s in the kit

| Item | Notes |
|------|--------|
| Oralable REV10 clip | Flash **1.0.70** before handoff ([`firmware/oralable_1.0.70_pcb00003_merged.hex`](./firmware/oralable_1.0.70_pcb00003_merged.hex)) |
| **Oralable magnetic charging case** | USB-C — matched LTC6990 TX for this clip |
| iPhone + **Oralable** (patient) app | TestFlight **4.3.3+** (FW **1.0.70** aligned) · vitals · Automatic · Device LED |
| This sheet | Keep with the clip |

**Out of scope:** **Oralable for Dentists**, CloudKit share-to-dentist, practice IAP. Export CSV / session logs from the patient app only.

**Charging:** Clip + case are one BOM (ADI **LTC4124** RX / **LTC6990** TX). **Not WPC Qi.**

**Wellness disclaimer:** Stage A wellness wearable validation — **not** a medical device.

---

## Critical: device placement

**Charge status** = LTC4124 **STAT** pattern (blink = charging; steady = taper/hold). Battery **%** is voltage-based and can read high on the pad — normal chemistry.

**Setup / Settings → Device placement** (applied on each BLE connect):

| Mode | When to use | LED (1.0.70) |
|------|-------------|--------------|
| **Automatic** | Preferred on **1.0.70** + Oralable case | Red flash while charging; solid on taper |
| **On wireless charger** | Force “on case” if Automatic unclear | Same red policy |
| **Off charger (not worn)** | Table / off case | Green flash / solid |
| **Worn on temple** | Vitals session | Status LEDs off; PPG sensing |

**Rules:**

1. Set placement **before** Connect when using manual modes (or leave **Automatic** on 1.0.70).
2. **Never** change placement while connected — disconnect first.
3. Use **Oralable case + USB-C** only — not MagSafe/Qi.

**App mirror:** Dashboard **Device LED** + **Dock / Charging / Taper** chips. Physical LED is dim by design.

---

## Day 1 — Charge on case (3 steps)

1. Seat clip on **Oralable magnetic case** (USB-C) → placement **Automatic** (or **On wireless charger**) → **Connect**. Confirm FW reads **1.0.70** if shown.
2. Confirm **red flash** (or app Charging + flash Device LED). Leave **30–60 min** (phone within ~30 cm).
3. When battery **≥50%**, ready for temple. % may step — watch the **trend**. After long charge, LED may go **solid red** (taper) while still on the case — that is expected, not “broken full at 4.2 V.”

Re-open the app later: it should **auto-reconnect**. One attempt; wait ~20 s.

---

## Day 1+ — Temple vitals (4 steps)

1. **Remove from case** → Settings → **Worn on temple** → **Connect** (or stay Automatic only if you will remount immediately — preferred: set **Worn** explicitly).
2. Mount on **temple**. Dashboard → **On body**, then **Vitals ready** when signal is good (~30 s).
3. Confirm **Heart rate** and **SpO₂** (“Good signal” when stable). Session **≥5 min**.
4. **Share** → export CSV. Do **not** use Protocol B (hidden in vitals phase).

Keep phone within **~30 cm**.

---

## What to record and send back

After each session (~5–10 min on temple):

1. **Share** → **Export nRF-style CSV** (Developer Settings) or session CSV from Share.
2. Email: placement mode, FW string if known, disconnects, phone distance, battery %.
3. Optional: screenshot of vitals card + Device LED row when **Vitals ready**.

Rename: `Oralable_VITALS_Ed_YYYYMMDD.csv` (or `Pedro_…`).

---

## LED quick reference (FW 1.0.70)

| Location | Condition | Device LED | App mirror |
|----------|-----------|------------|------------|
| On Oralable case | STAT blinking (charging) | **Red flash** | Flash red · Charging |
| On Oralable case | STAT steady (taper / hold) | **Solid red** | Solid red · Taper |
| Table / bench | Not full | Green flash | Flash green |
| Table / bench | Truly full (Vmax) | Solid green | Solid green |
| On temple (worn) | Any | PPG red/IR glow (status off) | LED off |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| On case but Dock/Charging off | Confirm **1.0.70**; wait one blink window (~few s); try **On wireless charger** then reconnect |
| Solid red on case before “100%” | Normal **taper** — current falling, not necessarily 4.2 V |
| Battery % jumps on pad | Voltage on wireless dock — watch **30 min trend** |
| “Firmware too old” | Need **≥ 1.0.63** — flash **1.0.70** |
| App warns below recommended | Kit still on **1.0.66** — flash **1.0.70** for Automatic |
| No HR / SpO₂ | **Worn on temple**; adjust pressure; wait 30 s |
| Green on case | Placement still off-dock — set **On wireless charger** or **Automatic** on 1.0.70; reconnect |
| Red on table | Set **Off charger (not worn)** |
| BLE drops after Worn | Charge to **≥50%** first; phone **~30 cm**; RSSI **≥ −70 dBm** |
| MagSafe / Qi used | Stop — use **Oralable case only** |
| No LED (black clip) | Power-cycle; charge 10 min on case; contact John |

---

## Success criteria (Phase 0)

Ed and Pedro each complete:

- [ ] FW **1.0.70** confirmed (`3A0FF006` or app)
- [ ] 1× charger check (red flash → optional solid taper on **Oralable case**)
- [ ] 1× bench LED check (green off pad)
- [ ] 3× temple sessions (≥5 min) with HR and SpO₂ at least once
- [ ] 1× CSV export per session

Protocol B / overnight muscle study **deferred** until vitals stable.

**Sign-off:** [VITALS_PILOT_TEST_PLAN.md § Sign-off](./VITALS_PILOT_TEST_PLAN.md#sign-off)
