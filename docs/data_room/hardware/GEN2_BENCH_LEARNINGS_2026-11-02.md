# Gen 2 bench — what we learned, and the path to the gate

**As at:** 11 Nov 2026  
**Hardware:** Gen 2 · BOM **REV9** · PCB **REV11** · Kaga **ES4L15BA1** (nRF54L15) · cell **LP270829**  
**Program:** [GEN2_PROGRAM_GATE.md](../../../../oralable_nrf/docs/GEN2_PROGRAM_GATE.md) · `oralable_nrf/gen2.program.json` (every gate still **pending**)  
**Earlier day:** [GEN2_WOUT_KAGA_STATUS_2026-10-30.md](./GEN2_WOUT_KAGA_STATUS_2026-10-30.md) is the 30 Oct record. This note is the one to follow.

**One-liner:** The radio, the accelerometer, and the PPG chip work on the REV11 pins. The 8 Nov image keeps red, IR, and green in stable fields and held a 14-minute stream off the charger. The capture check still fails. No gate is closed. Kits stay gated.

---

## Claim discipline

| Do say | Do not say |
|--------|------------|
| One board streamed PPG and accelerometer on 2 Nov until the cell fell under 5% | G2-P3 passed, or the stream matches Gen 1 |
| The 8 Nov image reads `2.0.0-gen2-nrfconnect`. Its hash is `55948a6e…`. The tree that built it was dirty | That string is a kit release, or kits shipped |
| A 14-minute off-charger stream scored near 50 Hz. The check failed on skipped frame counters | Charge-to-temple done, or G2-P4 closed |
| Charger pin on, cell about 3.4 V, on the 2 Nov logs | Charge-to-temple done |
| Gates in `gen2.program.json` are **pending** | A source build closed a gate |

---

## What the bench showed

### Pins, program, build

The DTS map is the one that answered. Do not use hardware doc §5.

| Net | GPIO | Config |
|-----|------|--------|
| SDA | P1.07 | TWIM22, pull-up. Wout, 25 Oct |
| SCL | P1.03 | TWIM22, pull-up. NFC2. Wout, 25 Oct |
| INT_ACC | P0.03 | Active high. LIS2DTW12 INT1 |
| INT_OPT | P1.06 | Active low, pull-up |
| SENS_EN | P1.08 | Active high. Wout, 25 Oct |
| CHRSTS | P1.02 | Active low, pull-up. NFC1 |
| BATEN | P1.04 | Active high |
| BATVOL | P1.05 | SAADC AIN1, gain 1, 12-bit |

NCS 2.9.3 applies `nfct-pins-as-gpios` only on the CMSE path. This image is not built that way, so `main` clears the NFCT pad config before I2C. The first board’s RTT read LIS2DTW12 WHO_AM_I `0x44` and MAXM86161 part id `0x36`. Later boards started both sensors even when the BLE pin-scan replay printed `lis44=0 maxid=0`. That scan is not the pin map. Do not copy its hits into the board file.

Program with J-Link **1057765387**, device `NRF54L15_M33`, speed 500. A connect that works prints SW-DP `0x6BA02477`. `VTref=3.300V` is the DK’s own reference. Do not run `nrfutil device recover`. Do not pass serial `1050090445`. One power source: do not feed the MAM from the DK while it is on the charger or the cell. Unplug Tag-Connect after reset. A debugger halt drops BATEN. P19 pin 3 (`SWD0_SELECT`) must be grounded by the 10-pin lead.

Build Gen 2 only with NCS 2.9.3 at `/Users/johnacogan67/ncs/v2.9.3` and toolchain `/opt/nordic/ncs/toolchains/b8efef2ad5`. `west.yml` stays on sdk-nrf v2.7.0 for Gen 1. Do not west-build the top of `$HOME/ncs/build_pcb00003_gen2_ota`. Rebuild the app with ninja, then `mergehex.py`. Do not flash Gen 1 **1.0.85** onto REV11. Do not overwrite `artifacts/oralable_gen2_ble.hex`.

### Why PPG used to stop

On the older image the phone lost the link about 7 s after PPG notify. Two faults were in that drop.

Red and IR, even dim, load a cell near 3.4 V. A green-only image still died, and the cell ADC moved only about 30 mV under green PA 5. The 3.3 V rail has not been metered. Firmware cannot see the regulator.

The link watchdog treated a quiet PPG notify as a dead radio and logged `fw: ble recover connected idle/stale stream` while the firmware log was still arriving. A firmware-log notify in the last 4 s now holds the link.

After that fix the link stayed up and PPG still stopped, after 8 frames, with `fw: ppg quiet err=0 empty=0`. The accelerometer kept running, so the I2C bus and the work queue were alive. An empty FIFO would have logged `empty=1`. A failed read would have logged a non-zero `err`. The 500 ms poll had been cancelled. `ppg_stop_streaming()` does that and leaves the chip marked running, so `ppg_ensure_awake()` does not turn the poll or the FIFO interrupt back on. The caller was the battery-LED path (`status-leds`), which runs once a PPG notify is more than 2 s old.

### What the image on the board does

While PPG notify is on, that call logs `fw: ppg poll keep status-leds` and leaves the poll running. In the 10:03–10:12 capture (`Oralable__9_.csv`, same connection as `__8_`) the link never dropped. PPG frame counters run 0 through 1116 with no gap. `ppg_err` stays 0. The accelerometer runs beside it (frame 1072, 12 counters missing). `status-leds` tries to stop the poll the whole time the stream is up, and the image refuses.

Samples are not a stuck word. Red and green sit near 12,100–12,700. IR moves from about 12,650 to about 15,500. `fw: ir_pulse worn=1` was logged. The AC number on that line is not a pulse size. This was green PA 5 only. Under 3900 mV the firmware holds green and does not apply red or IR.

### What stopped the stream

The soft floor is the Gen 1 gauge copied onto this board: empty 3300 mV, full 4200 mV, sensors off below 5% or below 3300 mV. 3344 mV is 4% and blocks. 3345 mV is 5% and passes. The MCU and the radio stay up.

In that capture the charger pin stayed on (`ch=1`) and the cell still fell from 3410 mV to 3322 mV. The floor stopped both sensors at 10:08:32 (3344 mV). They started again at 10:09:34 and ran until 10:10:37 (3333 mV), then stayed off through 10:12. After the floor the log shows `fw: ppg poll off set` and then `fw: ppg quiet err=0 empty=0`. That is the floor turning the reader off.

A charger pin reading on is not a charged cell. On the current gauge, 50% is 3750 mV. These logs never reached it. Every cell in them was about 3.32–3.55 V.

## What that means

Further pin scans will not move the gate. The streaming fault is no longer a missing chip, a wrong pin, or a dead radio.

G2-P4 is the next measurement. The 3300/4200 window and the 5% floor are Gen 1 CG-320B numbers. The floor did what the code says. Whether 3300 mV is empty for the LP270829 is unknown. Charge-to-temple, the ship gate, is a temple-ready state of charge on the Oralable charger (target at least 50%) and a short worn heart-rate and SpO2 session without brownout. SpO2 has not been run, because red and IR stay off under 3900 mV.

G2-P3 stays pending. The 2 Nov capture was one board, green only, a few minutes, and it has no Gen 1 comparison. The 8 Nov file is the longer stream. Stage C scores it. The accelerometer rate for that image is 50 Hz.

The keep-guard and the quiet-link change were in the **1.0.87** image. A later image is on the board. Stage C has its hash. The Stage A table below is the 3 Nov hex, not that image.

---

## Path to the gate

Do these in order. A low cell will fail a longer stream the same way the 10:12 capture did. Do not mark `gen2.program.json` passed from this note.

8 Nov 2026: Stage B no longer asks for a voltmeter on the cell or on the 3.3 V rail. Firmware `mv` is the voltage record.

9 Nov 2026: G2-P0 and G2-P1 were removed from the program list. Firmware `mv` remains the voltage record.

11 Nov 2026: G2-P3 no longer requires an approved Gen 1 capture beside the Gen 2 log.

| Stage | Toward | Work | Done when |
|-------|--------|------|-----------|
| A. Freeze a candidate | G2-P2 version read | Pin scan out. Poll-keep stays. Gen 2 string from `VERSION.gen2`. `app/VERSION` stays 1.0.87. Build with NCS 2.9.3. Record the git revision and the hex hash. | GATT `006` on a board reads `2.0.0-gen2-nrfconnect`. The manifest stays pending. |
| B. Charge, then stream | G2-P4 | One board, one power source. On the Oralable charger until firmware `mv` is at least 3750, and preferably near 4100. Log `ch=` and `mv` on the charger and after it comes off. Then turn on the log, the accelerometer, and PPG. | Ten minutes of PPG and accelerometer with no `fw: floor sensors off`. This does not close charge-to-temple. |
| C. Stream evidence | G2-P3 | Write down the accelerometer rate (50 Hz or 100 Hz) before calling parity. Capture log, accelerometer, then PPG. Run `verify_gen2_stream_capture.py`. Account for drops and errors in that capture. Keep the raw file out of git and store a hash. | The check passes and a reviewer can find the raw file. |
| D. Air contract | G2-P2 | On the Stage A image, read device id and firmware version, disconnect, and reconnect. | One log shows the version string, the TGM inventory, and a second connection. |
| E. Phone, then kits | G2-P6, then the ship gate | Follow `oralable_swift/docs/GEN2_COMPATIBILITY_GATE.md`. The app must not claim Gen 2 production compatibility until that is authorized. After the required gates have evidence and a reviewer, charge each unit to at least 50% and hold a short worn heart-rate and SpO2 session without brownout. | Five units with GATT `006` at 2.0.x can be packed. Until then the kits stay gated. No new date. |

The air-contract log and the phone soak use the candidate image. Red and IR stay off under 3900 mV.

## Stage C — checked 8 Nov 2026, not passed

The accelerometer rate for this image is **50 Hz**. `OralableCore` and `src/parser/log_parser.py` now stamp those samples at 20 ms. The iOS `AlgorithmSpec` declares the same rate. This note does not call parity.

The 14-minute file (`Oralable__21_.csv`, 06:19–06:33 UTC) delivered PPG at 50.2 Hz (42,763 samples, 851 s) and the accelerometer at 50.3 Hz (42,502 samples, 845 s). Both sit above the 45 Hz floor.

Raw file SHA-256 `b2e16fe0a22e3a99489cc016748fc9772097adf615622a6797628a7efb11f819`. Image SHA-256 `55948a6ec56883d0e5060f84a8c56f1f7c99e46c627363acebaaa9702f45238a`. The PPG reader is commit `cb269ff`. The rest of the firmware tree is still dirty, so this image is not that commit alone.

`verify_gen2_stream_capture.py --strict` returned 2. Three errors: the tree is dirty, PPG skips 184 frame counters, and the accelerometer skips 123. nRF Connect wrote 67 PPG notifies and 81 accelerometer notifies twice. Those identical copies are counted once and are not errors. Equal receipt times are no longer an error. No approved Gen 1 capture is in this repository. G2-P3 stays pending. This check is not a gate pass.

---

## Stage A — 3 Nov candidate, not the image now on the board

| Item | Value |
|------|--------|
| String in the ELF | `2.0.0-gen2-nrfconnect` |
| Boot line in the ELF | `Booting My Application v2.0.0-gen2-nrfconnect-7f4db62efa46` |
| `app/VERSION` | **1.0.87** (unchanged by this step) |
| Pin scan | `CONFIG_ORALABLE_I2C_PIN_SCAN` is not set. `i2cm` is not in the ELF. |
| Poll-keep | Still in this image |
| Git | `7f4db62efa46211269f64d115c9cd1776edf85a7` on a dirty tree. Not committed. |
| Hex | `/Users/johnacogan67/ncs/build_pcb00003_gen2_ota/merged.hex` |
| SHA-256 | `52a729e4ec436604f471613579f0961a806c97b776ac43f94e6b2af9e06acc80` |
| Built | 3 Nov 2026, 04:30, 811676 bytes. Log lines thinned: poll every 4 s, keep once per caller, hold-green on a 50 mV step, no wrapped infrared AC. |
| On a board | No. This hex was not programmed. The board later got the 8 Nov image in Stage C. |

The version string has been read on the 8 Nov image. This 3 Nov hex is a different file. Do not program it over the board.

## Stage E — log collected 8 Nov 2026, gate still pending

`Oralable__24_.csv` runs 07:18:30–07:20:46. SHA-256 `1293b729634abe1e6b4de47749c49ccf2b5d2322c527dfec2827e6bbf5330d97`. Service `3A0FF000` lists `001` through `00D` at 07:18:30 and again at 07:20:24. `005` and `006` have no descriptors. `00B` has no descriptors. The others have a client configuration descriptor. An SMP service is also discovered. That is not an update.

Reads: `005` is `0x6C06CBA8584D4300`. `006` is `2.0.0-gen2-nrfconnect` at 07:18:51 and again at 07:20:46. `004` is 4202 mV. `009` is dock=1, worn=0, state=1, 100%, charge=0. `00C` is 16 bytes: version 2, green PA 5, IR PA 128, red PA 32, stats period 5 s, battery interval 10 s, temp interval 1 s, stream mask 3, charger pin on, stable 2 s, mismatch 0. `00D` is 4224 mV, 100%, flags `0x03` (estimate, on the case), 0 mAh, minutes `0xFFFF`.

Notify on `00A` only. The first line is `fw: log stream open`, then charger pin on, phase 2, `ch=0`, `mv=4224`. PPG and accelerometer notifies stayed off. Disconnect is 07:20:15.776. Connecting is 07:20:20.934. Connected is 07:20:23.876. Two lines say `Service Discovery returned nil Services`. The thirteen characteristics are listed after those lines.

`Oralable__23_.csv` did not contain this inventory or this disconnect. G2-P2 stays pending until a reviewer accepts the log. This file does not close the gate.

## Stage D — note written 9 Nov 2026, gates still pending

The pin map above is `pcb00003_gen2_nrf54l15_cpuapp.dts` and its pinctrl file. Wout confirmed SDA, SCL, and SENS_EN on 25 Oct 2026. Hardware doc §5 no longer carries the hypothesis table that put SDA on P0.03 and BATVOL on P1.04.

9 Nov 2026, probe J-Link **1057765387**, device `NRF54L15_M33`, 500 kHz. The connect log is `outputs/gen2-reviewer-2026-11-09/swd_head.txt`, SHA-256 `bb68790c291b79269088feef2a7e31e8a853ed6ca399af98e72cb7375e655f50`. It prints SW-DP `0x6BA02477`. The flash read at `0x45000` is `flash_version.bin`, SHA-256 `3305cf77d46815e9c5743fc624f33bd4488e4e4226e6934401dbbf87fac884d9`. It contains `*** Booting My Application v2.0.0-gen2-nrfconnect-7f4db62efa46 ***` and the name Oralable.

A later read, with no reset, saved the RTT up-buffer at `0x20000010` to `outputs/gen2-reviewer-2026-11-09/rtt_up0_raw.bin`, SHA-256 `bc295f40ed4a6c1012221588c1411d301c51fb342acce8383f599cab93b42edd`. The connect log is `rtt_go9b_jlink.txt`, SHA-256 `1ce27fbaf9ffffe94f6e999f2f5de42d322b57a26b4901c2363381a047b23ee9`. It prints SW-DP `0x6BA02477`. The buffer contains `MAXM86161 part id 0x36 err=0` at 00:00:02.913 and `LIS2DTW12 WHO_AM_I 0x44 err=0` at 00:00:02.914, under the same boot line. A BLE line `lis44=0 maxid=0` is not an identity read.

G2-P0 and G2-P1 stay pending until a reviewer accepts this note.

## Stage F — signed zip built 9 Nov 2026, test not run

Commit `aeb542d` records the Gen 2 sources. It was built from a clean checkout of that commit with NCS v2.9.3 and toolchain `b8efef2ad5`. `app/VERSION` was not changed and is still the Gen 1 number. The Gen 2 board conf signs the image as **2.0.0+0**, which is newer than the **1.0.87+0** image on the module.

`imgtool` validated the signature against `root-ec-p256.pem`. The boot line in the file is `Booting My Application v2.0.0-gen2-nrfconnect-aeb542d395a0`. The zip is `outputs/gen2-reviewer-2026-11-09/oralable_gen2_2.0.0_dfu_application.zip`, SHA-256 `176c1914fd17cb96747505a9009f8048ffbd471910b5d57a04a640a8e647d527`. The signed bin is `oralable_gen2_2.0.0_app.signed.bin`, SHA-256 `c97d6906f8b1d8c7b92e90247697aaa752adc2ea6adca1f303e9fcbf40800afe`.

9 Nov 2026, the same signed file was also written over SWD with probe **1057765387**. The program log `program_2.0.0.txt` is empty. The read-back is the record. The slot at `0xC000` starts with MCUboot magic `0x96F3B83D` and version **2.0.0+0**. The boot line on the chip is `Booting My Application v2.0.0-gen2-nrfconnect-aeb542d395a0`. The bootloader at address 0 was not erased. An SWD load does not count as G2-P5. The file to refuse in Device Manager is `oralable_gen2_1.0.87_on_board_dfu_application.zip`, SHA-256 `e22fc45c0273bc951d8073e5bc987d5f6532c25905b62d2a13d268b4b6410b42`. The 2.0.0 zip is the same version as the chip, so Device Manager refuses that too.

Commit `aaa36e2` signs the next image as **2.0.1+0**. It was built from a clean checkout and was not programmed. `imgtool` validated it. The boot line in the file is `Booting My Application v2.0.0-gen2-nrfconnect-aaa36e2ba665`. The zip is `outputs/gen2-reviewer-2026-11-09/oralable_gen2_2.0.1_dfu_application.zip`, SHA-256 `346d54bde110c8e80bf12f3baf7fce9b7a6f4cfcc02360a2c824b353ed1dcc88`. The signed bin is `oralable_gen2_2.0.1_app.signed.bin`, SHA-256 `03000f19923bdecb7c6520890bf364e8a0d921278530deed6241d4dcafe18abb`. G2-P5 stays pending until a log shows the update, a reboot, a reconnect, and one failed update or rollback.

`Oralable__25_.csv` is an nRF Connect session from 08:46:23 to 08:47:46, SHA-256 `e64d4b8ab0ca90f78032b4fd93168ca52f7e7388888ac9fcaace310d1a777c80`. The phone connected at 08:46:30, found SMP and `001` through `00D`, and streamed PPG and the accelerometer until the file ends. There is no disconnect and no read of `006`. The only writes are notify switches on `009`, `00A`, `001`, and `002`. nRF Connect changed the appearance to McuMgrDFU because it saw SMP. The update was done in Nordic Device Manager, and that transfer is not in this file. Firmware log at 08:47:46: `ppg_ok=1818`, `ppg_err=2`, `acc_ok=1392`, `acc_err=3`, `ppg_ovf=1`, `ch=0`, `mv=4081`. The cell is off the charger. This file does not close a gate.

`Oralable__26_.csv` runs 08:54:34–08:55:47, SHA-256 `dbd15fe803d575ee6203c65c66d5ed80df29db779c29b54a281902f8441bd5fe`. It opens on a read of `009` and has no connect line, no disconnect, no `006` read, and no SMP write. `009` is dock 0, worn 0, state 0, 90%, charge 0. `004` is 4114 mV. The firmware log at 08:55:23 is `ppg_ok=5622`, `ppg_err=46`, `acc_ok=5045`, `acc_err=51`, `ppg_ovf=2`, `ch=0`, `mv=4114`. Those counters are higher than `Oralable__25_` at 08:47:46 (`ppg_ok=1818`, `ppg_err=2`), so this is the same boot. The module did not reboot between the two files. This file does not show which image is running.

`Oralable__27_.csv` runs 09:00:35–09:01:16, SHA-256 `0d54afa55117f4f098ff1365356c8525e47503491d46fb6a029d19f2cd76c636`. Connect is 09:00:42. SMP and `001` through `00D` are discovered. Two nil-services lines do not hide them. There is no disconnect, no `006` read, and no SMP write. `009` is dock 0, worn 0, state 0, 88%, charge 0. At 09:01:00 the counters are `ppg_ok=12674`, `ppg_err=73`, `acc_ok=11804`, `acc_err=114`, `ch=0`, `mv=4092`. They are higher than `Oralable__26_` at 08:55:23, so this is still the same boot. The appearance change to McuMgrDFU is nRF Connect seeing SMP. This file does not show which image is running.

`Oralable__28_.csv` runs 09:03:49–09:04:17, SHA-256 `f9a2a0a74dceade7af9fcbab27905cac238eb35b76bda030d45442c9ff8a398a`. Connect is 09:03:57. The `006` read at 09:04:17 is `2.0.0-gen2-nrfconnect`. That string is compiled into both the **2.0.0+0** image and the **2.0.1+0** image. This read does not separate them. There is no counter line and no SMP write.

`Oralable__29_.csv` runs 09:07:20–09:08:50, SHA-256 `5138a2b69fb2e424dcede8f174f4b37c9e9345d3367aae5e043bba26ce1b7e45`. The link drops at 09:07:23 and returns at 09:07:25. `006` at 09:07:53 is again `2.0.0-gen2-nrfconnect`. `00A` has no `ppg_ok` line. `ppg_ovf` is 3, then 4. It was 2 at 08:55, so this is still the same boot. `ch=0`, `mv=4092`. There is no SMP write.

`Oralable__30_.csv` runs 09:13:48–09:14:48, SHA-256 `40000804bb4ba8199288e4066db38f4a4ed6423905351ee7c513af27a05d6213`. It starts during discovery. `006` is still `2.0.0-gen2-nrfconnect`. There is no write to `00B` and no SMP write. Before PPG starts, `ppg_ovf=0` and the charger heartbeat count `ih=18`. At 09:08:50 those were `ppg_ovf=4` and `ih=255`. The module rebooted between the two files. `009` goes from worn 0 to worn 1 after PPG starts. `ch=0`, `mv=4070`. This reboot is not an update.

`Oralable__31_.csv` runs 09:13:48–09:20:04, SHA-256 `a60dab355865b10cd183b67aeb08355cee8f8754aa6f2485c21527273181f8b0`. It includes the session above. At 09:18:46 the phone writes `0A 01 00` to `00B`. The log answers `fw: cfg debug_reboot_s=1`. The link drops at 09:18:47 and returns at 09:19:07. `006` is still `2.0.0-gen2-nrfconnect`. At 09:19:41, before PPG starts, `ppg_ok=0`, `ppg_err=0`, `acc_ok=0`, and `ppg_ovf=0`. The bench reboot happened. There is no SMP upload. G2-P5 stays pending.

`Oralable__32_.csv` runs 09:39:29–09:41:20, SHA-256 `0aa24d9b9b749c3ef8763a4f5f21934534ffa1663e56467ff4512e55d5472523`. At 09:40:10 the phone writes `0A 01 00` to `00B` again. The link drops at 09:40:12 and returns at 09:40:17. `006` at 09:40:36 is `2.0.0-gen2-nrfconnect`. At 09:40:47, before PPG starts, `ppg_ok=0` and `ppg_ovf=0`. The heartbeat count is `ih=18`, `ch=0`, `mv=4070`. One write of `01` turns on SMP notifications. No image is uploaded. G2-P5 stays pending.

Commit `8c3177f` makes characteristic `006` follow the MCUboot sign version. The image signed **2.0.1+0** contains `2.0.1-gen2-nrfconnect`. `imgtool` validated it. The boot line in the file is `Booting My Application v2.0.1-gen2-nrfconnect-8c3177fa1fc8`. The zip is `outputs/gen2-reviewer-2026-11-09/oralable_gen2_2.0.1_readback_dfu_application.zip`, SHA-256 `fd7b24e4748a1e0e8414abdb15a4fa3468fb2be1d7e750a68883ffb5298d84d3`. It has not been programmed. `Oralable__33_.csv` runs 10:17:42–10:19:11, SHA-256 `5908478d2d45c7635e012aa26644f88d5162bf35be7cb4c755b7ea696ad51eb4`. At 10:18:10 characteristic `006` reads `322E 302E 312D 6765 6E32 2D6E 7266 636F 6E6E 6563 74`, which is `2.0.1-gen2-nrfconnect`. Before PPG starts, `ppg_ok=0` and `ppg_ovf=0`. The first heartbeat in the file is `ih=18`, `ch=0`, `mv=4081`. Status `009` is `00 00 00 56 00` (86%, not charging). The file has no SMP write and no disconnect. The image is on the module. The upload itself is not in this file. G2-P5 stays pending. The earlier `oralable_gen2_2.0.1_dfu_application.zip` is signed 2.0.1 but its `006` string is still `2.0.0-gen2-nrfconnect`.

On 9 Nov 2026 G2-P0, G2-P1, and G2-P5 were removed from `gen2.program.json`. The release check does not require a pin-map review, a sensor-identity review, or an update log.

`Oralable__34_.csv` runs 08:11:36–08:12:06 on 10 Nov 2026, SHA-256 `698048686b6286aec72dc79a9fcb1cdc7a9356b1ecd34c65fce7141c6b80f92f`. It opens on a disconnect, connects again at 08:11:42, and lists characteristics `001` through `00D`. `005` reads `0043 4D58 A8CB 066C`, the same device id as the 8 Nov inventory. `006` reads `2.0.1-gen2-nrfconnect`. The file ends on that read. No notification is enabled. G2-P2 stays pending.

`Oralable__35_.csv` runs 08:11:36–08:16:50 on 10 Nov 2026, SHA-256 `6e183bc8c098280735dc01ee73e46f927e3a6fb66cefc42f0a454504ae2769e2`. It includes the session above, then turns notifications on for `00A`, `009`, `002`, and `001`. Each one delivers a value. `009` is `00 00 00 42 00` (66%, not charging). The log reads `ch=0`, `mv=3894`, and `bat hold probe green`. The link drops at 08:16:50. PPG and accelerometer run for about 14 seconds. There is no read of `004`, `00C`, or `00D`. G2-P2 stays pending until a reviewer accepts the log. G2-P3 and G2-P4 stay pending.

`Oralable__37_.csv` runs 08:19:26–08:32:52 on 10 Nov 2026, SHA-256 `11171240b67a1fc0c21fb160f475e24cc0059d7aa2f309f7cb0f6fd5c4bf748a`. Notifications for `00A`, `002`, and `001` stay on for about 13 minutes. The device counters advance at 50 Hz. The phone log skips 4456 PPG counters and 4693 accelerometer counters. The first skip is at 08:19:48. During the file, `ppg_err` rises from 8933 to 11130 and `ppg_last` stays `-12`. `mv` moves between 3861 and 3905 with `ch=0`. The firmware keeps the green LED only (`bat hold probe green`). The file ends while both streams are still notifying. G2-P3 and G2-P4 stay pending.

The source now keeps a refused PPG or accelerometer frame and sends those same bytes again. The counter advances only after the send is accepted. That image is signed **2.0.2+0**. `imgtool` validated it. The zip is `outputs/gen2-reviewer-2026-11-09/oralable_gen2_2.0.2_dfu_application.zip`, SHA-256 `b5ee8b133c458314fb38c1f13263793be656a9b62d90c03a3dd32c923b0d5d8c`. It had not been programmed yet. The last read of 006 was still `2.0.1-gen2-nrfconnect`. On 10 Nov 2026 Device Manager refused that zip: there is no free slot. The spare slot is still in use after the 2.0.1 update. Tapping Read on the Image tab closes Device Manager before the slot list stays on screen. At 09:57 the same file was accepted and Device Manager sent 268899 bytes, the full app.signed.bin, at 30.21 kB/s. It then stopped with ResultLockError timeout. 006 was not read after that send. Oralable__38_.csv runs 10:03:38–10:16:06, SHA-256 `5633fb38f46377b16a3909b56e1932a58f85ddec6b3b2fc12cf0a7bbec6c990a`. It has no read of 006. The notify counts start again near zero. `ppg_err`, `acc_err`, and `ppg_last` stay 0. Recorded PPG counters run 1–1850 with no hole, 733 s, 50.4 Hz. Recorded accelerometer counters run 0–1483 with no hole, 738 s, 50.3 Hz. Counter 0 is absent from the PPG lines, and the firmware count is one higher. `mv` stays 3861 and `ch` stays 0. The log says the probe stayed green. The file ends while both streams are still notifying. Oralable__39_.csv runs 10:23:00–10:23:20, SHA-256 `43ef2b0aed7f2a783d75428f92c1457f9636981abaafd087bcdb4182e94c9e09`. At 10:23:13 the version read is `2.0.2-gen2-nrfconnect`. The notify counts continue from the __38_ boot: `ppg_ok` is 2916 and `ppg_err` is still 0. `mv` stays 3861 and `ch` stays 0. Oralable__40_.csv runs 11:28:09–11:32:14, SHA-256 `beb2c09b7ff0bb61dafb9af236341915452d1de6c7c8815225a317992f7b38d9`. It connects, lists `001` through `00D` plus SMP, and reads `005` (`0043 4D58 A8CB 066C`), `006` (`2.0.2-gen2-nrfconnect`), `004` (3850 mV), `00C`, and `00D`. Notifications on `00A`, `009`, `002`, and `001` each deliver a value. The link drops at 11:31:51 and returns at 11:31:56. The second discovery lists the same characteristics, and `006` at 11:32:14 is still `2.0.2-gen2-nrfconnect`. `mv` is 3850, `ch` is 0, and the log says the probe stayed dark. `ppg_last` is `-128` once. On 11 Nov 2026 a reviewer acceptance of this log was reported. `gen2.program.json` stays pending until that acceptance names the reviewer. Oralable__41_.csv runs 11:37:38–11:47:27, SHA-256 `f7f84d1c356de91ce48f642e397239a31d8f28ff8061d5d424cab8ffc1c30ece`. The file opens already on the charger: `on=1`, `ph=1`, `ch=1`, `mv=3905`. `004` reads 3905 mV. Status `009` is `01 00 01 43 01` (on the dock, 67%, charging). `mv` moves 3905, 3894, 3916, 3905, 3927, then 3938. At 11:46:53 the phase goes from 1 to 0, `ch` goes to 0, and `009` becomes `00 00 00 46 00` (off the dock, 70%, not charging). G2-P4 stays pending. Oralable__42_.csv runs 11:53:18–12:09:22, SHA-256 `9487820af308d0c16fbe9522ae2d969bd0aa77280b51c5996ac680c1fd506fcc`. PPG runs 919 s at 50.4 Hz, counters 3489–5808, no hole. Accelerometer runs 926 s at 50.2 Hz, counters 2828–4689, no hole. `ppg_err` stays 1 and `ppg_last` stays `-128`. The log says `ppg poll keep battery-dark` and `bat hold probe green`. Heartbeat `mv` is 3872–3883. A load reading of 3905 turns the dim red and IR on, and the next reading, 3872–3883, turns them off again. That pair repeats six times. G2-P3 stays pending. Oralable__43_.csv runs 15:47:07–16:01:26, SHA-256 `501cdc9081a7d47f79d304ac6736447505852948453bad9808d257d30c6fe32b`. It opens on the charger at `mv=3905`, `ch=1`. PPG and the accelerometer run 824 s and 827 s at 50.4 Hz and 50.3 Hz, counters 6283–8362 and 5068–6733, no hole. `ppg_err` stays 1. For about 15 s the firmware drives full red and IR (`dim=0`) and the IR slot moves. At 15:47:57 `bat hold probe green` reads `mv=3872`. At 15:48:06 the module leaves the charger. The heartbeat then reads 3850 and 3861 with `ch=0`. G2-P3 stays pending. Oralable__44_.csv runs 02:10:36–02:27:54 on 11 Nov 2026, SHA-256 `61767e32623fd0247fa4921f3cf5d0fea4d43e9160af35c1b4603b5caf534eb7`. Off the charger, heartbeat `mv` is 3938–3971 and `ch` is 0. Every under-load reading is 3916–3971, so full red and IR stay on. PPG runs 1019 s and the accelerometer 1024 s, both near 50 Hz. The phone log skips 150 PPG counters and 211 accelerometer counters. Each skip is one or two steps. `ppg_err` rises from 4 to 8 and `ppg_last` stays `-12`. The IR DC sits near 155000 and the worn bit flips. G2-P3 stays pending. The next image is signed **2.0.3+0**. `imgtool` validated it. The zip is `outputs/gen2-reviewer-2026-11-09/oralable_gen2_2.0.3_dfu_application.zip`, SHA-256 `d074eac2fd63a88288d8e4fb9c9e8a1b3684af65cebf86e508d05653949d080b`. It is not on the module. GATT 006 in the file is `2.0.3-gen2-nrfconnect`. The 20 ms poll still reads the sensors. A notify goes out only when the frame is full: 20 PPG samples, 25 accelerometer samples. At 02:56 on 11 Nov 2026 Device Manager shows that file, 269 kB, hash 0xA9AE40…87FCEF, and refuses it: there is no free slot. The upload has not started. Oralable__45_.csv runs 03:04:52–03:15:37 on 11 Nov 2026, SHA-256 `7480085a893fe2a8ded850a6b0dc0518df45ecb5cd5b8d4b6b2e7755dab9ad1b`. It has no read of 006. Off the charger, under-load is 3905–3960 mV and dim stays 0, so full red and IR stay on. PPG is 1571 frames of 20 samples, counters 0–1570, 622 s, with no hole. The accelerometer is 1266 frames of 25 samples, counters 0–1265, 628 s, with no hole. `ppg_err` stays 0. The IR word sits near 116000, then 172032, and the one `ir_pulse` line is worn=0, ac=24, dc=17200. G2-P3 stays pending. Oralable__46_.csv is one read at 03:22:27 on 11 Nov 2026, SHA-256 `e12e3a49303c69810183f3ebe10894317d5a489dd6ed886f3744901a5be9bd6a`. The 21 bytes are `2.0.3-gen2-nrfconnect`. The capture manifest for `Oralable__45_.csv` is `outputs/gen2-reviewer-2026-11-09/oralable_45_g2p3_capture.json`. `verify_gen2_stream_capture.py --strict` returned 2. PPG delivered 31420 samples at 50.485 Hz with 0 frame gaps. The accelerometer delivered 31650 samples at 50.350 Hz with 0 frame gaps. The one error is that the tree is dirty. The image on the module embeds `8c3177fa1fc8`. A clean rebuild of commit `220d241` embeds that commit id, so the signed hashes differ. G2-P3 stays pending. This check is not a gate pass. Commit `fe4e35d2d0f59ff88c785301f4ef5661b16bf11c` signs the image as **2.0.4+0**. It was built from a clean checkout of that commit. `imgtool` validated it. The boot line embeds `fe4e35d2d0f5`. GATT 006 in the file is `2.0.4-gen2-nrfconnect`. The zip is `outputs/gen2-reviewer-2026-11-09/oralable_gen2_2.0.4_dfu_application.zip`, SHA-256 `e44ace8da7d26918b192be825a81021dff6fe1dc92cc6df85082bc5ed6b3acbd`. The signed bin is 269284 bytes, SHA-256 `7b4263e32042a22705db104d9cd0b84dae0dd79b6b31b5c93ca8237e42b6d029`. It is not on the module. Oralable__47_.csv runs 03:45:43–03:56:19 on 11 Nov 2026, SHA-256 `6edf331e9d8c836d72f89366b100fd5c0654b98fa5304fe64da37c29dde1a080`. The first line is `2.0.4-gen2-nrfconnect`, 21 bytes, so that image is now running. The boot banner is not in the file. Notifications on `00A`, `002`, and `001` stay on. Off the charger, `ch` is 0 and every under-load reading is 3905–3938 with `dim=0`. PPG is 1550 frames of 20 samples, counters 0–1549, 613.5 s, 50.530 Hz, no hole. The accelerometer is 1243 frames of 25 samples, counters 0–1242, 616.8 s, 50.381 Hz, no hole. Both stats lines keep `ppg_err`, `acc_err`, and `log_err` at 0. `ppg_ovf` and `ppg_backlog` go to 1 at the start and stay there. One `ppg poll keep battery-dark` line does not turn the red and IR off. IR is 524287 on 28224 of 31000 samples. Green is 73728 on every sample. Red settles near 192000. The worn bit flips on that IR. The file ends while both streams are still notifying. G2-P3 stays pending. Oralable__48_.csv runs 04:09:55–04:11:45 on 11 Nov 2026, SHA-256 `124feb42ead530606a050c18307b881d2c0dbdeb88aad64ba50882806b92e7cc`. It reads `005` (`00434D58A8CB066C`), `006` (`2.0.4-gen2-nrfconnect`), `004` (3905 mV), `00C` (16 bytes: version 2, green PA 5, IR PA 128, red PA 32, stats 5 s, battery 10 s, temp 1 s, stream mask 3, charger fields 0), and `00D` (3905 mV, 67%, flags `0x01`, 0.54 mAh, 2852 minutes). Notifications on `00A`, `009`, `002`, and `001` each deliver a value. `009` is dock 0, worn 0, state 0, 67%, charge 0, then worn 1 and state 2. PPG in this file is 31 frames of 20 samples, counters 2472–2502. The accelerometer is 33 frames of 25 samples, counters 1979–2011. `ppg_err` and `acc_err` stay 0. `log_err` is already 1 and `log_last` is `-128`. The file has no connect line, no characteristic inventory, no disconnect, and no reconnect. The second `006` read is on the same link after the notifications are turned off. G2-P2 stays pending. Oralable__49_.csv runs 04:17:07–04:20:09 on 11 Nov 2026, SHA-256 `339f039498d6bf8f398dce34c03530fad013de171fb6dd86f57bae71eb4a18df`. It disconnects, connects at 04:17:10, and lists SMP plus `3A0FF000` with characteristics `001` through `00D`. It reads `005` (`00434D58A8CB066C`), `006` (`2.0.4-gen2-nrfconnect`), `004` (3916 mV), `00C` (the same 16 bytes as __48_), and `00D` (3916 mV, 68%, flags `0x01`, 0.57 mAh, 2843 minutes). Notifications on `00A`, `009`, `002`, and `001` each deliver a value. The link drops at 04:19:41 and returns at 04:19:52. The second discovery lists the same characteristics, and `006` at 04:20:09 is still `2.0.4-gen2-nrfconnect`. `ppg_err` and `acc_err` stay 0. `log_err` is 2 and `log_last` is `-128`. Two “nil Services” lines on each connect are the empty included-service result. On 11 Nov 2026 John reviewed this log. The copy is `outputs/gen2-reviewer-2026-11-09/Oralable__49_.csv`, same SHA-256. G2-P2 in `gen2.program.json` is passed. `release_eligible` stays false. G2-P3, G2-P4, and G2-P6 stay pending. Oralable__50_.csv runs 09:06:01–09:08:43 on 13 Nov 2026, SHA-256 `7545fa809e64c89dd04d82b5c696576bae3ed544d549cae5cea47abb04d4e171`. It disconnects, connects, and lists SMP plus `3A0FF000`. `006` is `2.0.4-gen2-nrfconnect`. Off the charger, the heartbeat is `mv=3839` and `ch=0`. One line says `bat hold probe green mv=3828 cached=3839 pa=5`. PPG is 249 frames of 20 samples, counters 2565–2813, 97.8 s, 50.904 Hz, no hole. The accelerometer is 212 frames of 25 samples, counters 2066–2277, 104.6 s, 50.679 Hz, no hole. `ppg_err` and `acc_err` stay 0. `log_err` is 3 and `log_last` is `-128`. Red and IR sit on 8360. Green settles near 10020 with about 20 counts of wander. The file ends while both streams are still notifying. G2-P3 stays pending. Oralable__51_.csv runs 14:56:21–14:57:44 on 13 Nov 2026, SHA-256 `d87e03553a8f264d8d4d7f610dc80f1a91be0cfabfe44d3e97677ca7c53b9dc6`. It connects and lists SMP plus `3A0FF000`. `006` is `2.0.4-gen2-nrfconnect`. The module is on the charger: `on=1`, `ph=1`, `ch=1`, `mv=3806`. Status `009` is dock 1, worn 0, state 1, 56%, charging. PPG is 66 frames of 20 samples, counters 3659–3724, 25.2 s, 52.377 Hz, no hole. The accelerometer is 58 frames of 25 samples, counters 2954–3011, 28.3 s, 51.199 Hz, no hole. `ppg_err` and `acc_err` stay 0. `log_err` is 4 and `log_last` is `-128`. `ppg_ovf` goes from 4 to 5 when PPG starts and stays there. One line says `ppg poll keep status-leds`. Red and IR sit near 10960. Green jumps by about 5000 counts on the charge blink and sits with them between blinks. The file ends while both streams are still notifying. G2-P3 stays pending.
