# Gen 2, Wout, and Kaga — where things stand

**As at:** 30 Oct 2026  
**Later bench:** [GEN2_BENCH_LEARNINGS_2026-11-02.md](./GEN2_BENCH_LEARNINGS_2026-11-02.md). The text below is the 30 Oct record. PPG and the accelerometer have since streamed on one board until the cell fell through the floor. Gates stay pending. Kits stay gated.  
**Hardware:** Gen 2 · BOM **REV9** · PCB **REV11** · Kaga **ES4L15BA1** (nRF54L15) · cell **LP270829**  
**Related:** [VERSION_ALIGNMENT.md](../VERSION_ALIGNMENT.md) · [KAGA_NCAB_PC53020_EQ_2026-08.md](./KAGA_NCAB_PC53020_EQ_2026-08.md) · [PCB00003_GEN2_REV11_HARDWARE.md](../../PCB00003_GEN2_REV11_HARDWARE.md) · [GEN1_GEN2_TRACKING.md](../../GEN1_GEN2_TRACKING.md)

**One-liner:** Work is Gen 2. Gen 1 is legacy. The nRF on the Gen 2 board runs, and one board advertised as Oralable. The PPG and the accelerometer do not answer. The next question is for Wout. It is not a reason to send the board back to Kaga.

---

## Claim discipline

| Do say | Do not say |
|--------|------------|
| Bring-up. Kits still **gated**. | Kits shipped, or ready to hand to Ed or Pedro |
| One blank board was programmed on **29 Oct 2026** with the 26 Oct merged hex | Kit firmware is **2.0.x**, or G2-P2 / G2-P5 are done |
| nRF Connect listed **Oralable** on a board that was already running | PPG or accelerometer samples were seen |
| Wout confirmed SDA **P1.07**, SCL **P1.03**, SENS_EN **P1.08** on **25 Oct 2026** | The hypothesis GPIO table in the hardware doc §5 |
| The 20-unit NCAB **PC53020** lot is a separate Kaga thread | That lot has arrived |

The image header on the board that was already running read **1.0.87**. That is not a kit version. GATT `3A0FF006` was not read after the 29 Oct program.

---

## Synopsis

Wout (WeeGee) is the designer. He confirmed the REV10 schematic is the REV11 design, and he confirmed the three sensor pins above, with 10 kΩ pull-ups on the SENS_EN net. Kaga supplies the **ES4L15BA1** module and, through Hosiden Besson and NCAB, the board lot. The August EQ on job **PC53020** is still that separate 20-unit thread. The boards on the bench are not that lot marked received.

On the bench the module is alive. A debugger on the nRF54L15 DK (J-Link **1057765387**) can halt a powered chip. One board that was already running advertised as **Oralable**, connected, and returned a firmware log. Cell voltage in that log was **3784 mV**. PPG notify and accelerometer notify were turned on. No samples arrived. Both sensor chips returned a NACK (`-EIO`), not their part ids. A one-shot pin scan in that image stored false hits (a byte of 0, or a timeout). Those hits are not the pin map. The scan is still in that image. Take it out before the image is the one to keep.

A different board, blank from Kaga, was programmed on **29 Oct 2026** with `artifacts/oralable_gen2_ota_merged.hex` (26 Oct). The write was **299008** bytes and verified. Tag-Connect was to come off after reset so a debugger halt does not drop the battery latch. That board was not seen advertising after the program. The earlier board was not overwritten: its bootloader matched the same hex, but the application image size in the header did not (**261636** bytes on the chip, **260632** in the file).

Idle voltages on the board that was already running, black lead on the black cell wire: cell **3.76 V**, 3V3 **3.26 V**, SENS_EN **3.26 V**, SDA **3.24 V**, SCL **3.24 V**. The pull-ups are up. The chips still do not answer. A voltmeter cannot show an I2C ACK, read a part id, or prove the parts are soldered.

The note to Wout is written and not sent. It asks whether U2 (MAXM86161) and U4 (LIS2DTW12) were on the pick-and-place for these REV11 boards, which net feeds VDD on each, and whether the TPS79733 enable is tied on. Kaga is not the next stop unless Wout says the module pins for SDA, SCL, and SENS_EN are not the ones already in the board file.

---

## Pins in use

Use the board file `pcb00003_gen2_nrf54l15_cpuapp.dts`. Do not use hardware doc §5.

| Net | GPIO | Note |
|-----|------|------|
| SDA | P1.07 | Wout, 25 Oct |
| SCL | P1.03 | Wout, 25 Oct |
| SENS_EN | P1.08 | MAXM86161 LDO_EN. 10 kΩ pull-ups on this net. Not the accelerometer supply. |
| BATEN | P1.04 | Latch. Firmware drives it high. A debugger reset can drop it. |
| BATVOL | P1.05 | ADC |
| CHRSTS | P1.02 | Charger status |

SENS_EN high means the GPIO is high. It does not mean the sensor chips answered.

---

## What is still open

| Item | Owner | State |
|------|--------|--------|
| Are U2 and U4 fitted on these boards, and which net feeds their VDD? Is the TPS79733 enable tied on? | Wout | Questions written. Not sent. |
| PPG and accelerometer samples | Bench | Not seen. G2-P1 and G2-P3 open. |
| Advertise as a kit gate (G2-P2) | Bench | One board listed **Oralable** before 29 Oct. The blank board programmed that day was not checked on air. Version string is not **2.0.x**. |
| OTA swap (G2-P5) | Bench | The merged hex includes the bootloader and was written over SWD. A Device Manager swap was not shown. |
| Remove the I2C pin scan | Firmware | Still in the image on the board that was already running. |
| 20-unit PC53020 lot | Kaga / NCAB | Separate. Not marked received. |
| Ed/Pedro kits | — | **Gated.** No new date. |
