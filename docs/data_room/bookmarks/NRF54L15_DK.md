# nRF54L15 DK — same SoC as Gen 2

**As at:** 17 Oct 2026  
**Tone:** which Nordic kit matches the Gen 2 chip. Not a clearance claim.  
**Related:** [HARDWARE_ROADMAP_nRF54L15.md](../../../../oralable_nrf/docs/HARDWARE_ROADMAP_nRF54L15.md) · [PCB00003_GEN2_REV11_HARDWARE.md](../../PCB00003_GEN2_REV11_HARDWARE.md)

**One-liner:** Buy the **nRF54L15 DK** (order code **nRF54L15-DK**, board **PCA10156**). It carries an nRF54L15, the same SoC as the Gen 2 module (Kaga **ES4L15BA1**, **nRF54L15-CAAA-R**). The nRF5340 DK on the bench (J-Link S/N **1050090445**) is only the programmer.

---

## Bookmarks

| # | Source | Why it matters |
|---|--------|----------------|
| 1 | [nRF54L15 DK product page](https://www.nordicsemi.com/Products/Development-hardware/nRF54L15-DK) | Order code **nRF54L15-DK**. The kit SoC is the nRF54L15. It can also emulate the smaller nRF54L10 and nRF54L05. |
| 2 | [nRF54L15 DK hardware user guide](https://docs.nordicsemi.com/bundle/ug_nrf54l15_dk/page/UG/nRF54L15_DK/intro/intro.html) | PCA10156. QFN48 on the board. On-board J-Link, 32 MHz and 32.768 kHz crystals, buttons and LEDs. |
| 3 | [Developing with the nRF54L series](https://docs.nordicsemi.com/bundle/ncs-latest/page/nrf/app_dev/device_guides/nrf54l/index.html) | Current nRF Connect SDK board target: `nrf54l15dk/nrf54l15/cpuapp`. |

## What the kit matches

| | Gen 2 MAM | nRF54L15 DK |
|--|-----------|-------------|
| SoC | nRF54L15-CAAA-R (CSP) inside ES4L15BA1 | nRF54L15-QFAA (QFN48) |
| Memory | 1.5 MB NVM, 256 KB RAM | Same |
| Radio | Bluetooth LE on the module antenna | Same radio, DK antenna |
| Board target in this repo | `pcb00003_gen2/nrf54l15/cpuapp` | Not in NCS **2.7.0** |

The package differs. The DK does not copy the module’s 15 pins, the certified antenna, or the MAM charger and power latch.

Skip a used board marked **nRF54L15 PDK**. That is the preview kit. NCS **2.7.0** in `oralable_nrf` only has that preview board (`nrf54l15pdk`). A new DK needs a current nRF Connect SDK.
