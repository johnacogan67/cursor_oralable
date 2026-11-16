# ES4L15BA1 — Kaga module datasheet

**As at:** 19 Oct 2026  
**Tone:** what the module maker requires for the antenna and the 32 kHz crystal. Not a radio measurement. Not a clearance claim.  
**Related:** [PCB00003_GEN2_REV11_HARDWARE.md](../../PCB00003_GEN2_REV11_HARDWARE.md) · [NRF54L15_DK.md](./NRF54L15_DK.md)

**One-liner:** The built-in antenna works only with pin 11 shorted to pin 12, and Kaga says not to put metal over that end. The 32 MHz clock is inside the module. The datasheet does not give a capacitor value for it.

---

## Bookmarks

| # | Source | Why it matters |
|---|--------|----------------|
| 1 | [ES4L15BA1 datasheet V1.0 (24 Jun 2026)](https://www.kagafei.com/jp/products/wireless-modules/bluetooth/File/__icsFiles/afieldfile/2026/07/07/ES4L15BA1_DataSheet_V1_0_20260624E.pdf) | Pin 11 `OUT_ANT` tied to pin 12 `OUT_MOD`, as short as possible. No metal over the antenna. No copper in the keep-out on any layer. Antenna is a monopole, max gain −4.8 dBi, tuned on FR-4. A ground shorter than 30 mm can weaken it. External 32.768 kHz crystal CL = 9.0 pF. DCC inductor 4.7 µH, close to the module. About 100 µF near a small cell for transmit droop. |
| 2 | [ES4L15BA1 one-pager V1.2 (26 Jun 2025)](https://www.kagafei.com/jp/products/wireless-modules/bluetooth/File/__icsFiles/afieldfile/2026/01/16/KFEI_BLE_ES4L15BA1_Onepaper_V1_2JE_20250626.pdf) | Outline 3.25 × 8.55 × 1.00 mm. Integrated clock and PCB antenna. nRF54L15. |
| 3 | [ES4L15BA1-EVB manual V1.0 (19 Dec 2025)](https://www.kagafei.com/jp/products/wireless-modules/bluetooth/File/__icsFiles/afieldfile/2026/06/15/ES4L15BA1_EVBManual_V1_0_20251219.pdf) | Kaga’s own board for the same module. Useful when a MAM layout question needs a known-good reference. |

## What this does not say

The REV11 schematic already shorts pin 11 to pin 12. On 19 Oct 2026 the Gen 2 image logged a finished transmit at +4 dBm while the board sat on the charge case, and the phone did not list Oralable. That is a bench fact. It is not a measurement that the layout meets Kaga’s keep-out.

The datasheet does not state the internal 32 MHz capacitor. Do not treat the 15 pF figure in the Gen 2 devicetree as a Kaga number.
