# IR-DC and PPG ADC Format

**Default configuration** (Oralable MAM cheek): `R_G_IR` channel order, IR-DC coupling 10M–70M raw.

## IR-DC

**IR-DC** is the low-frequency (&lt;1 Hz) component of the infrared PPG signal. It reflects:

- Baseline blood volume under the sensor
- Tissue perfusion at the measurement site
- **Hemodynamic occlusion** when blood flow is reduced (e.g. masseter clench)

For Oralable MAM on the cheek, a clench compresses tissue and reduces IR perfusion, so IR-DC drops. Validation uses this drop (e.g. ≥2.5%) to detect clench.

---

## MAXM86161 Sensor (oralable_nrf)

- **ADC:** 19-bit charge integrating ADC (0–524,287)
- **FIFO:** 3 bytes per channel × 3 channels = 9 bytes per sample
- **Driver:** `app/drivers/sensor/maxm86161/maxm86161.c` masks with `0x7ffff` (19-bit)

Firmware sends `struct ppg_sample { uint32_t red, ir, green }` over BLE.

---

## Observed Raw Range

Logs from Oralable_6 show IR in **~16M–67M** (32-bit unscaled). This indicates either:

1. Different firmware (no 19-bit mask, or different sensor config)
2. Different channel mapping (e.g. `R_G_IR` vs `R_IR_G`)

Run `scripts/check_ir_dc_scaling.py` to verify channel order and raw range for your logs.

---

## Coupling Check

`self_validate.py` uses `IR_DC_RAW_MIN` and `IR_DC_RAW_MAX` to flag sensor coupling:

- **19-bit firmware:** 30,000–400,000
- **32-bit firmware (observed):** 10,000,000–70,000,000

Update these in `src/validation/self_validate.py` (lines 46–47) to match your hardware.

---

## Channel Order

BLE payload per sample: 12 bytes (3 × uint32 LE). Order can be:

- **R_IR_G:** Red, IR, Green (per `tgm_service.h`)
- **R_G_IR:** Red, Green, IR (use when middle slot is constant/low; IR in slot2)

If `scripts/check_ir_dc_scaling.py` reports `slot1` constant, use `R_G_IR`.

---

## Voltage Conversion

`ADC_TO_V_SCALE = 3.3 / 65535` assumes 16-bit. For 32-bit raw values, voltage conversion is incorrect; use raw counts for coupling checks.
