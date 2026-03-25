# OTA Firmware

## File

**`app_update.bin`** – MCUboot-compatible application update for BLE DFU (Over-The-Air).

- **Size:** ~204 KB (includes Power Profiler)  
- **Board:** pcb00003 (nRF52832)  
- **Use:** Transfer via nRF Connect / mcumgr to Oralable device

## Contents

This build **includes Power Profiler & Battery Tracking**:
- State-aware mAh integration (ADV/CONN/STREAM)
- CG-320B battery curve (4.35V–3.0V)
- Critical low shutdown at 2.8V
- BatteryStats characteristic (3A0FF00A), 60s notify

## Rebuild

From `Projects/tgm_firmware` (with west):
```bash
west build -b pcb00003 --pristine
```
Then copy `build/zephyr/app_update.bin` here.
