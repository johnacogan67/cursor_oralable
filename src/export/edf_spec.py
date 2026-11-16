"""Shared Dual A / research EDF channel + annotation labels (parity with OralableCore EDFExportSpec)."""

from __future__ import annotations

FS_HZ = 50.0

# Ordinary signal labels (EDF 16-char max)
CH_IR_DC = "IR_DC"
CH_EMG = "EMG"
CH_SPO2 = "SpO2"
CH_SASHB = "SASHB_cum"
CH_RED = "Red"
CH_GREEN = "Green"
CH_IR = "IR"
CH_ACC_X = "AccX"
CH_ACC_Y = "AccY"
CH_ACC_Z = "AccZ"

ANN_EMG_BOUT = "EMG_bout"
ANN_SPO2_DESAT = "SpO2_desat"
ANN_LABEL_PREFIX = "label:"

CLAIM = (
    "Research Dual A / Oralable EDF+ — not PSG; not AHI/ODI; "
    "SASHB_cum is engineering SpO2<90 AUC, not Azarbarzin HB. "
    "Default: IR_DC viewer a.u. (occlusion↑), SpO2 %dev from 100, EMG, SASHB_cum; "
    "absolute IR-DC/SpO2 in CSV; raw Red/Green/IR omitted."
)
