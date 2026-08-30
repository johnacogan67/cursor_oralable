# Bookmark — ANR M40 temporalis sEMG concordance

> **Do not edit this bookmark for Dual A procedure.** Edit the full doc: [../../ANR_M40_CONCORDANCE.md](../../ANR_M40_CONCORDANCE.md).

**As at:** 30 Aug 2026 · Pack **1.1.68**  
**Canonical full doc:** [../../ANR_M40_CONCORDANCE.md](../../ANR_M40_CONCORDANCE.md)  
**Kit:** [ORALABLE_RESEARCH_KIT.md](../clinical/ORALABLE_RESEARCH_KIT.md) · photos [RESEARCH_KIT_PHOTO_SELECTION.md](../clinical/RESEARCH_KIT_PHOTO_SELECTION.md) (FIG-CO-026 / 031)  
**ANR public docs:** [anrcorp.com/documentation](https://www.anrcorp.com/documentation/)  
**Related:** [MEASUREMENT_CONSTRUCT_MAP.md](../clinical/MEASUREMENT_CONSTRUCT_MAP.md) · [TEMPORALIS_ANATOMY_AND_PLACEMENT.md](../clinical/TEMPORALIS_ANATOMY_AND_PLACEMENT.md) · [ACUPEBBLE_VS_ORALABLE_ANR.md](./ACUPEBBLE_VS_ORALABLE_ANR.md) · [BRUXOFF_PSG_GOLD_STANDARD.md](./BRUXOFF_PSG_GOLD_STANDARD.md) · [PAPER_A_FEASIBILITY_PROTOCOL.md](../clinical/PAPER_A_FEASIBILITY_PROTOCOL.md) · [SENSOR_CORROBORATION.md](./SENSOR_CORROBORATION.md) · [GEMINI_TEMPLE_PPG_AVENUES.md](./GEMINI_TEMPLE_PPG_AVENUES.md)

**One-liner:** ANR M40 is the Research Kit **temporalis sEMG** comparator. Mac Dual A remains the methods reference. Concordance nests Oralable SpO₂ with ANR EMG (AcuPebble-style burden context — **not** AHI) and writes research **`session.edf`**. Measured eng pack: `20260812_085110`. iOS Dual Protocol A (`showDualProtocolA`, default OFF) is optional research. Dual A overnight is later.

**Practical Dual A (defaults):** Seat **Oralable alone first** (clench → IR trough) → stack ANR → EMG gate **≥70** → IR drop **≥8%** → SpO₂ AC WARN (non-blocking) → Protocol A → align. Muscle pack OK with `spo2_qc=warn`. FW **1.0.71+:** keep Mac BLE up — Oralable sensors stop if the link drops. Setup + rationale: [canonical § Setup — seat Oralable alone first](../../ANR_M40_CONCORDANCE.md#setup--seat-oralable-alone-first).

---

## Primary sources (bookmarked)

| Source | Location | Use |
|--------|----------|-----|
| **BLE Design Guide** (A001S1M40A-DG-23-1) | Local NotebookLM: `/Users/johnacogan67/Library/CloudStorage/GoogleDrive-johnacogan67@gmail.com/My Drive/notebook_lm/Sources/BLE_DesignGuide.pdf` · also [ANR Documentation](https://www.anrcorp.com/documentation/) | UUID/rate truth: company `0x05DA`, Analog `0x2A58` uint16 0–1023 @ 100 ms notify, Digital ID `0x2A56` 1–24, Battery `0x2A19` |
| **iPhone App** (user guide / product page) | [anrcorp.com/iphoneapp](https://www.anrcorp.com/iphoneapp/) | Graph, log, export, biofeedback; up to 6 M40s; iOS 14+ — QC / ANR-only capture, not Dual A concordance |
| Documentation hub | [anrcorp.com/documentation](https://www.anrcorp.com/documentation/) | M40 product sheet, Android/iPhone guides, BLE guide |
| nRF Connect (iOS/desktop) | Nordic Semiconductor app | **Oralable:** primary BLE validation reference. **ANR:** optional GATT/hex inspect only — not ANR’s product reference (use BLE Design Guide and ANR iPhone app) |

### BLE validation reference (locked)

| Device | Reference of truth | nRF Connect role |
|--------|--------------------|------------------|
| Oralable | Firmware GATT + nRF Connect checklist | Primary |
| ANR M40 | BLE Design Guide + [ANR iPhone app](https://www.anrcorp.com/iphoneapp/) | Optional inspect |

| Script / path | Purpose |
|---------------|---------|
| `scripts/run_anr_emg_session.py` | Bleak ANR EMG logger → `data/raw/ANR_EMG_*.txt` |
| `scripts/run_dual_protocol_a_session.py` | Oralable + ANR Protocol A (EMG gate ≥70 + IR optical gate + SpO₂ AC WARN) |
| `scripts/align_anr_oralable_concordance.py` | Align @ 50 Hz → LP IR-DC/EMG F1 + **SpO₂∩EMG nest** (`spo2_qc`) + **`session.edf` (EMG)** → `plots/concordance/<session>/` |
| `scripts/export_dual_a_edf.py` | Convenience Dual A → pack including research EDF+ |
| `src/analysis/emg_spo2_nest.py` | Desat ≥10 s @ SpO₂ &lt; 90%; EMG∩desat; `spo2_qc` ok/warn/fail (not finger SpO₂) |
| iOS `showDualProtocolA` | Developer Settings Dual A + Share **4 files** incl. `session.edf` with ANR EMG (default **OFF**) |
| iOS `DUAL_PAIR` meta | Optional `skin_temp_mean_c` / `on_skin_fraction` when Oralable temp streams |

### SpO₂ ∩ EMG nest + EDF (pack **1.1.68**; first measured 12 Aug)

Mac align runs `ClinicalBiometricSuite` SpO₂ / SASHB on the Oralable 50 Hz frame and nests it with ANR EMG bouts:

- `desat_event_count` / `desat_events_per_hour` — descriptive (AcuPebble-style **label**, not claimed ODI/AHI)
- `emg_bouts_with_desat`, `frac_emg_bouts_with_desat`
- Overlay rows: **LP IR-DC** · EMG · SpO₂ · labels; `emg_ir_lag_zoom.png`
- `spo2_qc` — handoff visibility (`warn` OK for muscle Dual A; `fail` = flat AC)
- **`session.edf`** — research EDF+ with ANR `EMG` when Dual A; **not** PSG

**Measured precursor:** `plots/concordance/20260812_085110/` — SpO₂ yes; SASHB ≈ 929 %·s (SpO₂&lt;90 AUC — **not** Azarbarzin HB); **`spo2_qc=warn`**; median EMG→IR-DC lag ≈ 4.9 s; F1 vs Protocol A labels = 0 this pack (QC / placement).

**Claim discipline:** Concordance ≠ SB diagnosis; PSG-AV remains diagnostic gold standard. Nest ≠ AcuPebble AHI. Nest ≠ Bruxoff/GrindCare equivalence. Cite ANR docs; do not imply partnership unless contracted.
