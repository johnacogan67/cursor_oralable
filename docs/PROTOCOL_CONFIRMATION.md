# Protocol Confirmation — 1st 3-Tap Sync

**Confirmed:** All validation runs use **T=0 = 1st 3-tap sync**.

---

## How It Works

| Component | T=0 Definition | How Set |
|-----------|----------------|--------|
| **self_validate.py** | 1st sync | `--segment-from 1` → `t0_s = start_s` (1st anchor) |
| **validation_dashboard** | 1st sync | `segment_from_sync=1` → `anchors[0]` (1st anchor) |
| **clinical_summary** | 1st sync | Segments from 1st anchor via `find_all_three_tap_anchors` |
| **Protocol CSV** | `JOHN_COGAN_1ST_SYNC_PROTOCOL.csv` | Header: "T=0 = 1st 3-Tap Sync" |

---

## Run Commands (1st sync)

```bash
# Self-validation
python -m src.validation.self_validate data/raw/Oralable_20260304_090927.txt --segment-from 1 -o data/plots/self_validation_from_sync1.png

# Validation (auto-loads JOHN_COGAN_1ST_SYNC_PROTOCOL.csv when segment_from_sync=1)
python -c "
from pathlib import Path
from src.validation_dashboard import run_validation_dashboard
run_validation_dashboard(
    log_path=Path('data/raw/Oralable_20260304_090927.txt'),
    segment_from_sync=1,
    output_path=Path('data/plots/validation_dashboard_sync1.png'),
)
"
```

---

## Protocol Phases (T=0 = 1st 3-tap sync)

| Phase | ElapsedSeconds | Action |
|-------|----------------|--------|
| 0 | 0–5 | 3-Tap Sync |
| 1 | 30–45 | Max Tonic Clench |
| 2 | 45–60 | Rest |
| 3 | 60–105 | Phasic Grinding |
| 4 | 105–120 | Rest |
| 5 | 120–135 | Swallow/Control |
| 6 | 150–195 | Simulated Apnea |
| 7 | 210–270 | Natural Speech |

---

## Files

- **Protocol:** `data/validation_logs/JOHN_COGAN_1ST_SYNC_PROTOCOL.csv`
- **Plots:** `data/plots/ed_presentation/` (all from 1st sync)
