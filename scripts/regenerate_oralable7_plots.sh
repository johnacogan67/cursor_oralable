#!/bin/bash
# Regenerate all Oralable_7 validation plots.
# Oralable_7 = Oralable_6 trimmed from 2nd 3-tap sync onward.
# Run from project root: ./scripts/regenerate_oralable7_plots.sh

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/data/raw/Oralable_7.txt"
PLOTS="$ROOT/data/plots"

if [ ! -f "$LOG" ]; then
  echo "Error: $LOG not found. Run: python scripts/trim_log_to_sync.py data/raw/Oralable_6.txt data/raw/Oralable_7.txt --from-sync 2"
  exit 1
fi

PYTHON=".venv/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_oralable}"

echo "=== Oralable_7 Validation (trimmed from 2nd sync) ==="

# 1. Full session validation
echo ""
echo "1. Full session validation..."
$PYTHON -m src.validation.self_validate "$LOG" -o "$PLOTS/oralable7_self_validation.png"

# 2. From 1st sync (original 2nd sync = protocol anchor)
echo ""
echo "2. From 1st sync (protocol anchor)..."
$PYTHON -m src.validation.self_validate "$LOG" --segment-from 1 -o "$PLOTS/oralable7_from_sync1.png"

# 3. From 16 min
echo ""
echo "3. From 16 min..."
$PYTHON -m src.validation.self_validate "$LOG" --start-time 960 -o "$PLOTS/oralable7_from_16min.png"

# 4. From 3-tap sync ~5 min before end
echo ""
echo "4. From 3-tap sync ~5 min before end..."
$PYTHON -m src.validation.self_validate "$LOG" --segment-from-sync-near-end 5 -o "$PLOTS/oralable7_from_sync_5min_before_end.png"

# 5. Validation dashboard (from 1st sync)
echo ""
echo "5. Validation dashboard..."
$PYTHON -c "
from pathlib import Path
from src.validation_dashboard import run_validation_dashboard
run_validation_dashboard(
    log_path=Path('data/raw/Oralable_7.txt'),
    output_path=Path('data/plots/oralable7_validation_dashboard.png'),
    segment_from_sync=1,
)
"

# 6. Copy to ed_presentation for Ed & Pedro
ED7="$PLOTS/ed_presentation/oralable7"
mkdir -p "$ED7"
echo ""
echo "6. Copying to ed_presentation/oralable7..."
cp -f "$PLOTS/oralable7_self_validation.png" "$ED7/"
cp -f "$PLOTS/oralable7_from_sync1.png" "$ED7/"
cp -f "$PLOTS/oralable7_from_16min.png" "$ED7/"
cp -f "$PLOTS/oralable7_from_sync_5min_before_end.png" "$ED7/"
cp -f "$PLOTS/oralable7_validation_dashboard.png" "$ED7/"

echo ""
echo "Done. Plots in $PLOTS and $ED7"
