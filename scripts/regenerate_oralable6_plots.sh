#!/bin/bash
# Regenerate all Oralable_6 validation plots with cheek coupling config (R_G_IR, 10M-70M).
# Run from project root: ./scripts/regenerate_oralable6_plots.sh

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/data/raw/Oralable_6.txt"
PLOTS="$ROOT/data/plots"
ED="$PLOTS/ed_presentation"

if [ ! -f "$LOG" ]; then
  echo "Error: $LOG not found"
  exit 1
fi

PYTHON=".venv/bin/python"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_oralable}"

echo "=== Oralable_6 Validation (cheek coupling: R_G_IR, IR-DC 10M-70M) ==="

# 1. Full session validation
echo ""
echo "1. Full session validation..."
$PYTHON -m src.validation.self_validate "$LOG" -o "$PLOTS/self_validation.png"

# 2. From 2nd 3-tap sync
echo ""
echo "2. From 2nd 3-tap sync..."
$PYTHON -m src.validation.self_validate "$LOG" --segment-from 2 -o "$PLOTS/oralable6_from_sync2.png"

# 3. From 16 min (960s)
echo ""
echo "3. From 16 min..."
$PYTHON -m src.validation.self_validate "$LOG" --start-time 960 -o "$PLOTS/oralable6_from_16min.png"

# 4. From 3-tap sync ~5 min before end
echo ""
echo "4. From 3-tap sync ~5 min before end..."
$PYTHON -m src.validation.self_validate "$LOG" --segment-from-sync-near-end 5 -o "$PLOTS/oralable6_from_sync_5min_before_end.png"

# 5. Validation dashboard (from 2nd sync)
echo ""
echo "5. Validation dashboard..."
$PYTHON -c "
from pathlib import Path
from src.validation_dashboard import run_validation_dashboard
run_validation_dashboard(
    log_path=Path('data/raw/Oralable_6.txt'),
    output_path=Path('data/plots/validation_dashboard.png'),
    segment_from_sync=2,
)
"

# 6. Copy to ed_presentation
mkdir -p "$ED"
echo ""
echo "6. Copying to ed_presentation..."
cp -f "$PLOTS/self_validation.png" "$ED/self_validation.png"
cp -f "$PLOTS/oralable6_from_sync2.png" "$ED/oralable6_from_sync2.png"
cp -f "$PLOTS/oralable6_from_16min.png" "$ED/oralable6_from_16min.png"
cp -f "$PLOTS/oralable6_from_sync_5min_before_end.png" "$ED/oralable6_from_sync_5min_before_end.png"
cp -f "$PLOTS/validation_dashboard.png" "$ED/validation_dashboard.png"

echo ""
echo "Done. Plots in $PLOTS and $ED"
echo ""
echo "Note: Segment from 16 min may show coupling OUTSIDE (tail has lower IR)."
echo "      Full session and from 2nd sync use cheek config: R_G_IR, 10M-70M."
