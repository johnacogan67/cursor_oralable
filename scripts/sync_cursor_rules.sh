#!/usr/bin/env bash
# Sync always-on Cursor rules from cursor_oralable into sibling repos.
# Edit rules only in cursor_oralable/.cursor/rules/, then run this script.
set -euo pipefail
SRC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$SRC_ROOT/.cursor/rules"
RULES=(bookmark-sources.mdc plan-mode-switch.mdc prose-orwell.mdc workspace-topics.mdc)

for dest in oralable_nrf oralable_swift OralableCore; do
  D="$SRC_ROOT/../$dest/.cursor/rules"
  mkdir -p "$D"
  for r in "${RULES[@]}"; do
    cp "$SRC/$r" "$D/$r"
    echo "synced $dest/$r"
  done
done
echo "Done. Firmware-only nrf-connect-validation.mdc is not copied (stays in oralable_nrf)."
