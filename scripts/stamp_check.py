#!/usr/bin/env python3
"""Fail if data_room pack stamp drifts from VERSION file."""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
ver = (root / "docs/data_room/VERSION").read_text().strip()
readme = (root / "docs/data_room/README.md").read_text()
hub = (root / "docs/VERSION").read_text().strip()
ok = True
if f"**{ver}**" not in readme and f"v{ver}" not in readme and ver not in readme.split("Pack:")[1][:40]:
    # soft: require pack string appears near top
    if ver not in readme[:500]:
        print(f"FAIL: data_room/README.md does not cite pack {ver} in header")
        ok = False
align = (root / "docs/data_room/VERSION_ALIGNMENT.md").read_text()
if ver not in align[:800]:
    print(f"FAIL: VERSION_ALIGNMENT.md header missing pack {ver}")
    ok = False
print(f"pack={ver} hub={hub}")
if not ok:
    sys.exit(1)
print("OK stamp check")
