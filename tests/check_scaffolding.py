#!/usr/bin/env python3
"""
Temporary scaffolding check: data/raw contents, pandas/scipy availability, project structure.
"""
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    datasets_dir = root / "data" / "datasets"
    src_parser = root / "src" / "parser"
    src_processing = root / "src" / "processing"
    src_analysis = root / "src" / "analysis"
    notebooks_dir = root / "notebooks"
    models_coreml = root / "models" / "coreml"
    tests_dir = root / "tests"

    structure = [
        ("data/raw", raw_dir),
        ("data/processed", processed_dir),
        ("data/datasets", datasets_dir),
        ("src/parser", src_parser),
        ("src/processing", src_processing),
        ("src/analysis", src_analysis),
        ("notebooks", notebooks_dir),
        ("models/coreml", models_coreml),
        ("tests", tests_dir),
    ]

    print("=== Project structure ===")
    for name, path in structure:
        exists = path.exists()
        kind = "dir" if path.is_dir() else "file" if path.exists() else "missing"
        print(f"  {name}: {'OK' if exists else 'MISSING'} ({kind})")

    print("\n=== data/raw/ contents ===")
    if not raw_dir.exists():
        print("  data/raw/ does not exist.")
    else:
        entries = [p for p in raw_dir.iterdir() if p.name != ".keep"]
        if not entries:
            print("  (no files; add nRF Connect logs here)")
        else:
            for p in sorted(entries):
                print(f"  {p.name}")

    print("\n=== Dependencies ===")
    for pkg in ("pandas", "scipy"):
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  {pkg}: OK (version {ver})")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED")

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
