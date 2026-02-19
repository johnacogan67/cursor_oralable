import os

# Define the project structure
folders = [
    "data/raw",         # Raw .txt/csv logs from nrfConnect
    "data/processed",   # 50Hz normalized CSVs
    "data/datasets",    # Labeled training sets
    "src/parser",       # HEX to Decimal logic
    "src/processing",   # Filters and 50Hz resampling
    "src/analysis",     # HRV and Muscle Occlusion algorithms
    "notebooks",        # For signal visualization
    "models/coreml",    # Exported iOS models
    "tests"             # Unit tests for signal math
]

files = {
    "requirements.txt": "pandas\nnumpy\nscipy\nmatplotlib\ncoremltools\nscikit-learn\n",
    "README.md": "# Oralable MAM Signal Processing\n\n## Data Flow\n1. Place nrfConnect logs in `data/raw/`.\n2. Run `src/parser/log_parser.py` to convert HEX to initial CSV.\n3. Run `src/processing/resampler.py` to normalize to 50Hz.\n",
    ".gitignore": "data/\nmodels/*.pth\n__pycache__/\n.DS_Store\n"
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, ".keep"), "w") as f:
        f.write("")

# Create initial files
for filename, content in files.items():
    with open(filename, "w") as f:
        f.write(content)

print("✅ Project Scaffolding Complete!")