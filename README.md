# Oralable MAM Signal Processing

## Setup (pandas)

A virtual environment with pandas is in `.venv`. To use it:

```bash
source .venv/bin/activate   # or on Windows: .venv\Scripts\activate
# Or run directly:
.venv/bin/python src/parser/log_parser.py "data/raw/text.txt"
```

To create the venv and install dependencies (if needed):

```bash
python3 -m venv .venv
.venv/bin/pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## Data Flow
1. Place nrfConnect logs in `data/raw/`.
2. Run `src/parser/log_parser.py` to convert HEX to initial CSV.
3. Run `src/processing/resampler.py` to normalize to 50Hz.
