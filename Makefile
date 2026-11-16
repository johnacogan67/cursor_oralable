PYTHON ?= python3
VENV ?= .venv

.PHONY: bootstrap test verify verify-fast

bootstrap:
	PYTHON_BIN="$(PYTHON)" VENV_DIR="$(VENV)" ./scripts/bootstrap_python.sh

test:
	$(VENV)/bin/python -m pytest -q

verify:
	$(VENV)/bin/python scripts/verify_research_checkout.py

verify-fast:
	$(VENV)/bin/python scripts/verify_research_checkout.py --skip-tests
