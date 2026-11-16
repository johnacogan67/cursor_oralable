from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "oralable_app_services_v1.json"
VERIFIER = ROOT / "scripts" / "verify_app_service_contract.py"


def test_contract_declares_two_apps_six_products_and_migration_fields() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert payload["apps"]["consumer"]["bundle_identifier"] == "com.jacdental.oralable"
    assert payload["apps"]["professional"]["bundle_identifier"] == "com.jacdental.oralable.professionals"
    products = {
        *payload["storekit"]["consumer"]["products"].values(),
        *payload["storekit"]["professional"]["products"].values(),
    }
    assert len(products) == 6
    fields = payload["cloudkit"]["record_types"]["SharedPatientData"]["fields"]
    assert "professionalID" in fields
    assert "dentistID" in fields
    assert payload["healthkit"]["not_written"] == ["sleep stages", "HRV", "diagnosis", "treatment recommendation"]


def test_offline_app_service_verifier_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(VERIFIER), "--strict"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  app-service contract" in result.stdout
