"""Last MAM connection sidecar — firmware survives a brief / failed session."""

from pathlib import Path

from src.utils import mam_last_connection as mlc


def test_firmware_log_header_uses_persisted_firmware(tmp_path, monkeypatch):
    store = tmp_path / "last_mam_connection.json"
    monkeypatch.setattr(mlc, "STORE_PATH", store)

    mlc.save_last_connection(firmware="1.0.79", address="AA-BB")
    header = mlc.firmware_log_header(None, address=None)
    assert header.startswith("# MAM firmware 1.0.79")
    assert "AA-BB" in header


def test_failed_read_keeps_previous_firmware(tmp_path, monkeypatch):
    store = tmp_path / "last_mam_connection.json"
    monkeypatch.setattr(mlc, "STORE_PATH", store)

    mlc.save_last_connection(firmware="1.0.79", address="old")
    mlc.save_last_connection(firmware=None, address="new")
    data = mlc.load_last_connection()
    assert data is not None
    assert data["firmware"] == "1.0.79"
    assert data.get("firmware_source") == "previous"
    assert "1.0.79" in mlc.format_last_known()
    assert mlc.known_firmware() == "1.0.79"


def test_header_prefers_live_read(tmp_path, monkeypatch):
    store = tmp_path / "last_mam_connection.json"
    monkeypatch.setattr(mlc, "STORE_PATH", store)
    mlc.save_last_connection(firmware="1.0.70", address="x")
    header = mlc.firmware_log_header("1.0.79", address="y")
    assert "# MAM firmware 1.0.79 address=y" in header
