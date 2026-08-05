import pytest

from huggingface.upload_dataset import resolve_privacy


def _cfg(private_by_default: bool, permission_notes: str = "") -> dict:
    return {
        "dataset": {
            "private_by_default": private_by_default,
            "PERMISSION_NOTES": permission_notes,
        }
    }


def test_stays_private_with_no_permission(monkeypatch):
    monkeypatch.delenv("I_HAVE_REDISTRIBUTION_PERMISSION", raising=False)
    assert resolve_privacy(_cfg(True), i_have_permission=False) is True


def test_stays_private_with_only_cli_flag(monkeypatch):
    """--i-have-permission alone isn't enough -- also need the env flag and notes."""
    monkeypatch.delenv("I_HAVE_REDISTRIBUTION_PERMISSION", raising=False)
    assert resolve_privacy(_cfg(True), i_have_permission=True) is True


def test_goes_public_with_full_documented_permission(monkeypatch):
    monkeypatch.setenv("I_HAVE_REDISTRIBUTION_PERMISSION", "true")
    cfg = _cfg(True, permission_notes="Written permission from X on 2026-01-01.")
    assert resolve_privacy(cfg, i_have_permission=True) is False


def test_private_by_default_false_still_requires_gate(monkeypatch):
    monkeypatch.delenv("I_HAVE_REDISTRIBUTION_PERMISSION", raising=False)
    with pytest.raises(SystemExit):
        resolve_privacy(_cfg(False), i_have_permission=False)
