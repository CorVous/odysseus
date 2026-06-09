"""Tests for chat_stream_timeout_seconds in DEFAULT_SETTINGS."""
import pytest


def test_default_settings_has_chat_stream_timeout_seconds():
    """chat_stream_timeout_seconds must exist in DEFAULT_SETTINGS."""
    from src import settings
    assert "chat_stream_timeout_seconds" in settings.DEFAULT_SETTINGS, \
        "DEFAULT_SETTINGS missing 'chat_stream_timeout_seconds'"


def test_default_value_is_120():
    """Default should be 120 seconds (matching current DEFAULT_TIMEOUT_MS)."""
    from src import settings
    assert settings.DEFAULT_SETTINGS["chat_stream_timeout_seconds"] == 120, \
        "Expected default of 120 seconds"


def test_load_settings_returns_chat_stream_timeout(tmp_path, monkeypatch):
    """load_settings() should return the key with its default value."""
    from src import settings
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings, "SETTINGS_FILE", str(settings_file))
    settings._invalidate_caches()

    loaded = settings.load_settings()
    assert loaded["chat_stream_timeout_seconds"] == 120


def test_get_setting_returns_chat_stream_timeout(tmp_path, monkeypatch):
    """get_setting() should return the value for chat_stream_timeout_seconds."""
    from src import settings
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings, "SETTINGS_FILE", str(settings_file))
    settings._invalidate_caches()

    val = settings.get_setting("chat_stream_timeout_seconds", 60)
    assert val == 120


def test_save_and_reload_chat_stream_timeout(tmp_path, monkeypatch):
    """Saving a custom value should persist and reload correctly."""
    from src import settings
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings, "SETTINGS_FILE", str(settings_file))
    settings._invalidate_caches()

    # Save a custom value
    current = settings.load_settings()
    current["chat_stream_timeout_seconds"] = 300
    settings.save_settings(current)

    # Reload and verify
    settings._invalidate_caches()
    val = settings.get_setting("chat_stream_timeout_seconds", 60)
    assert val == 300


def test_is_setting_overridden_for_chat_stream_timeout(tmp_path, monkeypatch):
    """is_setting_overridden should reflect whether user changed the value."""
    from src import settings
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings, "SETTINGS_FILE", str(settings_file))
    settings._invalidate_caches()

    assert not settings.is_setting_overridden("chat_stream_timeout_seconds")

    current = settings.load_settings()
    current["chat_stream_timeout_seconds"] = 200
    settings.save_settings(current)

    settings._invalidate_caches()
    assert settings.is_setting_overridden("chat_stream_timeout_seconds")
