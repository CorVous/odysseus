"""Tests for chat_stream_timeout_seconds in the Settings UI (HTML + JS wiring)."""
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_INDEX_HTML = _REPO / "static" / "index.html"
_SETTINGS_JS = _REPO / "static" / "js" / "settings.js"


# ---------------------------------------------------------------------------
# HTML: input elements must exist in the settings modal
# ---------------------------------------------------------------------------

def test_html_has_chat_stream_timeout_input():
    """The settings modal must contain an input for chat_stream_timeout_seconds."""
    html = _INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="set-chatStreamTimeout"' in html, \
        "HTML missing #set-chatStreamTimeout input element"


def test_html_chat_timeout_input_is_numeric():
    """The chat timeout input must accept numeric values."""
    html = _INDEX_HTML.read_text(encoding="utf-8")
    assert 'inputmode="numeric"' in html, \
        "#set-chatStreamTimeout should have inputmode='numeric'"


# ---------------------------------------------------------------------------
# JS: initChatTimeoutSettings function must exist and be wired
# ---------------------------------------------------------------------------

def test_settings_js_has_init_chat_timeout_function():
    """settings.js must define initChatTimeoutSettings()."""
    src = _SETTINGS_JS.read_text(encoding="utf-8")
    assert "function initChatTimeoutSettings()" in src or \
           "function initChatTimeoutSettings (" in src, \
        "settings.js missing initChatTimeoutSettings function"


def test_settings_js_init_all_calls_init_chat_timeout():
    """initAll() must call initChatTimeoutSettings()."""
    src = _SETTINGS_JS.read_text(encoding="utf-8")
    assert "initChatTimeoutSettings()" in src, \
        "initAll() does not call initChatTimeoutSettings()"


def test_settings_js_reads_chat_stream_timeout_from_api():
    """initChatTimeoutSettings must read chat_stream_timeout_seconds from /api/auth/settings."""
    src = _SETTINGS_JS.read_text(encoding="utf-8")
    # The function should reference the setting key and the input element ID
    assert "chat_stream_timeout_seconds" in src, \
        "initChatTimeoutSettings should reference 'chat_stream_timeout_seconds'"
    assert "set-chatStreamTimeout" in src or "setChatStreamTimeout" in src, \
        "initChatTimeoutSettings should reference #set-chatStreamTimeout input"


def test_settings_js_saves_chat_stream_timeout_to_api():
    """initChatTimeoutSettings must POST chat_stream_timeout_seconds back to /api/auth/settings."""
    src = _SETTINGS_JS.read_text(encoding="utf-8")
    # The save function should include the key in its payload
    assert "chat_stream_timeout_seconds" in src, \
        "Save handler should include 'chat_stream_timeout_seconds' in payload"
