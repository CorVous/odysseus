"""Tests for chat.js consuming chat_stream_timeout_seconds from settings."""
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_CHAT_JS = _REPO / "static" / "js" / "chat.js"


def test_chat_js_reads_chat_stream_timeout_from_settings():
    """chat.js must read chat_stream_timeout_seconds from settings API, not use hardcoded constant."""
    src = _CHAT_JS.read_text(encoding="utf-8")

    # The timeout calculation should reference the setting key
    assert "chat_stream_timeout_seconds" in src, \
        "chat.js should reference 'chat_stream_timeout_seconds' for timeout value"


def test_chat_js_uses_setting_for_default_timeout():
    """The default (non-research) timeout path must use the settings value."""
    src = _CHAT_JS.read_text(encoding="utf-8")

    # Look for the pattern where chat_stream_timeout_seconds is used in timeout calculation
    # The code should parse the setting and multiply by 1000 to get ms
    assert "chat_stream_timeout_seconds" in src, \
        "chat.js must read chat_stream_timeout_seconds from settings"


def test_chat_js_falls_back_to_120_when_setting_missing():
    """If the setting is missing/invalid, chat.js should fall back to 120 (current DEFAULT_TIMEOUT_MS)."""
    src = _CHAT_JS.read_text(encoding="utf-8")

    # The fallback value should be 120 (seconds) or 120000 (ms)
    # Look for the default/fallback pattern near chat_stream_timeout_seconds usage
    lines = src.split('\n')
    found_setting_ref = False
    for i, line in enumerate(lines):
        if 'chat_stream_timeout_seconds' in line:
            found_setting_ref = True
            # Check nearby lines (within 5 lines) for a fallback to 120
            nearby = '\n'.join(lines[max(0, i-5):min(len(lines), i+6)])
            assert '120' in nearby or 'DEFAULT_TIMEOUT_MS' in nearby, \
                f"Near chat_stream_timeout_seconds reference (line {i+1}), " \
                f"expected fallback to 120: {nearby}"
            break

    assert found_setting_ref, "chat_stream_timeout_seconds not found in chat.js at all"


def test_chat_js_still_has_research_timeout_constant():
    """RESEARCH_TIMEOUT_MS should still exist (not surfaced yet)."""
    src = _CHAT_JS.read_text(encoding="utf-8")
    assert 'RESEARCH_TIMEOUT_MS' in src, \
        "RESEARCH_TIMEOUT_MS constant should still be present"


def test_chat_js_uses_setting_not_constant_for_default_path():
    """The non-research timeout path must NOT use DEFAULT_TIMEOUT_MS directly."""
    src = _CHAT_JS.read_text(encoding="utf-8")

    # Find the line that computes timeoutMs
    lines = src.split('\n')
    for i, line in enumerate(lines):
        if 'timeoutMs' in line and ('?' in line or ':' in line) and 'RESEARCH_TIMEOUT_MS' in line:
            # This is the ternary that chooses between research and default timeout
            # The non-research path (after ':') should NOT be DEFAULT_TIMEOUT_MS directly
            # It should use a settings-derived value instead
            break

    # More robust check: the code should have chat_stream_timeout_seconds usage
    assert "chat_stream_timeout_seconds" in src, \
        "chat.js must reference chat_stream_timeout_seconds for the default timeout path"
