"""Tests for chat_stream_timeout_seconds API round-trip and clamping."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock


def _auth_route_endpoint(path: str, method: str):
    """Find a route endpoint by path and HTTP method on the auth router."""
    from routes.auth_routes import setup_auth_routes

    auth_manager = MagicMock()
    router = setup_auth_routes(auth_manager)
    for route in router.routes:
        if getattr(route, "path", "") == path and method in getattr(route, "methods", set()):
            return auth_manager, route.endpoint
    raise AssertionError(f"{method} {path} route not registered")


def _fake_auth_request(token="session-token"):
    from routes.auth_routes import SESSION_COOKIE

    req = SimpleNamespace()
    req.cookies = {SESSION_COOKIE: token}
    req.client = SimpleNamespace(host="127.0.0.1")
    return req


# ---------------------------------------------------------------------------
# GET /api/auth/settings — must include chat_stream_timeout_seconds
# ---------------------------------------------------------------------------

def test_get_settings_returns_chat_stream_timeout_seconds():
    """GET /settings should expose chat_stream_timeout_seconds."""
    auth, target = _auth_route_endpoint("/api/auth/settings", "GET")
    auth.get_username_for_token.return_value = "admin"
    auth.is_admin.return_value = True

    request = _fake_auth_request()
    out = asyncio.run(target(request))

    assert "chat_stream_timeout_seconds" in out, \
        "GET /settings must include chat_stream_timeout_seconds"


def test_get_settings_returns_default_120(tmp_path):
    """GET /settings should return 120 as the default (with empty settings file)."""
    import src.settings as settings_mod
    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{}", encoding="utf-8")

    original = settings_mod.SETTINGS_FILE
    settings_mod.SETTINGS_FILE = str(settings_file)
    settings_mod._invalidate_caches()

    try:
        auth, target = _auth_route_endpoint("/api/auth/settings", "GET")
        auth.get_username_for_token.return_value = "admin"
        auth.is_admin.return_value = True

        request = _fake_auth_request()
        out = asyncio.run(target(request))

        assert out["chat_stream_timeout_seconds"] == 120, \
            "Default should be 120 seconds"
    finally:
        settings_mod.SETTINGS_FILE = original
        settings_mod._invalidate_caches()


# ---------------------------------------------------------------------------
# POST /api/auth/settings — clamping and persistence
# ---------------------------------------------------------------------------

class _JsonRequest:
    """Minimal mock of a FastAPI Request with an async json() method."""
    def __init__(self, data):
        self.cookies = {"session-token": "x"}
        self.app = SimpleNamespace(state=SimpleNamespace())
        self._data = data

    async def json(self):
        return self._data


def _post_settings(body_dict):
    """Helper: call POST /api/auth/settings with the given body dict."""
    auth, target = _auth_route_endpoint("/api/auth/settings", "POST")
    auth.get_username_for_token.return_value = "admin"
    auth.is_admin.return_value = True
    req = _JsonRequest(body_dict)
    return asyncio.run(target(req))


def test_post_clamps_below_minimum():
    """Values below the minimum (10) should be clamped to 10."""
    import src.settings as settings_mod
    store = {}
    settings_mod._invalidate_caches()
    original_load = settings_mod.load_settings
    original_save = settings_mod.save_settings

    def fake_load():
        return dict(store) if store else dict(settings_mod.DEFAULT_SETTINGS)
    def fake_save(d):
        store.clear()
        store.update(d)

    settings_mod.load_settings = fake_load
    settings_mod.save_settings = fake_save

    try:
        out = _post_settings({"chat_stream_timeout_seconds": 3})
        assert out["chat_stream_timeout_seconds"] == 10, \
            f"Expected clamped to 10, got {out['chat_stream_timeout_seconds']}"
    finally:
        settings_mod.load_settings = original_load
        settings_mod.save_settings = original_save


def test_post_clamps_above_maximum():
    """Values above the maximum (86400) should be clamped to 86400."""
    import src.settings as settings_mod
    store = {}
    settings_mod._invalidate_caches()
    original_load = settings_mod.load_settings
    original_save = settings_mod.save_settings

    def fake_load():
        return dict(store) if store else dict(settings_mod.DEFAULT_SETTINGS)
    def fake_save(d):
        store.clear()
        store.update(d)

    settings_mod.load_settings = fake_load
    settings_mod.save_settings = fake_save

    try:
        out = _post_settings({"chat_stream_timeout_seconds": 100000})
        assert out["chat_stream_timeout_seconds"] == 86400, \
            f"Expected clamped to 86400, got {out['chat_stream_timeout_seconds']}"
    finally:
        settings_mod.load_settings = original_load
        settings_mod.save_settings = original_save


def test_post_persists_valid_value():
    """Valid values should be persisted without clamping."""
    import src.settings as settings_mod
    store = {}
    settings_mod._invalidate_caches()
    original_load = settings_mod.load_settings
    original_save = settings_mod.save_settings

    def fake_load():
        return dict(store) if store else dict(settings_mod.DEFAULT_SETTINGS)
    def fake_save(d):
        store.clear()
        store.update(d)

    settings_mod.load_settings = fake_load
    settings_mod.save_settings = fake_save

    try:
        out = _post_settings({"chat_stream_timeout_seconds": 300})
        assert out["chat_stream_timeout_seconds"] == 300, \
            f"Expected 300, got {out['chat_stream_timeout_seconds']}"
    finally:
        settings_mod.load_settings = original_load
        settings_mod.save_settings = original_save


def test_post_boundary_value_at_minimum():
    """Value exactly at the minimum (10) should not be clamped."""
    import src.settings as settings_mod
    store = {}
    settings_mod._invalidate_caches()
    original_load = settings_mod.load_settings
    original_save = settings_mod.save_settings

    def fake_load():
        return dict(store) if store else dict(settings_mod.DEFAULT_SETTINGS)
    def fake_save(d):
        store.clear()
        store.update(d)

    settings_mod.load_settings = fake_load
    settings_mod.save_settings = fake_save

    try:
        out = _post_settings({"chat_stream_timeout_seconds": 10})
        assert out["chat_stream_timeout_seconds"] == 10, \
            f"Expected 10 at boundary, got {out['chat_stream_timeout_seconds']}"
    finally:
        settings_mod.load_settings = original_load
        settings_mod.save_settings = original_save


def test_post_boundary_value_at_maximum():
    """Value exactly at the maximum (86400) should not be clamped."""
    import src.settings as settings_mod
    store = {}
    settings_mod._invalidate_caches()
    original_load = settings_mod.load_settings
    original_save = settings_mod.save_settings

    def fake_load():
        return dict(store) if store else dict(settings_mod.DEFAULT_SETTINGS)
    def fake_save(d):
        store.clear()
        store.update(d)

    settings_mod.load_settings = fake_load
    settings_mod.save_settings = fake_save

    try:
        out = _post_settings({"chat_stream_timeout_seconds": 86400})
        assert out["chat_stream_timeout_seconds"] == 86400, \
            f"Expected 86400 at boundary, got {out['chat_stream_timeout_seconds']}"
    finally:
        settings_mod.load_settings = original_load
        settings_mod.save_settings = original_save
