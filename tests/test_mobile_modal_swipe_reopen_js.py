"""
Mobile modal swipe-dismiss-then-reopen bug (tasks / gallery / research).

Bug: after swiping a tool modal down to dismiss it, the FIRST tap on the same
tool tab fails to reopen it; the second tap works.

Cause: ui.js swipe-dismiss only adds `.hidden` and fires a `modal-dismissed`
event — it does NOT call the module's close function. Modules that track an
`_open` flag stay "open", so the next tab tap (a toggle) sees `isOpen()==true`
and calls close() instead of open(), spending the first tap.

Fix: each affected module listens for `modal-dismissed` (guarded on its own
modal id) and resets `_open=false`.

These are lightweight source-level checks (matching the repo's *_js.py
convention); behavioural coverage lives in tests/e2e/demo-reopen.spec.js.
"""

from pathlib import Path

import pytest

_JS = Path(__file__).resolve().parent.parent / "static" / "js"

# (js file, modal id, close fn whose body must still clear _open)
FIXED_MODULES = [
    ("tasks.js", "tasks-modal", "closeTasks"),
    ("gallery.js", "gallery-modal", "_doCloseGallery"),
    ("research/panel.js", "research-overlay", "closePanel"),
]


def _read(rel: str) -> str:
    return (_JS / rel).read_text(encoding="utf-8")


def _has_dismiss_listener_for(src: str, modal_id: str) -> bool:
    """True if a `modal-dismissed` listener guarded on `modal_id` is present."""
    i = src.find("modal-dismissed")
    while i != -1:
        if modal_id in src[i:i + 300]:
            return True
        i = src.find("modal-dismissed", i + 1)
    return False


def test_ui_dispatches_modal_dismissed_on_swipe():
    """The swipe handler must fire `modal-dismissed` for modules to react to."""
    src = _read("ui.js")
    assert "modal-dismissed" in src and "CustomEvent" in src


@pytest.mark.parametrize("js_file, modal_id, close_fn", FIXED_MODULES)
def test_module_has_open_flag(js_file, modal_id, close_fn):
    """The `_open` flag is what gets stuck `true` after a swipe-dismiss."""
    assert "let _open = false" in _read(js_file)


@pytest.mark.parametrize("js_file, modal_id, close_fn", FIXED_MODULES)
def test_module_resets_open_on_swipe_dismiss(js_file, modal_id, close_fn):
    """The fix: a `modal-dismissed` listener guarded on this modal id."""
    assert _has_dismiss_listener_for(_read(js_file), modal_id), (
        f"{js_file} must listen for 'modal-dismissed' guarded on '{modal_id}' "
        "and reset _open; without it the first re-tap after a swipe toggles "
        "the tool closed instead of reopening it."
    )


@pytest.mark.parametrize("js_file, modal_id, close_fn", FIXED_MODULES)
def test_close_fn_still_clears_open(js_file, modal_id, close_fn):
    """Normal close paths (button / click-outside / Esc) must still clear _open."""
    src = _read(js_file)
    idx = src.find(f"function {close_fn}(")
    assert idx != -1, f"{close_fn}() not found in {js_file}"
    assert "_open = false" in src[idx:idx + 600], \
        f"{close_fn}() must set _open=false"
