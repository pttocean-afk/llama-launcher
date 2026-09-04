"""Tests for the thinking-intensity (reasoning effort) feature.

Covers:
- argv builder: --reasoning / --reasoning-effort flag logic
- main-screen THINKING INTENSITY control (sync + enable/disable)
- SettingsDialog (per-model settings) effort persistence
- AddModelDialog effort default
"""
import threading

import pytest


from conftest import require_tk

def _import_app(tmp_path, monkeypatch):
    """Import llama_launcher.app with an isolated data and model dir
    (same isolation as tests/test_app_security.py)."""
    monkeypatch.setenv("LLAMA_LAUNCHER_DATA_DIR", str(tmp_path / "data"))
    models = tmp_path / "llama" / "models"
    models.mkdir(parents=True)
    (models / "a.gguf").write_bytes(b"")
    monkeypatch.setattr("llama_launcher.app.MODELS_DIR", models)
    monkeypatch.setattr("llama_launcher.app.invalidate_model_inventory",
                        lambda: None)
    monkeypatch.setattr("llama_launcher.app.llama_server_filename",
                        lambda: "llama-server")
    monkeypatch.setattr("llama_launcher.app.LLAMA_SERVER", models / "llama-server")
    monkeypatch.setattr("llama_launcher.app.VULKAN_SERVER", models / "llama-server")
    import llama_launcher.app as app

    return app


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    return _import_app(tmp_path, monkeypatch)


def _base_profile(**kw):
    p = {
        "name": "A", "model": "a.gguf", "mmproj": "",
        "vision_enabled": False, "default_ctx": 131072,
        "reasoning": "off", "reasoning_effort": "default",
        "gpu_split": "", "backend": "cuda", "starred": True,
        "favorite_order": 0,
        "extra_args": "-ctk q4_0 -ctv q4_0 --parallel 1",
    }
    p.update(kw)
    return p


# ---------------------------------------------------------------------------
# argv builder (pure functions, no display needed)
# ---------------------------------------------------------------------------

def test_reasoning_off_omits_effort_flag(app_mod):
    args = app_mod.build_reasoning_args(
        {"reasoning": "off", "reasoning_effort": "high"})
    assert args == ["--reasoning", "off"]


def test_reasoning_on_default_omits_effort_flag(app_mod):
    assert app_mod.build_reasoning_args({"reasoning": "on"}) == \
        ["--reasoning", "on"]
    assert app_mod.build_reasoning_args(
        {"reasoning": "on", "reasoning_effort": "default"}) == \
        ["--reasoning", "on"]


def test_reasoning_on_with_effort_appends_flag(app_mod):
    for level in ("minimal", "low", "medium", "high", "xhigh", "max"):
        assert app_mod.build_reasoning_args(
            {"reasoning": "on", "reasoning_effort": level}) == \
            ["--reasoning", "on", "--reasoning-effort", level]


def test_unknown_effort_treated_as_default(app_mod):
    assert app_mod.build_reasoning_args(
        {"reasoning": "on", "reasoning_effort": "ultra"}) == \
        ["--reasoning", "on"]
    # case/whitespace are normalized
    assert app_mod.build_reasoning_args(
        {"reasoning": "on", "reasoning_effort": " High "}) == \
        ["--reasoning", "on", "--reasoning-effort", "high"]
    assert app_mod.reasoning_effort_value(
        {"reasoning_effort": " High "}) == "high"


def test_reasoning_effort_value_defaults(app_mod):
    assert app_mod.reasoning_effort_value({}) == "default"
    assert app_mod.reasoning_effort_value({"reasoning_effort": ""}) == "default"
    assert app_mod.reasoning_effort_value(
        {"reasoning_effort": "bogus"}) == "default"
    assert app_mod.reasoning_effort_value(
        {"reasoning_effort": "medium"}) == "medium"


def test_default_profile_has_effort_default(app_mod):
    assert app_mod.DEFAULT_PROFILE["reasoning_effort"] == "default"
    assert app_mod.REASONING_EFFORTS[0] == "default"


def test_remote_profiles_include_reasoning_effort(app_mod):
    app = app_mod.LauncherApp.__new__(app_mod.LauncherApp)
    app.profiles = [
        _base_profile(reasoning="on", reasoning_effort="xhigh"),
        _base_profile(reasoning="off", model="b.gguf", name="B"),
    ]
    out = app.remote_profiles()
    by_model = {p["model"]: p for p in out}
    assert by_model["a.gguf"]["reasoning_effort"] == "xhigh"
    assert by_model["b.gguf"]["reasoning_effort"] == "default"


def test_dashboard_html_shows_effort_only_when_relevant(app_mod):
    html = app_mod.REMOTE_CONTROL_HTML
    # effort is appended only when reasoning is on and effort is not default
    assert ("if(x.reasoning==='on'&&x.reasoning_effort"
            "&&x.reasoning_effort!=='default')") in html
    assert "effort '+x.reasoning_effort" in html


# ---------------------------------------------------------------------------
# UI (display-bound; run under xvfb)
# ---------------------------------------------------------------------------

def _combo_state(widget):
    """ttk state option is list-valued; cget returns an index object."""
    return str(widget.cget("state"))


def _full_app(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(app_mod.messagebox, "showwarning", lambda *a, **k: None)
    monkeypatch.setattr(app_mod.messagebox, "showerror", lambda *a, **k: None)
    root = require_tk(app_mod.tk)
    app = app_mod.LauncherApp(root)
    return root, app


def test_main_screen_effort_control_syncs(app_mod, monkeypatch):
    root, app = _full_app(app_mod, monkeypatch)
    try:
        # THINKING dropdown leads with off, then the effort levels
        values = tuple(app.effort_combo["values"])
        assert values[0] == "off"
        assert set(values) == {"off", "default", "minimal", "low",
                               "medium", "high", "xhigh", "max"}
        # reasoning on + effort -> dropdown shows the effort level
        profile = _base_profile(reasoning="on", reasoning_effort="high")
        app.cfg["profiles"] = [dict(profile)]
        app.profiles = app_mod.merge_profiles(app.cfg)
        app.refresh_listbox()
        app.update_detail()
        assert app.effort_var.get() == "high"
        # reasoning off -> dropdown shows off (stored effort ignored)
        profile = _base_profile(reasoning="off", reasoning_effort="high")
        app.cfg["profiles"] = [dict(profile)]
        app.profiles = app_mod.merge_profiles(app.cfg)
        app.refresh_listbox()
        app.update_detail()
        assert app.effort_var.get() == "off"
        # unknown/legacy effort values normalize to default (thinking on)
        profile = _base_profile(reasoning="on", reasoning_effort="weird")
        app.cfg["profiles"] = [dict(profile)]
        app.profiles = app_mod.merge_profiles(app.cfg)
        app.refresh_listbox()
        app.update_detail()
        assert app.effort_var.get() == "default"
    finally:
        if app.control_server is not None:
            app.control_server.close()
        root.destroy()


def _fake_app(app_mod, profile):
    """Minimal app stand-in for dialog tests (no Tk root needed)."""
    app = app_mod.LauncherApp.__new__(app_mod.LauncherApp)
    app.cfg = {"profiles": [dict(profile)]}
    app.profiles = list(app_mod.merge_profiles(app.cfg))
    app.server_lock = threading.Lock()
    app.refresh_listbox = lambda *a, **k: None
    app.update_detail = lambda *a, **k: None
    return app


def test_settings_dialog_saves_reasoning_effort(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(app_mod.messagebox, "showerror", lambda *a, **k: None)
    profile = _base_profile(reasoning="on")
    app = _fake_app(app_mod, profile)
    root = require_tk(app_mod.tk)
    try:
        dlg = app_mod.SettingsDialog(root, app, dict(profile))
        try:
            assert dlg.reasoning_effort_var.get() == "default"
            # thinking on -> effort combo editable
            assert _combo_state(dlg.reasoning_effort_combo) == "readonly"
            dlg.reasoning_effort_var.set("xhigh")
            dlg.save()
            saved = app.cfg["profiles"][0]
            assert saved["reasoning_effort"] == "xhigh"
            # saved profile round-trips through the merge
            assert any(p.get("reasoning_effort") == "xhigh"
                       for p in app.profiles)
        finally:
            if dlg.winfo_exists():
                dlg.destroy()
    finally:
        root.destroy()


def test_settings_dialog_disables_effort_when_thinking_off(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(app_mod.messagebox, "showerror", lambda *a, **k: None)
    profile = _base_profile(reasoning="off", reasoning_effort="high")
    app = _fake_app(app_mod, profile)
    root = require_tk(app_mod.tk)
    try:
        dlg = app_mod.SettingsDialog(root, app, dict(profile))
        try:
            # value kept from profile but combo disabled (thinking off)
            assert dlg.reasoning_effort_var.get() == "high"
            assert _combo_state(dlg.reasoning_effort_combo) == "disabled"
            dlg.reasoning_var.set("on")
            assert _combo_state(dlg.reasoning_effort_combo) == "readonly"
        finally:
            if dlg.winfo_exists():
                dlg.destroy()
    finally:
        root.destroy()


def test_add_model_dialog_defaults(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(app_mod.messagebox, "showwarning", lambda *a, **k: None)
    app = _fake_app(app_mod, _base_profile())
    root = require_tk(app_mod.tk)
    try:
        dlg = app_mod.AddModelDialog(root, app)
        try:
            assert dlg.reasoning_effort_var.get() == "default"
            assert _combo_state(dlg.reasoning_effort_combo) == "disabled"
            dlg.reasoning_var.set("on")
            assert _combo_state(dlg.reasoning_effort_combo) == "readonly"
        finally:
            if dlg.winfo_exists():
                dlg.destroy()
    finally:
        root.destroy()


def test_settings_preview_handles_invalid_context(app_mod):
    """欄位收集失敗時預覽應顯示提示，不得先對 None 呼叫 get。"""
    class Preview:
        def __init__(self):
            self.value = ""

        def winfo_exists(self):
            return True

        def config(self, **_kwargs):
            pass

        def delete(self, *_args):
            self.value = ""

        def insert(self, _where, text):
            self.value += text

    dialog = app_mod.SettingsDialog.__new__(app_mod.SettingsDialog)
    dialog.preview_text = Preview()
    dialog._collect_fields = lambda silent=False: None
    dialog._update_preview()
    assert "Context 欄位格式有誤" in dialog.preview_text.value
