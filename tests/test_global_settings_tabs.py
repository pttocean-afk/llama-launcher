"""GlobalSettingsDialog 分頁與欄位保存回歸測試。"""
from types import SimpleNamespace

import pytest

from conftest import require_tk


def _import_app(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_LAUNCHER_DATA_DIR", str(tmp_path / "data"))
    import llama_launcher.app as app
    monkeypatch.setattr(app, "SETTINGS_PATH", app.settings_path())
    monkeypatch.setattr(app, "API_KEY_PATH", app.api_key_path())
    monkeypatch.setattr(app, "autostart_enabled", lambda: False)
    monkeypatch.setattr(app, "set_autostart", lambda enabled: True)
    monkeypatch.setattr(app.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(app.messagebox, "showwarning", lambda *a, **k: None)
    return app


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    return _import_app(tmp_path, monkeypatch)


def _fake_app():
    return SimpleNamespace(
        on_remote_access=lambda: None,
        on_migrate_legacy=lambda: None,
        open_config=lambda: None,
        cfg={"profiles": []},
        profiles=[],
        refresh_listbox=lambda *a, **k: None,
        update_detail=lambda: None,
    )


def _is_descendant(widget, ancestor):
    current = widget
    while current is not None:
        if current == ancestor:
            return True
        current = getattr(current, "master", None)
    return False


def test_global_settings_has_four_short_tabs(app_mod):
    root = require_tk(app_mod.tk)
    root.withdraw()
    try:
        dialog = app_mod.GlobalSettingsDialog(root, _fake_app())
        try:
            labels = [dialog.notebook.tab(tab, "text")
                      for tab in dialog.notebook.tabs()]
            assert labels == ["一般", "伺服器", "VRAM 預檢", "工具與資料"]
            assert _is_descendant(dialog.dir_entry, dialog.global_tabs["一般"])
            assert _is_descendant(dialog.api_key_entry,
                                  dialog.global_tabs["伺服器"])
            assert _is_descendant(dialog.preflight_procs_text,
                                  dialog.global_tabs["VRAM 預檢"])
            assert _is_descendant(dialog.preset_list,
                                  dialog.global_tabs["工具與資料"])
            assert dialog.preflight_var.get() == "關閉"
        finally:
            if dialog.winfo_exists():
                dialog.destroy()
    finally:
        root.destroy()


def test_global_settings_saves_chinese_preflight_label(app_mod):
    root = require_tk(app_mod.tk)
    root.withdraw()
    try:
        dialog = app_mod.GlobalSettingsDialog(root, _fake_app())
        dialog.preflight_var.set("只警告")
        dialog.port_var.set("8081")
        dialog.cache_ram_var.set("32768")
        dialog.save()
        saved = app_mod._load_settings()
        assert saved["vram_preflight"]["mode"] == "warn"
        assert saved["server"]["port"] == 8081
        assert saved["server"]["cache_ram_mb"] == 32768
    finally:
        root.destroy()


def test_global_settings_rejects_bad_cache_ram(app_mod, monkeypatch):
    errors = []
    monkeypatch.setattr(app_mod.messagebox, "showerror",
                        lambda *a, **k: errors.append(a))
    root = require_tk(app_mod.tk)
    root.withdraw()
    try:
        dialog = app_mod.GlobalSettingsDialog(root, _fake_app())
        try:
            dialog.cache_ram_var.set("not-a-number")
            dialog.save()
            assert errors and errors[-1][0] == "RAM cache 格式錯誤"
            assert dialog.winfo_exists()
        finally:
            if dialog.winfo_exists():
                dialog.destroy()
    finally:
        root.destroy()
