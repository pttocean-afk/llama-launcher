import json
import os
from pathlib import Path

from llama_launcher.migration import (
    MigrationPlan,
    _merge_profiles,
    _load_json,
    detect_legacy_dir,
    merge_legacy_into_config,
    migrate_legacy_data,
    plan_migration,
)


def _write_legacy(legacy: Path, profiles: list[dict] | None = None, token: str = "legacy-token"):
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "models.json").write_text(
        json.dumps({"profiles": list(profiles or [])}), encoding="utf-8")
    (legacy / "control-token.txt").write_text(token + "\n", encoding="utf-8")
    return legacy


def test_detect_legacy_dir(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not detect_legacy_dir(empty)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "models.json").write_text("{}", encoding="utf-8")
    assert detect_legacy_dir(legacy)


def test_plan_migration_reports_available_and_conflicts(tmp_path: Path):
    legacy = _write_legacy(tmp_path / "legacy")
    dest = tmp_path / "data"
    dest.mkdir()
    (dest / "secrets").mkdir()
    (dest / "secrets" / "control-token").write_text("new-token", encoding="utf-8")
    plan = plan_migration(legacy, dest)
    assert set(plan.available) == {"models.json", "control-token.txt"}
    assert set(plan.will_copy) == {"models.json"}
    assert plan.will_skip == ("control-token.txt",)
    assert not plan.is_empty

    empty_dest = tmp_path / "fresh"
    fresh = plan_migration(legacy, empty_dest)
    assert set(fresh.will_copy) == {"models.json", "control-token.txt"}
    assert fresh.will_skip == ()


def test_migrate_into_fresh_destination(tmp_path: Path):
    legacy = _write_legacy(tmp_path / "legacy", [{"model": "m.gguf"}])
    dest = tmp_path / "data"
    result = migrate_legacy_data(legacy, dest)
    assert set(result.copied) == {"models.json", "control-token.txt"}
    assert json.loads((dest / "profiles.json").read_text(encoding="utf-8"))["profiles"] == \
        [{"model": "m.gguf"}]
    assert (dest / "secrets" / "control-token").read_text().strip() == "legacy-token"


def test_migrate_existing_destination_merges_profiles(tmp_path: Path):
    legacy = _write_legacy(tmp_path / "legacy", [{"model": "m.gguf"}])
    dest = tmp_path / "data"
    dest.mkdir()
    (dest / "profiles.json").write_text(
        '{"profiles": [{"model": "kept.gguf", "name": "Kept"}]}', encoding="utf-8")
    result = migrate_legacy_data(legacy, dest)
    assert "models.json" in result.skipped          # merged in place, not overwritten
    assert "control-token.txt" in result.copied
    # the in-place merge (used by the GUI flow) combines both lists
    assert merge_legacy_into_config(legacy, dest / "profiles.json") == 1
    merged = json.loads((dest / "profiles.json").read_text(encoding="utf-8"))["profiles"]
    by_model = {p["model"]: p for p in merged}
    assert set(by_model) == {"kept.gguf", "m.gguf"}
    assert by_model["kept.gguf"]["name"] == "Kept"  # existing entry untouched
    # re-running is idempotent
    assert merge_legacy_into_config(legacy, dest / "profiles.json") == 0
    merged2 = json.loads((dest / "profiles.json").read_text(encoding="utf-8"))["profiles"]
    assert len(merged2) == 2


def test_merge_profiles_new_app_settings_win(tmp_path: Path):
    legacy_cfg = _load_json(_write_legacy(tmp_path / "legacy", [
        {"name": "A", "model": "a.gguf", "default_ctx": 1024, "starred": True,
         "favorite_order": 1},
        {"name": "B", "model": "b.gguf", "default_ctx": 2048, "starred": True,
         "favorite_order": 0},
        {"name": "C", "model": "c.gguf", "default_ctx": 4096},
        {"model": ""},
    ]) / "models.json")
    current_cfg = {"profiles": [
        {"name": "A-renamed", "model": "a.gguf", "default_ctx": 777, "starred": True,
         "favorite_order": 0},
    ]}
    merged = _merge_profiles(legacy_cfg, current_cfg)
    by_model = {p["model"]: p for p in merged}
    # new-app profile wins, legacy settings for it are not imported
    assert by_model["a.gguf"]["name"] == "A-renamed"
    assert by_model["a.gguf"]["default_ctx"] == 777
    assert by_model["b.gguf"]["default_ctx"] == 2048
    assert by_model["c.gguf"]["default_ctx"] == 4096
    assert "configured" not in by_model["b.gguf"]
    # starred keep favorite order; unstarred come after
    assert [p["model"] for p in merged] == ["a.gguf", "b.gguf", "c.gguf"]
    assert merged[0]["favorite_order"] == 0
    assert merged[1]["favorite_order"] == 1


def test_merge_into_empty_config(tmp_path: Path):
    legacy_cfg = _load_json(_write_legacy(tmp_path / "legacy", [
        {"name": "Z", "model": "z.gguf", "starred": False},
    ]) / "models.json")
    current_cfg: dict = {}
    merged = _merge_profiles(legacy_cfg, current_cfg)
    assert [p["model"] for p in merged] == ["z.gguf"]
    assert current_cfg["profiles"] == merged


def test_load_json_bad_input(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert _load_json(bad) == {}
    assert _load_json(tmp_path / "missing.json") == {}


def test_app_migration_flow(tmp_path, monkeypatch):
    """LauncherApp._do_migrate_legacy copies + merges + refreshes (no display)."""
    import llama_launcher.app as app

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    models = tmp_path / "llama" / "models"
    models.mkdir(parents=True)
    (models / "a.gguf").write_bytes(b"")
    # The app module computed SETTINGS_PATH / CONFIG_PATH at import time;
    # point them at this test's isolated data dir.
    monkeypatch.setattr(app, "SETTINGS_PATH", data_dir / "settings.json")
    monkeypatch.setattr(app, "CONFIG_PATH", data_dir / "profiles.json")

    monkeypatch.setattr(app, "MODELS_DIR", models)
    monkeypatch.setattr(app, "invalidate_model_inventory", lambda: None)
    monkeypatch.setattr(app, "llama_server_filename", lambda: "llama-server")
    monkeypatch.setattr(app, "LLAMA_SERVER", models / "llama-server")
    monkeypatch.setattr(app, "VULKAN_SERVER", models / "llama-server")
    monkeypatch.setattr(app.messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(app.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(app.messagebox, "showwarning", lambda *a, **k: None)

    legacy = _write_legacy(tmp_path / "legacy", [
        {"name": "Old", "model": "a.gguf", "default_ctx": 131072, "starred": True},
        {"name": "Newer", "model": "z.gguf", "default_ctx": 65536, "starred": True,
         "favorite_order": 0},
    ])

    # pre-existing profile in the new app must win over the legacy one
    app.save_config({"profiles": [{"name": "Newer-renamed", "model": "z.gguf",
                                   "default_ctx": 32768, "starred": True}]})
    app_instance = app.LauncherApp.__new__(app.LauncherApp)
    app_instance.cfg = app.load_config()
    app_instance.profiles = app.merge_profiles(app_instance.cfg)

    class _RefreshRecorder:
        def __init__(self):
            self.calls = []

        def __call__(self, *a, **k):
            self.calls.append((a, k))

    refresh = _RefreshRecorder()
    update = _RefreshRecorder()
    monkeypatch.setattr(app_instance, "refresh_listbox", refresh)
    monkeypatch.setattr(app_instance, "update_detail", update)

    result = app_instance._do_migrate_legacy(legacy)
    assert result is not None
    # existing config: token copied; profiles.json merged in place by the helper
    assert "control-token.txt" in result.copied
    assert (data_dir / "secrets" / "control-token").read_text().strip() == "legacy-token"
    merged_cfg = app.load_config()
    by_model = {p["model"]: p for p in merged_cfg["profiles"]}
    assert by_model["z.gguf"]["name"] == "Newer-renamed"   # new-app wins
    assert by_model["a.gguf"]["name"] == "Old"             # legacy imported
    assert [p["model"] for p in merged_cfg["profiles"]] == ["z.gguf", "a.gguf"]
    assert refresh.calls and update.calls
    # idempotent: running the merge again adds nothing
    assert app.merge_legacy_into_config(legacy, data_dir / "profiles.json") == 0
    assert len(app.load_config()["profiles"]) == 2
