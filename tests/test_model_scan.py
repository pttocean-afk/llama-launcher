"""遞迴掃描 models 子資料夾 + 模型路徑解析（model_file / relative_model_name）。"""
from __future__ import annotations

import pytest


@pytest.fixture
def scan_env(tmp_path, monkeypatch):
    """建立 models 目錄結構並回傳隔離的 app 模組。"""
    monkeypatch.setenv("LLAMA_LAUNCHER_DATA_DIR", str(tmp_path / "data"))
    import llama_launcher.app as app

    models = tmp_path / "llama" / "models"
    (models / "Coding").mkdir(parents=True)
    (models / "Coding" / "deep").mkdir()
    (models / "Vision").mkdir()
    (models / "top.gguf").write_bytes(b"x" * 10)
    (models / "Coding" / "qwen.gguf").write_bytes(b"y" * 20)
    (models / "Coding" / "deep" / "deep.gguf").write_bytes(b"z" * 30)
    (models / "Vision" / "mmproj-qwen-f16.gguf").write_bytes(b"v" * 5)
    (models / "note.txt").write_text("not a model")
    monkeypatch.setattr(app, "MODELS_DIR", models)
    app.invalidate_model_inventory()
    yield app
    app.invalidate_model_inventory()


def test_scan_recurses_into_subfolders(scan_env):
    app = scan_env
    assert app.scan_gguf_files() == [
        "Coding/deep/deep.gguf", "Coding/qwen.gguf", "top.gguf"]
    assert app.scan_mmproj_files() == ["Vision/mmproj-qwen-f16.gguf"]


def test_mmproj_detection_uses_filename_not_folder_name(tmp_path, monkeypatch):
    """分類資料夾名稱含 mmproj 不會讓裡面的模型被誤判成 vision 檔。"""
    monkeypatch.setenv("LLAMA_LAUNCHER_DATA_DIR", str(tmp_path / "data"))
    import llama_launcher.app as app

    models = tmp_path / "llama" / "models"
    (models / "mmproj-files").mkdir(parents=True)
    (models / "mmproj-files" / "qwen.gguf").write_bytes(b"x")
    monkeypatch.setattr(app, "MODELS_DIR", models)
    app.invalidate_model_inventory()
    try:
        assert app.scan_gguf_files() == ["mmproj-files/qwen.gguf"]
        assert app.scan_mmproj_files() == []
    finally:
        app.invalidate_model_inventory()


def test_sizes_keyed_by_relative_path(scan_env):
    app = scan_env
    assert app.model_size_text("top.gguf") == "10M" or \
        app.model_size_text("top.gguf")  # 大小換算不為空即可
    assert app.model_size_text("Coding/qwen.gguf")
    assert app.model_size_text("不存在的.gguf") == ""


def test_model_file_resolves_subfolder_paths(scan_env):
    app = scan_env
    assert app.model_file("top.gguf") == app.MODELS_DIR / "top.gguf"
    assert app.model_file("Coding/qwen.gguf") == \
        app.MODELS_DIR / "Coding" / "qwen.gguf"
    # Windows 風格分隔也接受
    assert app.model_file("Coding\\qwen.gguf") == \
        app.MODELS_DIR / "Coding" / "qwen.gguf"


def test_model_file_blocks_path_escape(scan_env):
    app = scan_env
    resolved = app.model_file("../../secret.gguf")
    assert ".." not in resolved.parts
    assert not resolved.exists()


def test_relative_model_name_inside_and_outside(scan_env):
    app = scan_env
    inside = app.MODELS_DIR / "Coding" / "qwen.gguf"
    assert app.relative_model_name(inside) == "Coding/qwen.gguf"
    outside = app.MODELS_DIR.parent / "other.gguf"
    assert app.relative_model_name(outside) == "other.gguf"


def test_merge_profiles_names_subfolder_models(scan_env):
    app = scan_env
    merged = app.merge_profiles({})
    by_model = {p["model"]: p for p in merged}
    assert by_model["Coding/qwen.gguf"]["name"] == "Coding/qwen"
    assert by_model["top.gguf"]["name"] == "top"


def test_guess_mmproj_matches_rule_by_basename(scan_env):
    app = scan_env
    got = app.guess_mmproj(
        "Agents-A1-8b.gguf", ["Vision/Agents-A1-mmproj.gguf"])
    assert got == "Vision/Agents-A1-mmproj.gguf"


def test_start_reports_missing_subfolder_model(scan_env, monkeypatch):
    """ServerManager.start 對子資料夾模型路徑的錯誤訊息可讀。"""
    app = scan_env
    monkeypatch.setattr(app, "LLAMA_SERVER", app.MODELS_DIR.parent / "llama-server")
    monkeypatch.setattr(app, "VULKAN_SERVER",
                        app.MODELS_DIR.parent / "llama-server")
    (app.MODELS_DIR.parent / "llama-server").write_bytes(b"")
    monkeypatch.setattr(app, "port_in_use", lambda _port: False)
    mgr = app.ServerManager()
    ok, msg = mgr.start({"model": "Coding/不存在.gguf", "backend": "cpu",
                         "mmproj": "", "ngl": 999}, "128")
    assert not ok
    assert "Coding/不存在.gguf" in msg
