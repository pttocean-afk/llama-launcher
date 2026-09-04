"""ServerManager.start 的指令組裝回歸測試。

重點：build_server_args 只回傳參數（不含執行檔），start() 必須把
伺服器二進位路徑放在指令最前面，否則 Popen 會拿參數當指令執行。
"""
import pytest


def _import_app(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_LAUNCHER_DATA_DIR", str(tmp_path / "data"))
    models = tmp_path / "llama" / "models"
    models.mkdir(parents=True)
    (models / "a.gguf").write_bytes(b"")
    import llama_launcher.app as app
    app.MODELS_DIR = models
    app.invalidate_model_inventory = lambda: None
    app._MODEL_INVENTORY_CACHE = None
    monkeypatch.setattr(app, "SETTINGS_PATH", app.settings_path())
    return app


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    return _import_app(tmp_path, monkeypatch)


def _manager(app_mod):
    # running 是 property（由 proc / external_pid 推導）
    manager = app_mod.ServerManager.__new__(app_mod.ServerManager)
    manager.proc = None
    manager.external_pid = None
    manager.external_command_line = ""
    manager.externally_adopted = False
    manager.log_path = None
    manager.log_fh = None
    manager.started_at = None
    manager.profile_name = ""
    manager.backend = ""
    manager.degraded_reason = ""
    manager.preflight_summary = ""
    return manager


def test_start_prepends_binary_path(app_mod, tmp_path, monkeypatch):
    """Popen 的第一個元素必須是伺服器執行檔，參數跟在其後。"""
    fake_binary = tmp_path / "llama-server"
    fake_binary.write_bytes(b"MZ")
    monkeypatch.setattr(app_mod, "LLAMA_SERVER", fake_binary)
    monkeypatch.setattr(app_mod, "VULKAN_SERVER", fake_binary)
    monkeypatch.setattr(app_mod, "port_in_use", lambda port: False)

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(app_mod.subprocess, "Popen", fake_popen)

    profile = {
        "name": "A", "model": "a.gguf", "scheme": "預設",
        "backend": "cuda", "default_ctx": 131072, "reasoning": "off",
        "mmproj": "", "kv_mode": "q4", "mtp": False, "parallel": 1,
        "ngl": 999, "gpu_split": "", "raw_args": "",
    }
    manager = _manager(app_mod)
    ok, msg = manager.start(profile, "128")
    assert ok, msg
    cmd = captured["cmd"]
    assert cmd[0] == str(fake_binary)
    assert cmd[1] == "-m"
    assert str(app_mod.MODELS_DIR / "a.gguf") in cmd
    # 預設 preflight off 不會擋啟動
    assert manager.preflight_summary == "VRAM preflight disabled"


def test_start_writes_full_command_to_log_header(app_mod, tmp_path, monkeypatch):
    fake_binary = tmp_path / "llama-server"
    fake_binary.write_bytes(b"MZ")
    monkeypatch.setattr(app_mod, "LLAMA_SERVER", fake_binary)
    monkeypatch.setattr(app_mod, "VULKAN_SERVER", fake_binary)
    monkeypatch.setattr(app_mod, "port_in_use", lambda port: False)

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(app_mod.subprocess, "Popen", fake_popen)
    profile = {
        "name": "A", "model": "a.gguf", "scheme": "預設",
        "backend": "cuda", "default_ctx": 131072, "reasoning": "off",
        "mmproj": "", "kv_mode": "q4", "mtp": False, "parallel": 1,
        "ngl": 999, "gpu_split": "", "raw_args": "",
    }
    manager = _manager(app_mod)
    ok, msg = manager.start(profile, "128")
    assert ok, msg
    header = manager.log_path.read_text(encoding="utf-8").splitlines()[2]
    assert header.startswith("# " + str(fake_binary))
    assert " ".join(captured["cmd"]) == header[2:]


def test_start_rejects_occupied_port(app_mod, tmp_path, monkeypatch):
    fake_binary = tmp_path / "llama-server"
    fake_binary.write_bytes(b"MZ")
    monkeypatch.setattr(app_mod, "LLAMA_SERVER", fake_binary)
    monkeypatch.setattr(app_mod, "port_in_use", lambda port: True)
    manager = _manager(app_mod)
    ok, msg = manager.start({"name": "A", "model": "a.gguf"}, "128")
    assert not ok
    assert "已被佔用" in msg


def test_start_rejects_missing_model(app_mod, tmp_path, monkeypatch):
    fake_binary = tmp_path / "llama-server"
    fake_binary.write_bytes(b"MZ")
    monkeypatch.setattr(app_mod, "LLAMA_SERVER", fake_binary)
    monkeypatch.setattr(app_mod, "port_in_use", lambda port: False)
    manager = _manager(app_mod)
    ok, msg = manager.start({"name": "A", "model": "nope.gguf"}, "128")
    assert not ok
    assert "找不到模型檔" in msg
