import json
import logging
import os
import socket
import tempfile
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest


def _import_app(tmp_path, monkeypatch):
    """Import llama_launcher.app with an isolated data dir and empty model dir."""
    monkeypatch.setenv("LLAMA_LAUNCHER_DATA_DIR", str(tmp_path / "data"))
    models = tmp_path / "llama" / "models"
    models.mkdir(parents=True)
    (models / "a.gguf").write_bytes(b"")
    monkeypatch.setattr("llama_launcher.app.MODELS_DIR", models)
    monkeypatch.setattr("llama_launcher.app.invalidate_model_inventory", lambda: None)
    monkeypatch.setattr("llama_launcher.app.llama_server_filename", lambda: "llama-server")
    monkeypatch.setattr("llama_launcher.app.LLAMA_SERVER", models / "llama-server")
    monkeypatch.setattr("llama_launcher.app.VULKAN_SERVER", models / "llama-server")
    import llama_launcher.app as app

    return app


class _FakeServer:
    def __init__(self, app_mod):
        self.mod = app_mod
        self.running = False
        self.log_path = None
        self.profile_name = ""
        self.backend = ""
        self.degraded_reason = ""
        self.externally_adopted = False
        self.started_at = None

    def scan_runtime_health(self):
        pass

    def pid_text(self):
        return "123"

    def uptime_text(self):
        return ""

    def start(self, profile, ctx):
        self.running = True
        self.profile_name = profile.get("name", "")
        return True, "started"

    def stop(self):
        self.running = False
        return "stopped"


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    return _import_app(tmp_path, monkeypatch)


def _make_app(app_mod):
    app = app_mod.LauncherApp.__new__(app_mod.LauncherApp)
    app.server = _FakeServer(app_mod)
    app.profiles = [
        {"name": "A", "model": "a.gguf", "backend": "cuda",
         "default_ctx": 131072, "reasoning": "off", "mmproj": ""},
        {"name": "XSS<script>", "model": "xss.gguf", "backend": "cuda",
         "default_ctx": 65536, "reasoning": "on", "mmproj": ""},
    ]
    app.server_lock = threading.Lock()
    app.control_server = None
    app.control_error = ""
    return app


def test_cuda_and_vulkan_always_require_vram_preflight(app_mod):
    assert app_mod.backend_requires_vram_preflight("cuda")
    assert app_mod.backend_requires_vram_preflight("CUDA")
    assert app_mod.backend_requires_vram_preflight("vulkan")
    assert not app_mod.backend_requires_vram_preflight("cpu")
    assert "EdgeGameAssist.exe" in app_mod.VRAM_CLEANUP_PROCESS_NAMES


def test_remote_start_and_stop_take_lock_and_report(app_mod):
    app = _make_app(app_mod)
    start = app.remote_start({"model": "a.gguf"})
    assert start["ok"]
    assert start["status"]["control"]["ok"] is False
    assert start["status"]["control"]["port"] == app_mod.CONTROL_PORT
    # already running → rejected without touching the server again
    busy = app.remote_start({"model": "a.gguf"})
    assert not busy["ok"]
    assert busy["error"] == "llama-server is already running"
    stop = app.remote_stop()
    assert stop["ok"]
    assert not app.server.running


def test_remote_start_rejects_unknown_model(app_mod):
    app = _make_app(app_mod)
    result = app.remote_start({"model": "nope.gguf"})
    assert not result["ok"]
    assert "not in models.json" in result["error"]


def test_control_server_bind_failure_is_observable(app_mod, monkeypatch):
    """LauncherApp with an occupied control port must log and expose the failure."""
    blocker = app_mod.ControlServer(_make_app(app_mod))
    blocker.start()
    logs = []
    handler = _CollectingHandler(logs)
    logger = app_mod.logging.getLogger("llama_launcher")
    logger.addHandler(handler)
    logger.setLevel(app_mod.logging.WARNING)
    monkeypatch.setattr(app_mod.messagebox, "showwarning", lambda *a, **k: None)
    try:
        root = app_mod.tk.Tk()
        app = app_mod.LauncherApp(root)
        root.destroy()
        assert app.control_server is None
        assert "8765" in app.control_error
        status = app.remote_status()
        assert status["control"]["ok"] is False
        assert "8765" in (status["control"]["error"] or "")
        assert any("8765" in record.getMessage() for record in logs), logs
    finally:
        logger.removeHandler(handler)
        logger.setLevel(0)
        blocker.close()


class _CollectingHandler(logging.Handler):
    def __init__(self, sink):
        super().__init__()
        self._sink = sink

    def emit(self, record):
        self._sink.append(record)


def test_dashboard_html_builds_profiles_with_dom(app_mod):
    """The remote dashboard must not interpolate profile data into innerHTML."""
    html = app_mod.REMOTE_CONTROL_HTML
    assert "renderProfiles" in html
    # profile name/model/backend no longer flow into an innerHTML template
    assert "${x.name}" not in html
    assert "${x.model" not in html
    assert "onclick=\"start(" not in html
    # values are assigned through textContent / addEventListener only
    assert "name.textContent=x.name" in html
    assert "addEventListener('click'" in html


def test_end_to_end_control_api_over_http(app_mod):
    """Real ControlServer over HTTP: token auth + DOM-safe profile rendering."""
    app = _make_app(app_mod)
    app.profiles = [
        {"name": "Evil<script><img src=x>", "model": "xss.gguf",
         "backend": "cuda", "default_ctx": 131072, "reasoning": "off"},
    ]
    server = app_mod.ControlServer(app)
    server.start()
    try:
        base = f"http://127.0.0.1:{app_mod.CONTROL_PORT}"
        # unauthenticated API access is rejected
        with pytest.raises(Exception):
            urlopen(Request(base + "/api/profiles"))
        token = server.token
        req = Request(base + "/api/profiles",
                      headers={"Authorization": f"Bearer {token}"})
        with urlopen(req) as resp:
            payload = json.loads(resp.read())
        names = [p["name"] for p in payload["profiles"]]
        assert "Evil<script><img src=x>" in names
        # the page itself serves the DOM-based renderer
        with urlopen(Request(base + "/")) as resp:
            html = resp.read().decode("utf-8")
        assert "renderProfiles" in html
        start_req = Request(
            base + "/api/start", data=b'{"model":"xss.gguf"}', method="POST",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"})
        with urlopen(start_req) as resp:
            assert json.loads(resp.read())["ok"] is True
    finally:
        server.close()


# ---------------------------------------------------------------- UI 卡頓防護
def test_scan_runtime_health_short_circuits_without_io(tmp_path, monkeypatch):
    """log 沒新增時 scan_runtime_health 不開檔（每秒輪詢不拖 UI）。"""
    import llama_launcher.app as app

    log = tmp_path / "server.log"
    log.write_bytes(b"hello world\n")
    manager = app.ServerManager.__new__(app.ServerManager)
    manager.log_path = log
    manager._health_scan_offset = 0
    manager._health_scan_size = -1
    manager.degraded_reason = ""
    manager.scan_runtime_health()
    assert manager._health_scan_size == log.stat().st_size
    assert manager._health_scan_offset == log.stat().st_size

    # 第二次：size 沒變 → 不開檔（用假的 open 驗證不會被呼叫）
    calls = {"open": 0}

    def fake_open(*_a, **_k):
        calls["open"] += 1
        raise AssertionError("should not open file")

    import builtins
    monkeypatch.setattr(builtins, "open", fake_open)
    manager.scan_runtime_health()
    assert calls["open"] == 0
