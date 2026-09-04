"""VRAM 預檢三級（off / warn / strict）行為測試。

重點回歸：
- 預設 off：不做任何查詢、不擋啟動（朋友只有單卡時不再誤擋）。
- 單張 GPU 也能正常讀取／檢查（舊版要求 >= 2 張才讀）。
- warn 超標只警告；strict 超時才擋，且訊息列出佔用程序。
"""
import pytest


def _import_app(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_LAUNCHER_DATA_DIR", str(tmp_path / "data"))
    import llama_launcher.app as app
    return app


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    app = _import_app(tmp_path, monkeypatch)
    # SETTINGS_PATH 在 import 時固定；每個測試的 tmp 目錄不同，必須重設
    monkeypatch.setattr(app, "SETTINGS_PATH", app.settings_path())
    return app


def _manager(app_mod, memory_sequence, owners=None):
    """建立不跑 __init__ 的 ServerManager，VRAM 查詢依序回傳。"""
    manager = app_mod.ServerManager.__new__(app_mod.ServerManager)
    state = {"i": 0}

    def fake_query_gpu_memory_mb():
        if state["i"] >= len(memory_sequence):
            return memory_sequence[-1]
        value = memory_sequence[state["i"]]
        state["i"] += 1
        return value

    manager.query_gpu_memory_mb = fake_query_gpu_memory_mb
    manager.query_gpu_count = lambda: (len(memory_sequence[-1])
                                       if memory_sequence[-1] is not None
                                       else None)
    manager.query_gpu_processes = (
        lambda idx: owners[idx] if owners and idx in owners else [])
    manager._stop_comfyui_if_active = lambda d, s: (True, "ComfyUI skipped")
    manager._close_vram_cleanup_allowlist = lambda names: None
    return manager


def _config(mode, limits=(), **kw):
    base = {
        "mode": mode,
        "gpu_limits_mb": list(limits),
        "kill_processes": ["TestApp.exe"],
        "comfyui": {"enabled": False, "distro": "Ubuntu",
                    "service": "comfyui.service"},
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------- off
def test_off_mode_does_not_query_gpu(app_mod, monkeypatch):
    called = []
    manager = app_mod.ServerManager.__new__(app_mod.ServerManager)
    manager.query_gpu_memory_mb = lambda: called.append(1) or []
    monkeypatch.setattr(app_mod, "vram_preflight_config",
                        lambda: _config("off"))
    ok, msg = manager.run_vram_preflight()
    assert ok is True
    assert "disabled" in msg
    assert called == []


def test_off_mode_is_default_when_settings_missing(app_mod, tmp_path):
    # 沒有寫過 settings.json → 預設 off
    assert app_mod.vram_preflight_config()["mode"] == "off"


# ---------------------------------------------------------------- warn
def test_warn_single_gpu_within_limit_passes(app_mod, monkeypatch):
    manager = _manager(app_mod, [[500]])
    monkeypatch.setattr(app_mod, "vram_preflight_config",
                        lambda: _config("warn", [800]))
    ok, msg = manager.run_vram_preflight()
    assert ok is True
    assert "OK" in msg


def test_warn_single_gpu_violation_only_warns(app_mod, monkeypatch):
    """單卡超標：warn 模式只警告、仍放行（不擋啟動）。"""
    manager = _manager(app_mod, [[2500]])
    monkeypatch.setattr(app_mod, "vram_preflight_config",
                        lambda: _config("warn", [800]))
    ok, msg = manager.run_vram_preflight()
    assert ok is True
    assert "警告" in msg
    assert "GPU0 2500 MB > 800 MB" in msg


def test_warn_gpu_unreadable_continues(app_mod, monkeypatch):
    manager = _manager(app_mod, [None])
    monkeypatch.setattr(app_mod, "vram_preflight_config",
                        lambda: _config("warn"))
    ok, msg = manager.run_vram_preflight()
    assert ok is True
    assert "讀取不到" in msg


# ---------------------------------------------------------------- strict
def test_strict_single_gpu_within_limit_passes(app_mod, monkeypatch):
    """回歸：只有 1 張 GPU 也能通過 strict（舊版要求 >= 2 張才讀取）。"""
    manager = _manager(app_mod, [[100], [80]])
    monkeypatch.setattr(app_mod, "vram_preflight_config",
                        lambda: _config("strict", [800, 500]))
    ok, msg = manager.run_vram_preflight()
    assert ok is True
    assert "GPU0 100→80 MB" in msg


def test_strict_gpu_unreadable_blocks(app_mod, monkeypatch):
    manager = _manager(app_mod, [None])
    monkeypatch.setattr(app_mod, "vram_preflight_config",
                        lambda: _config("strict", [800]))
    ok, msg = manager.run_vram_preflight()
    assert ok is False
    assert "無法讀取" in msg
    assert "已取消伺服器啟動" in msg


def test_strict_timeout_blocks_and_lists_owners(app_mod, monkeypatch):
    """strict 超時：擋住，並列出超標 GPU 的佔用程序。"""
    monkeypatch.setattr(app_mod, "VRAM_PREFLIGHT_WAIT_SECONDS", 0)
    monkeypatch.setattr(app_mod.time, "sleep", lambda s: None)
    manager = _manager(
        app_mod, [[3000], [3000], [3000]],
        owners={0: ["ComfyUI (PID 1234)"]})
    monkeypatch.setattr(app_mod, "vram_preflight_config",
                        lambda: _config("strict", [800, 500]))
    ok, msg = manager.run_vram_preflight()
    assert ok is False
    assert "已取消伺服器啟動" in msg
    assert "GPU0：3000 MB（需 ≤ 800 MB）" in msg
    assert "ComfyUI (PID 1234)" in msg


def test_strict_recover_after_cleanup_passes(app_mod, monkeypatch):
    """strict：清理後回到上限內 → 放行。"""
    monkeypatch.setattr(app_mod, "VRAM_PREFLIGHT_WAIT_SECONDS", 5)
    monkeypatch.setattr(app_mod.time, "sleep", lambda s: None)
    manager = _manager(app_mod, [[3000], [3000], [500]])
    monkeypatch.setattr(app_mod, "vram_preflight_config",
                        lambda: _config("strict", [800, 500]))
    ok, msg = manager.run_vram_preflight()
    assert ok is True
    assert "GPU0 3000→500 MB" in msg


def test_strict_no_limits_never_violates(app_mod, monkeypatch):
    manager = _manager(app_mod, [[999999]])
    monkeypatch.setattr(app_mod, "vram_preflight_config",
                        lambda: _config("strict"))
    ok, _ = manager.run_vram_preflight()
    assert ok is True


# ---------------------------------------------------------------- 靜態方法
def test_limit_violations(app_mod):
    V = app_mod.ServerManager._limit_violations
    assert V([100], [800]) == []
    assert V([900], [800]) == [(0, 900, 800)]
    assert V([100, 900], [800, 500]) == [(1, 900, 500)]
    # None 上限 = 不限制；多出的 GPU 不受限
    assert V([100, 900], [800, None]) == []
    assert V([100, 900, 99999], [800, 500]) == [(1, 900, 500)]
    assert V([100, 900, 99999], [800, 500, 100]) == [(1, 900, 500), (2, 99999, 100)]
    # 0 是合法上限
    assert V([1], [0]) == [(0, 1, 0)]


def test_vram_preflight_config_normalization(app_mod):
    from llama_launcher.paths import data_dir
    settings_path = data_dir() / "settings.json"  # 會自動建目錄
    import json
    settings_path.write_text(json.dumps({
        "vram_preflight": {
            "mode": "STRICT",
            "gpu_limits_mb": ["2304", "bad", 128],
            "kill_processes": ["A.exe", "  ", "B.exe"],
            "comfyui": {"enabled": True, "distro": "Ubuntu-24.04"},
        },
    }), encoding="utf-8")
    cfg = app_mod.vram_preflight_config()
    assert cfg["mode"] == "strict"
    assert cfg["gpu_limits_mb"] == [2304, None, 128]
    assert cfg["kill_processes"] == ["A.exe", "B.exe"]
    assert cfg["comfyui"]["enabled"] is True
    assert cfg["comfyui"]["distro"] == "Ubuntu-24.04"
    # 未知 mode → off
    settings_path.write_text(json.dumps(
        {"vram_preflight": {"mode": "yolo"}}), encoding="utf-8")
    assert app_mod.vram_preflight_config()["mode"] == "off"


def test_cleanup_process_names_default_list_kept(app_mod):
    """預設清理清單常量保留（相容性）；實際行為以設定為準。"""
    assert "EdgeGameAssist.exe" in app_mod.VRAM_CLEANUP_PROCESS_NAMES
