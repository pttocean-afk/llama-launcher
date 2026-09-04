"""同一模型多方案（scheme）行為測試。

覆蓋：
- merge_profiles：同一模型的多個方案都保留，未配置模型得到預設方案。
- remote_start：指定／未指定方案的配對（預設方案優先）。
- _persist_profile：按 (model, scheme) 更新，不誤傷同模型其他方案。
- 舊檔（沒有 scheme 欄位）相容：視為「預設」方案。
"""
import json

import pytest


def _import_app(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_LAUNCHER_DATA_DIR", str(tmp_path / "data"))
    models = tmp_path / "llama" / "models"
    models.mkdir(parents=True)
    for name in ("a.gguf", "b.gguf"):
        (models / name).write_bytes(b"")
    import llama_launcher.app as app
    app.MODELS_DIR = models
    app.invalidate_model_inventory = lambda: None
    return app


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    app = _import_app(tmp_path, monkeypatch)
    # 模型清單快取是模組級：每個測試用獨立 tmp 目錄，必須清空
    monkeypatch.setattr(app, "_MODEL_INVENTORY_CACHE", None, raising=False)
    return app


def _profile(model="a.gguf", scheme="預設", **kw):
    base = {
        "name": model.replace(".gguf", ""),
        "model": model,
        "scheme": scheme,
        "mmproj": "",
        "vision_enabled": False,
        "default_ctx": 131072,
        "reasoning": "off",
        "gpu_split": "",
        "backend": "cuda",
        "jinja": False,
        "extra_args": "",
        "kv_mode": "q4",
        "mtp": False,
        "parallel": 1,
        "starred": False,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------- merge
def test_merge_keeps_multiple_schemes_same_model(app_mod):
    cfg = {"profiles": [
        _profile(scheme="預設"),
        _profile(scheme="code"),
    ]}
    profiles = app_mod.merge_profiles(cfg)
    a = [p for p in profiles if p["model"] == "a.gguf"]
    assert {p["scheme"] for p in a} == {"預設", "code"}
    # b.gguf 在 models 目錄但未設定 → 只得到一個「預設」方案
    b = [p for p in profiles if p["model"] == "b.gguf"]
    assert len(b) == 1 and b[0]["scheme"] == "預設"
    assert len(profiles) == 3


def test_merge_legacy_profile_gets_default_scheme(app_mod):
    # 舊檔沒有 scheme 欄位
    cfg = {"profiles": [{"name": "A", "model": "a.gguf"}]}
    profiles = app_mod.merge_profiles(cfg)
    a = [p for p in profiles if p["model"] == "a.gguf"]
    assert len(a) == 1
    assert a[0]["scheme"] == "預設"


def test_merge_unconfigured_model_gets_default(app_mod):
    # models 目錄裡有 a.gguf 和 b.gguf；只有 a 有存檔設定
    cfg = {"profiles": [_profile("a.gguf")]}
    profiles = app_mod.merge_profiles(cfg)
    b = [p for p in profiles if p["model"] == "b.gguf"]
    assert len(b) == 1
    assert b[0]["scheme"] == "預設"
    assert b[0]["configured"] is False
    a = [p for p in profiles if p["model"] == "a.gguf"]
    assert len(a) == 1
    assert a[0]["configured"] is True


def test_merge_normalizes_parallel(app_mod):
    # 舊檔沒有 parallel 欄位 → 補上預設 1
    cfg = {"profiles": [{k: v for k, v in _profile("a.gguf").items()
                         if k != "parallel"}]}
    profiles = app_mod.merge_profiles(cfg)
    a = [p for p in profiles if p["model"] == "a.gguf"][0]
    assert a["parallel"] == 1
    # 存檔的 parallel（字串）原樣保留（下游 parse 會處理）
    cfg2 = {"profiles": [_profile("a.gguf", parallel="2")]}
    a2 = [p for p in app_mod.merge_profiles(cfg2)
          if p["model"] == "a.gguf"][0]
    assert a2["parallel"] == "2"
    assert app_mod.parallel_from_profile(a2) == 2


# ---------------------------------------------------------------- remote_start
class _Recaller:
    """記錄 start 被傳入的 profile，不真正啟動。"""

    def __init__(self, app_mod):
        self.mod = app_mod
        self.running = False
        self.log_path = None
        self.profile_name = ""
        self.backend = ""
        self.degraded_reason = ""
        self.externally_adopted = False
        self.started_at = None
        self.last_profile = None

    def scan_runtime_health(self):
        pass

    def pid_text(self):
        return "123"

    def uptime_text(self):
        return ""

    def start(self, profile, ctx):
        self.last_profile = dict(profile)
        self.running = True
        self.profile_name = profile.get("name", "")
        return True, "started"

    def stop(self):
        self.running = False
        return "stopped"


def _make_app(app_mod, profiles):
    import threading
    app = app_mod.LauncherApp.__new__(app_mod.LauncherApp)
    app.server = _Recaller(app_mod)
    app.profiles = profiles
    app.cfg = {"profiles": [dict(p) for p in profiles]}
    app.server_lock = threading.Lock()
    app.control_server = None
    app.control_error = ""
    return app


def test_remote_start_without_scheme_prefers_default(app_mod):
    app = _make_app(app_mod, [
        _profile(scheme="code"),
        _profile(scheme="預設"),
    ])
    result = app.remote_start({"model": "a.gguf"})
    assert result["ok"]
    assert app.server.last_profile["scheme"] == "預設"


def test_remote_start_without_scheme_falls_back_to_first(app_mod):
    app = _make_app(app_mod, [_profile(scheme="code")])
    result = app.remote_start({"model": "a.gguf"})
    assert result["ok"]
    assert app.server.last_profile["scheme"] == "code"


def test_remote_start_with_scheme(app_mod):
    app = _make_app(app_mod, [
        _profile(scheme="預設"),
        _profile(scheme="code"),
    ])
    result = app.remote_start({"model": "a.gguf", "scheme": "code"})
    assert result["ok"]
    assert app.server.last_profile["scheme"] == "code"


def test_remote_start_unknown_scheme_rejected(app_mod):
    app = _make_app(app_mod, [_profile(scheme="預設")])
    result = app.remote_start({"model": "a.gguf", "scheme": "ghost"})
    assert not result["ok"]
    assert "not in models.json" in result["error"]


def test_remote_start_unknown_model_rejected(app_mod):
    app = _make_app(app_mod, [_profile("a.gguf")])
    result = app.remote_start({"model": "zzz.gguf"})
    assert not result["ok"]
    assert "not in models.json" in result["error"]


def test_remote_profiles_expose_scheme(app_mod):
    app = _make_app(app_mod, [_profile(scheme="code")])
    listed = app.remote_profiles()
    assert listed[0]["scheme"] == "code"
    assert app_mod.normalize_scheme(None) == "預設"


# ---------------------------------------------------------------- 持久化
def test_persist_profile_updates_by_key_only(app_mod):
    app = _make_app(app_mod, [
        _profile(scheme="預設", default_ctx=131072),
        _profile(scheme="code", default_ctx=65536),
    ])
    target = [p for p in app.profiles if p["scheme"] == "code"][0]
    target["default_ctx"] = 32768
    app._persist_profile(target)
    saved = app.cfg["profiles"]
    code = [p for p in saved if p["scheme"] == "code"][0]
    default = [p for p in saved if p["scheme"] == "預設"][0]
    assert code["default_ctx"] == 32768
    assert default["default_ctx"] == 131072  # 未受影響
    assert len(saved) == 2


def test_persist_profile_appends_new_scheme(app_mod):
    app = _make_app(app_mod, [_profile(scheme="預設")])
    app._persist_profile(_profile(scheme="agent"))
    assert len(app.cfg["profiles"]) == 2


def test_persist_legacy_profile_maps_to_default_scheme(app_mod):
    # 舊檔沒有 scheme：_persist_profile 把它當成「預設」方案更新
    app = _make_app(app_mod, [{
        "name": "A", "model": "a.gguf", "default_ctx": 131072,
        "backend": "cuda", "reasoning": "off", "mmproj": "",
    }])
    app.profiles = app_mod.merge_profiles(app.cfg)
    target = app.profiles[0]
    target["default_ctx"] = 32768
    app._persist_profile(target)
    assert len(app.cfg["profiles"]) == 1
    assert app.cfg["profiles"][0]["default_ctx"] == 32768
    assert app.cfg["profiles"][0]["scheme"] == "預設"


# ---------------------------------------------------------------- 設定匯出
def test_export_import_roundtrip_with_schemes(app_mod, tmp_path):
    from llama_launcher.profiles import export_profiles, read_export
    profiles = [
        _profile("a.gguf", scheme="預設"),
        _profile("a.gguf", scheme="code", default_ctx=32768),
        _profile("b.gguf", scheme="chat"),
    ]
    out = export_profiles(profiles)
    # 同一模型多方案都進匯出檔
    models_in_export = [p["model"] for p in out["profiles"]]
    assert models_in_export.count("a.gguf") == 2
    path = tmp_path / "prof.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    loaded = read_export(path)
    assert len(loaded) == 3
    # 匯出保留 scheme；匯入合併時以 (model, scheme) 為鍵
    code = [p for p in loaded if p["model"] == "a.gguf"
            and p.get("scheme") == "code"][0]
    assert code["default_ctx"] == 32768
