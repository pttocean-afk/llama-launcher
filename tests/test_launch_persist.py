"""啟動時把主畫面參數存回模型設定（on_launch 持久化行為）+ DPI 縮放 helpers。"""
from __future__ import annotations

import threading

import pytest

import llama_launcher.ui_scale as ui_scale_mod


def _import_app(tmp_path, monkeypatch):
    """Import llama_launcher.app with an isolated data and model dir
    (same isolation as tests/test_reasoning_effort.py)."""
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


def _profile(**overrides) -> dict:
    p = {
        "name": "Test Model",
        "model": "test.gguf",
        "mmproj": "",
        "vision_enabled": False,
        "default_ctx": 131072,
        "reasoning": "off",
        "reasoning_effort": "default",
        "backend": "cuda",
        "starred": True,
        "favorite_order": 0,
    }
    p.update(overrides)
    return p


def _fake_app(app_mod, profile):
    """Minimal app stand-in：不需要 Tk widgets 即可跑 on_launch 的存檔路徑。"""
    app = app_mod.LauncherApp.__new__(app_mod.LauncherApp)
    app.cfg = {"profiles": [dict(profile)]}
    app.profiles = list(app_mod.merge_profiles(app.cfg))
    app.server_lock = threading.Lock()
    app.favorite_profiles = list(app.profiles)
    # UI 存取點全部 stub 掉；只驗證存檔邏輯
    app.current_profile = lambda: dict(profile)
    app.refresh_listbox = lambda *a, **k: None
    app.update_detail = lambda *a, **k: None
    app._update_server_ui = lambda: None
    app.show_window = lambda *a, **k: None
    app._reload_embedded_log = lambda *a, **k: None

    class _FakeServer:
        def __init__(self):
            self.started = None

        def start(self, profile, ctx_label):
            self.started = (dict(profile), ctx_label)
            return True, "ok"

    app.server = _FakeServer()
    return app


def test_on_launch_persists_ctx_and_controls_to_profile(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.messagebox, "showerror", lambda *a, **k: None)
    profile = _profile(default_ctx=32768, reasoning="off")
    app = _fake_app(app_mod, profile)

    # 使用者在主畫面改了 Context 與思考強度後按下啟動
    class _Vars:
        def __init__(self, value):
            self._value = value
        def get(self):
            return self._value
    app.ctx_var = _Vars("224")
    app.backend_var = _Vars("CUDA")
    app.vision_var = _Vars(False)
    app.effort_var = _Vars("high")

    app.on_launch()

    saved = app.cfg["profiles"][0]
    # Context 從 32K 被主畫面的 224K 覆寫並持久化
    assert saved["default_ctx"] == 224 * 1024
    assert saved["reasoning"] == "on"
    assert saved["reasoning_effort"] == "high"
    assert saved["backend"] == "cuda"
    # 啟動用的也是同一份設定
    started_profile, started_ctx = app.server.started
    assert started_profile["default_ctx"] == 224 * 1024
    assert started_ctx == "224"


def test_on_launch_invalid_context_aborts_without_saving(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.messagebox, "showerror", lambda *a, **k: None)
    profile = _profile(default_ctx=32768)
    app = _fake_app(app_mod, profile)

    class _Vars:
        def __init__(self, value):
            self._value = value
        def get(self):
            return self._value

    app.ctx_var = _Vars("not-a-number")
    app.backend_var = _Vars("CUDA")
    app.vision_var = _Vars(False)
    app.effort_var = _Vars("off")

    app.on_launch()

    # 格式錯誤：不啟動、也不改寫設定
    assert app.server.started is None
    assert app.cfg["profiles"][0]["default_ctx"] == 32768


def test_on_launch_thinking_off_resets_effort(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod.messagebox, "showerror", lambda *a, **k: None)
    profile = _profile(reasoning="on", reasoning_effort="high")
    app = _fake_app(app_mod, profile)

    class _Vars:
        def __init__(self, value):
            self._value = value
        def get(self):
            return self._value

    app.ctx_var = _Vars("128")
    app.backend_var = _Vars("Vulkan")
    app.vision_var = _Vars(False)
    app.effort_var = _Vars("off")

    app.on_launch()

    saved = app.cfg["profiles"][0]
    assert saved["reasoning"] == "off"
    assert saved["reasoning_effort"] == "default"
    assert saved["backend"] == "vulkan"
    assert saved["default_ctx"] == 128 * 1024


# ---------------------------------------------------------------------------
# DPI 縮放 helpers（headless 可測）
# ---------------------------------------------------------------------------

def test_dpi_scale_factor_bounds():
    ui_scale_mod.DpiScale.factor = 1.0

    class _Root:
        def __init__(self, dpi):
            self._dpi = dpi
        def winfo_fpixels(self, _spec):
            return self._dpi
        def winfo_screenwidth(self):
            return 1920
        def winfo_screenheight(self):
            return 1080

    assert ui_scale_mod.DpiScale.init(_Root(96)) == 1.0
    assert ui_scale_mod.DpiScale.init(_Root(144)) == 1.5
    assert ui_scale_mod.DpiScale.init(_Root(192)) == 2.0
    # 異常值都被夾住
    assert ui_scale_mod.DpiScale.init(_Root(960)) == 3.0
    assert ui_scale_mod.DpiScale.init(_Root(48)) == 0.5
    assert ui_scale_mod.DpiScale.init(_Root(0)) == 1.0
    assert ui_scale_mod.DpiScale.init(_Root(float("nan"))) == 1.0
    ui_scale_mod.DpiScale.factor = 1.0


def test_s_scales_pixels():
    ui_scale_mod.DpiScale.factor = 1.5
    assert ui_scale_mod.S(58) == 87
    assert ui_scale_mod.S(1) == 2  # 永不回 0
    ui_scale_mod.DpiScale.factor = 1.0
    assert ui_scale_mod.S(58) == 58
    ui_scale_mod.DpiScale.factor = 1.0


def test_fit_window_size_clamps_to_screen():
    ui_scale_mod.DpiScale.factor = 1.0

    class _Root:
        def winfo_screenwidth(self):
            return 1366
        def winfo_screenheight(self):
            return 768
        def winfo_fpixels(self, _spec):
            return 96.0

    # 大視窗被 clamp 進 90% 螢幕範圍（寬 1180 < 1229 不受影響，高超過）
    w, h = ui_scale_mod.fit_window_size(_Root(), 1180, 820)
    assert (w, h) == (1180, 691)
    # 超寬視窗也會被夾
    w, h = ui_scale_mod.fit_window_size(_Root(), 3000, 400)
    assert (w, h) == (1229, 400)
    # minsize 用 ratio=1.0 時只夾到螢幕大小
    w, h = ui_scale_mod.fit_window_size(_Root(), 980, 700, screen_ratio=1.0)
    assert (w, h) == (980, 700)
    # 小視窗不受影響
    assert ui_scale_mod.fit_window_size(_Root(), 520, 470) == (520, 470)
