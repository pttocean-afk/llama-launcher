#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama Launcher — cross-platform llama.cpp tray application.

The application stores profiles, settings, logs, and control secrets in the
host user-data directory. llama.cpp runtimes and GGUF models remain external
and are selected during first-run setup.
"""
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from .config import load_profiles, load_settings, save_json
from .diagnostics import collect_diagnostics, diagnostics_dict
from .host import (
    IS_WINDOWS,
    autostart_enabled,
    hidden_run_kwargs,
    list_llama_servers,
    llama_server_filename,
    nvidia_smi_path,
    open_external,
    pid_is_running,
    set_autostart,
    terminate_process_tree,
)
from .launch_args import (
    DEFAULT_CACHE_RAM_MB,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_SCHEME,
    KV_OPTIONS,
    REASONING_EFFORTS,
    build_reasoning_args,
    build_server_args,
    format_command,
    kv_label_from_mode,
    kv_mode_from_profile,
    normalize_scheme,
    parallel_from_profile,
    preserve_unmanaged_extra_args,
    profile_key,
    reasoning_effort_value,
    server_settings_from_dict,
)
from .migration import (
    detect_legacy_dir,
    merge_legacy_into_config,
    migrate_legacy_data,
    plan_migration,
)
from .paths import (
    api_key_path,
    logs_dir,
    profiles_path,
    resource_dir,
    settings_path,
    token_path,
)
from .remote_setup import configure_remote_access
from .security import ensure_control_token, read_api_key, write_api_key
from .tailscale import TailscaleManager

# 單一實例檢查用的本機 port（不會跟 8080 衝突）
SINGLE_INSTANCE_PORT = 45678

try:
    import pystray
    from PIL import Image, ImageDraw
    HAVE_TRAY = True
except Exception:
    HAVE_TRAY = False

APP_DIR = resource_dir()
CONFIG_PATH = profiles_path()
SETTINGS_PATH = settings_path()
LOGS_DIR = logs_dir()
CONTROL_TOKEN_PATH = token_path()
ICON_PNG = APP_DIR / "assets" / "llama-launcher-icon.png"
ICON_ICO = APP_DIR / "assets" / "llama-launcher-icon.ico"

# llama.cpp root and backend-specific runtimes are local machine settings.
def _load_settings() -> dict:
    return load_settings(SETTINGS_PATH)


def _save_settings(settings: dict) -> None:
    save_json(SETTINGS_PATH, settings)


def _server_path(root: Path, backend: str) -> Path:
    settings = _load_settings()
    configured = settings.get(f"{backend}_server")
    if configured:
        candidate = Path(configured)
        if candidate.exists():
            return candidate
    if backend == "vulkan":
        legacy = root / "benchmark-runtimes" / "b10509-vulkan" / llama_server_filename()
        if legacy.exists():
            return legacy
    return root / llama_server_filename()


def _find_llama_root() -> Path:
    settings = _load_settings()
    if settings.get("llama_dir"):
        candidate = Path(settings["llama_dir"])
        if (candidate / llama_server_filename()).exists():
            return candidate
    for candidate in (APP_DIR, APP_DIR.parent):
        if (candidate / llama_server_filename()).exists():
            return candidate
    return APP_DIR.parent if (APP_DIR.parent / "models").exists() else APP_DIR


LLAMA_DIR = _find_llama_root()
MODELS_DIR = LLAMA_DIR / "models"
LLAMA_SERVER = _server_path(LLAMA_DIR, "cuda")
VULKAN_SERVER = _server_path(LLAMA_DIR, "vulkan")
# 推理 port 可在全域設定調整；PORT 是記憶體中的目前值，
# 改設定後呼叫 update_server_settings() 重新讀取。
PORT = server_settings_from_dict(_load_settings().get("server"))["port"]
CONTROL_PORT = 8765
API_KEY_PATH = api_key_path()
_MODEL_INVENTORY_CACHE: tuple[list[str], list[str], dict[str, int]] | None = None


def invalidate_model_inventory() -> None:
    """讓下一次明確掃描重新讀 models 目錄；一般UI互動只使用快取。"""
    global _MODEL_INVENTORY_CACHE
    _MODEL_INVENTORY_CACHE = None


def update_llama_dir(new_dir: str) -> bool:
    """Update the local llama.cpp root. Runtime paths are host-specific."""
    global LLAMA_DIR, MODELS_DIR, LLAMA_SERVER, VULKAN_SERVER
    path = Path(new_dir).resolve()
    if not (path / llama_server_filename()).exists():
        return False
    settings = _load_settings()
    settings["llama_dir"] = str(path)
    _save_settings(settings)
    LLAMA_DIR = path
    MODELS_DIR = path / "models"
    LLAMA_SERVER = _server_path(path, "cuda")
    VULKAN_SERVER = _server_path(path, "vulkan")
    invalidate_model_inventory()
    return True


# 每次 GPU 啟動前的 VRAM 預檢與清理。整組政策存在 settings.json 的
# vram_preflight 區塊（每台電腦各自設定）：
#   mode:          off（預設，不做任何事）/ warn（超標只警告）/ strict（清理+擋住）
#   gpu_limits_mb: 每張 GPU 的 used VRAM 上限（MB，依 nvidia-smi index 排序）
#   kill_processes: strict 模式下啟動前強制結束的程序（僅 Windows taskkill）
#   comfyui:       啟動前檢查／停止 WSL 內的 ComfyUI service（僅 Windows）
# 預設 off：不殺任何程式、不擋啟動；需要乾淨基線的機器自行開嚴格模式。
NVIDIA_SMI = nvidia_smi_path()
VRAM_PREFLIGHT_WAIT_SECONDS = 10
VRAM_PREFLIGHT_MODES = ("off", "warn", "strict")
VRAM_CLEANUP_PROCESS_NAMES = [
    "NVIDIA Overlay.exe",
    "nvsphelper64.exe",
    "Blitz.exe",
    "steam.exe",
    "steamwebhelper.exe",
    "Riot Client.exe",
    "RiotClientServices.exe",
    "RiotClientCrashHandler.exe",
    "EdgeGameAssist.exe",
]


def vram_preflight_config() -> dict:
    """讀取並正規化 settings.json 的 vram_preflight 區塊。"""
    raw = _load_settings().get("vram_preflight") or {}
    mode = str(raw.get("mode") or "off").strip().lower()
    limits = []
    for value in raw.get("gpu_limits_mb") or []:
        try:
            n = int(str(value).strip())
            limits.append(max(0, n))
        except (TypeError, ValueError):
            limits.append(None)
    comfy = {
        "enabled": False,
        "distro": "Ubuntu",
        "service": "comfyui-raylight.service",
    }
    if isinstance(raw.get("comfyui"), dict):
        c = raw["comfyui"]
        comfy["enabled"] = bool(c.get("enabled", False))
        comfy["distro"] = str(c.get("distro") or "Ubuntu").strip() or "Ubuntu"
        comfy["service"] = str(c.get("service") or "").strip() or comfy["service"]
    return {
        "mode": mode if mode in VRAM_PREFLIGHT_MODES else "off",
        "gpu_limits_mb": limits,
        "kill_processes": [
            str(v).strip() for v in (raw.get("kill_processes") or []) if str(v).strip()
        ],
        "comfyui": comfy,
    }


def update_server_settings() -> None:
    """全域設定存檔後呼叫：把新的 server 區塊（port 等）載入記憶體。"""
    global PORT
    PORT = server_settings_from_dict(_load_settings().get("server"))["port"]


def current_server_settings() -> dict:
    """合併 settings.json 的 server 區塊與 secrets/ 裡的 api-key。"""
    settings = server_settings_from_dict(_load_settings().get("server"))
    settings["api_key"] = read_api_key(API_KEY_PATH)
    return settings

# Context 直接以 K 為單位輸入，例如 224 = 229376 tokens。
CONTEXT_MIN_K = 8
CONTEXT_MAX_K = 1024


def parse_context_k(value: str) -> int:
    """把首頁／設定輸入的 K 數字轉為 tokens；格式或範圍錯誤時丟 ValueError。"""
    text = str(value).strip().upper()
    if text.endswith("K"):
        text = text[:-1].strip()
    if not re.fullmatch(r"\d+", text):
        raise ValueError("Context 請輸入 K 單位整數，例如 224")
    context_k = int(text)
    if not CONTEXT_MIN_K <= context_k <= CONTEXT_MAX_K:
        raise ValueError(
            f"Context 必須介於 {CONTEXT_MIN_K}K～{CONTEXT_MAX_K}K")
    return context_k * 1024


def format_context_k(tokens: int) -> str:
    """把 profile token數轉成首頁／設定顯示的 K 整數。"""
    return str(max(1, int(tokens) // 1024))


def profile_vision_enabled(profile: dict) -> bool:
    """舊 profile沒有 vision_enabled時維持既有行為：有mmproj即預設開啟。"""
    return bool(profile.get("mmproj")) and bool(profile.get("vision_enabled", True))


# 預設組態（新模型未設定時的起點；對應原本 bat 的參數）。
# 同一模型可以有多個「方案」(scheme)，各自獨立保存這組欄位。
DEFAULT_PROFILE = {
    "name": "",
    "model": "",            # 相對 models\ 的檔名
    "scheme": DEFAULT_SCHEME,  # 啟動方案名（預設 / code / chat / ...）
    "mmproj": "",           # 相對 models\ 的檔名，可空
    "vision_enabled": False, # 保留mmproj配對，但可在啟動時暫停載入
    "default_ctx": 131072,
    "reasoning": "off",     # off / on / auto
    "reasoning_effort": "default",  # default / minimal / low / medium / high / xhigh / max
    "reasoning_format": "auto",  # auto / none / deepseek / deepseek-legacy
    "reasoning_preserve": "default",  # default / on / off
    "gpu_split": "",        # 各 GPU 層數（逗號分隔）；空 = 自動
    "backend": "cuda",      # cuda / vulkan
    "jinja": False,          # 使用 GGUF 內建 Jinja chat template
    "extra_args": "-ctk q4_0 -ctv q4_0 --parallel 1",
    "kv_mode": "q4",        # f16 / q8 / q5 / q4 / iq4_nl / custom
    "mtp": False,
    "spec_draft_n_max": "",  # 空 = llama.cpp 預設（MTP 建議 5）
    "flash_attn": "auto",   # auto / on / off
    "kv_unified": True,
    "fit": "on",            # on / off（llama.cpp 預設 on）
    "threads": "",          # 空 = llama.cpp 預設
    "threads_batch": "",
    "ctx_checkpoints": "",
    "parallel": 1,
    "ngl": 999,
    # 採樣預設（server 端 completion 預設值；空 = llama.cpp 預設）
    "temp": "",
    "top_p": "",
    "top_k": "",
    "min_p": "",
    "presence_penalty": "",
    "repeat_penalty": "",
    "raw_args": "",         # 完整參數模式：非空時覆蓋上方所有參數（bat 式）
    "starred": False,       # ★ 置頂
}


# ---------------------------------------------------------------- 設定檔
def load_config() -> dict:
    return load_profiles(CONFIG_PATH)


def save_config(cfg: dict) -> None:
    save_json(CONFIG_PATH, cfg)


# ---------------------------------------------------------------- 模型掃描
def _is_mmproj(name: str) -> bool:
    """判斷是否為視覺投影檔：檔名含 mmproj（不限開頭，如 Agents-A1-mmproj.gguf）。"""
    return "mmproj" in name.lower()

def scan_gguf_files() -> list[str]:
    """掃描 models 目錄下的 .gguf，排除 mmproj。"""
    models, _mmprojs, _sizes = _model_inventory()
    return list(models)


def scan_mmproj_files() -> list[str]:
    _models, mmprojs, _sizes = _model_inventory()
    return list(mmprojs)


def _model_inventory() -> tuple[list[str], list[str], dict[str, int]]:
    """一次掃描模型目錄並快取名稱與大小，避免每次點選/搜尋都碰磁碟。

    遞迴掃描子資料夾（網友會自行分類，如 models\\Coding\\qwen.gguf），
    名稱一律是相對於 models 的 POSIX 風格路徑（"Coding/qwen.gguf"）；
    頂層檔案名稱不變（"qwen.gguf"），所以既有 profiles.json 完全相容。
    不跟目錄符號連結，避免循環或重複掃描。"""
    global _MODEL_INVENTORY_CACHE
    if _MODEL_INVENTORY_CACHE is not None:
        return _MODEL_INVENTORY_CACHE
    models: list[str] = []
    mmprojs: list[str] = []
    sizes: dict[str, int] = {}
    if MODELS_DIR.exists():
        found: list[tuple[str, Path]] = []
        try:
            for dirpath, dirnames, filenames in os.walk(MODELS_DIR):
                # 排序保證掃描順序穩定（不同次啟動、不同檔案系統都一致）
                dirnames.sort(key=str.lower)
                for fname in sorted(filenames, key=str.lower):
                    if not fname.lower().endswith(".gguf"):
                        continue
                    f = Path(dirpath) / fname
                    # 相對路徑當名稱；頂層檔案就只是檔名
                    rel = f.relative_to(MODELS_DIR).as_posix()
                    found.append((rel, f))
        except OSError:
            pass
        found.sort(key=lambda item: item[0].lower())
        for rel, f in found:
            try:
                sizes[rel] = f.stat().st_size
            except OSError:
                sizes[rel] = 0
            # mmproj 判定只看檔名；資料夾名稱含 mmproj 不影響分類
            (mmprojs if _is_mmproj(f.name) else models).append(rel)
    _MODEL_INVENTORY_CACHE = (models, mmprojs, sizes)
    return _MODEL_INVENTORY_CACHE


def model_file(name: str) -> Path:
    """把 profile/model 名稱（支援子資料夾相對路徑）轉成實際檔案路徑。"""
    rel = Path(str(name or "").replace("\\", "/"))
    # 防呆：相對路徑不允許逃出 models 目錄（如 ..\\秘密.gguf）
    if not rel.is_absolute() and ".." in rel.parts:
        return MODELS_DIR / "___outside_models_dir___"
    return MODELS_DIR / rel


def relative_model_name(path: Path) -> str:
    """把使用者挑選的檔案轉成存檔用的名稱：models 內取相對路徑，外面退回檔名。"""
    try:
        return path.resolve().relative_to(MODELS_DIR.resolve()).as_posix()
    except (ValueError, OSError):
        return path.name


def format_file_size(nbytes: int) -> str:
    """把 bytes 轉成好讀的大小（G / M）。"""
    gb = nbytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.1f}G"
    mb = nbytes / (1024 ** 2)
    return f"{mb:.0f}M"


def model_size_text(name: str) -> str:
    """回傳 models 下某檔案的容量文字；讀不到回空字串。"""
    _models, _mmprojs, sizes = _model_inventory()
    size = sizes.get(name, 0)
    return format_file_size(size) if size else ""


def delete_file_to_recycle_bin(path: Path) -> bool:
    """用 Windows SHFileOperation 把檔案丟進資源回收筒（可還原）。

    比直接 os.remove 安全：誤刪還有機會救回來。
    失敗回 False（例如檔案被占用），不會硬刪。
    """
    try:
        import ctypes
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("wFunc", wintypes.UINT),
                ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR),
                ("fFlags", ctypes.c_ushort),
                ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", ctypes.c_void_p),
                ("lpszProgressTitle", wintypes.LPCWSTR),
            ]

        FO_DELETE = 3
        FOF_ALLOWUNDO = 0x40      # 允許復原 → 進資源回收筒
        FOF_NOCONFIRMATION = 0x10
        FOF_SILENT = 0x4
        src = str(path) + "\0"    # SHFileOperation 需要雙重 null 結尾
        op = SHFILEOPSTRUCTW()
        op.wFunc = FO_DELETE
        op.pFrom = src
        op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        return result == 0 and not op.fAnyOperationsAborted
    except Exception:
        return False


def guess_mmproj(model_name: str, mmproj_list: list[str]) -> str:
    """依檔名關鍵字猜 mmproj：先查已知規則表，再退回 token 交集比對。"""
    RULES = [
        ("agents-a1", "Agents-A1-mmproj.gguf"),
        ("huihui", "mmproj-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf"),
        ("thinkingcap", "mmproj-ThinkingCap-Qwen3.6-27B-f16.gguf"),
        ("gemma-4-26b", "mmproj-gemma4-26b-F16.gguf"),
        ("gemma-4-12b", "mmproj-gemma4-12b-F16.gguf"),
        ("qwen3.6-35b", "mmproj-qwen35-F16.gguf"),
        ("qwen35", "mmproj-qwen35-F16.gguf"),
        ("35b-a3b", "mmproj-qwen35-F16.gguf"),
        ("27b", "mmproj-model-f16.gguf"),
    ]
    base = model_name.lower()
    # mmproj 名稱可能是 "Coding/mmproj-x.gguf"，規則表比對用檔名部分
    mmproj_by_base = {m.rsplit("/", 1)[-1].lower(): m for m in mmproj_list}
    for key, mm in RULES:
        target = mmproj_by_base.get(mm.lower())
        if key in base and target:
            return target
    model_tokens = {t for t in re.split(r"[^a-z0-9]+", base) if len(t) >= 3}
    for mm in mmproj_list:
        key = re.sub(r"^mmproj[-_]?", "", mm.rsplit("/", 1)[-1], flags=re.I)
        key = re.sub(r"\.gguf$", "", key, flags=re.I).lower()
        mm_tokens = {t for t in re.split(r"[^a-z0-9]+", key) if len(t) >= 3}
        common = model_tokens & mm_tokens
        if len(common) >= 2:
            return mm
    return ""


def merge_profiles(cfg: dict) -> list[dict]:
    """已存設定 + 掃描到的模型合併成顯示清單（不覆寫已存設定）。

    同一模型可以有多個「方案」(scheme)：cfg 裡同 model 的多個 profile
    都會保留；模型檔存在但從未設定時，產生一個「預設」方案。"""
    saved_by_model: dict[str, list[dict]] = {}
    for p in cfg.get("profiles", []):
        if p.get("model"):
            saved_by_model.setdefault(str(p["model"]), []).append(p)
    found = scan_gguf_files()
    mmprojs = scan_mmproj_files()
    merged = []
    for name in found:
        if name in saved_by_model:
            for p in saved_by_model[name]:
                q = dict(p)
                q["scheme"] = normalize_scheme(p.get("scheme"))
                q["configured"] = True
                merged.append(q)
        else:
            p = dict(DEFAULT_PROFILE)
            p["name"] = name.replace(".gguf", "")
            p["model"] = name
            p["mmproj"] = guess_mmproj(name, mmprojs)
            p["default_ctx"] = 131072
            p["configured"] = False
            merged.append(p)
    # 舊檔沒有 parallel 欄位時由 extra_args 的 --parallel 推斷。
    for p in merged:
        if "parallel" not in p:
            p["parallel"] = parallel_from_profile(p)
    # ★ 置頂方案依使用者指定順序排列；未置頂按模型檔名 + 方案名排序。
    merged.sort(key=lambda p: (
        not bool(p.get("starred")),
        int(p.get("favorite_order", 999999)) if p.get("starred") else 999999,
        str(p.get("model", "")).lower(),
        p.get("scheme") or DEFAULT_SCHEME,
    ))
    return merged


# ---------------------------------------------------------------- 工具
def enable_dpi_awareness():
    """宣告 system-DPI-aware，避免 Windows 位圖縮放導致字體模糊。

    刻意不用 per-monitor v2（SetProcessDpiAwareness(2)）：Tk 8.6 只在啟動時
    讀一次 DPI，窗口移動/跨 DPI 邊界時不會重新縮放，Windows 被迫對 Tk 輸出
    做額外縮放處理，會讓拖動窗口明顯卡頓。system-DPI-aware 下 Tk 自己用
    `tk scaling` 調整字體，拖動最流暢。"""
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# ---------------------------------------------------------------- DPI 縮放
# DpiScale / S / fit_window_size 的實作與說明見 ui_scale.py。
from .ui_scale import DpiScale, S, fit_window_size


class SingleInstance:
    """單一實例檢查：重複啟動時通知舊實例顯示視窗並退出。

    用固定 port listen 判斷是否已有實例；新實例連上去送 "show"，
    舊實例收到後把主視窗從托盤叫出來。
    """

    def __init__(self, on_show=None):
        self.on_show = on_show
        self._sock = None
        self.is_primary = False

    def acquire(self) -> bool:
        """嘗試成為唯一實例。成功 True；失敗通知舊實例並回 False。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", SINGLE_INSTANCE_PORT))
            sock.listen(1)
            sock.settimeout(1.0)
        except OSError:
            sock.close()
            # 已有實例在跑 → 通知它顯示
            try:
                c = socket.create_connection(
                    ("127.0.0.1", SINGLE_INSTANCE_PORT), timeout=1)
                c.sendall(b"show")
                c.close()
            except Exception:
                pass
            return False
        self._sock = sock
        self.is_primary = True
        threading.Thread(target=self._listen, daemon=True).start()
        return True

    def _listen(self):
        while self._sock is not None:
            try:
                conn, _ = self._sock.accept()
                try:
                    data = conn.recv(16)
                    if data.strip() == b"show" and self.on_show:
                        self.on_show()
                finally:
                    conn.close()
            except socket.timeout:
                continue
            except OSError:
                break

    def release(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


def port_in_use(port: int) -> bool:
    """直接 socket 連線測試 port 是否有人監聽（比 parse netstat 可靠）。"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 本機連線測試 0.3s 綽綽有餘；縮短避免主執行緒被卡住。
    s.settimeout(0.3)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# ---------------------------------------------------------------- ServerManager
def backend_requires_vram_preflight(backend: str) -> bool:
    """CUDA and Vulkan both clean VRAM before every server start."""
    return str(backend).strip().lower() in {"cuda", "vulkan"}


class ServerManager:
    """管理 llama-server 背景 process：無視窗啟動、log 導流、狀態查詢。"""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.external_pid: int | None = None
        self.external_command_line = ""
        self.externally_adopted = False
        self.log_path: Path | None = None
        self.log_fh = None
        self.started_at: float | None = None
        self.profile_name = ""
        self.backend = ""
        self.degraded_reason = ""
        self.degraded_warning_shown = False
        self._health_scan_offset = 0
        self._health_scan_size = -1
        self.preflight_summary = ""

    @staticmethod
    def _pid_is_running(pid: int | None) -> bool:
        return pid_is_running(pid)

    @property
    def running(self) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            return True
        if self.external_pid and self._pid_is_running(self.external_pid):
            return True
        if self.external_pid:
            self.external_pid = None
            self.external_command_line = ""
            self.externally_adopted = False
        return False

    def adopt_existing_server(self) -> tuple[bool, str]:
        """Adopt the unique llama-server process on the configured inference port."""
        if self.running:
            return True, f"已管理 PID {self.pid_text()}"
        matches = list_llama_servers(PORT)
        if len(matches) != 1:
            if not matches:
                return False, f"沒有命令列明確使用port {PORT}的llama-server"
            return False, (f"找到 {len(matches)} 個port {PORT}候選，"
                           "為安全起見不自動接管")

        item = matches[0]
        pid = item.pid
        if not self._pid_is_running(pid):
            return False, "候選llama-server已結束"
        command = item.command_line
        model_match = re.search(r'(?:^|\s)-m\s+(?:"([^"]+)"|(\S+))', command)
        model_path = (model_match.group(1) or model_match.group(2)) if model_match else ""
        model_name = model_path.replace("\\", "/").rsplit("/", 1)[-1]
        self.external_pid = pid
        self.external_command_line = command
        self.externally_adopted = True
        self.profile_name = model_name.removesuffix(".gguf") or "External llama-server"
        self.backend = "VULKAN" if ("vulkan" in command.lower() or "--device Vulkan" in command) else "CUDA"
        self.started_at = None
        self.degraded_reason = ""
        self.degraded_warning_shown = False

        # 讓Log頁接回仍由child process持續寫入的原log，但跳過既有啟動內容。
        self.log_path = None
        if LOGS_DIR.exists() and model_name:
            for candidate in sorted(LOGS_DIR.glob("llama-server-*.log"),
                                    key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    with open(candidate, "rb") as fh:
                        header = fh.read(4096).decode("utf-8", errors="replace")
                    if model_name in header:
                        self.log_path = candidate
                        self._health_scan_offset = candidate.stat().st_size
                        self._health_scan_size = self._health_scan_offset
                        break
                except OSError:
                    continue
        return True, f"已接管 {self.profile_name} (PID {pid})"

    @staticmethod
    def _run_hidden(args: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **hidden_run_kwargs(),
        )

    def query_gpu_memory_mb(self) -> list[int] | None:
        """回傳 nvidia-smi 依 index 排列的 used VRAM MB；查詢失敗回 None。"""
        if NVIDIA_SMI is None or not NVIDIA_SMI.exists():
            return None
        try:
            result = self._run_hidden([
                str(NVIDIA_SMI),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ])
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        values = []
        for line in result.stdout.splitlines():
            match = re.search(r"\d+", line)
            if match:
                values.append(int(match.group()))
        return values if values else None

    def query_gpu_count(self) -> int | None:
        """nvidia-smi 可見的 GPU 數；讀不到回 None。"""
        values = self.query_gpu_memory_mb()
        return len(values) if values else None

    def query_gpu_processes(self, gpu_index: int) -> list[str]:
        """列出綁在指定 GPU（nvidia-smi index）的 process，供擋住啟動時說明。"""
        try:
            gpu_result = self._run_hidden([
                str(NVIDIA_SMI),
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ])
            app_result = self._run_hidden([
                str(NVIDIA_SMI),
                "--query-compute-apps=pid,process_name,gpu_uuid",
                "--format=csv,noheader,nounits",
            ])
        except (OSError, subprocess.SubprocessError):
            return []
        if gpu_result.returncode != 0 or app_result.returncode != 0:
            return []
        target_uuid = ""
        for line in gpu_result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 1)]
            if len(parts) == 2 and parts[0] == str(gpu_index):
                target_uuid = parts[1]
                break
        if not target_uuid:
            return []
        owners = []
        for line in app_result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) != 3 or parts[2] != target_uuid:
                continue
            owners.append(f"{Path(parts[1]).name} (PID {parts[0]})")
        return sorted(set(owners))

    def _stop_comfyui_if_active(self, distro: str, service: str) -> tuple[bool, str]:
        """Stop the configured WSL ComfyUI service (Windows hosts only)."""
        if not IS_WINDOWS:
            return True, "ComfyUI check skipped (not a Windows host)"
        check_args = ["wsl.exe", "-d", distro, "--",
                      "systemctl", "is-active", "--quiet", service]
        try:
            check = self._run_hidden(check_args)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"無法檢查 ComfyUI service：{exc}"
        if check.returncode != 0:
            return True, "ComfyUI inactive"
        stop_args = ["wsl.exe", "-d", distro, "--",
                     "systemctl", "stop", service]
        try:
            stopped = self._run_hidden(stop_args, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"停止 ComfyUI 失敗：{exc}"
        if stopped.returncode != 0:
            detail = (stopped.stderr or stopped.stdout).strip()
            return False, f"停止 ComfyUI 失敗：{detail or 'unknown error'}"
        return True, "ComfyUI stopped"

    def _close_vram_cleanup_allowlist(self, names: list[str]) -> None:
        """End the configured processes (taskkill, Windows hosts only)."""
        if not IS_WINDOWS:
            return
        for name in names:
            try:
                self._run_hidden(["taskkill.exe", "/IM", name, "/T", "/F"], timeout=10)
            except (OSError, subprocess.SubprocessError):
                pass

    @staticmethod
    def _limit_violations(values: list[int], limits: list) -> list[tuple[int, int, int]]:
        """回傳超標的 (index, used_mb, limit_mb) 清單。"""
        out = []
        for index, used in enumerate(values):
            if index < len(limits) and limits[index] is not None and used > limits[index]:
                out.append((index, used, limits[index]))
        return out

    def run_vram_preflight(self) -> tuple[bool, str]:
        """VRAM 預檢：政策來自 settings.json（預設 off＝不做任何事）。

        off：直接通過。
        warn：讀取 VRAM，超標只警告、不擋。
        strict：清理（ComfyUI + 指定程序）後等待回到上限內，超時才擋住。
        """
        cfg = vram_preflight_config()
        mode = cfg["mode"]
        if mode == "off":
            return True, "VRAM preflight disabled"
        limits = cfg["gpu_limits_mb"]

        before = self.query_gpu_memory_mb()
        if mode == "warn":
            if before is None:
                return True, "VRAM preflight: GPU 狀態讀取不到，繼續啟動"
            violations = self._limit_violations(before, limits)
            if violations:
                detail = ", ".join(
                    f"GPU{i} {u} MB > {l} MB" for i, u, l in violations)
                return True, f"VRAM preflight 警告：{detail}，繼續啟動"
            return True, "VRAM preflight OK"

        # strict
        if before is None:
            return False, "無法讀取 NVIDIA GPU 的 VRAM，已取消伺服器啟動"

        comfy_status = "ComfyUI check disabled"
        if cfg["comfyui"]["enabled"]:
            comfy_ok, comfy_status = self._stop_comfyui_if_active(
                cfg["comfyui"]["distro"], cfg["comfyui"]["service"])
            if not comfy_ok:
                return False, comfy_status
        self._close_vram_cleanup_allowlist(cfg["kill_processes"])

        deadline = time.monotonic() + VRAM_PREFLIGHT_WAIT_SECONDS
        after = before
        while True:
            current = self.query_gpu_memory_mb()
            if current is None:
                return False, "VRAM 清理後無法重新讀取 GPU 狀態，已取消啟動"
            after = current
            if not self._limit_violations(after, limits):
                break
            if time.monotonic() >= deadline:
                lines = []
                for index, used in enumerate(after):
                    limit = limits[index] if index < len(limits) else None
                    if limit is None or used <= limit:
                        continue
                    lines.append(f"GPU{index}：{used} MB（需 ≤ {limit} MB）")
                    owners = self.query_gpu_processes(index)
                    if owners:
                        lines.append(f"GPU{index} process：" + ", ".join(owners))
                detail = "\n".join(lines) or "VRAM 超標"
                return False, (
                    "VRAM 尚未回到安全基線，已取消伺服器啟動。\n"
                    f"{detail}\n請關閉其他 AI／GPU 程式後再試。"
                )
            time.sleep(0.5)

        parts = [f"GPU{i} {before[i]}→{after[i]} MB"
                 for i in range(len(after)) if i < len(before)]
        summary = f"VRAM preflight: {', '.join(parts)}, {comfy_status}"
        return True, summary

    def start(self, profile: dict, ctx_label: str) -> tuple[bool, str]:
        """啟動 llama-server（CREATE_NO_WINDOW，stdout/stderr 導流到 log 檔）。"""
        if self.running:
            return False, "llama-server 已在執行中"
        backend = str(profile.get("backend", "cuda")).strip().lower()
        windows_server = VULKAN_SERVER if backend == "vulkan" else LLAMA_SERVER
        if not windows_server.exists():
            return False, f"找不到 {windows_server}"
        model_path = model_file(profile["model"])
        if not model_path.exists():
            return False, f"找不到模型檔：models/{profile['model']}"
        if port_in_use(PORT):
            return False, (f"Port {PORT} 已被佔用。\n"
                           "請先關閉現有的 llama-server，\n"
                           "或按「停止伺服器」處理。")

        try:
            ctx = parse_context_k(ctx_label)
        except ValueError as exc:
            return False, str(exc)
        mmproj_path = None
        if profile_vision_enabled(profile):
            mmproj_path = model_file(profile["mmproj"])
            if not mmproj_path.exists():
                return False, f"找不到視覺模型檔：models/{profile['mmproj']}"
        # VRAM 預檢政策來自全域設定（預設 off）；CUDA/Vulkan 才適用。
        self.preflight_summary = "not required"
        if backend_requires_vram_preflight(backend):
            preflight_ok, preflight_msg = self.run_vram_preflight()
            if not preflight_ok:
                return False, preflight_msg
            self.preflight_summary = preflight_msg

        # 參數組裝與設定視窗的「最終指令預覽」共用同一個純函數。
        vulkan_gpu_count = None
        if backend == "vulkan" and IS_WINDOWS:
            vulkan_gpu_count = self.query_gpu_count()
        args = build_server_args(
            profile, ctx, current_server_settings(), model_path, mmproj_path,
            vulkan_gpu_count=vulkan_gpu_count)
        launch_cwd = windows_server.parent

        # log 檔
        LOGS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = LOGS_DIR / f"llama-server-{ts}.log"
        self.log_fh = open(self.log_path, "wb", buffering=0)
        # 寫入啟動指令（方便之後查用了什麼參數）
        full_command = format_command(windows_server, args)
        header = (f"# {datetime.now().isoformat()}  {profile.get('name','')}\n"
                  f"# {self.preflight_summary}\n"
                  f"# {full_command}\n"
                  f"{'='*80}\n").encode("utf-8")
        self.log_fh.write(header)

        try:
            self.proc = subprocess.Popen(
                [str(windows_server)] + args,
                cwd=str(launch_cwd),
                stdout=self.log_fh,
                stderr=subprocess.STDOUT,
                **hidden_run_kwargs(),
            )
        except Exception as e:
            if self.log_fh:
                self.log_fh.close()
            self.log_fh = None
            self.proc = None
            return False, f"啟動失敗：{e}"

        self.started_at = time.time()
        self.external_pid = None
        self.external_command_line = ""
        self.externally_adopted = False
        self.profile_name = profile.get("name", "")
        self.backend = backend.upper()
        self.degraded_reason = ""
        self.degraded_warning_shown = False
        self._health_scan_offset = 0
        self._health_scan_size = -1
        return True, f"已啟動 {self.profile_name} (PID {self.proc.pid})"

    def stop(self) -> str:
        """停止launcher啟動或重開tray後接管的llama-server。"""
        if not self.running:
            return "llama-server 未在執行"

        if self.proc is not None and self.proc.poll() is None:
            pid = self.proc.pid
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
            except Exception as e:
                return f"停止失敗：{e}"
            finally:
                if self.log_fh:
                    try:
                        self.log_fh.close()
                    except Exception:
                        pass
                    self.log_fh = None
            self.proc = None
            return f"已停止 (PID {pid})"

        pid = self.external_pid
        if not pid:
            return "llama-server 未在執行"
        ok, detail = terminate_process_tree(pid)
        if not ok:
            return f"停止接管的PID {pid}失敗：{detail}"
        self.external_pid = None
        self.external_command_line = ""
        self.externally_adopted = False
        return f"已停止接管的llama-server (PID {pid})"

    def uptime_text(self) -> str:
        if not self.running or self.started_at is None:
            return ""
        secs = int(time.time() - self.started_at)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"up {h}h{m:02d}m"
        return f"up {m}m{s:02d}s"

    def pid_text(self) -> str:
        if not self.running:
            return ""
        if self.proc is not None and self.proc.poll() is None:
            return str(self.proc.pid)
        return str(self.external_pid or "")


# ---------------------------------------------------------------- Remote control API
REMOTE_CONTROL_HTML = """<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>
<title>llama.cpp Launcher Control</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}html,body{max-width:100%;overflow-x:hidden}body{font:15px system-ui,-apple-system,sans-serif;background:#0b1220;color:#e8eef7;max-width:980px;margin:0 auto;padding:24px 16px}h1{margin:0;font-size:25px}h2{font-size:16px;margin:0 0 14px;color:#a9c2e2}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}.badge{padding:6px 12px;border-radius:99px;background:#26344b;color:#a9bad3}.badge.on{background:#164d37;color:#79e0a8}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.grid>div,.profile>div{min-width:0}.card{min-width:0;background:#151f30;border:1px solid #27364d;border-radius:12px;padding:16px;margin:12px 0}.metric{font-size:20px;font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.label{font-size:11px;color:#8fa3bf;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}button,select,input{font:inherit;padding:9px 12px;border-radius:7px;border:1px solid #405574;background:#202d42;color:#fff}input{width:100%;max-width:100%;min-width:0}button{cursor:pointer;background:#2869bb;border-color:#397ed4;font-weight:600}button.secondary{background:#27344a;border-color:#40516a}button.danger{background:#9e3e4d;border-color:#c95768}button:disabled{opacity:.45;cursor:not-allowed}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.row span{min-width:0;overflow-wrap:anywhere}.profile{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:12px;border:1px solid #2b3b54;border-radius:9px;margin:8px 0;background:#111a29}.profile b,.profile small{overflow-wrap:anywhere}.profile small{display:block;color:#8fa3bf;margin-top:4px}.profile button{flex:0 0 auto;white-space:nowrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:430px;overflow:auto;background:#0a101b;padding:12px;border-radius:8px;color:#b9c9dc;font-size:12px}@media(max-width:650px){body{padding:16px 10px}.card{padding:14px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.top{align-items:flex-start;gap:12px;flex-direction:column}.profile{align-items:flex-start}.profile button{padding:9px 11px}}@media(max-width:380px){.profile{flex-direction:column}.profile button{width:100%}}
</style><div class=top><div><h1>llama.cpp Launcher</h1><div style='color:#8fa3bf;margin-top:4px'>Tailscale remote control</div></div><span id=badge class=badge>Offline</span></div>
<div class=card><h2>Server status</h2><div id=metrics class=grid><div><div class=label>Model</div><div class=metric id=model>—</div></div><div><div class=label>Backend</div><div class=metric id=backend>—</div></div><div><div class=label>PID</div><div class=metric id=pid>—</div></div><div><div class=label>Uptime</div><div class=metric id=uptime>—</div></div></div><div class=row style='margin-top:16px'><button class=secondary onclick=status()>Refresh</button><button class=danger id=stopBtn onclick=stop() disabled>Stop server</button><span id=message style='color:#a9bad3'></span></div></div>
<div class=card><h2>Available profiles</h2><div id=profiles><div style='color:#8fa3bf'>Enter the token, then press Refresh.</div></div></div>
<div class=card><div class=row style='justify-content:space-between'><h2 style='margin:0'>Recent server log</h2><button class=secondary onclick=logs()>Load log</button></div><pre id=log>Press Load log to view the latest output.</pre></div>
<details class=card><summary style='cursor:pointer;color:#a9c2e2;font-weight:600'>Settings · Authentication</summary><div style='margin-top:14px'><div class=row><input id=token type=password placeholder='Paste control token' size=42><button class=secondary onclick=saveToken()>Save token</button></div><small style='display:block;color:#8fa3bf;margin-top:8px'>Token is stored only in this browser. You normally need to enter it only once.</small></div></details>
<script>
const $=id=>document.getElementById(id);const saved=localStorage.getItem('llamaToken');if(saved){$('token').value=saved;status()}
function saveToken(){localStorage.setItem('llamaToken',$('token').value.trim());status()}
async function call(path,method='GET',body){const token=$('token').value.trim();if(!token)throw Error('Please enter the control token');localStorage.setItem('llamaToken',token);const r=await fetch(path,{method,headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const text=await r.text();let d;try{d=JSON.parse(text)}catch{d={error:text}}if(!r.ok)throw Error(d.error||('HTTP '+r.status));return d}
function renderStatus(s){$('badge').textContent=s.running?'Running':'Stopped';$('badge').className='badge '+(s.running?'on':'');$('model').textContent=s.profile||'—';$('backend').textContent=s.backend||'—';$('pid').textContent=s.pid||'—';$('uptime').textContent=s.uptime||'—';$('stopBtn').disabled=!s.running}
function renderProfiles(list,running){const box=$('profiles');box.textContent='';if(!list.length){const none=document.createElement('div');none.style.color='#8fa3bf';none.textContent='No profiles found.';box.appendChild(none);return}for(const x of list){const row=document.createElement('div');row.className='profile';const info=document.createElement('div');const name=document.createElement('b');name.textContent=x.name;info.appendChild(name);const detail=document.createElement('small');let dtext=String(x.backend||'').toUpperCase()+' · '+((Number(x.default_ctx)||0)/1024).toFixed(0)+'K context · reasoning '+(x.reasoning||'');if(x.reasoning==='on'&&x.reasoning_effort&&x.reasoning_effort!=='default'){dtext+=' · effort '+x.reasoning_effort}detail.textContent=dtext;info.appendChild(detail);row.appendChild(info);const btn=document.createElement('button');btn.textContent='Start';if(!running){btn.addEventListener('click',()=>start(String(x.model||'')))}else{btn.disabled=true}row.appendChild(btn);box.appendChild(row)}}
async function status(){try{const [s,p]=await Promise.all([call('/api/status'),call('/api/profiles')]);renderStatus(s);renderProfiles(p.profiles||[],s.running);if(s.degraded)$('message').textContent='Warning: '+s.degraded}catch(e){$('message').textContent='Error: '+e.message}}
async function start(model){try{$('message').textContent='Starting...';await call('/api/start','POST',{model});await status();$('message').textContent='Started'}catch(e){$('message').textContent='Error: '+e.message}}
async function stop(){if(!confirm('Stop the current llama-server?'))return;try{$('message').textContent='Stopping...';await call('/api/stop','POST');await status();$('message').textContent='Stopped'}catch(e){$('message').textContent='Error: '+e.message}}
async function logs(){try{const d=await call('/api/logs');$('log').textContent=d.tail||'(No log available)'}catch(e){$('log').textContent='Error: '+e.message}}
setInterval(status,5000)
</script>"""


def _get_control_token() -> str:
    return ensure_control_token(CONTROL_TOKEN_PATH)


class ControlServer:
    """本機控制 API；透過 Tailscale Serve 對外，絕不直接暴露 8080。"""
    def __init__(self, app):
        self.app = app
        self.token = _get_control_token()
        self.bind_host = "127.0.0.1"
        self.bound = False
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _send(self, code, payload, content_type="application/json"):
                data = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(
                    payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", content_type + "; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authorized(self):
                supplied = self.headers.get("Authorization", "")
                return supplied.startswith("Bearer ") and compare_digest(
                    supplied[7:].strip(), parent.token)

            def _json_body(self):
                length = min(int(self.headers.get("Content-Length", "0")), 8192)
                return json.loads(self.rfile.read(length) or b"{}")

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                if path == "/":
                    self._send(200, REMOTE_CONTROL_HTML, "text/html")
                    return
                if not self._authorized():
                    self._send(401, {"error": "Bearer token required"})
                    return
                if path == "/api/status":
                    self._send(200, parent.app.remote_status())
                elif path == "/api/profiles":
                    self._send(200, {"profiles": parent.app.remote_profiles()})
                elif path == "/api/logs":
                    self._send(200, parent.app.remote_logs())
                elif path == "/api/health":
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "Not found"})

            def do_HEAD(self):
                path = urlparse(self.path).path
                if path == "/favicon.ico":
                    self.send_response(204)
                elif path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                elif path.startswith("/api/") and self._authorized():
                    self.send_response(200)
                else:
                    self.send_response(401 if path.startswith("/api/") else 404)
                self.end_headers()

            def do_POST(self):
                if not self._authorized():
                    self._send(401, {"error": "Bearer token required"})
                    return
                path = urlparse(self.path).path
                try:
                    body = self._json_body()
                    if path == "/api/start":
                        result = parent.app.remote_start(body)
                    elif path == "/api/stop":
                        result = parent.app.remote_stop()
                    else:
                        self._send(404, {"error": "Not found"})
                        return
                    if not result.get("ok") and not result.get("error"):
                        result["error"] = result.get("message", "Start/stop operation failed")
                    self._send(200 if result.get("ok") else 409, result)
                except (ValueError, KeyError, json.JSONDecodeError) as exc:
                    self._send(400, {"error": str(exc) or "Invalid JSON"})
                except Exception as exc:
                    self._send(500, {"error": f"Control request failed: {exc}"})

        self.httpd = ThreadingHTTPServer((self.bind_host, CONTROL_PORT), Handler)
        self.bound = True
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       name="llama-control-api", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------- GUI
# GPU 分配：口語選項 <-> 數值。
# 不做特定型號硬編碼（5070Ti 等只有特定使用者適用）；提供自動 + 自訂層數。
GPU_OPTIONS = [
    ("自動（讓程式決定）", ""),
    ("自訂層數分配…", "__custom__"),
]

def gpu_label_to_value(label: str) -> str:
    for name, val in GPU_OPTIONS:
        if name == label:
            return val
    # 未知 label 視為使用者直接輸入的數值（例如 "16,8"）
    return (label or "").strip()

def gpu_value_to_label(val: str) -> str:
    val = (val or "").strip()
    for name, v in GPU_OPTIONS:
        if v == val:
            return name
    # 自訂數值（如 "16,8"）直接顯示原值
    return val or "自動（讓程式決定）"


def gpu_preset_options() -> list[str]:
    """讀取全域「常用 GPU 分配」清單（settings.json 的 gpu_split_presets）。"""
    try:
        presets = _load_settings().get("gpu_split_presets") or []
        return [str(v).strip() for v in presets if str(v).strip()]
    except Exception:
        return []


def remember_gpu_preset(value: str) -> None:
    """把一個自訂 GPU 分配存進全域常用清單（去重、保留順序）。"""
    value = (value or "").strip()
    if not value:
        return
    settings = _load_settings()
    presets = [str(v).strip() for v in (settings.get("gpu_split_presets") or [])]
    if value not in presets:
        presets.append(value)
        settings["gpu_split_presets"] = presets
        _save_settings(settings)


def forget_gpu_preset(value: str) -> None:
    """從全域常用清單移除一個 GPU 分配（避免打錯的卡住）。"""
    value = (value or "").strip()
    settings = _load_settings()
    presets = [str(v).strip() for v in (settings.get("gpu_split_presets") or [])]
    if value in presets:
        presets.remove(value)
        settings["gpu_split_presets"] = presets
        _save_settings(settings)


class LogViewer(tk.Toplevel):
    """獨立 log 檢視視窗：讀取 log 檔尾部，自動更新。"""

    def __init__(self, app: "LauncherApp"):
        super().__init__(app.root)
        self.app = app
        self.title("llama-server Log")
        win_w, win_h = fit_window_size(self, S(860), S(540))
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(*fit_window_size(self, S(600), S(300), screen_ratio=1.0))

        top = tk.Frame(self, padx=8, pady=6)
        top.pack(fill="x")
        self.status_lbl = tk.Label(top, text="", font=("Segoe UI", 9),
                                   anchor="w", fg="#555")
        self.status_lbl.pack(side="left", fill="x", expand=True)
        tk.Button(top, text="開啟 log 檔", command=self.open_log_file).pack(side="right")
        tk.Button(top, text="重新載入", command=self.reload).pack(side="right", padx=4)
        tk.Button(top, text="清空檢視", command=self.clear_view).pack(side="right", padx=4)

        body = tk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.text = tk.Text(body, font=("Consolas", 9), state="disabled",
                            bg="#10141c", fg="#d5dbe5", wrap="none")
        sb = tk.Scrollbar(body, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        # 自動捲動：預設 True；使用者往上捲時停用，捲回底部時恢復
        self._auto_scroll = True
        self.text.bind("<MouseWheel>", self._on_wheel)
        self.text.bind("<Button-4>", self._on_wheel)   # Linux 滾輪上
        self.text.bind("<Button-5>", self._on_wheel)   # Linux 滾輪下
        sb.bind("<MouseWheel>", self._on_wheel)
        # 拖 scrollbar thumb 也更新狀態
        sb.config(command=self._on_scrollbar)

        self._last_offset = 0
        self._max_lines = 4000
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(600, self._poll)
        self.reload()

    def _on_wheel(self, event):
        """滾輪事件：往上捲就停用自動捲，捲到底就恢復。"""
        delta = getattr(event, "delta", 0)
        if delta == 0:
            # Linux Button-4/5
            delta = 120 if event.num == 4 else -120
        if delta > 0:
            self._auto_scroll = False
        elif self.text.yview()[1] >= 0.999:
            self._auto_scroll = True

    def _on_scrollbar(self, *args):
        """Scrollbar 拖動：交給原本的 yview，並更新自動捲狀態。"""
        # args[0] = 'moveto'/'scroll'
        if args and args[0] == "moveto":
            try:
                frac = float(args[1])
                if frac >= 0.999:
                    self._auto_scroll = True
                else:
                    self._auto_scroll = False
            except Exception:
                pass
        return self.text.yview(*args)

    def _poll(self):
        if self.winfo_exists():
            self.reload(append_only=True)
            self.after(600, self._poll)

    def reload(self, append_only=False):
        p = self.app.server.log_path
        if p is None or not p.exists():
            self.status_lbl.config(text="尚未啟動 llama-server（啟動後 log 會顯示在這裡）")
            return
        try:
            size = p.stat().st_size
            if append_only and self._last_offset == size:
                self._update_status(p)
                return
            if append_only and self._last_offset > size:
                self._last_offset = 0  # log 檔被輪替
            with open(p, "rb") as f:
                f.seek(self._last_offset)
                chunk = f.read(size - self._last_offset)
                self._last_offset = f.tell()
            text = chunk.decode("utf-8", errors="replace")
            if text:
                self.text.config(state="normal")
                self.text.insert("end", text)
                # 限制行數
                lines = int(self.text.index("end-1c").split(".")[0])
                if lines > self._max_lines:
                    self.text.delete("1.0", f"{lines - self._max_lines}.0")
                # 自動滾到底（除非使用者主動往上捲）
                if self._auto_scroll:
                    self.text.see("end")
                self.text.config(state="disabled")
            self._update_status(p)
        except Exception:
            pass

    def _update_status(self, p: Path):
        running = self.app.server.running
        if running and self.app.server.degraded_reason:
            status = f"⚠ DEGRADED  {self.app.server.degraded_reason}"
        elif running:
            status = (f"● running  {self.app.server.profile_name}  "
                      f"PID {self.app.server.pid_text()}  "
                      f"{self.app.server.uptime_text()}")
        else:
            status = "○ stopped"
        self.status_lbl.config(text=f"{status}    |    {p.name}")

    def open_log_file(self):
        if self.app.server.log_path and self.app.server.log_path.exists():
            open_external(self.app.server.log_path)

    def clear_view(self):
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")
        self._last_offset = 0

    def close(self):
        self.destroy()


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg = load_config()
        self.profiles = merge_profiles(self.cfg)
        self.tray_icon = None
        self.quitting = False
        self.server = ServerManager()
        self.server.adopt_existing_server()
        # Tk 主執行緒與 remote HTTP thread 共用，避免同時 start/stop 同一個 server。
        self.server_lock = threading.Lock()
        # psutil 全系統掃描很貴（Windows 上可能幾百 ms）；adopt 嘗試做節流。
        self._last_adopt_attempt = 0.0
        self.control_server = None
        self.control_error = ""
        try:
            self.control_server = ControlServer(self)
            self.control_server.start()
        except OSError as exc:
            self.control_server = None
            self.control_error = (
                f"Control API 無法監聽 {CONTROL_PORT}（{exc}）。"
                "Remote dashboard 將不可用。")
            logging.getLogger("llama_launcher").warning(
                "Control API bind failed on %s: %s", CONTROL_PORT, exc)
        self.log_viewer: LogViewer | None = None
        self.performance_viewer = None  # 效能分析視窗（可重複開啟，聚焦既有）

        # ---- Modern dark dashboard: favorites/control on the left, full-height log on the right.
        DpiScale.init(root)
        root.title("llama.cpp Launcher")
        win_w, win_h = fit_window_size(root, S(1180), S(820))
        root.geometry(f"{win_w}x{win_h}")
        root.minsize(*fit_window_size(root, S(980), S(700), screen_ratio=1.0))
        root.configure(bg="#0d1118")
        self.window_icon = None
        try:
            if ICON_PNG.exists():
                self.window_icon = tk.PhotoImage(file=str(ICON_PNG))
                root.iconphoto(True, self.window_icon)
            if ICON_ICO.exists():
                root.iconbitmap(default=str(ICON_ICO))
        except tk.TclError:
            pass
        root.protocol("WM_DELETE_WINDOW", self.hide_to_tray)
        if self.control_error:
            messagebox.showwarning(
                "Remote control unavailable",
                f"{self.control_error}\n\n"
                f"Control API 需要 {CONTROL_PORT} 不被其他程式佔用。",
            )

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Dashboard.TCombobox", fieldbackground="#202838",
                        background="#202838", foreground="#eef2f8",
                        arrowcolor="#9fb1c8", bordercolor="#354055")
        style.map("Dashboard.TCombobox",
                  fieldbackground=[("readonly", "#202838")],
                  foreground=[("readonly", "#eef2f8")],
                  selectbackground=[("readonly", "#202838")],
                  selectforeground=[("readonly", "#eef2f8")])

        header = tk.Frame(root, bg="#141a24", height=S(58))
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="LLAMA  CONTROL  CENTER",
                 font=("Segoe UI", 15, "bold"), fg="#f4f7fb",
                 bg="#141a24", padx=18).pack(side="left", fill="y")
        self.status_dot = tk.Label(header, text="○  Server stopped",
                                   font=("Segoe UI", 10, "bold"), fg="#f0ad4e",
                                   bg="#141a24", padx=18)
        self.status_dot.pack(side="right", fill="y")

        dashboard = tk.PanedWindow(root, orient="horizontal", sashwidth=S(7),
                                   sashrelief="flat", bg="#0d1118",
                                   bd=0, relief="flat")
        dashboard.pack(fill="both", expand=True, padx=S(12), pady=S(12))

        left = tk.Frame(dashboard, bg="#171e2a", width=S(350),
                        highlightthickness=1, highlightbackground="#293244")
        right = tk.Frame(dashboard, bg="#0c1119",
                         highlightthickness=1, highlightbackground="#293244")
        dashboard.add(left, minsize=S(310), width=S(350))
        dashboard.add(right, minsize=S(520))

        tk.Label(left, text="★  FAVORITE MODELS", font=("Segoe UI", 10, "bold"),
                 fg="#9fb7d7", bg="#171e2a", anchor="w").pack(
                     fill="x", padx=16, pady=(16, 7))
        favorite_box = tk.Frame(left, bg="#171e2a")
        favorite_box.pack(fill="both", expand=True, padx=16)
        self.listbox = tk.Listbox(
            favorite_box, font=("Segoe UI", 10), exportselection=False,
            height=7, bg="#101620", fg="#edf2f8", selectbackground="#326fd1",
            selectforeground="white", relief="flat", bd=0,
            highlightthickness=1, highlightbackground="#303b4e",
            activestyle="none")
        # expand：視窗變高時列表吃掉多餘空間，避免下方大片空白
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.refresh_listbox()

        order_row = tk.Frame(left, bg="#171e2a")
        order_row.pack(fill="x", padx=16, pady=(6, 12))
        tk.Button(order_row, text="↑ Move up", command=lambda: self.move_favorite(-1),
                  font=("Segoe UI", 9), bg="#252f40", fg="#e7edf5",
                  activebackground="#344159", activeforeground="white",
                  relief="flat", padx=10, pady=4).pack(side="left")
        tk.Button(order_row, text="↓ Move down", command=lambda: self.move_favorite(1),
                  font=("Segoe UI", 9), bg="#252f40", fg="#e7edf5",
                  activebackground="#344159", activeforeground="white",
                  relief="flat", padx=10, pady=4).pack(side="left", padx=6)
        tk.Button(order_row, text="All models", command=self.open_model_library,
                  font=("Segoe UI", 9, "bold"), bg="#252f40", fg="#e7edf5",
                  activebackground="#344159", activeforeground="white",
                  relief="flat", padx=10, pady=4).pack(side="right")

        self.detail_text = tk.Text(
            left, height=5, font=("Consolas", 9), state="disabled",
            bg="#101620", fg="#aebbd0", relief="flat", bd=0,
            highlightthickness=1, highlightbackground="#303b4e",
            padx=10, pady=8)
        self.detail_text.pack(fill="x", padx=16, pady=(0, 12))

        controls = tk.Frame(left, bg="#171e2a")
        controls.pack(fill="x", padx=16)
        for col in range(2):
            controls.columnconfigure(col, weight=1)

        def control_label(text, row, col):
            tk.Label(controls, text=text, font=("Segoe UI", 8, "bold"),
                     fg="#8293aa", bg="#171e2a", anchor="w").grid(
                         row=row, column=col, sticky="ew",
                         padx=(0, 6) if col == 0 else (6, 0), pady=(0, 3))

        def combo(var, values, row, col):
            w = ttk.Combobox(controls, textvariable=var, values=values,
                             state="readonly", style="Dashboard.TCombobox")
            w.grid(row=row, column=col, sticky="ew",
                   padx=(0, 6) if col == 0 else (6, 0), pady=(0, 10))
            return w

        self.ctx_var = tk.StringVar(value="128")
        self.backend_var = tk.StringVar(value="CUDA")
        self.vision_var = tk.BooleanVar(value=False)
        # THINKING 下拉：off = 不思考；default/minimal/…/max = 思考 + 強度
        self.effort_var = tk.StringVar(value="off")
        control_label("CONTEXT (K)", 0, 0)
        control_label("ENABLE VISION", 0, 1)
        self.ctx_entry = tk.Entry(
            controls, textvariable=self.ctx_var, font=("Segoe UI", 10),
            bg="#202838", fg="#eef2f8", insertbackground="white",
            relief="flat", highlightthickness=1, highlightbackground="#303b4e")
        self.ctx_entry.grid(row=1, column=0, sticky="ew", padx=(0, 6),
                            pady=(0, 10), ipady=5)
        self.vision_check = tk.Checkbutton(
            controls, text="Enable vision projector", variable=self.vision_var,
            font=("Segoe UI", 9, "bold"), bg="#171e2a", fg="#dce6f3",
            activebackground="#171e2a", activeforeground="white",
            selectcolor="#253045", anchor="w")
        self.vision_check.grid(row=1, column=1, sticky="ew", padx=(6, 0),
                               pady=(0, 10))
        control_label("BACKEND", 2, 0)
        control_label("THINKING", 2, 1)
        self.backend_combo = combo(self.backend_var, ["CUDA", "Vulkan"], 3, 0)
        self.effort_combo = combo(self.effort_var,
                                  ["off"] + list(REASONING_EFFORTS), 3, 1)

        self.power_btn = tk.Button(
            left, text="▶  START SERVER", font=("Segoe UI", 12, "bold"),
            bg="#2f74d0", fg="white", activebackground="#3e86e2",
            activeforeground="white", relief="flat", bd=0, pady=11,
            command=self.on_toggle_server)
        self.power_btn.pack(fill="x", padx=16, pady=(6, 8))
        tk.Button(left, text="Open WebUI", command=self.open_chat_ui,
                  font=("Segoe UI", 10, "bold"), bg="#253045", fg="#dce6f3",
                  activebackground="#34445f", activeforeground="white",
                  relief="flat", bd=0, pady=8).pack(fill="x", padx=16)
        utility_row = tk.Frame(left, bg="#171e2a")
        # 底部也留 16px，跟左右 padx=16 一致，避免按鈕貼到面板底邊。
        utility_row.pack(fill="x", padx=16, pady=(8, 16))
        tk.Button(utility_row, text="Settings", command=self.on_global_settings,
                  font=("Segoe UI", 9, "bold"), bg="#253045", fg="#dce6f3",
                  activebackground="#34445f", activeforeground="white",
                  relief="flat", padx=6, pady=9).pack(
                      side="left", fill="x", expand=True)
        tk.Button(utility_row, text="Diagnostics", command=self.on_diagnostics,
                  font=("Segoe UI", 9, "bold"), bg="#253045", fg="#dce6f3",
                  activebackground="#34445f", activeforeground="white",
                  relief="flat", padx=6, pady=9).pack(
                      side="left", fill="x", expand=True, padx=(8, 0))
        tk.Button(utility_row, text="📊 效能分析",
                  command=self.open_performance_viewer,
                  font=("Segoe UI", 9, "bold"), bg="#253045", fg="#dce6f3",
                  activebackground="#34445f", activeforeground="white",
                  relief="flat", padx=6, pady=9).pack(
                      side="left", fill="x", expand=True, padx=(8, 0))

        log_header = tk.Frame(right, bg="#141a24", height=S(46))
        log_header.pack(fill="x")
        log_header.pack_propagate(False)
        tk.Label(log_header, text="LIVE SERVER LOG", font=("Segoe UI", 10, "bold"),
                 fg="#dfe7f2", bg="#141a24", padx=14).pack(side="left", fill="y")
        tk.Button(log_header, text="Bottom", command=self.scroll_log_to_bottom,
                  font=("Segoe UI", 9), bg="#252f40", fg="#d7e0ec",
                  activebackground="#344159", activeforeground="white",
                  relief="flat", padx=10).pack(side="right", padx=(0, 10), pady=8)
        tk.Button(log_header, text="Clear", command=self.clear_embedded_log,
                  font=("Segoe UI", 9), bg="#252f40", fg="#d7e0ec",
                  activebackground="#344159", activeforeground="white",
                  relief="flat", padx=10).pack(side="right", padx=6, pady=8)

        log_frame = tk.Frame(right, bg="#0c1119")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text = tk.Text(
            log_frame, wrap="word", font=("Cascadia Mono", 9), state="disabled",
            bg="#090d13", fg="#c7d2e1", insertbackground="white",
            selectbackground="#2d5d9f", relief="flat", bd=0, padx=12, pady=10)
        log_scroll = tk.Scrollbar(log_frame, command=self._log_scrollbar,
                                  bg="#1d2634", troughcolor="#0c1119")
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self._log_auto_scroll = True
        self._log_last_offset = 0
        self._log_max_lines = 5000
        self._log_max_chunk_bytes = 512 * 1024
        self.log_text.bind("<MouseWheel>", self._on_log_wheel, add="+")

        self.update_detail()
        self.listbox.focus_set()
        self.root.after(400, self.init_tray)
        self.root.after(800, self._poll_server_status)
        self.root.after(500, self._poll_embedded_log)
        if not LLAMA_SERVER.exists():
            self.root.after(900, self.run_first_setup)
        # First run with an empty profile list: offer to migrate the old launcher data.
        if not self.profiles:
            self.root.after(1500, self._maybe_offer_legacy_migration)

    def _maybe_offer_legacy_migration(self):
        if self.profiles:
            return
        candidates = self._candidate_legacy_dirs()
        if not candidates:
            return
        if messagebox.askyesno(
                "Import old launcher data?",
                f"Found old launcher data in\n{candidates[0]}\n\n"
                "Copy profiles and control token into the new data directory?"):
            self._do_migrate_legacy(candidates[0])

    # ---------------- setup / diagnostics
    def run_first_setup(self):
        if LLAMA_SERVER.exists():
            return
        messagebox.showinfo(
            "Welcome to Llama Launcher",
            f"Choose the llama.cpp folder containing {llama_server_filename()}.\n\n"
            "Models are expected in its models subfolder. This path is stored only on this PC.",
        )
        selected = filedialog.askdirectory(title="Choose llama.cpp folder")
        if not selected:
            return
        if not update_llama_dir(selected):
            messagebox.showerror("Invalid folder", f"The selected folder does not contain {llama_server_filename()}.")
            return
        invalidate_model_inventory()
        self.cfg = load_config()
        self.profiles = merge_profiles(self.cfg)
        self.refresh_listbox()
        self.update_detail()
        if messagebox.askyesno(
                "Remote access", "Configure secure Tailscale Serve remote access now?"):
            self.on_remote_access()

    # ---------------- legacy migration
    def _candidate_legacy_dirs(self) -> list[Path]:
        """Folders worth offering as migration sources (most specific first)."""
        candidates: list[Path] = []
        try:
            current = _load_settings().get("llama_dir")
            if current:
                candidates.extend([Path(current) / "launcher-app", Path(current)])
        except (OSError, ValueError):
            pass
        if os.name == "nt":
            drive = Path(os.environ.get("SystemDrive", "C:") or "C:")
            candidates.extend([
                drive / "llama-cpp" / "launcher-app",
                drive / "llama-cpp",
            ])
            local = os.environ.get("LOCALAPPDATA")
            if local:
                candidates.append(Path(local) / "llama-cpp" / "launcher-app")
        else:
            home = Path.home()
            candidates.extend([home / "llama-cpp" / "launcher-app", home / "llama-cpp"])
        seen: set[Path] = set()
        unique: list[Path] = []
        for cand in candidates:
            try:
                cand = cand.expanduser()
            except (OSError, ValueError):
                continue
            if cand in seen or not cand.is_dir():
                continue
            seen.add(cand)
            if detect_legacy_dir(cand):
                unique.append(cand)
        return unique

    def on_migrate_legacy(self, auto=False):
        """Migrate old-launcher profiles/settings/token into the new data dir.

        With auto=True (startup) the first detected folder migrates without
        prompts; otherwise the user picks the folder in a dialog.
        """
        candidates = self._candidate_legacy_dirs()
        legacy = candidates[0] if (auto and candidates) else None
        if legacy is None and not auto:
            if candidates:
                legacy = filedialog.askdirectory(
                    title="Choose the old launcher folder (llama-launcher.pyw + models.json)",
                    initialdir=str(candidates[0]), mustexist=True)
            else:
                legacy = filedialog.askdirectory(
                    title="Choose the old launcher folder (llama-launcher.pyw + models.json)",
                    mustexist=True)
        if not legacy:
            if not auto:
                messagebox.showinfo("No legacy folder found",
                                    "No old launcher folder with models.json was found.")
            return
        if not detect_legacy_dir(legacy):
            if not auto:
                messagebox.showwarning(
                    "Not a legacy folder",
                    f"{legacy} does not contain models.json, settings.json, or control-token.txt.")
            return
        self._do_migrate_legacy(Path(legacy))

    def _do_migrate_legacy(self, legacy: Path):
        """Run the migration and refresh the UI. Returns the result for tests."""
        data_dir = SETTINGS_PATH.parent
        plan = plan_migration(legacy, data_dir)
        if plan.is_empty:
            messagebox.showwarning(
                "Nothing to migrate",
                f"{legacy} contains no recognizable launcher data.")
            return None
        if plan.will_copy:
            confirm = (f"Copy {', '.join(plan.will_copy)} "
                       + (f"(skip existing: {', '.join(plan.will_skip)}) " if plan.will_skip else "")
                       + f"from\n{legacy}?\n\nProfiles already present in the new app keep their settings.")
        else:
            confirm = ("All files are already present in the new data directory.\n"
                       "Merge profile lists anyway?")
        if not messagebox.askyesno("Migrate old launcher data", confirm):
            return None
        result = migrate_legacy_data(legacy, data_dir)
        merged_note = ""
        added = merge_legacy_into_config(legacy, CONFIG_PATH)
        if added:
            merged_note = f"\n\nMerged {added} profile(s) not already present."
        cfg = load_config()
        if result.copied or result.skipped:
            detail = []
            if result.copied:
                detail.append("Copied: " + ", ".join(result.copied))
            if result.skipped:
                detail.append("Skipped (already present): " + ", ".join(result.skipped))
            messagebox.showinfo(
                "Migration complete",
                "\n".join(detail) + merged_note +
                "\n\nProfiles and token are now in the new data directory.",
            )
        self.cfg = cfg
        self.profiles = merge_profiles(self.cfg)
        self.refresh_listbox()
        self.update_detail()
        return result

    # ---------------- profile export / import
    def export_profiles(self, profiles: list[dict] | None = None) -> Path | None:
        """Save profiles to a JSON file. Returns the path, or None if cancelled."""
        if profiles is None:
            profiles = self.profiles
        if not profiles:
            messagebox.showinfo("Export", "No profiles to export.")
            return None
        from .profiles import export_profiles
        payload = export_profiles(profiles)
        default_name = "llama-launcher-profiles.json"
        path = filedialog.asksaveasfilename(
            title="Export profiles",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return None
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo(
            "Export complete",
            f"{len(profiles)} profile(s) saved to\n{path}")
        return Path(path)

    def import_profiles(self) -> None:
        """Load profiles from a JSON export file and merge them in."""
        from .profiles import read_export, merge_imported
        path = filedialog.askopenfilename(
            title="Import profiles",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            imported = read_export(Path(path))
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            messagebox.showerror("Import failed", f"Cannot read profile file:\n{exc}")
            return
        if not imported:
            messagebox.showinfo("Import", "No valid profiles found in file.")
            return
        current = list(self.cfg.get("profiles", []))
        merged, added, updated = merge_imported(current, imported)
        self.cfg["profiles"] = merged
        save_config(self.cfg)
        self.profiles = merge_profiles(self.cfg)
        self._normalize_favorite_orders()
        self.refresh_listbox()
        self.update_detail()
        msg = f"Imported {added} new, updated {updated} existing."
        messagebox.showinfo("Import complete", msg)

    def on_remote_access(self):
        manager = TailscaleManager()
        if manager.executable is None:
            messagebox.showerror(
                "Tailscale not found",
                "Install and sign in to Tailscale, then press Remote Access again.\n\n"
                "https://tailscale.com/download/windows",
            )
            return
        result = configure_remote_access(manager, CONTROL_PORT)
        if result.authorization_url:
            open_external(result.authorization_url)
            messagebox.showinfo(
                "Tailscale authorization required",
                "The Tailscale authorization page has been opened.\n"
                "Approve Serve, then press Remote Access again.\n\n"
                f"{result.authorization_url}",
            )
            return
        if not result.ok or not result.https_url:
            messagebox.showerror("Remote access setup failed", result.detail)
            return
        remote_url = str(result.https_url)
        settings = _load_settings()
        settings["tailscale_url"] = remote_url
        _save_settings(settings)
        token = _get_control_token()
        dialog = tk.Toplevel(self.root)
        dialog.title("Remote Access")
        win_w, win_h = fit_window_size(dialog, S(620), S(280))
        dialog.geometry(f"{win_w}x{win_h}")
        dialog.transient(self.root)
        dialog.grab_set()
        body = tk.Frame(dialog, padx=18, pady=18)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Tailscale remote access is ready",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(body, text="HTTPS URL", fg="#666").pack(anchor="w", pady=(16, 2))
        url_entry = tk.Entry(body)
        url_entry.insert(0, remote_url)
        url_entry.config(state="readonly")
        url_entry.pack(fill="x")
        tk.Label(body, text="Control token (keep private)", fg="#666").pack(anchor="w", pady=(12, 2))
        token_entry = tk.Entry(body, show="•")
        token_entry.insert(0, token)
        token_entry.config(state="readonly")
        token_entry.pack(fill="x")
        buttons = tk.Frame(body)
        buttons.pack(fill="x", pady=(16, 0))
        tk.Button(buttons, text="Open remote page",
                  command=lambda: open_external(remote_url)).pack(side="left")
        tk.Button(buttons, text="Copy URL",
                  command=lambda: self._copy_text(remote_url)).pack(side="left", padx=8)
        tk.Button(buttons, text="Copy token",
                  command=lambda: self._copy_text(token)).pack(side="left")
        tk.Button(buttons, text="Close", command=dialog.destroy).pack(side="right")

    def _copy_text(self, value: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(value)
        self.root.update_idletasks()

    def on_diagnostics(self):
        report = diagnostics_dict(collect_diagnostics(LLAMA_DIR, inference_port=PORT))
        labels = {
            "llama_dir": "llama.cpp folder",
            "llama_server_exists": llama_server_filename(),
            "models_dir_exists": "Models folder",
            "model_count": "Model count",
            "inference_port": f"Inference port {PORT}",
            "control_port_8765": "Control port 8765",
            "tailscale_installed": "Tailscale installed",
            "tailscale_serve_url": "Tailscale HTTPS URL",
        }
        lines = []
        for key, value in report.items():
            if isinstance(value, bool):
                value = "OK" if value else "Not ready"
            lines.append(f"{labels.get(key, key)}: {value or 'Not configured'}")
        messagebox.showinfo("Diagnostics", "\n".join(lines))

    def on_global_settings(self):
        GlobalSettingsDialog(self.root, self)

    # ---------------- remote control
    def remote_status(self) -> dict:
        running = self.server.running
        return {
            "ok": True,
            "running": running,
            "profile": self.server.profile_name if running else None,
            "backend": self.server.backend if running else None,
            "pid": self.server.pid_text() if running else None,
            "uptime": self.server.uptime_text() if running else None,
            "adopted": bool(running and self.server.externally_adopted),
            "degraded": self.server.degraded_reason or None,
            "log": str(self.server.log_path) if self.server.log_path else None,
            "control": {"host": self.control_server.bind_host if self.control_server else None,
                        "port": CONTROL_PORT,
                        "ok": bool(self.control_server is not None and self.control_server.bound),
                        "error": self.control_error or None},
        }

    def remote_profiles(self) -> list[dict]:
        return [{"name": p.get("name", ""), "model": p.get("model", ""),
                 "scheme": normalize_scheme(p.get("scheme")),
                 "backend": p.get("backend", "cuda"),
                 "default_ctx": p.get("default_ctx", 131072),
                 "reasoning": p.get("reasoning", "off"),
                 "reasoning_effort": reasoning_effort_value(p)}
                for p in self.profiles if p.get("model")]

    def remote_logs(self) -> dict:
        path = self.server.log_path
        if path is None or not path.exists():
            return {"path": None, "tail": ""}
        try:
            with open(path, "rb") as fh:
                fh.seek(max(0, path.stat().st_size - 64 * 1024))
                tail = fh.read().decode("utf-8", errors="replace")
            return {"path": str(path), "tail": tail}
        except OSError as exc:
            return {"path": str(path), "error": str(exc), "tail": ""}

    def remote_start(self, body: dict) -> dict:
        model = str(body.get("model", "")).strip()
        if not model:
            return {"ok": False, "error": "model is required"}
        scheme = str(body.get("scheme") or "").strip()
        matches = [p for p in self.profiles if p.get("model") == model]
        if scheme:
            profile = next((p for p in matches
                            if normalize_scheme(p.get("scheme")) == normalize_scheme(scheme)),
                           None)
        else:
            # 未指定方案：優先「預設」方案，否則該模型的第一個方案。
            profile = next((p for p in matches
                            if normalize_scheme(p.get("scheme")) == DEFAULT_SCHEME),
                           None) or (matches[0] if matches else None)
        if profile is None:
            return {"ok": False, "error": "profile is not in models.json"}
        with self.server_lock:
            if self.server.running:
                return {"ok": False, "error": "llama-server is already running",
                        "status": self.remote_status()}
            ctx = format_context_k(profile.get("default_ctx", 131072))
            ok, message = self.server.start(dict(profile), ctx)
        return {"ok": ok, "message": message, "status": self.remote_status()}

    def remote_stop(self) -> dict:
        with self.server_lock:
            message = self.server.stop()
        return {"ok": not self.server.running, "message": message,
                "status": self.remote_status()}

    # ---------------- listbox
    def _persist_profile(self, profile: dict):
        self.cfg.setdefault("profiles", [])
        key = profile_key(profile)
        for i, saved in enumerate(self.cfg["profiles"]):
            if profile_key(saved) == key:
                self.cfg["profiles"][i] = dict(profile)
                break
        else:
            self.cfg["profiles"].append(dict(profile))

    @staticmethod
    def _display_name(p: dict) -> str:
        """列表顯示名：非「預設」方案附方案名（同一模型多方案時區分用）。"""
        scheme = normalize_scheme(p.get("scheme"))
        return p["name"] if scheme == DEFAULT_SCHEME else f"{p['name']} · {scheme}"

    def refresh_listbox(self, select_model=None):
        """首頁只顯示置頂方案；完整清單由 ModelLibraryDialog 管理。

        select_model：(model, scheme) 元組或 model 字串；None = 保持目前選取。"""
        if select_model is None:
            current = self.current_profile() if hasattr(self, "favorite_profiles") else None
            select_model = profile_key(current) if current else None
        if isinstance(select_model, str):
            select_model = (select_model, DEFAULT_SCHEME)
        self.favorite_profiles = [p for p in self.profiles if p.get("starred")]
        self.listbox.delete(0, "end")
        # 列表顯示名稱＋方案（詳細資訊在右側 detail 面板看，避免被擠掉）
        for p in self.favorite_profiles:
            self.listbox.insert("end", self._display_name(p))
        if not self.favorite_profiles:
            self.listbox.insert("end", "No favorite models — open All models")
            self.listbox.itemconfig(0, fg="#7f8b9d")
            return
        index = next((i for i, p in enumerate(self.favorite_profiles)
                      if profile_key(p) == select_model), 0)
        self.listbox.selection_set(index)
        self.listbox.activate(index)
        self.listbox.see(index)

    def set_profile_starred(self, profile: dict, starred: bool):
        p = dict(profile)
        p["starred"] = starred
        if starred:
            key = profile_key(p)
            orders = [int(x.get("favorite_order", -1)) for x in self.profiles
                      if x.get("starred") and profile_key(x) != key]
            if "favorite_order" not in p:
                p["favorite_order"] = max(orders, default=-1) + 1
        else:
            p.pop("favorite_order", None)
        self._persist_profile(p)
        save_config(self.cfg)
        self.profiles = merge_profiles(self.cfg)
        self._normalize_favorite_orders()
        self.refresh_listbox(profile_key(p) if starred else None)
        self.update_detail()

    def _normalize_favorite_orders(self):
        favorites = [p for p in self.profiles if p.get("starred")]
        changed = False
        for order, p in enumerate(favorites):
            if p.get("favorite_order") != order:
                p["favorite_order"] = order
                self._persist_profile(p)
                changed = True
        if changed:
            save_config(self.cfg)
            self.profiles = merge_profiles(self.cfg)

    def move_favorite_model(self, model: str, delta: int):
        """置頂清單排序：model 可為 (model, scheme) 元組或 model 字串。"""
        if isinstance(model, str):
            key = (model, DEFAULT_SCHEME)
        else:
            key = tuple(model)
        favorites = [p for p in self.profiles if p.get("starred")]
        index = next((i for i, p in enumerate(favorites)
                      if profile_key(p) == key), -1)
        target = index + delta
        if index < 0 or target < 0 or target >= len(favorites):
            return
        favorites[index], favorites[target] = favorites[target], favorites[index]
        for order, p in enumerate(favorites):
            p["favorite_order"] = order
            self._persist_profile(p)
        save_config(self.cfg)
        self.profiles = merge_profiles(self.cfg)
        self.refresh_listbox(key)
        self.update_detail()

    def move_favorite(self, delta: int):
        p = self.current_profile()
        if p:
            self.move_favorite_model(profile_key(p), delta)

    def toggle_star(self):
        p = self.current_profile()
        if p:
            self.set_profile_starred(p, not bool(p.get("starred")))

    def open_model_library(self):
        ModelLibraryDialog(self.root, self)

    def open_performance_viewer(self):
        """📊 硬體能力與模型選型；重複點擊聚焦既有視窗。"""
        from .capability_viewer import CapabilityViewer
        viewer = self.performance_viewer
        if viewer is None or not viewer.winfo_exists():
            viewer = CapabilityViewer(self.root, LOGS_DIR, LLAMA_DIR)
            self.performance_viewer = viewer
        else:
            viewer.lift()
            viewer.focus_force()

    def on_select(self, _event=None):
        self.update_detail()

    def current_profile(self) -> dict | None:
        sel = self.listbox.curselection()
        profiles = getattr(self, "favorite_profiles", [])
        if not sel or sel[0] >= len(profiles):
            return None
        return profiles[sel[0]]

    def update_detail(self):
        p = self.current_profile()
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")
        if p is None:
            self.detail_text.insert("end", "Pin models from All models to show them here.")
        else:
            size = model_size_text(p.get("model", ""))
            vision_state = ("TEXT" if not p.get("mmproj") else
                            ("VISION ON" if profile_vision_enabled(p) else "VISION OFF"))
            scheme = normalize_scheme(p.get("scheme"))
            title = p["name"] if scheme == DEFAULT_SCHEME \
                else f"{p['name']}  ·  {scheme}"
            lines = [
                title,
                f"Size: {size or '?'}   ·   {p.get('backend','cuda').upper()}",
                vision_state,
                f"Default context: {format_context_k(p.get('default_ctx', 131072))}K",
            ]
            effort = reasoning_effort_value(p)
            if p.get("reasoning", "off") == "on":
                thinking_state = f"ON · effort {effort}" if effort != "default" else "ON"
            else:
                thinking_state = "OFF"
            lines.append(f"Thinking: {thinking_state}")
            self.detail_text.insert("end", "\n".join(lines))
        self.detail_text.config(state="disabled")
        if p:
            self.backend_var.set(p.get("backend", "cuda").upper())
            self.ctx_var.set(format_context_k(p.get("default_ctx", 131072)))
            if p.get("reasoning", "off") == "on":
                self.effort_var.set(reasoning_effort_value(p))
            else:
                self.effort_var.set("off")
            has_mmproj = bool(p.get("mmproj"))
            self.vision_var.set(profile_vision_enabled(p))
            self.vision_check.config(state="normal" if has_mmproj else "disabled")
        else:
            self.vision_var.set(False)
            self.effort_var.set("off")
            self.vision_check.config(state="disabled")

    # ---------------- 動作
    def on_launch(self):
        p = self.current_profile()
        if p is None:
            messagebox.showinfo("提示", "請先選擇一個模型")
            return
        p = dict(p)
        try:
            ctx_val = parse_context_k(self.ctx_var.get())
        except ValueError as exc:
            messagebox.showerror("Context格式錯誤", str(exc))
            return
        p["backend"] = self.backend_var.get().lower()
        p["vision_enabled"] = bool(p.get("mmproj")) and self.vision_var.get()
        thinking = self.effort_var.get()
        if thinking == "off":
            p["reasoning"] = "off"
            p["reasoning_effort"] = "default"
        else:
            p["reasoning"] = "on"
            p["reasoning_effort"] = thinking
        # 主畫面上的參數一併存回模型設定（含 Context）：
        # 下次啟動或到 Settings 開啟時，看到的就是上次實際啟動的組合，
        # 不用再到設定視窗重設一遍。
        p["default_ctx"] = ctx_val
        self.cfg.setdefault("profiles", [])
        key = (p["model"], normalize_scheme(p.get("scheme")))
        for i, sp in enumerate(self.cfg["profiles"]):
            if profile_key(sp) == key:
                self.cfg["profiles"][i] = p
                break
        else:
            self.cfg["profiles"].append(p)
        save_config(self.cfg)
        self.profiles = merge_profiles(self.cfg)
        # 讓 detail 面板顯示存檔後的設定（Default context 等）
        self.refresh_listbox(key)
        self.update_detail()

        with self.server_lock:
            ok, msg = self.server.start(p, self.ctx_var.get())
        if not ok:
            messagebox.showerror("啟動失敗", msg)
            return
        self._update_server_ui()
        # 啟動後留在主視窗，讓主人直接看到整合式 Log。
        self.show_window()
        self._reload_embedded_log()

    def on_toggle_server(self):
        if self.server.running:
            self.on_stop()
        else:
            self.on_launch()

    def on_stop(self):
        with self.server_lock:
            self.server.stop()
        self._update_server_ui()

    def open_log(self):
        self.show_window()
        self.scroll_log_to_bottom()

    def _on_log_wheel(self, event):
        delta = getattr(event, "delta", 0)
        if delta > 0:
            self._log_auto_scroll = False
        elif self.log_text.yview()[1] >= 0.999:
            self._log_auto_scroll = True

    def _log_scrollbar(self, *args):
        if args and args[0] == "moveto":
            try:
                self._log_auto_scroll = float(args[1]) >= 0.999
            except (IndexError, ValueError):
                pass
        self.log_text.yview(*args)

    def scroll_log_to_bottom(self):
        self._log_auto_scroll = True
        self.log_text.see("end")

    def clear_embedded_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self._log_last_offset = 0

    def _reload_embedded_log(self):
        p = self.server.log_path
        if p is None or not p.exists():
            return
        try:
            size = p.stat().st_size
            # 沒新增內容就不開檔（每秒輪詢時避免磁碟 IO）
            if size == self._log_last_offset and self._log_last_offset > 0:
                return
            if self._log_last_offset > size:
                self._log_last_offset = 0
            skipped = False
            start = self._log_last_offset
            if size - start > self._log_max_chunk_bytes:
                start = max(0, size - self._log_max_chunk_bytes)
                skipped = True
            with open(p, "rb") as fh:
                fh.seek(start)
                chunk = fh.read(size - start)
                self._log_last_offset = fh.tell()
            if chunk:
                self.log_text.config(state="normal")
                if skipped:
                    self.log_text.insert("end", "\n[較舊的Log已略過，顯示最新512KB]\n")
                self.log_text.insert("end", chunk.decode("utf-8", errors="replace"))
                lines = int(self.log_text.index("end-1c").split(".")[0])
                if lines > self._log_max_lines:
                    self.log_text.delete("1.0", f"{lines - self._log_max_lines}.0")
                if self._log_auto_scroll:
                    self.log_text.see("end")
                self.log_text.config(state="disabled")
        except (OSError, tk.TclError):
            pass

    def _poll_embedded_log(self):
        if self.quitting:
            return
        if self.root.state() != "withdrawn":
            self._reload_embedded_log()
        # 1s→2s：log 更新不需要那麼頻繁，減少 Windows 主執行緒活動。
        self.root.after(2000, self._poll_embedded_log)

    def on_add_model(self):
        AddModelDialog(self.root, self)

    def on_delete_model(self):
        p = self.current_profile()
        if p is None:
            messagebox.showinfo("提示", "請先選擇要刪除的模型")
            return
        self.delete_profile(p)

    def delete_profile(self, p: dict, parent=None):
        """刪除指定模型：檔案進資源回收筒，並移除 profile。"""
        model_name = p.get("name", p.get("model", ""))
        model_path = model_file(p["model"])
        if self.server.running and self.server.profile_name == p.get("name"):
            messagebox.showerror(
                "無法刪除", f"「{model_name}」正在執行中。請先停止伺服器。",
                parent=parent)
            return False

        size = model_size_text(p["model"]) or "?"
        files_to_delete = [("主模型", model_path)]
        mmproj_name = p.get("mmproj") or ""
        keep_mmproj_note = ""
        if mmproj_name:
            mmproj_file = model_file(mmproj_name)
            shared_by_others = any(
                sp.get("model") != p["model"] and sp.get("mmproj") == mmproj_name
                for sp in self.profiles)
            if shared_by_others:
                keep_mmproj_note = (
                    f"\n\n{mmproj_name} 也被其他模型共用，該檔案會保留。")
            else:
                files_to_delete.append(("vision 檔 (mmproj)", mmproj_file))

        file_lines = "\n".join(
            f"  {label}: {path.name}  [{size if label == '主模型' else ''}]"
            for label, path in files_to_delete)
        ok = messagebox.askyesno(
            "確認刪除模型",
            f"確定要刪除「{model_name}」嗎？\n\n{file_lines}\n\n"
            "檔案會移到資源回收筒（可還原），設定也會移除。"
            + keep_mmproj_note,
            icon="warning", parent=parent)
        if not ok:
            return False

        failed = []
        for label, path in files_to_delete:
            if path.exists() and not delete_file_to_recycle_bin(path):
                failed.append(f"{label}: {path.name}")
        invalidate_model_inventory()
        self.cfg.setdefault("profiles", [])
        self.cfg["profiles"] = [
            sp for sp in self.cfg["profiles"] if sp.get("model") != p["model"]]
        save_config(self.cfg)
        self.profiles = merge_profiles(self.cfg)
        self.refresh_listbox()
        self.update_detail()

        if failed:
            messagebox.showwarning(
                "部分檔案無法刪除",
                "設定已移除，但以下檔案無法移到資源回收筒：\n\n"
                + "\n".join(failed), parent=parent)
        else:
            messagebox.showinfo(
                "已刪除", f"「{model_name}」已移除（可從資源回收筒還原）。",
                parent=parent)
        return True

    def on_settings(self):
        """開啟口語化設定視窗（目前選中的模型）。"""
        p = self.current_profile()
        if p is None:
            messagebox.showinfo("提示", "請先選擇一個模型再開設定")
            return
        SettingsDialog(self.root, self, p)

    def on_rescan(self):
        invalidate_model_inventory()
        self.profiles = merge_profiles(self.cfg)
        self.refresh_listbox()
        self.update_detail()

    def open_models_dir(self):
        MODELS_DIR.mkdir(exist_ok=True)
        open_external(MODELS_DIR)

    def open_config(self):
        if not CONFIG_PATH.exists():
            save_config(self.cfg)
        open_external(CONFIG_PATH)

    # ---------------- 狀態輪詢
    # adopt 掃描節流秒數：避免每秒 psutil 全系統掃描拖住 UI。
    ADOPT_SCAN_INTERVAL = 10.0

    def _poll_server_status(self):
        if self.quitting:
            return
        if not self.server.running and port_in_use(PORT):
            # 8080 被占但不確定是不是 llama-server：只有掃描能確認。
            # 節流：10 秒內最多掃一次，避免每秒全系統 psutil 掃描。
            now = time.monotonic()
            if now - self._last_adopt_attempt >= self.ADOPT_SCAN_INTERVAL:
                self._last_adopt_attempt = now
                self.server.adopt_existing_server()
        self._update_server_ui()
        # server 執行中 2s 輪詢（狀態/uptime 要即時）；idle 時 5s 就夠，
        # 盡量減少 Windows 上主執行緒的定時活動。
        delay = 2000 if self.server.running else 5000
        self.root.after(delay, self._poll_server_status)

    def _update_server_ui(self):
        running = self.server.running
        if running:
            if self.server.degraded_reason:
                self.status_dot.config(
                    text=f"⚠  DEGRADED  ·  {self.server.profile_name}  ·  "
                         f"PID {self.server.pid_text()}  ·  {self.server.uptime_text()}",
                    fg="#ff6b6b")
            else:
                status_parts = ["●"]
                if self.server.externally_adopted:
                    status_parts.append("ADOPTED")
                status_parts.extend([
                    self.server.profile_name,
                    self.server.backend,
                    f"PID {self.server.pid_text()}",
                ])
                uptime = self.server.uptime_text()
                if uptime:
                    status_parts.append(uptime)
                self.status_dot.config(
                    text="  ·  ".join(status_parts), fg="#67d58b")
            self.power_btn.config(
                text="■  STOP SERVER", bg="#b94b55",
                activebackground="#ce5b66", state="normal")
        else:
            self.status_dot.config(text="○  Server stopped", fg="#f0ad4e")
            self.power_btn.config(
                text="▶  START SERVER", bg="#2f74d0",
                activebackground="#3e86e2", state="normal")

    # ---------------- 托盤
    def init_tray(self):
        if not HAVE_TRAY:
            return
        img = self.make_tray_image()
        menu = pystray.Menu(
            pystray.MenuItem("顯示視窗", self.show_window, default=True),
            pystray.MenuItem("啟動目前模型", self.tray_launch),
            pystray.MenuItem("停止伺服器", self.tray_stop),
            pystray.MenuItem("開啟聊天介面", self.tray_chat_ui),
            pystray.MenuItem("查看 Log", self.tray_log),
            pystray.MenuItem("重新掃描", self.tray_rescan),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("結束", self.quit_app),
        )
        self.tray_icon = pystray.Icon("llama-launcher", img,
                                      "llama.cpp Launcher", menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    @staticmethod
    def make_tray_image():
        if ICON_PNG.exists():
            try:
                with Image.open(ICON_PNG) as source:
                    return source.convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)
            except Exception:
                pass
        # 圖示檔損壞時的高辨識度fallback，不再畫兔耳。
        img = Image.new("RGB", (64, 64), "#111827")
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([3, 3, 60, 60], radius=13,
                            fill="#172554", outline="#38bdf8", width=4)
        d.text((20, 23), "AI", fill="white")
        d.ellipse([47, 47, 59, 59], fill="#fb7185")
        return img

    def hide_to_tray(self):
        settings = _load_settings()
        close_to_tray = bool(settings.get("close_to_tray", IS_WINDOWS))
        if close_to_tray and self.tray_icon is not None:
            self.root.withdraw()
        else:
            self.quit_app()

    def show_window(self, icon=None, item=None):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self._reload_embedded_log()

    def show_window_from_tray(self):
        """供 SingleInstance 回呼（背景 thread 呼叫）：轉到 tk 主執行緒執行。"""
        try:
            self.root.after(0, self.show_window)
        except Exception:
            pass

    def open_chat_ui(self):
        """開啟聊天 WebUI（llama-server 的 / 頁面）。"""
        url = f"http://127.0.0.1:{PORT}/"
        if not open_external(url):
            messagebox.showinfo("開啟瀏覽器", f"無法自動開啟，請手動開：\n{url}")

    def tray_launch(self, icon=None, item=None):
        self.root.after(0, self.on_launch)

    def tray_stop(self, icon=None, item=None):
        self.root.after(0, self.on_stop)

    def tray_chat_ui(self, icon=None, item=None):
        self.root.after(0, self.open_chat_ui)

    def tray_log(self, icon=None, item=None):
        self.root.after(0, self.open_log)

    def tray_rescan(self, icon=None, item=None):
        self.root.after(0, self.on_rescan)

    def quit_app(self, icon=None, item=None):
        """結束：先問是否要一併關閉 llama-server。"""
        if self.server.running:
            if not messagebox.askyesno(
                    "結束",
                    "llama-server 還在執行。\n要一併關閉它嗎？\n\n"
                    "（選「否」則 llama-server 會繼續在背景跑，log 仍會寫入）"):
                pass  # 讓 server 繼續跑
            else:
                with self.server_lock:
                    self.server.stop()
        self.quitting = True
        if self.control_server is not None:
            try:
                self.control_server.close()
            except Exception:
                pass
        if self.tray_icon is not None:
            self.tray_icon.stop()
        self.root.after(50, self.root.destroy)


class ModelLibraryDialog(tk.Toplevel):
    """完整模型庫：搜尋、置頂、排序、設定與刪除。"""

    def __init__(self, parent, app: LauncherApp):
        super().__init__(parent)
        self.app = app
        self.filtered: list[dict] = []
        self.title("Model Library")
        win_w, win_h = fit_window_size(self, S(780), S(620))
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(*fit_window_size(self, S(680), S(500), screen_ratio=1.0))
        self.configure(bg="#111722")
        self.transient(parent)

        header = tk.Frame(self, bg="#171e2a", height=S(60))
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="MODEL LIBRARY", font=("Segoe UI", 15, "bold"),
                 fg="#f0f4f9", bg="#171e2a", padx=18).pack(side="left", fill="y")
        tk.Button(header, text="＋ Add model", command=self.add_model,
                  font=("Segoe UI", 9, "bold"), bg="#2f74d0", fg="white",
                  activebackground="#3e86e2", activeforeground="white",
                  relief="flat", padx=14, pady=6).pack(side="right", padx=16, pady=12)
        tk.Button(header, text="Models folder", command=self.app.open_models_dir,
                  font=("Segoe UI", 9), bg="#273246", fg="white",
                  activebackground="#3b4a63", activeforeground="white",
                  relief="flat", padx=12, pady=6).pack(side="right", pady=12)

        search_row = tk.Frame(self, bg="#111722")
        search_row.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(search_row, text="Search", font=("Segoe UI", 9, "bold"),
                 fg="#8293aa", bg="#111722").pack(side="left")
        self.search_var = tk.StringVar()
        entry = tk.Entry(search_row, textvariable=self.search_var,
                         font=("Segoe UI", 10), bg="#202838", fg="#eef2f8",
                         insertbackground="white", relief="flat")
        entry.pack(side="left", fill="x", expand=True, padx=(10, 0), ipady=6)
        self.search_var.trace_add("write", lambda *_: self.refresh())

        list_frame = tk.Frame(self, bg="#111722")
        list_frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        self.listbox = tk.Listbox(
            list_frame, font=("Segoe UI", 10), exportselection=False,
            bg="#0c1119", fg="#e3eaf3", selectbackground="#326fd1",
            selectforeground="white", relief="flat", bd=0,
            highlightthickness=1, highlightbackground="#303b4e",
            activestyle="none")
        scroll = tk.Scrollbar(list_frame, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.listbox.bind("<Double-Button-1>", lambda _e: self.settings())

        actions = tk.Frame(self, bg="#111722")
        actions.pack(fill="x", padx=16, pady=(0, 16))
        # 兩排按鈕，避免太擁擠
        action_rows = (
            (("★ Pin / Unpin", self.toggle_pin), ("↑", lambda: self.move(-1)),
             ("↓", lambda: self.move(1)), ("Settings", self.settings),
             ("Delete", self.delete)),
            (("Rescan", self.rescan), ("Export", self._export),
             ("Import", self._import), ("Open config", self.app.open_config),
             ("Close", self.destroy)),
        )
        for row_index, row in enumerate(action_rows):
            for col_index, (text, command) in enumerate(row):
                actions.columnconfigure(col_index, weight=1)
                color = {
                    "★ Pin / Unpin": "#2f74d0",
                    "Delete": "#8f3f48",
                    "Close": "#273246",
                }.get(text, "#273246")
                tk.Button(actions, text=text, command=command,
                          font=("Segoe UI", 9, "bold"), bg=color, fg="white",
                          activebackground="#3b4a63", activeforeground="white",
                          relief="flat", padx=6, pady=6).grid(
                              row=row_index, column=col_index, sticky="ew",
                              padx=(0, 6) if col_index < 4 else (0, 0),
                              pady=(0, 6) if row_index == 0 else (0, 0))
        self.refresh()
        entry.focus_set()

    def current(self) -> dict | None:
        sel = self.listbox.curselection()
        if not sel or sel[0] >= len(self.filtered):
            return None
        return self.filtered[sel[0]]

    def refresh(self, select_model=None):
        """select_model：(model, scheme) 元組或 model 字串；None = 保持目前。"""
        current = self.current()
        if select_model is None and current:
            select_model = profile_key(current)
        elif isinstance(select_model, str):
            select_model = (select_model, DEFAULT_SCHEME)
        query = self.search_var.get().strip().lower()
        self.filtered = [p for p in self.app.profiles
                         if not query or query in p.get("name", "").lower()
                         or query in p.get("model", "").lower()]
        self.listbox.delete(0, "end")
        for p in self.filtered:
            star = "★" if p.get("starred") else "☆"
            if not p.get("mmproj"):
                media = "TEXT"
            else:
                media = "VISION ON" if profile_vision_enabled(p) else "VISION OFF"
            size = model_size_text(p.get("model", "")) or "?"
            backend = p.get("backend", "cuda").upper()
            self.listbox.insert(
                "end", f"{star}  {self.app._display_name(p)}    [{size}]    "
                       f"{media} · {backend}")
        if self.filtered:
            index = next((i for i, p in enumerate(self.filtered)
                          if profile_key(p) == tuple(select_model)), 0)
            self.listbox.selection_set(index)
            self.listbox.see(index)

    def toggle_pin(self):
        p = self.current()
        if not p:
            return
        self.app.set_profile_starred(p, not bool(p.get("starred")))
        self.refresh(select_model=profile_key(p))

    def move(self, delta: int):
        p = self.current()
        if not p:
            return
        if not p.get("starred"):
            messagebox.showinfo("提示", "請先把模型設為置頂。", parent=self)
            return
        self.app.move_favorite_model(profile_key(p), delta)
        self.refresh(select_model=profile_key(p))

    def settings(self):
        p = self.current()
        if not p:
            return
        dialog = SettingsDialog(self, self.app, p)
        self.wait_window(dialog)
        self.refresh(select_model=profile_key(p))

    def add_model(self):
        dialog = AddModelDialog(self, self.app)
        self.wait_window(dialog)
        self.refresh()

    def delete(self):
        p = self.current()
        if p and self.app.delete_profile(p, parent=self):
            self.refresh()

    def rescan(self):
        self.app.on_rescan()
        self.refresh()

    def _export(self):
        """Export the currently selected profile, or all if none selected."""
        p = self.current()
        profiles = [p] if p else self.app.profiles
        self.app.export_profiles(profiles)
        self.refresh()

    def _import(self):
        self.app.import_profiles()
        self.refresh()


class GlobalSettingsDialog(tk.Toplevel):
    """全域設定：llama.cpp 路徑、開機啟動、關閉行為、Remote Access、舊資料匯入。

    與 SettingsDialog（個別模型設定）分開，避免兩種設定混在一起。"""

    def __init__(self, parent, app: LauncherApp):
        super().__init__(parent)
        self.app = app
        self.title("設定（全域）")
        win_w, win_h = fit_window_size(self, S(620), S(760))
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(*fit_window_size(self, S(540), S(640), screen_ratio=1.0))
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        # 可捲動內容區（新增區塊後高度不夠時不會擠掉按鈕）
        outer = tk.Frame(self)
        outer.pack(side="top", fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        body = tk.Frame(canvas, padx=18, pady=14)
        canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>",
                  lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        tk.Label(body, text="全域設定",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(body, text="這些設定影響整個 launcher，與個別模型無關。",
                 font=("Segoe UI", 9), fg="#888").pack(anchor="w", pady=(2, 10))

        # ---- llama.cpp 資料夾
        tk.Label(body, text="llama.cpp 資料夾位置",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(body, text="包含 llama-server 的那一層；模型在它的 models 子資料夾。",
                 font=("Segoe UI", 8), fg="#888").pack(anchor="w", pady=(1, 4))
        dir_row = tk.Frame(body)
        dir_row.pack(fill="x")
        self.dir_var = tk.StringVar(value=str(LLAMA_DIR))
        self.dir_entry = tk.Entry(dir_row, textvariable=self.dir_var,
                                  font=("Segoe UI", 9))
        self.dir_entry.pack(side="left", fill="x", expand=True, ipady=3)

        def browse_dir():
            f = filedialog.askdirectory(initialdir=str(LLAMA_DIR))
            if f:
                self.dir_var.set(f)
        tk.Button(dir_row, text="瀏覽…", command=browse_dir,
                  padx=10, pady=3).pack(side="left", padx=(6, 0))
        tk.Frame(body, height=1, bg="#ddd").pack(fill="x", pady=12)

        # ---- 伺服器（host / port / alias / api-key）
        server = _load_settings().get("server") or {}
        tk.Label(body, text="伺服器", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(body, text="llama-server 的網路參數（對所有模型生效）。",
                 font=("Segoe UI", 8), fg="#888").pack(anchor="w", pady=(1, 4))
        srv = server_settings_from_dict(server)
        srv_grid = tk.Frame(body)
        srv_grid.pack(fill="x")
        self.host_var = tk.StringVar(value=srv["host"])
        self.port_var = tk.StringVar(value=str(srv["port"]))
        self.alias_var = tk.StringVar(value=str(server.get("alias") or ""))
        self.api_key_var = tk.StringVar(value="")
        existing_key = read_api_key(API_KEY_PATH)
        self._api_key_existing = existing_key
        for row, (label, var, width) in enumerate((
                ("Host", self.host_var, None),
                ("Port", self.port_var, 8),
                ("Alias（API 別名）", self.alias_var, None))):
            tk.Label(srv_grid, text=label, font=("Segoe UI", 9)).grid(
                row=row, column=0, sticky="w", pady=2)
            if width:
                tk.Entry(srv_grid, textvariable=var, width=width,
                         font=("Consolas", 9)).grid(row=row, column=1, sticky="w")
            else:
                tk.Entry(srv_grid, textvariable=var,
                         font=("Consolas", 9)).grid(
                             row=row, column=1, sticky="ew")
        srv_grid.columnconfigure(1, weight=1)
        key_row = tk.Frame(body)
        key_row.pack(fill="x", pady=(6, 0))
        tk.Label(key_row, text="API Key", font=("Segoe UI", 9)).pack(side="left")
        self.api_key_entry = tk.Entry(key_row, textvariable=self.api_key_var,
                                      show="•", font=("Consolas", 9))
        self.api_key_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
        key_status = ("已設定（%s…%s）" % (existing_key[:4], existing_key[-4:])) \
            if len(existing_key) >= 8 else ("已設定" if existing_key else "未設定")
        tk.Label(key_row, text=f"目前：{key_status}", font=("Segoe UI", 8),
                 fg="#888").pack(side="left")
        tk.Button(key_row, text="清除", command=self._clear_api_key,
                  font=("Segoe UI", 8)).pack(side="left", padx=(4, 0))
        tk.Label(body, text="API Key 留空＝維持目前值；填入新值＝覆蓋。"
                            "存於 secrets/ 目錄，不寫進 settings.json。",
                 font=("Segoe UI", 8), fg="#888", anchor="w",
                 wraplength=560, justify="left").pack(anchor="w", pady=(2, 0))
        tk.Frame(body, height=1, bg="#ddd").pack(fill="x", pady=12)

        # ---- VRAM 預檢
        pf = vram_preflight_config()
        tk.Label(body, text="VRAM 預檢（GPU 啟動前）", font=("Segoe UI", 10,
                  "bold")).pack(anchor="w")
        tk.Label(body, text="預設關閉。需要乾淨顯存基線（例如固定層數分配）的機器"
                            "可開嚴格模式。",
                 font=("Segoe UI", 8), fg="#888").pack(anchor="w", pady=(1, 4))
        self.preflight_var = tk.StringVar(
            value=pf["mode"] if pf["mode"] in VRAM_PREFLIGHT_MODES else "off")
        pf_row = tk.Frame(body)
        pf_row.pack(fill="x")
        tk.Label(pf_row, text="模式：", font=("Segoe UI", 9)).pack(side="left")
        ttk.Combobox(pf_row, textvariable=self.preflight_var,
                     values=["off", "warn", "strict"], state="readonly",
                     width=8).pack(side="left", padx=(4, 12))
        tk.Label(pf_row, text="off＝不檢查；warn＝超標只警告；"
                              "strict＝清理+超時擋住",
                 font=("Segoe UI", 8), fg="#888").pack(side="left")
        self.preflight_limits_var = tk.StringVar(
            value=",".join(str(v) for v in pf["gpu_limits_mb"]
                           if v is not None))
        limits_row = tk.Frame(body)
        limits_row.pack(fill="x", pady=(4, 0))
        tk.Label(limits_row, text="VRAM 上限（MB，依 GPU 順序）：",
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(limits_row, textvariable=self.preflight_limits_var, width=14,
                 font=("Consolas", 9)).pack(side="left", padx=(4, 0))
        tk.Label(limits_row, text="例：2304,128；留空＝不限制",
                 font=("Segoe UI", 8), fg="#888").pack(side="left", padx=(6, 0))
        tk.Label(body, text="啟動前強制結束的程序（僅 strict、僅 Windows，一行一個）",
                 font=("Segoe UI", 9), anchor="w").pack(anchor="w", pady=(6, 2))
        self.preflight_procs_text = tk.Text(body, height=4, font=("Consolas", 9))
        self.preflight_procs_text.pack(fill="x")
        self.preflight_procs_text.insert(
            "1.0", "\n".join(pf["kill_processes"]))
        comfy = pf["comfyui"]
        self.comfyui_enabled_var = tk.BooleanVar(value=comfy["enabled"])
        self.comfyui_distro_var = tk.StringVar(value=comfy["distro"])
        self.comfyui_service_var = tk.StringVar(value=comfy["service"])
        tk.Checkbutton(body, text="啟動前停止 WSL 裡的 ComfyUI service（僅 strict）",
                       variable=self.comfyui_enabled_var, anchor="w",
                       font=("Segoe UI", 9)).pack(anchor="w")
        comfy_row = tk.Frame(body)
        comfy_row.pack(fill="x", pady=(2, 0))
        tk.Label(comfy_row, text="WSL 分區：", font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(comfy_row, textvariable=self.comfyui_distro_var, width=10,
                 font=("Consolas", 9)).pack(side="left", padx=(4, 12))
        tk.Label(comfy_row, text="Service：", font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(comfy_row, textvariable=self.comfyui_service_var, width=26,
                 font=("Consolas", 9)).pack(side="left", padx=(4, 0))
        tk.Frame(body, height=1, bg="#ddd").pack(fill="x", pady=12)

        # ---- 其他（RAM cache / Vulkan 裝置）
        tk.Label(body, text="其他", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        other_grid = tk.Frame(body)
        other_grid.pack(fill="x", pady=(4, 0))
        self.cache_ram_var = tk.StringVar(
            value=str(server.get("cache_ram_mb", DEFAULT_CACHE_RAM_MB)))
        self.vulkan_devices_var = tk.StringVar(
            value=str(server.get("vulkan_devices") or ""))
        tk.Label(other_grid, text="RAM prompt cache 上限（MB，-cram）：",
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        tk.Entry(other_grid, textvariable=self.cache_ram_var, width=10,
                 font=("Consolas", 9)).grid(row=0, column=1, sticky="w",
                                            padx=(4, 16))
        tk.Label(other_grid, text="Vulkan --device（留空＝自動）：",
                 font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w")
        tk.Entry(other_grid, textvariable=self.vulkan_devices_var, width=20,
                 font=("Consolas", 9)).grid(row=1, column=1, sticky="w", padx=(4, 0))
        tk.Frame(body, height=1, bg="#ddd").pack(fill="x", pady=12)

        # ---- 隨開機啟動
        tk.Label(body, text="啟動行為", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.autostart_var = tk.BooleanVar(value=autostart_enabled())
        tk.Checkbutton(
            body, text="隨 Windows / Linux 登入自動啟動",
            variable=self.autostart_var, anchor="w",
            font=("Segoe UI", 9)).pack(fill="x", pady=(4, 0))
        self.close_to_tray_var = tk.BooleanVar(
            value=bool(_load_settings().get("close_to_tray", IS_WINDOWS)))
        tk.Checkbutton(
            body, text="關閉視窗時縮到系統匣（不勾 = 直接結束）",
            variable=self.close_to_tray_var, anchor="w",
            font=("Segoe UI", 9)).pack(fill="x", pady=(2, 0))
        tk.Frame(body, height=1, bg="#ddd").pack(fill="x", pady=12)

        # ---- Remote access
        tk.Label(body, text="遠端控制", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(body, text="透過 Tailscale 從手機／其他電腦控制 llama-server。",
                 font=("Segoe UI", 8), fg="#888").pack(anchor="w", pady=(1, 4))
        tk.Button(body, text="設定 Remote Access（Tailscale Serve + Token）",
                  command=self.app.on_remote_access,
                  bg="#2f74d0", fg="white", activebackground="#3e86e2",
                  activeforeground="white", relief="flat",
                  pady=7).pack(fill="x")
        tk.Frame(body, height=1, bg="#ddd").pack(fill="x", pady=12)

        # ---- 舊資料匯入
        tk.Label(body, text="資料", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Button(body, text="Import old launcher data（舊版 models.json / token）",
                  command=self.app.on_migrate_legacy,
                  font=("Segoe UI", 9), bg="#253045", fg="#dce6f3",
                  activebackground="#34445f", activeforeground="white",
                  relief="flat", pady=7).pack(fill="x")
        tk.Button(body, text="開啟 profiles.json（手動編輯模型清單）",
                  command=self.app.open_config,
                  font=("Segoe UI", 9), bg="#253045", fg="#dce6f3",
                  activebackground="#34445f", activeforeground="white",
                  relief="flat", pady=7).pack(fill="x", pady=(6, 0))

        # ---- 常用 GPU 分配
        tk.Label(body, text="常用 GPU 分配", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(body, text="在個別模型設定自訂過的層數分配，之後可直接選。",
                 font=("Segoe UI", 8), fg="#888").pack(anchor="w", pady=(1, 4))
        preset_row = tk.Frame(body)
        preset_row.pack(fill="x")
        self.preset_list = tk.Listbox(preset_row, height=4,
                                      font=("Consolas", 9),
                                      exportselection=False)
        self.preset_list.pack(side="left", fill="x", expand=True)
        preset_btns = tk.Frame(preset_row)
        preset_btns.pack(side="left", padx=(6, 0))
        tk.Button(preset_btns, text="刪除選取",
                  command=self._delete_preset, padx=8, pady=3).pack(fill="x")
        tk.Button(preset_btns, text="清空",
                  command=self._clear_presets, padx=8, pady=3).pack(fill="x", pady=(4, 0))
        self._reload_presets()
        tk.Frame(body, height=1, bg="#ddd").pack(fill="x", pady=12)

        btns = tk.Frame(self, padx=18, pady=12)
        btns.pack(fill="x")
        tk.Button(btns, text="儲存", command=self.save,
                  font=("Segoe UI", 10, "bold"), width=12,
                  bg="#2f6fed", fg="white",
                  activebackground="#2456c0", activeforeground="white",
                  padx=16, pady=7).pack(side="right")
        tk.Button(btns, text="取消", command=self.destroy,
                  font=("Segoe UI", 10), width=12,
                  padx=16, pady=7).pack(side="right", padx=(0, 10))

    def _clear_api_key(self):
        self.api_key_var.set("")
        self._api_key_existing = ""
        self._api_key_dirty = True

    def _reload_presets(self):
        self.preset_list.delete(0, "end")
        for preset in gpu_preset_options():
            self.preset_list.insert("end", preset)

    def _delete_preset(self):
        sel = self.preset_list.curselection()
        if not sel:
            return
        value = self.preset_list.get(sel[0])
        forget_gpu_preset(value)
        self._reload_presets()

    def _clear_presets(self):
        for value in gpu_preset_options():
            forget_gpu_preset(value)
        self._reload_presets()

    def save(self):
        new_dir = self.dir_var.get().strip()
        if new_dir and str(Path(new_dir).resolve()) != str(LLAMA_DIR.resolve()):
            ok = update_llama_dir(new_dir)
            if not ok:
                messagebox.showerror(
                    "資料夾位置錯誤",
                    f"在 {new_dir} 找不到 {llama_server_filename()}。\n"
                    f"請確認資料夾位置正確（要能看得到 {llama_server_filename()} 那層）。",
                    parent=self)
                return
            invalidate_model_inventory()
            self.app.cfg = load_config()
            self.app.profiles = merge_profiles(self.app.cfg)
            self.app.refresh_listbox()
            self.app.update_detail()

        # 伺服器參數
        try:
            port = int(self.port_var.get().strip() or DEFAULT_PORT)
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            messagebox.showerror("Port 格式錯誤",
                                 "Port 請輸入 1～65535 的整數。", parent=self)
            return
        settings = _load_settings()
        settings["server"] = {
            "host": self.host_var.get().strip() or DEFAULT_HOST,
            "port": port,
            "alias": self.alias_var.get().strip(),
            "cache_ram_mb": self.cache_ram_var.get().strip() or DEFAULT_CACHE_RAM_MB,
            "vulkan_devices": self.vulkan_devices_var.get().strip(),
        }
        # VRAM 預檢
        limits = []
        for value in self.preflight_limits_var.get().split(","):
            value = value.strip()
            if not value:
                continue
            try:
                limits.append(max(0, int(value)))
            except ValueError:
                messagebox.showerror(
                    "VRAM 上限格式錯誤",
                    "上限請用逗號分隔的整數 MB，例如 2304,128。", parent=self)
                return
        settings["vram_preflight"] = {
            "mode": self.preflight_var.get(),
            "gpu_limits_mb": limits,
            "kill_processes": [line.strip()
                               for line in self.preflight_procs_text.get(
                                   "1.0", "end").splitlines() if line.strip()],
            "comfyui": {
                "enabled": bool(self.comfyui_enabled_var.get()),
                "distro": self.comfyui_distro_var.get().strip() or "Ubuntu",
                "service": self.comfyui_service_var.get().strip(),
            },
        }
        settings["close_to_tray"] = bool(self.close_to_tray_var.get())
        _save_settings(settings)
        update_server_settings()

        # API key：填新值＝覆蓋；清空按過「清除」＝刪除；留空＝維持
        key_text = self.api_key_var.get().strip()
        if key_text:
            write_api_key(API_KEY_PATH, key_text)
        elif getattr(self, "_api_key_dirty", False):
            write_api_key(API_KEY_PATH, "")

        if not set_autostart(bool(self.autostart_var.get())):
            messagebox.showwarning(
                "開機啟動設定失敗",
                "無法寫入開機啟動設定（可能權限不足），其他設定已儲存。",
                parent=self)
        self.destroy()
        messagebox.showinfo("已儲存", "全域設定已更新。")


class SettingsDialog(tk.Toplevel):
    """口語化設定視窗：把技術參數翻譯成一般人看得懂的選項。

    同一模型可以有多個「方案」（不同用途各存一套參數）；進階頁籤提供
    完整參數（raw）與最終指令預覽，raw 非空時覆蓋所有 GUI 參數。"""

    def __init__(self, parent, app: LauncherApp, profile: dict):
        super().__init__(parent)
        self.app = app
        self.profile = profile
        self.original_key = profile_key(profile)
        scheme_text = "" if normalize_scheme(profile.get("scheme")) == DEFAULT_SCHEME \
            else f" · {normalize_scheme(profile.get('scheme'))}"
        self.title(f"設定 — {profile.get('name','')}{scheme_text}")
        win_w, win_h = fit_window_size(self, S(700), S(620))
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(*fit_window_size(self, S(620), S(500), screen_ratio=1.0))
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        # 底部按鈕列先 pack，固定保留空間，不會被上方內容擠出視窗。
        btns = tk.Frame(self, padx=16)
        btns.pack(side="bottom", fill="x", pady=(10, 16))
        tk.Label(btns, text="同一模型可存多套參數（不同用途），按「另存新方案」複製一份。",
                 font=("Segoe UI", 8), fg="#888").pack(side="left", fill="x", expand=True)
        tk.Button(
            btns, text="另存新方案", command=self.save_as_new_scheme,
            font=("Segoe UI", 10, "bold"), width=12,
            bg="#2f6fed", fg="white",
            activebackground="#2456c0", activeforeground="white",
            padx=12, pady=8,
        ).pack(side="right")
        tk.Button(
            btns, text="儲存", command=self.save,
            font=("Segoe UI", 10, "bold"), width=8,
            bg="#2f6fed", fg="white",
            activebackground="#2456c0", activeforeground="white",
            padx=16, pady=8,
        ).pack(side="right", padx=(0, 8))
        tk.Button(
            btns, text="取消", command=self.destroy,
            font=("Segoe UI", 10), width=12,
            padx=16, pady=8,
        ).pack(side="right", padx=(0, 8))

        root_body = tk.Frame(self)
        root_body.pack(side="top", fill="both", expand=True)

        # 分頁：模型 / 加速 / 進階 —— 避免整串選項把視窗撐爆
        nb = ttk.Notebook(root_body)
        nb.pack(fill="both", expand=True)
        tab_model = tk.Frame(nb, padx=18, pady=14)
        tab_perf = tk.Frame(nb, padx=18, pady=14)
        tab_adv = tk.Frame(nb, padx=18, pady=14)
        nb.add(tab_model, text="模型")
        nb.add(tab_perf, text="加速")
        nb.add(tab_adv, text="進階")

        # ============ 頁籤一：模型 ============
        body = tab_model
        tk.Label(body, text=f"模型：{profile.get('name','')}",
                 font=("Segoe UI", 9), fg="#666", anchor="w").pack(anchor="w", pady=(0, 8))

        # ---- 方案（同一模型的多套啟動參數）
        tk.Label(body, text="方案名稱（同一模型可有多套參數，如 code / chat / agent）",
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", pady=(2, 2))
        self.scheme_var = tk.StringVar(
            value=normalize_scheme(profile.get("scheme")))
        scheme_row = tk.Frame(body)
        scheme_row.pack(fill="x")
        tk.Entry(scheme_row, textvariable=self.scheme_var,
                 font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True, ipady=3)
        tk.Button(scheme_row, text="另存為…", command=self.save_as_new_scheme,
                  font=("Segoe UI", 9)).pack(side="left", padx=(6, 0))

        # ---- 視覺模型（mmproj）
        tk.Label(body, text="視覺模型（mmproj）", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(2, 2))
        mmproj_values = ["（無 vision）"] + scan_mmproj_files()
        current_mmproj = profile.get("mmproj") or "（無 vision）"
        if current_mmproj not in mmproj_values:
            mmproj_values.append(current_mmproj)
        self.mmproj_var = tk.StringVar(value=current_mmproj)
        self.vision_enabled_var = tk.BooleanVar(value=profile_vision_enabled(profile))
        mmproj_row = tk.Frame(body)
        mmproj_row.pack(fill="x")
        self.mmproj_combo = ttk.Combobox(
            mmproj_row, textvariable=self.mmproj_var, values=mmproj_values,
            state="readonly")
        self.mmproj_combo.pack(side="left", fill="x", expand=True)

        def update_vision_toggle(enable_selected=False):
            has_mmproj = self.mmproj_var.get().strip() != "（無 vision）"
            self.vision_check.config(state="normal" if has_mmproj else "disabled")
            if not has_mmproj:
                self.vision_enabled_var.set(False)
            elif enable_selected:
                self.vision_enabled_var.set(True)

        def browse_mmproj():
            f = filedialog.askopenfilename(
                initialdir=str(MODELS_DIR), filetypes=[("Vision projector GGUF", "*.gguf")])
            if f:
                self.mmproj_var.set(relative_model_name(Path(f)))
                update_vision_toggle(enable_selected=True)

        tk.Button(mmproj_row, text="瀏覽…", command=browse_mmproj).pack(
            side="left", padx=(6, 0))
        self.vision_check = tk.Checkbutton(
            body, text="啟用視覺模型（關閉時保留配對檔，下次可直接開回來）",
            variable=self.vision_enabled_var, anchor="w")
        self.vision_check.pack(fill="x", pady=(4, 0))
        self.mmproj_combo.bind(
            "<<ComboboxSelected>>", lambda _e: update_vision_toggle(enable_selected=True))
        update_vision_toggle()
        # ---- 記憶長度
        tk.Label(body, text="記憶長度（Context，K）", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(6, 2))
        self.ctx_var = tk.StringVar(
            value=format_context_k(profile.get("default_ctx", 131072)))
        ctx_row = tk.Frame(body)
        ctx_row.pack(fill="x")
        self.ctx_entry = tk.Entry(ctx_row, textvariable=self.ctx_var, width=10)
        self.ctx_entry.pack(side="left")
        tk.Label(ctx_row, text="K", font=("Segoe UI", 9)).pack(side="left", padx=(5, 0))
        # ---- 預設運算後端
        tk.Label(body, text="預設運算後端（Backend）", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        self.backend_var = tk.StringVar(value=profile.get("backend", "cuda").upper())
        self.backend_combo = ttk.Combobox(
            body, textvariable=self.backend_var, values=["CUDA", "Vulkan"],
            state="readonly")
        self.backend_combo.pack(fill="x")

        # ---- 思考模式
        tk.Label(body, text="思考模式（Reasoning / Thinking）", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        reasoning = str(profile.get("reasoning", "off")).strip().lower()
        if reasoning not in {"on", "off", "auto"}:
            reasoning = "off"
        self.reasoning_var = tk.StringVar(value=reasoning)
        rf = tk.Frame(body)
        rf.pack(fill="x")
        tk.Radiobutton(rf, text="關閉（回應較快）", variable=self.reasoning_var,
                       value="off").pack(side="left")
        tk.Radiobutton(rf, text="開啟（會先想再回答，較慢）", variable=self.reasoning_var,
                       value="on").pack(side="left")
        tk.Radiobutton(rf, text="auto（依模型模板）", variable=self.reasoning_var,
                       value="auto").pack(side="left")

        # ---- 思考強度
        tk.Label(body, text="思考強度（Reasoning effort）", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        self.reasoning_effort_var = tk.StringVar(
            value=reasoning_effort_value(profile))
        self.reasoning_effort_combo = ttk.Combobox(
            body, textvariable=self.reasoning_effort_var,
            values=list(REASONING_EFFORTS), state="readonly")
        self.reasoning_effort_combo.pack(fill="x")
        tk.Label(
            body,
            text="只在思考模式「開啟」時生效；default＝由模型模板決定。",
            font=("Segoe UI", 8), fg="#888", anchor="w",
            wraplength=540, justify="left",
        ).pack(anchor="w", pady=(2, 0))

        def _sync_settings_effort_state(*_args):
            self.reasoning_effort_combo.config(
                state="readonly" if self.reasoning_var.get() == "on"
                else "disabled")
        self.reasoning_var.trace_add("write", _sync_settings_effort_state)
        _sync_settings_effort_state()


        # ============ 頁籤二：加速 ============
        body = tab_perf
        # ---- MTP speculative decoding
        tk.Label(body, text="MTP 多 Token 預測", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        self.mtp_var = tk.StringVar(value="On" if profile.get("mtp", False) else "Off")
        self.mtp_combo = ttk.Combobox(
            body, textvariable=self.mtp_var, values=["Off", "On"],
            state="readonly")
        self.mtp_combo.pack(fill="x")
        mtp_row = tk.Frame(body)
        mtp_row.pack(fill="x", pady=(2, 0))
        tk.Label(mtp_row, text="--spec-draft-n-max", font=("Consolas", 9),
                 fg="#888").pack(side="left")
        self.spec_n_max_var = tk.StringVar(value=str(profile.get("spec_draft_n_max") or ""))
        tk.Entry(mtp_row, textvariable=self.spec_n_max_var, width=6,
                 font=("Consolas", 9)).pack(side="left", padx=(6, 0))
        tk.Label(mtp_row, text="每輪猜測 token 數；留空＝llama.cpp 預設（MTP 建議 5）",
                 font=("Segoe UI", 8), fg="#888").pack(side="left", padx=(8, 0))

        # ---- Jinja chat template
        tk.Label(body, text="Jinja 聊天模板", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        legacy_jinja = "--jinja" in (profile.get("extra_args", "") or "").split()
        self.jinja_var = tk.BooleanVar(value=bool(profile.get("jinja", legacy_jinja)))
        tk.Checkbutton(
            body,
            text="啟用 --jinja（使用 GGUF 內建聊天模板）",
            variable=self.jinja_var,
            anchor="w",
        ).pack(fill="x")

        # ---- 顯示卡分配
        tk.Label(body, text="顯示卡分配", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        gpu_values = [o[0] for o in GPU_OPTIONS]
        current_gpu = gpu_value_to_label(profile.get("gpu_split", ""))
        # 常用分配（使用者自訂過、存進全域清單的）排在中間
        for preset in gpu_preset_options():
            if preset not in gpu_values:
                gpu_values.append(preset)
        if current_gpu not in gpu_values and current_gpu:
            gpu_values.append(current_gpu)
        self.gpu_var = tk.StringVar(value=current_gpu)
        self.gpu_custom_var = tk.StringVar(
            value=profile.get("gpu_split", "") or "")
        self.gpu_combo = ttk.Combobox(body, textvariable=self.gpu_var,
                                      values=gpu_values,
                                      state="readonly", width=28)
        self.gpu_combo.pack(fill="x")

        def on_gpu_custom(_event=None):
            if self.gpu_var.get() != "自訂層數分配…":
                return
            value = simpledialog.askstring(
                "自訂顯示卡分配",
                "請輸入各 GPU 的層數，逗號分隔（例：16,8 = GPU0 16 層、GPU1 8 層）。\n"
                "留空 = 自動。\n\n"
                "儲存後會加入「常用分配」清單，之後可直接選。",
                initialvalue=self.gpu_custom_var.get() or "16,8",
                parent=self)
            if value is None:
                self.gpu_var.set(current_gpu or "自動（讓程式決定）")
                return
            value = value.strip()
            self.gpu_custom_var.set(value)
            if value:
                self.gpu_var.set(f"自訂：{value}")
            else:
                self.gpu_var.set("自動（讓程式決定）")

        self.gpu_combo.bind("<<ComboboxSelected>>", on_gpu_custom)

        # ---- KV 快取精度
        tk.Label(body, text="KV 快取精度", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        self.kv_var = tk.StringVar(value=kv_label_from_mode(kv_mode_from_profile(profile)))
        self.kv_combo = ttk.Combobox(
            body, textvariable=self.kv_var, values=[label for label, _ in KV_OPTIONS],
            state="readonly")
        self.kv_combo.pack(fill="x")
        # ---- 並行請求
        tk.Label(body, text="同時服務幾個請求", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        self.parallel_var = tk.StringVar(value=str(parallel_from_profile(profile)))
        pr = tk.Frame(body)
        pr.pack(fill="x")
        for val, label in [("1", "1（一般）"), ("2", "2"), ("4", "4")]:
            tk.Radiobutton(pr, text=label, variable=self.parallel_var,
                           value=val).pack(side="left")

        # ---- 其他 runtime 選項（留空／auto＝llama.cpp 預設）
        tk.Label(body, text="其他加速選項（auto／留空＝llama.cpp 預設）",
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", pady=(8, 2))
        opt_grid = tk.Frame(body)
        opt_grid.pack(fill="x")
        flash_attn = str(profile.get("flash_attn") or "auto").strip().lower()
        if flash_attn not in {"auto", "on", "off"}:
            flash_attn = "auto"
        self.flash_attn_var = tk.StringVar(value=flash_attn)
        tk.Label(opt_grid, text="Flash Attention：", font=("Segoe UI", 9),
                 anchor="e").grid(row=0, column=0, sticky="e")
        ttk.Combobox(opt_grid, textvariable=self.flash_attn_var,
                     values=["auto", "on", "off"], state="readonly",
                     width=7).grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.kv_unified_var = tk.BooleanVar(value=bool(profile.get("kv_unified", True)))
        self.fit_var = tk.BooleanVar(value=str(profile.get("fit") or "on").strip() != "off")
        tk.Checkbutton(opt_grid, text="KV 統一緩衝（--kv-unified，需搭配 RAM cache）",
                       variable=self.kv_unified_var, anchor="w",
                       font=("Segoe UI", 9)).grid(row=1, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(opt_grid, text="自動調整未設定參數以塞進顯存（--fit，預設開）",
                       variable=self.fit_var, anchor="w",
                       font=("Segoe UI", 9)).grid(row=2, column=0, columnspan=2, sticky="w")
        num_grid = tk.Frame(body)
        num_grid.pack(fill="x", pady=(6, 0))
        self.threads_var = tk.StringVar(value=str(profile.get("threads") or ""))
        self.threads_batch_var = tk.StringVar(value=str(profile.get("threads_batch") or ""))
        self.ctx_checkpoints_var = tk.StringVar(value=str(profile.get("ctx_checkpoints") or ""))
        for row, (label, var) in enumerate((
                ("Threads（CPU 執行緒）", self.threads_var),
                ("Threads-batch（batch 執行緒）", self.threads_batch_var),
                ("Context checkpoints", self.ctx_checkpoints_var))):
            tk.Label(num_grid, text=label, font=("Segoe UI", 8),
                     fg="#888").grid(row=row, column=0, sticky="w")
            tk.Entry(num_grid, textvariable=var, width=8,
                     font=("Consolas", 9)).grid(row=row, column=1, sticky="w")
        tk.Label(body, text="留空＝不傳該參數，由 llama.cpp 用預設值。",
                 font=("Segoe UI", 8), fg="#888", anchor="w").pack(anchor="w")

        # ---- GPU offload 層數（-ngl）
        tk.Label(body, text="GPU 卸載層數（-ngl）", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        ngl_values = ["全部（999）", "64", "32", "16"]
        current_ngl = str(profile.get("ngl", 999))
        self.ngl_var = tk.StringVar(
            value=current_ngl if current_ngl in ngl_values else "全部（999）")
        self.ngl_custom_var = tk.StringVar(value=current_ngl)
        self.ngl_combo = ttk.Combobox(body, textvariable=self.ngl_var,
                                      values=ngl_values, state="readonly")
        self.ngl_combo.pack(fill="x")

        def on_ngl_custom(_event=None):
            if self.ngl_var.get() != "全部（999）":
                return
            value = simpledialog.askstring(
                "自訂 GPU 層數",
                "輸入要卸載到 GPU 的層數（例如 48）。\n"
                "999 = 全部層數都放 GPU；較小值 = 部分層數留給 CPU。",
                initialvalue=self.ngl_custom_var.get() or "999",
                parent=self)
            if value is None:
                self.ngl_var.set(self.ngl_custom_var.get() or "全部（999）")
                return
            value = value.strip()
            if value and value != "999":
                self.ngl_custom_var.set(value)
                self.ngl_var.set(value)
            else:
                self.ngl_var.set("全部（999）")

        self.ngl_combo.bind("<<ComboboxSelected>>", on_ngl_custom)

        # ============ 頁籤三：進階 ============
        body = tab_adv

        # ---- 採樣預設（server 端 completion 預設值）
        tk.Label(body, text="採樣預設（server 端預設值；留空＝llama.cpp 預設）",
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", pady=(8, 2))
        sam_grid = tk.Frame(body)
        sam_grid.pack(fill="x")
        self.temp_var = tk.StringVar(value=str(profile.get("temp") or ""))
        self.top_p_var = tk.StringVar(value=str(profile.get("top_p") or ""))
        self.top_k_var = tk.StringVar(value=str(profile.get("top_k") or ""))
        self.min_p_var = tk.StringVar(value=str(profile.get("min_p") or ""))
        self.presence_penalty_var = tk.StringVar(
            value=str(profile.get("presence_penalty") or ""))
        self.repeat_penalty_var = tk.StringVar(
            value=str(profile.get("repeat_penalty") or ""))
        sampling_fields = (
            ("Temp", self.temp_var), ("Top-P", self.top_p_var),
            ("Top-K", self.top_k_var), ("Min-P", self.min_p_var),
            ("Presence Penalty", self.presence_penalty_var),
            ("Repeat Penalty", self.repeat_penalty_var),
        )
        for i, (label, var) in enumerate(sampling_fields):
            row, col = divmod(i, 2)
            tk.Label(sam_grid, text=label, font=("Segoe UI", 8),
                     fg="#888").grid(row=row, column=col * 2, sticky="w")
            tk.Entry(sam_grid, textvariable=var, width=7,
                     font=("Consolas", 9)).grid(row=row, column=col * 2 + 1,
                                                sticky="w", padx=(4, 16))

        # ---- 進階參數（extra args）
        tk.Label(body, text="進階參數（extra args）", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        self.extra_args_var = tk.StringVar(value=profile.get("extra_args", "") or "")
        tk.Entry(body, textvariable=self.extra_args_var,
                 font=("Consolas", 9)).pack(fill="x", ipady=3)
        tk.Label(body, text="給進階使用者：直接編輯額外的 llama-server 參數。"
                            "KV 快取、並行數、Jinja 由上方選項管理，會自動剔除重複。"
                            "KV 選「自訂」時，這裡的 -ctk/-ctv 會原樣保留。",
                 font=("Segoe UI", 8), fg="#888", anchor="w",
                 wraplength=600, justify="left").pack(anchor="w", pady=(2, 0))

        # ---- 完整參數（raw，bat 式全權控制）
        tk.Label(body, text="完整啟動參數（raw；非空時覆蓋上方所有選項，bat 式）",
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", pady=(8, 2))
        self.raw_args_var = tk.StringVar(value=profile.get("raw_args", "") or "")
        raw_frame = tk.Frame(body)
        raw_frame.pack(fill="x")
        self.raw_args_text = tk.Text(raw_frame, height=4, font=("Consolas", 9),
                                     wrap="none")
        raw_sb = tk.Scrollbar(raw_frame, command=self.raw_args_text.yview)
        self.raw_args_text.configure(yscrollcommand=raw_sb.set)
        raw_sb.pack(side="right", fill="y")
        self.raw_args_text.pack(side="left", fill="both", expand=True)
        self.raw_args_text.insert("1.0", self.raw_args_var.get())
        self.raw_args_text.bind("<KeyRelease>", lambda _e: self._update_preview())
        self.raw_args_text.bind("<FocusOut>", lambda _e: self._update_preview())

        # ---- 最終指令預覽
        tk.Label(body, text="最終啟動指令預覽（實際會執行的完整參數）",
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", pady=(8, 2))
        prev_frame = tk.Frame(body)
        prev_frame.pack(fill="x")
        self.preview_text = tk.Text(prev_frame, height=6, font=("Consolas", 8),
                                    state="disabled", bg="#10141c", fg="#d5dbe5",
                                    wrap="none")
        prev_sb = tk.Scrollbar(prev_frame, command=self.preview_text.yview)
        self.preview_text.configure(yscrollcommand=prev_sb.set)
        prev_sb.pack(side="right", fill="y")
        self.preview_text.pack(side="left", fill="both", expand=True)
        self.after(100, self._update_preview)

    # ---------------- 參數收集／預覽
    def _collect_fields(self, silent: bool = False) -> dict | None:
        """把視窗欄位收集成 profile dict（不含 scheme）；驗證失敗回 None。"""
        p = dict(self.profile)
        # mmproj / vision
        mmproj = self.mmproj_var.get().strip()
        p["mmproj"] = "" if mmproj == "（無 vision）" else mmproj
        p["vision_enabled"] = bool(p["mmproj"] and self.vision_enabled_var.get())
        # context
        try:
            p["default_ctx"] = parse_context_k(self.ctx_var.get())
        except ValueError as exc:
            if not silent:
                messagebox.showwarning("Context格式錯誤", str(exc), parent=self)
            return None
        p["backend"] = self.backend_var.get().lower()
        # reasoning
        p["reasoning"] = self.reasoning_var.get()
        p["reasoning_effort"] = reasoning_effort_value(
            {"reasoning_effort": self.reasoning_effort_var.get()})
        # 顯示卡分配
        gpu_label = self.gpu_var.get()
        if gpu_label == "自訂層數分配…":
            gpu_value = self.gpu_custom_var.get().strip()
        elif gpu_label.startswith("自訂："):
            gpu_value = gpu_label.split("：", 1)[1].strip()
        elif gpu_label == "自動（讓程式決定）":
            gpu_value = ""
        else:
            gpu_value = gpu_label_to_value(gpu_label)
        if gpu_value:
            remember_gpu_preset(gpu_value)
        p["gpu_split"] = gpu_value
        # KV 模式
        p["kv_mode"] = next((v for label, v in KV_OPTIONS
                             if label == self.kv_var.get()), "q4")
        p["parallel"] = int(self.parallel_var.get() or 1)
        # ngl：combo 值為「全部（999）」/「64」/「32」/「16」/自訂數字字串
        ngl_label = self.ngl_var.get()
        if ngl_label == "全部（999）":
            p["ngl"] = 999
        elif ngl_label.isdigit():
            p["ngl"] = max(1, int(ngl_label))
        else:
            try:
                p["ngl"] = max(1, int(self.ngl_custom_var.get() or 999))
            except ValueError:
                if not silent:
                    messagebox.showwarning("NGL 格式錯誤", "請輸入整數層數", parent=self)
                return None
        # 加速選項
        p["mtp"] = self.mtp_var.get() == "On"
        n_max = self.spec_n_max_var.get().strip()
        p["spec_draft_n_max"] = n_max
        p["jinja"] = self.jinja_var.get()
        p["flash_attn"] = self.flash_attn_var.get()
        p["kv_unified"] = self.kv_unified_var.get()
        p["fit"] = "on" if self.fit_var.get() else "off"
        p["threads"] = self.threads_var.get().strip()
        p["threads_batch"] = self.threads_batch_var.get().strip()
        p["ctx_checkpoints"] = self.ctx_checkpoints_var.get().strip()
        # 進階參數：GUI 管理項由選項重組，其餘原樣保留
        extra = preserve_unmanaged_extra_args(
            self.extra_args_var.get(), manage_kv=(p["kv_mode"] != "custom"))
        p["extra_args"] = " ".join(extra)
        # 採樣預設
        p["temp"] = self.temp_var.get().strip()
        p["top_p"] = self.top_p_var.get().strip()
        p["top_k"] = self.top_k_var.get().strip()
        p["min_p"] = self.min_p_var.get().strip()
        p["presence_penalty"] = self.presence_penalty_var.get().strip()
        p["repeat_penalty"] = self.repeat_penalty_var.get().strip()
        # raw 完整參數
        p["raw_args"] = self.raw_args_text.get("1.0", "end").strip()
        # 保留置頂狀態
        p["starred"] = bool(self.profile.get("starred", False))
        if p["starred"] and self.profile.get("favorite_order") is not None:
            p["favorite_order"] = self.profile.get("favorite_order")
        return p

    def _update_preview(self, *_args):
        """重建「最終啟動指令預覽」；欄位有誤時顯示錯誤而不是崩掉。"""
        if getattr(self, "preview_text", None) is None or not self.preview_text.winfo_exists():
            return
        p = self._collect_fields(silent=True)
        binary = VULKAN_SERVER if p.get("backend") == "vulkan" else LLAMA_SERVER
        if p is None:
            text = "（Context 欄位格式有誤，無法組裝預覽）"
        else:
            try:
                ctx = parse_context_k(self.ctx_var.get())
                mmproj = model_file(p["mmproj"]) if p.get("mmproj") else None
                args = build_server_args(
                    p, ctx, current_server_settings(), model_file(p["model"]),
                    mmproj)
                text = format_command(binary, args)
            except Exception as exc:
                text = f"（預覽失敗：{exc}）"
        self.preview_text.config(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("end", text)
        self.preview_text.config(state="disabled")

    def save(self):
        self._save_current_scheme()

    def _save_current_scheme(self):
        p = self._collect_fields()
        if p is None:
            return
        scheme = normalize_scheme(self.scheme_var.get())
        new_key = (p["model"], scheme)
        cfg = self.app.cfg
        cfg.setdefault("profiles", [])
        for sp in cfg["profiles"]:
            if profile_key(sp) == new_key and new_key != self.original_key:
                messagebox.showerror(
                    "方案名稱重複",
                    f"「{scheme}」已存在。\n請換一個名稱，或先用「另存新方案」建立新方案。",
                    parent=self)
                return
        p["scheme"] = scheme
        self._persist_to_cfg(p)
        self.original_key = new_key
        self.app.refresh_listbox(new_key)
        self.app.update_detail()
        self.destroy()
        messagebox.showinfo("已儲存", f"設定已更新（方案：{scheme}）。")

    def save_as_new_scheme(self):
        """把目前欄位值複製成一個新方案。"""
        p = self._collect_fields()
        if p is None:
            return
        name = simpledialog.askstring(
            "另存新方案",
            "新方案名稱（例如 code / chat / agent）：",
            initialvalue=normalize_scheme(self.scheme_var.get()), parent=self)
        if name is None:
            return
        scheme = normalize_scheme(name)
        if scheme == self.original_key[1]:
            messagebox.showerror(
                "名稱重複", "新方案名稱不能與目前方案相同。", parent=self)
            return
        cfg = self.app.cfg
        cfg.setdefault("profiles", [])
        if any(profile_key(sp) == (p["model"], scheme) for sp in cfg["profiles"]):
            messagebox.showerror("方案名稱重複", f"「{scheme}」已存在。", parent=self)
            return
        p["scheme"] = scheme
        p["starred"] = False
        p.pop("favorite_order", None)
        self._persist_to_cfg(p)
        # 切換到新方案繼續編輯
        self.original_key = (p["model"], scheme)
        self.profile = p
        self.scheme_var.set(scheme)
        self.app.refresh_listbox(self.original_key)
        self.app.update_detail()
        messagebox.showinfo("已建立", f"已建立新方案「{scheme}」，可繼續調整。")

    def _persist_to_cfg(self, p: dict):
        key = profile_key(p)
        cfg = self.app.cfg
        cfg.setdefault("profiles", [])
        for i, sp in enumerate(cfg["profiles"]):
            if profile_key(sp) == key:
                cfg["profiles"][i] = dict(p)
                break
        else:
            cfg["profiles"].append(dict(p))
        save_config(cfg)
        self.app.profiles = merge_profiles(cfg)

class AddModelDialog(tk.Toplevel):
    """加入新模型：名稱 + GGUF + mmproj（可空）+ 預設 ctx + reasoning + 思考強度。"""

    def __init__(self, parent, app: LauncherApp):
        super().__init__(parent)
        self.app = app
        self.title("加入新模型")
        win_w, win_h = fit_window_size(self, S(520), S(470))
        self.geometry(f"{win_w}x{win_h}")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        body = tk.Frame(self, padx=16, pady=12)
        body.pack(fill="both", expand=True)

        def field(label):
            tk.Label(body, text=label, font=("Segoe UI", 9),
                     anchor="w").pack(fill="x", pady=(8, 2))
            frame = tk.Frame(body)
            frame.pack(fill="x")
            entry = tk.Entry(frame)
            entry.pack(side="left", fill="x", expand=True)
            return entry

        tk.Label(body, text="加入新模型（存到 models.json，不需改程式）",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")

        self.name_entry = field("顯示名稱")
        self.model_entry = field("GGUF 檔 (models\\)")

        def browse_model():
            f = filedialog.askopenfilename(
                initialdir=str(MODELS_DIR), filetypes=[("GGUF", "*.gguf")])
            if f:
                self.model_entry.delete(0, "end")
                self.model_entry.insert(0, relative_model_name(Path(f)))
        tk.Button(body, text="瀏覽…", command=browse_model).pack(pady=(2, 0), anchor="e")

        self.mmproj_entry = field("mmproj 檔 (可留空 = 無 vision)")

        def browse_mmproj():
            f = filedialog.askopenfilename(
                initialdir=str(MODELS_DIR), filetypes=[("GGUF", "*.gguf")])
            if f:
                self.mmproj_entry.delete(0, "end")
                self.mmproj_entry.insert(0, relative_model_name(Path(f)))
        tk.Button(body, text="瀏覽…", command=browse_mmproj).pack(pady=(2, 0), anchor="e")

        tk.Label(body, text="預設 Context（K）", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        self.ctx_var = tk.StringVar(value="128")
        ctx_row = tk.Frame(body)
        ctx_row.pack(fill="x")
        self.ctx_entry = tk.Entry(ctx_row, textvariable=self.ctx_var)
        self.ctx_entry.pack(side="left", fill="x", expand=True)
        tk.Label(ctx_row, text="K").pack(side="left", padx=(5, 0))

        tk.Label(body, text="Reasoning (thinking)", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        self.reasoning_var = tk.StringVar(value="off")
        rf = tk.Frame(body)
        rf.pack(fill="x")
        tk.Radiobutton(rf, text="off", variable=self.reasoning_var,
                       value="off").pack(side="left")
        tk.Radiobutton(rf, text="on", variable=self.reasoning_var,
                       value="on").pack(side="left")

        tk.Label(body, text="思考強度 (reasoning effort)", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        self.reasoning_effort_var = tk.StringVar(value="default")
        self.reasoning_effort_combo = ttk.Combobox(
            body, textvariable=self.reasoning_effort_var,
            values=list(REASONING_EFFORTS), state="readonly")
        self.reasoning_effort_combo.pack(fill="x")
        tk.Label(body, text="只在思考模式 on 時生效；default＝由模型模板決定。",
                 font=("Segoe UI", 8), fg="#888", anchor="w").pack(anchor="w")

        def _sync_add_effort_state(*_args):
            self.reasoning_effort_combo.config(
                state="readonly" if self.reasoning_var.get() == "on"
                else "disabled")
        self.reasoning_var.trace_add("write", _sync_add_effort_state)
        _sync_add_effort_state()

        btns = tk.Frame(self, padx=16, pady=10)
        btns.pack(fill="x")
        tk.Button(btns, text="儲存", command=self.save,
                  bg="#2f6fed", fg="white", padx=20).pack(side="right")
        tk.Button(btns, text="取消", command=self.destroy).pack(side="right", padx=6)

    def save(self):
        name = self.name_entry.get().strip()
        model = self.model_entry.get().strip()
        if not name or not model:
            messagebox.showwarning("請填完整", "名稱與 GGUF 檔為必填")
            return
        if not model_file(model).exists():
            messagebox.showwarning("檔案不存在",
                                   f"models\\{model} 不存在，請確認檔名")
            return
        try:
            ctx_val = parse_context_k(self.ctx_var.get())
        except ValueError as exc:
            messagebox.showwarning("Context格式錯誤", str(exc), parent=self)
            return
        mmproj = self.mmproj_entry.get().strip()
        if mmproj and not model_file(mmproj).exists():
            messagebox.showwarning(
                "視覺模型不存在", f"models\\{mmproj} 不存在，請確認檔名", parent=self)
            return
        cfg = self.app.cfg
        cfg.setdefault("profiles", [])
        existing = [p for p in cfg["profiles"] if p.get("model") == model]
        if existing:
            messagebox.showerror(
                "模型已存在",
                f"「{model}」已在清單中（{len(existing)} 個方案）。\n\n"
                "要加另一套參數：選該模型 →「Settings」→「另存新方案」。",
                parent=self)
            return
        profile = dict(DEFAULT_PROFILE)
        profile.update({
            "name": name,
            "model": model,
            "mmproj": mmproj,
            "vision_enabled": bool(mmproj),
            "default_ctx": ctx_val,
            "reasoning": self.reasoning_var.get(),
            "reasoning_effort": reasoning_effort_value(
                {"reasoning_effort": self.reasoning_effort_var.get()}),
        })
        cfg["profiles"].append(profile)
        save_config(cfg)
        invalidate_model_inventory()
        self.app.profiles = merge_profiles(cfg)
        self.app.refresh_listbox()
        self.app.update_detail()
        self.destroy()
        messagebox.showinfo("已加入", f"「{name}」已加入清單。")


def main():
    enable_dpi_awareness()
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = LauncherApp(root)
    # 單一實例：重複啟動時喚醒舊實例並退出
    single = SingleInstance(on_show=app.show_window_from_tray)
    if not single.acquire():
        root.destroy()
        return
    app.single_instance = single
    root.mainloop()


if __name__ == "__main__":
    main()
