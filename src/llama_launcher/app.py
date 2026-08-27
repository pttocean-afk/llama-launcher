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
from .migration import (
    detect_legacy_dir,
    merge_legacy_into_config,
    migrate_legacy_data,
    plan_migration,
)
from .paths import logs_dir, profiles_path, resource_dir, settings_path, token_path
from .remote_setup import configure_remote_access
from .security import ensure_control_token
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
PORT = 8080
CONTROL_PORT = 8765
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


# Vulkan 224K 以上啟動前的安全 VRAM 預檢。
NVIDIA_SMI = nvidia_smi_path()
VRAM_PREFLIGHT_GPU0_LIMIT_MB = 2304
VRAM_PREFLIGHT_GPU1_LIMIT_MB = 128
VRAM_PREFLIGHT_WAIT_SECONDS = 10
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


KV_OPTIONS = [
    ("F16 / F16（最高精度，最吃顯存）", "f16"),
    ("Q4 / Q4（正式版，現有設定）", "q4"),
    ("IQ4-NL / IQ4-NL（K/V 同 IQ4-NL，品質略優於 Q4）", "iq4_nl"),
]


def kv_mode_from_profile(profile: dict) -> str:
    """相容舊 models.json：優先讀 kv_mode，否則由 extra_args 推斷。"""
    mode = profile.get("kv_mode")
    if mode in {"f16", "q4", "iq4_nl"}:
        return mode
    extra = profile.get("extra_args", "") or ""
    if "-ctk iq4_nl" in extra and "-ctv iq4_nl" in extra:
        return "iq4_nl"
    if "-ctk q4_0" in extra and "-ctv q4_0" in extra:
        return "q4"
    return "f16"


def kv_label_from_mode(mode: str) -> str:
    return next((label for label, value in KV_OPTIONS if value == mode), KV_OPTIONS[1][0])


def preserve_unmanaged_extra_args(extra: str) -> list[str]:
    """移除 GUI 管理的 KV/parallel/Jinja，保留使用者的 -b/-ub 等手動參數。"""
    parts = (extra or "").split()
    out = []
    i = 0
    managed = {"-ctk", "--cache-type-k", "-ctv", "--cache-type-v", "--parallel", "-np"}
    while i < len(parts):
        part = parts[i]
        if part == "--jinja":
            i += 1
            continue
        if part in managed:
            i += 2
        else:
            out.append(part)
            i += 1
    return out

# 預設組態（對應原本 bat 的參數）
DEFAULT_PROFILE = {
    "name": "",
    "model": "",            # 相對 models\ 的檔名
    "mmproj": "",           # 相對 models\ 的檔名，可空
    "vision_enabled": False, # 保留mmproj配對，但可在啟動時暫停載入
    "default_ctx": 131072,
    "reasoning": "off",     # off / on
    "gpu_split": "16,8",
    "backend": "cuda",      # cuda / vulkan
    "jinja": False,          # 使用 GGUF 內建 Jinja chat template
    "extra_args": "-ctk q4_0 -ctv q4_0 --parallel 1",
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
    """掃描 models\ 下的 .gguf，排除 mmproj。"""
    models, _mmprojs, _sizes = _model_inventory()
    return list(models)


def scan_mmproj_files() -> list[str]:
    _models, mmprojs, _sizes = _model_inventory()
    return list(mmprojs)


def _model_inventory() -> tuple[list[str], list[str], dict[str, int]]:
    """一次掃描模型目錄並快取名稱與大小，避免每次點選/搜尋都碰磁碟。"""
    global _MODEL_INVENTORY_CACHE
    if _MODEL_INVENTORY_CACHE is not None:
        return _MODEL_INVENTORY_CACHE
    models: list[str] = []
    mmprojs: list[str] = []
    sizes: dict[str, int] = {}
    if MODELS_DIR.exists():
        try:
            entries = sorted(MODELS_DIR.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            entries = []
        for f in entries:
            if f.suffix.lower() != ".gguf":
                continue
            try:
                sizes[f.name] = f.stat().st_size
            except OSError:
                sizes[f.name] = 0
            (mmprojs if _is_mmproj(f.name) else models).append(f.name)
    _MODEL_INVENTORY_CACHE = (models, mmprojs, sizes)
    return _MODEL_INVENTORY_CACHE


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
    for key, mm in RULES:
        if key in base and mm in mmproj_list:
            return mm
    model_tokens = {t for t in re.split(r"[^a-z0-9]+", base) if len(t) >= 3}
    for mm in mmproj_list:
        key = re.sub(r"^mmproj[-_]?", "", mm, flags=re.I)
        key = re.sub(r"\.gguf$", "", key, flags=re.I).lower()
        mm_tokens = {t for t in re.split(r"[^a-z0-9]+", key) if len(t) >= 3}
        common = model_tokens & mm_tokens
        if len(common) >= 2:
            return mm
    return ""


def merge_profiles(cfg: dict) -> list[dict]:
    """已存設定 + 掃描到的模型合併成顯示清單（不覆寫已存設定）。"""
    saved = {p.get("model"): p for p in cfg.get("profiles", [])}
    found = scan_gguf_files()
    mmprojs = scan_mmproj_files()
    merged = []
    for name in found:
        if name in saved:
            p = dict(saved[name])
            p["configured"] = True
        else:
            p = dict(DEFAULT_PROFILE)
            p["name"] = name.replace(".gguf", "")
            p["model"] = name
            p["mmproj"] = guess_mmproj(name, mmprojs)
            p["default_ctx"] = 131072
            p["configured"] = False
        merged.append(p)
    # ★ 置頂模型依使用者指定順序排列；未置頂模型照名稱排序。
    merged.sort(key=lambda p: (
        not bool(p.get("starred")),
        int(p.get("favorite_order", 999999)) if p.get("starred") else 999999,
        p["name"].lower(),
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
        """Adopt the unique llama-server process explicitly configured for port 8080."""
        if self.running:
            return True, f"已管理 PID {self.pid_text()}"
        matches = list_llama_servers(PORT)
        if len(matches) != 1:
            if not matches:
                return False, "沒有命令列明確使用port 8080的llama-server"
            return False, f"找到 {len(matches)} 個port 8080候選，為安全起見不自動接管"

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
        return values if len(values) >= 2 else None

    def query_gpu1_processes(self) -> list[str]:
        """列出 nvidia-smi 可見、綁在第二張 GPU 的 process，供阻止啟動時說明。"""
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
        gpu1_uuid = ""
        for line in gpu_result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 1)]
            if len(parts) == 2 and parts[0] == "1":
                gpu1_uuid = parts[1]
                break
        if not gpu1_uuid:
            return []
        owners = []
        for line in app_result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) != 3 or parts[2] != gpu1_uuid:
                continue
            owners.append(f"{Path(parts[1]).name} (PID {parts[0]})")
        return sorted(set(owners))

    def _stop_comfyui_if_active(self) -> tuple[bool, str]:
        """Apply the legacy Windows/WSL ComfyUI policy only on Windows hosts."""
        if not IS_WINDOWS:
            return True, "Linux host cleanup policy not configured"
        check_args = [
            "wsl.exe", "-d", "Ubuntu", "--",
            "systemctl", "is-active", "--quiet", "comfyui-raylight.service",
        ]
        try:
            check = self._run_hidden(check_args)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"無法檢查 ComfyUI service：{exc}"
        if check.returncode != 0:
            return True, "ComfyUI inactive"
        stop_args = [
            "wsl.exe", "-d", "Ubuntu", "--",
            "systemctl", "stop", "comfyui-raylight.service",
        ]
        try:
            stopped = self._run_hidden(stop_args, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"停止 ComfyUI 失敗：{exc}"
        if stopped.returncode != 0:
            detail = (stopped.stderr or stopped.stdout).strip()
            return False, f"停止 ComfyUI 失敗：{detail or 'unknown error'}"
        return True, "ComfyUI stopped"

    def _close_vram_cleanup_allowlist(self):
        """Apply the approved Windows process cleanup allowlist only on Windows."""
        if not IS_WINDOWS:
            return
        for name in VRAM_CLEANUP_PROCESS_NAMES:
            try:
                self._run_hidden(["taskkill.exe", "/IM", name, "/T", "/F"], timeout=10)
            except (OSError, subprocess.SubprocessError):
                pass

    def run_vram_preflight(self) -> tuple[bool, str]:
        """Apply the current host-specific high-context Vulkan safety policy."""
        if not IS_WINDOWS:
            return True, "VRAM preflight skipped: no Linux host cleanup policy configured"
        before = self.query_gpu_memory_mb()
        if before is None:
            return False, "無法讀取兩張 NVIDIA GPU 的 VRAM，已取消 Vulkan 224K+ 啟動"

        comfy_ok, comfy_status = self._stop_comfyui_if_active()
        if not comfy_ok:
            return False, comfy_status
        self._close_vram_cleanup_allowlist()

        deadline = time.monotonic() + VRAM_PREFLIGHT_WAIT_SECONDS
        after = before
        while True:
            current = self.query_gpu_memory_mb()
            if current is None:
                return False, "VRAM 清理後無法重新讀取 GPU 狀態，已取消啟動"
            after = current
            if (after[0] <= VRAM_PREFLIGHT_GPU0_LIMIT_MB and
                    after[1] <= VRAM_PREFLIGHT_GPU1_LIMIT_MB):
                break
            if time.monotonic() >= deadline:
                owners = self.query_gpu1_processes()
                owners_text = ("\nGPU1 process：" + ", ".join(owners)) if owners else ""
                return False, (
                    "VRAM 尚未回到安全基線，已取消 Vulkan 224K+ 啟動。\n"
                    f"GPU0：{after[0]} MB（需 ≤ {VRAM_PREFLIGHT_GPU0_LIMIT_MB}）\n"
                    f"GPU1：{after[1]} MB（需 ≤ {VRAM_PREFLIGHT_GPU1_LIMIT_MB}）"
                    f"{owners_text}\n"
                    "請關閉其他 AI／GPU 程式後再試。"
                )
            time.sleep(0.5)

        summary = (
            f"VRAM preflight: GPU0 {before[0]}→{after[0]} MB, "
            f"GPU1 {before[1]}→{after[1]} MB, {comfy_status}"
        )
        return True, summary

    def start(self, profile: dict, ctx_label: str) -> tuple[bool, str]:
        """啟動 llama-server（CREATE_NO_WINDOW，stdout/stderr 導流到 log 檔）。"""
        if self.running:
            return False, "llama-server 已在執行中"
        mtp_enabled = bool(profile.get("mtp", False))
        backend = profile.get("backend", "cuda")
        windows_server = VULKAN_SERVER if backend == "vulkan" else LLAMA_SERVER
        if not windows_server.exists():
            return False, f"找不到 {windows_server}"
        model_rel = Path("models") / profile["model"]
        if not (LLAMA_DIR / model_rel).exists():
            return False, f"找不到模型檔：{model_rel}"
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
            mmproj_path = LLAMA_DIR / Path("models") / profile["mmproj"]
            if not mmproj_path.exists():
                return False, f"找不到視覺模型檔：models\\{profile['mmproj']}"
        self.preflight_summary = "not required"
        if backend == "vulkan" and ctx >= 229376:
            preflight_ok, preflight_msg = self.run_vram_preflight()
            if not preflight_ok:
                return False, preflight_msg
            self.preflight_summary = preflight_msg
        server_args = [str(windows_server), "-m", str(LLAMA_DIR / model_rel)]
        if mmproj_path is not None:
            server_args += ["--mmproj", str(mmproj_path)]
        server_args += [
            "-ngl", str(int(profile.get("ngl", 999))),
            "-c", str(ctx),
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "-sm", "layer",
            # RAM prompt cache 上限 24GB（全 model 共用；本機 64GB RAM，多輪 agent 重用 prefix 省 prefill）
            "-cram", "24000",
        ]
        if backend == "vulkan":
            server_args += ["--device", "Vulkan0,Vulkan1"]
        gpu_split = (profile.get("gpu_split") or "").strip()
        if gpu_split:
            server_args += ["-ts", gpu_split]
        server_args += ["-mg", "0"]
        # extra_args（KV 快取、--parallel 等）；--parallel 由這裡決定，缺了才補 1
        # 先放共用 runtime 參數，再放 reasoning，排列與原本 bat 對齊。
        if profile.get("extra_args"):
            server_args += profile["extra_args"].split()
        if "--parallel" not in server_args:
            server_args += ["--parallel", "1"]
        if profile.get("reasoning") == "on":
            server_args += ["--reasoning", "on"]
        else:
            server_args += ["--reasoning", "off"]
        if profile.get("jinja", False) and "--jinja" not in server_args:
            server_args += ["--jinja"]
        if mtp_enabled:
            server_args += ["--spec-type", "draft-mtp"]

        args = server_args
        launch_cwd = windows_server.parent

        # log 檔
        LOGS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = LOGS_DIR / f"llama-server-{ts}.log"
        self.log_fh = open(self.log_path, "wb", buffering=0)
        # 寫入啟動指令（方便之後查用了什麼參數）
        header = (f"# {datetime.now().isoformat()}  {profile.get('name','')}\n"
                  f"# {self.preflight_summary}\n"
                  f"# {' '.join(args)}\n"
                  f"{'='*80}\n").encode("utf-8")
        self.log_fh.write(header)

        try:
            self.proc = subprocess.Popen(
                args,
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

    def scan_runtime_health(self):
        """從新增 log 內容偵測會讓雙 GPU 推論大幅掉速的 runtime fallback。

        先 stat 比對大小，log 沒新增就不開檔（每秒輪詢時避免磁碟 IO）。"""
        p = self.log_path
        if p is None or not p.exists():
            return
        try:
            size = p.stat().st_size
            if size == self._health_scan_size and self._health_scan_offset > 0:
                return
            if self._health_scan_offset > size:
                self._health_scan_offset = 0
            with open(p, "rb") as fh:
                fh.seek(self._health_scan_offset)
                chunk = fh.read(size - self._health_scan_offset)
                self._health_scan_offset = fh.tell()
                self._health_scan_size = size
            text = chunk.decode("utf-8", errors="replace")
            if "retrying without pipeline parallelism" in text:
                self.degraded_reason = (
                    "Vulkan compute buffer 配置失敗，已退回無 pipeline parallelism 慢速模式"
                )
        except OSError:
            pass

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
:root{color-scheme:dark}*{box-sizing:border-box}body{font:15px system-ui,-apple-system,sans-serif;background:#0b1220;color:#e8eef7;max-width:980px;margin:0 auto;padding:24px 16px}h1{margin:0;font-size:25px}h2{font-size:16px;margin:0 0 14px;color:#a9c2e2}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}.badge{padding:6px 12px;border-radius:99px;background:#26344b;color:#a9bad3}.badge.on{background:#164d37;color:#79e0a8}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.card{background:#151f30;border:1px solid #27364d;border-radius:12px;padding:16px;margin:12px 0}.metric{font-size:20px;font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.label{font-size:11px;color:#8fa3bf;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}button,select,input{font:inherit;padding:9px 12px;border-radius:7px;border:1px solid #405574;background:#202d42;color:#fff}input{width:100%;max-width:100%;min-width:0}button{cursor:pointer;background:#2869bb;border-color:#397ed4;font-weight:600}button.secondary{background:#27344a;border-color:#40516a}button.danger{background:#9e3e4d;border-color:#c95768}button:disabled{opacity:.45;cursor:not-allowed}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.profile{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:12px;border:1px solid #2b3b54;border-radius:9px;margin:8px 0;background:#111a29}.profile small{display:block;color:#8fa3bf;margin-top:4px}.profile button{white-space:nowrap}pre{white-space:pre-wrap;max-height:430px;overflow:auto;background:#0a101b;padding:12px;border-radius:8px;color:#b9c9dc;font-size:12px}@media(max-width:650px){.grid{grid-template-columns:repeat(2,1fr)}.top{align-items:flex-start;gap:12px;flex-direction:column}}
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
function renderProfiles(list,running){const box=$('profiles');box.textContent='';if(!list.length){const none=document.createElement('div');none.style.color='#8fa3bf';none.textContent='No profiles found.';box.appendChild(none);return}for(const x of list){const row=document.createElement('div');row.className='profile';const info=document.createElement('div');const name=document.createElement('b');name.textContent=x.name;info.appendChild(name);const detail=document.createElement('small');detail.textContent=String(x.backend||'').toUpperCase()+' · '+((Number(x.default_ctx)||0)/1024).toFixed(0)+'K context · reasoning '+(x.reasoning||'');info.appendChild(detail);row.appendChild(info);const btn=document.createElement('button');btn.textContent='Start';if(!running){btn.addEventListener('click',()=>start(String(x.model||'')))}else{btn.disabled=true}row.appendChild(btn);box.appendChild(row)}}
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
        self.geometry("860x540")
        self.minsize(600, 300)

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

        # ---- Modern dark dashboard: favorites/control on the left, full-height log on the right.
        root.title("llama.cpp Launcher")
        root.geometry("1180x820")
        root.minsize(980, 700)
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

        header = tk.Frame(root, bg="#141a24", height=58)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="LLAMA  CONTROL  CENTER",
                 font=("Segoe UI", 15, "bold"), fg="#f4f7fb",
                 bg="#141a24", padx=18).pack(side="left", fill="y")
        self.status_dot = tk.Label(header, text="○  Server stopped",
                                   font=("Segoe UI", 10, "bold"), fg="#f0ad4e",
                                   bg="#141a24", padx=18)
        self.status_dot.pack(side="right", fill="y")

        dashboard = tk.PanedWindow(root, orient="horizontal", sashwidth=7,
                                   sashrelief="flat", bg="#0d1118",
                                   bd=0, relief="flat")
        # 四邊留白均勻：左右 dashboard padx 12 + 內容 padx 16 = 28；
        # 底部也留 28，讓左下角按鈕不會貼邊。
        dashboard.pack(fill="both", expand=True, padx=12, pady=(12, 28))

        left = tk.Frame(dashboard, bg="#171e2a", width=350,
                        highlightthickness=1, highlightbackground="#293244")
        right = tk.Frame(dashboard, bg="#0c1119",
                         highlightthickness=1, highlightbackground="#293244")
        dashboard.add(left, minsize=310, width=350)
        dashboard.add(right, minsize=520)

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
        self.thinking_var = tk.BooleanVar(value=False)
        control_label("CONTEXT (K)", 0, 0)
        control_label("BACKEND", 0, 1)
        self.ctx_entry = tk.Entry(
            controls, textvariable=self.ctx_var, font=("Segoe UI", 10),
            bg="#202838", fg="#eef2f8", insertbackground="white",
            relief="flat", highlightthickness=1, highlightbackground="#303b4e")
        self.ctx_entry.grid(row=1, column=0, sticky="ew", padx=(0, 6),
                            pady=(0, 10), ipady=5)
        self.backend_combo = combo(self.backend_var, ["CUDA", "Vulkan"], 1, 1)
        self.vision_check = tk.Checkbutton(
            controls, text="Enable vision projector", variable=self.vision_var,
            font=("Segoe UI", 9, "bold"), bg="#171e2a", fg="#dce6f3",
            activebackground="#171e2a", activeforeground="white",
            selectcolor="#253045", anchor="w")
        self.vision_check.grid(row=2, column=0, sticky="ew", padx=(0, 6),
                               pady=(0, 10))
        self.thinking_check = tk.Checkbutton(
            controls, text="Enable Thinking", variable=self.thinking_var,
            font=("Segoe UI", 9, "bold"), bg="#171e2a", fg="#dce6f3",
            activebackground="#171e2a", activeforeground="white",
            selectcolor="#253045", anchor="w")
        self.thinking_check.grid(row=2, column=1, sticky="ew", padx=(6, 0),
                                 pady=(0, 10))

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
        utility_row.pack(fill="x", padx=16, pady=(8, 0))
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

        log_header = tk.Frame(right, bg="#141a24", height=46)
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
        dialog.geometry("620x280")
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
        report = diagnostics_dict(collect_diagnostics(LLAMA_DIR))
        labels = {
            "llama_dir": "llama.cpp folder",
            "llama_server_exists": llama_server_filename(),
            "models_dir_exists": "Models folder",
            "model_count": "Model count",
            "inference_port_8080": "Inference port 8080",
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
        self.server.scan_runtime_health()
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
                 "backend": p.get("backend", "cuda"),
                 "default_ctx": p.get("default_ctx", 131072),
                 "reasoning": p.get("reasoning", "off")}
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
        profile = next((p for p in self.profiles if p.get("model") == model), None)
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
        for i, saved in enumerate(self.cfg["profiles"]):
            if saved.get("model") == profile.get("model"):
                self.cfg["profiles"][i] = dict(profile)
                break
        else:
            self.cfg["profiles"].append(dict(profile))

    def refresh_listbox(self, select_model: str | None = None):
        """首頁只顯示置頂模型；完整清單由 ModelLibraryDialog 管理。"""
        if select_model is None:
            current = self.current_profile() if hasattr(self, "favorite_profiles") else None
            select_model = current.get("model") if current else None
        self.favorite_profiles = [p for p in self.profiles if p.get("starred")]
        self.listbox.delete(0, "end")
        # 列表只顯示名稱（詳細資訊在右側 detail 面板看，避免被擠掉）
        for p in self.favorite_profiles:
            self.listbox.insert("end", p["name"])
        if not self.favorite_profiles:
            self.listbox.insert("end", "No favorite models — open All models")
            self.listbox.itemconfig(0, fg="#7f8b9d")
            return
        index = next((i for i, p in enumerate(self.favorite_profiles)
                      if p.get("model") == select_model), 0)
        self.listbox.selection_set(index)
        self.listbox.activate(index)
        self.listbox.see(index)

    def set_profile_starred(self, profile: dict, starred: bool):
        p = dict(profile)
        p["starred"] = starred
        if starred:
            orders = [int(x.get("favorite_order", -1)) for x in self.profiles
                      if x.get("starred") and x.get("model") != p.get("model")]
            if "favorite_order" not in p:
                p["favorite_order"] = max(orders, default=-1) + 1
        else:
            p.pop("favorite_order", None)
        self._persist_profile(p)
        save_config(self.cfg)
        self.profiles = merge_profiles(self.cfg)
        self._normalize_favorite_orders()
        self.refresh_listbox(select_model=p.get("model") if starred else None)
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
        favorites = [p for p in self.profiles if p.get("starred")]
        index = next((i for i, p in enumerate(favorites)
                      if p.get("model") == model), -1)
        target = index + delta
        if index < 0 or target < 0 or target >= len(favorites):
            return
        favorites[index], favorites[target] = favorites[target], favorites[index]
        for order, p in enumerate(favorites):
            p["favorite_order"] = order
            self._persist_profile(p)
        save_config(self.cfg)
        self.profiles = merge_profiles(self.cfg)
        self.refresh_listbox(select_model=model)
        self.update_detail()

    def move_favorite(self, delta: int):
        p = self.current_profile()
        if p:
            self.move_favorite_model(p["model"], delta)

    def toggle_star(self):
        p = self.current_profile()
        if p:
            self.set_profile_starred(p, not bool(p.get("starred")))

    def open_model_library(self):
        ModelLibraryDialog(self.root, self)

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
            lines = [
                p["name"],
                f"Size: {size or '?'}   ·   {p.get('backend','cuda').upper()}",
                vision_state,
                f"Default context: {format_context_k(p.get('default_ctx', 131072))}K",
            ]
            self.detail_text.insert("end", "\n".join(lines))
        self.detail_text.config(state="disabled")
        if p:
            self.backend_var.set(p.get("backend", "cuda").upper())
            self.ctx_var.set(format_context_k(p.get("default_ctx", 131072)))
            self.thinking_var.set(p.get("reasoning", "off") == "on")
            has_mmproj = bool(p.get("mmproj"))
            self.vision_var.set(profile_vision_enabled(p))
            self.vision_check.config(state="normal" if has_mmproj else "disabled")
        else:
            self.vision_var.set(False)
            self.thinking_var.set(False)
            self.vision_check.config(state="disabled")

    # ---------------- 動作
    def on_launch(self):
        p = self.current_profile()
        if p is None:
            messagebox.showinfo("提示", "請先選擇一個模型")
            return
        p = dict(p)
        try:
            parse_context_k(self.ctx_var.get())
        except ValueError as exc:
            messagebox.showerror("Context格式錯誤", str(exc))
            return
        p["backend"] = self.backend_var.get().lower()
        p["vision_enabled"] = bool(p.get("mmproj")) and self.vision_var.get()
        p["reasoning"] = "on" if self.thinking_var.get() else "off"
        # 首頁 Context只影響本次啟動；預設值在 Settings裡修改。
        self.cfg.setdefault("profiles", [])
        for i, sp in enumerate(self.cfg["profiles"]):
            if sp.get("model") == p["model"]:
                self.cfg["profiles"][i] = p
                break
        else:
            self.cfg["profiles"].append(p)
        save_config(self.cfg)
        self.profiles = merge_profiles(self.cfg)

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
        model_file = MODELS_DIR / p["model"]
        if self.server.running and self.server.profile_name == p.get("name"):
            messagebox.showerror(
                "無法刪除", f"「{model_name}」正在執行中。請先停止伺服器。",
                parent=parent)
            return False

        size = model_size_text(p["model"]) or "?"
        files_to_delete = [("主模型", model_file)]
        mmproj_name = p.get("mmproj") or ""
        keep_mmproj_note = ""
        if mmproj_name:
            mmproj_file = MODELS_DIR / mmproj_name
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
        self.server.scan_runtime_health()
        if self.server.degraded_reason and not self.server.degraded_warning_shown:
            self.server.degraded_warning_shown = True
            reason = self.server.degraded_reason
            with self.server_lock:
                stop_message = self.server.stop()
            messagebox.showwarning(
                "llama-server 慢速模式（已自動停止）",
                f"{reason}\n\n"
                "這個狀態可能讓生成速度降到約 7–12 t/s，"
                "launcher 已自動停止伺服器。\n"
                f"{stop_message}\n\n"
                "目前即使 nvidia-smi 顯示 GPU1 0 MB，Vulkan allocator仍可能失敗。\n"
                "請重新啟動 Windows，或改用較低 context後再試。",
            )
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
        self.geometry("780x620")
        self.minsize(680, 500)
        self.configure(bg="#111722")
        self.transient(parent)

        header = tk.Frame(self, bg="#171e2a", height=60)
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

    def refresh(self, select_model: str | None = None):
        current = self.current()
        if select_model is None and current:
            select_model = current.get("model")
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
                "end", f"{star}  {p['name']}    [{size}]    {media} · {backend}")
        if self.filtered:
            index = next((i for i, p in enumerate(self.filtered)
                          if p.get("model") == select_model), 0)
            self.listbox.selection_set(index)
            self.listbox.see(index)

    def toggle_pin(self):
        p = self.current()
        if not p:
            return
        self.app.set_profile_starred(p, not bool(p.get("starred")))
        self.refresh(select_model=p.get("model"))

    def move(self, delta: int):
        p = self.current()
        if not p:
            return
        if not p.get("starred"):
            messagebox.showinfo("提示", "請先把模型設為置頂。", parent=self)
            return
        self.app.move_favorite_model(p["model"], delta)
        self.refresh(select_model=p["model"])

    def settings(self):
        p = self.current()
        if not p:
            return
        dialog = SettingsDialog(self, self.app, p)
        self.wait_window(dialog)
        self.refresh(select_model=p.get("model"))

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
        self.geometry("560x720")
        self.minsize(520, 640)
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        body = tk.Frame(self, padx=18, pady=14)
        body.pack(fill="both", expand=True)

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

        settings = _load_settings()
        settings["close_to_tray"] = bool(self.close_to_tray_var.get())
        _save_settings(settings)

        if not set_autostart(bool(self.autostart_var.get())):
            messagebox.showwarning(
                "開機啟動設定失敗",
                "無法寫入開機啟動設定（可能權限不足），其他設定已儲存。",
                parent=self)
        self.destroy()
        messagebox.showinfo("已儲存", "全域設定已更新。")


class SettingsDialog(tk.Toplevel):
    """口語化設定視窗：把技術參數翻譯成一般人看得懂的選項。"""

    def __init__(self, parent, app: LauncherApp, profile: dict):
        super().__init__(parent)
        self.app = app
        self.profile = profile
        self.title(f"設定 — {profile.get('name','')}")
        self.geometry("620x900")
        self.minsize(580, 820)
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        # 底部按鈕列先 pack，固定保留空間，不會被上方內容擠出視窗。
        btns = tk.Frame(self, padx=16)
        btns.pack(side="bottom", fill="x", pady=(10, 16))
        tk.Button(
            btns, text="儲存", command=self.save,
            font=("Segoe UI", 10, "bold"), width=12,
            bg="#2f6fed", fg="white",
            activebackground="#2456c0", activeforeground="white",
            padx=16, pady=8,
        ).pack(side="right")
        tk.Button(
            btns, text="取消", command=self.destroy,
            font=("Segoe UI", 10), width=12,
            padx=16, pady=8,
        ).pack(side="right", padx=(0, 10))

        body = tk.Frame(self, padx=16, pady=12)
        body.pack(side="top", fill="both", expand=True)

        tk.Label(body, text="個別模型設定",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(body, text="全域設定（llama.cpp 路徑、開機啟動、Remote Access）請到主畫面 Settings。",
                 font=("Segoe UI", 8), fg="#888", anchor="w").pack(anchor="w", pady=(2, 6))

        tk.Label(body, text=f"模型：{profile.get('name','')}",
                 font=("Segoe UI", 9), fg="#666", anchor="w").pack(anchor="w", pady=(0, 8))

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
                self.mmproj_var.set(Path(f).name)
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
        # ---- MTP speculative decoding
        tk.Label(body, text="MTP 多 Token 預測", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        self.mtp_var = tk.StringVar(value="On" if profile.get("mtp", False) else "Off")
        self.mtp_combo = ttk.Combobox(
            body, textvariable=self.mtp_var, values=["Off", "On"],
            state="readonly")
        self.mtp_combo.pack(fill="x")
        tk.Label(
            body,
            text="只適用含 MTP / NextN 權重的模型；會增加 VRAM 使用量。預設關閉。",
            font=("Segoe UI", 8), fg="#888", anchor="w",
        ).pack(anchor="w")

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
        tk.Label(
            body,
            text="可能改善 Qwen 多輪 Thinking／工具格式；若客戶端不相容可關閉。",
            font=("Segoe UI", 8), fg="#888", anchor="w",
        ).pack(anchor="w")

        # ---- 思考模式
        tk.Label(body, text="思考模式（Reasoning / Thinking）", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        self.reasoning_var = tk.StringVar(value=profile.get("reasoning", "off"))
        rf = tk.Frame(body)
        rf.pack(fill="x")
        tk.Radiobutton(rf, text="關閉（回應較快）", variable=self.reasoning_var,
                       value="off").pack(side="left")
        tk.Radiobutton(rf, text="開啟（會先想再回答，較慢）", variable=self.reasoning_var,
                       value="on").pack(side="left")

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
        tk.Label(body, text="例：16,8 = 第一張卡分 16 層、第二張分 8 層。"
                            "實際可用層數取決於模型與 VRAM。",
                 font=("Segoe UI", 8), fg="#888", anchor="w").pack(anchor="w")
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
        self.parallel_var = tk.StringVar(value="1")
        try:
            extra = profile.get("extra_args", "") or ""
            parts = extra.split()
            if "--parallel" in parts:
                i = parts.index("--parallel")
                if i + 1 < len(parts):
                    self.parallel_var.set(parts[i + 1])
        except Exception:
            pass
        pr = tk.Frame(body)
        pr.pack(fill="x")
        for val, label in [("1", "1（一般）"), ("2", "2"), ("4", "4")]:
            tk.Radiobutton(pr, text=label, variable=self.parallel_var,
                           value=val).pack(side="left")

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

        # ---- 進階參數（extra args）
        tk.Label(body, text="進階參數（extra args）", font=("Segoe UI", 9),
                 anchor="w").pack(fill="x", pady=(8, 2))
        self.extra_args_var = tk.StringVar(value=profile.get("extra_args", "") or "")
        tk.Entry(body, textvariable=self.extra_args_var,
                 font=("Consolas", 9)).pack(fill="x", ipady=3)
        tk.Label(body, text="給進階使用者：直接編輯額外的 llama-server 參數。"
                            "KV 快取、並行數、Jinja 由上方選項管理，會自動保留。",
                 font=("Segoe UI", 8), fg="#888", anchor="w",
                 wraplength=540, justify="left").pack(anchor="w", pady=(2, 0))

    def save(self):
        """把口語選項寫回 profile 並存檔（全域設定在主畫面 Settings）。"""
        p = self.profile
        try:
            ctx_val = parse_context_k(self.ctx_var.get())
        except ValueError as exc:
            messagebox.showerror("Context格式錯誤", str(exc), parent=self)
            return
        mmproj = self.mmproj_var.get().strip()
        mmproj = "" if mmproj == "（無 vision）" else mmproj
        if mmproj and not (MODELS_DIR / mmproj).exists():
            messagebox.showerror(
                "視覺模型不存在",
                f"找不到 models\\{mmproj}。\n請重新掃描或選擇正確的 GGUF 檔。",
                parent=self)
            return

        p["default_ctx"] = ctx_val
        p["reasoning"] = self.reasoning_var.get()
        gpu_label = self.gpu_var.get()
        if gpu_label == "自訂層數分配…":
            # combo 只選了選項但沒輸入數值：直接取暫存的自訂值
            p["gpu_split"] = self.gpu_custom_var.get().strip()
        elif gpu_label.startswith("自訂："):
            p["gpu_split"] = gpu_label[3:].strip()
        else:
            p["gpu_split"] = gpu_label_to_value(gpu_label)
        # 自訂分配加入全域常用清單，之後每個模型的設定都能直接選
        remember_gpu_preset(p["gpu_split"])
        p["backend"] = self.backend_var.get().lower()
        p["mtp"] = self.mtp_var.get() == "On"
        p["jinja"] = self.jinja_var.get()
        p["mmproj"] = mmproj
        p["vision_enabled"] = bool(mmproj) and self.vision_enabled_var.get()

        # 組 extra_args：GUI 管理的 KV/parallel 由選項產生；
        # 使用者在進階參數框輸入的其他參數（-b/-ub 等）保留。
        extra = preserve_unmanaged_extra_args(self.extra_args_var.get())
        kv_label = self.kv_var.get()
        kv_mode = next((value for label, value in KV_OPTIONS if label == kv_label), "q4")
        p["kv_mode"] = kv_mode
        if kv_mode == "q4":
            extra += ["-ctk", "q4_0", "-ctv", "q4_0"]
        elif kv_mode == "iq4_nl":
            extra += ["-ctk", "iq4_nl", "-ctv", "iq4_nl"]
        else:
            extra += ["-ctk", "f16", "-ctv", "f16"]
        extra += ["--parallel", self.parallel_var.get()]
        p["extra_args"] = " ".join(extra)

        # GPU 卸載層數
        ngl_label = self.ngl_var.get()
        if ngl_label == "全部（999）":
            p["ngl"] = 999
        elif ngl_label.isdigit():
            p["ngl"] = int(ngl_label)
        else:
            p["ngl"] = int(self.ngl_custom_var.get() or 999)

        # 存回 cfg
        cfg = self.app.cfg
        cfg.setdefault("profiles", [])
        for i, sp in enumerate(cfg["profiles"]):
            if sp.get("model") == p["model"]:
                cfg["profiles"][i] = dict(p)
                break
        else:
            cfg["profiles"].append(dict(p))
        save_config(cfg)
        self.app.profiles = merge_profiles(cfg)
        self.app.refresh_listbox()
        self.app.update_detail()
        self.destroy()
        messagebox.showinfo("已儲存", "設定已更新。")


class AddModelDialog(tk.Toplevel):
    """加入新模型：名稱 + GGUF + mmproj（可空）+ 預設 ctx + reasoning。"""

    def __init__(self, parent, app: LauncherApp):
        super().__init__(parent)
        self.app = app
        self.title("加入新模型")
        self.geometry("520x380")
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
                self.model_entry.insert(0, Path(f).name)
        tk.Button(body, text="瀏覽…", command=browse_model).pack(pady=(2, 0), anchor="e")

        self.mmproj_entry = field("mmproj 檔 (可留空 = 無 vision)")

        def browse_mmproj():
            f = filedialog.askopenfilename(
                initialdir=str(MODELS_DIR), filetypes=[("GGUF", "*.gguf")])
            if f:
                self.mmproj_entry.delete(0, "end")
                self.mmproj_entry.insert(0, Path(f).name)
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
        if not (MODELS_DIR / model).exists():
            messagebox.showwarning("檔案不存在",
                                   f"models\\{model} 不存在，請確認檔名")
            return
        try:
            ctx_val = parse_context_k(self.ctx_var.get())
        except ValueError as exc:
            messagebox.showwarning("Context格式錯誤", str(exc), parent=self)
            return
        mmproj = self.mmproj_entry.get().strip()
        if mmproj and not (MODELS_DIR / mmproj).exists():
            messagebox.showwarning(
                "視覺模型不存在", f"models\\{mmproj} 不存在，請確認檔名", parent=self)
            return
        profile = {
            "name": name,
            "model": model,
            "mmproj": mmproj,
            "vision_enabled": bool(mmproj),
            "default_ctx": ctx_val,
            "reasoning": self.reasoning_var.get(),
            "gpu_split": "16,8",
            "backend": "cuda",
            "extra_args": "-ctk q4_0 -ctv q4_0 --parallel 1",
        }
        cfg = self.app.cfg
        cfg.setdefault("profiles", [])
        cfg["profiles"] = [p for p in cfg["profiles"] if p.get("model") != model]
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
