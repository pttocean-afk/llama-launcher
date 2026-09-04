from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "LlamaLauncher"


def install_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    override = os.environ.get("LLAMA_LAUNCHER_DATA_DIR")
    if override:
        root = Path(override).expanduser()
    elif (install_dir() / "portable.mode").exists():
        root = install_dir() / "data"
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def logs_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def backups_dir() -> Path:
    path = data_dir() / "backups"
    path.mkdir(parents=True, exist_ok=True)
    return path


def profiles_path() -> Path:
    return data_dir() / "profiles.json"


def settings_path() -> Path:
    return data_dir() / "settings.json"


def token_path() -> Path:
    secrets_dir = data_dir() / "secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    return secrets_dir / "control-token"


def api_key_path() -> Path:
    """llama-server --api-key 的存放位置（與 settings.json 分開，不隨設定外洩）。"""
    secrets_dir = data_dir() / "secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    return secrets_dir / "api-key"
