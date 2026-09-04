from __future__ import annotations

import os
import secrets
from pathlib import Path


def ensure_control_token(path: Path) -> str:
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    if token:
        return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return token


def read_api_key(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def write_api_key(path: Path, value: str) -> None:
    """value 為空字串時清除既有關鍵字。"""
    value = value.strip()
    if not value:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
