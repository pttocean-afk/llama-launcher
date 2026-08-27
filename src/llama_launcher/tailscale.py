from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .host import hidden_run_kwargs

TAILSCALE_URL_RE = re.compile(r"https://[a-zA-Z0-9.-]+\.ts\.net/?")
AUTH_URL_RE = re.compile(r"https://login\.tailscale\.com/[^\s]+")


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int


class TailscaleManager:
    def __init__(self, executable: Path | None = None):
        self.executable = executable or self.locate()

    @staticmethod
    def locate() -> Path | None:
        located = shutil.which("tailscale.exe") or shutil.which("tailscale")
        candidates = [located, r"C:\Program Files\Tailscale\tailscale.exe"]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return Path(candidate)
        return None

    def run(self, *args: str, timeout: int = 30) -> CommandResult:
        if self.executable is None:
            return CommandResult(False, "", "Tailscale is not installed", 127)
        try:
            proc = subprocess.run(
                [str(self.executable), *args], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=timeout,
                **hidden_run_kwargs(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return CommandResult(False, "", str(exc), 1)
        return CommandResult(proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip(), proc.returncode)

    def serve_status(self) -> CommandResult:
        return self.run("serve", "status")

    def enable_control_serve(self, port: int = 8765) -> CommandResult:
        return self.run("serve", "--bg", f"http://127.0.0.1:{port}", timeout=60)

    @staticmethod
    def extract_https_url(text: str) -> str | None:
        match = TAILSCALE_URL_RE.search(text)
        return match.group(0).rstrip("/") if match else None

    @staticmethod
    def extract_authorization_url(text: str) -> str | None:
        match = AUTH_URL_RE.search(text)
        return match.group(0) if match else None
