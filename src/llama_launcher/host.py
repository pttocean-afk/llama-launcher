from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import psutil

IS_WINDOWS = os.name == "nt"
CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    executable: str
    command_line: str


def llama_server_filename() -> str:
    return "llama-server.exe" if IS_WINDOWS else "llama-server"


def command_uses_port(command: list[str] | tuple[str, ...], port: int) -> bool:
    expected = str(port)
    for index, arg in enumerate(command):
        if arg == "--port" and index + 1 < len(command) and command[index + 1] == expected:
            return True
        if arg == f"--port={expected}":
            return True
    return False


def list_llama_servers(port: int) -> list[ProcessInfo]:
    matches: list[ProcessInfo] = []
    for process in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            info = process.info
            name = str(info.get("name") or "").lower()
            executable = str(info.get("exe") or "")
            cmdline = tuple(str(part) for part in (info.get("cmdline") or ()))
            executable_name = Path(executable).name.lower() if executable else ""
            if "llama-server" not in name and "llama-server" not in executable_name:
                continue
            if not command_uses_port(cmdline, port):
                continue
            matches.append(ProcessInfo(int(info["pid"]), executable, subprocess.list2cmdline(cmdline)))
        except (psutil.Error, OSError, ValueError):
            continue
    return matches


def pid_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        return psutil.Process(int(pid)).is_running()
    except (psutil.Error, ValueError):
        return False


def terminate_process_tree(pid: int, timeout: float = 10.0) -> tuple[bool, str]:
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return True, f"Process {pid} is not running"
    processes = parent.children(recursive=True) + [parent]
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(processes, timeout=timeout)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(alive, timeout=3)
    return (not alive, f"Stopped PID {pid}" if not alive else f"PID {pid} is still running")


def hidden_run_kwargs() -> dict:
    return {"creationflags": CREATE_NO_WINDOW} if IS_WINDOWS else {}


def nvidia_smi_path() -> Path | None:
    located = shutil.which("nvidia-smi.exe" if IS_WINDOWS else "nvidia-smi")
    if located:
        return Path(located)
    if IS_WINDOWS:
        candidate = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        return candidate if candidate.exists() else None
    return None


def open_external(target: str | Path) -> bool:
    value = str(target)
    if re.match(r"^https?://", value, re.I):
        return webbrowser.open(value)
    if IS_WINDOWS:
        try:
            os.startfile(value)  # type: ignore[attr-defined]
            return True
        except OSError:
            return False
    opener = shutil.which("xdg-open") or shutil.which("gio")
    if not opener:
        return False
    args = [opener, "open", value] if Path(opener).name == "gio" else [opener, value]
    try:
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------- autostart
def _launch_command() -> str:
    """Command that starts the launcher at sign-in (frozen or dev)."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    return f'"{sys.executable}" -m llama_launcher'


def autostart_enabled() -> bool:
    """Whether the launcher is registered to start at sign-in."""
    if IS_WINDOWS:
        try:
            import winreg
            with winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Run") as key:
                winreg.QueryValueEx(key, "LlamaLauncher")
            return True
        except OSError:
            return False
    desktop = Path.home() / ".config" / "autostart" / "llama-launcher.desktop"
    return desktop.exists()


def set_autostart(enabled: bool) -> bool:
    """Register / unregister the launcher at sign-in. Returns success."""
    if IS_WINDOWS:
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0, winreg.KEY_SET_VALUE)
            if enabled:
                winreg.SetValueEx(key, "LlamaLauncher", 0,
                                  winreg.REG_SZ, _launch_command())
            else:
                try:
                    winreg.DeleteValue(key, "LlamaLauncher")
                except FileNotFoundError:
                    pass
            winreg.CloseKey(key)
            return True
        except OSError:
            return False
    autostart_dir = Path.home() / ".config" / "autostart"
    desktop = autostart_dir / "llama-launcher.desktop"
    if enabled:
        try:
            autostart_dir.mkdir(parents=True, exist_ok=True)
            desktop.write_text(
                "[Desktop Entry]\nType=Application\n"
                "Name=Llama Launcher\n"
                f"Exec={_launch_command()}\n"
                "X-GNOME-Autostart-enabled=true\n",
                encoding="utf-8")
            return True
        except OSError:
            return False
    try:
        desktop.unlink(missing_ok=True)
        return True
    except OSError:
        return False
