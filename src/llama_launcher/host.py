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
