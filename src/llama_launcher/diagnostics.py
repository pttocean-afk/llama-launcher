from __future__ import annotations

import socket
from dataclasses import asdict, dataclass
from pathlib import Path

from .host import llama_server_filename
from .tailscale import TailscaleManager


@dataclass(frozen=True)
class DiagnosticReport:
    llama_dir: str
    llama_server_exists: bool
    models_dir_exists: bool
    model_count: int
    inference_port: bool
    control_port_8765: bool
    tailscale_installed: bool
    tailscale_serve_url: str | None


def port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def collect_diagnostics(llama_dir: Path, tailscale: TailscaleManager | None = None,
                        inference_port: int = 8080) -> DiagnosticReport:
    manager = tailscale or TailscaleManager()
    models = llama_dir / "models"
    try:
        model_count = sum(1 for p in models.glob("*.gguf") if "mmproj" not in p.name.lower())
    except OSError:
        model_count = 0
    serve = manager.serve_status() if manager.executable else None
    serve_text = "\n".join((serve.stdout, serve.stderr)) if serve else ""
    return DiagnosticReport(
        llama_dir=str(llama_dir),
        llama_server_exists=(llama_dir / llama_server_filename()).exists(),
        models_dir_exists=models.exists(),
        model_count=model_count,
        inference_port=port_open("127.0.0.1", inference_port),
        control_port_8765=port_open("127.0.0.1", 8765),
        tailscale_installed=manager.executable is not None,
        tailscale_serve_url=manager.extract_https_url(serve_text),
    )


def diagnostics_dict(report: DiagnosticReport) -> dict:
    return asdict(report)
