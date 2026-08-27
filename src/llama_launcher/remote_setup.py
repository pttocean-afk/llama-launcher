from __future__ import annotations

from dataclasses import dataclass

from .tailscale import TailscaleManager


@dataclass(frozen=True)
class RemoteSetupResult:
    ok: bool
    https_url: str | None
    authorization_url: str | None
    detail: str


def configure_remote_access(manager: TailscaleManager, port: int = 8765) -> RemoteSetupResult:
    enabled = manager.enable_control_serve(port)
    combined = "\n".join(part for part in (enabled.stdout, enabled.stderr) if part)
    auth_url = manager.extract_authorization_url(combined)
    if auth_url:
        return RemoteSetupResult(False, None, auth_url, combined)
    status = manager.serve_status()
    status_text = "\n".join(part for part in (status.stdout, status.stderr) if part)
    https_url = manager.extract_https_url(status_text or combined)
    return RemoteSetupResult(
        bool(enabled.ok and status.ok and https_url),
        https_url,
        None,
        status_text or combined or "No Tailscale output",
    )
