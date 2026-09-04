import sys
from pathlib import Path

import pytest

from llama_launcher.host import command_uses_port, llama_server_filename


def test_command_uses_explicit_port_forms():
    assert command_uses_port(("llama-server", "--port", "8080"), 8080)
    assert command_uses_port(("llama-server", "--port=8080"), 8080)
    assert not command_uses_port(("llama-server", "--port", "8081"), 8080)
    assert not command_uses_port(("llama-server", "--other", "8080"), 8080)


def test_platform_server_filename():
    assert llama_server_filename() in {"llama-server", "llama-server.exe"}


@pytest.mark.skipif(sys.platform == "win32",
                    reason="Linux-only: sets ~/.config/autostart desktop file")
def test_autostart_toggle(tmp_path, monkeypatch):
    """Linux autostart writes/removes ~/.config/autostart/llama-launcher.desktop."""
    import llama_launcher.host as host
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    assert host.autostart_enabled() is False
    assert host.set_autostart(True) is True
    assert host.autostart_enabled() is True
    desktop = home / ".config" / "autostart" / "llama-launcher.desktop"
    assert desktop.exists()
    assert "[Desktop Entry]" in desktop.read_text(encoding="utf-8")
    assert host.set_autostart(False) is True
    assert host.autostart_enabled() is False
    assert not desktop.exists()
