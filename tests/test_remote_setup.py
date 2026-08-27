from llama_launcher.remote_setup import configure_remote_access
from llama_launcher.tailscale import CommandResult


class FakeManager:
    def __init__(self, enable, status):
        self._enable = enable
        self._status = status

    def enable_control_serve(self, port):
        assert port == 8765
        return self._enable

    def serve_status(self):
        return self._status

    @staticmethod
    def extract_https_url(text):
        import re
        match = re.search(r"https://[^\s]+\.ts\.net", text)
        return match.group(0) if match else None

    @staticmethod
    def extract_authorization_url(text):
        import re
        match = re.search(r"https://login\.tailscale\.com/[^\s]+", text)
        return match.group(0) if match else None


def test_remote_setup_success():
    manager = FakeManager(
        CommandResult(True, "ok", "", 0),
        CommandResult(True, "https://pc.tail.ts.net\nproxy", "", 0),
    )
    result = configure_remote_access(manager)
    assert result.ok
    assert result.https_url == "https://pc.tail.ts.net"


def test_remote_setup_surfaces_authorization():
    manager = FakeManager(
        CommandResult(False, "https://login.tailscale.com/f/serve?node=x", "", 1),
        CommandResult(False, "", "unused", 1),
    )
    result = configure_remote_access(manager)
    assert not result.ok
    assert result.authorization_url.endswith("node=x")
