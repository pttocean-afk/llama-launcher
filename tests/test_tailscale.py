from llama_launcher.tailscale import TailscaleManager


def test_extract_serve_url():
    text = "Available within your tailnet:\nhttps://my-pc.tail123.ts.net/\n|-- proxy"
    assert TailscaleManager.extract_https_url(text) == "https://my-pc.tail123.ts.net"


def test_extract_authorization_url():
    text = "To enable, visit:\nhttps://login.tailscale.com/f/serve?node=abc"
    assert TailscaleManager.extract_authorization_url(text).endswith("node=abc")
