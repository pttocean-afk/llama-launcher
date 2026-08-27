from llama_launcher.host import command_uses_port, llama_server_filename


def test_command_uses_explicit_port_forms():
    assert command_uses_port(("llama-server", "--port", "8080"), 8080)
    assert command_uses_port(("llama-server", "--port=8080"), 8080)
    assert not command_uses_port(("llama-server", "--port", "8081"), 8080)
    assert not command_uses_port(("llama-server", "--other", "8080"), 8080)


def test_platform_server_filename():
    assert llama_server_filename() in {"llama-server", "llama-server.exe"}
