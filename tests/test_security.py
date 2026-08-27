from pathlib import Path

from llama_launcher.security import ensure_control_token


def test_token_is_stable_and_not_short(tmp_path: Path):
    path = tmp_path / "secrets" / "token"
    first = ensure_control_token(path)
    second = ensure_control_token(path)
    assert first == second
    assert len(first) >= 40
    assert path.read_text(encoding="utf-8").endswith("\n")
