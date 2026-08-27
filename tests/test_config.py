import json
from pathlib import Path

from llama_launcher.config import load_profiles, save_json


def test_save_json_is_valid_and_newline_terminated(tmp_path: Path):
    path = tmp_path / "profiles.json"
    save_json(path, {"profiles": [{"name": "demo"}]})
    assert path.read_bytes().endswith(b"\n")
    assert json.loads(path.read_text(encoding="utf-8"))["profiles"][0]["name"] == "demo"


def test_invalid_profiles_fails_closed(tmp_path: Path):
    path = tmp_path / "profiles.json"
    path.write_text("{bad json", encoding="utf-8")
    assert load_profiles(path) == {"profiles": []}
