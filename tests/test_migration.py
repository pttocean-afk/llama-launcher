import json
from pathlib import Path

from llama_launcher.migration import migrate_legacy_data


def test_migration_renames_models_and_preserves_existing_destination(tmp_path: Path):
    legacy = tmp_path / "legacy"
    dest = tmp_path / "data"
    legacy.mkdir()
    (legacy / "models.json").write_text(
        '{"profiles": [{"model": "m.gguf"}]}', encoding="utf-8")
    (legacy / "control-token.txt").write_text("secret\n", encoding="utf-8")
    result = migrate_legacy_data(legacy, dest)
    assert "models.json" in result.copied
    assert (dest / "profiles.json").exists()
    assert json.loads((dest / "profiles.json").read_text())["profiles"] == [{"model": "m.gguf"}]
    assert (dest / "secrets" / "control-token").read_text().strip() == "secret"
    # a second run keeps the existing profiles (idempotent merge) and the token
    second = migrate_legacy_data(legacy, dest)
    assert "models.json" in second.skipped
    assert json.loads((dest / "profiles.json").read_text())["profiles"] == [{"model": "m.gguf"}]
