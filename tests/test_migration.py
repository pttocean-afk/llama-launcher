from pathlib import Path

from llama_launcher.migration import migrate_legacy_data


def test_migration_renames_models_and_preserves_existing_destination(tmp_path: Path):
    legacy = tmp_path / "legacy"
    dest = tmp_path / "data"
    legacy.mkdir()
    (legacy / "models.json").write_text('{"profiles": []}', encoding="utf-8")
    (legacy / "control-token.txt").write_text("secret\n", encoding="utf-8")
    result = migrate_legacy_data(legacy, dest)
    assert "models.json" in result.copied
    assert (dest / "profiles.json").exists()
    assert (dest / "secrets" / "control-token").read_text().strip() == "secret"
    second = migrate_legacy_data(legacy, dest)
    assert "models.json" in second.skipped
