from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MigrationResult:
    copied: tuple[str, ...]
    skipped: tuple[str, ...]


def migrate_legacy_data(legacy_dir: Path, destination: Path, include_token: bool = True) -> MigrationResult:
    destination.mkdir(parents=True, exist_ok=True)
    mapping = {
        "models.json": destination / "profiles.json",
        "settings.json": destination / "settings.json",
    }
    if include_token:
        mapping["control-token.txt"] = destination / "secrets" / "control-token"
    copied, skipped = [], []
    for source_name, target in mapping.items():
        source = legacy_dir / source_name
        if not source.exists() or target.exists():
            skipped.append(source_name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(source_name)
    return MigrationResult(tuple(copied), tuple(skipped))
