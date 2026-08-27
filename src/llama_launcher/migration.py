from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

LEGACY_MARKERS = ("models.json", "control-token.txt", "settings.json")


@dataclass(frozen=True)
class MigrationResult:
    copied: tuple[str, ...]
    skipped: tuple[str, ...]


@dataclass(frozen=True)
class MigrationPlan:
    """What a legacy data folder contains and what would happen to it."""

    legacy_dir: Path
    available: tuple[str, ...]
    will_copy: tuple[str, ...]
    will_skip: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.available


def detect_legacy_dir(legacy_dir: Path) -> bool:
    """A folder qualifies as legacy launcher data if it holds at least one
    known file (models.json / settings.json / control-token.txt)."""
    legacy_dir = Path(legacy_dir)
    return any((legacy_dir / name).is_file() for name in LEGACY_MARKERS)


def plan_migration(legacy_dir: Path, destination: Path, include_token: bool = True) -> MigrationPlan:
    """Preview the migration without touching the filesystem."""
    legacy_dir = Path(legacy_dir)
    destination = Path(destination)
    available = tuple(name for name in LEGACY_MARKERS if (legacy_dir / name).is_file())
    targets = {
        "models.json": destination / "profiles.json",
        "settings.json": destination / "settings.json",
    }
    if include_token:
        targets["control-token.txt"] = destination / "secrets" / "control-token"
    will_copy, will_skip = [], []
    for name in LEGACY_MARKERS:
        if name not in available:
            continue
        if name == "control-token.txt" and not include_token:
            will_skip.append(name)
        elif targets[name].exists():
            will_skip.append(name)
        else:
            will_copy.append(name)
    return MigrationPlan(
        legacy_dir=legacy_dir,
        available=available,
        will_copy=tuple(will_copy),
        will_skip=tuple(will_skip),
    )


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
        if not source.exists():
            skipped.append(source_name)
            continue
        if source_name == "models.json":
            # Fresh destination: write the merged config (nothing to merge into).
            # Existing config with profiles: merge happens in place via
            # merge_legacy_into_config (called by the GUI flow), so report skip.
            existing_cfg = _load_json(target) if target.exists() else {}
            if existing_cfg.get("profiles"):
                skipped.append(source_name)
            else:
                _merge_profiles(_load_json(source), existing_cfg)
                save_cfg(target, existing_cfg)
                copied.append(source_name)
            continue
        if target.exists():
            skipped.append(source_name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(source_name)
    return MigrationResult(tuple(copied), tuple(skipped))


def save_cfg(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def merge_legacy_into_config(legacy_dir: Path, config_path: Path) -> int:
    """Merge the legacy models.json profiles into the config file in place
    (new-app settings win). Returns the number of newly added profiles.
    A missing legacy file is a no-op returning 0; the destination config is
    always rewritten with the (idempotent) merge result."""
    legacy_cfg = _load_json(Path(legacy_dir) / "models.json")
    if not legacy_cfg.get("profiles"):
        return 0
    config = _load_json(Path(config_path))
    before = {p.get("model") for p in config.get("profiles", []) if p.get("model")}
    _merge_profiles(legacy_cfg, config)
    added = {p.get("model") for p in config.get("profiles", []) if p.get("model")} - before
    save_cfg(Path(config_path), config)
    return len(added)


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _merge_profiles(legacy_cfg: dict, current_cfg: dict) -> list[dict]:
    """Merge legacy profiles into the current config without clobbering
    models the user already configured in the new app.

    Returns the updated profile list; ``current_cfg`` is mutated in place.
    """
    current = {p.get("model"): dict(p) for p in current_cfg.get("profiles", []) if p.get("model")}
    for profile in legacy_cfg.get("profiles", []):
        model = profile.get("model")
        if not model:
            continue
        if model in current:
            continue  # new-app settings win
        profile = dict(profile)
        profile.pop("configured", None)
        current[model] = profile
    ordered = []
    starred = [p for p in current.values() if p.get("starred")]
    starred.sort(key=lambda p: (int(p.get("favorite_order", 1 << 30)), str(p.get("name", "")).lower()))
    rest = [p for p in current.values() if not p.get("starred")]
    rest.sort(key=lambda p: str(p.get("name", "")).lower())
    for order, profile in enumerate(starred):
        profile["favorite_order"] = order
        ordered.append(profile)
    ordered.extend(rest)
    current_cfg["profiles"] = ordered
    return ordered
