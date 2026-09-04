"""Portable profile export/import (no local absolute paths)."""
from __future__ import annotations

import json
from pathlib import Path

# Fields that travel with a portable profile. Everything else (e.g. paths,
# host-specific settings) is stripped on export.
PORTABLE_FIELDS = (
    "name",
    "model",          # relative filename inside models/
    "scheme",         # 啟動方案（同一模型可有多個，如 預設 / code / chat）
    "mmproj",         # relative filename, may be empty
    "vision_enabled",
    "default_ctx",
    "reasoning",      # "on" / "off" / "auto"
    "reasoning_effort",  # "default" / "minimal" / "low" / "medium" / "high" / "xhigh" / "max"
    "reasoning_format",  # "auto" / "none" / "deepseek" / "deepseek-legacy"
    "reasoning_preserve",  # "default" / "on" / "off"
    "gpu_split",
    "backend",        # "cuda" / "vulkan"
    "jinja",
    "extra_args",
    "kv_mode",        # f16 / q8 / q5 / q4 / iq4_nl / custom
    "mtp",
    "spec_draft_n_max",
    "flash_attn",     # auto / on / off
    "kv_unified",
    "fit",            # on / off
    "threads",
    "threads_batch",
    "ctx_checkpoints",
    "parallel",
    "ngl",
    "temp",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repeat_penalty",
    "raw_args",       # 完整參數模式（非空時覆蓋所有 GUI 參數）
    "starred",
    "favorite_order",
)

EXPORT_VERSION = 1


def export_profiles(profiles: list[dict]) -> dict:
    """Return a JSON-serialisable dict for the given profiles."""
    items = []
    for p in profiles:
        item = {k: p[k] for k in PORTABLE_FIELDS if k in p}
        # Strip any value that looks like an absolute path (safety net).
        for key, val in list(item.items()):
            if isinstance(val, str) and (
                val.startswith("/") or val.startswith("\\")
                or (len(val) > 1 and val[1] == ":")
            ):
                item.pop(key, None)
        items.append(item)
    return {"version": EXPORT_VERSION, "profiles": items}


def read_export(path: Path) -> list[dict]:
    """Load a portable profile export file and return the profile list.

    Raises ValueError on malformed input."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "profiles" not in data:
        raise ValueError("Not a valid LlamaLauncher profile export")
    version = data.get("version", 0)
    if version > EXPORT_VERSION:
        raise ValueError(
            f"Export version {version} is newer than supported {EXPORT_VERSION}")
    profiles = data["profiles"]
    if not isinstance(profiles, list):
        raise ValueError("profiles must be a list")
    # Normalise: keep only known fields, drop anything path-like.
    clean = []
    for p in profiles:
        if not isinstance(p, dict):
            continue
        item = {k: p[k] for k in PORTABLE_FIELDS if k in p}
        for key, val in list(item.items()):
            if isinstance(val, str) and (
                val.startswith("/") or val.startswith("\\")
                or (len(val) > 1 and val[1] == ":")
            ):
                item.pop(key, None)
        if item.get("model"):
            clean.append(item)
    return clean


def _profile_key(profile: dict) -> tuple[str, str]:
    scheme = str(profile.get("scheme") or "").strip() or "預設"
    return (str(profile.get("model") or ""), scheme)


def merge_imported(current_profiles: list[dict], imported: list[dict]) -> tuple[list[dict], int, int]:
    """Merge imported profiles into the current list.

    Returns (merged_list, added, updated). Existing profiles (matched by
    model + scheme) keep their settings unless the imported version has
    fields the current one lacks. Starred order is preserved for existing
    entries; new entries get appended after starred ones."""
    by_key = {_profile_key(p): dict(p) for p in current_profiles if p.get("model")}
    added = updated = 0
    for item in imported:
        if not item.get("model"):
            continue
        key = _profile_key(item)
        if key in by_key:
            # Fill in missing fields from the import (e.g. kv_mode added later).
            cur = by_key[key]
            changed = False
            for k, v in item.items():
                if k not in cur and v not in (None, "", False, 0):
                    cur[k] = v
                    changed = True
            if changed:
                updated += 1
        else:
            item = dict(item)
            item.pop("configured", None)
            by_key[key] = item
            added += 1
    # Re-order: starred by favorite_order, then unstarred by model/scheme.
    merged = list(by_key.values())
    starred = [p for p in merged if p.get("starred")]
    starred.sort(key=lambda p: (
        int(p.get("favorite_order", 1 << 30)), str(p.get("name", "")).lower()))
    for i, p in enumerate(starred):
        p["favorite_order"] = i
    rest = [p for p in merged if not p.get("starred")]
    rest.sort(key=lambda p: (str(p.get("model", "")).lower(),
                             str(p.get("scheme") or "預設")))
    return starred + rest, added, updated
