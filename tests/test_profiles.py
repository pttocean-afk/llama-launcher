import json
from pathlib import Path

from llama_launcher.profiles import (
    EXPORT_VERSION,
    merge_imported,
    read_export,
    export_profiles,
)


def _profile(name="A", model="a.gguf", **kw):
    base = {
        "name": name,
        "model": model,
        "mmproj": "",
        "vision_enabled": False,
        "default_ctx": 131072,
        "reasoning": "off",
        "gpu_split": "16,8",
        "backend": "cuda",
        "jinja": False,
        "extra_args": "-ctk q4_0 -ctv q4_0 --parallel 1",
        "kv_mode": "q4",
        "mtp": False,
        "starred": False,
    }
    base.update(kw)
    return base


def test_export_strips_absolute_paths():
    p = _profile()
    p["model"] = "/absolute/path/to/model.gguf"
    p["mmproj"] = "C:\\models\\mm.gguf"
    p["extra_args"] = "-b 512"
    out = export_profiles([p])
    assert out["version"] == EXPORT_VERSION
    item = out["profiles"][0]
    assert "model" not in item       # absolute path stripped
    assert "mmproj" not in item       # absolute path stripped
    assert item["extra_args"] == "-b 512"


def test_export_keeps_relative_paths():
    p = _profile(model="a.gguf", mmproj="mm.gguf")
    out = export_profiles([p])
    item = out["profiles"][0]
    assert item["model"] == "a.gguf"
    assert item["mmproj"] == "mm.gguf"


def test_export_roundtrip(tmp_path: Path):
    profiles = [
        _profile("A", "a.gguf", starred=True, favorite_order=0),
        _profile("B", "b.gguf", backend="vulkan", reasoning="on"),
    ]
    out = export_profiles(profiles)
    path = tmp_path / "export.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    loaded = read_export(path)
    assert len(loaded) == 2
    by_model = {p["model"]: p for p in loaded}
    assert by_model["a.gguf"]["starred"] is True
    assert by_model["b.gguf"]["backend"] == "vulkan"
    assert by_model["b.gguf"]["reasoning"] == "on"


def test_export_roundtrip_reasoning_effort(tmp_path: Path):
    profiles = [_profile("B", "b.gguf", reasoning="on",
                         reasoning_effort="high")]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(export_profiles(profiles)), encoding="utf-8")
    loaded = read_export(path)
    assert loaded[0]["reasoning_effort"] == "high"


def test_merge_imported_fills_missing_reasoning_effort():
    current = [_profile("A", "a.gguf")]
    imported = [_profile("A", "a.gguf", reasoning_effort="xhigh")]
    merged, added, updated = merge_imported(current, imported)
    assert added == 0
    assert updated == 1
    a = next(p for p in merged if p["model"] == "a.gguf")
    assert a["reasoning_effort"] == "xhigh"


def test_read_export_rejects_bad_file(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"wrong": 1}', encoding="utf-8")
    try:
        read_export(bad)
        assert False, "should raise"
    except ValueError as e:
        assert "not a valid" in str(e).lower() or "version" in str(e).lower()


def test_read_export_rejects_future_version(tmp_path: Path):
    f = tmp_path / "future.json"
    f.write_text(json.dumps({"version": 99, "profiles": []}), encoding="utf-8")
    try:
        read_export(f)
        assert False, "should raise"
    except ValueError as e:
        assert "newer" in str(e).lower()


def test_read_export_drops_path_like_values(tmp_path: Path):
    f = tmp_path / "export.json"
    f.write_text(json.dumps({
        "version": 1,
        "profiles": [
            {"model": "a.gguf", "name": "A", "extra_args": "/usr/bin/llama"},
        ],
    }), encoding="utf-8")
    loaded = read_export(f)
    assert loaded[0]["model"] == "a.gguf"
    assert "extra_args" not in loaded[0]


def test_merge_imported_adds_new():
    current = [_profile("A", "a.gguf")]
    imported = [_profile("B", "b.gguf")]
    merged, added, updated = merge_imported(current, imported)
    assert added == 1
    assert updated == 0
    assert len(merged) == 2
    models = {p["model"] for p in merged}
    assert models == {"a.gguf", "b.gguf"}


def test_merge_imported_keeps_existing_settings():
    current = [_profile("A", "a.gguf", default_ctx=262144, starred=True, favorite_order=0)]
    imported = [_profile("A", "a.gguf", default_ctx=131072)]
    merged, added, updated = merge_imported(current, imported)
    assert added == 0
    assert updated == 0
    a = next(p for p in merged if p["model"] == "a.gguf")
    assert a["default_ctx"] == 262144   # existing wins


def test_merge_imported_fills_missing_fields():
    # current profile lacks kv_mode and mtp entirely
    current = [{"name": "A", "model": "a.gguf", "default_ctx": 131072}]
    imported = [{"name": "A", "model": "a.gguf", "kv_mode": "f16", "mtp": True}]
    merged, added, updated = merge_imported(current, imported)
    assert added == 0
    assert updated == 1
    a = next(p for p in merged if p["model"] == "a.gguf")
    assert a["kv_mode"] == "f16"
    assert a["mtp"] is True


def test_merge_imported_preserves_starred_order():
    current = [
        _profile("S1", "s1.gguf", starred=True, favorite_order=0),
        _profile("S2", "s2.gguf", starred=True, favorite_order=1),
        _profile("U1", "u1.gguf"),
    ]
    imported = [_profile("S3", "s3.gguf", starred=True, favorite_order=5)]
    merged, added, updated = merge_imported(current, imported)
    assert added == 1
    starred = [p for p in merged if p.get("starred")]
    assert [p["model"] for p in starred] == ["s1.gguf", "s2.gguf", "s3.gguf"]
    assert starred[2]["favorite_order"] == 2   # re-numbered


def test_merge_imported_ignores_empty_model():
    current = [_profile("A", "a.gguf")]
    imported = [{"name": "no-model", "model": ""}]
    merged, added, updated = merge_imported(current, imported)
    assert added == 0
    assert len(merged) == 1
