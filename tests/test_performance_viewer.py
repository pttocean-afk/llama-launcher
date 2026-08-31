"""Display-free tests for llama_launcher.performance_viewer (TDD task 6).

The module must import headless (no $DISPLAY): all helpers and the scan
model are tested without constructing any Tk widgets.
"""
from __future__ import annotations

import threading
from pathlib import Path

from llama_launcher import performance_viewer as pv
from llama_launcher.log_analysis import (
    DEFAULT_MIN_GENERATED,
    parse_log_file,
)

FIXTURES = Path(__file__).parent / "fixtures" / "logs"
BEE = FIXTURES / "bee-qwen38-small.log"
B10621 = FIXTURES / "b10621-cuda-small.log"
VULKAN = FIXTURES / "vulkan-small.log"


def test_module_imports_without_display():
    # reaching this line means the import succeeded headless
    assert pv.PerformanceViewer is not None
    assert pv.PerformanceModel is not None


def test_log_cache_key_uses_stat():
    st = BEE.stat()
    assert pv.log_cache_key(BEE) == (str(BEE.resolve()), st.st_size,
                                     st.st_mtime_ns)


def test_log_cache_reuses_unchanged_file():
    cache = pv.LogCache()
    a = cache.get_or_parse(BEE)
    b = cache.get_or_parse(BEE)
    assert a is b
    assert a.samples  # sanity: fixture actually parsed


def test_log_cache_invalidates_grown_file(tmp_path):
    log = tmp_path / "growing.log"
    body = (
        "# 2026-08-30T12:00:00.000000  Test\n"
        "# C:\\bin\\llama-server.exe -m C:\\models\\m.gguf -c 4096\n"
        + "=" * 80 + "\n"
        "0.00.100.000 I slot launch_slot_: id 0 | task 0 | "
        "processing task, is_child = 0\n"
        "0.00.100.100 I slot print_timing: id 0 | task 0 | "
        "prompt eval time = 100.00 ms / 10 tokens (10.00 tokens per second)\n"
        "0.00.100.200 I slot print_timing: id 0 | task 0 | "
        "        eval time = 500.00 ms / 40 tokens (8.00 tokens per second)\n"
        "0.00.100.300 I slot      release: id 0 | task 0 | "
        "stop processing: n_tokens = 50, truncated = 0\n"
    )
    log.write_text(body, encoding="utf-8")
    cache = pv.LogCache()
    first = cache.get_or_parse(log)
    assert len(first.samples) == 1
    # the active log grows: (size, mtime_ns) change -> reparse only this file
    with log.open("a", encoding="utf-8") as f:
        f.write("0.00.200.000 I slot      release: id 0 | task 1 | "
                "stop processing: n_tokens = 90, truncated = 0\n")
    second = cache.get_or_parse(log)
    assert second is not first
    assert len(second.samples) == 2


def test_legacy_logs_dir(tmp_path):
    llama = tmp_path / "llama"
    (llama / "launcher-app" / "logs").mkdir(parents=True)
    assert pv.legacy_logs_dir(llama) == llama / "launcher-app" / "logs"
    assert pv.legacy_logs_dir(tmp_path / "nowhere") is None
    assert pv.legacy_logs_dir(None) is None


def test_select_scan_paths_dedupes(tmp_path):
    d1 = tmp_path / "d1"
    d1.mkdir()
    d2 = tmp_path / "d2"
    d2.mkdir()
    (d1 / "a.log").write_text("x", encoding="utf-8")
    (d1 / "b.log").write_text("x", encoding="utf-8")
    (d1 / "notalog.txt").write_text("x", encoding="utf-8")
    (d2 / "c.log").write_text("x", encoding="utf-8")
    extra = d1 / "extra.log"
    extra.write_text("x", encoding="utf-8")
    paths = pv.select_scan_paths(d1, [d2],
                                 [extra, d1 / "missing.log", d1 / "a.log"])
    names = {p.name for p in paths}
    assert names == {"a.log", "b.log", "c.log", "extra.log"}
    # deterministic: sorted by resolved full path, deduplicated
    resolved = [str(p.resolve()) for p in paths]
    assert resolved == sorted(resolved)
    assert len(resolved) == len(set(resolved))


def test_filter_options_lists_distinct_values():
    pool = [parse_log_file(p) for p in (BEE, B10621, VULKAN)]
    opts = pv.filter_options(pool)
    assert set(opts["backend"]) == {"cuda", "vulkan"}
    assert set(opts["runtime"]) == {"BeeLlama v0.4.4", "b10621", "b10509"}
    assert set(opts["kv"]) == {"q4_0/q4_0", "unknown/unknown"}
    assert opts["model"] == ("Qwen3.8-27B-IQ4_NL",)
    assert "196608" in opts["context"] and "262144" in opts["context"]
    assert set(opts["vision"]) == {"yes", "no"}


def test_build_filters_ignores_all_and_maps_values():
    f = pv.build_filters({"model": "M", "backend": "cuda",
                          "context": "196608", "vision": "yes",
                          "min_generated": "50"})
    assert f.model == "M"
    assert f.backend == "cuda"
    assert f.configured_context == 196608
    assert f.vision_loaded is True
    assert f.min_generated == 50
    g = pv.build_filters({"model": "All", "vision": "all",
                          "context": "unknown"})
    assert g.model is None
    assert g.vision_loaded is None
    assert g.configured_context is None
    assert g.min_generated == DEFAULT_MIN_GENERATED


def test_format_series_label():
    assert pv.format_series_label("cuda", 2, 5) == \
        "cuda (2 logs, 5 samples)"


def test_dimensions_cover_plan():
    assert set(pv.DIMENSIONS.values()) == {
        "runtime", "backend", "kv", "reasoning", "reasoning_effort",
        "context", "vision", "run"}


def test_performance_model_scan_collects_logs():
    model = pv.PerformanceModel()
    done = threading.Event()
    results = []
    model.request_scan([BEE, B10621],
                       on_done=lambda r: (results.append(r), done.set()))
    assert done.wait(10)
    assert len(model.parsed) == 2
    assert model.errors == []
    assert len(results) == 1 and results[0][1] == []


def test_performance_model_scan_reports_unreadable_file(tmp_path):
    missing = tmp_path / "nope.log"
    model = pv.PerformanceModel()
    done = threading.Event()
    model.request_scan([BEE, missing],
                       on_done=lambda r: done.set())
    assert done.wait(10)
    assert len(model.parsed) == 1
    assert len(model.errors) == 1
    assert str(missing) in model.errors[0][0]


def test_performance_model_current_generation_applies():
    model = pv.PerformanceModel()
    fired = []
    # generation 0 is the current one: worker result applies
    model._worker(0, [BEE, VULKAN], lambda r: fired.append(r))
    assert len(fired) == 1
    assert len(model.parsed) == 2


def test_performance_model_stale_scan_cannot_overwrite():
    model = pv.PerformanceModel()
    fired = []
    # a worker claiming an old generation must be dropped entirely
    model._worker(99, [BEE], lambda r: fired.append(r))
    assert fired == []
    assert model.parsed == []
    assert model.errors == []
