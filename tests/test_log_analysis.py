"""Parser contract tests for llama_launcher.log_analysis (TDD task 1).

Fixtures in tests/fixtures/logs are minimized versions of real Launcher logs
(BeeLlama v0.4.4 CUDA, b10621 CUDA native, b10509 Vulkan) with CRLF endings.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from llama_launcher.log_analysis import (
    Filters,
    ParsedLog,
    RunMetadata,
    ScanError,
    TimingSample,
    aggregate_runs,
    bucket_index,
    fairness_warnings,
    parse_log_file,
    parse_log_text,
    percentile,
    scan_log_paths,
)

FIXTURES = Path(__file__).parent / "fixtures" / "logs"


def _read_fixture(name: str) -> str:
    """Read fixture preserving CRLF line endings (no universal-newline
    translation; bytes -> str decode never rewrites \\r\\n)."""
    return (FIXTURES / name).read_bytes().decode("utf-8")


def _parse_fixture(name: str):
    return parse_log_text(_read_fixture(name), source_path=str(FIXTURES / name))


# ---------------------------------------------------------------------------
# metadata: owner-requested dimensions + runtime
# ---------------------------------------------------------------------------

def test_bee_metadata_all_dimensions():
    meta = _parse_fixture("bee-qwen38-small.log").metadata
    # runtime/build dimension
    assert meta.runtime_label == "BeeLlama v0.4.4"
    assert meta.runtime_version == "0.4.4"
    assert meta.executable_path.endswith(
        r"E:\llama-cpp\benchmark-runtimes\beellama-v0.4.4-cuda-13.3\llama-server.exe")
    # backend dimension
    assert meta.backend == "cuda"
    # KV dimension
    assert meta.kv_k == "q4_0"
    assert meta.kv_v == "q4_0"
    # reasoning dimension
    assert meta.reasoning == "on"
    # configured max context dimension
    assert meta.configured_context == 196608
    # vision loaded dimension
    assert meta.vision_loaded is True
    assert meta.mmproj_path.endswith("mmproj-Qwen3.8-27B-Uncensored-vision-f16.gguf")
    # other metadata
    assert meta.model_name == "Qwen3.8-27B-IQ4_NL"
    assert meta.model_path.endswith("Qwen3.8-27B-IQ4_NL.gguf")
    assert meta.profile_name == "Qwen3.8-27B-IQ4_NL"
    assert meta.started_at == "2026-08-30T18:23:31.114749"
    assert meta.gpu_split == "16,8"
    assert meta.parallel == 1
    assert meta.jinja is False
    flags = dict(zip(meta.extra_flags[::2], meta.extra_flags[1::2]))
    assert flags.get("-cram") == "24000"
    assert flags.get("-ngl") == "999"
    assert flags.get("-sm") == "layer"


def test_b10621_metadata():
    meta = _parse_fixture("b10621-cuda-small.log").metadata
    assert meta.runtime_label == "b10621"
    assert meta.runtime_version == "10621"
    assert meta.backend == "cuda"
    assert meta.kv_k == "q4_0"
    assert meta.kv_v == "q4_0"
    assert meta.reasoning == "on"
    assert meta.configured_context == 196608
    assert meta.vision_loaded is True
    assert meta.gpu_split == "16,8"


def test_vulkan_metadata():
    meta = _parse_fixture("vulkan-small.log").metadata
    # --device Vulkan flag and vulkan exe path both indicate vulkan
    assert meta.backend == "vulkan"
    assert meta.runtime_label == "b10509"
    assert meta.runtime_version == "10509"
    assert meta.reasoning == "off"
    assert meta.configured_context == 262144
    assert meta.vision_loaded is False
    assert meta.mmproj_path is None
    # no -ctk/-ctv flags: parser must not guess a KV type
    assert meta.kv_k == "unknown"
    assert meta.kv_v == "unknown"
    assert meta.gpu_split == "16,8"


def test_header_is_authority_for_command_config():
    meta = _parse_fixture("bee-qwen38-small.log").metadata
    # header says 196608; body n_ctx_slot also 196608 -> no conflict warning
    assert not any("context" in w.lower() for w in meta.warnings)


# ---------------------------------------------------------------------------
# task-correlated timing extraction
# ---------------------------------------------------------------------------

def test_bee_samples_task_correlated():
    parsed = _parse_fixture("bee-qwen38-small.log")
    samples = parsed.samples
    assert len(samples) == 3
    probe, cached, incomplete = samples

    # task 0: the one-token firewall probe
    assert probe.task_id == 0
    assert probe.slot_id == 0
    assert probe.completed is True
    assert probe.prompt_tokens == 54
    assert probe.prompt_ms == pytest.approx(890.32)
    assert probe.prefill_tps == pytest.approx(60.65)
    assert probe.generated_tokens == 1
    assert probe.decode_ms == 0.0
    assert probe.decode_tps == 0.0
    assert probe.used_context == 54
    assert probe.used_context_source == "stop_processing"

    # task 4: completed request with RAM prompt-cache hit
    assert cached.task_id == 4
    assert cached.completed is True
    assert cached.prompt_tokens == 2366
    assert cached.prefill_tps == pytest.approx(1195.50)
    assert cached.generated_tokens == 185
    assert cached.decode_ms == pytest.approx(4816.12)
    assert cached.decode_tps == pytest.approx(38.20)
    assert cached.used_context == 2550
    assert cached.used_context_source == "stop_processing"
    assert cached.cache_source == "ram"
    assert cached.cached_tokens == 2366
    assert cached.reprocessed_tokens == 0

    # task 9: incomplete (no eval/total/stop lines) -> checkpoint only
    assert incomplete.task_id == 9
    assert incomplete.completed is False
    assert incomplete.prompt_tokens == 1000
    assert incomplete.prefill_tps == pytest.approx(200.00)
    assert incomplete.generated_tokens == 60  # latest n_gen checkpoint
    # rolling tg on n_gen lines is never the official decode result
    assert incomplete.decode_ms is None
    assert incomplete.decode_tps is None
    assert incomplete.used_context == 1060  # 1000 prompt + 60 generated
    assert incomplete.used_context_source == "checkpoint"


def test_b10621_samples():
    samples = _parse_fixture("b10621-cuda-small.log").samples
    assert len(samples) == 2
    t0, t479 = samples
    assert t0.completed is True
    assert t0.prompt_tokens == 1003
    assert t0.prompt_ms == pytest.approx(1460.52)
    assert t0.prefill_tps == pytest.approx(686.74)
    assert t0.generated_tokens == 475
    assert t0.decode_ms == pytest.approx(13262.73)
    assert t0.decode_tps == pytest.approx(35.74)
    assert t0.used_context == 1477
    assert t0.error_kind is None
    assert t479.completed is True
    assert t479.used_context == 5294
    assert t479.generated_tokens == 678
    # line offsets are 1-based and traceable
    assert 1 <= t0.line_start < t0.line_end
    assert t479.line_start > t0.line_end


def test_vulkan_samples():
    samples = _parse_fixture("vulkan-small.log").samples
    assert [s.task_id for s in samples] == [0, 7]
    assert all(s.completed for s in samples)
    assert samples[0].used_context == 699
    assert samples[1].used_context == 1239


def test_incomplete_task_counted_separately():
    parsed = _parse_fixture("bee-qwen38-small.log")
    completed = [s for s in parsed.samples if s.completed]
    incomplete = [s for s in parsed.samples if not s.completed]
    assert len(completed) == 2
    assert len(incomplete) == 1


def test_one_token_probe_retained_raw():
    """The probe stays in the raw sample list; aggregation filters it later."""
    parsed = _parse_fixture("bee-qwen38-small.log")
    probe = parsed.samples[0]
    assert probe.generated_tokens == 1
    assert probe in parsed.samples  # retained, not dropped at parse time


# ---------------------------------------------------------------------------
# malformed / incomplete handling
# ---------------------------------------------------------------------------

def test_task_without_timing_lines_dropped_with_warning():
    text = (
        "# 2026-08-30T12:00:00.000000  Test\n"
        "# C:\\bin\\llama-server.exe -m C:\\models\\m.gguf -c 8192\n"
        + "=" * 80 + "\n"
        "0.01.000.000 I slot launch_slot_: id  0 | task 5 | processing task, is_child = 0\n"
        "0.02.000.000 I slot print_timing: id  0 | task 5 | prompt eval time = 123 ms /\n"
    )
    parsed = parse_log_text(text, source_path="synthetic")
    assert parsed.samples == ()
    assert any("task 5" in w for w in parsed.warnings)


def test_oom_associated_to_task():
    text = (
        "# 2026-08-30T12:00:00.000000  Test\n"
        "# C:\\bin\\llama-server.exe -m C:\\models\\m.gguf -c 8192\n"
        + "=" * 80 + "\n"
        "0.01.000.000 I slot launch_slot_: id  0 | task 5 | processing task, is_child = 0\n"
        "0.02.000.000 E ggml_backend_cuda_buffer_type_alloc_buffer: "
        "cudaMalloc failed: out of memory\n"
    )
    parsed = parse_log_text(text, source_path="synthetic")
    assert len(parsed.samples) == 1
    sample = parsed.samples[0]
    assert sample.task_id == 5
    assert sample.error_kind == "oom"
    assert sample.completed is False
    assert sample.used_context is None


def test_unassociated_oom_is_warning_not_error_kind():
    """OOM before any task (init-time retry) must not stick to a task."""
    parsed = _parse_fixture("bee-qwen38-small.log")
    assert all(s.error_kind is None for s in parsed.samples)
    assert any("out of memory" in w or "oom" in w.lower() for w in parsed.warnings)


def test_scan_one_bad_file_does_not_abort():
    good = FIXTURES / "b10621-cuda-small.log"
    missing = FIXTURES / "does-not-exist.log"
    parsed, errors = scan_log_paths([good, missing])
    assert len(parsed) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ScanError)
    assert errors[0].path == str(missing)


def test_binary_garbage_file_parses_without_crash():
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as fh:
        fh.write(b"\x00\xff\xfe garbage \x80\x81\x82\n" * 5)
        path = Path(fh.name)
    try:
        parsed = parse_log_file(path)
        assert parsed.samples == ()
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# encoding / line-ending tolerance
# ---------------------------------------------------------------------------

def test_crlf_and_lf_parse_identically():
    raw = _read_fixture("bee-qwen38-small.log")
    assert "\r\n" in raw  # fixture really is CRLF
    crlf = parse_log_text(raw, source_path="bee")
    lf = parse_log_text(raw.replace("\r\n", "\n"), source_path="bee")
    assert crlf.metadata == lf.metadata
    assert crlf.samples == lf.samples


def test_replacement_char_tolerance():
    raw = (FIXTURES / "bee-qwen38-small.log").read_bytes()
    # corrupt one byte inside an init line (after the header)
    marker = b"common_params_print_info"
    idx = raw.index(marker) + len(marker)
    corrupted = raw[:idx] + b"\xff" + raw[idx:]
    parsed = parse_log_text(corrupted.decode("utf-8", errors="replace"),
                            source_path="corrupted")
    assert len(parsed.samples) == 3
    assert parsed.metadata.model_name == "Qwen3.8-27B-IQ4_NL"


# ---------------------------------------------------------------------------
# actual image use boundary
# ---------------------------------------------------------------------------

def test_actual_image_unknown_without_task_evidence():
    # vision loaded, but no per-task image evidence -> unknown, not false
    for s in _parse_fixture("bee-qwen38-small.log").samples:
        assert s.actual_image_used == "unknown"
    for s in _parse_fixture("b10621-cuda-small.log").samples:
        assert s.actual_image_used == "unknown"


def test_actual_image_no_when_vision_not_loaded():
    for s in _parse_fixture("vulkan-small.log").samples:
        assert s.actual_image_used == "no"


def test_actual_image_yes_with_task_evidence():
    text = (
        "# 2026-08-30T12:00:00.000000  Test\n"
        "# C:\\bin\\llama-server.exe "
        "--mmproj C:\\models\\mm.gguf -m C:\\models\\m.gguf -c 8192\n"
        + "=" * 80 + "\n"
        "0.01.000.000 I slot launch_slot_: id  0 | task 12 | processing task, is_child = 0\n"
        "0.01.500.000 I slot add_image: id  0 | task 12 | image processed: "
        "512x512, 690 image tokens\n"
        "0.02.000.000 I slot print_timing: id  0 | task 12 | "
        "prompt eval time =     100.00 ms /    10 tokens (   10.00 ms per token,    10.00 tokens per second)\n"
        "0.03.000.000 I slot print_timing: id  0 | task 12 | "
        "eval time =     200.00 ms /    20 tokens (   10.00 ms per token,    20.00 tokens per second)\n"
        "0.03.500.000 I slot print_timing: id  0 | task 12 | "
        "total time =     300.00 ms /    30 tokens\n"
        "0.04.000.000 I slot      release: id  0 | task 12 | stop processing: n_tokens = 29, truncated = 0\n"
    )
    parsed = parse_log_text(text, source_path="synthetic")
    assert len(parsed.samples) == 1
    assert parsed.samples[0].actual_image_used == "yes"
    assert parsed.samples[0].used_context == 29


# ---------------------------------------------------------------------------
# command-line / header parsing
# ---------------------------------------------------------------------------

def test_quoted_paths_with_spaces():
    text = (
        "# 2026-08-30T12:00:00.000000  Test\n"
        '# "C:\\Program Files\\llama\\llama-server.exe" '
        '-m "C:\\my models\\My Model.gguf" -c 4096\n'
        + "=" * 80 + "\n"
        "0.01.000.000 I srv    load_model: listening on http://0.0.0.0:8080\n"
    )
    meta = parse_log_text(text, source_path="synthetic").metadata
    assert meta.executable_path == r"C:\Program Files\llama\llama-server.exe"
    assert meta.model_path == r"C:\my models\My Model.gguf"
    assert meta.model_name == "My Model"
    assert meta.configured_context == 4096


def test_no_header_infers_from_log_content():
    raw = _read_fixture("vulkan-small.log")
    # strip the two header lines + separator
    lines = raw.split("\r\n")
    body = "\r\n".join(lines[3:]).lstrip("\r\n")
    parsed = parse_log_text(body, source_path="raw")
    meta = parsed.metadata
    assert meta.started_at is None
    assert meta.profile_name is None
    assert meta.model_name == "Qwen3.8-27B-IQ4_NL"
    assert meta.model_path.endswith("Qwen3.8-27B-IQ4_NL.gguf")
    # n_ctx_slot from body is the only context evidence
    assert meta.configured_context == 262144
    assert any("header" in w.lower() for w in parsed.warnings)
    # without a command line, backend falls back to content evidence
    assert meta.backend in ("vulkan", "unknown")
    assert meta.vision_loaded is False


def test_backend_never_guessed_from_model_name():
    text = (
        "# 2026-08-30T12:00:00.000000  Test\n"
        "# C:\\bin\\llama-server.exe -m C:\\models\\my-cuda-model.gguf -c 4096\n"
        + "=" * 80 + "\n"
        "0.01.000.000 I srv  llama_server: model loaded\n"
    )
    meta = parse_log_text(text, source_path="synthetic").metadata
    assert meta.backend == "unknown"
    assert meta.runtime_label == "unknown"


def test_legacy_two_line_header():
    """Legacy launcher header has no preflight line (vulkan fixture)."""
    parsed = _parse_fixture("vulkan-small.log")
    assert parsed.metadata.profile_name == "Qwen3.8-27B-IQ4_NL"
    assert parsed.metadata.started_at == "2026-08-24T20:11:02.000000"


def test_reasoning_omitted_is_unknown_not_off():
    text = (
        "# 2026-08-30T12:00:00.000000  Test\n"
        "# C:\\bin\\llama-server.exe -m C:\\models\\m.gguf -c 4096\n"
        + "=" * 80 + "\n"
        "0.01.000.000 I srv  llama_server: model loaded\n"
    )
    meta = parse_log_text(text, source_path="synthetic").metadata
    assert meta.reasoning == "unknown"


def _header_log(argv_extra: str) -> str:
    return (
        "# 2026-08-30T12:00:00.000000  Test\n"
        f"# C:\\bin\\llama-server.exe -m C:\\models\\m.gguf -c 4096 "
        f"{argv_extra}\n"
        + "=" * 80 + "\n"
        "0.01.000.000 I srv  llama_server: model loaded\n"
    )


def test_reasoning_effort_parsed_from_header():
    meta = parse_log_text(
        _header_log("--reasoning on --reasoning-effort high"),
        source_path="synthetic").metadata
    assert meta.reasoning == "on"
    assert meta.reasoning_effort == "high"


def test_reasoning_effort_defaults_when_flag_absent():
    meta = parse_log_text(
        _header_log("--reasoning on"), source_path="synthetic").metadata
    assert meta.reasoning_effort == "default"


def test_reasoning_effort_value_is_case_normalized():
    meta = parse_log_text(
        _header_log("--reasoning on --reasoning-effort HIGH"),
        source_path="synthetic").metadata
    assert meta.reasoning_effort == "high"


def test_reasoning_effort_invalid_value_warns_and_marks_unknown():
    parsed = parse_log_text(
        _header_log("--reasoning on --reasoning-effort ultra"),
        source_path="synthetic")
    assert parsed.metadata.reasoning_effort == "unknown"
    assert any("reasoning-effort" in w for w in parsed.metadata.warnings)


def test_reasoning_effort_all_known_levels_accepted():
    from llama_launcher.log_analysis import REASONING_EFFORT_LEVELS
    for level in REASONING_EFFORT_LEVELS:
        meta = parse_log_text(
            _header_log(f"--reasoning-effort {level}"),
            source_path="synthetic").metadata
        assert meta.reasoning_effort == level


# ---------------------------------------------------------------------------
# data model invariants
# ---------------------------------------------------------------------------

def test_records_are_immutable():
    parsed = _parse_fixture("bee-qwen38-small.log")
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.metadata.backend = "vulkan"
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.samples[0].decode_tps = 99.0


# ---------------------------------------------------------------------------
# aggregation (task 3)
# ---------------------------------------------------------------------------

def _meta(**kw):
    """Build a RunMetadata with sensible defaults; kw overrides fields."""
    base = dict(
        source_path=r"C:\logs\run.log",
        started_at="2026-08-30T10:00:00",
        profile_name="Test",
        model_path=r"C:\models\model.gguf",
        model_name="model",
        executable_path=r"C:\bin\llama-server.exe",
        runtime_label="legacy",
        runtime_version="unknown",
        backend="cuda",
        kv_k="q4_0",
        kv_v="q4_0",
        reasoning="on",
        reasoning_effort="default",
        configured_context=196608,
        vision_loaded=True,
        mmproj_path=r"C:\models\mm.gguf",
        gpu_split="16,8",
        batch=None,
        ubatch=None,
        parallel=1,
        jinja=False,
        extra_flags=(),
    )
    base.update(kw)
    return RunMetadata(**base)


def _sample(used_context=2550, generated_tokens=185, completed=True,
            decode_tps=38.2, prefill_tps=1195.5, error_kind=None, task_id=4):
    """Build a TimingSample with sensible defaults; kw overrides fields."""
    return TimingSample(
        task_id=task_id,
        slot_id=0,
        used_context=used_context,
        used_context_source="stop_processing",
        prompt_tokens=2366,
        prompt_ms=1979.0,
        prefill_tps=prefill_tps,
        generated_tokens=generated_tokens,
        decode_ms=4816.12,
        decode_tps=decode_tps,
        cache_source=None,
        cached_tokens=None,
        reprocessed_tokens=None,
        actual_image_used="unknown",
        completed=completed,
        error_kind=error_kind,
        line_start=10,
        line_end=20,
    )


def _parsed(meta, samples):
    return ParsedLog(metadata=meta, samples=tuple(samples))


def test_bucket_index_boundaries():
    assert bucket_index(0) == 0
    assert bucket_index(9999) == 0
    assert bucket_index(10000) == 1
    assert bucket_index(19999) == 1
    assert bucket_index(20000) == 2
    assert bucket_index(200000) == 20
    # custom bucket sizes
    assert bucket_index(4999, 5000) == 0
    assert bucket_index(5000, 5000) == 1
    assert bucket_index(19999, 20000) == 0
    assert bucket_index(20000, 20000) == 1
    with pytest.raises(ValueError):
        bucket_index(100, 0)
    with pytest.raises(ValueError):
        bucket_index(100, -5)


def test_percentile_deterministic():
    # numpy "linear" interpolation
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 25) == pytest.approx(1.75)
    assert percentile([1, 2, 3, 4], 75) == pytest.approx(3.25)
    assert percentile([7], 50) == 7  # n=1 -> the value itself
    assert percentile([5, 1, 3], 50) == pytest.approx(3)  # unsorted input ok
    assert percentile([5, 1, 3], 25) == pytest.approx(2)
    assert percentile([5, 1, 3], 75) == pytest.approx(4)
    assert percentile([10, 20, 30, 40, 50], 90) == pytest.approx(46)
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1], -1)
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_aggregate_default_filters_probe_and_incomplete():
    parsed = _parse_fixture("bee-qwen38-small.log")
    result = aggregate_runs([parsed], dimension="run")
    assert len(result.series) == 1
    series = result.series[0]
    assert series.log_count == 1
    # probe (1 generated token < 20) and the incomplete task are excluded
    assert series.sample_count == 1
    assert result.excluded_counts["incomplete"] == 1
    assert result.excluded_counts["below_min_generated"] == 1
    assert result.excluded_counts["missing_used_context"] == 0
    assert [b.bucket_start for b in series.buckets] == [0]
    assert series.buckets[0].decode.median == pytest.approx(38.20)
    assert series.buckets[0].prefill.median == pytest.approx(1195.50)


def test_grouping_by_each_dimension():
    pool = [
        _parse_fixture("bee-qwen38-small.log"),
        _parse_fixture("b10621-cuda-small.log"),
        _parse_fixture("vulkan-small.log"),
    ]

    def keys(dimension):
        return {s.series_key
                for s in aggregate_runs(pool, dimension=dimension).series}

    assert keys("runtime") == {"BeeLlama v0.4.4", "b10621", "b10509"}
    assert keys("backend") == {"cuda", "vulkan"}
    backend = aggregate_runs(pool, dimension="backend")
    cuda = next(s for s in backend.series if s.series_key == "cuda")
    assert cuda.log_count == 2  # bee + b10621 pooled under cuda
    assert keys("kv") == {"q4_0/q4_0", "unknown/unknown"}
    assert keys("reasoning") == {"on", "off"}
    assert keys("context") == {"196608", "262144"}
    assert keys("vision") == {"yes", "no"}
    assert len(aggregate_runs(pool, dimension="run").series) == 3
    with pytest.raises(ValueError):
        aggregate_runs(pool, dimension="nonsense")


def test_reasoning_effort_dimension_grouping():
    pool = [
        _parsed(_meta(source_path=r"C:\logs\high.log",
                      reasoning_effort="high"), [_sample()]),
        _parsed(_meta(source_path=r"C:\logs\low.log",
                      reasoning_effort="low"), [_sample()]),
        _parsed(_meta(source_path=r"C:\logs\def.log"), [_sample()]),
    ]
    result = aggregate_runs(pool, dimension="reasoning_effort")
    assert {s.series_key for s in result.series} == {"high", "low", "default"}
    # compared dimension is exempt from fairness warnings
    assert not any("reasoning_effort" in w for w in result.warnings)


def test_reasoning_effort_fairness_warning_when_mixed():
    pool = [
        _parsed(_meta(source_path=r"C:\logs\high.log",
                      reasoning_effort="high"), [_sample()]),
        _parsed(_meta(source_path=r"C:\logs\default.log"), [_sample()]),
    ]
    result = aggregate_runs(pool, dimension="run")
    assert any("reasoning_effort" in w for w in result.warnings)
    # uniform effort -> no effort warning
    uniform = aggregate_runs([
        _parsed(_meta(source_path=r"C:\logs\d1.log"), [_sample()]),
        _parsed(_meta(source_path=r"C:\logs\d2.log"), [_sample()]),
    ], dimension="run")
    assert not any("reasoning_effort" in w for w in uniform.warnings)


def test_missing_buckets_not_interpolated():
    samples = [_sample(used_context=100),
               _sample(used_context=9999),
               _sample(used_context=20000)]
    result = aggregate_runs([_parsed(_meta(), samples)], dimension="run")
    series = result.series[0]
    # bucket 1 (10000..19999) has no samples and is NOT fabricated
    assert [b.bucket_start for b in series.buckets] == [0, 20000]
    assert series.buckets[0].bucket_end == 10000
    assert series.buckets[1].bucket_end == 30000


def test_decode_prefill_computed_independently():
    samples = [
        _sample(decode_tps=30.0, prefill_tps=None),
        _sample(decode_tps=None, prefill_tps=900.0),
    ]
    result = aggregate_runs([_parsed(_meta(), samples)], dimension="run")
    bucket = result.series[0].buckets[0]
    assert bucket.decode.n == 1
    assert bucket.decode.median == pytest.approx(30.0)
    assert bucket.prefill.n == 1
    assert bucket.prefill.median == pytest.approx(900.0)


def test_model_isolation_and_warning():
    alpha = _parsed(
        _meta(source_path=r"C:\logs\alpha.log", model_name="Alpha",
              model_path=r"C:\models\Alpha.gguf"),
        [_sample()])
    beta = _parsed(
        _meta(source_path=r"C:\logs\beta.log", model_name="Beta",
              model_path=r"C:\models\Beta.gguf"),
        [_sample()])
    no_filter = aggregate_runs([alpha, beta], dimension="run")
    assert len(no_filter.series) == 2
    assert any("model" in w for w in no_filter.warnings)
    isolated = aggregate_runs([alpha, beta], dimension="run",
                              filters=Filters(model="Alpha"))
    assert len(isolated.series) == 1
    assert isolated.series[0].series_key == "alpha.log"
    assert not any("model" in w for w in isolated.warnings)


def test_fairness_warning_noncompared_dimensions():
    bee = _parse_fixture("bee-qwen38-small.log")
    vulkan = _parse_fixture("vulkan-small.log")
    warnings = fairness_warnings([bee, vulkan], dimension="backend")
    joined = "\n".join(warnings)
    # non-compared fields that differ across the two runs are flagged
    assert "kv" in joined
    assert "context" in joined
    assert "runtime_label" in joined
    assert "reasoning" in joined
    assert "vision_loaded" in joined
    # the compared dimension itself is never flagged
    assert not any("'backend'" in w for w in warnings)
    # fields that are identical (model, gpu split) produce no warning
    assert "model_name" not in joined
    assert "gpu_split" not in joined
    # aggregate_runs carries the same fairness warnings
    result = aggregate_runs([bee, vulkan], dimension="backend")
    assert all(w in result.warnings for w in warnings)


def test_bucket_stats_fields():
    samples = [_sample(decode_tps=v) for v in (10.0, 20.0, 30.0)]
    result = aggregate_runs([_parsed(_meta(), samples)], dimension="run")
    stats = result.series[0].buckets[0].decode
    assert stats.n == 3
    assert stats.median == pytest.approx(20.0)
    assert stats.p25 == pytest.approx(15.0)
    assert stats.p75 == pytest.approx(25.0)
    assert stats.min == pytest.approx(10.0)
    assert stats.max == pytest.approx(30.0)
    assert stats.mean == pytest.approx(20.0)
    assert stats.anomaly_count == 0


def test_custom_bucket_sizes():
    samples = [_sample(used_context=4900),
               _sample(used_context=5100),
               _sample(used_context=24000)]
    result = aggregate_runs([_parsed(_meta(), samples)], dimension="run",
                            bucket_size=5000)
    assert [b.bucket_start for b in result.series[0].buckets] == \
        [0, 5000, 20000]


def test_anomaly_count():
    samples = [
        _sample(used_context=1000, error_kind="oom", decode_tps=25.0),
        _sample(used_context=2000, decode_tps=0.0),
        _sample(used_context=3000, decode_tps=40.0),
    ]
    result = aggregate_runs([_parsed(_meta(), samples)], dimension="run")
    stats = result.series[0].buckets[0].decode
    assert stats.n == 3
    # error_kind sample + non-positive value sample are anomalies
    assert stats.anomaly_count == 2
