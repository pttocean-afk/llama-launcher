from llama_launcher.capability_analysis import (
    aggregate_capabilities,
    model_usage,
    pareto_rows,
    render_capability_csv,
    runtime_options,
)
from llama_launcher.log_analysis import ParsedLog, RunMetadata, TimingSample, parse_log_text


def _meta(**overrides):
    data = dict(
        source_path="run.log", started_at="2026-01-01T00:00:00",
        profile_name="Qwen", model_path=r"C:\models\qwen.gguf",
        model_name="qwen", executable_path=r"C:\b10770\llama-server.exe",
        runtime_label="b10770", runtime_version="10770", backend="cuda",
        kv_k="q4_0", kv_v="q4_0", reasoning="off",
        reasoning_effort="default", configured_context=131072,
        vision_loaded=False, mmproj_path=None, gpu_split="16,8",
        batch=None, ubatch=None, parallel=1, jinja=False, extra_flags=(),
        vision_requested=False, vision_ready=False, startup_status="ready",
        startup_error_kind=None,
    )
    data.update(overrides)
    return RunMetadata(**data)


def _sample(ctx=5000, speed=38.0, generated=100, completed=True,
            error=None):
    return TimingSample(
        task_id=1, slot_id=0, used_context=ctx,
        used_context_source="stop_processing", prompt_tokens=max(ctx - generated, 0),
        prompt_ms=1000, prefill_tps=500, generated_tokens=generated,
        decode_ms=2000, decode_tps=speed, cache_source=None,
        cached_tokens=None, reprocessed_tokens=None, actual_image_used="no",
        completed=completed, error_kind=error, line_start=1, line_end=5,
    )


def _log(meta=None, samples=()):
    return ParsedLog(metadata=meta or _meta(), samples=tuple(samples))


def test_parser_records_ready_and_confirmed_vision():
    text = """# 2026-01-01T00:00:00  Qwen
# C:\\b10770\\llama-server.exe -m C:\\qwen.gguf --mmproj C:\\mm.gguf -c 131072
================================================================================
load_model: loaded multimodal model, 'C:\\mm.gguf'
llama_server: model loaded
llama_server: listening on http://0.0.0.0:8080
"""
    meta = parse_log_text(text).metadata
    assert meta.vision_requested is True
    assert meta.vision_ready is True
    assert meta.startup_status == "ready"
    assert meta.startup_error_kind is None


def test_parser_marks_unrecovered_startup_oom_failed():
    text = """# 2026-01-01T00:00:00  Qwen
# C:\\b10770\\llama-server.exe -m C:\\qwen.gguf -c 262144
================================================================================
cudaMalloc failed: out of memory
failed to load model
"""
    meta = parse_log_text(text).metadata
    assert meta.startup_status == "failed"
    assert meta.startup_error_kind == "oom"


def test_ready_wins_over_recoverable_oom():
    text = """# 2026-01-01T00:00:00  Qwen
# C:\\b10770\\llama-server.exe -m C:\\qwen.gguf -c 131072
================================================================================
cudaMalloc failed: out of memory
llama_server: listening on http://0.0.0.0:8080
"""
    assert parse_log_text(text).metadata.startup_status == "ready"


def test_capability_context_envelope_and_speeds():
    logs = [
        _log(_meta(configured_context=131072),
             [_sample(3000, 40), _sample(4000, 38), _sample(5000, 39)]),
        _log(_meta(configured_context=196608), [_sample(5000, 34)]),
        _log(_meta(configured_context=262144, startup_status="failed",
                   startup_error_kind="oom"), []),
    ]
    report = aggregate_capabilities(logs, runtime_label="b10770")
    env = report.envelopes[0]
    assert env.max_ready_context == 196608
    assert env.max_inference_context == 196608
    assert env.first_failed_context == 262144
    assert env.run_count == 3
    assert env.inference_runs == 2
    row = next(r for r in report.rows if r.key.configured_context == 131072)
    assert row.stable_decode_tps == 39
    assert row.observed_max_decode_tps == 40
    assert row.status == "inference"


def test_short_outputs_prove_inference_but_do_not_distort_curve():
    report = aggregate_capabilities([_log(samples=[_sample(generated=1, speed=99)])])
    row = report.rows[0]
    assert row.inference_runs == 1
    assert row.sample_count == 0
    assert row.stable_decode_tps is None


def test_runtime_and_model_usage_ordering():
    logs = [
        _log(_meta(model_name="rare", runtime_label="b1",
                   started_at="2026-01-01"), [_sample()]),
        _log(_meta(model_name="popular", runtime_label="b2",
                   started_at="2026-02-01"), [_sample()]),
        _log(_meta(model_name="popular", runtime_label="b2",
                   started_at="2026-02-02"), [_sample()]),
    ]
    assert runtime_options(logs) == ("b2", "b1")
    usage = model_usage(logs, runtime_label="b2")
    assert [(u.model_name, u.run_count) for u in usage] == [("popular", 2)]


def test_filters_runtime_and_multiple_models():
    logs = [
        _log(_meta(model_name="a"), [_sample()]),
        _log(_meta(model_name="b"), [_sample()]),
        _log(_meta(model_name="c", runtime_label="b10600"), [_sample()]),
    ]
    report = aggregate_capabilities(
        logs, runtime_label="b10770", selected_models={"a", "b"})
    assert {r.key.model_name for r in report.rows} == {"a", "b"}


def test_none_bucket_and_stable_fields():
    r = aggregate_capabilities([_log(_meta(configured_context=131072))])
    row = r.rows[0]
    assert row.stable_decode_tps is None
    assert row.stable_bucket is None
    assert row.stable_sample_count == 0


def test_capability_csv_contains_verified_limits():
    report = aggregate_capabilities([
        _log(_meta(configured_context=131072), [_sample(speed=38)]),
        _log(_meta(configured_context=262144, startup_status="failed",
                   startup_error_kind="oom"), []),
    ])
    csv_text = render_capability_csv(report)
    assert "max_inference_context" in csv_text
    assert "131072" in csv_text
    assert "262144" in csv_text
    assert "38.0000" in csv_text


def test_pareto_frontier():
    logs = [
        _log(_meta(model_name="fast-small", configured_context=100_000),
             [_sample(speed=50)]),
        _log(_meta(model_name="dominated", configured_context=80_000),
             [_sample(speed=30)]),
        _log(_meta(model_name="large", configured_context=200_000),
             [_sample(speed=35)]),
    ]
    rows = aggregate_capabilities(logs, stable_min_samples=1).rows
    assert {r.key.model_name for r in pareto_rows(rows)} == {"fast-small", "large"}
