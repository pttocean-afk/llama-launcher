"""Pure capability aggregation for the model-selection dashboard.

The existing log chart answers "how fast were requests?".  This module adds
an operational view: for one llama.cpp build, which model/backend/Vision
configuration was observed to start, complete inference, and at what context
and speed.  All limits are deliberately *observed/verified*, never claimed as
the physical hardware maximum.
"""
from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .log_analysis import ParsedLog, TimingSample, percentile


@dataclass(frozen=True, order=True)
class CapabilityKey:
    runtime_label: str
    runtime_version: str
    model_name: str
    backend: str
    configured_context: int | None
    vision_enabled: bool
    kv: str
    reasoning: str
    reasoning_effort: str
    gpu_split: str
    parallel: int | None


@dataclass(frozen=True)
class CurveBucket:
    bucket_start: int
    bucket_end: int
    n: int
    median: float
    p25: float
    p75: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class CapabilityRow:
    key: CapabilityKey
    run_count: int
    ready_runs: int
    inference_runs: int
    failed_runs: int
    unknown_runs: int
    vision_confirmed_runs: int
    sample_count: int
    stable_decode_tps: float | None
    stable_bucket: tuple[int, int] | None
    stable_sample_count: int
    observed_max_decode_tps: float | None
    curve: tuple[CurveBucket, ...]
    last_started_at: str | None

    @property
    def status(self) -> str:
        if self.inference_runs:
            return "inference"
        if self.ready_runs:
            return "ready"
        if self.failed_runs:
            return "failed"
        return "unknown"


@dataclass(frozen=True, order=True)
class EnvelopeKey:
    runtime_label: str
    runtime_version: str
    model_name: str
    backend: str
    vision_enabled: bool
    kv: str
    reasoning: str
    reasoning_effort: str
    gpu_split: str
    parallel: int | None


@dataclass(frozen=True)
class CapabilityEnvelope:
    key: EnvelopeKey
    run_count: int
    ready_runs: int
    inference_runs: int
    failed_runs: int
    vision_confirmed_runs: int
    max_ready_context: int | None
    max_inference_context: int | None
    first_failed_context: int | None
    stable_decode_tps: float | None
    stable_bucket: tuple[int, int] | None
    stable_sample_count: int
    observed_max_decode_tps: float | None
    sample_count: int
    row_keys: tuple[CapabilityKey, ...]


@dataclass(frozen=True)
class ModelUsage:
    model_name: str
    run_count: int
    inference_runs: int
    last_started_at: str | None


@dataclass(frozen=True)
class CapabilityReport:
    rows: tuple[CapabilityRow, ...]
    envelopes: tuple[CapabilityEnvelope, ...]
    model_usage: tuple[ModelUsage, ...]
    runtime_options: tuple[str, ...]
    selected_runtime: str | None


def _sample_ok(sample: TimingSample, min_generated: int) -> bool:
    return bool(
        sample.completed
        and sample.error_kind is None
        and sample.used_context is not None
        and sample.decode_tps is not None
        and sample.decode_tps > 0
        and (sample.generated_tokens or 0) >= min_generated
    )


def _inference_ok(sample: TimingSample) -> bool:
    return bool(
        sample.completed
        and sample.error_kind is None
        and sample.decode_tps is not None
        and sample.decode_tps > 0
        and (sample.generated_tokens or 0) > 0
    )


def _key(log: ParsedLog) -> CapabilityKey:
    m = log.metadata
    return CapabilityKey(
        runtime_label=m.runtime_label,
        runtime_version=m.runtime_version,
        model_name=m.model_name,
        backend=m.backend,
        configured_context=m.configured_context,
        vision_enabled=bool(getattr(m, "vision_requested", m.vision_loaded)
                            or m.vision_loaded),
        kv=f"{m.kv_k}/{m.kv_v}",
        reasoning=m.reasoning,
        reasoning_effort=m.reasoning_effort,
        gpu_split=m.gpu_split or "unknown",
        parallel=m.parallel,
    )


def _identity(key: CapabilityKey) -> tuple:
    """The part of a key that makes two runs "the same configuration".

    KV quantisation, reasoning settings, GPU split and parallelism are
    deliberate knobs, not distinct configurations: treat runs that differ
    only in those as one row so the list does not show near-duplicates.
    """
    return (key.runtime_label, key.runtime_version, key.model_name,
            key.backend, key.configured_context, key.vision_enabled)


def _dominant(values: Iterable[str], default: str = "") -> str:
    counts = Counter(v for v in values if v)
    return counts.most_common(1)[0][0] if counts else default


def _dominant_key(keys: Sequence[CapabilityKey]) -> CapabilityKey:
    """Merged key: identity fields fixed, knobs taken from the majority run."""
    first = keys[0]
    return CapabilityKey(
        runtime_label=first.runtime_label,
        runtime_version=first.runtime_version,
        model_name=first.model_name,
        backend=first.backend,
        configured_context=first.configured_context,
        vision_enabled=first.vision_enabled,
        kv=_dominant((k.kv for k in keys), "unknown/unknown"),
        reasoning=_dominant((k.reasoning for k in keys), "unknown"),
        reasoning_effort=_dominant((k.reasoning_effort for k in keys), "default"),
        gpu_split=_dominant((k.gpu_split for k in keys), "unknown"),
        parallel=next((k.parallel for k in keys if k.parallel is not None), None),
    )


def _envelope_key(key: CapabilityKey) -> EnvelopeKey:
    return EnvelopeKey(
        runtime_label=key.runtime_label,
        runtime_version=key.runtime_version,
        model_name=key.model_name,
        backend=key.backend,
        vision_enabled=key.vision_enabled,
        kv=key.kv,
        reasoning=key.reasoning,
        reasoning_effort=key.reasoning_effort,
        gpu_split=key.gpu_split,
        parallel=key.parallel,
    )


def _curve(samples: Iterable[TimingSample], bucket_size: int,
           min_generated: int) -> tuple[CurveBucket, ...]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for sample in samples:
        if not _sample_ok(sample, min_generated):
            continue
        idx = int(sample.used_context) // bucket_size
        buckets[idx].append(float(sample.decode_tps))
    out = []
    for idx in sorted(buckets):
        values = buckets[idx]
        out.append(CurveBucket(
            bucket_start=idx * bucket_size,
            bucket_end=(idx + 1) * bucket_size,
            n=len(values),
            median=percentile(values, 50),
            p25=percentile(values, 25),
            p75=percentile(values, 75),
            minimum=min(values),
            maximum=max(values),
        ))
    return tuple(out)


def _stable_bucket(curve: Sequence[CurveBucket],
                   minimum_samples: int) -> CurveBucket | None:
    """Lowest sufficiently sampled bucket; fall back visibly to low confidence.

    Real personal logs often have only one run for a precise configuration.
    Returning that median keeps the dashboard useful, while ``n`` is retained
    so the UI can mark it as low-confidence rather than pretending it is a
    statistically stable result.
    """
    if not curve:
        return None
    for bucket in curve:
        if bucket.n >= minimum_samples:
            return bucket
    return curve[0]


def _latest(logs: Sequence[ParsedLog]) -> str | None:
    values = [p.metadata.started_at for p in logs if p.metadata.started_at]
    return max(values) if values else None


def runtime_options(parsed_logs: Sequence[ParsedLog]) -> tuple[str, ...]:
    """Runtime labels ordered by most recently used, then usage count."""
    stats: dict[str, tuple[str, int]] = {}
    for log in parsed_logs:
        label = log.metadata.runtime_label
        last, count = stats.get(label, ("", 0))
        started = log.metadata.started_at or ""
        stats[label] = (max(last, started), count + 1)
    return tuple(sorted(stats, key=lambda k: (stats[k][0], stats[k][1], k),
                        reverse=True))


def model_usage(parsed_logs: Sequence[ParsedLog], *,
                runtime_label: str | None = None) -> tuple[ModelUsage, ...]:
    grouped: dict[str, list[ParsedLog]] = defaultdict(list)
    for log in parsed_logs:
        if runtime_label is not None and log.metadata.runtime_label != runtime_label:
            continue
        grouped[log.metadata.model_name].append(log)
    out = []
    for name, logs in grouped.items():
        out.append(ModelUsage(
            model_name=name,
            run_count=len(logs),
            inference_runs=sum(any(_inference_ok(s) for s in p.samples)
                               for p in logs),
            last_started_at=_latest(logs),
        ))
    return tuple(sorted(out, key=lambda x: (
        -x.run_count, -x.inference_runs, x.model_name.lower())))


def aggregate_capabilities(
    parsed_logs: Sequence[ParsedLog], *,
    runtime_label: str | None = None,
    selected_models: Iterable[str] | None = None,
    bucket_size: int = 10_000,
    min_generated: int = 20,
    stable_min_samples: int = 3,
) -> CapabilityReport:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    if min_generated < 0:
        raise ValueError("min_generated must be non-negative")
    selected = set(selected_models) if selected_models is not None else None
    options = runtime_options(parsed_logs)
    logs = [p for p in parsed_logs
            if (runtime_label is None or p.metadata.runtime_label == runtime_label)
            and (selected is None or p.metadata.model_name in selected)]

    grouped: dict[tuple, list[ParsedLog]] = defaultdict(list)
    for log in logs:
        grouped[_identity(_key(log))].append(log)

    rows: list[CapabilityRow] = []
    for ident, key_logs in grouped.items():
        merged_key = _dominant_key([_key(p) for p in key_logs])
        samples = [s for p in key_logs for s in p.samples]
        curve = _curve(samples, bucket_size, min_generated)
        inference_flags = [any(_inference_ok(s) for s in p.samples)
                           for p in key_logs]
        ready_flags = [p.metadata.startup_status == "ready" or inferred
                       for p, inferred in zip(key_logs, inference_flags)]
        failed_flags = [p.metadata.startup_status == "failed" and not ready
                        for p, ready in zip(key_logs, ready_flags)]
        ready_runs = sum(ready_flags)
        inference_runs = sum(inference_flags)
        failed_runs = sum(failed_flags)
        values = [s.decode_tps for s in samples if _sample_ok(s, min_generated)]
        stable = _stable_bucket(curve, stable_min_samples)
        rows.append(CapabilityRow(
            key=merged_key,
            run_count=len(key_logs),
            ready_runs=ready_runs,
            inference_runs=inference_runs,
            failed_runs=failed_runs,
            unknown_runs=len(key_logs) - ready_runs - failed_runs,
            vision_confirmed_runs=sum(bool(getattr(p.metadata, "vision_ready", False))
                                      for p in key_logs),
            sample_count=len(values),
            stable_decode_tps=stable.median if stable else None,
            stable_bucket=((stable.bucket_start, stable.bucket_end)
                           if stable else None),
            stable_sample_count=stable.n if stable else 0,
            observed_max_decode_tps=max(values) if values else None,
            curve=curve,
            last_started_at=_latest(key_logs),
        ))
    rows.sort(key=lambda r: (
        r.key.model_name.lower(), r.key.backend, not r.key.vision_enabled,
        -(r.key.configured_context or -1)))

    env_groups: dict[EnvelopeKey, list[CapabilityRow]] = defaultdict(list)
    for row in rows:
        env_groups[_envelope_key(row.key)].append(row)
    envelopes: list[CapabilityEnvelope] = []
    for key, env_rows in env_groups.items():
        ready_ctx = [r.key.configured_context for r in env_rows
                     if r.ready_runs and r.key.configured_context is not None]
        infer_ctx = [r.key.configured_context for r in env_rows
                     if r.inference_runs and r.key.configured_context is not None]
        baseline = max(infer_ctx) if infer_ctx else (max(ready_ctx) if ready_ctx else None)
        failed_ctx = [r.key.configured_context for r in env_rows
                      if r.failed_runs and not r.ready_runs
                      and r.key.configured_context is not None
                      and (baseline is None or r.key.configured_context > baseline)]
        speed_row = None
        target_ctx = max(infer_ctx) if infer_ctx else None
        if target_ctx is not None:
            candidates = [r for r in env_rows
                          if r.key.configured_context == target_ctx
                          and r.stable_decode_tps is not None]
            speed_row = candidates[0] if candidates else None
        if speed_row is None:
            candidates = [r for r in env_rows if r.stable_decode_tps is not None]
            speed_row = max(candidates,
                            key=lambda r: r.key.configured_context or -1) \
                if candidates else None
        envelopes.append(CapabilityEnvelope(
            key=key,
            run_count=sum(r.run_count for r in env_rows),
            ready_runs=sum(r.ready_runs for r in env_rows),
            inference_runs=sum(r.inference_runs for r in env_rows),
            failed_runs=sum(r.failed_runs for r in env_rows),
            vision_confirmed_runs=sum(r.vision_confirmed_runs for r in env_rows),
            max_ready_context=max(ready_ctx) if ready_ctx else None,
            max_inference_context=max(infer_ctx) if infer_ctx else None,
            first_failed_context=min(failed_ctx) if failed_ctx else None,
            stable_decode_tps=(speed_row.stable_decode_tps if speed_row else None),
            stable_bucket=(speed_row.stable_bucket if speed_row else None),
            stable_sample_count=(speed_row.stable_sample_count if speed_row else 0),
            observed_max_decode_tps=max(
                (r.observed_max_decode_tps for r in env_rows
                 if r.observed_max_decode_tps is not None), default=None),
            sample_count=sum(r.sample_count for r in env_rows),
            row_keys=tuple(r.key for r in env_rows),
        ))
    envelopes.sort(key=lambda e: (
        -(e.max_inference_context or e.max_ready_context or -1),
        -(e.stable_decode_tps or -1), e.key.model_name.lower()))

    return CapabilityReport(
        rows=tuple(rows),
        envelopes=tuple(envelopes),
        model_usage=model_usage(parsed_logs, runtime_label=runtime_label),
        runtime_options=options,
        selected_runtime=runtime_label,
    )


def render_capability_csv(report: CapabilityReport) -> str:
    """One row per model/backend/Vision envelope for portable comparison."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow((
        "runtime", "runtime_version", "model", "backend", "vision",
        "kv", "reasoning", "reasoning_effort", "gpu_split", "parallel",
        "runs", "ready_runs", "inference_runs", "failed_runs",
        "vision_confirmed_runs", "max_inference_context", "max_ready_context", "first_failed_context",
        "stable_decode_tps", "stable_bucket_start", "stable_bucket_end",
        "stable_samples", "observed_max_decode_tps", "samples",
    ))
    for env in report.envelopes:
        k = env.key
        writer.writerow((
            k.runtime_label, k.runtime_version, k.model_name, k.backend,
            "on" if k.vision_enabled else "off", k.kv, k.reasoning,
            k.reasoning_effort, k.gpu_split,
            "" if k.parallel is None else k.parallel,
            env.run_count, env.ready_runs, env.inference_runs, env.failed_runs,
            env.vision_confirmed_runs,
            "" if env.max_inference_context is None else env.max_inference_context,
            "" if env.max_ready_context is None else env.max_ready_context,
            "" if env.first_failed_context is None else env.first_failed_context,
            "" if env.stable_decode_tps is None else f"{env.stable_decode_tps:.4f}",
            "" if env.stable_bucket is None else env.stable_bucket[0],
            "" if env.stable_bucket is None else env.stable_bucket[1],
            env.stable_sample_count,
            "" if env.observed_max_decode_tps is None else
            f"{env.observed_max_decode_tps:.4f}", env.sample_count,
        ))
    return buf.getvalue()


def pareto_rows(rows: Sequence[CapabilityRow]) -> tuple[CapabilityRow, ...]:
    """Non-dominated verified rows by configured context and stable speed."""
    candidates = [r for r in rows
                  if r.inference_runs and r.key.configured_context is not None
                  and r.stable_decode_tps is not None]
    result = []
    for row in candidates:
        ctx = int(row.key.configured_context)
        speed = float(row.stable_decode_tps)
        dominated = any(
            other is not row
            and int(other.key.configured_context) >= ctx
            and float(other.stable_decode_tps) >= speed
            and (int(other.key.configured_context) > ctx
                 or float(other.stable_decode_tps) > speed)
            for other in candidates
        )
        if not dominated:
            result.append(row)
    return tuple(sorted(result, key=lambda r: int(r.key.configured_context)))
