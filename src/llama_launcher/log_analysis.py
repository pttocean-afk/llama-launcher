"""Parse llama-server log files into normalized, UI-independent records.

The Launcher writes each server log with a ``#`` header (timestamp/profile,
optional preflight summary, then the exact command line) followed by an
``=`` separator and the raw server output.  Older launcher builds used a
two-line header, and externally captured logs may have no header at all; in
those cases metadata is inferred from the log body.

This module is read-only with respect to logs: it never truncates, rewrites,
rotates, renames, or deletes a source file.
"""
from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Sequence

__all__ = [
    "RunMetadata",
    "TimingSample",
    "ParsedLog",
    "ScanError",
    "parse_log_text",
    "parse_log_file",
    "scan_log_paths",
    "DEFAULT_MIN_GENERATED",
    "bucket_index",
    "percentile",
    "Filters",
    "MetricStats",
    "BucketStats",
    "SeriesAggregation",
    "AggregationResult",
    "aggregate_runs",
    "fairness_warnings",
    "series_key",
    "REASONING_EFFORT_LEVELS",
]

#: Default quality filter: samples that generated fewer tokens than this are
#: excluded from curve aggregates (they stay in raw views/exports).
DEFAULT_MIN_GENERATED = 20

UNKNOWN = "unknown"

#: llama.cpp --reasoning-effort levels accepted by recent builds (b10621+).
#: "default" means the flag is not passed and the model's chat template
#: decides the thinking effort.
REASONING_EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")

# --- body-line patterns --------------------------------------------------------

_LAUNCH_RE = re.compile(r"launch_slot_:.*?\bid\s*(\d+)\s*\|\s*task\s*(\d+)\s*\|")
_ID_RE = re.compile(r"\bid\s*(\d+)\s*\|\s*task\s*(\d+)\s*\|")
_PROMPT_RE = re.compile(
    r"prompt eval time =\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    r"(?:\s*\([^)]*?([\d.]+)\s*tokens per second\))?")
_DECODE_RE = re.compile(
    r"(?<!prompt )eval time =\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    r"(?:\s*\([^)]*?([\d.]+)\s*tokens per second\))?")
_TOTAL_RE = re.compile(r"total time =\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
_STOP_RE = re.compile(
    r"stop processing:\s*n_tokens\s*=\s*(\d+)\s*,\s*truncated\s*=\s*(\d+)")
_NGEN_RE = re.compile(r"n_gen =\s*(\d+)\s*,\s*tg =\s*[\d.]+\s*t/s")
_PROMPT_CACHE_RE = re.compile(
    r"prompt cache:\s*source = ([A-Za-z_]+),\s*cached = (\d+) tokens?,"
    r"\s*reprocessed = (\d+) tokens?")
_IMAGE_EVIDENCE_RE = re.compile(r"add_image|image processed|image tokens")
_TASK_ID_IN_LINE_RE = re.compile(r"\btask\s+(\d+)\b")

_CUDA_EVIDENCE_RE = re.compile(
    r"cudaMalloc|ggml_backend_cuda|\bCUDA\d*\s*buffer|\bCUDA:\s|cuInit|cublas",
    re.IGNORECASE)
_VULKAN_EVIDENCE_RE = re.compile(
    r"ggml_backend_vulkan|vulkan\s*backend|VK_LAYER", re.IGNORECASE)

_OOM_RE = re.compile(
    r"cudaMalloc failed|out of memory|failed to allocate|KV cache is full",
    re.IGNORECASE)
_DECODE_FAIL_RE = re.compile(
    r"failed to decode|decode failure|decode error", re.IGNORECASE)

_MODEL_RE = re.compile(r"loading model '([^']+)'")
_MMPROJ_RE = re.compile(r"loaded multimodal model, '([^']+)'")
_NCTX_RE = re.compile(r"n_ctx_slot = (\d+)")
_BUILD_RE = re.compile(r"build\s*=\s*(\d{4,6})")
_BNUM_RE = re.compile(r"(?<![0-9a-z])b(\d{4,6})(?![0-9])")
_SERVER_READY_RE = re.compile(
    r"llama_server:\s*listening\s+on\s+https?://", re.IGNORECASE)
_MODEL_LOADED_RE = re.compile(
    r"llama_server:\s*model\s+loaded", re.IGNORECASE)
_STARTUP_FATAL_RE = re.compile(
    r"failed to load model|error loading model|unable to load model|"
    r"failed to create (?:llama_)?context|model loading failed|"
    r"error while loading model",
    re.IGNORECASE)

_SEPARATOR_RE = re.compile(r"^=+\s*$")


# --- normalized data model ------------------------------------------------------

@dataclass(frozen=True)
class RunMetadata:
    """One record per server log: the configuration of that run."""
    source_path: str
    started_at: str | None
    profile_name: str | None
    model_path: str | None
    model_name: str
    executable_path: str | None
    runtime_label: str
    runtime_version: str
    backend: str
    kv_k: str
    kv_v: str
    reasoning: str
    reasoning_effort: str
    configured_context: int | None
    vision_loaded: bool
    mmproj_path: str | None
    gpu_split: str | None
    batch: int | None
    ubatch: int | None
    parallel: int | None
    jinja: bool
    extra_flags: tuple[str, ...]
    # ``vision_loaded`` is retained as the historical/configured Vision flag.
    # The fields below distinguish what was requested from what the server
    # actually confirmed in its output.
    vision_requested: bool = False
    vision_ready: bool = False
    # ready = listening line (or completed legacy sample), failed = fatal
    # startup evidence without readiness, unknown = incomplete/old log.
    startup_status: str = UNKNOWN
    startup_error_kind: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TimingSample:
    """One record per request/task seen in a log."""
    task_id: int | None
    slot_id: int | None
    used_context: int | None
    used_context_source: str
    prompt_tokens: int | None
    prompt_ms: float | None
    prefill_tps: float | None
    generated_tokens: int | None
    decode_ms: float | None
    decode_tps: float | None
    cache_source: str | None
    cached_tokens: int | None
    reprocessed_tokens: int | None
    actual_image_used: str
    completed: bool
    error_kind: str | None
    line_start: int
    line_end: int


@dataclass(frozen=True)
class ParsedLog:
    metadata: RunMetadata
    samples: tuple[TimingSample, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScanError:
    path: str
    error: str


# --- header / command-line parsing ---------------------------------------------

def _split_command(cmd: str) -> list[str]:
    """Windows-command-aware split: quoted paths with spaces stay intact.

    posix=True would be wrong here (backslash is an escape character there
    and corrupts Windows paths); posix=False keeps the quote characters in
    the tokens, so balanced surrounding quotes are stripped afterwards.
    """
    try:
        tokens = shlex.split(cmd, posix=False)
    except ValueError:
        tokens = cmd.split()
    cleaned: list[str] = []
    for tok in tokens:
        if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
            tok = tok[1:-1]
        cleaned.append(tok)
    return cleaned


def _looks_like_path(token: str) -> bool:
    return (
        "\\" in token
        or "/" in token
        or token.lower().endswith(".exe")
        or token.lower().endswith(".bin")
    )


def _model_name(model_path: str) -> str:
    """Normalize a model filename from a (possibly Windows) path."""
    path = PureWindowsPath(model_path) if "\\" in model_path \
        else PurePosixPath(model_path)
    return path.stem or UNKNOWN


def _parse_header(lines: list[str], warnings: list[str]) -> tuple[dict, int]:
    """Return (header_info, body_start_index)."""
    header_lines: list[str] = []
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        header_lines.append(lines[i])
        i += 1
    if not header_lines:
        return {}, 0
    if i < len(lines) and _SEPARATOR_RE.match(lines[i]):
        i += 1

    info: dict = {}
    first = header_lines[0][1:].lstrip()
    parts = first.split(None, 1)
    if parts:
        try:
            datetime.fromisoformat(parts[0])
            info["started_at"] = parts[0]
            info["profile_name"] = parts[1].strip() if len(parts) > 1 else None
        except ValueError:
            warnings.append(f"unparseable header timestamp: {parts[0]!r}")
            info["started_at"] = None
            info["profile_name"] = first.strip() or None

    # The command line is the last '#' line whose first token looks like a
    # program path (the preflight summary line starts with plain words).
    for line in reversed(header_lines[1:]):
        tokens = _split_command(line[1:].strip())
        if tokens and _looks_like_path(tokens[0]):
            info["argv"] = tokens
            break
    return info, i


_VALUE_FLAGS = {
    "-m": "model_path",
    "--model": "model_path",
    "--mmproj": "mmproj_path",
    "-c": "configured_context",
    "--ctx": "configured_context",
    "-ts": "gpu_split",
    "-b": "batch",
    "--batch": "batch",
    "-ub": "ubatch",
    "--ubatch": "ubatch",
    "--parallel": "parallel",
    "-ctk": "kv_k",
    "--cache-type-k": "kv_k",
    "-ctv": "kv_v",
    "--cache-type-v": "kv_v",
    "--reasoning": "reasoning",
    "--reasoning-effort": "reasoning_effort",
    "--device": "device",
}
_INT_FLAGS = {"configured_context", "batch", "ubatch", "parallel"}


def _parse_argv(argv: list[str], warnings: list[str]) -> dict:
    """Parse llama-server argv (argv[0] already removed) into flag values."""
    flags: dict = {}
    extra: list[str] = []

    def _set(key: str, value: str) -> None:
        if key in flags:
            warnings.append(f"duplicate flag for {key}; first occurrence kept")
            return
        if key in _INT_FLAGS:
            try:
                value = str(int(value))
            except ValueError:
                warnings.append(f"non-integer value for {key}: {value!r}")
        flags[key] = value

    i = 0
    while i < len(argv):
        tok = argv[i]
        i += 1
        if tok == "--jinja":
            flags["jinja"] = True
            continue
        if not tok.startswith("-"):
            continue  # stray positional token
        if tok in _VALUE_FLAGS:
            if i >= len(argv):
                warnings.append(f"flag {tok} missing value")
                continue
            _set(_VALUE_FLAGS[tok], argv[i])
            i += 1
        else:
            # Unknown flag: keep (flag, value) pairs for diagnostics.
            if i < len(argv) and not argv[i].startswith("-"):
                extra.extend([tok, argv[i]])
                i += 1
            else:
                extra.extend([tok, ""])
    flags.setdefault("jinja", False)
    flags["extra_flags"] = tuple(extra)
    return flags


def _detect_backend(flags: dict, executable_path: str | None,
                    body_cuda: bool, body_vulkan: bool) -> str:
    """Backend priority: --device flag, executable path, then log evidence.
    Never inferred from model names."""
    device = (flags.get("device") or "").strip()
    if device.lower().startswith("vulkan"):
        return "vulkan"
    if device.lower().startswith("cuda"):
        return "cuda"
    if executable_path:
        exe = executable_path.lower()
        if "vulkan" in exe:
            return "vulkan"
        if "cuda" in exe:
            return "cuda"
    if body_cuda:
        return "cuda"
    if body_vulkan:
        return "vulkan"
    return UNKNOWN


def _detect_runtime(executable_path: str | None, body_build: str | None,
                    backend: str) -> tuple[str, str]:
    if executable_path:
        for segment in re.split(r"[\\/]+", executable_path):
            low = segment.lower()
            if "beellama" in low:
                m = re.search(r"v?(\d+(?:\.\d+)+)", segment)
                if m:
                    return f"BeeLlama v{m.group(1)}", m.group(1)
                return "BeeLlama", UNKNOWN
            m = _BNUM_RE.search(segment)
            if m:
                return f"b{m.group(1)}", m.group(1)
    if body_build:
        return f"b{body_build}", body_build
    if backend != UNKNOWN:
        return "legacy", UNKNOWN
    return UNKNOWN, UNKNOWN


# --- body scan -------------------------------------------------------------------

@dataclass
class _TaskState:
    task_id: int | None = None
    slot_id: int | None = None
    line_start: int = 0
    line_end: int = 0
    launched: bool = False
    has_timing: bool = False
    prompt_tokens: int | None = None
    prompt_ms: float | None = None
    prefill_tps: float | None = None
    generated_tokens: int | None = None
    decode_ms: float | None = None
    decode_tps: float | None = None
    total_tokens: int | None = None
    stop_n_tokens: int | None = None
    truncated: int | None = None
    cache_source: str | None = None
    cached_tokens: int | None = None
    reprocessed_tokens: int | None = None
    last_n_gen: int | None = None
    image_evidence: bool = False
    error_kind: str | None = None
    closed: bool = False


def _scan_body(lines: list[str], body_start: int, warnings: list[str]
               ) -> tuple[list[_TaskState], dict]:
    """Walk body lines, pairing timing lines by (slot, task) id.

    Returns (task states in first-seen order, content evidence dict).
    """
    tasks: dict[tuple[int | None, int | None], _TaskState] = {}
    order: list[tuple[int | None, int | None]] = []
    active: _TaskState | None = None
    evidence: dict = {
        "cuda": False, "vulkan": False, "build": None,
        "model_path": None, "mmproj_path": None, "n_ctx": None,
        "server_ready": False, "model_loaded": False,
        "startup_oom": False, "startup_fatal": False,
    }

    def _get_state(slot: int, task: int, lineno: int) -> _TaskState:
        key = (slot, task)
        state = tasks.get(key)
        if state is None:
            state = _TaskState(task_id=task, slot_id=slot,
                               line_start=lineno, line_end=lineno)
            tasks[key] = state
            order.append(key)
        return state

    for lineno, line in enumerate(lines[body_start:], start=body_start + 1):
        if not line.strip():
            continue

        # content evidence (metadata fallbacks + backend hints)
        if evidence["model_path"] is None:
            m = _MODEL_RE.search(line)
            if m:
                evidence["model_path"] = m.group(1)
        if evidence["mmproj_path"] is None:
            m = _MMPROJ_RE.search(line)
            if m:
                evidence["mmproj_path"] = m.group(1)
        if evidence["n_ctx"] is None:
            m = _NCTX_RE.search(line)
            if m:
                evidence["n_ctx"] = int(m.group(1))
        if evidence["build"] is None:
            m = _BUILD_RE.search(line)
            if m:
                evidence["build"] = m.group(1)
        if not evidence["cuda"] and _CUDA_EVIDENCE_RE.search(line):
            evidence["cuda"] = True
        if not evidence["vulkan"] and _VULKAN_EVIDENCE_RE.search(line):
            evidence["vulkan"] = True
        if _SERVER_READY_RE.search(line):
            evidence["server_ready"] = True
        if _MODEL_LOADED_RE.search(line):
            evidence["model_loaded"] = True
        if _STARTUP_FATAL_RE.search(line) and not evidence["server_ready"]:
            evidence["startup_fatal"] = True

        # error lines: associate with the active, not-yet-closed task
        if _OOM_RE.search(line):
            if active is not None and not active.closed:
                if active.error_kind is None:
                    active.error_kind = "oom"
                active.has_timing = True  # an observed error keeps the task
                active.line_end = lineno
            else:
                # Startup allocation errors have no active task.  A runtime
                # may recover (for example by retrying another device), so a
                # later listening line still wins over this evidence.
                if not evidence["server_ready"]:
                    evidence["startup_oom"] = True
                warnings.append(
                    f"unassociated OOM at line {lineno}: {line.strip()[:160]}")
            continue
        if _DECODE_FAIL_RE.search(line):
            if active is not None and not active.closed:
                if active.error_kind is None:
                    active.error_kind = "decode_failure"
                active.has_timing = True
                active.line_end = lineno
            else:
                warnings.append(
                    f"unassociated decode failure at line {lineno}")
            continue

        m = _LAUNCH_RE.search(line)
        if m:
            slot, task = int(m.group(1)), int(m.group(2))
            state = _get_state(slot, task, lineno)
            if state.launched and (state.has_timing or state.error_kind):
                warnings.append(
                    f"task {task} relaunched before completion; "
                    f"previous partial state replaced")
            state.launched = True
            if not state.has_timing and state.error_kind is None:
                state.line_start = lineno
            active = state
            continue

        idm = _ID_RE.search(line)
        if not idm:
            continue
        slot, task = int(idm.group(1)), int(idm.group(2))
        state = _get_state(slot, task, lineno)
        if state.closed:
            warnings.append(
                f"timing line after stop for task {task} at line {lineno}; "
                f"ignored")
            continue

        pm = _PROMPT_RE.search(line)
        if pm:
            state.prompt_ms = float(pm.group(1))
            state.prompt_tokens = int(pm.group(2))
            if pm.group(3):
                state.prefill_tps = float(pm.group(3))
            state.has_timing = True
            state.line_end = lineno
            continue

        dm = _DECODE_RE.search(line)
        if dm:
            state.decode_ms = float(dm.group(1))
            state.generated_tokens = int(dm.group(2))
            if dm.group(3):
                state.decode_tps = float(dm.group(3))
            state.has_timing = True
            state.line_end = lineno
            continue

        tm = _TOTAL_RE.search(line)
        if tm:
            state.total_tokens = int(tm.group(2))
            state.has_timing = True
            state.line_end = lineno
            continue

        sm = _STOP_RE.search(line)
        if sm:
            state.stop_n_tokens = int(sm.group(1))
            state.truncated = int(sm.group(2))
            state.has_timing = True
            state.closed = True
            state.line_end = lineno
            active = None
            continue

        gm = _NGEN_RE.search(line)
        if gm:
            state.last_n_gen = int(gm.group(1))
            state.has_timing = True
            state.line_end = lineno
            continue

        cm = _PROMPT_CACHE_RE.search(line)
        if cm:
            state.cache_source = cm.group(1)
            state.cached_tokens = int(cm.group(2))
            state.reprocessed_tokens = int(cm.group(3))
            state.has_timing = True
            state.line_end = lineno
            continue

        if _IMAGE_EVIDENCE_RE.search(line):
            tmatch = _TASK_ID_IN_LINE_RE.search(line)
            if tmatch is None or int(tmatch.group(1)) == task:
                state.image_evidence = True
                state.line_end = lineno

    return [tasks[key] for key in order], evidence


def _build_samples(states: list[_TaskState], vision_loaded: bool,
                   warnings: list[str]) -> tuple[TimingSample, ...]:
    samples: list[TimingSample] = []
    for state in states:
        if not (state.has_timing or state.error_kind):
            warnings.append(f"task {state.task_id}: no timing lines, dropped")
            continue
        if state.stop_n_tokens is not None:
            used_context = state.stop_n_tokens
            source = "stop_processing"
        elif state.total_tokens is not None:
            used_context = state.total_tokens
            source = "total_time"
        elif state.last_n_gen is not None:
            used_context = (state.prompt_tokens or 0) + state.last_n_gen
            source = "checkpoint"
        else:
            used_context = None
            source = "unknown"

        if state.image_evidence:
            actual_image = "yes"
        elif not vision_loaded:
            actual_image = "no"
        else:
            actual_image = "unknown"

        samples.append(TimingSample(
            task_id=state.task_id,
            slot_id=state.slot_id,
            used_context=used_context,
            used_context_source=source,
            prompt_tokens=state.prompt_tokens,
            prompt_ms=state.prompt_ms,
            prefill_tps=state.prefill_tps,
            generated_tokens=(state.generated_tokens
                              if state.generated_tokens is not None
                              else state.last_n_gen),
            decode_ms=state.decode_ms,
            decode_tps=state.decode_tps,
            cache_source=state.cache_source,
            cached_tokens=state.cached_tokens,
            reprocessed_tokens=state.reprocessed_tokens,
            actual_image_used=actual_image,
            completed=(state.stop_n_tokens is not None
                       or state.total_tokens is not None),
            error_kind=state.error_kind,
            line_start=state.line_start or 1,
            line_end=state.line_end or state.line_start or 1,
        ))
    return tuple(samples)


# --- public API ---------------------------------------------------------------------

def parse_log_text(text: str, *, source_path: str = "<text>") -> ParsedLog:
    """Parse raw log text (CRLF/LF tolerant) into a ParsedLog."""
    warnings: list[str] = []
    lines = text.splitlines()

    header, body_start = _parse_header(lines, warnings)

    argv = header.get("argv") or []
    flags = _parse_argv(argv[1:], warnings) if len(argv) > 1 else {}
    flags.setdefault("jinja", False)
    flags.setdefault("extra_flags", ())
    executable_path = argv[0] if argv else None
    model_path = flags.get("model_path")
    mmproj_path = flags.get("mmproj_path")
    ctx = flags.get("configured_context")
    gpu_split = flags.get("gpu_split")
    reasoning_raw = flags.get("reasoning")
    if reasoning_raw is not None and reasoning_raw not in ("on", "off", "auto"):
        warnings.append(f"unrecognized --reasoning value: {reasoning_raw!r}")
        reasoning = UNKNOWN
    else:
        reasoning = reasoning_raw or UNKNOWN

    effort_raw = flags.get("reasoning_effort")
    if effort_raw is None:
        reasoning_effort = "default"
    elif str(effort_raw).strip().lower() in REASONING_EFFORT_LEVELS:
        reasoning_effort = str(effort_raw).strip().lower()
    else:
        warnings.append(f"unrecognized --reasoning-effort value: {effort_raw!r}")
        reasoning_effort = UNKNOWN

    states, evidence = _scan_body(lines, body_start, warnings)

    if not header:
        warnings.insert(0, "no launcher header found; metadata inferred "
                           "from log content")
    if model_path is None and evidence["model_path"] is not None:
        model_path = evidence["model_path"]
    vision_requested = (mmproj_path is not None
                        or evidence["mmproj_path"] is not None)
    if mmproj_path is None and evidence["mmproj_path"] is not None:
        mmproj_path = evidence["mmproj_path"]
    # Compatibility field used by existing filters means Vision configured;
    # vision_ready below is the stronger server-confirmed signal.
    vision_loaded = vision_requested
    vision_ready = evidence["mmproj_path"] is not None

    samples = _build_samples(states, vision_loaded, warnings)
    inference_seen = any(
        s.completed and s.error_kind is None
        and ((s.generated_tokens or 0) > 0 or (s.prompt_tokens or 0) > 0)
        for s in samples)
    if evidence["server_ready"] or inference_seen:
        startup_status = "ready"
        startup_error_kind = None
    elif evidence["startup_oom"]:
        startup_status = "failed"
        startup_error_kind = "oom"
    elif evidence["startup_fatal"]:
        startup_status = "failed"
        startup_error_kind = "model_load"
    else:
        startup_status = UNKNOWN
        startup_error_kind = None

    if ctx is None:
        ctx = evidence["n_ctx"]
    elif (evidence["n_ctx"] is not None
          and evidence["n_ctx"] != int(ctx)):
        warnings.append(
            f"configured context conflict: header -c {ctx} vs body "
            f"n_ctx_slot {evidence['n_ctx']}; header kept")

    if ctx is not None:
        try:
            ctx = int(ctx)
        except (TypeError, ValueError):
            warnings.append(f"non-integer configured context: {ctx!r}")
            ctx = None

    backend = _detect_backend(flags, executable_path,
                              evidence["cuda"], evidence["vulkan"])
    runtime_label, runtime_version = _detect_runtime(
        executable_path, evidence["build"], backend)

    metadata = RunMetadata(
        source_path=source_path,
        started_at=header.get("started_at"),
        profile_name=header.get("profile_name"),
        model_path=model_path,
        model_name=_model_name(model_path) if model_path else UNKNOWN,
        executable_path=executable_path,
        runtime_label=runtime_label,
        runtime_version=runtime_version,
        backend=backend,
        kv_k=flags.get("kv_k") or UNKNOWN,
        kv_v=flags.get("kv_v") or UNKNOWN,
        reasoning=reasoning,
        reasoning_effort=reasoning_effort,
        configured_context=ctx,
        vision_loaded=vision_loaded,
        mmproj_path=mmproj_path,
        gpu_split=gpu_split,
        batch=int(flags["batch"]) if flags.get("batch") else None,
        ubatch=int(flags["ubatch"]) if flags.get("ubatch") else None,
        parallel=int(flags["parallel"]) if flags.get("parallel") else None,
        jinja=bool(flags.get("jinja", False)),
        extra_flags=flags.get("extra_flags", ()),
        vision_requested=vision_requested,
        vision_ready=vision_ready,
        startup_status=startup_status,
        startup_error_kind=startup_error_kind,
        warnings=tuple(warnings),
    )
    return ParsedLog(metadata=metadata, samples=samples,
                     warnings=tuple(warnings))


def parse_log_file(path: Path | str) -> ParsedLog:
    """Parse a log file. Bytes are decoded with errors='replace' so a bad
    byte never aborts the parse; callers get warnings, not crashes."""
    p = Path(path)
    raw = p.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    return parse_log_text(text, source_path=str(p))


def scan_log_paths(paths: Iterable[Path | str]) -> tuple[list[ParsedLog],
                                                         list[ScanError]]:
    """Parse many logs; one unreadable file never aborts the scan."""
    parsed: list[ParsedLog] = []
    errors: list[ScanError] = []
    for p in paths:
        sp = str(p)
        try:
            parsed.append(parse_log_file(p))
        except OSError as exc:
            errors.append(ScanError(path=sp, error=str(exc)))
        except Exception as exc:  # defensive: keep folder imports alive
            errors.append(ScanError(path=sp,
                                    error=f"{type(exc).__name__}: {exc}"))
    return parsed, errors


# --- aggregation -------------------------------------------------------------------

def bucket_index(used_context: int, bucket_size: int = 10_000) -> int:
    """Index of the context bucket a sample falls into (floor division)."""
    if bucket_size <= 0:
        raise ValueError(f"bucket_size must be positive, got {bucket_size}")
    return used_context // bucket_size


def percentile(values: Sequence[float], p: float) -> float:
    """Deterministic percentile with numpy "linear" interpolation.

    rank = (p/100)*(n-1); result x[lo] + (x[hi]-x[lo])*frac on the sorted
    values.  n=1 returns the value itself; empty input or p outside 0..100
    raises ValueError.
    """
    vals = sorted(values)
    if not vals:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile p must be within 0..100, got {p}")
    n = len(vals)
    if n == 1:
        return float(vals[0])
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(vals[lo] + (vals[hi] - vals[lo]) * frac)


@dataclass(frozen=True)
class Filters:
    """Optional pool filters for aggregate_runs; None values mean "no filter".

    ``min_generated`` is the only non-optional quality knob: it defaults to
    DEFAULT_MIN_GENERATED and is applied per sample, not per log.
    """
    model: str | None = None
    min_generated: int = DEFAULT_MIN_GENERATED
    backend: str | None = None
    runtime: str | None = None          # matches runtime_label
    kv: str | None = None               # "q4_0/q4_0" style string
    reasoning: str | None = None
    configured_context: int | None = None
    vision_loaded: bool | None = None
    gpu_split: str | None = None


@dataclass(frozen=True)
class MetricStats:
    n: int
    median: float
    p25: float
    p75: float
    min: float
    max: float
    mean: float
    anomaly_count: int


@dataclass(frozen=True)
class BucketStats:
    bucket_start: int
    bucket_end: int
    decode: MetricStats | None
    prefill: MetricStats | None


@dataclass(frozen=True)
class SeriesAggregation:
    dimension: str
    series_key: str
    buckets: tuple[BucketStats, ...]   # sorted, non-empty buckets only
    log_count: int
    sample_count: int                  # quality-passed samples


@dataclass(frozen=True)
class AggregationResult:
    series: tuple[SeriesAggregation, ...]
    warnings: tuple[str, ...]
    excluded_counts: dict


#: Aggregatable dimensions -> (series key, metadata fields compared by the
#: fairness check).  A dimension's own fields are exempt from fairness
#: warnings (that is what is being compared).
_DIMENSION_FIELDS: dict[str, tuple[str, ...]] = {
    "runtime": ("runtime_label",),
    "backend": ("backend",),
    "kv": ("kv_k", "kv_v"),
    "reasoning": ("reasoning",),
    "reasoning_effort": ("reasoning_effort",),
    "context": ("configured_context",),
    "vision": ("vision_loaded",),
    "run": (),
}

#: Metadata fields checked for cross-run fairness (all except the compared
#: dimension's own fields).
_FAIRNESS_FIELDS = (
    "model_name", "runtime_label", "backend", "kv_k", "kv_v",
    "reasoning", "reasoning_effort", "configured_context", "vision_loaded",
    "gpu_split", "batch", "ubatch",
)

#: Quality-filter exclusion counters (each sample counts at most once, at the
#: first failing check, in this order).
_EXCLUSION_KEYS = ("incomplete", "below_min_generated",
                   "missing_used_context")


def _basename(path: str) -> str:
    """Last path segment for both Windows and POSIX style paths."""
    if "\\" in path:
        return PureWindowsPath(path).name or path
    return PurePosixPath(path).name or path


def series_key(meta: RunMetadata, dimension: str) -> str:
    """Series key for one log under a comparison dimension.

    Public so exports and the viewer can map logs to series without
    duplicating the mapping rules.  Raises ValueError on unknown
    dimensions.
    """
    if dimension == "runtime":
        return meta.runtime_label
    if dimension == "backend":
        return meta.backend
    if dimension == "kv":
        return f"{meta.kv_k}/{meta.kv_v}"
    if dimension == "reasoning":
        return meta.reasoning
    if dimension == "reasoning_effort":
        return meta.reasoning_effort
    if dimension == "context":
        return (str(meta.configured_context)
                if meta.configured_context is not None else "unknown")
    if dimension == "vision":
        return "yes" if meta.vision_loaded else "no"
    if dimension == "run":
        return _basename(meta.source_path)
    raise ValueError(f"unknown aggregation dimension: {dimension!r}")


def _log_passes_filters(meta: RunMetadata, f: Filters) -> bool:
    if f.model is not None and meta.model_name != f.model:
        return False
    if f.backend is not None and meta.backend != f.backend:
        return False
    if f.runtime is not None and meta.runtime_label != f.runtime:
        return False
    if f.kv is not None and f"{meta.kv_k}/{meta.kv_v}" != f.kv:
        return False
    if f.reasoning is not None and meta.reasoning != f.reasoning:
        return False
    if (f.configured_context is not None
            and meta.configured_context != f.configured_context):
        return False
    if f.vision_loaded is not None and meta.vision_loaded != f.vision_loaded:
        return False
    if f.gpu_split is not None and meta.gpu_split != f.gpu_split:
        return False
    return True


def _metric_stats(pairs: list[tuple[float, bool]]) -> MetricStats | None:
    """Stats for one metric in one bucket; pairs are (value, is_anomaly)."""
    if not pairs:
        return None
    values = sorted(v for v, _ in pairs)
    return MetricStats(
        n=len(values),
        median=percentile(values, 50),
        p25=percentile(values, 25),
        p75=percentile(values, 75),
        min=float(values[0]),
        max=float(values[-1]),
        mean=sum(values) / len(values),
        anomaly_count=sum(1 for _, a in pairs if a),
    )


def _group_series(parsed_logs: Sequence[ParsedLog],
                  dimension: str) -> dict[str, list[ParsedLog]]:
    series: dict[str, list[ParsedLog]] = {}
    for p in parsed_logs:
        series.setdefault(series_key(p.metadata, dimension), []).append(p)
    return series


def fairness_warnings(parsed_logs: Sequence[ParsedLog], *,
                      dimension: str) -> tuple[str, ...]:
    """Flag configuration differences that make the comparison observational.

    For every non-compared metadata field:
    - a series with >= 2 logs whose values differ -> within-series warning;
    - every series uniform but the uniform values differ across series ->
      cross-series warning.  With dimension "run" each series holds one log,
      so only the cross-series check can fire.
    """
    compared = _DIMENSION_FIELDS.get(dimension)
    if compared is None:
        raise ValueError(f"unknown aggregation dimension: {dimension!r}")
    series = _group_series(parsed_logs, dimension)
    warnings: list[str] = []
    for field in _FAIRNESS_FIELDS:
        if field in compared:
            continue
        distinct: dict[str, set] = {
            key: {getattr(p.metadata, field) for p in logs}
            for key, logs in series.items()
        }
        for key in sorted(series):
            if len(series[key]) >= 2 and len(distinct[key]) > 1:
                vals = ", ".join(str(v)
                                 for v in sorted(distinct[key], key=str))
                warnings.append(
                    f"series '{key}' varies in field '{field}' across its "
                    f"runs: {{{vals}}}")
        if all(len(v) == 1 for v in distinct.values()) \
                and len(distinct) > 1:
            uniform = {k: next(iter(v)) for k, v in distinct.items()}
            if len({str(v) for v in uniform.values()}) > 1:
                pairs = ", ".join(f"{k}={uniform[k]}"
                                  for k in sorted(uniform))
                warnings.append(
                    f"field '{field}' differs across series: {pairs} — "
                    "observational, not a controlled comparison")
    return tuple(warnings)


def aggregate_runs(parsed_logs: Sequence[ParsedLog], *, dimension: str,
                   bucket_size: int = 10_000,
                   filters: Filters | None = None) -> AggregationResult:
    """Aggregate timing samples into per-series, per-context-bucket stats.

    Logs are first narrowed by ``filters``; samples that fail the quality
    filter (incomplete / below min generated / missing used context) are
    excluded and counted.  Decode and prefill stats are computed
    independently per bucket; empty buckets are not fabricated.  Missing
    buckets are left to the UI to render as gaps.
    """
    if dimension not in _DIMENSION_FIELDS:
        raise ValueError(f"unknown aggregation dimension: {dimension!r}")
    if bucket_size <= 0:
        raise ValueError(f"bucket_size must be positive, got {bucket_size}")
    f = filters or Filters()
    pool = [p for p in parsed_logs if _log_passes_filters(p.metadata, f)]

    warnings: list[str] = []
    if f.model is None and len({p.metadata.model_name
                                for p in pool}) > 1:
        warnings.append(
            "multiple models present without a model filter; series pool "
            "different models — select a model for a controlled comparison")
    warnings.extend(fairness_warnings(pool, dimension=dimension))

    series_logs = _group_series(pool, dimension)
    excluded_counts = {key: 0 for key in _EXCLUSION_KEYS}
    series_out: list[SeriesAggregation] = []
    for key in sorted(series_logs):
        logs = series_logs[key]
        buckets: dict[int, dict[str, list[tuple[float, bool]]]] = {}
        sample_count = 0
        for p in logs:
            for s in p.samples:
                if not s.completed:
                    excluded_counts["incomplete"] += 1
                    continue
                if (s.generated_tokens or 0) < f.min_generated:
                    excluded_counts["below_min_generated"] += 1
                    continue
                if s.used_context is None:
                    excluded_counts["missing_used_context"] += 1
                    continue
                sample_count += 1
                idx = bucket_index(s.used_context, bucket_size)
                per_metric = buckets.setdefault(
                    idx, {"decode": [], "prefill": []})
                for metric in ("decode", "prefill"):
                    value = s.decode_tps if metric == "decode" \
                        else s.prefill_tps
                    if value is None:
                        continue
                    anomaly = s.error_kind is not None or value <= 0
                    per_metric[metric].append((value, anomaly))
        bucket_stats = tuple(
            BucketStats(
                bucket_start=idx * bucket_size,
                bucket_end=(idx + 1) * bucket_size,
                decode=_metric_stats(buckets[idx]["decode"]),
                prefill=_metric_stats(buckets[idx]["prefill"]),
            )
            for idx in sorted(buckets))
        series_out.append(SeriesAggregation(
            dimension=dimension,
            series_key=key,
            buckets=bucket_stats,
            log_count=len(logs),
            sample_count=sample_count,
        ))
    return AggregationResult(series=tuple(series_out),
                             warnings=tuple(warnings),
                             excluded_counts=excluded_counts)
