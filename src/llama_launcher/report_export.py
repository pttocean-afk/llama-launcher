"""Self-contained report exports for performance aggregations.

Formats:
- HTML  — inline CSS + inline SVG charts, no CDN/script/external URL.
- SVG   — editable vector chart (median lines + P25-P75 band + per-bucket n).
- PNG   — the same chart rendered with Pillow (dark theme).
- Raw CSV    — one row per timing sample plus its run metadata.
- Aggregate CSV — one row per series/bucket/metric with the full stat set.
- Markdown — comparison table, key findings, fairness warnings, file list.

All render functions are pure (return text/bytes); ``export_all`` writes to
disk.  Output names carry model, comparison dimension, bucket size and a
timestamp; existing files are never overwritten (an incremented numeric
suffix is used instead).
"""
from __future__ import annotations

import csv
import html
import io
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from llama_launcher.log_analysis import (
    AggregationResult,
    BucketStats,
    MetricStats,
    ParsedLog,
    series_key,
)

__all__ = [
    "RAW_CSV_COLUMNS",
    "AGGREGATE_CSV_COLUMNS",
    "ChartPoint",
    "ChartModel",
    "chart_model",
    "format_point_label",
    "suggested_filename",
    "safe_output_path",
    "render_chart_svg",
    "render_chart_png",
    "render_html",
    "render_raw_csv",
    "render_aggregate_csv",
    "render_markdown",
    "export_all",
]

RAW_CSV_COLUMNS = (
    # run metadata (one record per log)
    "source_path", "started_at", "profile_name", "model_name",
    "executable_path", "runtime_label", "runtime_version", "backend",
    "kv_k", "kv_v", "reasoning", "reasoning_effort",
    "configured_context", "vision_loaded",
    "mmproj_path", "gpu_split", "batch", "ubatch", "parallel", "jinja",
    # timing sample fields
    "task_id", "slot_id", "used_context", "used_context_source",
    "prompt_tokens", "prompt_ms", "prefill_tps", "generated_tokens",
    "decode_ms", "decode_tps", "cache_source", "cached_tokens",
    "reprocessed_tokens", "actual_image_used", "completed", "error_kind",
    "line_start", "line_end",
)
_METADATA_COLUMNS = RAW_CSV_COLUMNS[:20]
_SAMPLE_COLUMNS = RAW_CSV_COLUMNS[20:]

AGGREGATE_CSV_COLUMNS = (
    "series", "bucket_start", "bucket_end", "metric",
    "n", "median", "p25", "p75", "min", "max", "mean", "anomaly_count",
)

#: Distinct series colors (dark-theme friendly).
PALETTE = [
    (77, 163, 255),    # blue
    (255, 112, 67),    # orange
    (102, 187, 106),   # green
    (171, 71, 188),    # purple
    (255, 202, 40),    # yellow
    (38, 198, 218),    # cyan
    (240, 98, 146),    # pink
    (156, 136, 255),   # lavender
]

_BG = (24, 24, 32)


# --- naming / collision safety ---------------------------------------------------

def _sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-")
    return s or "unknown"


def suggested_filename(model: str, dimension: str, bucket_size: int,
                       extension: str, *,
                       now: datetime | None = None) -> str:
    """Deterministic report file name: model-dimension-bucket-timestamp.ext."""
    stamp = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")
    return (f"{_sanitize(model)}-{_sanitize(dimension)}-"
            f"bucket{bucket_size}-{stamp}.{_sanitize(extension).lstrip('.')}"
            or f"report.{stamp}.{_sanitize(extension).lstrip('.')}")


def safe_output_path(path: Path | str) -> Path:
    """A path that will not clobber an existing file.

    ``report.csv`` taken -> ``report-1.csv`` -> ``report-2.csv`` ...
    """
    p = Path(path)
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    i = 1
    while True:
        cand = p.with_name(f"{stem}-{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


# --- geometry shared by SVG and PNG -------------------------------------------------

@dataclass(frozen=True)
class ChartPoint:
    x: int          # bucket midpoint (used-context tokens)
    median: float
    p25: float
    p75: float
    n: int


@dataclass(frozen=True)
class ChartModel:
    """Data extracted from an AggregationResult for one metric."""
    metric: str
    series: tuple[tuple[str, tuple[ChartPoint, ...]], ...]
    x_max: int
    y_max: float

    @property
    def has_data(self) -> bool:
        return any(pts for _, pts in self.series)


def format_point_label(value: float) -> str:
    """Point label on charts: the t/s value itself (1 decimal < 100,
    integer >= 100) instead of a separate sample count."""
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def chart_model(result: AggregationResult, metric: str) -> ChartModel:
    series: list[tuple[str, tuple[ChartPoint, ...]]] = []
    x_max = 0
    y_max = 0.0
    for s in result.series:
        pts: list[ChartPoint] = []
        for b in s.buckets:
            st = b.decode if metric == "decode" else b.prefill
            if st is None:
                continue
            pts.append(ChartPoint(x=(b.bucket_start + b.bucket_end) // 2,
                              median=st.median, p25=st.p25, p75=st.p75,
                              n=st.n))
            x_max = max(x_max, b.bucket_end)
            y_max = max(y_max, st.p75, st.median)
        series.append((s.series_key, tuple(pts)))
    if not series or not any(pts for _, pts in series):
        x_max, y_max = 10_000, 1.0
    return ChartModel(metric=metric, series=tuple(series),
                       x_max=x_max, y_max=_nice_ceil(y_max * 1.15 or 1.0))


def _nice_ceil(v: float) -> float:
    if v <= 0:
        return 1.0
    exp = math.floor(math.log10(v))
    base = 10.0 ** exp
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * base >= v:
            return m * base
    return 10.0 * base


def _fmt_k(v: int) -> str:
    return "0" if v == 0 else f"{v / 1000:g}K"


def _blend(c: tuple[int, int, int], bg: tuple[int, int, int],
           t: float) -> tuple[int, int, int]:
    return tuple(int(a + (b - a) * t) for a, b in zip(c, bg))


def _css(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


# --- SVG ---------------------------------------------------------------------------

def render_chart_svg(result: AggregationResult, *, metric: str = "decode",
                     title: str | None = None, bucket_size: int = 10_000,
                     width: int = 860, height: int = 460) -> str:
    """One metric as an editable vector chart (median + P25-P75 + n labels)."""
    model = chart_model(result, metric)
    t = html.escape(title or f"{metric} t/s vs used context")
    ylabel = f"{metric} t/s"
    xlab = f"used context ({bucket_size / 1000:g}K buckets)"
    L, R, T, B = 64, 150, 48, 56
    pw, ph = width - L - R, height - T - B

    def X(tok: int) -> float:
        return L + pw * tok / max(model.x_max, 1)

    def Y(v: float) -> float:
        return (T + ph) - ph * v / model.y_max

    o: list[str] = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" viewBox="0 0 {width} {height}" '
             'font-family="Segoe UI, Arial, sans-serif" '
             f'style="background:{_css(_BG)}">')
    o.append(f'<text x="{width / 2}" y="26" fill="#e8e8f0" font-size="16" '
             f'text-anchor="middle">{t}</text>')
    # axes
    o.append(f'<line x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}" '
             'stroke="#888" />')
    o.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T + ph}" '
             'stroke="#888" />')
    # x ticks at bucket boundaries
    for b in range(0, model.x_max + 1, bucket_size):
        x = X(b)
        o.append(f'<line x1="{x:.1f}" y1="{T + ph}" x2="{x:.1f}" '
                 f'y2="{T + ph + 5}" stroke="#888" />')
        o.append(f'<text x="{x:.1f}" y="{T + ph + 18}" fill="#bbb" '
                 f'font-size="10" text-anchor="middle">{_fmt_k(b)}</text>')
    # y ticks (5 divisions)
    for i in range(6):
        v = model.y_max * i / 5
        y = Y(v)
        o.append(f'<line x1="{L - 5}" y1="{y:.1f}" x2="{L}" y2="{y:.1f}" '
                 'stroke="#888" />')
        o.append(f'<text x="{L - 8}" y="{y + 3.5:.1f}" fill="#bbb" '
                 f'font-size="10" text-anchor="end">{v:g}</text>')
    # axis titles
    o.append(f'<text x="{L + pw / 2}" y="{height - 12}" fill="#e8e8f0" '
             f'font-size="12" text-anchor="middle">{html.escape(xlab)}</text>')
    o.append(f'<text x="16" y="{T + ph / 2}" fill="#e8e8f0" font-size="12" '
             f'text-anchor="middle" '
             f'transform="rotate(-90 16 {T + ph / 2})">'
             f'{html.escape(ylabel)}</text>')

    if not model.has_data:
        o.append(f'<text x="{L + pw / 2}" y="{T + ph / 2}" fill="#999" '
                 f'font-size="14" text-anchor="middle">no data</text>')
    else:
        for i, (key, pts) in enumerate(model.series):
            color = _css(PALETTE[i % len(PALETTE)])
            if len(pts) >= 2:
                top = " ".join(f"{X(p.x):.1f},{Y(p.p75):.1f}" for p in pts)
                bot = " ".join(f"{X(p.x):.1f},{Y(p.p25):.1f}"
                               for p in reversed(pts))
                o.append(f'<polygon points="{top} {bot}" fill="{color}" '
                         'fill-opacity="0.14" stroke="none" />')
            line_pts = " ".join(f"{X(p.x):.1f},{Y(p.median):.1f}"
                                for p in pts)
            if len(pts) >= 2:
                o.append(f'<polyline points="{line_pts}" fill="none" '
                         f'stroke="{color}" stroke-width="2" />')
            for p in pts:
                x, y = X(p.x), Y(p.median)
                o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" '
                         f'fill="{color}" />')
                o.append(f'<text x="{x:.1f}" y="{y - 8:.1f}" fill="{color}" '
                         f'font-size="10" text-anchor="middle">'
                         f'{format_point_label(p.median)}</text>')
        # legend
        lx = L + pw + 16
        ly = T + 8
        o.append(f'<text x="{lx}" y="{ly - 4}" fill="#e8e8f0" '
                 'font-size="11">series</text>')
        for i, (key, pts) in enumerate(model.series):
            color = _css(PALETTE[i % len(PALETTE)])
            label = key if len(key) <= 28 else key[:27] + "…"
            o.append(f'<rect x="{lx}" y="{ly + 4}" width="12" height="12" '
                     f'fill="{color}" />')
            o.append(f'<text x="{lx + 18}" y="{ly + 14}" fill="#ddd" '
                     f'font-size="11">{html.escape(label)}</text>')
            ly += 20
    o.append("</svg>")
    return "\n".join(o)


# --- PNG (Pillow) --------------------------------------------------------------------

def _default_font(size: int = 11):
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, ValueError):
        return ImageFont.load_default()


def render_chart_png(result: AggregationResult, *, metric: str = "decode",
                     title: str | None = None, bucket_size: int = 10_000,
                     width: int = 860, height: int = 460) -> bytes:
    """The same chart as render_chart_svg, rasterized with Pillow."""
    model = chart_model(result, metric)
    img = Image.new("RGB", (width, height), _BG)
    d = ImageDraw.Draw(img)
    f10 = _default_font(10)
    f12 = _default_font(12)
    f16 = _default_font(16)
    t = title or f"{metric} t/s vs used context"
    xlab = f"used context ({bucket_size / 1000:g}K buckets)"
    L, R, T, B = 64, 150, 48, 56
    pw, ph = width - L - R, height - T - B

    def X(tok: int) -> float:
        return L + pw * tok / max(model.x_max, 1)

    def Y(v: float) -> float:
        return (T + ph) - ph * v / model.y_max

    d.text((width / 2, 22), t, fill=(232, 232, 240), font=f16,
           anchor="mm")
    d.line([(L, T + ph), (L + pw, T + ph)], fill=(136, 136, 136), width=1)
    d.line([(L, T), (L, T + ph)], fill=(136, 136, 136), width=1)
    for b in range(0, model.x_max + 1, bucket_size):
        x = X(b)
        d.line([(x, T + ph), (x, T + ph + 5)], fill=(136, 136, 136))
        d.text((x, T + ph + 16), _fmt_k(b), fill=(187, 187, 187),
               font=f10, anchor="mm")
    for i in range(6):
        v = model.y_max * i / 5
        y = Y(v)
        d.line([(L - 5, y), (L, y)], fill=(136, 136, 136))
        d.text((L - 8, y), f"{v:g}", fill=(187, 187, 187), font=f10,
               anchor="rm")
    d.text((L + pw / 2, height - 12), xlab, fill=(232, 232, 240),
           font=f12, anchor="mm")
    d.text((16, T + ph / 2), f"{metric} t/s", fill=(232, 232, 240),
           font=f12, anchor="mm",
           )
    if not model.has_data:
        d.text((L + pw / 2, T + ph / 2), "no data", fill=(153, 153, 153),
               font=f12, anchor="mm")
    else:
        for i, (key, pts) in enumerate(model.series):
            color = PALETTE[i % len(PALETTE)]
            if len(pts) >= 2:
                poly = ([ (X(p.x), Y(p.p75)) for p in pts ]
                        + [ (X(p.x), Y(p.p25)) for p in reversed(pts) ])
                d.polygon(poly, fill=_blend(color, _BG, 0.86))
            if pts:
                if len(pts) >= 2:
                    d.line([ (X(p.x), Y(p.median)) for p in pts ],
                           fill=color, width=2)
                for p in pts:
                    x, y = X(p.x), Y(p.median)
                    d.ellipse([x - 3.5, y - 3.5, x + 3.5, y + 3.5],
                              fill=color)
                    d.text((x, y - 11), format_point_label(p.median),
                           fill=color, font=f10, anchor="mm")
            label = key if len(key) <= 28 else key[:27] + "…"
            ly = T + 12 + 20 * i
            d.rectangle([L + pw + 16, ly, L + pw + 28, ly + 12],
                        fill=color)
            d.text((L + pw + 34, ly + 6), label, fill=(221, 221, 221),
                   font=f12, anchor="lm")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- CSV ---------------------------------------------------------------------------

def _cell(value) -> str:
    return "" if value is None else str(value)


def render_raw_csv(parsed_logs: Sequence[ParsedLog]) -> str:
    """One row per timing sample, prefixed with its run metadata."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(RAW_CSV_COLUMNS)
    for p in parsed_logs:
        meta_cells = [getattr(p.metadata, c) for c in _METADATA_COLUMNS]
        for s in p.samples:
            w.writerow([_cell(v) for v in meta_cells]
                       + [_cell(getattr(s, c)) for c in _SAMPLE_COLUMNS])
    return buf.getvalue()


def render_aggregate_csv(result: AggregationResult) -> str:
    """One row per series/bucket/metric with measured stats only."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(AGGREGATE_CSV_COLUMNS)
    for s in result.series:
        for b in s.buckets:
            for metric, st in (("decode", b.decode), ("prefill", b.prefill)):
                if st is None:
                    continue
                w.writerow([s.series_key, b.bucket_start, b.bucket_end,
                            metric, st.n, _cell(st.median), _cell(st.p25),
                            _cell(st.p75), _cell(st.min), _cell(st.max),
                            _cell(st.mean), st.anomaly_count])
    return buf.getvalue()


# --- pool helpers ------------------------------------------------------------------

def _pool_model_label(parsed_logs: Sequence[ParsedLog]) -> str:
    models = sorted({p.metadata.model_name for p in parsed_logs})
    return models[0] if len(models) == 1 else "multiple-models"


def _series_config_rows(parsed_logs: Sequence[ParsedLog],
                        dimension: str) -> list[tuple[str, list[str]]]:
    """Per series: (key, formatted non-compared configuration values)."""
    by_key: dict[str, list] = {}
    for p in parsed_logs:
        by_key.setdefault(series_key(p.metadata, dimension), []).append(
            p.metadata)
    fields = ("model_name", "runtime_label", "backend",
              "kv", "reasoning", "reasoning_effort", "configured_context",
              "vision_loaded", "gpu_split", "batch", "ubatch")
    out: list[tuple[str, list[str]]] = []
    for key in sorted(by_key):
        metas = by_key[key]
        cells = []
        for f in fields:
            vals = set()
            for m in metas:
                if f == "kv":
                    vals.add(f"{m.kv_k}/{m.kv_v}")
                elif f == "vision_loaded":
                    vals.add("yes" if m.vision_loaded else "no")
                else:
                    v = getattr(m, f)
                    vals.add("unknown" if v is None else str(v))
            if len(vals) == 1:
                cells.append(next(iter(vals)))
            else:
                cells.append("varies: "
                             + ", ".join(sorted(vals, key=str)))
        out.append((key, cells))
    return out


def _series_config_header() -> list[str]:
    return ("series", "model", "runtime", "backend", "KV", "reasoning",
            "effort", "max ctx", "vision", "split", "batch", "ubatch")


# --- HTML --------------------------------------------------------------------------

_HTML_CSS = """
body { background:#181820; color:#e8e8f0; margin:24px;
       font-family:'Segoe UI',Arial,sans-serif; }
h1 { font-size:20px; } h2 { font-size:15px; margin-top:24px; }
table { border-collapse:collapse; margin-top:8px; }
td,th { border:1px solid #3a3a48; padding:4px 10px; font-size:12px; }
th { background:#23232e; }
.warn { color:#ffca28; font-size:12px; }
.ok { color:#66bb6a; font-size:12px; }
.meta { color:#9a9aa8; font-size:12px; }
svg { background:#181820; }
"""


def render_html(result: AggregationResult, *,
                parsed_logs: Sequence[ParsedLog], dimension: str,
                bucket_size: int = 10_000,
                now: datetime | None = None) -> str:
    """Self-contained HTML report: inline CSS + inline SVG charts only."""
    ts = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    model_label = _pool_model_label(parsed_logs)
    n_logs = len(parsed_logs)
    n_samples = sum(len(p.samples) for p in parsed_logs)
    n_passed = sum(s.sample_count for s in result.series)
    excl = ", ".join(f"{k} {v}" for k, v in result.excluded_counts.items())

    o: list[str] = []
    o.append("<!DOCTYPE html>")
    o.append('<html lang="en">')
    o.append("<head>")
    o.append('<meta charset="utf-8">')
    o.append(f"<title>Performance report — {html.escape(model_label)} · "
             f"{html.escape(dimension)}</title>")
    o.append(f"<style>{_HTML_CSS}</style>")
    o.append("</head>")
    o.append("<body>")
    o.append(f"<h1>Performance report — {html.escape(model_label)} · "
             f"compare by {html.escape(dimension)}</h1>")
    o.append(f'<p class="meta">Generated {ts} · {n_logs} logs / '
             f"{n_samples} samples · quality-passed {n_passed} · "
             f"excluded: {excl or 'none'}</p>")

    o.append("<h2>Fairness warnings</h2>")
    if result.warnings:
        o.append('<ul class="warn">')
        for w in result.warnings:
            o.append(f"<li>⚠ {html.escape(w)}</li>")
        o.append("</ul>")
    else:
        o.append('<p class="ok">No fairness warnings: series pool is '
                 "uniform in all non-compared fields.</p>")

    o.append("<h2>Series configuration</h2>")
    o.append("<table>")
    o.append("<tr>" + "".join(f"<th>{h}</th>"
                              for h in _series_config_header())
             + "</tr>")
    for key, cells in _series_config_rows(parsed_logs, dimension):
        o.append("<tr>" + f"<td>{html.escape(key)}</td>"
                 + "".join(f"<td>{html.escape(c)}</td>" for c in cells)
                 + "</tr>")
    o.append("</table>")

    for metric, mtitle in (("decode", "Decode t/s vs used context"),
                           ("prefill", "Prefill t/s vs used context")):
        o.append(f"<h2>{mtitle}</h2>")
        o.append(render_chart_svg(result, metric=metric,
                                  bucket_size=bucket_size))

    o.append(f"<h2>Summary ({bucket_size / 1000:g}K buckets)</h2>")
    o.append("<table>")
    o.append("<tr><th>series</th><th>bucket</th><th>metric</th><th>n</th>"
             "<th>median</th><th>p25</th><th>p75</th><th>min</th>"
             "<th>max</th></tr>")
    for s in result.series:
        for b in s.buckets:
            for metric, st in (("decode", b.decode), ("prefill", b.prefill)):
                if st is None:
                    continue
                o.append(
                    f"<tr><td>{html.escape(s.series_key)}</td>"
                    f"<td>{b.bucket_start}-{b.bucket_end}</td>"
                    f"<td>{metric}</td><td>{st.n}</td>"
                    f"<td>{st.median:g}</td><td>{st.p25:g}</td>"
                    f"<td>{st.p75:g}</td><td>{st.min:g}</td>"
                    f"<td>{st.max:g}</td></tr>")
    o.append("</table>")
    o.append("</body>")
    o.append("</html>")
    return "\n".join(o)


# --- Markdown ------------------------------------------------------------------------

def render_markdown(result: AggregationResult, *,
                    parsed_logs: Sequence[ParsedLog], dimension: str,
                    bucket_size: int = 10_000,
                    exported_files: Sequence[str] = (),
                    now: datetime | None = None) -> str:
    ts = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    model_label = _pool_model_label(parsed_logs)
    n_logs = len(parsed_logs)
    n_samples = sum(len(p.samples) for p in parsed_logs)
    n_passed = sum(s.sample_count for s in result.series)
    excl = ", ".join(f"{k} {v}" for k, v in result.excluded_counts.items())

    o: list[str] = []
    o.append(f"# Performance report — {model_label} · compare by "
             f"{dimension}")
    o.append("")
    o.append(f"- Generated: {ts}")
    o.append(f"- Pool: {n_logs} logs, {n_samples} samples")
    o.append(f"- Quality-passed: {n_passed} (excluded: {excl or 'none'})")
    o.append(f"- Bucket size: {bucket_size} used tokens")
    o.append("")
    o.append("## Fairness warnings")
    o.append("")
    if result.warnings:
        for w in result.warnings:
            o.append(f"- ⚠ {w}")
    else:
        o.append("- none")
    o.append("")
    o.append(f"## Comparison ({bucket_size / 1000:g}K buckets)")
    o.append("")
    o.append("| series | bucket | metric | n | median | p25 | p75 "
             "| min | max |")
    o.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for s in result.series:
        for b in s.buckets:
            for metric, st in (("decode", b.decode), ("prefill", b.prefill)):
                if st is None:
                    continue
                o.append(
                    f"| {s.series_key} | {b.bucket_start}-{b.bucket_end} "
                    f"| {metric} | {st.n} | {st.median:g} | {st.p25:g} "
                    f"| {st.p75:g} | {st.min:g} | {st.max:g} |")
    o.append("")
    # key findings: best median per bucket across series (decode)
    findings: list[str] = []
    per_bucket: dict[tuple[int, int], tuple[str, float]] = {}
    for s in result.series:
        for b in s.buckets:
            if b.decode is None:
                continue
            k = (b.bucket_start, b.bucket_end)
            cur = per_bucket.get(k)
            if cur is None or b.decode.median > cur[1]:
                per_bucket[k] = (s.series_key, b.decode.median)
    for k in sorted(per_bucket):
        key, med = per_bucket[k]
        findings.append(f"- fastest decode at {k[0]}-{k[1]}: "
                        f"{key} ({med:g} t/s)")
    o.append("## Key findings")
    o.append("")
    if findings:
        o.extend(findings[:10])
    else:
        o.append("- no quality-passed samples")
    o.append("")
    o.append("## Exported files")
    o.append("")
    for f in exported_files:
        o.append(f"- {f}")
    o.append("")
    return "\n".join(o)


# --- bundle ---------------------------------------------------------------------------

def export_all(result: AggregationResult,
               parsed_logs: Sequence[ParsedLog], *, dimension: str,
               bucket_size: int = 10_000, out_dir: Path | str,
               now: datetime | None = None) -> dict[str, Path]:
    """Write all six export formats; never overwrites existing files.

    Returns {format: final_path}.  File names carry model, dimension,
    bucket size and timestamp.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_label = _pool_model_label(parsed_logs)
    stem = suggested_filename(model_label, dimension, bucket_size, "x",
                              now=now)
    stem = stem[:-len(".x")]  # keep model-dimension-bucket-timestamp

    def _write(name: str, text: str) -> Path:
        p = safe_output_path(out / name)
        p.write_text(text, encoding="utf-8")
        return p

    def _write_bytes(name: str, data: bytes) -> Path:
        p = safe_output_path(out / name)
        p.write_bytes(data)
        return p

    html_path = _write(f"{stem}.html",
                       render_html(result, parsed_logs=parsed_logs,
                                   dimension=dimension,
                                   bucket_size=bucket_size, now=now))
    svg_path = _write(f"{stem}.svg",
                      render_chart_svg(result, metric="decode",
                                       bucket_size=bucket_size))
    png_path = _write_bytes(f"{stem}.png",
                            render_chart_png(result, metric="decode",
                                             bucket_size=bucket_size))
    raw_path = _write(f"{stem}-raw.csv", render_raw_csv(parsed_logs))
    agg_path = _write(f"{stem}-aggregate.csv",
                      render_aggregate_csv(result))
    md_path = _write(
        f"{stem}.md",
        render_markdown(result, parsed_logs=parsed_logs, dimension=dimension,
                        bucket_size=bucket_size,
                        exported_files=[html_path.name, svg_path.name,
                                        png_path.name, raw_path.name,
                                        agg_path.name],
                        now=now))
    return {"html": html_path, "svg": svg_path, "png": png_path,
            "raw_csv": raw_path, "aggregate_csv": agg_path,
            "markdown": md_path}
