"""Export engine tests for llama_launcher.report_export (TDD task 5).

Exports must be self-contained (no CDN/script/external URLs), deterministic,
and never overwrite existing files.
"""
from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET
from datetime import datetime

from PIL import Image

from llama_launcher.log_analysis import aggregate_runs
from llama_launcher.report_export import (
    AGGREGATE_CSV_COLUMNS,
    RAW_CSV_COLUMNS,
    export_all,
    format_point_label,
    render_aggregate_csv,
    render_chart_png,
    render_chart_svg,
    render_html,
    render_markdown,
    render_raw_csv,
    safe_output_path,
    suggested_filename,
)

from test_log_analysis import _meta, _parsed, _sample

NOW = datetime(2026, 8, 30, 21, 30, 0)


def _two_series_pool():
    """Two same-model logs, different backend/runtime/KV/ctx/vision:
    2 logs, 5 samples, all quality-passed; backend is the compared
    dimension so several fairness warnings must fire."""
    a = _parsed(
        _meta(source_path=r"C:\logs\a.log", backend="cuda",
              runtime_label="b10621", model_name="Model",
              model_path=r"C:\models\Model.gguf"),
        [_sample(task_id=1, used_context=2550, decode_tps=38.2,
                 prefill_tps=1195.5),
         _sample(task_id=2, used_context=5000, decode_tps=30.0,
                 prefill_tps=None),
         _sample(task_id=3, used_context=20000, decode_tps=25.0,
                 prefill_tps=900.0)])
    b = _parsed(
        _meta(source_path=r"C:\logs\b.log", backend="vulkan",
              runtime_label="b10509", kv_k="unknown", kv_v="unknown",
              model_name="Model", model_path=r"C:\models\Model.gguf",
              configured_context=262144, vision_loaded=False,
              mmproj_path=None),
        [_sample(task_id=4, used_context=3000, decode_tps=20.0,
                 prefill_tps=500.0),
         _sample(task_id=5, used_context=15000, decode_tps=18.0,
                 prefill_tps=450.0)])
    return [a, b]


# ---------------------------------------------------------------------------
# file naming / collision safety
# ---------------------------------------------------------------------------

def test_suggested_filename_contents():
    name = suggested_filename("Qwen 3.8/27B", "backend", 10_000, "csv",
                              now=NOW)
    assert name == "Qwen-3.8-27B-backend-bucket10000-20260830T213000.csv"


def test_safe_output_path_never_overwrites(tmp_path):
    p = tmp_path / "report.csv"
    assert safe_output_path(p) == p  # free
    p.write_text("x", encoding="utf-8")
    p2 = safe_output_path(p)
    assert p2 != p and p2.name == "report-1.csv"
    p2.write_text("y", encoding="utf-8")
    p3 = safe_output_path(p)
    assert p3.name == "report-2.csv"


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

def test_raw_csv_headers_and_rows():
    pool = [
        _parsed(_meta(source_path=r"C:\logs\a.log"),
                [_sample(), _sample(task_id=5)]),
        _parsed(_meta(source_path=r"C:\logs\b.log", backend="vulkan"),
                [_sample(task_id=7)]),
    ]
    rows = list(csv.reader(io.StringIO(render_raw_csv(pool))))
    assert rows[0] == list(RAW_CSV_COLUMNS)
    assert len(rows) == 1 + 3  # one row per sample
    row = dict(zip(rows[0], rows[1]))
    assert row["source_path"] == r"C:\logs\a.log"
    assert row["task_id"] == "4"
    assert row["decode_tps"] == "38.2"
    assert row["completed"] == "True"
    # None fields render as empty cells
    assert row["error_kind"] == ""


def test_aggregate_csv_rows():
    pool = [
        _parsed(_meta(source_path=r"C:\logs\a.log"),
                [_sample(used_context=2550, decode_tps=38.2,
                         prefill_tps=1195.5),
                 _sample(used_context=20000, decode_tps=25.0,
                         prefill_tps=None)]),
    ]
    result = aggregate_runs(pool, dimension="run")
    rows = list(csv.reader(io.StringIO(render_aggregate_csv(result))))
    assert rows[0] == list(AGGREGATE_CSV_COLUMNS)
    # bucket 0: decode + prefill; bucket 20000: decode only
    assert len(rows) == 1 + 3
    first = dict(zip(rows[0], rows[1]))
    assert first["series"] == "a.log"
    assert first["bucket_start"] == "0"
    assert first["metric"] == "decode"
    assert first["n"] == "1"
    assert first["median"] == "38.2"


# ---------------------------------------------------------------------------
# chart rendering
# ---------------------------------------------------------------------------

def test_svg_valid_xml_with_labels():
    pool = _two_series_pool()
    result = aggregate_runs(pool, dimension="backend")
    svg = render_chart_svg(result, metric="decode", title="Decode curve")
    root = ET.fromstring(svg)  # valid XML
    assert "cuda" in svg and "vulkan" in svg
    assert "decode t/s" in svg
    assert "used context" in svg
    assert "Decode curve" in svg
    assert "<polyline" in svg  # median lines
    assert "<polygon" in svg   # P25-P75 bands


def test_single_point_series_renders_without_lines():
    # a series with exactly one used bucket must not crash any renderer —
    # lines need >= 2 points, but the point + t/s label stay visible
    pool = [_parsed(
        _meta(runtime_label="b10621", backend="cuda",
              model_name="Model", kv_k="q4_0", kv_v="q4_0",
              configured_context=262144),
        [_sample(task_id=1, used_context=19950,
                 decode_tps=19.8,
                 prefill_tps=1200.0)])]
    result = aggregate_runs(pool, dimension="runtime")
    svg = render_chart_svg(result, metric="decode")
    ET.fromstring(svg)  # still valid XML
    assert "19.8" in svg  # t/s label present, no n= label
    assert "n=" not in svg
    png = render_chart_png(result, metric="decode")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_png_signature_and_dimensions():
    pool = _two_series_pool()
    result = aggregate_runs(pool, dimension="backend")
    data = render_chart_png(result, metric="decode",
                            width=860, height=460)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(data))
    assert img.size == (860, 460)


def test_chart_renders_with_no_data():
    pool = [_parsed(_meta(), [_sample(completed=False)])]
    result = aggregate_runs(pool, dimension="run")
    svg = render_chart_svg(result, metric="decode")
    ET.fromstring(svg)  # still valid XML
    data = render_chart_png(result, metric="decode")
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

def test_html_self_contained():
    pool = _two_series_pool()
    result = aggregate_runs(pool, dimension="backend")
    html = render_html(result, parsed_logs=pool, dimension="backend",
                       bucket_size=10_000, now=NOW)
    assert "<style" in html          # inline CSS
    assert "<svg" in html            # inline charts
    assert "cuda" in html and "vulkan" in html
    # no external dependency of any kind
    assert "<script" not in html
    assert "<link" not in html
    assert 'src="http' not in html
    assert 'href="http' not in html
    assert "@import" not in html
    # fairness warnings visible
    assert "differs across series" in html
    assert "kv" in html
    # data summary
    assert "2 logs" in html
    assert "5 samples" in html


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------

def test_markdown_contains_warnings_and_counts():
    pool = _two_series_pool()
    result = aggregate_runs(pool, dimension="backend")
    md = render_markdown(result, parsed_logs=pool, dimension="backend",
                         bucket_size=10_000,
                         exported_files=["report.html", "report.png"],
                         now=NOW)
    assert "differs across series" in md   # fairness warnings
    assert "2 logs" in md
    assert "5 samples" in md
    assert "| series |" in md              # comparison table
    assert "report.html" in md and "report.png" in md


# ---------------------------------------------------------------------------
# end-to-end export bundle
# ---------------------------------------------------------------------------

def test_export_all_writes_six_formats(tmp_path):
    pool = _two_series_pool()
    result = aggregate_runs(pool, dimension="backend", bucket_size=10_000)
    files = export_all(result, pool, dimension="backend",
                       bucket_size=10_000, out_dir=tmp_path, now=NOW)
    assert set(files) == {"html", "svg", "png", "raw_csv",
                          "aggregate_csv", "markdown"}
    for p in files.values():
        assert p.exists()
    assert files["html"].suffix == ".html"
    assert files["png"].suffix == ".png"
    assert files["markdown"].suffix == ".md"
    # names carry model, dimension and bucket size
    name = files["html"].name
    assert "Model" in name and "backend" in name and "bucket10000" in name
    # each format is structurally sound
    html = files["html"].read_text(encoding="utf-8")
    assert "<svg" in html and "<script" not in html
    ET.fromstring(files["svg"].read_text(encoding="utf-8"))
    assert files["png"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    raw_rows = list(csv.reader(
        io.StringIO(files["raw_csv"].read_text(encoding="utf-8"))))
    assert len(raw_rows) == 1 + 5
    agg_rows = list(csv.reader(
        io.StringIO(files["aggregate_csv"].read_text(encoding="utf-8"))))
    assert agg_rows[0] == list(AGGREGATE_CSV_COLUMNS)


def test_format_point_label_shows_value_not_n():
    assert format_point_label(19.84) == "19.8"
    assert format_point_label(38.2) == "38.2"
    assert format_point_label(125.0) == "125"
    assert format_point_label(15316.7) == "15317"
