#!/usr/bin/env python3
"""Non-destructive validation of llama_launcher.log_analysis against real logs.

This is a development tool, not part of the package: every path is a CLI
argument (no private paths are baked in as defaults), and it only ever
READS log files.  Run it with the repo venv:

    ./.venv-linux-system/bin/python scripts/validate_log_analysis.py scan DIR [DIR ...]
        Parse every *.log in each directory and report file/sample/warning
        counts, per-directory and in total.

    ./.venv-linux-system/bin/python scripts/validate_log_analysis.py \
        compare REF_JSON LOG [--tolerance 0.01]
        Compare a report JSON with a "points" array (per-point prompt/decode
        tps + generated tokens) against one parsed log, matching samples in
        task order.  Exits non-zero on any mismatch.

    ./.venv-linux-system/bin/python scripts/validate_log_analysis.py \
        curve LOG [--bucket-size 10000] [--min-generated 20]
        Print the per-bucket decode/prefill median table for one log so the
        numbers can be eyeballed against report documents.

    ./.venv-linux-system/bin/python scripts/validate_log_analysis.py \
        expect LOG KEY=VALUE [KEY=VALUE ...]
        Assert metadata fields of one log (RunMetadata attribute names).
        Values are parsed as bool/None/int when they look like them.

Exit codes: 0 = all checks passed, 1 = at least one check failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llama_launcher.log_analysis import (  # noqa: E402
    Filters,
    aggregate_runs,
    parse_log_file,
)

_METADATA_KEYS = {
    "source_path", "started_at", "profile_name", "model_path", "model_name",
    "executable_path", "runtime_label", "runtime_version", "backend",
    "kv_k", "kv_v", "reasoning", "configured_context", "vision_loaded",
    "mmproj_path", "gpu_split", "batch", "ubatch", "parallel", "jinja",
}


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        return value


def cmd_scan(args: argparse.Namespace) -> int:
    grand_files = grand_parsed = grand_errors = 0
    grand_samples = grand_completed = grand_incomplete = 0
    grand_error_kind = grand_warnings = 0
    ok = True
    for d in args.dirs:
        path = Path(d)
        if not path.is_dir():
            print(f"ERROR: not a directory: {d}")
            return 1
        files = sorted(path.glob("*.log"))
        parsed = errors = 0
        samples = completed = incomplete = error_kind = 0
        warning_counter: Counter[str] = Counter()
        for f in files:
            try:
                p = parse_log_file(f)
            except Exception as exc:  # defensive; parser should not raise
                errors += 1
                print(f"  PARSE-ERROR {f.name}: {type(exc).__name__}: {exc}")
                continue
            parsed += 1
            for s in p.samples:
                samples += 1
                if s.completed:
                    completed += 1
                else:
                    incomplete += 1
                if s.error_kind:
                    error_kind += 1
            warning_counter.update(p.warnings)
        total_warnings = sum(warning_counter.values())
        grand_files += len(files)
        grand_parsed += parsed
        grand_errors += errors
        grand_samples += samples
        grand_completed += completed
        grand_incomplete += incomplete
        grand_error_kind += error_kind
        grand_warnings += total_warnings
        if errors or parsed == 0:
            ok = False
        print(f"== {path}")
        print(f"   files: {len(files)}  parsed: {parsed}  parse-errors: {errors}")
        print(f"   samples: {samples}  completed: {completed}  "
              f"incomplete: {incomplete}  error_kind: {error_kind}")
        print(f"   warnings: {total_warnings} "
              f"(distinct: {len(warning_counter)})")
        for msg, n in warning_counter.most_common(5):
            print(f"     {n:4d}  {msg[:150]}")
    print(f"== TOTAL files: {grand_files} parsed: {grand_parsed} "
          f"parse-errors: {grand_errors} samples: {grand_samples} "
          f"completed: {grand_completed} incomplete: {grand_incomplete} "
          f"error_kind: {grand_error_kind} warnings: {grand_warnings}")
    return 0 if ok else 1


def cmd_compare(args: argparse.Namespace) -> int:
    ref = json.loads(Path(args.ref_json).read_text(encoding="utf-8"))
    points = ref.get("points")
    if not points:
        print(f"ERROR: {args.ref_json} has no 'points' array")
        return 1
    parsed = parse_log_file(args.log)
    samples = parsed.samples
    print(f"ref: {args.ref_json} ({len(points)} points)")
    print(f"log: {args.log} ({len(samples)} samples)")
    if len(samples) < len(points):
        print("FAIL: fewer samples than reference points")
        return 1
    ok = True
    for i, (point, s) in enumerate(zip(points, samples)):
        checks = [
            ("decode_tps", point.get("decode_tps"), s.decode_tps),
            ("prompt_tps", point.get("prompt_tps"), s.prefill_tps),
            ("generated_tokens", point.get("generated_tokens"),
             s.generated_tokens),
        ]
        row_ok = True
        parts = []
        for name, expected, actual in checks:
            if expected is None:
                continue
            if actual is None:
                parts.append(f"{name}: MISSING")
                row_ok = False
                continue
            if isinstance(expected, (int, float)) and \
                    isinstance(actual, (int, float)):
                if expected == 0:
                    good = actual == 0
                    delta = 0.0
                else:
                    delta = abs(actual - expected) / abs(expected)
                    good = delta <= args.tolerance
            else:
                good = expected == actual
                delta = 0.0
            parts.append(f"{name}: exp={expected} got={actual} "
                         f"d={delta:.4f}" + ("" if good else "  MISMATCH"))
            row_ok = row_ok and good
        print(f"  point {i:2d} task {s.task_id}: " + " | ".join(parts))
        ok = ok and row_ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_curve(args: argparse.Namespace) -> int:
    parsed = parse_log_file(args.log)
    meta = parsed.metadata
    print(f"log: {args.log}")
    print(f"   runtime={meta.runtime_label} backend={meta.backend} "
          f"kv={meta.kv_k}/{meta.kv_v} ctx={meta.configured_context} "
          f"model={meta.model_name}")
    print(f"   samples: {len(parsed.samples)}")
    result = aggregate_runs(
        [parsed], dimension="run", bucket_size=args.bucket_size,
        filters=Filters(min_generated=args.min_generated))
    series = result.series[0]
    print(f"   quality-passed: {series.sample_count} "
          f"excluded: {result.excluded_counts}")
    print(f"   {'bucket':>14}  {'decode n':>8}  {'decode med':>10}  "
          f"{'prefill n':>9}  {'prefill med':>11}")
    for b in series.buckets:
        d = b.decode
        pf = b.prefill
        print(f"   {b.bucket_start:>6d}-{b.bucket_end:<7d}  "
              f"{d.n if d else 0:>8d}  "
              f"{d.median if d else float('nan'):>10.2f}  "
              f"{pf.n if pf else 0:>9d}  "
              f"{pf.median if pf else float('nan'):>11.2f}")
    return 0


def cmd_expect(args: argparse.Namespace) -> int:
    parsed = parse_log_file(args.log)
    meta = parsed.metadata
    ok = True
    for spec in args.kv:
        key, _, raw = spec.partition("=")
        if key not in _METADATA_KEYS:
            print(f"  FAIL {key}: not a RunMetadata field")
            ok = False
            continue
        expected = _coerce(raw)
        actual = getattr(meta, key)
        good = actual == expected
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'} {key}: expected={expected!r} "
              f"actual={actual!r}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="parse log dirs, print counts")
    p.add_argument("dirs", nargs="+")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("compare", help="compare report JSON vs parsed log")
    p.add_argument("ref_json")
    p.add_argument("log")
    p.add_argument("--tolerance", type=float, default=0.01)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("curve", help="per-bucket median table for one log")
    p.add_argument("log")
    p.add_argument("--bucket-size", type=int, default=10_000)
    p.add_argument("--min-generated", type=int, default=20)
    p.set_defaults(fn=cmd_curve)

    p = sub.add_parser("expect", help="assert metadata fields of one log")
    p.add_argument("log")
    p.add_argument("kv", nargs="+", metavar="KEY=VALUE")
    p.set_defaults(fn=cmd_expect)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
