"""📊 效能分析 — performance viewer window (Tk) plus display-free helpers.

The module is import-safe without a display: all pure helpers (cache keys,
path selection, filter options, aggregation glue) live outside the widget.
Only ``PerformanceViewer`` needs Tk.

Scans run in a worker thread; each scan carries a generation token so a slow
scan can never overwrite the result of a newer one.  Parsed logs are cached
by (absolute path, size, mtime_ns) — a still-growing active log therefore
invalidates only itself on Rescan.  Logs are only ever READ.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .log_analysis import (
    DEFAULT_MIN_GENERATED,
    AggregationResult,
    Filters,
    ParsedLog,
    aggregate_runs,
    parse_log_file,
)
from .report_export import (
    PALETTE,
    chart_model,
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

__all__ = [
    "log_cache_key",
    "LogCache",
    "legacy_logs_dir",
    "select_scan_paths",
    "filter_options",
    "build_filters",
    "format_series_label",
    "pool_model_label",
    "PerformanceModel",
    "PerformanceViewer",
    "DIMENSIONS",
    "BUCKET_SIZES",
    "MIN_GENERATED_CHOICES",
]

#: Comparison dimension dropdown labels -> log_analysis dimension names.
DIMENSIONS = {
    "Runtime": "runtime",
    "Backend (CUDA/Vulkan)": "backend",
    "KV pair": "kv",
    "Reasoning": "reasoning",
    "Reasoning effort": "reasoning_effort",
    "Max ctx": "context",
    "Vision": "vision",
    "Individual run": "run",
}

BUCKET_SIZES = ("5000", "10000", "20000")
MIN_GENERATED_CHOICES = ("0", "10", "20", "50", "100")

# dashboard palette
_BG = "#0d1118"
_PANEL = "#171e2a"
_SECTION = "#141a24"
_BORDER = "#293244"
_INPUT_BG = "#202838"
_INPUT_FG = "#eef2f8"
_LABEL = "#8293aa"
_TEXT = "#dce6f3"
_BTN = "#253045"
_BTN_HOVER = "#34445f"
_CHART_BG = "#0c1119"
_WARN_FG = "#ffca28"
_OK_FG = "#66bb6a"


# --- pure helpers (no Tk) -----------------------------------------------------------

def log_cache_key(path: Path | str) -> tuple[str, int, int]:
    """Cache identity for a log: (absolute path, size, mtime_ns)."""
    p = Path(path).resolve()
    st = p.stat()
    return (str(p), st.st_size, st.st_mtime_ns)


class LogCache:
    """ParsedLog cache keyed by (path, size, mtime_ns).

    A file that grew (active run) or changed re-parses; unchanged files are
    reused as-is.  Entries for the same path but older stat are pruned.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, int, int], ParsedLog] = {}

    def get_or_parse(self, path: Path | str) -> ParsedLog:
        key = log_cache_key(path)
        hit = self._entries.get(key)
        if hit is not None:
            return hit
        parsed = parse_log_file(key[0])
        for old in [k for k in self._entries if k[0] == key[0]]:
            del self._entries[old]
        self._entries[key] = parsed
        return parsed

    def clear(self) -> None:
        self._entries.clear()


def legacy_logs_dir(llama_dir: Path | str | None) -> Path | None:
    """Legacy launcher logs live under <llama dir>/launcher-app/logs.

    Discovered, never hardcoded: returns None when absent.
    """
    if llama_dir is None:
        return None
    d = Path(llama_dir) / "launcher-app" / "logs"
    return d if d.is_dir() else None


def select_scan_paths(logs_dir: Path | str, extra_dirs=(),
                      extra_files=()) -> list[Path]:
    """All *.log files to scan, deduplicated by resolved path, sorted."""
    out: dict[str, Path] = {}

    def _add(p: Path) -> None:
        try:
            out.setdefault(str(p.resolve()), p)
        except OSError:
            pass

    for base in (Path(logs_dir), *(Path(d) for d in extra_dirs)):
        try:
            if base.is_dir():
                for f in sorted(base.glob("*.log")):
                    _add(f)
        except OSError:
            continue
    for f in extra_files:
        f = Path(f)
        try:
            if f.is_file():
                _add(f)
        except OSError:
            continue
    return [out[k] for k in sorted(out)]


def filter_options(parsed_logs) -> dict[str, tuple[str, ...]]:
    """Distinct metadata values per filterable field (for dropdowns)."""
    metas = [p.metadata for p in parsed_logs]

    def distinct(fn) -> tuple[str, ...]:
        return tuple(sorted({fn(m) for m in metas}, key=str))

    return {
        "model": distinct(lambda m: m.model_name),
        "runtime": distinct(lambda m: m.runtime_label),
        "backend": distinct(lambda m: m.backend),
        "kv": distinct(lambda m: f"{m.kv_k}/{m.kv_v}"),
        "reasoning": distinct(lambda m: m.reasoning),
        "context": distinct(
            lambda m: str(m.configured_context)
            if m.configured_context is not None else "unknown"),
        "vision": distinct(lambda m: "yes" if m.vision_loaded else "no"),
        "gpu_split": distinct(lambda m: m.gpu_split or "unknown"),
        "batch": distinct(
            lambda m: str(m.batch) if m.batch is not None else "unknown"),
        "ubatch": distinct(
            lambda m: str(m.ubatch) if m.ubatch is not None else "unknown"),
    }


def build_filters(selections: dict) -> Filters:
    """Map UI selections to a Filters object (""/"All" = no filter)."""
    def opt(key: str):
        v = selections.get(key)
        return None if v in (None, "", "All", "all") else v

    ctx = opt("context")
    configured_context = None
    if ctx is not None and ctx != "unknown":
        try:
            configured_context = int(ctx)
        except ValueError:
            pass
    vis = opt("vision")
    vision_loaded = (vis == "yes") if vis is not None else None
    mg = selections.get("min_generated")
    return Filters(
        model=opt("model"),
        min_generated=int(mg) if mg is not None else DEFAULT_MIN_GENERATED,
        backend=opt("backend"),
        runtime=opt("runtime"),
        kv=opt("kv"),
        reasoning=opt("reasoning"),
        configured_context=configured_context,
        vision_loaded=vision_loaded,
        gpu_split=opt("gpu_split"),
    )


def format_series_label(key: str, log_count: int,
                        sample_count: int) -> str:
    return f"{key} ({log_count} logs, {sample_count} samples)"


def pool_model_label(parsed_logs) -> str:
    models = sorted({p.metadata.model_name for p in parsed_logs})
    return models[0] if len(models) == 1 else "multiple-models"


# --- scan model (worker thread + generation token, display-free) ---------------------

class PerformanceModel:
    """Owns the parse cache and runs scans off the UI thread.

    ``request_scan`` bumps the generation counter and starts a daemon
    thread; the worker drops its result when a newer scan was requested, so
    stale scans cannot overwrite current state.
    """

    def __init__(self) -> None:
        self.cache = LogCache()
        self._generation = 0
        self._lock = threading.Lock()
        self.paths: list[Path] = []
        self.parsed: list[ParsedLog] = []
        self.errors: list[tuple[str, str]] = []

    def request_scan(self, paths, on_done) -> int:
        """Start a scan of the given paths; returns the generation token."""
        with self._lock:
            self._generation += 1
            gen = self._generation
            self.paths = sorted({Path(p) for p in paths})
        thread = threading.Thread(
            target=self._worker,
            args=(gen, list(self.paths), on_done),
            daemon=True,
        )
        thread.start()
        return gen

    def _worker(self, gen: int, paths: list[Path], on_done) -> None:
        parsed: list[ParsedLog] = []
        errors: list[tuple[str, str]] = []
        for p in paths:
            try:
                parsed.append(self.cache.get_or_parse(p))
            except OSError as exc:
                errors.append((str(p), str(exc)))
            except Exception as exc:  # keep a folder scan alive
                errors.append((str(p), f"{type(exc).__name__}: {exc}"))
        with self._lock:
            stale = gen != self._generation
            if not stale:
                self.parsed, self.errors = parsed, errors
        if not stale and on_done is not None:
            on_done((parsed, errors))


# --- Tk window ------------------------------------------------------------------------

class PerformanceViewer(tk.Toplevel):
    """📊 效能分析 window: scan → filter → aggregate → charts/table/export."""

    def __init__(self, master, logs_dir: Path | str,
                 llama_dir: Path | str | None = None):
        super().__init__(master)
        self.title("📊 效能分析 — Llama Launcher")
        self.geometry("1100x800")
        self.minsize(900, 580)
        self.configure(bg=_BG)
        self.logs_dir = Path(logs_dir)
        self.llama_dir = Path(llama_dir) if llama_dir is not None else None
        self.model = PerformanceModel()
        self.result: AggregationResult | None = None
        self._extra_dirs: list[Path] = []
        self._extra_files: list[Path] = []
        self._combo_widgets: dict[str, ttk.Combobox] = {}
        # worker threads may only enqueue; Tk is touched from the main loop
        self._scan_queue: "queue.Queue" = queue.Queue()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(100, self._poll_scan_queue)
        # auto-scan current Launcher logs on open
        self.scan_launcher_logs()

    # ---- UI construction ---------------------------------------------------------

    def _build_ui(self) -> None:
        # --- sources
        src = self._section("SOURCES / 來源 (read-only)")
        tk.Button(src, text="掃描 Launcher Logs",
                  command=self.scan_launcher_logs, **self._btn_kw()).pack(
                      side="left", padx=(0, 6))
        tk.Button(src, text="匯入 Log", command=self.import_logs,
                  **self._btn_kw()).pack(side="left", padx=6)
        tk.Button(src, text="匯入資料夾", command=self.import_folder,
                  **self._btn_kw()).pack(side="left", padx=6)
        tk.Button(src, text="重新掃描", command=self.rescan,
                  **self._btn_kw()).pack(side="left", padx=6)
        self.status_lbl = tk.Label(src, text="", font=("Segoe UI", 9),
                                   fg=_LABEL, anchor="e")
        self.status_lbl.pack(side="right", fill="x", expand=True)

        # --- filters
        flt = self._section("FILTERS / 篩選")
        self._combos: dict[str, tk.StringVar] = {}
        rows = [
            ("model", "Model", "model"),
            ("compare", "Compare by", None),
            ("runtime", "Runtime", "runtime"),
            ("backend", "Backend", "backend"),
            ("kv", "KV", "kv"),
            ("reasoning", "Reasoning", "reasoning"),
        ]
        for i, (key, label, optkey) in enumerate(rows):
            r, c = divmod(i, 3)
            self._filter_widget(flt, r, c, key, label, optkey)
        rows2 = [
            ("context", "Max ctx", "context"),
            ("vision", "Vision", "vision"),
            ("bucket", "bucket", None),
            ("mingen", "min output", None),
        ]
        for i, (key, label, optkey) in enumerate(rows2):
            r, c = divmod(i, 3)
            self._filter_widget(flt, 1 + r, c, key, label, optkey)
        tk.Button(flt, text="套用", command=self.apply_filters,
                  bg="#2f74d0", fg="white",
                  activebackground="#3e86e2", activeforeground="white",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  padx=14, pady=3).grid(row=1, column=3,
                                        sticky="ew", padx=(8, 0))

        # --- warnings
        wrn = self._section("FAIRNESS / PARSER WARNINGS")
        self.warn_text = tk.Text(wrn, height=3, wrap="word", state="disabled",
                                 bg="#090d13", fg=_WARN_FG,
                                 font=("Segoe UI", 9), relief="flat",
                                 bd=0, padx=10, pady=6)
        self.warn_text.pack(fill="x")

        # --- charts
        dec = self._section("DECODE T/S VS USED CONTEXT")
        self.decode_canvas = tk.Canvas(dec, height=230, bg=_CHART_BG,
                                       highlightthickness=0)
        self.decode_canvas.pack(fill="both", expand=True)
        pre = self._section("PREFILL T/S VS USED CONTEXT")
        self.prefill_canvas = tk.Canvas(pre, height=230, bg=_CHART_BG,
                                        highlightthickness=0)
        self.prefill_canvas.pack(fill="both", expand=True)
        for cv in (self.decode_canvas, self.prefill_canvas):
            cv.bind("<Configure>", lambda _e: self._schedule_chart_redraw())

        # --- table
        tab = self._section("10K SUMMARY TABLE")
        cols = ("series", "bucket", "metric", "n", "median", "p25", "p75",
                "min", "max")
        self.table = ttk.Treeview(tab, columns=cols, show="headings",
                                  height=10)
        widths = {"series": 220, "bucket": 110, "metric": 60, "n": 50,
                  "median": 70, "p25": 70, "p75": 70, "min": 70, "max": 70}
        for col in cols:
            self.table.heading(col, text=col)
            self.table.column(col, width=widths[col],
                              anchor="w" if col == "series" else "e")
        tsb = ttk.Scrollbar(tab, command=self.table.yview)
        self.table.configure(yscrollcommand=tsb.set)
        self.table.pack(side="left", fill="both", expand=True)
        tsb.pack(side="right", fill="y")

        # --- export
        exp = self._section("EXPORT")
        for label, fmt in (
                ("HTML", "html"), ("PNG", "png"), ("SVG", "svg"),
                ("Raw CSV", "raw_csv"), ("Aggregate CSV", "aggregate_csv"),
                ("Markdown", "markdown")):
            tk.Button(exp, text=label, command=lambda f=fmt: self._export(f),
                      **self._btn_kw()).pack(side="left", padx=(0, 6))

    def _btn_kw(self) -> dict:
        return dict(font=("Segoe UI", 9), bg=_BTN, fg=_TEXT,
                    activebackground=_BTN_HOVER, activeforeground="white",
                    relief="flat", padx=10, pady=3)

    def _section(self, title: str) -> tk.Frame:
        frame = tk.Frame(self, bg=_BG)
        frame.pack(fill="both", expand=False, padx=12, pady=(8, 0))
        tk.Label(frame, text=title, font=("Segoe UI", 8, "bold"),
                 fg=_LABEL, bg=_BG, anchor="w").pack(fill="x")
        body = tk.Frame(frame, bg=_PANEL,
                        highlightthickness=1, highlightbackground=_BORDER)
        body.pack(fill="both", expand=True, pady=(2, 0))
        return body

    def _filter_widget(self, parent: tk.Frame, row: int, col: int,
                       key: str, label: str,
                       options_key: str | None) -> None:
        wrap = tk.Frame(parent, bg=_PANEL)
        wrap.grid(row=row, column=col, sticky="ew", padx=6, pady=5)
        parent.columnconfigure(col, weight=1)
        tk.Label(wrap, text=label, font=("Segoe UI", 8),
                 fg=_LABEL, bg=_PANEL).pack(anchor="w")
        var = tk.StringVar(value="All" if options_key else "")
        if options_key is None and key == "compare":
            values = tuple(DIMENSIONS)
            var.set("Runtime")
        elif options_key is None and key == "bucket":
            values = BUCKET_SIZES
            var.set("10000")
        elif options_key is None and key == "mingen":
            values = MIN_GENERATED_CHOICES
            var.set(str(DEFAULT_MIN_GENERATED))
        else:
            values = ("All",)
        cb = ttk.Combobox(wrap, textvariable=var, values=values,
                          state="readonly", width=14)
        cb.pack(fill="x", pady=(1, 0))
        self._combos[key] = var
        self._combo_widgets[key] = cb
        self._combo_widget(key, cb)

    def _combo_widget(self, key: str, cb: ttk.Combobox) -> None:
        cb.configure(style="Perf.TCombobox")
        try:
            style = ttk.Style(self)
            style.configure("Perf.TCombobox", fieldbackground=_INPUT_BG,
                            background=_INPUT_BG, foreground=_INPUT_FG,
                            arrowcolor=_TEXT, bordercolor=_BORDER,
                            lightcolor=_INPUT_BG, darkcolor=_INPUT_BG,
                            selectbackground=_INPUT_BG,
                            selectforeground=_INPUT_FG)
            style.map("Perf.TCombobox",
                      fieldbackground=[("readonly", _INPUT_BG)],
                      foreground=[("readonly", _INPUT_FG)])
        except tk.TclError:
            pass

    # ---- scans ---------------------------------------------------------------------

    def _scan_target_paths(self) -> list[Path]:
        dirs = list(self._extra_dirs)
        files = list(self._extra_files)
        legacy = legacy_logs_dir(self.llama_dir)
        if legacy is not None and legacy.resolve() != self.logs_dir.resolve():
            dirs.append(legacy)
        return select_scan_paths(self.logs_dir, dirs, files)

    def _start_scan(self, note: str = "") -> None:
        paths = self._scan_target_paths()
        self.status_lbl.config(
            text=f"Scanning {len(paths)} files… {note}".strip())
        self.warn_text.config(state="normal")
        self.warn_text.delete("1.0", "end")
        self.warn_text.config(state="disabled")
        # on_done runs on the worker thread: enqueue only; the main loop
        # (_poll_scan_queue) applies the result on the Tk thread.
        self.model.request_scan(
            paths, on_done=lambda res: self._scan_queue.put(res))

    def _poll_scan_queue(self) -> None:
        try:
            while True:
                self._on_scan_done(self._scan_queue.get_nowait())
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll_scan_queue)

    def _on_scan_done(self, _result) -> None:
        if not self.winfo_exists():
            return
        parsed = self.model.parsed
        n_samples = sum(len(p.samples) for p in parsed)
        self.status_lbl.config(
            text=f"{len(self.model.paths)} files / {n_samples} samples"
                 + (f" · {len(self.model.errors)} unreadable"
                    if self.model.errors else ""))
        self._populate_filter_options()
        self.apply_filters()

    def scan_launcher_logs(self) -> None:
        """Reset imports and scan Launcher logs dir (+ auto-detected legacy)."""
        self._extra_dirs = []
        self._extra_files = []
        legacy = legacy_logs_dir(self.llama_dir)
        note = f" (+ legacy {legacy})" if legacy is not None else ""
        self._start_scan(note)

    def rescan(self) -> None:
        """Re-scan the same paths; the cache invalidates only changed files."""
        self._start_scan()

    def import_logs(self) -> None:
        paths = filedialog.askopenfilenames(
            title="匯入 log 檔",
            filetypes=[("log files", "*.log"), ("all files", "*")])
        if not paths:
            return
        self._extra_files.extend(Path(p) for p in paths)
        self._start_scan(f"({len(self._extra_files)} imported files)")

    def import_folder(self) -> None:
        d = filedialog.askdirectory(title="匯入資料夾")
        if not d:
            return
        self._extra_dirs.append(Path(d))
        self._start_scan(f"({len(self._extra_dirs)} imported folders)")

    # ---- filters / aggregation -------------------------------------------------------

    def _populate_filter_options(self) -> None:
        opts = filter_options(self.model.parsed)
        mapping = {"model": "model", "runtime": "runtime",
                   "backend": "backend", "kv": "kv", "reasoning": "reasoning",
                   "context": "context", "vision": "vision"}
        for varkey, optkey in mapping.items():
            var = self._combos.get(varkey)
            if var is None:
                continue
            current = var.get()
            values = ("All",) + tuple(opts[optkey])
            widget = self._combo_widgets.get(varkey)
            if widget is not None:
                widget.configure(values=values)
            if current not in values:
                var.set("All")

    def _selections(self) -> dict:
        g = {k: self._combos[k].get()
             for k in ("model", "runtime", "backend", "kv", "reasoning",
                       "context", "vision")}
        g["min_generated"] = self._combos["mingen"].get()
        return g

    def apply_filters(self) -> None:
        if not self.model.parsed:
            return
        dimension = DIMENSIONS.get(self._combos["compare"].get(), "runtime")
        bucket_size = int(self._combos["bucket"].get())
        self.result = aggregate_runs(
            self.model.parsed, dimension=dimension, bucket_size=bucket_size,
            filters=build_filters(self._selections()))
        self._render()

    # ---- rendering ---------------------------------------------------------------------

    def _schedule_chart_redraw(self) -> None:
        if self.result is not None and self.winfo_exists():
            self.after(120, self._render_charts)

    def _render(self) -> None:
        if self.result is None:
            return
        # warnings
        self.warn_text.config(state="normal")
        self.warn_text.delete("1.0", "end")
        if self.result.warnings:
            for w in self.result.warnings:
                self.warn_text.insert("end", f"⚠ {w}\n")
            self.warn_text.config(fg=_WARN_FG)
        else:
            self.warn_text.insert("end",
                                  "No fairness warnings: pool is uniform in "
                                  "all non-compared fields.\n")
            self.warn_text.config(fg=_OK_FG)
        self.warn_text.config(state="disabled")
        # table
        self.table.delete(*self.table.get_children())
        for s in self.result.series:
            for b in s.buckets:
                for metric, st in (("decode", b.decode),
                                   ("prefill", b.prefill)):
                    if st is None:
                        continue
                    self.table.insert(
                        "", "end",
                        values=(s.series_key,
                                f"{b.bucket_start}-{b.bucket_end}",
                                metric, st.n,
                                f"{st.median:.2f}", f"{st.p25:.2f}",
                                f"{st.p75:.2f}", f"{st.min:.2f}",
                                f"{st.max:.2f}"))
        self._render_charts()

    def _render_charts(self) -> None:
        if self.result is None:
            return
        self._draw_chart(self.decode_canvas, self.result, "decode")
        self._draw_chart(self.prefill_canvas, self.result, "prefill")

    def _draw_chart(self, canvas: tk.Canvas, result: AggregationResult,
                    metric: str) -> None:
        canvas.delete("all")
        model = chart_model(result, metric)
        w = max(canvas.winfo_width(), 640)
        h = max(canvas.winfo_height(), 200)
        L, R, T, B = 56, 150, 30, 44
        pw, ph = w - L - R, h - T - B

        def X(tok: int) -> float:
            return L + pw * tok / max(model.x_max, 1)

        def Y(v: float) -> float:
            return (T + ph) - ph * v / model.y_max

        canvas.create_text(w / 2, 14,
                           text=f"{metric} t/s vs used context (10K buckets)",
                           fill="#e8e8f0", font=("Segoe UI", 10, "bold"))
        canvas.create_line(L, T + ph, L + pw, T + ph, fill="#888888")
        canvas.create_line(L, T, L, T + ph, fill="#888888")
        bucket = self._current_bucket_size()
        b = 0
        while b <= model.x_max:
            x = X(b)
            canvas.create_line(x, T + ph, x, T + ph + 4, fill="#888888")
            canvas.create_text(x, T + ph + 14,
                               text=("0" if b == 0 else f"{b / 1000:g}K"),
                               fill="#bbbbbb", font=("Segoe UI", 8))
            b += bucket
        for i in range(6):
            v = model.y_max * i / 5
            y = Y(v)
            canvas.create_line(L - 4, y, L, y, fill="#888888")
            canvas.create_text(L - 8, y, text=f"{v:g}", fill="#bbbbbb",
                               font=("Segoe UI", 8), anchor="e")
        canvas.create_text(L + pw / 2, h - 10,
                           text="used context (10K buckets)",
                           fill="#e8e8f0", font=("Segoe UI", 9))
        canvas.create_text(14, T + ph / 2, text=f"{metric} t/s",
                           fill="#e8e8f0", font=("Segoe UI", 9),
                           anchor="w")
        if not model.has_data:
            canvas.create_text(L + pw / 2, T + ph / 2, text="no data",
                               fill="#999999", font=("Segoe UI", 10))
            return
        for i, (key, pts) in enumerate(model.series):
            color = "#%02x%02x%02x" % PALETTE[i % len(PALETTE)]
            if len(pts) >= 2:
                band = [c for p in pts for c in ((X(p.x), Y(p.p75)),)]
                band += [c for p in reversed(pts) for c in ((X(p.x), Y(p.p25)),)]
                canvas.create_polygon(*[v for pair in band for v in pair],
                                      fill=color, outline="")
            if pts:
                if len(pts) >= 2:
                    line = [v for p in pts
                            for v in (X(p.x), Y(p.median))]
                    canvas.create_line(*line, fill=color, width=2)
                for p in pts:
                    x, y = X(p.x), Y(p.median)
                    canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                       fill=color, outline="")
                    canvas.create_text(x, y - 10,
                                       text=format_point_label(p.median),
                                       fill=color, font=("Segoe UI", 8))
            label = key if len(key) <= 26 else key[:25] + "…"
            ly = T + 12 + 20 * i
            canvas.create_rectangle(L + pw + 14, ly, L + pw + 26, ly + 12,
                                    fill=color, outline="")
            canvas.create_text(L + pw + 32, ly + 6, text=label,
                               fill="#dddddd", font=("Segoe UI", 9),
                               anchor="w")

    def _current_bucket_size(self) -> int:
        try:
            return int(self._combos["bucket"].get())
        except (KeyError, tk.TclError, ValueError):
            return 10_000

    # ---- export ---------------------------------------------------------------------

    def _export(self, fmt: str) -> None:
        if self.result is None or not self.model.parsed:
            messagebox.showinfo("效能分析", "先掃描或匯入 logs 再匯出。",
                                parent=self)
            return
        dimension = DIMENSIONS.get(self._combos["compare"].get(), "runtime")
        bucket_size = int(self._combos["bucket"].get())
        label = pool_model_label(self.model.parsed)
        ext = {"html": "html", "svg": "svg", "png": "png",
               "raw_csv": "csv", "aggregate_csv": "csv",
               "markdown": "md"}[fmt]
        default = suggested_filename(label, dimension, bucket_size, ext,
                                     now=datetime.now())
        chosen = filedialog.asksaveasfilename(
            parent=self, defaultextension=f".{ext}", initialfile=default,
            filetypes=[(ext.upper(), f".{ext}")])
        if not chosen:
            return
        final = safe_output_path(chosen)
        try:
            if fmt == "html":
                final.write_text(
                    render_html(self.result,
                                parsed_logs=self.model.parsed,
                                dimension=dimension,
                                bucket_size=bucket_size),
                    encoding="utf-8")
            elif fmt == "svg":
                final.write_text(
                    render_chart_svg(self.result, metric="decode",
                                     bucket_size=bucket_size),
                    encoding="utf-8")
            elif fmt == "png":
                final.write_bytes(
                    render_chart_png(self.result, metric="decode",
                                     bucket_size=bucket_size))
            elif fmt == "raw_csv":
                final.write_text(
                    render_raw_csv(self.model.parsed), encoding="utf-8")
            elif fmt == "aggregate_csv":
                final.write_text(
                    render_aggregate_csv(self.result), encoding="utf-8")
            else:
                final.write_text(
                    render_markdown(self.result,
                                    parsed_logs=self.model.parsed,
                                    dimension=dimension,
                                    bucket_size=bucket_size,
                                    exported_files=[]),
                    encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("效能分析", f"匯出失敗: {exc}", parent=self)
            return
        if final.name != Path(chosen).name:
            self.status_lbl.config(
                text=f"匯出（檔名已加後避免覆蓋）: {final.name}")
        else:
            self.status_lbl.config(text=f"已匯出: {final.name}")
