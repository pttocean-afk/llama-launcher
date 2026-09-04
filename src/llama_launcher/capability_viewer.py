"""Hardware capability / model-selection dashboard for Launcher logs."""
from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from .capability_analysis import (
    CapabilityEnvelope,
    CapabilityKey,
    CapabilityReport,
    CapabilityRow,
    aggregate_capabilities,
    model_usage,
    pareto_rows,
    render_capability_csv,
    runtime_options,
)
from .performance_viewer import (
    DEFAULT_MIN_GENERATED,
    MIN_GENERATED_CHOICES,
    PerformanceModel,
    legacy_logs_dir,
    select_scan_paths,
)
from .report_export import PALETTE

_BG = "#0d1118"
_PANEL = "#171e2a"
_SECTION = "#141a24"
_BORDER = "#293244"
_INPUT_BG = "#202838"
_TEXT = "#dce6f3"
_LABEL = "#8293aa"
_MUTED = "#627086"
_BLUE = "#4da3ff"
_GREEN = "#66bb6a"
_YELLOW = "#ffca28"
_RED = "#ef5350"
_CHART_BG = "#0c1119"


def format_context(value: int | None) -> str:
    if value is None:
        return "?"
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}K"
    if value >= 1000:
        return f"{value / 1000:.0f}K"
    return str(value)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _midclip(text: str, max_chars: int, tail: int = 14) -> str:
    """Ellipsize the middle so unique model tails stay visible."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= tail + 2:
        return text[:max_chars - 1] + "…"
    head = max_chars - tail - 1
    return f"{text[:head]}…{text[-tail:]}"


class CapabilityViewer(tk.Toplevel):
    """Primary performance view: verified limits, ranking, curves and Pareto."""

    def __init__(self, master, logs_dir: Path | str,
                 llama_dir: Path | str | None = None):
        super().__init__(master)
        from .ui_scale import DpiScale, S, fit_window_size
        DpiScale.init(self)
        width, height = fit_window_size(self, S(1240), S(820))
        x = max((self.winfo_screenwidth() - width) // 2, 0)
        y = max((self.winfo_screenheight() - height) // 2, 0)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(*fit_window_size(self, S(980), S(640), screen_ratio=1.0))
        self.title("📊 硬體能力與模型選型 v2 — Llama Launcher")
        self.configure(bg=_BG)
        self.logs_dir = Path(logs_dir)
        self.llama_dir = Path(llama_dir) if llama_dir is not None else None
        self.model = PerformanceModel()
        self.report: CapabilityReport | None = None
        self._extra_dirs: list[Path] = []
        self._extra_files: list[Path] = []
        self._scan_queue: queue.Queue = queue.Queue()
        self._model_vars: dict[str, tk.BooleanVar] = {}
        self._selected_keys: set[CapabilityKey] = set()
        self._row_by_key: dict[CapabilityKey, CapabilityRow] = {}
        self._config_iids: dict[str, CapabilityKey] = {}
        self._env_iids: dict[str, CapabilityEnvelope] = {}
        self._model_colors: dict[str, str] = {}
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(100, self._poll_scan_queue)
        self.scan_launcher_logs()

    # ---- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self, bg=_BG)
        toolbar.pack(fill="x", padx=12, pady=(10, 4))
        for text, command in (
            ("掃描 Launcher Logs", self.scan_launcher_logs),
            ("匯入 Log", self.import_logs),
            ("匯入資料夾", self.import_folder),
            ("重新掃描", self.rescan),
            ("套用條件", self.apply_selection),
            ("匯出能力 CSV", self.export_capability_csv),
            ("舊版詳細統計 / 匯出", self.open_legacy_viewer),
        ):
            tk.Button(toolbar, text=text, command=command, bg="#253045",
                      fg=_TEXT, activebackground="#34445f",
                      activeforeground="white", relief="flat",
                      font=("Segoe UI", 9), padx=10, pady=4).pack(
                          side="left", padx=(0, 6))
        self.status_label = tk.Label(toolbar, text="", bg=_BG, fg=_LABEL,
                                     anchor="e", font=("Segoe UI", 9))
        self.status_label.pack(side="right", fill="x", expand=True)

        filters = tk.Frame(self, bg=_PANEL, highlightthickness=1,
                           highlightbackground=_BORDER)
        filters.pack(fill="x", padx=12, pady=4)
        self.runtime_var = tk.StringVar(value="")
        self.metric_var = tk.StringVar(value="穩定中位數")
        self.bucket_var = tk.StringVar(value="10000")
        self.min_gen_var = tk.StringVar(value=str(DEFAULT_MIN_GENERATED))
        self._combo(filters, "llama.cpp Build（單選）", self.runtime_var, (), 0)
        self._combo(filters, "速度排名", self.metric_var,
                    ("穩定中位數", "觀測最高"), 1)
        self._combo(filters, "曲線 bucket", self.bucket_var,
                    ("5000", "10000", "20000"), 2)
        self._combo(filters, "最少輸出 tokens", self.min_gen_var,
                    MIN_GENERATED_CHOICES, 3)
        self.runtime_combo.bind("<<ComboboxSelected>>",
                                lambda _e: self._runtime_changed())
        self.metric_var.trace_add("write", lambda *_: self._render_overview())

        body = tk.PanedWindow(self, orient="horizontal", bg=_BG,
                              sashwidth=5, sashrelief="flat")
        body.pack(fill="both", expand=True, padx=12, pady=(4, 10))
        left = tk.Frame(body, bg=_SECTION, width=250,
                        highlightthickness=1, highlightbackground=_BORDER)
        right = tk.Frame(body, bg=_BG)
        body.add(left, minsize=215, width=260)
        body.add(right, minsize=650)
        self._build_model_picker(left)
        self._build_tabs(right)

    def _combo(self, parent, label, variable, values, column) -> None:
        wrap = tk.Frame(parent, bg=_PANEL)
        wrap.grid(row=0, column=column, padx=7, pady=6, sticky="ew")
        parent.columnconfigure(column, weight=1)
        tk.Label(wrap, text=label, bg=_PANEL, fg=_LABEL,
                 font=("Segoe UI", 8)).pack(anchor="w")
        cb = ttk.Combobox(wrap, textvariable=variable, values=values,
                          state="readonly", width=8)
        cb.pack(fill="x", pady=(2, 0))
        if variable is self.runtime_var:
            self.runtime_combo = cb

    def _build_model_picker(self, parent) -> None:
        tk.Label(parent, text="MODELS / 模型複選", bg=_SECTION, fg=_LABEL,
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(
                     fill="x", padx=10, pady=(10, 4))
        tk.Label(parent, text="依使用次數排序", bg=_SECTION, fg=_MUTED,
                 font=("Segoe UI", 8), anchor="w").pack(fill="x", padx=10)
        actions = tk.Frame(parent, bg=_SECTION)
        actions.pack(fill="x", padx=8, pady=6)
        tk.Button(actions, text="全選", command=lambda: self._set_all_models(True),
                  bg="#253045", fg=_TEXT, relief="flat").pack(side="left", padx=2)
        tk.Button(actions, text="清除", command=lambda: self._set_all_models(False),
                  bg="#253045", fg=_TEXT, relief="flat").pack(side="left", padx=2)
        self.model_canvas = tk.Canvas(parent, bg=_SECTION, highlightthickness=0)
        scroll = ttk.Scrollbar(parent, orient="vertical",
                               command=self.model_canvas.yview)
        self.model_list = tk.Frame(self.model_canvas, bg=_SECTION)
        self.model_window = self.model_canvas.create_window(
            (0, 0), window=self.model_list, anchor="nw")
        self.model_canvas.configure(yscrollcommand=scroll.set)
        self.model_canvas.pack(side="left", fill="both", expand=True, padx=(6, 0))
        scroll.pack(side="right", fill="y")
        self.model_list.bind("<Configure>", lambda _e: self.model_canvas.configure(
            scrollregion=self.model_canvas.bbox("all")))
        self.model_canvas.bind("<Configure>", lambda e: self.model_canvas.itemconfigure(
            self.model_window, width=e.width))

    def _build_tabs(self, parent) -> None:
        style = ttk.Style(self)
        try:
            style.configure("Capability.TNotebook", background=_BG, borderwidth=0)
            # Stick with the default theme tab layout (vista/xpnative); do not
            # override it — the theme's own selected-tab sizing is reliable.
            # Only change colours and font so the active tab stands out without
            # shrinking.
            style.configure("Capability.TNotebook.Tab",
                            background=_INPUT_BG, foreground="#c8d2e4",
                            font=("Segoe UI", 9, "bold"))
            style.map("Capability.TNotebook.Tab",
                      background=[("selected", "#2f74d0"),
                                  ("!selected", _INPUT_BG)],
                      foreground=[("selected", "white"),
                                  ("!selected", "#c8d2e4")])
            style.configure("Capability.Treeview", background=_PANEL,
                            fieldbackground=_PANEL, foreground=_TEXT,
                            rowheight=24, borderwidth=0)
            style.configure("Capability.Treeview.Heading", background=_INPUT_BG,
                            foreground=_TEXT, relief="flat")
            style.map("Capability.Treeview",
                      background=[("selected", "#245a91")],
                      foreground=[("selected", "white")])
        except tk.TclError:
            pass
        self.tabs = ttk.Notebook(parent, style="Capability.TNotebook")
        self.tabs.pack(fill="both", expand=True)
        overview = tk.Frame(self.tabs, bg=_BG)
        curves = tk.Frame(self.tabs, bg=_BG)
        tradeoff = tk.Frame(self.tabs, bg=_BG)
        self.tabs.add(overview, text="能力總覽")
        self.tabs.add(curves, text="速度曲線比較")
        self.tabs.add(tradeoff, text="Context / 速度取捨")

        self.summary_label = tk.Label(
            overview, text="", bg=_BG, fg=_TEXT, anchor="w",
            justify="left", font=("Segoe UI", 9))
        overview.columnconfigure(0, weight=1)
        overview.rowconfigure(1, weight=3)
        overview.rowconfigure(2, weight=2)
        self.summary_label.grid(row=0, column=0, sticky="ew",
                                padx=8, pady=(5, 2))
        chart_wrap = tk.Frame(overview, bg=_CHART_BG)
        chart_wrap.grid(row=1, column=0, sticky="nsew", padx=5, pady=3)
        self.overview_canvas = tk.Canvas(chart_wrap, bg=_CHART_BG,
                                         highlightthickness=0, height=220)
        oy = ttk.Scrollbar(chart_wrap, orient="vertical",
                           command=self.overview_canvas.yview)
        self.overview_canvas.configure(yscrollcommand=oy.set)
        self.overview_canvas.pack(side="left", fill="both", expand=True)
        oy.pack(side="right", fill="y")
        self.overview_canvas.bind("<Configure>", lambda _e: self._render_overview())
        cols = ("model", "backend", "vision", "detail", "infer", "ready",
                "failed", "speed", "runs")
        self.envelope_table = ttk.Treeview(
            overview, columns=cols, show="headings", height=7,
            style="Capability.Treeview")
        headings = {
            "model": "Model", "backend": "Backend", "vision": "Vision",
            "detail": "KV / Reasoning", "infer": "最大可推論",
            "ready": "最大可啟動", "failed": "首次失敗",
            "speed": "穩定 / Max T/S", "runs": "Runs",
        }
        widths = {"model": 220, "backend": 70, "vision": 65, "detail": 160,
                  "infer": 85, "ready": 85, "failed": 75,
                  "speed": 120, "runs": 55}
        for col in cols:
            self.envelope_table.heading(col, text=headings[col])
            self.envelope_table.column(col, width=widths[col],
                                       anchor="w" if col == "model" else "center")
        env_sy = ttk.Scrollbar(overview, orient="vertical",
                               command=self.envelope_table.yview)
        env_sx = ttk.Scrollbar(overview, orient="horizontal",
                               command=self.envelope_table.xview)
        self.envelope_table.configure(yscrollcommand=env_sy.set,
                                      xscrollcommand=env_sx.set)
        self.envelope_table.grid(row=2, column=0, sticky="nsew",
                                 padx=(5, 0), pady=(2, 5))
        env_sy.grid(row=2, column=1, sticky="ns", pady=(2, 5))
        env_sx.grid(row=3, column=0, sticky="ew", padx=(5, 0))
        overview.rowconfigure(3, weight=0)
        self.envelope_table.bind("<Double-1>", self._toggle_envelope_from_table)

        curve_help = tk.Label(
            curves, text="雙擊左側配置或點能力圖橫條可加入/移除比較；實線為中位數，陰影為 P25–P75。",
            bg=_BG, fg=_LABEL, anchor="w", font=("Segoe UI", 9))
        curve_help.pack(fill="x", padx=8, pady=(6, 2))
        curve_body = tk.PanedWindow(curves, orient="horizontal", bg=_BG, sashwidth=4)
        curve_body.pack(fill="both", expand=True, padx=5, pady=3)
        config_wrap = tk.Frame(curve_body, bg=_SECTION)
        curve_wrap = tk.Frame(curve_body, bg=_CHART_BG)
        curve_body.add(config_wrap, width=280, minsize=240)
        curve_body.add(curve_wrap, minsize=450)
        cfg_cols = ("pick", "config")
        self.config_table = ttk.Treeview(
            config_wrap, columns=cfg_cols, show="headings",
            style="Capability.Treeview")
        for col, text, width in (
            ("pick", "比較", 48), ("config", "配置 / 狀態 / 速度", 300)):
            self.config_table.heading(col, text=text)
            self.config_table.column(col, width=width,
                                     anchor="w" if col == "config" else "center")
        sy = ttk.Scrollbar(config_wrap, orient="vertical",
                           command=self.config_table.yview)
        sx = ttk.Scrollbar(config_wrap, orient="horizontal",
                           command=self.config_table.xview)
        self.config_table.configure(yscrollcommand=sy.set,
                                    xscrollcommand=sx.set)
        self.config_table.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        config_wrap.rowconfigure(0, weight=1)
        config_wrap.rowconfigure(1, weight=0)
        config_wrap.columnconfigure(0, weight=1)
        self.config_table.bind("<Double-1>", self._toggle_config_from_table)
        self.curve_canvas = tk.Canvas(curve_wrap, bg=_CHART_BG,
                                      highlightthickness=0)
        self.curve_canvas.pack(fill="both", expand=True)
        self.curve_canvas.bind("<Configure>", lambda _e: self._render_curves())
        controls = tk.Frame(curves, bg=_BG)
        controls.pack(fill="x", padx=6, pady=3)
        tk.Button(controls, text="清除比較", command=self._clear_curves,
                  bg="#253045", fg=_TEXT, relief="flat").pack(side="left")
        self.selection_label = tk.Label(controls, text="", bg=_BG, fg=_LABEL)
        self.selection_label.pack(side="left", padx=10)

        tk.Label(tradeoff,
                 text="右上角越好；亮線為 Pareto frontier。點圓點可加入速度曲線比較。",
                 bg=_BG, fg=_LABEL, anchor="w", font=("Segoe UI", 9)).pack(
                     fill="x", padx=8, pady=(6, 2))
        self.tradeoff_canvas = tk.Canvas(tradeoff, bg=_CHART_BG,
                                         highlightthickness=0)
        self.tradeoff_canvas.pack(fill="both", expand=True, padx=5, pady=4)
        self.tradeoff_canvas.bind("<Configure>", lambda _e: self._render_tradeoff())
        self.tradeoff_canvas.bind("<MouseWheel>", self._tradeoff_wheel)
        self.tradeoff_canvas.bind("<ButtonPress-1>", self._tradeoff_mouse_down)
        self.tradeoff_canvas.bind("<B1-Motion>", self._tradeoff_mouse_drag)
        self._tradeoff_zoom = 1.0
        zoombar = tk.Frame(tradeoff, bg=_BG)
        zoombar.pack(fill="x", padx=6, pady=3)
        self._zoom_label = tk.Label(zoombar, text="", bg=_BG, fg=_LABEL,
                                    font=("Segoe UI", 9))
        self._zoom_label.pack(side="left")
        tk.Button(zoombar, text="重置縮放", command=self._tradeoff_reset_zoom,
                  bg="#253045", fg=_TEXT, relief="flat").pack(side="left", padx=(8, 0))
        tk.Label(zoombar, text="（左鍵拖曳平移，滾輪以游標為中心縮放）", bg=_BG, fg=_MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=4)

    # ---- scanning ------------------------------------------------------------

    def _paths(self) -> list[Path]:
        dirs = list(self._extra_dirs)
        legacy = legacy_logs_dir(self.llama_dir)
        if legacy is not None and legacy.resolve() != self.logs_dir.resolve():
            dirs.append(legacy)
        return select_scan_paths(self.logs_dir, dirs, self._extra_files)

    def _start_scan(self, note="") -> None:
        paths = self._paths()
        self.status_label.config(text=f"掃描 {len(paths)} 個 logs… {note}")
        self.model.request_scan(paths, lambda result: self._scan_queue.put(result))

    def _poll_scan_queue(self) -> None:
        try:
            while True:
                self._scan_queue.get_nowait()
                try:
                    self._scan_complete()
                except Exception as exc:
                    self.status_label.config(
                        text=f"掃描處理錯誤：{exc}")
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll_scan_queue)

    def _scan_complete(self) -> None:
        options = runtime_options(self.model.parsed)
        self.runtime_combo.configure(values=options)
        if self.runtime_var.get() not in options:
            self.runtime_var.set(options[0] if options else "")
        self._populate_models(initial=True)
        samples = sum(len(p.samples) for p in self.model.parsed)
        self.status_label.config(
            text=f"{len(self.model.parsed)} logs · {samples} samples"
                 + (f" · {len(self.model.errors)} unreadable" if self.model.errors else ""))
        self.apply_selection()

    def scan_launcher_logs(self) -> None:
        self._extra_dirs.clear()
        self._extra_files.clear()
        self._start_scan()

    def rescan(self) -> None:
        self._start_scan()

    def import_logs(self) -> None:
        paths = filedialog.askopenfilenames(title="匯入 log",
                                            filetypes=[("Log", "*.log"), ("All", "*")])
        if paths:
            self._extra_files.extend(Path(p) for p in paths)
            self._start_scan()

    def import_folder(self) -> None:
        path = filedialog.askdirectory(title="匯入 Log 資料夾")
        if path:
            self._extra_dirs.append(Path(path))
            self._start_scan()

    def export_capability_csv(self) -> None:
        if self.report is None:
            return
        runtime = (self.report.selected_runtime or "all").replace("/", "-")
        path = filedialog.asksaveasfilename(
            title="匯出硬體能力 CSV", defaultextension=".csv",
            initialfile=f"llama-capability-{runtime}.csv",
            filetypes=[("CSV", "*.csv")])
        if path:
            Path(path).write_text(render_capability_csv(self.report),
                                  encoding="utf-8-sig")

    def open_legacy_viewer(self) -> None:
        from .performance_viewer import PerformanceViewer
        PerformanceViewer(self, self.logs_dir, self.llama_dir)

    # ---- filtering -----------------------------------------------------------

    def _runtime_changed(self) -> None:
        self._populate_models(initial=False)
        self.apply_selection()

    def _populate_models(self, initial: bool) -> None:
        old = {name for name, var in self._model_vars.items() if var.get()}
        for child in self.model_list.winfo_children():
            child.destroy()
        self._model_vars.clear()
        usage = model_usage(self.model.parsed,
                            runtime_label=self.runtime_var.get() or None)
        for index, item in enumerate(usage):
            selected = (item.model_name in old) if old else (initial and index < 8)
            var = tk.BooleanVar(value=selected)
            text = f"{item.model_name}\n  {item.run_count} runs · {item.inference_runs} 可推論"
            cb = tk.Checkbutton(self.model_list, text=text, variable=var,
                                command=self.apply_selection, bg=_SECTION, fg=_TEXT,
                                activebackground=_SECTION, activeforeground="white",
                                selectcolor=_INPUT_BG, justify="left", anchor="w",
                                wraplength=220, font=("Segoe UI", 9))
            cb.pack(fill="x", padx=4, pady=2)
            self._model_vars[item.model_name] = var

    def _set_all_models(self, value: bool) -> None:
        for var in self._model_vars.values():
            var.set(value)
        self.apply_selection()

    def apply_selection(self) -> None:
        if not self.model.parsed:
            return
        selected = {name for name, var in self._model_vars.items() if var.get()}
        if not selected and self._model_vars:
            self.report = aggregate_capabilities(
                self.model.parsed, runtime_label=self.runtime_var.get() or None,
                selected_models=set(), bucket_size=int(self.bucket_var.get()),
                min_generated=int(self.min_gen_var.get()))
        else:
            self.report = aggregate_capabilities(
                self.model.parsed, runtime_label=self.runtime_var.get() or None,
                selected_models=selected, bucket_size=int(self.bucket_var.get()),
                min_generated=int(self.min_gen_var.get()))
        self._row_by_key = {r.key: r for r in self.report.rows}
        self._selected_keys.intersection_update(self._row_by_key)
        if not self._selected_keys:
            ranked = sorted((r for r in self.report.rows if r.stable_decode_tps is not None),
                            key=lambda r: r.stable_decode_tps or 0, reverse=True)
            self._selected_keys.update(r.key for r in ranked[:3])
        self._assign_colors()
        self._render_all()

    def _assign_colors(self) -> None:
        models = sorted({r.key.model_name for r in self.report.rows}) if self.report else []
        self._model_colors = {name: _hex(PALETTE[i % len(PALETTE)])
                              for i, name in enumerate(models)}

    # ---- overview ------------------------------------------------------------

    def _speed(self, env: CapabilityEnvelope) -> float | None:
        return (env.observed_max_decode_tps if self.metric_var.get() == "觀測最高"
                else env.stable_decode_tps)

    def _resolve_envelope_row(self, env: CapabilityEnvelope) -> CapabilityRow | None:
        target = env.max_inference_context or env.max_ready_context
        candidates = [self._row_by_key[k] for k in env.row_keys
                      if k in self._row_by_key and k.configured_context == target]
        if not candidates:
            candidates = [self._row_by_key[k] for k in env.row_keys
                          if k in self._row_by_key]
        return max(candidates, key=lambda r: r.stable_decode_tps or -1) \
            if candidates else None

    def _render_all(self) -> None:
        self._render_overview()
        self._render_tables()
        self._render_curves()
        self._render_tradeoff()

    def _render_overview(self) -> None:
        canvas = getattr(self, "overview_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        report = self.report
        if report is None or not report.envelopes:
            canvas.create_text(20, 20, anchor="nw", fill=_LABEL,
                               text="沒有符合條件的能力紀錄")
            canvas.configure(scrollregion=(0, 0, 1, 1))
            return
        envs = sorted(report.envelopes,
                      key=lambda e: self._speed(e) or -1, reverse=True)
        values = [self._speed(e) or 0 for e in envs]
        max_speed = max(values) if values else 1
        width = max(canvas.winfo_width(), 1)
        left, right, top, row_h = 430, 95, 34, 42
        plot_w = max(width - left - right, 180)
        canvas.create_text(left, 10, anchor="nw", fill=_LABEL,
                           text=f"0 → {max_speed:.1f} T/S（{self.metric_var.get()}）")
        for i, env in enumerate(envs):
            y = top + i * row_h
            ctx = env.max_inference_context or env.max_ready_context
            status = "✓ 推論" if env.max_inference_context else (
                "◐ 啟動" if env.max_ready_context else (
                    "✕ 失敗" if env.failed_runs else "? 未知"))
            # Line 1: full model name (never truncated — many names share a
            # prefix). Line 2: compact backend/ctx/vision/speed parameters.
            label_x = 8
            canvas.create_text(label_x, y + 4, anchor="nw", fill=_TEXT,
                               text=env.key.model_name,
                               font=("Segoe UI", 9, "bold"))
            param2 = (f"{env.key.backend.upper()} · "
                      f"{format_context(ctx)} · "
                      f"V{'ON' if env.key.vision_enabled else 'OFF'} · {status}")
            canvas.create_text(label_x, y + 22, anchor="nw", fill=_LABEL,
                               text=_midclip(param2, 34),
                               font=("Segoe UI", 8))
            speed = self._speed(env)
            bar_w = plot_w * ((speed or 0) / max_speed) if max_speed else 0
            row = self._resolve_envelope_row(env)
            selected = row is not None and row.key in self._selected_keys
            color = self._model_colors.get(env.key.model_name, _BLUE)
            tag = f"env{i}"
            canvas.create_rectangle(left, y + 3, left + max(bar_w, 2), y + 31,
                                    fill=color,
                                    outline="white" if selected else color,
                                    width=2 if selected else 1, tags=tag)
            speed_text = ((f"{speed:.1f} T/S (n={env.stable_sample_count})"
                           if self.metric_var.get() == "穩定中位數"
                           else f"{speed:.1f} T/S")
                          if speed is not None else "無速度樣本")
            text_x = left + bar_w + 6
            anchor = "w"
            text_color = _TEXT
            if text_x > width - 105:
                text_x = left + bar_w - 7
                anchor = "e"
                text_color = "#07101c"
            canvas.create_text(text_x, y + 17, anchor=anchor,
                               fill=text_color, text=speed_text, tags=tag)
            canvas.tag_bind(tag, "<Button-1>",
                            lambda _e, item=env: self._toggle_envelope(item))
        total_h = top + len(envs) * row_h + 10
        canvas.configure(scrollregion=(0, 0, width, total_h))
        verified = sum(e.max_inference_context is not None for e in envs)
        failed = sum(e.failed_runs > 0 for e in envs)
        self.summary_label.config(
            text=(f"Build {report.selected_runtime or '全部'} · {len(envs)} 組配置 · "
                  f"{verified} 組完成推論 · {failed} 組曾啟動失敗\n"
                  "上限均為 Log 中『最大已驗證』，不是未測試過的理論硬體極限。"))

    def _render_tables(self) -> None:
        if not self.report:
            return
        self._env_iids.clear()
        for item in self.envelope_table.get_children():
            self.envelope_table.delete(item)
        for i, env in enumerate(self.report.envelopes):
            iid = f"e{i}"
            self._env_iids[iid] = env
            speed = (f"{env.stable_decode_tps:.1f} (n={env.stable_sample_count}) / "
                     f"{env.observed_max_decode_tps:.1f}"
                     if env.stable_decode_tps is not None
                     and env.observed_max_decode_tps is not None else "—")
            self.envelope_table.insert("", "end", iid=iid, values=(
                env.key.model_name, env.key.backend.upper(),
                (("ON ✓" if env.vision_confirmed_runs else "ON ?")
                 if env.key.vision_enabled else "OFF"),
                f"{env.key.kv} · R:{env.key.reasoning}",
                format_context(env.max_inference_context),
                format_context(env.max_ready_context),
                format_context(env.first_failed_context), speed, env.run_count))

        self._config_iids.clear()
        for item in self.config_table.get_children():
            self.config_table.delete(item)
        for i, row in enumerate(self.report.rows):
            if row.status == "failed":
                continue  # 從未成功啟動的配置不列入比較清單
            iid = f"r{i}"
            self._config_iids[iid] = row.key
            picked = "☑" if row.key in self._selected_keys else "☐"
            status = {"inference": "✓ 推論", "ready": "◐ 啟動",
                      "failed": "✕ 失敗", "unknown": "? 未知"}[row.status]
            speed = (f"{row.stable_decode_tps:.1f} (n={row.stable_sample_count})"
                     if row.stable_decode_tps else "—")
            self.config_table.insert("", "end", iid=iid, values=(
                picked, (f"{row.key.model_name} | "
                         f"{row.key.backend.upper()} "
                         f"{format_context(row.key.configured_context)} "
                         f"V{'ON' if row.key.vision_enabled else 'OFF'} | "
                         f"{status} · {speed} · N{row.sample_count}")))
        self.selection_label.config(text=f"已選 {len(self._selected_keys)} 條曲線")

    def _toggle_envelope_from_table(self, _event=None) -> None:
        selected = self.envelope_table.selection()
        if selected and selected[0] in self._env_iids:
            self._toggle_envelope(self._env_iids[selected[0]])

    def _toggle_envelope(self, env: CapabilityEnvelope) -> None:
        row = self._resolve_envelope_row(env)
        if row:
            self._toggle_key(row.key)
            self.tabs.select(1)

    def _toggle_config_from_table(self, _event=None) -> None:
        selected = self.config_table.selection()
        if selected and selected[0] in self._config_iids:
            self._toggle_key(self._config_iids[selected[0]])

    def _toggle_key(self, key: CapabilityKey) -> None:
        if key in self._selected_keys:
            self._selected_keys.remove(key)
        else:
            self._selected_keys.add(key)
        self._render_all()

    def _clear_curves(self) -> None:
        self._selected_keys.clear()
        self._render_all()

    # ---- curve chart ---------------------------------------------------------

    def _render_curves(self) -> None:
        canvas = getattr(self, "curve_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        rows = sorted(
            (self._row_by_key[k] for k in self._selected_keys
             if k in self._row_by_key and self._row_by_key[k].curve),
            key=lambda r: (r.key.runtime_label, r.key.model_name,
                           r.key.backend, r.key.configured_context or -1,
                           "v" if r.key.vision_enabled else "n",
                           r.key.kv, r.key.gpu_split, r.key.parallel or 0))
        if not rows:
            canvas.create_text(20, 20, anchor="nw", fill=_LABEL,
                               text="尚未選擇有速度樣本的配置")
            return
        width, height = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        left, right, top = 70, 25, 28
        legend_cols = 2
        legend_rows = (len(rows) + legend_cols - 1) // legend_cols
        bottom = 34 + legend_rows * 32 + 12  # axis ticks + legend + title
        plot_w, plot_h = width - left - right, height - top - bottom
        max_x = max(b.bucket_end for row in rows for b in row.curve)
        max_y = max(b.p75 for row in rows for b in row.curve) * 1.12 or 1
        canvas.create_line(left, top, left, top + plot_h, fill=_MUTED)
        canvas.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill=_MUTED)
        for i in range(5):
            value = max_y * i / 4
            y = top + plot_h - plot_h * i / 4
            canvas.create_line(left, y, left + plot_w, y, fill="#202838")
            canvas.create_text(left - 8, y, anchor="e", fill=_LABEL,
                               text=f"{value:.0f}")
        for i in range(5):
            value = max_x * i / 4
            x = left + plot_w * i / 4
            canvas.create_line(x, top, x, top + plot_h, fill="#202838")
            canvas.create_text(x, top + plot_h + 8, anchor="n", fill=_LABEL,
                               text=format_context(int(value)))
        canvas.create_text(8, top, anchor="nw", fill=_LABEL, text="Decode T/S")
        legend_y = top + plot_h + 30
        for i, row in enumerate(rows):
            model_color = self._model_colors.get(row.key.model_name)
            color = model_color or _hex(PALETTE[i % len(PALETTE)])
            points = []
            upper, lower = [], []
            for b in row.curve:
                x_value = (b.bucket_start + b.bucket_end) / 2
                x = left + plot_w * x_value / max_x
                y = top + plot_h - plot_h * b.median / max_y
                points.extend((x, y))
                upper.extend((x, top + plot_h - plot_h * b.p75 / max_y))
                lower[:0] = [x, top + plot_h - plot_h * b.p25 / max_y]
            if len(upper) >= 4:
                canvas.create_polygon(*(upper + lower), fill=color, stipple="gray25",
                                      outline="")
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2, smooth=True)
            for x, y in zip(points[::2], points[1::2]):
                canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                   fill=color, outline="white")
            lx = left + (i % 2) * (plot_w / 2)
            ly = legend_y + (i // legend_cols) * 32
            canvas.create_line(lx, ly + 4, lx + 18, ly + 4, fill=color, width=3)
            canvas.create_text(lx + 23, ly, anchor="nw", fill=_TEXT,
                               text=row.key.model_name,
                               font=("Segoe UI", 8, "bold"))
            canvas.create_text(lx + 23, ly + 14, anchor="nw", fill=_LABEL,
                               text=(f"{row.key.backend.upper()} · "
                                     f"{format_context(row.key.configured_context)} · "
                                     f"V{'ON' if row.key.vision_enabled else 'OFF'}"),
                               font=("Segoe UI", 8))
        canvas.create_text(left + plot_w / 2, height - 14, fill=_LABEL,
                           text="實際使用 Context（雙擊配置可加入/移除）")

    # ---- trade-off -----------------------------------------------------------

    def _tradeoff_wheel(self, event: tk.Event) -> None:
        old_zoom = float(getattr(self, "_tradeoff_zoom", 1.0))
        old_cx = float(getattr(self, "_tradeoff_cx", 0.0))
        old_cy = float(getattr(self, "_tradeoff_cy", 0.0))
        pw = max(self.tradeoff_canvas.winfo_width() - 75 - 35, 1)
        ph = max(self.tradeoff_canvas.winfo_height() - 35 - 65, 1)
        ctx_at = old_cx + (event.x - 75) / pw * self._tradeoff_range_x()
        tps_at = old_cy + (ph - (event.y - 35)) / ph * self._tradeoff_range_y()
        if event.delta > 0:
            self._tradeoff_zoom = min(old_zoom * 1.25, 8.0)
        else:
            self._tradeoff_zoom = max(old_zoom / 1.25, 0.3)
        self._tradeoff_cx = ctx_at - (event.x - 75) / pw * self._tradeoff_range_x()
        self._tradeoff_cy = tps_at - (ph - (event.y - 35)) / ph * self._tradeoff_range_y()
        if abs(self._tradeoff_zoom - old_zoom) < 0.001 and abs(self._tradeoff_cx - old_cx) < 1e-6:
            return
        self._render_tradeoff()

    def _tradeoff_range_x(self) -> float:
        rows = [r for r in (self.report.rows if self.report else [])
                if r.inference_runs and r.key.configured_context is not None
                and r.stable_decode_tps is not None]
        max_ctx = max((int(r.key.configured_context) for r in rows), default=131072)
        return max_ctx * 1.08 / max(float(getattr(self, "_tradeoff_zoom", 1.0)), 0.01)

    def _tradeoff_range_y(self) -> float:
        rows = [r for r in (self.report.rows if self.report else [])
                if r.inference_runs and r.key.configured_context is not None
                and r.stable_decode_tps is not None]
        max_tps = max((float(r.stable_decode_tps) for r in rows), default=1)
        return max_tps * 1.12 / max(float(getattr(self, "_tradeoff_zoom", 1.0)), 0.01)

    def _tradeoff_mouse_down(self, event: tk.Event) -> None:
        self._tradeoff_drag = (event.x, event.y,
                               float(getattr(self, "_tradeoff_cx", 0.0)),
                               float(getattr(self, "_tradeoff_cy", 0.0)))

    def _tradeoff_mouse_drag(self, event: tk.Event) -> None:
        drag = getattr(self, "_tradeoff_drag", None)
        if drag is None:
            return
        sx, sy, scx, scy = drag
        pw = max(self.tradeoff_canvas.winfo_width() - 75 - 35, 1)
        ph = max(self.tradeoff_canvas.winfo_height() - 35 - 65, 1)
        dx = -(event.x - sx) / pw * self._tradeoff_range_x()
        dy =  (event.y - sy) / ph * self._tradeoff_range_y()
        self._tradeoff_cx = scx + dx
        self._tradeoff_cy = scy + dy
        self._render_tradeoff()

    def _tradeoff_reset_zoom(self) -> None:
        self._tradeoff_zoom = 1.0
        self._tradeoff_cx = 0.0
        self._tradeoff_cy = 0.0
        self._render_tradeoff()

    def _render_tradeoff(self) -> None:
        canvas = getattr(self, "tradeoff_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        zoom = float(getattr(self, "_tradeoff_zoom", 1.0))
        cx = float(getattr(self, "_tradeoff_cx", 0.0))
        cy = float(getattr(self, "_tradeoff_cy", 0.0))
        self._zoom_label.config(text=f"縮放 {zoom:.1f}x")
        if not self.report:
            return
        rows = [r for r in self.report.rows if r.inference_runs
                and r.key.configured_context is not None
                and r.stable_decode_tps is not None]
        if not rows:
            canvas.create_text(20, 20, anchor="nw", fill=_LABEL,
                               text="沒有完成推論且具速度樣本的配置")
            return
        frontier = pareto_rows(rows)
        width, height = max(canvas.winfo_width(), 650), max(canvas.winfo_height(), 450)
        left, right, top, bottom = 75, 35, 35, 65
        pw, ph = width - left - right, height - top - bottom
        max_ctx = max(int(r.key.configured_context) for r in rows)
        range_x = max_ctx * 1.08 / max(zoom, 0.01)
        max_tps = max(float(r.stable_decode_tps) for r in rows)
        range_y = max_tps * 1.12 / max(zoom, 0.01)

        def xy(row):
            """data coords → canvas pixels (viewport transforms)."""
            return (left + (int(row.key.configured_context) - cx) / range_x * pw,
                    top + ph - (float(row.stable_decode_tps) - cy) / range_y * ph)

        def point_text(row) -> str:
            spec = (f"{row.key.backend.upper()} · "
                    f"{format_context(row.key.configured_context)} · "
                    f"V{'ON' if row.key.vision_enabled else 'OFF'}")
            if row.stable_bucket:
                spec += (f" · {row.stable_decode_tps:.0f}T/s@"
                         f"{format_context(row.stable_bucket[0])}")
            return f"{row.key.model_name}\n{spec}"

        # axes
        canvas.create_line(left, top, left, top + ph, fill=_MUTED)
        canvas.create_line(left, top + ph, left + pw, top + ph, fill=_MUTED)
        # grid lines
        for i in range(5):
            xv = cx + range_x * i / 4
            x = left + pw * i / 4
            yv = cy + range_y * i / 4
            y = top + ph - ph * i / 4
            if xv >= 0:
                canvas.create_line(x, top, x, top + ph, fill="#202838")
                canvas.create_text(x, top + ph + 12, anchor="n", fill=_LABEL,
                                   text=format_context(int(xv)))
            if yv >= 0:
                canvas.create_line(left, y, left + pw, y, fill="#202838")
                canvas.create_text(left - 8, y, anchor="e", fill=_LABEL,
                                   text=f"{yv:.0f}")
        canvas.create_text(10, top, anchor="nw", fill=_LABEL, text="穩定 Decode T/S")
        canvas.create_text(left + pw / 2, height - 18, fill=_LABEL,
                           text="配置 Context（最大已驗證可推論）")
        # Pareto frontier
        if len(frontier) >= 2:
            line = []
            for row in frontier:
                line.extend(xy(row))
            canvas.create_line(*line, fill=_YELLOW, width=3, dash=(5, 3))
        # points + labels (always visible)
        for i, row in enumerate(rows):
            px, py = xy(row)
            color = self._model_colors.get(row.key.model_name, _BLUE)
            outline = _YELLOW if row in frontier else (
                "white" if row.key in self._selected_keys else color)
            tag = f"point{i}"
            radius = 7 if row in frontier else 5
            canvas.create_oval(px - radius, py - radius, px + radius, py + radius,
                               fill=color, outline=outline, width=2, tags=tag)
            label_text = point_text(row)
            label_x = px + 9
            label_anchor = "sw"
            if px > width - 240:
                label_x = px - 9
                label_anchor = "se"
            canvas.create_text(label_x, py - 6, anchor=label_anchor,
                               fill=_TEXT, text=label_text,
                               font=("Segoe UI", 8), tags=tag)
            canvas.tag_bind(tag, "<Button-1>",
                            lambda _e, key=row.key: self._tradeoff_pick(key))

    def _tradeoff_pick(self, key: CapabilityKey) -> None:
        self._toggle_key(key)
        self.tabs.select(1)
