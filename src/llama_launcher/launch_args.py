"""Pure launch-argument construction for llama-server.

Kept free of Tk and file I/O so the final command line can be built, previewed
and tested without a display. ``ServerManager.start`` uses the same function,
so the settings-dialog preview is always what actually gets executed.
"""
from __future__ import annotations

DEFAULT_SCHEME = "預設"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_CACHE_RAM_MB = 24000

# KV 快取精度選項：(GUI 標籤, profile 的 kv_mode 值)
# 型別對照見 llama-server --cache-type-k 的 allowed values。
KV_OPTIONS = [
    ("F16 / F16（最高精度，最吃顯存）", "f16"),
    ("Q8 / Q8（精度高、省顯存，大型模型推薦）", "q8"),
    ("Q5 / Q5（Q5_0，精度與容量折衷）", "q5"),
    ("Q4 / Q4（正式版，現有設定）", "q4"),
    ("IQ4-NL / IQ4-NL（K/V 同 IQ4-NL，品質略優於 Q4）", "iq4_nl"),
    ("自訂（使用進階參數裡的 -ctk / -ctv）", "custom"),
]
KV_TYPE_BY_MODE = {"f16": "f16", "q8": "q8_0", "q5": "q5_0", "q4": "q4_0", "iq4_nl": "iq4_nl"}
KNOWN_KV_MODES = set(KV_TYPE_BY_MODE) | {"custom"}
KV_MODE_BY_TYPE = {v: k for k, v in KV_TYPE_BY_MODE.items()}

REASONING_EFFORTS = ("default", "minimal", "low", "medium", "high", "xhigh", "max")
REASONING_FORMATS = ("auto", "none", "deepseek", "deepseek-legacy")

# 採樣預設（llama-server --temp 等：server 端 completion 預設值）
SAMPLING_FIELDS = (
    ("temp", "--temp"),
    ("top_p", "--top-p"),
    ("top_k", "--top-k"),
    ("min_p", "--min-p"),
    ("presence_penalty", "--presence-penalty"),
    ("repeat_penalty", "--repeat-penalty"),
)


def normalize_scheme(value) -> str:
    text = str(value or "").strip()
    return text or DEFAULT_SCHEME


def profile_key(profile: dict) -> tuple[str, str]:
    """Profile 的唯一鍵：同一模型可以有多個「方案」(scheme)。"""
    return (str(profile.get("model") or ""), normalize_scheme(profile.get("scheme")))


# ---------------------------------------------------------------- 數值解析
def parse_number(value):
    """解析 profile 裡的數字欄位：空/無效 → None；整數值回 int。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer() and abs(number) < 2 ** 31:
        return int(number)
    return number


def format_number(value) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:g}"


def _arg_value(parts: list[str], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in parts:
            i = parts.index(name)
            if i + 1 < len(parts):
                return parts[i + 1]
    return None


# ---------------------------------------------------------------- KV 模式
def kv_mode_from_profile(profile: dict) -> str:
    """正規化 profile 的 KV 模式。

    舊檔沒有 kv_mode 時由 extra_args 推斷；推不出來（或 K/V 型別不一致、
    非已知型別）一律回 "custom"：保留使用者自己寫的 -ctk/-ctv，不靜默改寫。
    """
    mode = str(profile.get("kv_mode") or "").strip().lower()
    if mode in KNOWN_KV_MODES:
        return mode
    if mode:
        # 存檔裡有明確但未知的值：視為 custom，不靜默改寫
        return "custom"
    # 舊檔沒有 kv_mode：由 extra_args 推斷
    extra = str(profile.get("extra_args") or "").split()
    k = _arg_value(extra, ("-ctk", "--cache-type-k"))
    v = _arg_value(extra, ("-ctv", "--cache-type-v"))
    if not k:
        return "f16"
    if k == v and k in KV_MODE_BY_TYPE:
        return KV_MODE_BY_TYPE[k]
    return "custom"


def kv_label_from_mode(mode: str) -> str:
    return next((label for label, value in KV_OPTIONS if value == mode), KV_OPTIONS[0][0])


def preserve_unmanaged_extra_args(extra: str, manage_kv: bool = True) -> list[str]:
    """移除 GUI 管理的 KV/parallel/Jinja，保留使用者的 -b/-ub 等手動參數。

    manage_kv=False（kv_mode=custom）時保留使用者自己的 -ctk/-ctv。"""
    parts = (extra or "").split()
    out = []
    i = 0
    managed = {"--parallel", "-np"}
    kv_managed = {"-ctk", "--cache-type-k", "-ctv", "--cache-type-v"}
    while i < len(parts):
        part = parts[i]
        if part == "--jinja":
            i += 1
            continue
        if part in managed:
            i += 2
            continue
        if manage_kv and part in kv_managed:
            i += 2
            continue
        out.append(part)
        i += 1
    return out


def parallel_from_profile(profile: dict) -> int:
    """正規化並行請求數（舊檔的 --parallel 在 extra_args 裡，順帶推斷）。"""
    value = parse_number(profile.get("parallel"))
    if value is not None and value >= 1:
        return int(value)
    extra = str(profile.get("extra_args") or "").split()
    value = parse_number(_arg_value(extra, ("--parallel", "-np")))
    if value is not None and value >= 1:
        return int(value)
    return 1


# ---------------------------------------------------------------- Reasoning
def reasoning_effort_value(profile: dict) -> str:
    """正規化 profile 的 reasoning_effort（缺省/未知值 → "default"）。"""
    value = str(profile.get("reasoning_effort") or "default").strip().lower()
    return value if value in REASONING_EFFORTS else "default"


def build_reasoning_args(profile: dict) -> list[str]:
    """組 --reasoning / --reasoning-effort / --reasoning-format / --reasoning-preserve。

    reasoning: off → --reasoning off；on → --reasoning on（+ 相關參數）；
    auto → 不傳旗標，由模型模板決定。
    effort 只在思考模式開啟時有意義，"default" 不傳旗標。
    """
    reasoning = str(profile.get("reasoning") or "off").strip().lower()
    if reasoning not in {"on", "off", "auto"}:
        reasoning = "off"
    args = []
    if reasoning != "auto":
        args += ["--reasoning", reasoning]
    if reasoning == "on":
        effort = reasoning_effort_value(profile)
        if effort != "default":
            args += ["--reasoning-effort", effort]
        fmt = str(profile.get("reasoning_format") or "auto").strip().lower()
        if fmt != "auto":
            args += ["--reasoning-format", fmt]
        preserve = str(profile.get("reasoning_preserve") or "default").strip().lower()
        if preserve == "on":
            args.append("--reasoning-preserve")
        elif preserve == "off":
            args.append("--no-reasoning-preserve")
    return args


# ---------------------------------------------------------------- Server 設定
def server_settings_from_dict(raw: dict | None) -> dict:
    """正規化 settings.json 的 server 區塊（host/port/alias/cache_ram/vulkan_devices）。

    api_key 不放在 settings.json（屬敏感值，存在 secrets/ 目錄），由呼叫方補上。
    """
    raw = raw or {}
    host = str(raw.get("host") or "").strip() or DEFAULT_HOST
    port = DEFAULT_PORT
    try:
        port = int(str(raw.get("port") or "").strip() or DEFAULT_PORT)
    except ValueError:
        pass
    if not 1 <= port <= 65535:
        port = DEFAULT_PORT
    alias = str(raw.get("alias") or "").strip()
    cache_ram = DEFAULT_CACHE_RAM_MB
    try:
        cache_ram = int(str(raw.get("cache_ram_mb") or "").strip() or DEFAULT_CACHE_RAM_MB)
    except ValueError:
        pass
    if cache_ram < 0:
        cache_ram = DEFAULT_CACHE_RAM_MB
    vulkan_devices = str(raw.get("vulkan_devices") or "").strip()
    return {
        "host": host,
        "port": port,
        "alias": alias,
        "api_key": str(raw.get("api_key") or "").strip(),
        "cache_ram_mb": cache_ram,
        "vulkan_devices": vulkan_devices,
    }


def vulkan_device_list(gpu_count: int | None) -> str:
    """Vulkan 後端的 --device 值：依 GPU 數自動產生；未知時退回第一張卡。"""
    if gpu_count is None or gpu_count < 1:
        return "Vulkan0"
    return ",".join(f"Vulkan{i}" for i in range(gpu_count))


# ---------------------------------------------------------------- 主建構
def build_server_args(
    profile: dict,
    ctx_tokens: int,
    server_settings: dict,
    model_path,
    mmproj_path=None,
    vulkan_gpu_count: int | None = None,
) -> list[str]:
    """組出 llama-server 完整參數列（不含執行檔路徑）。

    raw_args 非空時：「完整參數模式」，只保留 -m/--mmproj，其餘完全由
    raw_args 決定（bat 式全權控制）。
    """
    args = ["-m", str(model_path)]
    if mmproj_path is not None:
        args += ["--mmproj", str(mmproj_path)]

    raw = str(profile.get("raw_args") or "").strip()
    if raw:
        args += raw.split()
        return args

    ngl = parse_number(profile.get("ngl"))
    if ngl is None or ngl < 1:
        ngl = 999
    args += [
        "-ngl", str(int(ngl)),
        "-c", str(int(ctx_tokens)),
        "--host", server_settings["host"],
        "--port", str(server_settings["port"]),
    ]
    if server_settings.get("alias"):
        args += ["--alias", server_settings["alias"]]
    if server_settings.get("api_key"):
        args += ["--api-key", server_settings["api_key"]]
    args += ["-sm", "layer"]
    if server_settings["cache_ram_mb"] >= 0:
        args += ["-cram", str(server_settings["cache_ram_mb"])]

    backend = str(profile.get("backend", "cuda")).strip().lower()
    if backend == "vulkan":
        configured_devices = str(
            server_settings.get("vulkan_devices") or "").strip()
        args += ["--device", configured_devices or
                 vulkan_device_list(vulkan_gpu_count)]
    gpu_split = str(profile.get("gpu_split") or "").strip()
    if gpu_split:
        args += ["-ts", gpu_split]
    args += ["-mg", "0"]

    # 可選 runtime 參數（留空 = 不傳，維持 llama.cpp 預設）
    flash_attn = str(profile.get("flash_attn") or "auto").strip().lower()
    if flash_attn in ("on", "off"):
        args += ["--flash-attn", flash_attn]
    args.append("--kv-unified" if bool(profile.get("kv_unified", True)) else "--no-kv-unified")
    if str(profile.get("fit") or "on").strip().lower() == "off":
        args += ["--fit", "off"]
    threads = parse_number(profile.get("threads"))
    if threads is not None:
        args += ["--threads", format_number(threads)]
    threads_batch = parse_number(profile.get("threads_batch"))
    if threads_batch is not None:
        args += ["--threads-batch", format_number(threads_batch)]
    ctx_checkpoints = parse_number(profile.get("ctx_checkpoints"))
    if ctx_checkpoints is not None:
        args += ["--ctx-checkpoints", format_number(ctx_checkpoints)]

    # extra_args（-b/-ub 等手動參數）；KV/parallel 由 GUI 管理
    kv_mode = kv_mode_from_profile(profile)
    if profile.get("extra_args"):
        args += preserve_unmanaged_extra_args(
            str(profile["extra_args"]), manage_kv=(kv_mode != "custom"))
    if kv_mode in KV_TYPE_BY_MODE:
        kv_type = KV_TYPE_BY_MODE[kv_mode]
        args += ["-ctk", kv_type, "-ctv", kv_type]
    if "--parallel" not in args:
        args += ["--parallel", str(parallel_from_profile(profile))]

    args += build_reasoning_args(profile)
    if profile.get("jinja") and "--jinja" not in args:
        args.append("--jinja")
    if profile.get("mtp"):
        args += ["--spec-type", "draft-mtp"]
        n_max = parse_number(profile.get("spec_draft_n_max"))
        if n_max is not None:
            args += ["--spec-draft-n-max", format_number(n_max)]

    for field, flag in SAMPLING_FIELDS:
        value = parse_number(profile.get(field))
        if value is not None:
            args += [flag, format_number(value)]
    return args


def format_command(binary, args: list[str]) -> str:
    """把參數列排成可讀的一行指令（預覽／log 用）。"""
    return " ".join([str(binary)] + [str(a) for a in args])
