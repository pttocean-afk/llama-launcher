"""Tests for the pure launch-argument builder (launch_args.py).

Covers the bat-parameter matrix, the KV no-silent-rewrite regression,
raw-args override, and server-settings normalization.
"""
from llama_launcher.launch_args import (
    DEFAULT_CACHE_RAM_MB,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_SCHEME,
    build_reasoning_args,
    build_server_args,
    format_command,
    kv_label_from_mode,
    kv_mode_from_profile,
    normalize_scheme,
    parallel_from_profile,
    preserve_unmanaged_extra_args,
    profile_key,
    reasoning_effort_value,
    server_settings_from_dict,
    vulkan_device_list,
)


def _profile(**kw):
    base = {
        "name": "A",
        "model": "a.gguf",
        "scheme": DEFAULT_SCHEME,
        "mmproj": "",
        "default_ctx": 131072,
        "reasoning": "off",
        "gpu_split": "",
        "backend": "cuda",
        "jinja": False,
        "extra_args": "",
        "kv_mode": "q4",
        "mtp": False,
        "ngl": 999,
        "parallel": 1,
        "flash_attn": "auto",
        "kv_unified": True,
        "fit": "on",
        "starred": False,
    }
    base.update(kw)
    return base


def _settings(**kw):
    base = server_settings_from_dict({})
    base.update(kw)
    return base


def has_seq(args, seq):
    """subsequence 檢查：seq 是否為 args 的連續子列。"""
    seq = list(seq)
    for i in range(len(args) - len(seq) + 1):
        if args[i:i + len(seq)] == seq:
            return True
    return False


def _args(profile, settings=None, **kw):
    mmproj = profile.get("mmproj")
    kw.setdefault("mmproj_path", f"/models/{mmproj}" if mmproj else None)
    return build_server_args(
        profile, profile.get("default_ctx", 131072),
        _settings() if settings is None else settings,
        "/models/a.gguf", **kw)


# ------------------------------------------------------------------ 基本組合
def test_default_args():
    args = _args(_profile())
    assert args[:2] == ["-m", "/models/a.gguf"]
    assert has_seq(args, ["-ngl", "999"])
    assert has_seq(args, ["-c", "131072"])
    assert has_seq(args, ["--host", "0.0.0.0"])
    assert has_seq(args, ["--port", "8080"])
    assert has_seq(args, ["-sm", "layer"])
    assert has_seq(args, ["-cram", "24000"])
    assert has_seq(args, ["-mg", "0"])
    assert has_seq(args, ["--kv-unified"])
    assert has_seq(args, ["-ctk", "q4_0", "-ctv", "q4_0"])
    assert has_seq(args, ["--parallel", "1"])
    assert has_seq(args, ["--reasoning", "off"])
    # 未設定的可選參數不傳
    for flag in ("--flash-attn", "--fit", "--threads", "--threads-batch",
                 "--ctx-checkpoints", "--alias", "--api-key", "--jinja"):
        assert flag not in args


def test_host_port_alias_api_key():
    settings = _settings(host="127.0.0.1", port=8081,
                         alias="qwen-code", api_key="sk-llama-test")
    args = _args(_profile(), settings)
    assert has_seq(args, ["--host", "127.0.0.1"])
    assert has_seq(args, ["--port", "8081"])
    assert has_seq(args, ["--alias", "qwen-code"])
    assert has_seq(args, ["--api-key", "sk-llama-test"])


# ------------------------------------------------------------------ KV 快取
def test_kv_q8_emits_q8_0():
    """回歸：選 Q8 時必須傳 q8_0，不得被靜默改寫成 f16。"""
    args = _args(_profile(kv_mode="q8"))
    assert has_seq(args, ["-ctk", "q8_0", "-ctv", "q8_0"])
    assert "f16" not in args


def test_kv_mode_matrix():
    for mode, expected in [("f16", "f16"), ("q8", "q8_0"), ("q5", "q5_0"),
                           ("q4", "q4_0"), ("iq4_nl", "iq4_nl")]:
        args = _args(_profile(kv_mode=mode))
        assert [t for t in args if t == "-ctk"] and \
            args[args.index("-ctk") + 1] == expected, mode
        assert args[args.index("-ctv") + 1] == expected, mode


def test_kv_custom_preserves_user_types():
    """custom 模式：保留使用者自己寫的 -ctk/-ctv（含 K/V 不同型別）。"""
    p = _profile(kv_mode="custom", extra_args="-ctk q8_0 -ctv f16")
    args = _args(p)
    assert args.index("-ctk") < args.index("q8_0")
    assert args[args.index("-ctk") + 1] == "q8_0"
    assert args[args.index("-ctv") + 1] == "f16"
    assert args.count("-ctk") == 1


def test_kv_inferred_from_legacy_extra_args():
    # 舊檔沒有 kv_mode：由 extra_args 推斷
    assert kv_mode_from_profile(
        {"extra_args": "-ctk q8_0 -ctv q8_0 --parallel 1"}) == "q8"
    assert kv_mode_from_profile({"extra_args": "--parallel 1"}) == "f16"
    # K/V 不一致 → custom（不靜默改寫）
    assert kv_mode_from_profile(
        {"extra_args": "-ctk q8_0 -ctv f16"}) == "custom"
    # 未知型別 → custom
    assert kv_mode_from_profile({"kv_mode": "weird"}) == "custom"


def test_kv_label_lookup():
    assert "Q8" in kv_label_from_mode("q8")
    assert kv_label_from_mode("bogus")  # 回第一項，不炸


# ------------------------------------------------------------------ raw 模式
def test_raw_args_override_everything():
    p = _profile(kv_mode="q8", mtp=True, jinja=True, ngl=64,
                 raw_args="--host 0.0.0.0 --port 8081 -ngl 99 --jinja")
    args = _args(p)
    assert args == ["-m", "/models/a.gguf",
                    "--host", "0.0.0.0", "--port", "8081",
                    "-ngl", "99", "--jinja"]


def test_raw_args_keeps_mmproj():
    p = _profile(raw_args="-ngl 99", mmproj="mm.gguf")
    args = _args(p)
    assert args[:4] == ["-m", "/models/a.gguf", "--mmproj", "/models/mm.gguf"]
    assert args[4:] == ["-ngl", "99"]


# ------------------------------------------------------------------ MTP
def test_mtp_with_n_max():
    args = _args(_profile(mtp=True, spec_draft_n_max="5"))
    i = args.index("--spec-type")
    assert args[i:i + 4] == ["--spec-type", "draft-mtp",
                             "--spec-draft-n-max", "5"]


def test_mtp_without_n_max_uses_default():
    args = _args(_profile(mtp=True))
    assert args.count("--spec-type") == 1
    assert "--spec-draft-n-max" not in args


def test_mtp_off_no_spec_flags():
    args = _args(_profile())
    assert "--spec-type" not in args


# ------------------------------------------------------------------ 採樣
def test_sampling_flags():
    p = _profile(temp="0.8", top_p="0.95", top_k="20", min_p="0.05",
                 presence_penalty="0.1", repeat_penalty="1.0")
    args = _args(p)
    assert has_seq(args, ["--temp", "0.8"])
    assert has_seq(args, ["--top-p", "0.95"])
    assert has_seq(args, ["--top-k", "20"])
    assert has_seq(args, ["--min-p", "0.05"])
    assert has_seq(args, ["--presence-penalty", "0.1"])
    assert has_seq(args, ["--repeat-penalty", "1"])  # 1.0 格式化成 "1"


def test_sampling_empty_omitted():
    args = _args(_profile())
    for flag in ("--temp", "--top-p", "--top-k", "--min-p",
                 "--presence-penalty", "--repeat-penalty"):
        assert flag not in args


# ------------------------------------------------------------------ runtime
def test_flash_attn():
    assert "--flash-attn" not in _args(_profile(flash_attn="auto"))
    assert has_seq(_args(_profile(flash_attn="on")), ["--flash-attn", "on"])
    assert has_seq(_args(_profile(flash_attn="off")), ["--flash-attn", "off"])


def test_kv_unified_and_fit():
    on = _args(_profile(kv_unified=True, fit="on"))
    assert "--kv-unified" in on and "--no-kv-unified" not in on
    assert "--fit" not in on
    off = _args(_profile(kv_unified=False, fit="off"))
    assert "--no-kv-unified" in off
    assert has_seq(off, ["--fit", "off"])


def test_threads_and_checkpoints():
    p = _profile(threads="6", threads_batch="10", ctx_checkpoints="32")
    args = _args(p)
    assert has_seq(args, ["--threads", "6"])
    assert has_seq(args, ["--threads-batch", "10"])
    assert has_seq(args, ["--ctx-checkpoints", "32"])
    empty = _args(_profile())
    for flag in ("--threads", "--threads-batch", "--ctx-checkpoints"):
        assert flag not in empty


def test_gpu_split_and_ngl():
    args = _args(_profile(gpu_split="16,8", ngl=99))
    assert has_seq(args, ["-ts", "16,8"])
    assert has_seq(args, ["-ngl", "99"])


def test_vulkan_device_list():
    two = build_server_args(_profile(backend="vulkan"), 4096,
                            _settings(), "/models/a.gguf",
                            vulkan_gpu_count=2)
    assert has_seq(two, ["--device", "Vulkan0,Vulkan1"])
    one = build_server_args(_profile(backend="vulkan"), 4096,
                            _settings(), "/models/a.gguf", vulkan_gpu_count=1)
    assert has_seq(one, ["--device", "Vulkan0"])
    unknown = build_server_args(_profile(backend="vulkan"), 4096,
                                _settings(), "/models/a.gguf")
    assert has_seq(unknown, ["--device", "Vulkan0"])
    configured = build_server_args(
        _profile(backend="vulkan"), 4096,
        _settings(vulkan_devices="Vulkan1"), "/models/a.gguf",
        vulkan_gpu_count=2)
    assert has_seq(configured, ["--device", "Vulkan1"])
    assert "Vulkan0,Vulkan1" not in configured
    cuda = _args(_profile(backend="cuda"))
    assert "--device" not in cuda


def test_vulkan_device_list_function():
    assert vulkan_device_list(0) == "Vulkan0"
    assert vulkan_device_list(None) == "Vulkan0"
    assert vulkan_device_list(1) == "Vulkan0"
    assert vulkan_device_list(3) == "Vulkan0,Vulkan1,Vulkan2"


def test_cache_ram():
    assert has_seq(_args(_profile()), ["-cram", str(DEFAULT_CACHE_RAM_MB)])
    custom = _args(_profile(), _settings(cache_ram_mb=32768))
    assert has_seq(custom, ["-cram", "32768"])
    disabled = _args(_profile(), _settings(cache_ram_mb=-1))
    assert "-cram" not in disabled


# ------------------------------------------------------------------ reasoning
def test_reasoning_matrix():
    assert build_reasoning_args({"reasoning": "off"}) == ["--reasoning", "off"]
    assert build_reasoning_args({"reasoning": "auto"}) == []
    assert build_reasoning_args(
        {"reasoning": "on"}) == ["--reasoning", "on"]
    on = build_reasoning_args(
        {"reasoning": "on", "reasoning_effort": "low",
         "reasoning_format": "deepseek", "reasoning_preserve": "on"})
    assert ["--reasoning", "on", "--reasoning-effort", "low",
            "--reasoning-format", "deepseek", "--reasoning-preserve"] == on
    default_effort = build_reasoning_args(
        {"reasoning": "on", "reasoning_effort": "default"})
    assert default_effort == ["--reasoning", "on"]
    preserve_off = build_reasoning_args(
        {"reasoning": "on", "reasoning_preserve": "off"})
    assert preserve_off == ["--reasoning", "on", "--no-reasoning-preserve"]
    # 未知 reasoning 值 → off
    assert build_reasoning_args({"reasoning": "weird"}) == ["--reasoning", "off"]


def test_reasoning_effort_normalization():
    assert reasoning_effort_value({}) == "default"
    assert reasoning_effort_value({"reasoning_effort": "high"}) == "high"
    assert reasoning_effort_value({"reasoning_effort": "nope"}) == "default"


def test_reasoning_in_full_args():
    p = _profile(reasoning="on", reasoning_effort="low",
                 reasoning_format="auto", reasoning_preserve="default")
    args = _args(p)
    assert has_seq(args, ["--reasoning", "on", "--reasoning-effort", "low"])


# ------------------------------------------------------------------ 其他
def test_jinja():
    assert "--jinja" in _args(_profile(jinja=True))
    assert "--jinja" not in _args(_profile(jinja=False))


def test_extra_args_preserved_but_managed_stripped():
    p = _profile(extra_args="-b 512 --parallel 4 -ctk q8_0 -ctv q8_0 --jinja")
    args = _args(p)
    assert "-b" in args and "512" in args
    # GUI 管理的 KV / parallel / jinja 被剔除，由選項產生
    assert args.count("-ctk") == 1  # 只有 q4_0 那一組
    assert args[args.index("-ctk") + 1] == "q4_0"


def test_preserve_unmanaged_extra_args():
    assert preserve_unmanaged_extra_args("-b 512 --parallel 4 -ctk q8_0") == \
        ["-b", "512"]
    kept = preserve_unmanaged_extra_args("-ctk q8_0 -ctv f16", manage_kv=False)
    assert kept == ["-ctk", "q8_0", "-ctv", "f16"]


def test_parallel_from_profile():
    assert parallel_from_profile({}) == 1
    assert parallel_from_profile({"parallel": "4"}) == 4
    assert parallel_from_profile({"parallel": 0}) == 1
    assert parallel_from_profile({"extra_args": "--parallel 3"}) == 3
    assert parallel_from_profile({"parallel": "2",
                                  "extra_args": "--parallel 9"}) == 2


def test_server_settings_normalization():
    s = server_settings_from_dict(None)
    assert s["host"] == DEFAULT_HOST
    assert s["port"] == DEFAULT_PORT
    assert s["cache_ram_mb"] == DEFAULT_CACHE_RAM_MB
    s = server_settings_from_dict(
        {"host": " ", "port": "not-a-port", "cache_ram_mb": "x",
         "alias": " a ", "vulkan_devices": "Vulkan0"})
    assert s["host"] == DEFAULT_HOST
    assert s["port"] == DEFAULT_PORT
    assert s["cache_ram_mb"] == DEFAULT_CACHE_RAM_MB
    assert s["alias"] == "a"
    assert s["vulkan_devices"] == "Vulkan0"
    assert server_settings_from_dict({"port": 99999})["port"] == DEFAULT_PORT
    assert server_settings_from_dict({"port": 8081})["port"] == 8081


# ------------------------------------------------------------------ 方案鍵
def test_profile_key_and_scheme():
    assert normalize_scheme(None) == DEFAULT_SCHEME
    assert normalize_scheme(" code ") == "code"
    assert normalize_scheme("") == DEFAULT_SCHEME
    p = _profile(scheme="code")
    assert profile_key(p) == ("a.gguf", "code")
    assert profile_key(_profile()) == ("a.gguf", DEFAULT_SCHEME)


def test_format_command():
    text = format_command("llama-server.exe", ["-m", "a.gguf", "--port", "8081"])
    assert text == "llama-server.exe -m a.gguf --port 8081"


def test_user_bat_profile_reproduces():
    """網友的 bat 參數以 raw_args 輸入時，輸出與 bat 一致（除 -m 外）。"""
    p = _profile(
        raw_args=(
            "--host 0.0.0.0 --port 8081 --api-key sk-llama-test "
            "--alias qwen3.8-27b-code --reasoning-format auto "
            "--reasoning-preserve --fit off -ngl 99 --flash-attn on --jinja "
            "--kv-unified --threads 6 --threads-batch 10 --parallel 1 "
            "--cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp "
            "--spec-draft-n-max 5 --ctx-checkpoints 32 --reasoning on "
            "--reasoning-effort low --cache-ram 32768 --temp 0.8 "
            "--top-p 0.95 --top-k 20 --min-p 0.05 --presence-penalty 0.1 "
            "--repeat-penalty 1.0 --ctx-size 200000"))
    args = _args(p)
    assert args[0] == "-m"
    assert has_seq(args, ["--port", "8081"])
    assert has_seq(args, ["--spec-draft-n-max", "5"])
    assert args.count("--cache-type-k") == 1
    assert args[-2:] == ["--ctx-size", "200000"]
