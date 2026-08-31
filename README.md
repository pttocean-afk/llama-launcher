<div align="center">

# 🦙 Llama Launcher

**本機 `llama.cpp` 伺服器的跨平台桌面控制中心 — Windows / Linux**

啟動、接管、檢視、遠端控制，全部在一個畫面搞定，不用再開 terminal 打指令。

![Windows](https://img.shields.io/badge/Windows-✓-0078D6) ![Linux](https://img.shields.io/badge/Linux-x64-✓-A855F7) ![License](https://img.shields.io/badge/License-MIT-blue) ![tests](https://img.shields.io/badge/tests-38%20passed-green)

</div>

---

## 為什麼用它

跑本機 `llama.cpp` 其實很繁瑣：要記得一堆參數、在 terminal 看 log、手動管理 GPU 分配、還想從手機連回來。**Llama Launcher** 把這些全部包起來：

- ✅ 把各模型存成 profile，一鍵啟動／停止
- ✅ 自動接管已在跑的 `llama-server --port 8080`，不重啟、不停你的東西
- ✅ 整合 Log 檢視，看到 `Vulkan` fallback 還會主動提醒並停止
- ✅ **Tailscale 遠端控制**：從手機／別台電腦安全的連回來
- ✅ 雙平台共用同一套核心，Windows / Linux 行為一致

---

## ✨ 功能總覽

| 功能 | 說明 |
|---|---|
| 🗂️ Model profiles | 每個模型的 context、GPU 分配、KV 精度、reasoning、思考強度、vision 各自獨立存檔 |
| 🧠 思考強度 | llama.cpp `--reasoning-effort`（default/minimal/low/medium/high/xhigh/max），思考模式開啟時可選 |
| 🎮 GPU 分配 | 自動，或自訂各卡層數（如 `16,8`），常用分配自動記住、可刪除 |
| 🚀 一鍵啟動 | 幫你組好 `-ngl -c -ts -ctk/ctv --parallel --reasoning-effort` 等參數，直接 launch |
| 📄 即時 Log | 內嵌 log 面板，偵測到 Vulkan 慢速退化會警告 |
| 🔌 Process 接管 | 偵測既有的 8080 llama-server，直接納管，不重啟 |
| 📶 Tailscale Serve | HTTPS 遠端控制頁＋Bearer token，只綁 localhost 再透過 Tailscale 代理 |
| 💼 全域／個別設定分離 | 開機啟動、llama.cpp 路徑、遠端存取＝全域；模型參數＝個別 |
| ♻️ 舊資料遷移 | 從舊版 launcher 匯入 profiles／token，一鍵搬家 |
| 📦 Profile 匯出／匯入 | 可攜 JSON（不含本機路徑），換機/分享輕鬆 |
| 📊 效能分析 | 解析 logs 依 10K used-context bucket 聚合 decode/prefill 曲線、公平性警告、6 種格式匯出 |

---

## 📸 畫面

![LlamaLauncher 主畫面](docs/screenshot.png)

*主畫面：Favorite Models、整合 Log、一鍵 Start/Stop*

![LlamaLauncher 遠端啟動介面](docs/remote.png)

*Tailscale 遠端控制頁：狀態、模型、Start/Stop、Log*

---

## 🚀 快速開始

### Windows

1. 到本專案的 [Releases](https://github.com/pttocean-afk/llama-launcher/releases)下載 `LlamaLauncher-Setup-x64.exe` 安裝，或解壓 `LlamaLauncher-Portable-x64.zip`
2. 啟動後選擇**包含 `llama-server.exe` 的資料夾**
3. 把 GGUF 模型放進它的 `models` 子資料夾
4. 選好模型 → 按 **START SERVER**

### Linux (x86_64)

```bash
wget -c https://github.com/pttocean-afk/llama-launcher/releases/latest/download/LlamaLauncher-Linux-x86_64.tar.gz  # 把 your-user 換成你的帳號
tar -xzf LlamaLauncher-Linux-x86_64.tar.gz
./LlamaLauncher/LlamaLauncher
```

> 首次啟動會請你挑 `llama-server` 所在資料夾。

---

## 📶 遠端控制（Tailscale）

1. 主畫面 → **Settings → Remote Access**
2. 系統偵測 Tailscale（需先安裝並登入）
3. 完成後取得：
   - **HTTPS 網址**（例：`https://your-pc.tail.ts.net`）
   - **控制 Token**（私人字串）
4. 手機／別台電腦開網址 → 貼 Token → 即可看狀態、選模型、啟動／停止

**安全設計**：控制 API 只綁 `127.0.0.1:8765`，對外一律透過 Tailscale Serve HTTPS 代理；`/api/*` 全需 Bearer token。推理埠 `8080` 不直接暴露。

---

## 🗂️ 資料存放位置

| 平台 | 使用者資料 |
|---|---|
| Windows | `%LOCALAPPDATA%\LlamaLauncher` |
| Linux | `$XDG_DATA_HOME/LlamaLauncher` 或 `~/.local/share/LlamaLauncher` |
| 測試覆寫 | 用 `LLAMA_LAUNCHER_DATA_DIR` 環境變數 |

程式與資料分離：移除「程式」不影響你的 profiles／設定。

---

## 🖥️ 平台行為

| | Windows | Linux |
|---|---|---|
| 執行檔 | `llama-server.exe` | `llama-server` |
| Process 探測 | `psutil` | `psutil` |
| 遠端存取 | Tailscale Serve HTTPS | Tailscale Serve HTTPS |
| 關閉按鈕預設 | 縮到系統匣 | 安全退出（tray 支援因桌面而異） |
| 發佈產物 | Setup EXE ＋ Portable ZIP | Portable x86_64 tar.gz |

電腦專屬路徑皆為本機設定；匯入 profile 不假設 Windows 路徑在 Linux 也有效。

---

## 📊 效能分析

主畫面 → **📊 效能分析**：解析所有 log，依「10K used context」bucket 聚合效能曲線，比較不同 runtime／backend／KV／reasoning 的 decode／prefill 表現。

- **來源**：預設掃描使用者資料目錄（Windows `%LOCALAPPDATA%\LlamaLauncher\logs`、Linux `~/.local/share/LlamaLauncher/logs`）下所有 `*.log`；若存在 `<llama.cpp 目錄>/launcher-app/logs`（舊版 launcher）會自動偵測納入，也可另匯入單一 log 或整個資料夾。**log 一律唯讀**——工具不會截斷、覆寫、輪替或刪除任何 log。
- **支援格式**：本 launcher 寫入 header（`# <timestamp>  <profile>` ＋ `# <argv>`）的 `llama-server` log；body 需含 `print_timing`／`stop processing` 行。缺 header 時 metadata 會推斷（附警告）；無法解析的檔案會列在錯誤清單、不影響其他檔案。
- **比較維度**：Runtime（BeeLlama v0.4.4 / b10621 / legacy…）、Backend（CUDA/Vulkan）、KV pair（`q4_0/q4_0`）、Reasoning（on/off/auto/unknown）、Reasoning effort（default/minimal/low/medium/high/xhigh/max）、Max ctx、Vision（yes/no）、個別 run（每個 log 檔）。
- **Vision loaded**：啟動時是否載入 `--mmproj` 多模態模型檔（yes＝有載入、no＝沒有、unknown＝無法判斷），反映「啟動設定」，不代表某筆請求實際發了圖片。
- **Max ctx vs used context**：Max ctx 是啟動時 `-c` 的設定上限；曲線與表格使用「實際 used context」（請求結束時的上下文位置，如 `stop processing` 的 `n_tokens`），歸入 `used_context // 10000 × 10000` 的 bucket（0、10000、20000…）。
- **10K bucket 統計**：每個 bucket 獨立計算 decode／prefill 的 n、中位數、P25、P75、min、max（percentile 為確定性線性內插）；**空 bucket 不補值**，沒資料就不顯示。
- **公平性警告**：同一系列內 model／runtime／backend／KV／reasoning／context／vision／GPU 分配／batch 不一致會顯示警告（例如同系列 reasoning on/off 混合）；跨系列差異標註「observational, not a controlled comparison」。
- **排除規則**：預設排除 generated < 20 tokens 的請求（UI 可調 0/10/20/50/100）；未完成（無 stop）與缺 used context 的請求也排除，HTML/Markdown 匯出會顯示各排除類型的計數。
- **匯出**：HTML（自包含、無外部依賴）、PNG、SVG、raw CSV（每請求一行）、aggregate CSV（每 bucket×metric 一行）、Markdown；**永不覆寫**既有檔，自動加 `-1`、`-2`… 後綴。

---

## 🧪 開發 / 測試

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

**Windows build**

```powershell
.\scripts\build-windows.ps1
.\scripts\build-installer.ps1
```

> 本機手動打包的完整 SOP（WSL→Windows interop：Python/PyInstaller/Inno Setup 路徑、逐條指令、踩坑清單）見 [docs/BUILD-WINDOWS.md](docs/BUILD-WINDOWS.md)。

**Linux build**（用 distro 系統 Python，須含 Tk）

```bash
sudo apt-get install python3-tk python3-venv xvfb
python3 -m venv .venv-build
PATH="$PWD/.venv-build/bin:$PATH" bash scripts/build-linux.sh
```

CI：`.github/workflows/` 提供 Windows / Linux 自動 build＋test。

---

## 🔒 安全性

- 控制 Token 固定長度比對，防時序攻擊
- 資料檔 atomic 寫入，避免中途損毀
- 遠端 dashboard 以 DOM 建構渲染（無 `innerHTML` 注入）
- start/stop 共用作業鎖，避免多執行緒同時啟停
- 本機資料目錄結構在 `.gitignore` 排除 Token／模型／絕對路徑／Log

---

## 📄 License

本專案以 **MIT License** 釋出。請見 [LICENSE](LICENSE)。

---

## 🙏 致謝

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [pystray](https://github.com/moses-palmer/pystray)、[psutil](https://github.com/giampaolo/psutil)、[Tailscale](https://tailscale.com)

---

# English

## 🦙 Llama Launcher

**A cross-platform desktop control center for local `llama.cpp` servers — Windows / Linux.**

Launch, adopt, log, and remotely control your local llama.cpp server from one
window — no terminal, no remembering flags.

## Why

Running local llama.cpp is fiddly: model-specific flags, terminal logs, manual
GPU split, and remote access are a pain. **Llama Launcher** wraps it all up:

- **Model profiles** — save each model's context, GPU split, KV precision,
  reasoning, and vision settings independently, then launch with one click.
- **Process adoption** — detects an already-running `llama-server --port 8080`
  and manages it without a restart.
- **Integrated log panel** — warnings when the Vulkan runtime falls back to a
  slow no-parallel-pipeline mode.
- **Tailscale remote control** — a secure HTTPS control page (Bearer token)
  accessible from your phone or another machine.
- **One shared core on both OSes** — identical behavior on Windows and Linux.

## Feature snapshot

| Feature | Description |
|---|---|
| Model profiles | Per-model context / GPU split / KV / reasoning / thinking intensity / vision |
| Thinking intensity | llama.cpp `--reasoning-effort` (default/minimal/low/medium/high/xhigh/max), chosen when reasoning is on |
| GPU split | Auto, or custom layers per GPU (e.g. `16,8`); presets remembered & removable |
| One-click start | Builds `-ngl -c -ts -ctk/ctv --parallel --reasoning-effort` args for you |
| Live logs | Embedded log viewer with Vulkan-fallback warnings |
| Process adoption | Manages an existing 8080 server without restarting it |
| Tailscale Serve | HTTPS remote page + Bearer token, localhost-bound behind Tailscale |
| Global / per-model settings | Autostart, llama.cpp path, remote access = global |
| Legacy migration | Import old profiles / token in one click |
| Profile export/import | Portable JSON (no local absolute paths) |
| Performance analysis | 10K used-context bucket decode/prefill curves, fairness warnings, 6-format export |

## Quick start

**Windows:** grab `LlamaLauncher-Setup-x64.exe` or `LlamaLauncher-Portable-x64.zip`
from [Releases](https://github.com/pttocean-afk/llama-launcher/releases). Pick the
folder containing `llama-server.exe`, drop GGUFs into its `models` subfolder,
select a model, press **START SERVER**.

**Linux (x86_64):**

```bash
wget -c https://github.com/pttocean-afk/llama-launcher/releases/latest/download/LlamaLauncher-Linux-x86_64.tar.gz
tar -xzf LlamaLauncher-Linux-x86_64.tar.gz
./LlamaLauncher/LlamaLauncher
```

## Remote access (Tailscale)

Main screen → **Settings → Remote Access**. The app sets up Tailscale Serve and
shows an HTTPS URL plus a private control token. Open the URL elsewhere, paste
the token, and you can view status, pick a model, and start/stop the server.

The control API binds only to `127.0.0.1:8765` and is exposed exclusively
through Tailscale Serve HTTPS; every `/api/*` route requires a Bearer token.
The inference port `8080` is never exposed as the control channel.

## Data locations

| Platform | User data |
|---|---|
| Windows | `%LOCALAPPDATA%\LlamaLauncher` |
| Linux | `$XDG_DATA_HOME/LlamaLauncher` or `~/.local/share/LlamaLauncher` |
| Test override | `LLAMA_LAUNCHER_DATA_DIR` env var |

## Performance analysis

Main screen → **📊 效能分析 (Performance analysis)**: parses every log and
aggregates timings into 10K used-context buckets, so you can compare
decode/prefill performance across runtimes, backends, KV precision, reasoning,
and more.

- **Sources** — scans all `*.log` in the user data directory by default; a
  legacy `<llama dir>/launcher-app/logs` is auto-detected. You can also import
  individual logs or whole folders. **Logs are read-only**: the tool never
  truncates, rewrites, rotates, or deletes a log.
- **Supported format** — `llama-server` logs written with this launcher's
  header (`# <timestamp>  <profile>` + `# <argv>`); the body needs
  `print_timing` / `stop processing` lines. Without a header the metadata is
  inferred (with a warning); unparseable files are listed and skipped without
  affecting the others.
- **Comparison dimensions** — runtime (BeeLlama v0.4.4 / b10621 / legacy…),
  backend (CUDA/Vulkan), KV pair (`q4_0/q4_0`), reasoning (on/off/auto/unknown),
  reasoning effort (default/minimal/low/medium/high/xhigh/max), max ctx,
  vision (yes/no), or individual run (each log file).
- **Vision loaded** — whether an `--mmproj` multimodal model file was loaded at
  launch (yes / no / unknown). It reflects the launch configuration, not
  whether a particular request actually sent an image.
- **Max ctx vs used context** — max ctx is the configured `-c` ceiling; the
  curves and table use the actual used context at request end (e.g. the
  `n_tokens` in `stop processing`), bucketed as `used_context // 10000 × 10000`
  (0, 10000, 20000…).
- **10K bucket stats** — per bucket and per metric: n, median, P25, P75, min,
  max (deterministic linear-interpolation percentiles). Empty buckets are never
  fabricated — no data, no row.
- **Fairness warnings** — a series that mixes model / runtime / backend / KV /
  reasoning / reasoning effort / context / vision / GPU split / batch shows a
  warning; cross-series differences are labelled "observational, not a
  controlled comparison".
- **Exclusion rules** — requests with generated < 20 tokens are excluded by
  default (UI adjustable 0/10/20/50/100); incomplete requests (no stop) and
  requests without a used context are also excluded, with per-class counts in
  the HTML/Markdown exports.
- **Exports** — self-contained HTML (no external dependencies), PNG, SVG, raw
  CSV (one row per request), aggregate CSV (one row per bucket × metric),
  Markdown. Existing files are never overwritten — a `-1`, `-2`… suffix is
  added instead.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

**Windows build**

```powershell
.\scripts\build-windows.ps1
.\scripts\build-installer.ps1
```

> Manual local packaging SOP (WSL→Windows interop: Python/PyInstaller/Inno Setup
> paths, exact commands, and known pitfalls) — see [docs/BUILD-WINDOWS.md](docs/BUILD-WINDOWS.md).

**Linux build** (use the distribution system Python with Tk installed)

```bash
sudo apt-get install python3-tk python3-venv xvfb
python3 -m venv .venv-build
PATH="$PWD/.venv-build/bin:$PATH" bash scripts/build-linux.sh
```

## License

MIT — see [LICENSE](LICENSE).
