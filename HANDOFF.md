# HANDOFF — Llama Launcher productization（交接給 DSH 接手）

> 產生：2026-08-27，Lucy。此文件是「現況 + 待辦」的唯一入口；先讀完再動工。
> 主人在另一台環境用 DSH 接手，所有執行都必須以真實 build/test/HTTP 驗證，不允許只改不驗。

## 一、Repo 位置與基線

- 新 repo：`/home/pttocean/projects/llama-launcher`（WSL），已 git init，本地已 commit 2 筆，工作樹乾淨。
- 舊日常 launcher（尚未切換，屬「可用基準」，不可刪）：`/mnt/e/llama-cpp/launcher-app/llama-launcher.pyw`。
- 產品硬性要求：**必須同時支援 Windows 與 Linux 本機**（llama.cpp 不一定裝在 Windows）。共用核心不得含平台專屬假設。
- 使用者資料與程式分離：Windows `%LOCALAPPDATA%\LlamaLauncher`；Linux `$XDG_DATA_HOME/LlamaLauncher`；可用 `LLAMA_LAUNCHER_DATA_DIR` 覆寫（測試用）。

## 二、已完成的架構（不要重做）

- 模組：`config.py`（atomic JSON）、`security.py`（token 產生/恆時比對）、`tailscale.py`（CLI 偵測/Serve/授權 URL 解析）、`remote_setup.py`、`diagnostics.py`、`migration.py`、`paths.py`（LOCALAPPDATA/XDG/portable marker）、`host.py`（跨平台：executable 名稱、psutil 掃描/接管/停止、external open、nvidia-smi 位置）。
- `app.py`（~2575 行）＝從舊 monolith 移植的 GUI + ServerManager + ControlServer，已接上共用模組。
- 首次啟動精靈（選 llama.cpp 目錄）、Remote Access 按鈕（一鍵 Tailscale Serve + 授權網址 + 顯示 URL/Token）、Diagnostics 按鈕。
- 安全設計：Control API 只綁 `127.0.0.1:8765`，Tailscale Serve 代理該 localhost（**不要**綁 100.x IP，實測 Serve→100.x 會 timeout）；Bearer token；`/api/*` 全保護；`/favicon.ico` 204；只允許既有 profile 啟動；`--port 8080` 唯一明確接管。
- VRAM preflight（Vulkan ≥224K）保留為 **Windows-only host policy**；Linux 必須 skip（不可執行 wsl.exe/taskkill/Windows allowlist）。
- Linux tray 預設 close-to-exit（GNOME/KDE 無 tray host 時不會變幽靈程式）。

## 三、已驗證的事實（勿重複踩坑）

| 項目 | 結果 |
|---|---|
| Linux 測試（WSL Python 3.11 + system 3.12） | 10 passed |
| Windows 測試（Python 3.11） | 10 passed |
| Windows Portable EXE + Setup EXE | 已建，真實驗證：啟動、HTTPS 200、接管既有 PID 22092 不重啟 |
| Linux artifact（system Python + python3-tk 重建） | 已建，Xvfb 啟動成功、dashboard HTTP 200 |
| **Linux 打包鐵則** | 必須用 distro system Python（`python3 -m venv` + `python3-tk`）；用 Hermes bundled Python 打包會缺 `libtcl9.0.so` 直接失敗 |
| 產物位置 | `dist/LlamaLauncher-Setup-x64.exe`、`dist/LlamaLauncher-Portable-x64.zip`、`dist/LlamaLauncher-Linux-x86_64.tar.gz`（已被 .gitignore 排除，不入 git） |

## 四、接手後第一優先（已完成 2026-08-27 by DSH，commit 見下）

來源：`/home/pttocean/.hermes/profiles/lucy/cache/delegation/subagent-summary-0-20260827_085305_350664.txt`

1. ~~**遠端 dashboard XSS**~~ ✅ 已改 DOM 建構（`renderProfiles`：`createElement`/`textContent`/`addEventListener`，model 只走 `POST /api/start` JSON body）。
2. ~~**start/stop 無 operation lock**~~ ✅ `LauncherApp.server_lock`（`threading.Lock`）：`remote_start`/`remote_stop`/`on_launch`/`on_stop`、degraded 自動停止、quit 停止全部共用。
3. ~~**ControlServer bind 失敗靜默**~~ ✅ 寫 `llama_launcher` logger warning + `/api/status` 回 `control.ok`/`control.error` + 啟動時彈出警告 dialog。

驗證：15 passed（Xvfb 下；headless 14，bind 測試需 display）、compileall OK、Linux artifact 重建並真機 Xvfb 驗證（dashboard 200、`/api/status` token 401/200、新 HTML 生效）。
**注意：Windows artifact 尚未用本次改動重建**（本環境無法產出/執行 .exe）——下次在 Windows 上跑 `scripts/build-windows.ps1` 重建並驗證。

## 五、v1.0 前剩餘工作（依序）

1. ✅ 3 個安全修復 + 測試（已完成；Linux artifact 已重驗，**Windows artifact 待重建**）。
2. 真實 Linux GPU 主機驗證 `llama-server` start/stop/adopt（目前只有 Windows 驗過）。
3. 真實 GNOME/KDE session 驗 tray（Xvfb 無 tray manager，不算數）。
4. 模組細拆（audit 建議，不急於 v1.0）：`profiles.py`、`inventory.py`、`command.py`（argv builder）、`preflight.py`、`remote.py`（HTML 移出 py 檔）、`single_instance.py`；用 characterization tests 鎖定 CUDA/Vulkan argv。
5. 設定遷移 UI：舊 `E:\llama-cpp\launcher-app` 的 models.json/settings/token → 新資料目錄（migration.py 已有核心邏輯）。
6. profile export/import（不帶本機絕對路徑）。
7. 依主人決定：GitHub repo 公開/private + License（icon 來源授權也要確認）；主人驗收後才切換日常使用。

## 六、硬性規則

- 不得把 `control-token.txt`、logs、GGUF、本機絕對路徑、Tailscale hostname、settings/profiles 提交進 git。
- 改完任何東西：跑 `pytest` + `compileall`，Windows/Linux 各驗一次 artifact 啟動（Xvfb / 實際 EXE），再回報。
- 不主動啟動/停止主人正在跑的 llama-server（現況：Windows PID 22092 由舊 launcher 管理，別碰）。
- 需要我確認的決策（GitHub 可見性、License）→ 問主人，不要自行決定。
