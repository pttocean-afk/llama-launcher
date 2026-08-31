# Windows 打包 SOP（本機 WSL → Windows interop）

> 最後驗證：2026-08-31（含 thinking-intensity 功能版）。
> 用途：每次改版後在**本機**重新壓 Windows 安裝檔，照這份文件依序執行即可，不要重新摸索工具鏈。
> CI 替代方案：push 到 `main` 或 tag `v*` 會觸發 `.github/workflows/windows-build.yml` 自動建置（見文末注意事項）。

## 0. 前置工具鏈（皆已裝好，勿重裝）

| 工具 | WSL 路徑 | 版本 |
|---|---|---|
| Windows Python（interop 啟動器） | `/mnt/c/Windows/py.exe` | 3.14.2 |
|   └ 實際執行檔 | `C:\Users\pttoc\AppData\Local\Python\pythoncore-3.14-64\python.exe` | — |
| PyInstaller（裝在上述 Python 內） | `py.exe -m PyInstaller` | 6.18.0 |
| Inno Setup 6 | `/mnt/c/Users/pttoc/AppData/Local/Programs/Inno Setup 6/ISCC.exe` | 6.7.3 |
| PowerShell（interop） | `/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe` | — |

⚠️ **Windows 使用者設定檔資料夾是 `C:\Users\pttoc`**（WSL 的家目錄才是 `/home/pttocean`）。
找 Inno Setup、Python 位置時，請先找 `pttoc` 底下，不要找 `pttocean`。

測試命令（用於確認環境正常）：

```bash
/mnt/c/Windows/py.exe --version                          # Python 3.14.2
/mnt/c/Windows/py.exe -m PyInstaller --version          # 6.18.0
ls "/mnt/c/Users/pttoc/AppData/Local/Programs/Inno Setup 6/ISCC.exe"
```

## 1. 打包步驟（依序照抄）

以 repo 根目錄為 `$REPO`（WSL：`/home/pttocean/projects/llama-launcher`）。

### 1.1 複製工作樹到 Windows 側 build 目錄

每次都全新複製，**不要**做差異同步（避免 `__pycache__` / `build` / `dist` 殘留）：

```bash
cd "$REPO"
rm -rf /mnt/e/llama-launcher-build
mkdir -p /mnt/e/llama-launcher-build
rsync -a \
  --exclude '.git' --exclude '.venv*' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude 'build' --exclude 'dist' \
  --exclude '*.pyc' \
  ./ /mnt/e/llama-launcher-build/
cd /mnt/e/llama-launcher-build
```

### 1.2 （僅首次或 pytest 消失時）Windows Python 裝 dev 依賴

```bash
/mnt/c/Windows/py.exe -m pip install -e ".[dev]"
```

### 1.3 （可選）Windows 側測試

```bash
/mnt/c/Windows/py.exe -m pytest -q
```

預期：約 118 passed／4 failed，**4 個失敗都是環境性、非程式問題**（清單見 §2），可以直接忽略。

### 1.4 PyInstaller 打包主程式

注意：`--add-data` 的資料夾分隔符在 Windows 上是 `;`（不是 `:`）。

```bash
cd /mnt/e/llama-launcher-build
/mnt/c/Windows/py.exe -m PyInstaller \
  --noconfirm --clean --windowed \
  --name LlamaLauncher \
  --icon "src/llama_launcher/assets/llama-launcher-icon.ico" \
  --add-data "src/llama_launcher/assets;assets" \
  --paths "src" \
  scripts/launcher_entry.py
```

產出：`dist/LlamaLauncher/`（含 `LlamaLauncher.exe` 與 `_internal/`）。
成功標記：log 尾端出現 `Build complete! ... E:\llama-launcher-build\dist`。

### 1.5 複製 README 進 Portable 目錄

```bash
cd /mnt/e/llama-launcher-build
cp README.md dist/LlamaLauncher/README.md
```

### 1.6 壓 Portable ZIP（用 PowerShell interop）

路徑要用 Windows 格式（`E:\...`），不是 `/mnt/e/...`：

```bash
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
  "Compress-Archive -Path 'E:\llama-launcher-build\dist\LlamaLauncher\*' \
   -DestinationPath 'E:\llama-launcher-build\dist\LlamaLauncher-Portable-x64.zip' -Force"
```

### 1.7 編譯 Setup EXE（Inno Setup）

```bash
cd /mnt/e/llama-launcher-build/installer
"/mnt/c/Users/pttoc/AppData/Local/Programs/Inno Setup 6/ISCC.exe" LlamaLauncher.iss
```

成功標記：`Successful compile` + `dist/LlamaLauncher-Setup-x64.exe`。
（`LlamaLauncher.iss` 內用相對路徑 `..\dist\LlamaLauncher\*`，因此必須在 `installer/` 目錄下執行。）

### 1.8 產物複製回 repo 的 `dist/` 並驗證

```bash
cd "$REPO"
cp -f /mnt/e/llama-launcher-build/dist/LlamaLauncher-Setup-x64.exe     dist/
cp -f /mnt/e/llama-launcher-build/dist/LlamaLauncher-Portable-x64.zip  dist/
ls -la dist/
file dist/LlamaLauncher-Setup-x64.exe          # 期望: PE32 executable (GUI) ...
unzip -l dist/LlamaLauncher-Portable-x64.zip   # 期望含 LlamaLauncher.exe + _internal
```

> `dist/` 已列入 `.gitignore`，產物不入 git；要發布才上傳 GitHub Releases。

## 2. Windows 側 pytest 的 4 個「正常」失敗（勿誤判為回歸）

| 測試 | 原因 |
|---|---|
| `test_app_security.py::test_control_server_bind_failure_is_observable` | tkinter 需要桌面 session（interop session 無桌面） |
| `test_app_security.py::test_end_to_end_control_api_over_http` | 同上（urllib/Tk 環境） |
| `test_host.py::test_autostart_toggle` | 依賴 Windows registry/autostart 狀態 |
| `test_reasoning_effort.py::test_settings_dialog_*`（偶發 1 個） | tkinter/Tcl 初始化（`tcl_findLibrary`），有桌面 session 就會過 |

**程式邏輯測試的權威來源是 Linux 側**：`xvfb-run -a .venv-build/bin/python -m pytest -q`（目前 122 passed）。

## 3. 踩過的坑（勿重蹈）

1. **Inno Setup 安裝**
   - 從 WSL interop 靜默安裝會**假裝成功**（exit 0）但什麼都沒裝（`//VERYSILENT` 參數被吃），不要再試。
   - 下載要直連 GitHub release：`https://github.com/jrsoftware/issrc/releases/download/is-6_7_3/innosetup-6.7.3.exe`；
     `https://jrsoftware.org/download.php/is.exe` 只會回 HTML 跳轉頁（curl 抓下來不是 exe）。
   - 安裝後路徑在 `C:\Users\pttoc\AppData\Local\Programs\Inno Setup 6\`（`Compil32.exe`、`ISCC.exe`）。
2. **`cmd.exe` 不在 WSL PATH**：不要寫 `cmd.exe /c ...`；直接用 interop 執行 `ISCC.exe` 絕對路徑即可。
3. **使用者資料夾是 `pttoc`**：所有 Windows 側路徑先找 `C:\Users\pttoc`。
4. **PyInstaller `--add-data` 分隔符**：Windows 用 `;`（`src\...\assets;assets`）；寫成 `:` 會在 Windows 產生物件檔路徑錯誤。
5. **PowerShell interop 的路徑**要用 `E:\...` 形式；`Compress-Archive` 的 `-Force` 記得加，避免舊檔殘留報錯。
6. **Linux 打包別用 Hermes Python**：要用 distro system Python（`python3 -m venv` + `python3-tk`），否則缺 `libtcl9.0.so`。
7. **CI installer 步驟**：`.github/workflows/windows-build.yml` 的 `build-installer.ps1` 需要 Inno Setup 6，但 workflow 目前沒有安裝步驟；若 CI 上 installer 步驟失敗，先在此 workflow 加 `winget install JRSoftware.InnoSetup`（或等價步驟），本機打包不受影響。

## 4. 相關檔案

- `scripts/build-windows.ps1` — CI/Windows 本機用的 PyInstaller + ZIP 腳本（本 SOP 與之等價）
- `scripts/build-installer.ps1` — Inno Setup 編譯腳本
- `installer/LlamaLauncher.iss` — Inno 安裝腳本（AppId/名稱/Reg 設定）
- `.github/workflows/windows-build.yml`、`linux-build.yml` — push main / tag `v*` 自動建置
- Linux 手動打包：`PATH="$PWD/.venv-build/bin:$PATH" bash scripts/build-linux.sh` → `dist/LlamaLauncher-Linux-x86_64.tar.gz`