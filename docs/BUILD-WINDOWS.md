# Windows 本機打包 SOP（WSL → Windows）

本專案只正式維護 **Windows 10/11 x64**。這份文件是唯一的本機 release build 流程；不要再手動同步零散檔案或建立第二份 `llama_launcher/`。

## 一行打包

在 WSL repo 根目錄執行：

```bash
cd /home/pttocean/projects/llama-launcher
bash scripts/build-release-windows.sh
```

腳本會依序完成：

1. 刪除並重建乾淨的 `E:\llama-launcher-build`。
2. 只同步 repo source（排除 `.git`、venv、cache、舊 `build/dist`、spec、egg-info）。
3. 使用 Windows Python 3.11 安裝 dev 依賴並執行 pytest。
4. 使用 PyInstaller 建置 Portable app。
5. 使用 PowerShell 建立 Portable ZIP。
6. 使用已安裝的 Inno Setup 6 建立 Setup EXE。
7. 將兩個最終產物複製回 repo 的 `dist/`，列出 SHA-256。
8. 清除 mirror 裡的 PyInstaller 中間檔、解壓 app、cache 與 egg-info，只保留兩個最終產物。

正式產物（`X.Y.Z` 為 `pyproject.toml` 版本）：

```text
dist/LlamaLauncher-Setup-X.Y.Z-x64.exe
dist/LlamaLauncher-Portable-X.Y.Z-x64.zip
```

同一份產物也會留在：

```text
E:\llama-launcher-build\dist\LlamaLauncher-Setup-X.Y.Z-x64.exe
E:\llama-launcher-build\dist\LlamaLauncher-Portable-X.Y.Z-x64.zip
```

> 發行檔名必須包含版號，而且既有產物不可覆蓋。若同版號檔案已存在，
> 腳本會停止；請先增加版本號。完整規則見根目錄 `DEVELOPMENT.md`。

若 pytest 已在同一 commit 完整通過，只想重建產物：

```bash
bash scripts/build-release-windows.sh --skip-tests
```

## 前置工具（本機已安裝）

| 工具 | 固定路徑 |
|---|---|
| Windows Python 3.11 | `/mnt/c/Users/pttoc/AppData/Local/Programs/Python/Python311/python.exe` |
| Inno Setup 6 | `/mnt/c/Users/pttoc/AppData/Local/Programs/Inno Setup 6/ISCC.exe` |
| PowerShell | `/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe` |
| Windows scratch mirror | `/mnt/e/llama-launcher-build` |

Windows 使用者是 `pttoc`；WSL 使用者才是 `pttocean`。

需要改用其他位置時，可設定：

```bash
LLAMA_LAUNCHER_BUILD_DIR=/mnt/e/other-build \
LLAMA_LAUNCHER_PYTHON=/mnt/c/path/to/python.exe \
bash scripts/build-release-windows.sh
```

## 打包前檢查

- `pyproject.toml` 的版本與 `installer/LlamaLauncher.iss` 的 `MyAppVersion` 必須一致。
- `git status --short` 應乾淨；不要把開發中未確認的 UI 直接打包。
- GitHub Actions `windows-build.yml` 應在相同 commit 全綠。
- 若 source 測試中的 Launcher 還在執行，先關閉，避免 pip 更新或 port 8765 衝突。

## 打包後 smoke test

至少確認：

1. Portable `LlamaLauncher.exe` 能啟動。
2. 能讀取 `%LOCALAPPDATA%\LlamaLauncher` 既有 profiles/logs。
3. 📊 效能分析可開啟能力總覽、速度曲線與 Context/速度取捨。
4. 取捨圖可用滾輪以游標為中心縮放，並可按住左鍵拖曳。
5. Setup EXE 可覆蓋安裝並正常啟動。
6. 關閉程式後 port 8765 已釋放。

## CI 與 GitHub Release

- push `master` 或 `v*` tag 會觸發 `.github/workflows/windows-build.yml`。
- Windows pytest 必須全綠；不接受「環境性失敗可忽略」。
- CI Test step 已跑過後，建置 step 使用 `scripts/build-windows.ps1 -SkipTests`，避免同一 runner 重複初始化 Tk/Tcl。
- 建立 release 時使用新 tag，不覆蓋舊 tag；只發布 Windows Setup EXE 與 Portable ZIP。

## 保留與清理規則

應保留：

- repo source、`.venv-build-sys`（WSL 測試環境）。
- repo `dist/` 與 mirror `dist/` 內當前版本的 Setup EXE、Portable ZIP。

應刪除／由腳本自動重建：

- `build/`、`LlamaLauncher.spec`、`__pycache__/`、`.pytest_cache/`、`*.egg-info/`。
- `E:\llama-launcher-build\llama_launcher`（錯誤的 root package 副本；正式路徑只有 `src\llama_launcher`）。
- UI 測試截圖、`*_temp.py`、`*_temp.ps1`。
- 舊 Linux venv、Linux bundle 與 Linux packaging 檔案。

不要再使用 `rsync src/ ...`：尾端斜線會把 `src/llama_launcher` 錯放成 mirror root 的 `llama_launcher/`。只允許由 canonical script 同步完整 repo 根目錄。
