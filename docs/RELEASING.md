# Windows 打包與發布 SOP

本專案目前僅正式維護 **Windows 10/11 x64**。Linux CI、安裝說明與發佈產物暫停；每次 release 只發布 Setup EXE 與 Portable ZIP。

## 流程總覽

```text
完成功能 → 本機測試 → 讓使用者以 source 實測 → 確認 OK →
更新 README/截圖 → Windows 打包 → smoke test → GitHub Release → 驗證下載
```

> 不要為了每次 UI 微調反覆打包。使用者確認 source 版本後才進入打包階段。

## 1. 前置檢查

```bash
cd /home/pttocean/projects/llama-launcher
PATH="$PWD/.venv-build-sys/bin:$PATH" xvfb-run -a python -m pytest -q
.venv-build-sys/bin/python -m compileall -q src tests
git diff --check
git status --short
```

Windows GitHub Actions（`.github/workflows/windows-build.yml`）也必須成功。Workflow 監聽 `master` 與 `v*` tag。

## 2. Windows 本機打包

完整 WSL→Windows interop 與 Inno Setup 細節見 [BUILD-WINDOWS.md](BUILD-WINDOWS.md)。核心流程：

```bash
cd /home/pttocean/projects/llama-launcher
rsync -a --delete \
  --exclude '.git' --exclude '.venv*' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude 'build' --exclude 'dist' --exclude '*.pyc' \
  ./ /mnt/e/llama-launcher-build/

cd /mnt/e/llama-launcher-build
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile \
  -ExecutionPolicy Bypass -File 'E:\llama-launcher-build\scripts\build-windows.ps1'
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile \
  -ExecutionPolicy Bypass -File 'E:\llama-launcher-build\scripts\build-installer.ps1'
```

產物：

- `dist/LlamaLauncher-Portable-x64.zip`
- `dist/LlamaLauncher-Setup-x64.exe`

若 CI 的獨立 Test step 已經完成，build script 可使用 `-SkipTests`；正式手動 release 預設不要跳過。

## 3. 打包後 smoke test

至少確認：

1. Portable EXE 能啟動，沒有缺少 Tcl/Tk 或 asset。
2. 能讀取 `%LOCALAPPDATA%\LlamaLauncher` 的既有 profiles/logs。
3. 主畫面模型子資料夾掃描正常。
4. 📊 效能分析可開啟能力總覽、速度曲線與取捨圖。
5. `127.0.0.1:8765` Control API 啟動正常，關閉程式後 port 已釋放。
6. Setup EXE 可安裝、啟動及解除安裝。

## 4. GitHub Release

1. Commit 並 push `master`，等待 Windows CI 綠燈。
2. 建立新 tag（不要覆蓋既有 release tag）：

```bash
VERSION=vX.Y.Z
git tag "$VERSION"
git push origin "$VERSION"
```

3. Tag workflow 可自動上傳 Windows 產物；若需手動補傳，使用 GitHub Releases 網頁或已儲存的 git credential 呼叫 GitHub API。
4. Release 說明只列 Windows installer 與 portable，不再宣稱 Linux 支援。

## 5. 發布後驗證

```bash
for u in \
  https://github.com/pttocean-afk/llama-launcher/releases/download/vX.Y.Z/LlamaLauncher-Setup-x64.exe \
  https://github.com/pttocean-afk/llama-launcher/releases/download/vX.Y.Z/LlamaLauncher-Portable-x64.zip ; do
  curl -sL -o /dev/null -w "%{http_code} $u\n" "$u"
done
```

兩個網址都應回 HTTP 200。

## 常見問題

| 問題 | 處理方式 |
|---|---|
| 同一 Windows job 重複跑 Tk 測試後 Tcl/Tk 缺檔 | Workflow 保留獨立 Test；打包 step 使用 `build-windows.ps1 -SkipTests` |
| Windows PyInstaller `--add-data` 寫成 `:` | Windows 分隔符必須是 `;`（`assets;assets`） |
| 重複建立同一 tag | 改新版本 tag，或只補傳既有 release 資產 |
| `dist/` 沒出現在 git | 正常；產物被 gitignore，只上傳 GitHub Releases |
| README 與實際平台不一致 | 中文、英文與 screenshots 必須同步更新；目前正式平台只有 Windows |
