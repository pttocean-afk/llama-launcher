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

先確認 `pyproject.toml` 與 `installer/LlamaLauncher.iss` 的版本一致，然後只執行 canonical script：

```bash
cd /home/pttocean/projects/llama-launcher
bash scripts/build-release-windows.sh
```

若相同 commit 已完整測試，可使用：

```bash
bash scripts/build-release-windows.sh --skip-tests
```

腳本會從乾淨的 Windows mirror 建置、清理中間檔，並把最終產物複製回 repo：

- `dist/LlamaLauncher-Portable-X.Y.Z-x64.zip`
- `dist/LlamaLauncher-Setup-X.Y.Z-x64.exe`

**發行檔名必須含完整版號，且不得覆蓋既有版本。** 若同版號產物已存在，
打包腳本必須停止；先增加版本號再重打。上一版需保留供回滾。完整守則見
根目錄 [DEVELOPMENT.md](../DEVELOPMENT.md)。

不要手動 `rsync src/`、逐檔複製 package 或保留第二份 root `llama_launcher/`。完整工具路徑與除錯規則見 [BUILD-WINDOWS.md](BUILD-WINDOWS.md)。

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
  https://github.com/pttocean-afk/llama-launcher/releases/download/vX.Y.Z/LlamaLauncher-Setup-X.Y.Z-x64.exe \
  https://github.com/pttocean-afk/llama-launcher/releases/download/vX.Y.Z/LlamaLauncher-Portable-X.Y.Z-x64.zip ; do
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
