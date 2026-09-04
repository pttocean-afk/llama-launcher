# 開發守則

本文件列出維護 Llama Launcher 時不可省略的專案規則。

## 發行產物命名與保存

1. **所有正式產物檔名必須包含完整版本號**：
   - `LlamaLauncher-Setup-X.Y.Z-x64.exe`
   - `LlamaLauncher-Portable-X.Y.Z-x64.zip`
2. `pyproject.toml` 的 `version` 與 `installer/LlamaLauncher.iss` 的
   `MyAppVersion` 必須一致。
3. **不得覆蓋或刪除既有版本的正式產物**。要重打正式包時，先增加版本號；
   canonical build script 偵測到同版號檔案時必須直接失敗。
4. 打包前先確認上一版產物仍存在；需要整理時移至
   `dist/previous-X.Y.Z/`，檔名仍保留版本號。
5. 發布到 GitHub Release 的 asset 名稱也必須使用上述含版本號格式，
   release tag 使用 `vX.Y.Z`，不得覆蓋既有 tag。

## 發行前檢查

- 執行完整 pytest（Linux/Xvfb 與 Windows Python）。
- 執行 `python -m compileall -q src tests` 與 `git diff --check`。
- 對變更過的 GUI 視窗做啟動 smoke test。
- 確認安裝檔與便攜版的 SHA-256，並確認上一版回滾檔仍完整。
- 唯一正式 WSL → Windows 打包入口是：
  `bash scripts/build-release-windows.sh`。

更完整的流程見 `docs/RELEASING.md` 與 `docs/BUILD-WINDOWS.md`。
