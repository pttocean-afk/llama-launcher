#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -e '.[dev]'
# tkinter 測試需要顯示伺服器；本機 CI（GitHub Actions Ubuntu）沒有 DISPLAY，
# 統一用 xvfb-run 包住（樓上有裝 xvfb 時才可跑；沒有 xvfb 時退回直接跑）。
if command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a python -m pytest
else
    python -m pytest
fi
python -m PyInstaller --noconfirm --clean --windowed \
  --name LlamaLauncher \
  --icon src/llama_launcher/assets/llama-launcher-icon.png \
  --add-data 'src/llama_launcher/assets:assets' \
  --paths src \
  scripts/launcher_entry.py

cp README.md dist/LlamaLauncher/README.md
mkdir -p dist/LlamaLauncher/share/applications
cp packaging/linux/llama-launcher.desktop dist/LlamaLauncher/share/applications/
tar -C dist -czf dist/LlamaLauncher-Linux-x86_64.tar.gz LlamaLauncher
printf 'Linux bundle: %s\n' "dist/LlamaLauncher-Linux-x86_64.tar.gz"
