#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -e '.[dev]'
python -m pytest
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
