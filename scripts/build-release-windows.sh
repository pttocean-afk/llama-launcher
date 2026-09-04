#!/usr/bin/env bash
# Canonical local release build: WSL -> Windows Python/PyInstaller/Inno Setup.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIRROR="${LLAMA_LAUNCHER_BUILD_DIR:-/mnt/e/llama-launcher-build}"
WIN_PY="${LLAMA_LAUNCHER_PYTHON:-/mnt/c/Users/pttoc/AppData/Local/Programs/Python/Python311/python.exe}"
POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
ISCC="/mnt/c/Users/pttoc/AppData/Local/Programs/Inno Setup 6/ISCC.exe"
SKIP_TESTS=0
[[ "${1:-}" == "--skip-tests" ]] && SKIP_TESTS=1
MIRROR_WIN="$(wslpath -w "$MIRROR")"

for tool in "$WIN_PY" "$POWERSHELL" "$ISCC"; do
  [[ -f "$tool" ]] || { echo "Missing required tool: $tool" >&2; exit 1; }
done

PROJECT_VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$REPO/pyproject.toml" | head -1)"
INSTALLER_VERSION="$(sed -n 's/^#define MyAppVersion "\([^"]*\)"/\1/p' "$REPO/installer/LlamaLauncher.iss")"
[[ -n "$PROJECT_VERSION" && "$PROJECT_VERSION" == "$INSTALLER_VERSION" ]] || {
  echo "Version mismatch: pyproject=$PROJECT_VERSION installer=$INSTALLER_VERSION" >&2
  exit 1
}

# The mirror is disposable. Recreate it to prevent duplicate package trees,
# stale __pycache__, PyInstaller specs, and old build/dist files.
rm -rf "$MIRROR"
mkdir -p "$MIRROR"
rsync -a \
  --exclude '.git' --exclude '.venv*' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude 'build' --exclude 'dist' \
  --exclude '*.pyc' --exclude '*.spec' --exclude '*.egg-info' \
  "$REPO/" "$MIRROR/"

cd "$MIRROR"
"$WIN_PY" -m pip install -e '.[dev]'
if (( ! SKIP_TESTS )); then
  "$WIN_PY" -m pytest -q
fi
"$WIN_PY" -m PyInstaller \
  --noconfirm --clean --windowed \
  --name LlamaLauncher \
  --icon 'src/llama_launcher/assets/llama-launcher-icon.ico' \
  --add-data 'src/llama_launcher/assets;assets' \
  --paths src \
  scripts/launcher_entry.py

cp README.md dist/LlamaLauncher/README.md
"$POWERSHELL" -NoProfile -Command \
  "Compress-Archive -Path '$MIRROR_WIN\\dist\\LlamaLauncher\\*' -DestinationPath '$MIRROR_WIN\\dist\\LlamaLauncher-Portable-x64.zip' -Force"
(
  cd "$MIRROR/installer"
  "$ISCC" LlamaLauncher.iss
)

mkdir -p "$REPO/dist"
rm -f "$REPO/dist/LlamaLauncher-Setup-x64.exe" \
      "$REPO/dist/LlamaLauncher-Portable-x64.zip"
cp "$MIRROR/dist/LlamaLauncher-Setup-x64.exe" "$REPO/dist/"
cp "$MIRROR/dist/LlamaLauncher-Portable-x64.zip" "$REPO/dist/"

# Keep only final distributables in the disposable mirror.
rm -rf "$MIRROR/build" "$MIRROR/dist/LlamaLauncher" "$MIRROR/LlamaLauncher.spec"
find "$MIRROR" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$MIRROR" -type d -name '*.egg-info' -prune -exec rm -rf {} +

printf '\nBuild complete (version %s):\n' "$PROJECT_VERSION"
sha256sum "$REPO/dist/LlamaLauncher-Setup-x64.exe" \
          "$REPO/dist/LlamaLauncher-Portable-x64.zip"
printf '\nArtifacts:\n  %s\n  %s\n' \
  "$REPO/dist/LlamaLauncher-Setup-x64.exe" \
  "$REPO/dist/LlamaLauncher-Portable-x64.zip"
