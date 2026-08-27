$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

python -m pip install -e ".[dev]"
python -m pytest
python -m PyInstaller --noconfirm --clean --windowed `
  --name LlamaLauncher `
  --icon "src/llama_launcher/assets/llama-launcher-icon.ico" `
  --add-data "src/llama_launcher/assets;assets" `
  --paths "src" `
  "scripts/launcher_entry.py"

Copy-Item "README.md" "dist/LlamaLauncher/README.md" -Force
Compress-Archive -Path "dist/LlamaLauncher/*" -DestinationPath "dist/LlamaLauncher-Portable-x64.zip" -Force
Write-Host "Portable build: dist/LlamaLauncher-Portable-x64.zip"
