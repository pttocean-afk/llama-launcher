$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

# 注意：$ErrorActionPreference=Stop 不會因原生程式的非零結束碼失敗，
# 必須在每個原生指令後明確檢查 $LASTEXITCODE。
python -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
python -m pytest
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
python -m PyInstaller --noconfirm --clean --windowed `
  --name LlamaLauncher `
  --icon "src/llama_launcher/assets/llama-launcher-icon.ico" `
  --add-data "src/llama_launcher/assets;assets" `
  --paths "src" `
  "scripts/launcher_entry.py"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Copy-Item "README.md" "dist/LlamaLauncher/README.md" -Force
Compress-Archive -Path "dist/LlamaLauncher/*" -DestinationPath "dist/LlamaLauncher-Portable-x64.zip" -Force
Write-Host "Portable build: dist/LlamaLauncher-Portable-x64.zip"
