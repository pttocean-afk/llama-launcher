param(
  [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$VersionLine = Select-String -Path "pyproject.toml" -Pattern '^version = "([^"]+)"$'
if (-not $VersionLine) { throw "Cannot read version from pyproject.toml" }
$Version = $VersionLine.Matches[0].Groups[1].Value
$PortableName = "LlamaLauncher-Portable-$Version-x64.zip"
$PortablePath = Join-Path $Repo "dist/$PortableName"
if (Test-Path $PortablePath) {
  throw "Refusing to overwrite existing release artifact: $PortablePath. Bump the version first."
}

# 注意：$ErrorActionPreference=Stop 不會因原生程式的非零結束碼失敗，

# 必須在每個原生指令後明確檢查 $LASTEXITCODE。
python -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
if (-not $SkipTests) {
  python -m pytest
  if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
}
python -m PyInstaller --noconfirm --clean --windowed `
  --name LlamaLauncher `
  --icon "src/llama_launcher/assets/llama-launcher-icon.ico" `
  --add-data "src/llama_launcher/assets;assets" `
  --paths "src" `
  "scripts/launcher_entry.py"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Copy-Item "README.md" "dist/LlamaLauncher/README.md" -Force
Compress-Archive -Path "dist/LlamaLauncher/*" -DestinationPath $PortablePath
Write-Host "Portable build: $PortablePath"
