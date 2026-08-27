$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Candidates = @(
  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
  "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)
$Iscc = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) {
  throw "Inno Setup 6 was not found. Install JRSoftware.InnoSetup with winget."
}
& $Iscc "$Repo\installer\LlamaLauncher.iss"
if ($LASTEXITCODE -ne 0) {
  throw "Inno Setup compilation failed with exit code $LASTEXITCODE"
}
Write-Host "Installer: $Repo\dist\LlamaLauncher-Setup-x64.exe"
