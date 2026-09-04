$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$VersionLine = Select-String -Path "$Repo\pyproject.toml" -Pattern '^version = "([^"]+)"$'
if (-not $VersionLine) { throw "Cannot read version from pyproject.toml" }
$Version = $VersionLine.Matches[0].Groups[1].Value
$InstallerPath = "$Repo\dist\LlamaLauncher-Setup-$Version-x64.exe"
if (Test-Path $InstallerPath) {
  throw "Refusing to overwrite existing release artifact: $InstallerPath. Bump the version first."
}
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
Write-Host "Installer: $InstallerPath"
