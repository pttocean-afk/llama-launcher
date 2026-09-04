param(
  [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

if (-not $SkipInstall) {
  py -m pip install -e .
  if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

# Close an installed/portable LlamaLauncher first, otherwise its localhost
# Control API (port 8765) will correctly prevent a second instance binding.
py scripts\launcher_entry.py
if ($LASTEXITCODE -ne 0) { throw "Llama Launcher source run failed" }
