param(
    [int]$Seconds = 30,
    [int]$IntervalMs = 1000
)

$ErrorActionPreference = 'Continue'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outDir = "E:\llama-cpp\diagnostics\slow-path-$stamp"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Get-CimInstance Win32_Process |
    Where-Object { $_.Name -in @('llama-server.exe', 'LlamaLauncher.exe') } |
    Select-Object ProcessId, ParentProcessId, Name, CreationDate, ExecutablePath, CommandLine |
    Format-List | Out-File "$outDir\processes.txt" -Encoding utf8

Get-NetTCPConnection -LocalPort 8080 -ErrorAction SilentlyContinue |
    Select-Object State, LocalAddress, LocalPort, OwningProcess |
    Format-List | Out-File "$outDir\port-8080.txt" -Encoding utf8

& curl.exe -sS http://127.0.0.1:8080/slots 2>&1 |
    Out-File "$outDir\slots-start.json" -Encoding utf8

$logDir = Join-Path $env:LOCALAPPDATA 'LlamaLauncher\logs'
$latestLog = Get-ChildItem $logDir -Filter 'llama-server-*.log' -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($latestLog) {
    Copy-Item $latestLog.FullName "$outDir\$($latestLog.Name)" -Force
}

'captured_at,gpu_index,name,pstate,gpu_util_pct,memory_util_pct,memory_used_mib,memory_total_mib,temp_c,power_w,graphics_clock_mhz,memory_clock_mhz,pcie_gen,pcie_width' |
    Set-Content "$outDir\gpu.csv" -Encoding ascii

$samples = [Math]::Max(1, [Math]::Ceiling(($Seconds * 1000) / $IntervalMs))
for ($i = 0; $i -lt $samples; $i++) {
    $capturedAt = (Get-Date).ToString('o')
    $rows = & nvidia-smi.exe --query-gpu=index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>&1
    foreach ($row in $rows) {
        "$capturedAt,$row" | Add-Content "$outDir\gpu.csv" -Encoding ascii
    }
    Start-Sleep -Milliseconds $IntervalMs
}

& curl.exe -sS http://127.0.0.1:8080/slots 2>&1 |
    Out-File "$outDir\slots-end.json" -Encoding utf8

if ($latestLog) {
    Copy-Item $latestLog.FullName "$outDir\$($latestLog.BaseName)-end.log" -Force
}

Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty CounterSamples |
    Where-Object CookedValue -gt 0 |
    Select-Object InstanceName, Path, @{Name='MiB';Expression={[Math]::Round($_.CookedValue / 1MB, 2)}} |
    Export-Csv "$outDir\gpu-process-memory.csv" -NoTypeInformation -Encoding utf8

Write-Output $outDir
