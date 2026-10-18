# End of the night: once every queued run has a result (or nothing has trained for 10 min), queue the best-of
# round (bestof.py); once that is done, rebuild the report, stop the rolling queue and release the GPU lease.
Set-Location (Split-Path $PSScriptRoot)
$py = ".venv\Scripts\python.exe"
function Training { @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train_(modern|ppo|2018)\.py' }).Count }
$idle = 0
while (-not (Test-Path jobs\queue\20-bestof.txt)) {
    Start-Sleep 300
    $idle = if ((Training) -eq 0) { $idle + 1 } else { 0 }
    if ($idle -ge 2) { & $py bestof.py --force } else { & $py bestof.py }
}
"bestof queued $(Get-Date -f HH:mm)"
Get-Content jobs\queue\20-bestof.txt
$names = Get-Content jobs\queue\20-bestof.txt | Where-Object { $_ -match '--name (\S+)' } | ForEach-Object { $Matches[1] }
$idle = 0
while ($idle -lt 2 -and ($names | Where-Object { -not (Test-Path "results\$_.json") })) {
    Start-Sleep 300
    $idle = if ((Training) -eq 0) { $idle + 1 } else { 0 }
}
& $py report.py | Out-Null
New-Item jobs\queue\STOP -ItemType File -Force | Out-Null
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'Start-Sleep 43200' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
"finished $(Get-Date -f HH:mm)"
