# Rolling queue: after the overnight batch, run every jobs\queue\*.txt in name order, again and again,
# picking up files added later (batch.py skips finished runs). Create jobs\queue\STOP to end it.
Set-Location (Split-Path $PSScriptRoot)
$py = ".venv\Scripts\python.exe"
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'overnight.ps1' }) { Start-Sleep 30 }
while (-not (Test-Path jobs\queue\STOP)) {
    $ran = $false
    foreach ($f in Get-ChildItem jobs\queue\*.txt | Sort-Object Name) {
        $out = & $py -u batch.py $f.FullName --parallel 3
        $out | Write-Output
        if ($out -notmatch '^0 of ') { $ran = $true }
        & $py report.py | Out-Null
    }
    if (-not $ran) { Start-Sleep 60 }
}
