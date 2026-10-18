# Runs batch 5 once the overnight queue has finished, then refreshes the report.
Set-Location (Split-Path $PSScriptRoot)
$py = ".venv\Scripts\python.exe"
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'overnight.ps1' }) { Start-Sleep 30 }
& $py -u batch.py jobs\batch5.txt --parallel 3
& $py report.py
