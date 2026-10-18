# Overnight: MCTS on the quiet machine first (4 games at once, one core each), then the experiments,
# then the report. Waits for any light batch still running.
Set-Location (Split-Path $PSScriptRoot)
$py = ".venv\Scripts\python.exe"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'batch.py' }) { Start-Sleep 30 }
& $py -u run_mcts.py
& $py -u batch.py jobs\overnight.txt --parallel 3
& $py report.py
