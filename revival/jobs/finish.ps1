# After the running batches: redo missing modern runs, then MCTS on the quiet machine, then the report.
Set-Location (Split-Path $PSScriptRoot)
$py = ".venv\Scripts\python.exe"
function Wait-Batch($pattern) {
    while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $pattern }) { Start-Sleep 30 }
}
Wait-Batch 'batch.py runs.all-modern'
& $py -u batch.py runs\all-modern.txt --parallel 3
Wait-Batch 'batch.py runs.all-2018'
& $py -u batch.py runs\all-2018.txt --parallel 5   # anything that failed in the 2018 phase
& $py -u run_mcts.py
& $py report.py
