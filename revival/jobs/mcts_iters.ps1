# MCTS at fixed search budgets (CPU only, idle priority): runs next to GPU training once the 40 ms
# MCTS (which needs a quiet machine) is done.
Set-Location (Split-Path $PSScriptRoot)
$py = ".venv\Scripts\python.exe"
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_mcts.py' -and $_.CommandLine -notmatch '--iters' }) { Start-Sleep 30 }
while (-not (Test-Path results\mcts-waferthinmints2018.json)) { Start-Sleep 30 }
foreach ($it in 50, 200, 800) {
    $p = Start-Process $py -ArgumentList "-u", "run_mcts.py", "--iters", $it -NoNewWindow -PassThru -RedirectStandardOutput "runs\mcts-i$it.log"
    $p.PriorityClass = 'Idle'
    $p.WaitForExit()
}
