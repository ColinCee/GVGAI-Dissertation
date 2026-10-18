# Runs every training batch in order, 2018 replay first. Restartable: finished runs are skipped,
# 2018 runs resume from checkpoints. Run under the GVGAI GPU lease.
Set-Location (Split-Path $PSScriptRoot)
$py = ".venv\Scripts\python.exe"
Get-Content jobs\phase2.txt, jobs\ablation2018.txt | Set-Content runs\all-2018.txt
& $py -u batch.py runs\all-2018.txt --parallel 5  # 8 ran out of RAM (each 2018 run grows to ~5-9 GB as its replay fills)
Get-Content jobs\ladder.txt, jobs\phase3.txt | Set-Content runs\all-modern.txt
& $py -u batch.py runs\all-modern.txt --parallel 3
& $py report.py
