Set-Location c:\repos\ai-portfolio
& "c:\repos\ai-portfolio\.venv\Scripts\python.exe" "c:\repos\ai-portfolio\scripts\add_regression_hints.py" *>&1 | Tee-Object -FilePath "c:\repos\ai-portfolio\hint_run_output.txt"
Write-Host "Exit code: $LASTEXITCODE"
