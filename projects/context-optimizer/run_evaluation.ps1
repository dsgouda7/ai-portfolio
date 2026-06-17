$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

python -m pip install -r evaluation/requirements.txt
python evaluation/run_dual_benchmark.py
