$RootSetup = Join-Path $PSScriptRoot "..\setup.ps1"
& $RootSetup @args
exit $LASTEXITCODE
