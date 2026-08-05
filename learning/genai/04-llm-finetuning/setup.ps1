<#
.SYNOPSIS
    Runs the shared applied-LLM environment setup for this chapter.
#>
param(
    [switch]$SkipKernel
)

$ErrorActionPreference = "Stop"
$SharedSetup = Join-Path $PSScriptRoot "..\_llm-shared\setup.ps1"

& $SharedSetup @PSBoundParameters
