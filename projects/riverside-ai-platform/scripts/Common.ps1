Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-NonEmptyString {
    param([object]$Value, [string]$Name)
    if ($Value -isnot [string] -or [string]::IsNullOrWhiteSpace($Value)) {
        throw "Required string '$Name' is missing."
    }
    if ($Value.Contains("`r") -or $Value.Contains("`n")) {
        throw "Input '$Name' must be a single line."
    }
    return $Value
}

function Assert-Sha256 {
    param([object]$Value, [string]$Name)
    $text = Assert-NonEmptyString $Value $Name
    if ($text -cnotmatch '^[a-f0-9]{64}$') {
        throw "Input '$Name' must be a lowercase SHA-256 digest."
    }
    return $text
}

function Get-RequiredProperty {
    param([object]$Object, [string]$Name, [string]$Context = 'input')
    if ($null -eq $Object -or $Object.PSObject.Properties.Name -notcontains $Name) {
        throw "Required property '$Context.$Name' is missing."
    }
    return $Object.$Name
}

function Read-JsonObject {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "JSON input does not exist: $Path"
    }
    $value = Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json -Depth 100
    if ($null -eq $value -or $value -is [System.Array]) {
        throw "JSON input must contain one object: $Path"
    }
    return $value
}

function Get-FileSha256 {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file does not exist: $Path"
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-DirectorySha256 {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Required directory does not exist: $Path"
    }
    $root = (Resolve-Path -LiteralPath $Path).Path
    $files = @(Get-ChildItem -LiteralPath $root -File -Recurse | Sort-Object {
        [IO.Path]::GetRelativePath($root, $_.FullName).Replace('\', '/')
    })
    if ($files.Count -eq 0) {
        throw "Directory contains no files: $Path"
    }
    $entries = foreach ($file in $files) {
        $relative = [IO.Path]::GetRelativePath($root, $file.FullName).Replace('\', '/')
        "$relative`t$(Get-FileSha256 $file.FullName)"
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($entries -join "`n") + "`n")
    $hash = [Security.Cryptography.SHA256]::HashData($bytes)
    return [Convert]::ToHexString($hash).ToLowerInvariant()
}

function Assert-Digest {
    param([string]$Path, [string]$Expected, [string]$Name)
    $expectedDigest = Assert-Sha256 $Expected $Name
    $actual = if (Test-Path -LiteralPath $Path -PathType Container) {
        Get-DirectorySha256 $Path
    } else {
        Get-FileSha256 $Path
    }
    if ($actual -cne $expectedDigest) {
        throw "SHA-256 mismatch for '$Name': expected $expectedDigest, observed $actual."
    }
    return $actual
}

function ConvertTo-TemplateScalar {
    param([object]$Value, [string]$Name)
    $text = Assert-NonEmptyString ([string]$Value) $Name
    if ($text -notmatch '^[A-Za-z0-9][A-Za-z0-9._:/@+\\-]*$') {
        throw "Input '$Name' contains characters that are unsafe for scalar template substitution."
    }
    return $text.Replace('\', '/')
}

function Expand-TemplateFile {
    param(
        [string]$TemplatePath,
        [string]$OutputPath,
        [hashtable]$Values,
        [hashtable]$LiteralValues = @{}
    )
    if (-not (Test-Path -LiteralPath $TemplatePath -PathType Leaf)) {
        throw "Template does not exist: $TemplatePath"
    }
    $content = Get-Content -LiteralPath $TemplatePath -Raw -Encoding utf8
    foreach ($key in ($Values.Keys | Sort-Object)) {
        $token = "__$key`__"
        if (-not $content.Contains($token)) {
            throw "Template token '$token' was not found in $TemplatePath."
        }
        $content = $content.Replace($token, [string]$Values[$key])
    }
    foreach ($sentinel in ($LiteralValues.Keys | Sort-Object)) {
        if (-not $content.Contains($sentinel)) {
            throw "Template sentinel '$sentinel' was not found in $TemplatePath."
        }
        $content = $content.Replace($sentinel, [string]$LiteralValues[$sentinel])
    }
    $unresolved = [regex]::Matches($content, '__RIVERSIDE_[A-Z0-9_]+__') | ForEach-Object Value | Sort-Object -Unique
    $remainingSentinels = @($LiteralValues.Keys | Where-Object { $content.Contains($_) })
    if ($unresolved.Count -gt 0 -or $remainingSentinels.Count -gt 0) {
        throw "Unresolved template values in ${TemplatePath}: $(@($unresolved) + $remainingSentinels -join ', ')"
    }
    $parent = Split-Path -Parent $OutputPath
    if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    [IO.File]::WriteAllText($OutputPath, $content, [Text.UTF8Encoding]::new($false))
}

function Assert-AzureContext {
    param([string]$SubscriptionId, [string]$TenantId)
    Assert-NonEmptyString $SubscriptionId 'subscription_id' | Out-Null
    Assert-NonEmptyString $TenantId 'tenant_id' | Out-Null
    $accountJson = & az account show --only-show-errors --output json
    if ($LASTEXITCODE -ne 0) { throw 'Azure CLI is not authenticated. Use az login or workload/managed identity login.' }
    $account = $accountJson | ConvertFrom-Json
    if ($account.id -cne $SubscriptionId -or $account.tenantId -cne $TenantId) {
        throw "Azure CLI context does not match the explicit subscription and tenant inputs."
    }
}

function Invoke-AzChecked {
    param([string[]]$Arguments, [switch]$CaptureJson)
    Write-Host ('az ' + ($Arguments -join ' '))
    $output = & az @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI command failed: az $($Arguments -join ' ')"
    }
    if ($CaptureJson) {
        if ([string]::IsNullOrWhiteSpace(($output -join "`n"))) { return $null }
        return ($output -join "`n") | ConvertFrom-Json -Depth 100
    }
    return $output
}

function Write-JsonEvidence {
    param([object]$Value, [string]$Path)
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $json = $Value | ConvertTo-Json -Depth 100
    [IO.File]::WriteAllText($Path, $json + "`n", [Text.UTF8Encoding]::new($false))
}
