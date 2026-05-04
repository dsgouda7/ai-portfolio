$outputFile = Join-Path (Resolve-Path .).Path 'scripts\check_md_links_output.txt'
$repo = Resolve-Path .
$mds = Get-ChildItem -Path $repo -Recurse -Include '*.md' -File -ErrorAction SilentlyContinue
$genRefs = @()
$broken = @()
foreach ($md in $mds) {
    try {
        $text = Get-Content $md.FullName -Raw -ErrorAction Stop
    } catch { continue }
    $pattern = '\[[^\]]*\]\(([^)]+)\)'
    $matches = [regex]::Matches($text, $pattern)
    foreach ($m in $matches) {
        $link = $m.Groups[1].Value.Trim()
        if ($link -match '^(http|https):') { continue }
        if ($link -match '^#') { continue }
        $linkNoAnchor = $link -replace '#.*$',''
        if ([System.IO.Path]::IsPathRooted($linkNoAnchor)) { $abs = $linkNoAnchor } else { $abs = Join-Path (Split-Path $md.FullName -Parent) $linkNoAnchor }
        if ($linkNoAnchor -match 'GenScripts') { $genRefs += "$($md.FullName) -> $link" }
        if (-not (Test-Path $abs)) { $broken += "$($md.FullName) -> $link" }
    }
}
$sb = New-Object System.Text.StringBuilder
if ($genRefs.Count -gt 0) { $sb.AppendLine('FOUND_GENSCRIPTS_REFS:') | Out-Null; $genRefs | Sort-Object -Unique | ForEach-Object { $sb.AppendLine($_) | Out-Null } } else { $sb.AppendLine('NO_GENSCRIPTS_REFS_FOUND') | Out-Null }
if ($broken.Count -gt 0) { $sb.AppendLine('BROKEN_LINKS:') | Out-Null; $broken | Sort-Object -Unique | ForEach-Object { $sb.AppendLine($_) | Out-Null } } else { $sb.AppendLine('NO_BROKEN_LOCAL_LINKS') | Out-Null }
[System.IO.File]::WriteAllText($outputFile, $sb.ToString())
Write-Output "WROTE_OUTPUT: $outputFile"
