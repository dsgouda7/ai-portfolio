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
if ($genRefs.Count -gt 0) { Write-Output 'FOUND_GENSCRIPTS_REFS:'; $genRefs | Sort-Object -Unique } else { Write-Output 'NO_GENSCRIPTS_REFS_FOUND' }
if ($broken.Count -gt 0) { Write-Output 'BROKEN_LINKS:'; $broken | Sort-Object -Unique } else { Write-Output 'NO_BROKEN_LOCAL_LINKS' }
