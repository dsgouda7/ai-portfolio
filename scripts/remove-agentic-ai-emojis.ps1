# Remove all decorative emojis from 03b-agentic-ai markdown files
# and replace emoji-based callouts with text equivalents

$targetDir = "c:\repos\ai-portfolio\notes\03b-agentic-ai"
$mdFiles = Get-ChildItem -Path $targetDir -Filter *.md -Recurse

# Define emoji replacement mappings
$emojiMap = @{
    # Decorative emojis to remove entirely
    '🎯 ' = ''
    '🚀 ' = ''
    '✅ ' = ''
    '✅' = '[Complete]'
    '⚡ ' = ''
    '📊 ' = ''
    '💡 ' = ''
    '⚠️ ' = ''
    '🔍 ' = ''
    '📝 ' = ''
    '🎓 ' = ''
    '🏆 ' = ''
    '💻 ' = ''
    '🔧 ' = ''
    '🎨 ' = ''
    '📈 ' = ''
    '📉 ' = ''
    '🔄 ' = ''
    '🌟 ' = ''
    '⭐ ' = ''
    '✨ ' = ''
    '🎉 ' = ''
    '🔥 ' = ''
    '💪 ' = ''
    '👍 ' = ''
    '👎 ' = ''
    '❌ ' = ''
    '✔️ ' = ''
    '➡️ ' = ''
    '⬆️ ' = ''
    '⬇️ ' = ''
    '📌 ' = ''
    '🎪 ' = ''
    '🏃‍♂️ ' = ''
    '🎭 ' = ''
    '🎬 ' = ''
    '🧠 ' = ''
    '💰 ' = ''
    '🎁 ' = ''
    '🔑 ' = ''
    '🎮 ' = ''
    '🔬 ' = ''
    '🎤 ' = ''
    '🧪 ' = ''
    '📚 ' = ''
    '🔔 ' = ''
    '📖 ' = ''
    '🧩 ' = ''
    '🔗 ' = ''
    '⚙️ ' = ''
    '💬 ' = ''
    '🎙️ ' = ''
    '📢 ' = ''
    '🛡️ ' = ''
    '🎲 ' = ''
    '🧮 ' = ''
    '📋 ' = ''
    '🏁 ' = ''
    '🛋️ ' = ''
    '📽️ ' = ''
    '⚖️ ' = ''
    '👷 ' = ''
    '🚨 ' = ''

    # Callout replacements
    '💡 **Insight' = '**Insight'
    '⚠️ **Warning' = '**Warning'
    '⚡ **Constraint' = '**Constraint'
    '📖 **Optional' = '**Optional'
    '➡️ **Forward' = '**Forward'
}

$totalReplacements = 0
$filesModified = 0

foreach ($file in $mdFiles) {
    Write-Host "Processing: $($file.FullName)" -ForegroundColor Cyan
    $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
    $originalContent = $content
    $fileReplacements = 0

    # Apply all emoji replacements
    foreach ($emoji in $emojiMap.Keys) {
        $replacement = $emojiMap[$emoji]
        $matchCount = ([regex]::Matches($content, [regex]::Escape($emoji))).Count
        if ($matchCount -gt 0) {
            $content = $content -replace [regex]::Escape($emoji), $replacement
            $fileReplacements += $matchCount
            Write-Host "  Replaced $matchCount instance(s) of '$emoji'" -ForegroundColor Yellow
        }
    }

    # Save if changes were made
    if ($content -ne $originalContent) {
        Set-Content -Path $file.FullName -Value $content -Encoding UTF8 -NoNewline
        $filesModified++
        $totalReplacements += $fileReplacements
        Write-Host "  Total replacements in file: $fileReplacements" -ForegroundColor Green
    } else {
        Write-Host "  No emojis found" -ForegroundColor Gray
    }
}

Write-Host "`n=== SUMMARY ===" -ForegroundColor Magenta
Write-Host "Files processed: $($mdFiles.Count)" -ForegroundColor White
Write-Host "Files modified: $filesModified" -ForegroundColor Green
Write-Host "Total emoji replacements: $totalReplacements" -ForegroundColor Green
