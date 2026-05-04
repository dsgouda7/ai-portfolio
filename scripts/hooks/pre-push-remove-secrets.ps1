# Pre-push hook: Strip secrets from notebooks before push (Windows)

Write-Host "🔍 Scanning notebooks for secrets..." -ForegroundColor Cyan

$notebooks = git diff --cached --name-only --diff-filter=ACM | Where-Object { $_ -match '\.ipynb$' }

foreach ($notebook in $notebooks) {
    $content = Get-Content $notebook -Raw
    
    if ($content -match 'API_KEY.*=.*"[^"]+"') {
        Write-Host "❌ Found secrets in $notebook" -ForegroundColor Red
        Write-Host "   Stripping secrets..." -ForegroundColor Yellow
        
        # Replace with empty strings
        $content = $content -replace '(API_KEY.*=.*")([^"]+)(")', '$1$3'
        $content = $content -replace '(SECRET.*=.*")([^"]+)(")', '$1$3'
        
        Set-Content $notebook -Value $content
        git add $notebook
        
        Write-Host "✅ Cleaned $notebook" -ForegroundColor Green
    }
}

Write-Host "✅ Secret scan complete" -ForegroundColor Green
