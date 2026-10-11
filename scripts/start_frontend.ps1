param([switch]$Development)
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location "$projectRoot\frontend"
if ($Development) {
    npx vite --host 0.0.0.0 --port 5173
} else {
    if (-not (Test-Path -LiteralPath "$projectRoot\frontend\dist\index.html")) {
        npm run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed' }
    }
    npm run serve
}
