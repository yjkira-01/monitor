param([switch]$CheckOnly)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw '未找到Python，请先安装网页后端运行环境。' }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw '未找到Node.js/npm，请先安装网页前端运行环境。' }
foreach ($path in @('backend/app/main.py','frontend/node_modules/vite/package.json','frontend/dist/index.html')) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $path))) { throw "缺少网页资源：$path；请按README安装依赖并构建。" }
}
if($CheckOnly){Write-Output '网页版源码、已安装依赖和页面构建均存在。';return}
$logs=Join-Path $projectRoot 'artifacts/web-runtime'
$null=New-Item -ItemType Directory -Path $logs -Force
function Listeners([int]$Port) { @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) }
function Wait-Address([string]$Address,[string]$Expected,[int]$Seconds=75) {
    $deadline=(Get-Date).AddSeconds($Seconds)
    while((Get-Date) -lt $deadline) {
        try { $response=Invoke-WebRequest -Uri $Address -UseBasicParsing -TimeoutSec 2; if($response.StatusCode -eq 200 -and $response.Content.Contains($Expected)){return} }
        catch {}
        Start-Sleep -Milliseconds 300
    }
    throw "网页服务暂未就绪，请查看 $logs 中的启动日志。"
}
if(-not (Listeners 8000).Count) {
    # First launch uses a public sample. A camera can be selected later in the page.
    if(-not $env:VIDEO_SOURCE){$env:VIDEO_SOURCE=Join-Path $projectRoot 'data/raw/GMDCSA24/Subject 4/Fall/01.mp4'}
    $arguments=@('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+(Join-Path $PSScriptRoot 'start_backend.ps1')+'"'),'-SampleOnly')
    $null=Start-Process -FilePath powershell.exe -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'backend.log') -RedirectStandardError (Join-Path $logs 'backend-error.log')
}
Wait-Address 'http://127.0.0.1:8000/openapi.json' '"/api/video/status"'
if(-not (Listeners 5173).Count) {
    $arguments=@('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+(Join-Path $PSScriptRoot 'start_frontend.ps1')+'"'))
    $null=Start-Process -FilePath powershell.exe -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'frontend.log') -RedirectStandardError (Join-Path $logs 'frontend-error.log')
}
Wait-Address 'http://127.0.0.1:5173' 'id="root"' 30
Start-Process 'http://127.0.0.1:5173'
Write-Output '网页版已打开。初始账号：admin / admin123。'
