$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if(-not (Get-Command python -ErrorAction SilentlyContinue)){throw '请先安装Python 3.10或更新版本，并加入PATH。'}
if(-not (Get-Command npm -ErrorAction SilentlyContinue)){throw '请先安装Node.js 20或更新版本。'}
python -m pip install -r requirements-web.txt
if($LASTEXITCODE -ne 0){throw 'Python依赖安装失败。'}
python scripts/fetch_web_resources.py
if($LASTEXITCODE -ne 0){throw '运行资源准备失败。'}
Push-Location -LiteralPath (Join-Path $projectRoot 'frontend')
try{
    npm ci
    if($LASTEXITCODE -ne 0){throw '页面依赖安装失败。'}
    npm run build
    if($LASTEXITCODE -ne 0){throw '页面构建失败。'}
}finally{Pop-Location}
Write-Output '网页版准备完成，请双击根目录的“启动网页版.cmd”。'
