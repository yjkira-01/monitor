$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
python -u web/app.py --host 0.0.0.0 --port 5000 @args
