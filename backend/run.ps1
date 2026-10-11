$backendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $backendRoot
$env:PYTHONPATH = $backendRoot
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
