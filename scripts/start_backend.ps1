param([switch]$SampleOnly)
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$env:PYTHONPATH = "$projectRoot\.deps;$projectRoot\backend;$projectRoot\src"
if ($SampleOnly) { $env:RADAR_PORT = '' }
if (-not $SampleOnly -and -not $env:RADAR_PORT) {
    # Only choose an unambiguous CP2105 DATA interface; configuration remains a separate device step.
    $env:RADAR_PORT = python -c "from serial.tools import list_ports; ports = [p.device for p in list_ports.comports() if p.vid == 0x10c4 and p.pid == 0xea70 and 'Standard' in p.description]; print(ports[0] if len(ports) == 1 else '')" 2>$null
}
if (-not $env:VIDEO_CAMERA_INDEX) { $env:VIDEO_CAMERA_INDEX = '1' }
if (-not $env:VIDEO_SOURCE) { $env:VIDEO_SOURCE = $env:VIDEO_CAMERA_INDEX }
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log --log-level warning
