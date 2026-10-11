param([string]$App='CareMonitor/bin/Release/net8.0-windows/CareMonitor.exe',[string]$Evidence='../artifacts/device_validation',[switch]$PhysicalDevices)
$ErrorActionPreference='Stop'
$executable=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot $App))
if(-not (Test-Path -LiteralPath $executable)){
    if($PSBoundParameters.ContainsKey('App')){throw '指定的应用不存在，请先构建或提供有效路径'}
    $env:DOTNET_CLI_HOME=Join-Path $PSScriptRoot '.dotnet'
    $env:NUGET_PACKAGES=Join-Path $PSScriptRoot '.nuget'
    $env:MSBuildEnableWorkloadResolver='false'
    dotnet build (Join-Path $PSScriptRoot 'CareMonitor/CareMonitor.csproj') -c Release --source (Join-Path $PSScriptRoot 'packages')
    if($LASTEXITCODE -ne 0){throw '验证应用构建失败'}
}
$directory=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Evidence))
if(Test-Path -LiteralPath $directory){throw '验证目录已存在，请选择新目录'}
New-Item -ItemType Directory -Path $directory | Out-Null
$root=Split-Path $PSScriptRoot -Parent
$checks=[Collections.Generic.List[object]]::new()
function Run-Test([string]$Name,[string[]]$Arguments,[string]$Result){
    $start=[Diagnostics.ProcessStartInfo]::new($executable)
    $start.WorkingDirectory=$root;$start.UseShellExecute=$false;$start.WindowStyle=[Diagnostics.ProcessWindowStyle]::Hidden
    foreach($argument in $Arguments){$start.ArgumentList.Add($argument)}
    $process=[Diagnostics.Process]::Start($start)
    try{if(-not $process.WaitForExit(240000)){throw "验证超时 $Name，进程 $($process.Id)"};if($process.ExitCode -ne 0){throw "验证失败 $Name，退出码 $($process.ExitCode)"}}finally{$process.Dispose()}
    $data=Get-Content -LiteralPath $Result -Raw | ConvertFrom-Json
    if($Name -eq 'replay'){if($data.state -ne 'complete'){throw '公开回放未完成'}}
    elseif(-not $data.passed){throw "结果未通过 $Name"}
    if($Name -eq 'ui' -and (Test-Path -LiteralPath (Join-Path (Split-Path $Result -Parent) 'error.txt'))){throw 'UI 验证包含错误'}
    $copy=Join-Path $directory ($Name+'.json')
    if([IO.Path]::GetFullPath($Result) -ne [IO.Path]::GetFullPath($copy)){Copy-Item -LiteralPath $Result -Destination $copy}
    $checks.Add([pscustomobject]@{name=$Name;passed=$true})
    Write-Output "通过：$Name"
}
try{
    foreach($mode in @('cpu','gpu')){
        $arguments=@('--self-test','--realtime','--fall-model','gmd');if($mode -eq 'gpu'){$arguments+=@('--gpu','--profile-models')}
        Run-Test $mode $arguments (Join-Path (Split-Path $executable -Parent) 'self-test-result.json')
    }
    $video=Join-Path $root 'data/raw/GMDCSA24/Subject 4/ADL/01.mp4'
    $lifecycle=Join-Path $directory 'source-lifecycle.json'
    Run-Test 'lifecycle' @('--self-test','--gpu','--lifecycle-video',$video,'--eval-output',$lifecycle) $lifecycle
    $replay=Join-Path $directory 'public-replay.json'
    Run-Test 'replay' @('--self-test','--gpu','--recognition-replay',(Join-Path $root 'artifacts/device_upgrade_20261010/replay_cases.json'),'--fall-model','gmd','--eval-output',$replay) $replay
    Run-Test 'ui' @('--ui-smoke','--fall-model','gmd','--test-video',$video) (Join-Path (Split-Path $executable -Parent) 'ui-validation/result.json')
    if($PhysicalDevices){
        $devices=Join-Path $directory 'physical.json'
        Run-Test 'physical' @('--self-test','--gpu','--device-check','--eval-output',$devices) $devices
    }
    [IO.File]::WriteAllText((Join-Path $directory 'result.json'),([pscustomobject]@{passed=$true;checks=$checks}|ConvertTo-Json -Depth 4))
}catch{
    [IO.File]::WriteAllText((Join-Path $directory 'result.json'),([pscustomobject]@{passed=$false;checks=$checks;error=$_.ToString()}|ConvertTo-Json -Depth 4));throw
}
