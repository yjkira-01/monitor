param(
    [string]$BuildReport = 'release/CareMonitor-0.3.2-embedded-win-x64.build.json',
    [string]$Artifacts = '../artifacts/model_optimization_20261011/package',
    [switch]$NativeChecks
)
$ErrorActionPreference='Stop'
$build=Get-Content -LiteralPath (Join-Path $PSScriptRoot $BuildReport) -Raw | ConvertFrom-Json
$manifest=Get-Content -LiteralPath $build.embeddedManifest -Raw | ConvertFrom-Json
$evidence=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Artifacts))
if(Test-Path -LiteralPath $evidence){throw '验证输出已存在，请选用新目录'}
New-Item -ItemType Directory -Path $evidence | Out-Null
$single=Join-Path $evidence '单文件 空目录'
$cache=Join-Path $single 'Runtime'
New-Item -ItemType Directory -Path $single | Out-Null
$executable=Join-Path $single 'CareMonitor.exe'
Copy-Item -LiteralPath $build.executable -Destination $executable
$env:DOTNET_BUNDLE_EXTRACT_BASE_DIR=Join-Path $evidence 'dotnet-bundle'
$checks=[Collections.Generic.List[object]]::new()

function Assert-Check([bool]$Condition,[string]$Name){
    if(-not $Condition){throw "验证失败：$Name"}
    $checks.Add([pscustomobject]@{name=$Name;passed=$true})
    Write-Host "通过：$Name"
}
function Start-Exact([string[]]$Arguments){
    $start=[Diagnostics.ProcessStartInfo]::new($executable)
    $start.WorkingDirectory=$evidence
    $start.UseShellExecute=$false
    $start.CreateNoWindow=$true
    $start.WindowStyle=[Diagnostics.ProcessWindowStyle]::Hidden
    # Self-contained hosts must not need the SDK or a machine-wide runtime.
    $start.Environment['DOTNET_ROOT']=Join-Path $evidence 'no-installed-dotnet'
    $start.Environment['DOTNET_ROOT_X64']=Join-Path $evidence 'no-installed-dotnet'
    foreach($argument in $Arguments){$start.ArgumentList.Add($argument)}
    return [Diagnostics.Process]::Start($start)
}
function Run-Exact([string]$Name,[string[]]$Arguments,[int]$Expected=0){
    $report=Join-Path $evidence ($Name+'.json')
    $all=@('--bootstrap-report',$report)+$Arguments
    $watch=[Diagnostics.Stopwatch]::StartNew()
    $process=Start-Exact $all
    try{
        if(-not $process.WaitForExit(240000)){throw "验证进程超时：$Name，进程编号 $($process.Id)"}
        Assert-Check ($process.ExitCode -eq $Expected) "$Name 退出码 $Expected"
    }finally{$process.Dispose()}
    $result=Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
    $checks.Add([pscustomobject]@{name=$Name;elapsedMs=$watch.Elapsed.TotalMilliseconds;installation=$result.installation})
    return $result
}
function Verify-Payload([string]$Directory){
    foreach($file in $manifest.files){
        $path=Join-Path $Directory $file.path
        if(-not (Test-Path -LiteralPath $path)){throw "缺少释放资源：$($file.path)"}
        if((Get-Item -LiteralPath $path).Length -ne $file.length -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256){throw "释放资源字节不一致：$($file.path)"}
    }
}

try{
    Assert-Check (@(Get-ChildItem -LiteralPath (Split-Path $build.executable -Parent) -Force).Count -eq 1) '发布目录只有一个 EXE'
    $first=Run-Exact 'first' @('--bootstrap-quiet','--bootstrap-extract-only')
    Assert-Check ($first.installed -and -not $first.installation.reused -and -not $first.installation.repaired) '首次离线释放完整资源'
    $runtime=$first.installation.directory
    Assert-Check ($runtime.StartsWith($cache+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) '默认释放到 EXE 旁 Runtime，忽略外部工作目录'
    Verify-Payload $runtime
    Assert-Check $true '所有释放文件与内嵌清单 SHA256 一致'
    $model=Join-Path $runtime 'Assets/Realtime/rtmo.onnx'
    $time=(Get-Item -LiteralPath $model).LastWriteTimeUtc
    $repeat=Run-Exact 'repeat' @('--bootstrap-quiet','--bootstrap-extract-only')
    Assert-Check ($repeat.installation.reused -and (Get-Item -LiteralPath $model).LastWriteTimeUtc -eq $time) '再次启动复用且不改写模型'

    $icon=Join-Path $runtime 'Assets/AppIcon.ico'
    Remove-Item -LiteralPath $icon
    $missing=Run-Exact 'repair-missing' @('--bootstrap-quiet','--bootstrap-extract-only')
    Assert-Check ($missing.installation.repaired -and (Test-Path -LiteralPath $icon)) '缺失图标自动修复'
    $originalTime=(Get-Item -LiteralPath $model).LastWriteTimeUtc
    $stream=[IO.File]::Open($model,[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    try{$value=$stream.ReadByte();$stream.Position=0;$stream.WriteByte([byte]($value -bxor 1))}finally{$stream.Dispose()}
    [IO.File]::SetLastWriteTimeUtc($model,$originalTime)
    $damaged=Run-Exact 'repair-corrupt' @('--bootstrap-quiet','--bootstrap-extract-only')
    Verify-Payload $runtime
    Assert-Check $damaged.installation.repaired '同长度同修改时间的模型损坏自动修复'

    $parallelCache=Join-Path $evidence 'parallel-cache'
    $processes=@()
    for($index=0;$index -lt 2;$index++){
        $arguments=@('--bootstrap-cache',$parallelCache,'--bootstrap-report',(Join-Path $evidence "parallel-$index.json"),'--bootstrap-quiet','--bootstrap-extract-only')
        $processes+=Start-Exact $arguments
    }
    foreach($process in $processes){try{if(-not $process.WaitForExit(240000) -or $process.ExitCode -ne 0){throw '并发释放失败'}}finally{$process.Dispose()}}
    $parallel=@(0,1 | ForEach-Object {Get-Content -LiteralPath (Join-Path $evidence "parallel-$_.json") -Raw | ConvertFrom-Json})
    Assert-Check (@($parallel | Where-Object {-not $_.installation.reused}).Count -eq 1) '两个独立进程只释放一次'
    Verify-Payload $parallel[0].installation.directory

    $null=Run-Exact 'startup-window' @('--bootstrap-extract-only')
    Assert-Check $true '准备界面启动与关闭正常'
    $failure=Run-Exact 'child-failure' @('--bootstrap-quiet','--self-test','--evaluation-list',(Join-Path $evidence '不存在的清单.json')) 1
    Assert-Check ($failure.exitCode -eq 1 -and (Test-Path -LiteralPath (Join-Path $runtime 'self-test-error.txt'))) '子程序失败退出码正确转发'
    Remove-Item -LiteralPath (Join-Path $runtime 'self-test-error.txt')

    $moved=Join-Path $evidence '移动后 程序目录'
    $sourcePath=[IO.Path]::GetFullPath($single)
    $targetPath=[IO.Path]::GetFullPath($moved)
    $boundary=$evidence+[IO.Path]::DirectorySeparatorChar
    if(-not $sourcePath.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase) -or -not $targetPath.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase) -or ((Get-Item -LiteralPath $single -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw '测试移动路径越界或为目录链接'}
    Move-Item -LiteralPath $sourcePath -Destination $targetPath
    $single=$targetPath
    $executable=Join-Path $single 'CareMonitor.exe'
    $cache=Join-Path $single 'Runtime'
    $relocated=Run-Exact 'relocated' @('--bootstrap-quiet','--bootstrap-extract-only')
    $runtime=$relocated.installation.directory
    Assert-Check ($relocated.installation.reused -and $runtime.StartsWith($cache+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) '移动程序文件夹后复用资源并使用新路径'
    Verify-Payload $runtime

    if($NativeChecks){
        foreach($mode in @('cpu','gpu')){
            $arguments=@('--bootstrap-quiet','--self-test','--realtime','--fall-model','gmd')
            if($mode -eq 'gpu'){$arguments+=@('--gpu','--profile-models')}
            $result=Run-Exact "native-$mode" $arguments
            $self=Join-Path $runtime 'self-test-result.json'
            Assert-Check ((Get-Content -LiteralPath $self -Raw | ConvertFrom-Json).passed -and $result.exitCode -eq 0) "$mode 原生自检通过"
            Copy-Item -LiteralPath $self -Destination (Join-Path $evidence "native-$mode-result.json")
        }
        $video=Join-Path (Split-Path $PSScriptRoot -Parent) 'data/raw/GMDCSA24/Subject 4/ADL/01.mp4'
        $result=Run-Exact 'native-ui' @('--bootstrap-quiet','--ui-smoke','--test-video',$video)
        $ui=Join-Path $runtime 'ui-validation/result.json'
        $uiError=Join-Path $runtime 'ui-validation/error.txt'
        Assert-Check ((Test-Path -LiteralPath $ui) -and -not (Test-Path -LiteralPath $uiError)) '公开回放界面未产生错误'
        $uiResult=Get-Content -LiteralPath $ui -Raw | ConvertFrom-Json
        Assert-Check ($uiResult.passed -and $uiResult.navigation_clicks -eq 7 -and $uiResult.window_icon -and $uiResult.single_person_score_checks -gt 0) '识别更新、八页导航、详情和图标通过'
        Assert-Check ($uiResult.report_native_formatting -and $uiResult.legacy_report_evidence_fold -and $uiResult.local_report_generation) '原生报告排版、历史依据与生成通过'
        Assert-Check ($uiResult.monitor_zone_cancel_delete_source_isolation) '区域取消、删除与来源隔离通过'
        Assert-Check ($uiResult.short_window_details_visible -and $uiResult.playback_controls_unthrottled) '短窗口详情与不受详情节流的播放按钮通过'
        $idle=Join-Path $runtime 'ui-validation/idle-close.json'
        Assert-Check ((Test-Path -LiteralPath $idle) -and (Get-Content -LiteralPath $idle -Raw | ConvertFrom-Json).passed) '设备断开后关闭无窗口重入'
        Copy-Item -LiteralPath $ui -Destination (Join-Path $evidence 'native-ui-result.json')
        Copy-Item -LiteralPath (Join-Path $runtime 'ui-validation/monitor-reopened.png') -Destination (Join-Path $evidence 'monitor.png')
    }
    $adjacent=@(Get-ChildItem -LiteralPath $single -Force | Sort-Object Name)
    Assert-Check ($adjacent.Count -eq 2 -and $adjacent[0].Name -eq 'CareMonitor.exe' -and $adjacent[1].Name -eq 'Runtime') '运行后资源仅位于 EXE 旁 Runtime 目录'
    Assert-Check ((Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant() -eq $build.executableSha256) '测试前后 EXE 哈希一致'
    [IO.File]::WriteAllText((Join-Path $evidence 'result.json'),([pscustomobject]@{passed=$true;nativeChecks=[bool]$NativeChecks;checks=$checks} | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
    Write-Output "单文件验证完成：$evidence"
}catch{
    [IO.File]::WriteAllText((Join-Path $evidence 'result.json'),([pscustomobject]@{passed=$false;nativeChecks=[bool]$NativeChecks;checks=$checks;error=$_.ToString()} | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
    throw
}
