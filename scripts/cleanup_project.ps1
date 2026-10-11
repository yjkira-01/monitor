param([switch]$Execute,[switch]$GeneratedOnly)
$ErrorActionPreference='Stop'
$workspaceRoot=[IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent)).TrimEnd('\')
if($workspaceRoot -ne 'D:\dachaung'){throw '清理范围必须为已确认的 D:\dachaung 项目'}
$evidence=Join-Path $workspaceRoot 'artifacts/cleanup_20261011'
New-Item -ItemType Directory -Path $evidence -Force | Out-Null
$buildPath=Join-Path $workspaceRoot 'desktop/release/CareMonitor-0.3.2-embedded-win-x64.build.json'
$build=Get-Content -LiteralPath $buildPath -Raw | ConvertFrom-Json
$currentBuild=Split-Path $build.embeddedManifest -Parent
$latest=Split-Path $build.executable -Parent
$protected=@($build.executable,$build.zip,$buildPath,$build.embeddedManifest,$build.embeddedArchive,
    (Join-Path $workspaceRoot 'desktop/model-assets.sha256.json')) | ForEach-Object {[IO.Path]::GetFullPath($_)}
$active=@(Get-CimInstance Win32_Process | Where-Object {$_.ProcessId -ne $PID})
$targets=[Collections.Generic.List[object]]::new()
function Assert-SafePath([string]$Path){
    $resolved=[IO.Path]::GetFullPath($Path).TrimEnd('\')
    if(-not $resolved.StartsWith($workspaceRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw "目标越出项目：$resolved"}
    if($protected | Where-Object {$_ -eq $resolved -or $_.StartsWith($resolved+'\',[StringComparison]::OrdinalIgnoreCase)}){throw "目标含受保护的当前资源：$resolved"}
    $probe=$resolved
    while($probe -ne $workspaceRoot){
        if(Test-Path -LiteralPath $probe){
            if((Get-Item -LiteralPath $probe -Force).Attributes -band [IO.FileAttributes]::ReparsePoint){throw "目标含链接：$probe"}
        }
        $probe=Split-Path $probe -Parent
    }
    if(Test-Path -LiteralPath $resolved -PathType Container){
        $links=@(Get-ChildItem -LiteralPath $resolved -Recurse -Force -Attributes ReparsePoint)
        if($links.Count){throw "目录含链接，不执行递归删除：$resolved"}
    }
    return $resolved
}
function Add-Target([string]$Path,[string]$Reason){
    if(-not (Test-Path -LiteralPath $Path)){return}
    $full=Assert-SafePath $Path
    if($targets | Where-Object {$_.path -eq $full -or $full.StartsWith($_.path+'\',[StringComparison]::OrdinalIgnoreCase)}){return}
    $item=Get-Item -LiteralPath $full -Force
    $files=if($item.PSIsContainer){@(Get-ChildItem -LiteralPath $full -Recurse -File -Force)}else{@($item)}
    $targets.Add([pscustomobject]@{path=$full;relative=$full.Substring($workspaceRoot.Length+1);reason=$Reason;
        bytes=[long](($files | Measure-Object Length -Sum).Sum);files=$files.Count;status='planned'})
}

# Only generated releases are selected; the current EXE, ZIP and build metadata are protected.
foreach($item in Get-ChildItem -LiteralPath (Join-Path $workspaceRoot 'desktop/release') -Force){
    if($item.Name -like 'CareMonitor*' -and $item.FullName -ne $latest -and $item.FullName -notin $protected){Add-Target $item.FullName '旧版发布目录或安装包'}
}
Add-Target (Join-Path $latest 'Runtime') '当前应用可重新释放的重复运行资源'
foreach($item in Get-ChildItem -LiteralPath (Join-Path $workspaceRoot 'desktop/.packaging') -Force){
    if($item.FullName -ne $currentBuild){Add-Target $item.FullName '旧打包工作目录或未交付候选'}
    else{foreach($child in Get-ChildItem -LiteralPath $item.FullName -Force){if($child.FullName -notin $protected){Add-Target $child.FullName '当前打包重复构建输出'}}}
}
foreach($item in Get-ChildItem -LiteralPath (Join-Path $workspaceRoot 'desktop') -Directory -Force){
    if($item.Name -like 'validation-*'){Add-Target $item.FullName '重复验证程序；最终结果已单独保存'}
}
foreach($project in @('ByteTrack','CareMonitor','CareMonitor.Bootstrap','CareMonitor.Bootstrap.Tests','CareMonitor.Tests','ModelBench')){
    foreach($output in @('bin','obj')){Add-Target (Join-Path $workspaceRoot "desktop/$project/$output") '可重新生成的编译输出'}
}
foreach($relative in @('artifacts/embedded_desktop_20261010/package','artifacts/embedded_desktop_20261010/package-local',
    'artifacts/device_upgrade_20261010/package','artifacts/recognition_repair_20261010/package-final')){
    foreach($item in Get-ChildItem -LiteralPath (Join-Path $workspaceRoot $relative) -Directory -Force){
        if($item.Name -ne 'ui-validation'){Add-Target $item.FullName '封包测试释放的重复程序和运行库；保留JSON结果'}
    }
}
foreach($item in Get-ChildItem -LiteralPath (Join-Path $workspaceRoot 'artifacts') -Directory -Force){
    if($item.Name -like 'browser-*' -or $item.Name -eq 'spatial-preview-profile'){Add-Target $item.FullName '隔离浏览器测试缓存'}
    elseif($item.Name -like 'pytest*' -or $item.Name -like 'tests_*' -or $item.Name -eq 'pip-temp'){Add-Target $item.FullName '测试或依赖安装临时目录'}
}
foreach($relative in @('.pytest_cache','.pip-tmp','.tmp_mmfall','.tmp_mmfi','.tmp_upfall_code')){Add-Target (Join-Path $workspaceRoot $relative) '缓存或未接入运行链路的临时研究克隆'}
foreach($item in Get-ChildItem -LiteralPath $workspaceRoot -File -Filter 'onnxruntime_profile__*.json'){Add-Target $item.FullName '一次性推理性能跟踪'}
foreach($relative in @('__pycache__','backend','src','tests','scripts','edge','fusion','web')){
    $directory=Join-Path $workspaceRoot $relative
    if($relative -eq '__pycache__'){Add-Target $directory 'Python字节码缓存'}
    elseif(Test-Path -LiteralPath $directory){foreach($cache in Get-ChildItem -LiteralPath $directory -Directory -Recurse -Filter '__pycache__'){Add-Target $cache.FullName 'Python字节码缓存'}}
}
Add-Target (Join-Path $workspaceRoot 'artifacts/pretrained/people_rf_detr_medium.failed_topk.onnx') '导出失败且未部署的模型文件'

# Delete transfer chunks only after checking the complete checkpoint and concatenated chunks.
$preview=Join-Path $workspaceRoot 'artifacts/training_preview_20261009'
$checkpoint=Join-Path $preview 'internvideo2_preview.pt'
if(Test-Path -LiteralPath $checkpoint){
    $metadata=Get-Content -LiteralPath (Join-Path $preview 'checkpoint.json') -Raw | ConvertFrom-Json
    $hash=(Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    if($hash -ne $metadata.checkpoint_sha256){throw '预览模型校验失败，不清理下载分片'}
    $parts=Join-Path $preview ('parts-'+$hash.Substring(0,12))
    if(Test-Path -LiteralPath $parts){
        $null=Assert-SafePath $parts
        $chunks=@(Get-ChildItem -LiteralPath $parts -File | Sort-Object {[int]$_.Name})
        $digest=[Security.Cryptography.IncrementalHash]::CreateHash([Security.Cryptography.HashAlgorithmName]::SHA256)
        $buffer=[byte[]]::new(1048576)
        try{
            for($i=0;$i -lt $chunks.Count;$i++){
                if($chunks[$i].Name -ne [string]$i){throw '下载分片编号不连续'}
                $stream=[IO.File]::OpenRead($chunks[$i].FullName)
                try{while(($count=$stream.Read($buffer,0,$buffer.Length)) -gt 0){$digest.AppendData($buffer,0,$count)}}finally{$stream.Dispose()}
            }
            $joined=[Convert]::ToHexString($digest.GetHashAndReset()).ToLowerInvariant()
        }finally{$digest.Dispose()}
        if($joined -ne $hash){throw '下载分片与完整模型不一致，不执行删除'}
        Add-Target $parts '已按SHA256确认与完整模型重复的下载分片'
    }
}

if($Execute -and -not $GeneratedOnly){throw '整批清理未获自动审批。执行仅支持已核验的发布/构建生成文件 -GeneratedOnly；其他条目只预览。'}
if($GeneratedOnly){
    $audit=Get-Content -LiteralPath (Join-Path $evidence 'generated-audit.json') -Raw | ConvertFrom-Json
    if(-not $audit.all_targets_in_workspace -or $audit.unknown_models.Count -gt 0 -or $audit.model_files -ne $audit.models_with_retained_copies){throw '发布/构建资源的保留副本核验未完成'}
    $audited=@($audit.generated_targets.path)
    $selected=[Collections.Generic.List[object]]::new()
    foreach($target in $targets){if($target.path -in $audited){$selected.Add($target)}}
    $targets=$selected
}
$plan=[ordered]@{root=$workspaceRoot;execute=[bool]$Execute;generatedOnly=[bool]$GeneratedOnly;created=(Get-Date).ToString('o');
    protected=$protected;targets=$targets;plannedBytes=($targets | Measure-Object bytes -Sum).Sum}
[IO.File]::WriteAllText((Join-Path $evidence 'plan.json'),($plan | ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
$summary=@($targets | Group-Object reason | ForEach-Object {[pscustomobject]@{reason=$_.Name;targets=$_.Count;gib=[Math]::Round(($_.Group | Measure-Object bytes -Sum).Sum/1GB,3)}})
$summary | ConvertTo-Json -Depth 3 | Write-Output
if(-not $Execute){Write-Output "预览：$($targets.Count)项，约$([Math]::Round($plan.plannedBytes/1GB,2))GiB，尚未删除";return}

$freeBefore=(Get-PSDrive -Name D).Free
foreach($target in $targets){
    try{
        $full=Assert-SafePath $target.path
        $inUse=@($active | Where-Object {
            ($_.ExecutablePath -and ($_.ExecutablePath -eq $full -or $_.ExecutablePath.StartsWith($full+'\',[StringComparison]::OrdinalIgnoreCase))) -or
            ($_.CommandLine -and ($_.CommandLine.IndexOf($full,[StringComparison]::OrdinalIgnoreCase) -ge 0 -or $_.CommandLine.IndexOf($full.Replace('\','/'),[StringComparison]::OrdinalIgnoreCase) -ge 0))
        })
        if($inUse.Count){$target.status='skipped-active-process';continue}
        Remove-Item -LiteralPath $full -Recurse -Force
        if(Test-Path -LiteralPath $full){throw '删除后路径仍存在'}
        $target.status='deleted'
    }catch{$target.status='failed';$target | Add-Member -NotePropertyName error -NotePropertyValue $_.Exception.Message}
}
$result=[ordered]@{root=$workspaceRoot;completed=(Get-Date).ToString('o');
    deletedTargets=@($targets | Where-Object status -eq 'deleted').Count;
    deletedBytes=($targets | Where-Object status -eq 'deleted' | Measure-Object bytes -Sum).Sum;
    freeBefore=$freeBefore;freeAfter=(Get-PSDrive -Name D).Free;targets=$targets}
[IO.File]::WriteAllText((Join-Path $evidence 'result.json'),($result | ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
Write-Output "删除$($result.deletedTargets)项，约$([Math]::Round($result.deletedBytes/1GB,2))GiB"
$pending=@($targets | Where-Object status -ne 'deleted')
if($pending.Count){$pending | Select-Object relative,status,error | ConvertTo-Json -Depth 3 | Write-Output}
