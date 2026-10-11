param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)).TrimEnd('\')
if ($workspaceRoot -ne 'D:\dachaung') { throw '只能在已核对的 D:\dachaung 项目内整理。' }
$archiveRoot = Join-Path $workspaceRoot 'archive'
$evidence = Join-Path $workspaceRoot 'artifacts/cleanup_20261011'
$null = New-Item -ItemType Directory -Path $archiveRoot,$evidence -Force
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$active = @(Get-CimInstance Win32_Process | Where-Object ProcessId -ne $PID)
$protected = @('.deps','frontend','data','desktop','backend/app.db','MEMORY.md','archive') | ForEach-Object { Join-Path $workspaceRoot $_ }
function Assert-Bounded([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    if (-not $full.StartsWith($workspaceRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw "越出项目：$full" }
    foreach ($keep in $protected) {
        if ($full -eq $keep -or $keep.StartsWith($full+'\',[StringComparison]::OrdinalIgnoreCase) -or $full.StartsWith($keep+'\',[StringComparison]::OrdinalIgnoreCase)) { throw "属于保留内容：$full" }
    }
    for ($probe=$full; $probe -ne $workspaceRoot; $probe=Split-Path $probe -Parent) {
        if ((Get-Item -LiteralPath $probe -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "含链接：$probe" }
    }
    if ((Get-Item -LiteralPath $full -Force).PSIsContainer) {
        if (@(Get-ChildItem -LiteralPath $full -Recurse -Force -Attributes ReparsePoint).Count) { throw "目录内含链接：$full" }
    }
    return $full
}
function Get-Sha([IO.Stream]$Stream) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($Stream))).Replace('-','').ToLowerInvariant() }
    finally { $sha.Dispose() }
}
$groups = [ordered]@{
    'research-dependencies' = @('artifacts/model_libraries','artifacts/action_export_dependencies','artifacts/fall_export_dependencies','artifacts/onnx_export_dependencies','artifacts/transfer_dependencies')
    'test-workspaces-20261011' = @('.pytest_cache','.pip-tmp')
    'legacy-tools' = @('.tmp_mmfall','.tmp_mmfi','.tmp_upfall_code','cursor-orchestrator','tasks','.venv')
    'profiling-cache-20261011' = @('__pycache__')
}
foreach ($dir in Get-ChildItem -LiteralPath (Join-Path $workspaceRoot 'artifacts') -Directory -Force) {
    if ($dir.Name -like 'browser-*' -or $dir.Name -eq 'spatial-preview-profile' -or $dir.Name -like 'pytest*' -or $dir.Name -like 'tests_*' -or $dir.Name -eq 'pip-temp') {
        $groups['test-workspaces-20261011'] += $dir.FullName.Substring($workspaceRoot.Length+1)
    }
}
foreach ($file in Get-ChildItem -LiteralPath $workspaceRoot -File -Filter 'onnxruntime_profile__*.json') { $groups['profiling-cache-20261011'] += $file.Name }
foreach ($source in @('backend','src','tests','scripts','edge','fusion','web')) {
    foreach ($cache in Get-ChildItem -LiteralPath (Join-Path $workspaceRoot $source) -Directory -Recurse -Filter '__pycache__') {
        $groups['profiling-cache-20261011'] += $cache.FullName.Substring($workspaceRoot.Length+1)
    }
}
$plan = @()
foreach ($groupName in $groups.Keys) {
    foreach ($relative in $groups[$groupName]) {
        $path = Join-Path $workspaceRoot $relative
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $path = Assert-Bounded $path
        $users = @($active | Where-Object {
            ($_.ExecutablePath -and ($_.ExecutablePath -eq $path -or $_.ExecutablePath.StartsWith($path+'\',[StringComparison]::OrdinalIgnoreCase))) -or
            ($_.CommandLine -and ($_.CommandLine.Contains($path) -or $_.CommandLine.Contains($path.Replace('\','/'))))
        })
        if ($users.Count) { Write-Output "跳过使用中的目录：$relative"; continue }
        $item = Get-Item -LiteralPath $path -Force
        $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $path -File -Recurse -Force) } else { @($item) }
        $plan += [pscustomobject]@{group=$groupName;relative=$relative.Replace('\','/');path=$path;files=$files.Count;bytes=[long](($files | Measure-Object Length -Sum).Sum)}
    }
}
$planReport = [ordered]@{root=$workspaceRoot;created=(Get-Date).ToString('o');execute=[bool]$Execute;targets=$plan;files=($plan | Measure-Object files -Sum).Sum;bytes=($plan | Measure-Object bytes -Sum).Sum}
[IO.File]::WriteAllText((Join-Path $evidence 'consolidation-plan.json'),($planReport | ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
$plan | Group-Object group | ForEach-Object { [pscustomobject]@{archive=$_.Name;targets=$_.Count;files=($_.Group | Measure-Object files -Sum).Sum;GiB=[Math]::Round(($_.Group | Measure-Object bytes -Sum).Sum/1GB,3)} } | ConvertTo-Json
if (-not $Execute) { Write-Output '只预览；执行时先完整归档并核对逐文件SHA256，原内容均可恢复。'; return }
$freeBefore = (Get-PSDrive -Name D).Free
$results = @()
foreach ($batch in $plan | Group-Object group) {
    $zipPath = Join-Path $archiveRoot ($batch.Name+'.zip')
    $manifestPath = Join-Path $archiveRoot ($batch.Name+'.manifest.json')
    if (Test-Path -LiteralPath $zipPath) { throw "归档已存在，不覆盖：$zipPath" }
    $records = [Collections.Generic.List[object]]::new()
    $directories = [Collections.Generic.List[string]]::new()
    $zip = [IO.Compression.ZipFile]::Open($zipPath,[IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($target in $batch.Group) {
            $path = Assert-Bounded $target.path
            $item = Get-Item -LiteralPath $path -Force
            $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $path -File -Recurse -Force) } else { @($item) }
            if ($item.PSIsContainer) {
                $allDirs = @($item) + @(Get-ChildItem -LiteralPath $path -Directory -Recurse -Force)
                foreach ($dir in $allDirs) {
                    $name = $dir.FullName.Substring($workspaceRoot.Length+1).Replace('\','/')+'/'
                    $directories.Add($name)
                    $null = $zip.CreateEntry($name)
                }
            }
            foreach ($file in $files) {
                $name = $file.FullName.Substring($workspaceRoot.Length+1).Replace('\','/')
                $sha = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                $entry = [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$file.FullName,$name,[IO.Compression.CompressionLevel]::Optimal)
                $records.Add([pscustomobject]@{path=$name;bytes=$file.Length;sha256=$sha})
            }
        }
    } finally { $zip.Dispose() }
    Write-Output "$($batch.Name)：已归档$($records.Count)份文件，正在逐份校验。"
    $zip = [IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        if ($zip.Entries.Count -ne $records.Count+$directories.Count) { throw '归档条目数量不一致。' }
        foreach ($record in $records) {
            $entry = $zip.GetEntry($record.path)
            if (-not $entry -or $entry.Length -ne $record.bytes) { throw "长度不一致：$($record.path)" }
            $stream = $entry.Open()
            try { $sha = Get-Sha $stream } finally { $stream.Dispose() }
            if ($sha -ne $record.sha256) { throw "归档内容校验失败：$($record.path)" }
            if ((Get-FileHash -LiteralPath (Join-Path $workspaceRoot $record.path) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sha) { throw "源文件期间变化：$($record.path)" }
        }
    } finally { $zip.Dispose() }
    $manifest = [ordered]@{root=$workspaceRoot;created=(Get-Date).ToString('o');archive=(Split-Path $zipPath -Leaf);archiveSha256=(Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant();targets=@($batch.Group.relative);directories=$directories;files=$records;verified=$true}
    [IO.File]::WriteAllText($manifestPath,($manifest | ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
    # Original content is removed only after a complete, independently read and verified copy exists.
    foreach ($target in $batch.Group) {
        $path = Assert-Bounded $target.path
        $live = @(Get-CimInstance Win32_Process | Where-Object ProcessId -ne $PID | Where-Object {
            ($_.ExecutablePath -and ($_.ExecutablePath -eq $path -or $_.ExecutablePath.StartsWith($path+'\',[StringComparison]::OrdinalIgnoreCase))) -or
            ($_.CommandLine -and ($_.CommandLine.Contains($path) -or $_.CommandLine.Contains($path.Replace('\','/'))))
        })
        if ($live.Count) { Write-Output "归档后保留使用中的路径：$($target.relative)"; continue }
        Remove-Item -LiteralPath $path -Recurse -Force
        if (Test-Path -LiteralPath $path) { throw "原路径未成功收拢：$path" }
        $results += $target
    }
    Write-Output "$($batch.Name)：校验通过，已收拢到 $zipPath。"
}
$result = [ordered]@{completed=(Get-Date).ToString('o');targets=$results;consolidatedFiles=($results | Measure-Object files -Sum).Sum;originalBytes=($results | Measure-Object bytes -Sum).Sum;archives=@(Get-ChildItem -LiteralPath $archiveRoot -File -Filter '*.zip' | Select-Object Name,Length);freeBefore=$freeBefore;freeAfter=(Get-PSDrive -Name D).Free}
[IO.File]::WriteAllText((Join-Path $evidence 'consolidation-result.json'),($result | ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
Write-Output "已归档收拢$($result.consolidatedFiles)份文件，所有原内容可按清单恢复。"
