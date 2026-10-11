param(
    [string]$Output = 'release/CareMonitor-0.3.2-embedded-win-x64',
    [string]$ZipName = 'CareMonitor-0.3.2-embedded-win-x64.zip',
    [switch]$SkipZip
)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.IO.Compression,System.IO.Compression.FileSystem
$desktopRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$env:DOTNET_CLI_HOME=Join-Path $desktopRoot '.dotnet'
$env:NUGET_PACKAGES=Join-Path $desktopRoot '.nuget'
$env:MSBuildEnableWorkloadResolver='false'
$env:DOTNET_CLI_TELEMETRY_OPTOUT='1'
$destination=[IO.Path]::GetFullPath((Join-Path $desktopRoot $Output))
if(-not $destination.StartsWith($desktopRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw '发布路径必须在desktop目录内'}
if(Test-Path -LiteralPath $destination){throw '请选择尚不存在的新发布目录，避免覆盖已交付应用'}
if([IO.Path]::GetFileName($ZipName) -ne $ZipName){throw 'ZipName只能是文件名'}
$zip=Join-Path (Split-Path $destination -Parent) $ZipName
if(-not $SkipZip -and (Test-Path -LiteralPath $zip)){throw 'ZIP已存在，请使用新的发布文件名'}
$name=[IO.Path]::GetFileName($destination)
$work=Join-Path $desktopRoot ('.packaging/'+$name+'-'+[Guid]::NewGuid().ToString('N'))
$payload=Join-Path $work 'payload'
$archivePath=Join-Path $work 'payload.zip'
$manifestPath=Join-Path $work 'payload.json'
$bootstrapOutput=Join-Path $work 'bootstrap'
$feed=Join-Path $desktopRoot 'packages'
$restoreSource=if(Test-Path -LiteralPath $feed){@('--source',$feed)}else{@()}
New-Item -ItemType Directory -Path $work -Force | Out-Null

# Keep build outputs away from any development application the user may be running.
$appBuild=Join-Path $work 'app-build/'
dotnet publish (Join-Path $desktopRoot 'CareMonitor/CareMonitor.csproj') -c Release -r win-x64 --self-contained true -p:RuntimeFrameworkVersion=8.0.14 -p:PublishSingleFile=false -p:PublishTrimmed=false -p:DebugType=None "-p:OutputPath=$appBuild" @restoreSource -o $payload
if($LASTEXITCODE -ne 0){throw '桌面应用发布失败'}
Copy-Item -LiteralPath (Join-Path $desktopRoot 'README.md') -Destination (Join-Path $payload '使用说明.md')
$docDirectory=Join-Path $payload 'docs'
New-Item -ItemType Directory -Path $docDirectory -Force | Out-Null
foreach($doc in @('binary_fall_selection_20261009.md','realtime_recognition_20261010.md','embedded_desktop_20261010.md','device_upgrade_20261010.md','recognition_repair_20261011.md','stereo_calibration_guide.md','model_optimization_20261011.md')){
    Copy-Item -LiteralPath (Join-Path (Split-Path $desktopRoot -Parent) ('docs/'+$doc)) -Destination (Join-Path $docDirectory $doc)
}
$files=@(Get-ChildItem -LiteralPath $payload -Recurse -File | Where-Object Extension -ne '.pdb' | Sort-Object FullName)
foreach($file in $files){
    $relative=$file.FullName.Substring($payload.Length+1)
    if($relative -match '^(ui-validation|self-test-data)[\\/]' -or $file.Name -match '^(self-test|ui-startup-error)'){
        throw "发布资源混入测试文件：$relative"
    }
}
$hashes=@($files | ForEach-Object { [pscustomobject]@{file=$_.FullName.Substring($payload.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()} })
[IO.File]::WriteAllText((Join-Path $payload 'SHA256.json'),($hashes | ConvertTo-Json -Depth 4),[Text.UTF8Encoding]::new($false))
$files=@(Get-ChildItem -LiteralPath $payload -Recurse -File | Where-Object Extension -ne '.pdb' | Sort-Object FullName)
$entries=@($files | ForEach-Object { [pscustomobject]@{path=$_.FullName.Substring($payload.Length+1).Replace('\','/');length=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()} })
$stream=[IO.File]::Open($archivePath,[IO.FileMode]::CreateNew)
$archive=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create)
try {
    foreach($file in $files){
        $relative=$file.FullName.Substring($payload.Length+1).Replace('\','/')
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$file.FullName,$relative,[IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
} finally {$archive.Dispose();$stream.Dispose()}
$archiveHash=(Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$manifest=[pscustomobject]@{version='0.3.2';archiveSha256=$archiveHash;files=$entries}
[IO.File]::WriteAllText($manifestPath,($manifest | ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
dotnet publish (Join-Path $desktopRoot 'CareMonitor.Bootstrap/CareMonitor.Bootstrap.csproj') -c Release -r win-x64 --self-contained true -p:RuntimeFrameworkVersion=8.0.14 "-p:PayloadArchive=$archivePath" "-p:PayloadManifest=$manifestPath" @restoreSource -o $bootstrapOutput
if($LASTEXITCODE -ne 0){throw '内嵌资源启动器发布失败'}
$bootstrapFiles=@(Get-ChildItem -LiteralPath $bootstrapOutput -File)
if($bootstrapFiles.Count -ne 1 -or $bootstrapFiles[0].Name -ne 'CareMonitor.Bootstrap.exe') {throw '启动器未生成独立单文件，拒绝交付'}
New-Item -ItemType Directory -Path $destination | Out-Null
$executable=Join-Path $destination 'CareMonitor.exe'
Copy-Item -LiteralPath $bootstrapFiles[0].FullName -Destination $executable
if(-not $SkipZip){
    $stream=[IO.File]::Open($zip,[IO.FileMode]::CreateNew)
    $archive=[IO.Compression.ZipArchive]::new($stream,[IO.Compression.ZipArchiveMode]::Create)
    try {
        # The application already contains a compressed payload. Avoid compressing its large ZIP again.
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$executable,'CareMonitor.exe',[IO.Compression.CompressionLevel]::NoCompression) | Out-Null
    } finally {$archive.Dispose();$stream.Dispose()}
}
$buildReport=[pscustomobject]@{version='0.3.2';executable=$executable;executableBytes=(Get-Item -LiteralPath $executable).Length;executableSha256=(Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant();embeddedArchive=$archivePath;embeddedManifest=$manifestPath;archiveSha256=$archiveHash;resourceFiles=$entries.Count;resourceBytes=($files | Measure-Object Length -Sum).Sum;zip=if($SkipZip){$null}else{$zip}}
[IO.File]::WriteAllText((Join-Path (Split-Path $destination -Parent) ($name+'.build.json')),($buildReport | ConvertTo-Json -Depth 4),[Text.UTF8Encoding]::new($false))
Write-Output "独立单文件应用：$executable"
Write-Output "内嵌资源：$($entries.Count) 文件，首次运行自动释放"
if(-not $SkipZip){Write-Output "便携包（仅一个EXE）：$zip"}
