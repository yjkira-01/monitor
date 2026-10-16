param(
    [string]$Archive = "$PSScriptRoot\..\blurred_videos.zip",
    [string]$Output = "$PSScriptRoot\..\artifacts\research_execution_20261012\rgb_audit",
    [int[]]$Subjects = @(1),
    [int[]]$Views = @(0, 1)
)
$ErrorActionPreference = 'Stop'
$expectedHash = '1a95d3993d564ba028aa7f7f53c1a23ac35edd4f7196a951399529738873149e'
if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
    throw 'The video archive does not match the verified Dryad release.'
}
if ($Subjects.Count -eq 0 -or ($Subjects | Where-Object { $_ -lt 1 -or $_ -gt 20 }) -or
    $Views.Count -eq 0 -or ($Views | Where-Object { $_ -notin @(0, 1) })) { throw 'Invalid subject or view.' }
[void][System.IO.Directory]::CreateDirectory([System.IO.Path]::GetFullPath($Output))
# The release uses Deflate64, which Python's standard ZIP reader cannot decode.
$videoZip = [System.IO.Compression.ZipFile]::OpenRead([System.IO.Path]::GetFullPath($Archive))
try {
    foreach ($subject in $Subjects) {
        foreach ($view in $Views) {
            $basename = "subject${subject}_color${view}.mp4"
            $entries = @($videoZip.Entries | Where-Object { $_.Name -eq $basename })
            if ($entries.Count -ne 1) { throw "Expected one entry for $basename" }
            $entry = $entries[0]
            if ($entry.Length -le 0 -or $entry.Length -gt 1GB) { throw 'Unexpected video member size.' }
            $target = Join-Path $Output $basename
            if (Test-Path -LiteralPath $target) { throw "Output already exists: $target" }
            if (([System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($target)))).AvailableFreeSpace -lt $entry.Length + 1GB) {
                throw 'Insufficient space for selected video and a 1 GiB reserve.'
            }
            $partial = "$target.partial"
            $source = $entry.Open()
            try {
                $destination = [System.IO.File]::Open($partial, [System.IO.FileMode]::CreateNew)
                try { $source.CopyTo($destination) } finally { $destination.Dispose() }
            } finally { $source.Dispose() }
            if ((Get-Item -LiteralPath $partial).Length -ne $entry.Length) { throw 'Incomplete extraction.' }
            [System.IO.File]::Move($partial, $target)
            @{ video = $basename; bytes = $entry.Length } | ConvertTo-Json -Compress
        }
    }
} finally { $videoZip.Dispose() }
