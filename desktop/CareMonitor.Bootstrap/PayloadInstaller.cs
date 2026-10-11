using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace CareMonitor.Bootstrap;

internal sealed record PayloadFile(string Path, long Length, string Sha256);
internal sealed record PayloadManifest(string Version, string ArchiveSha256, PayloadFile[] Files)
{
    public static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true };
    public string CacheKey => Version + "-" + ArchiveSha256[..16];
    public long TotalBytes => Files.Sum(file => file.Length);

    public static PayloadManifest Read(Stream source)
    {
        var manifest = JsonSerializer.Deserialize<PayloadManifest>(source, Json) ?? throw new InvalidDataException("内嵌资源清单为空。");
        manifest.Validate();
        return manifest;
    }

    public void Validate()
    {
        if (!System.Version.TryParse(Version, out _) || Version.Length > 40 || !IsHash(ArchiveSha256))
            throw new InvalidDataException("内嵌资源版本或校验码无效。");
        if (Files is null || Files.Length == 0 || Files.Length > 10000)
            throw new InvalidDataException("内嵌资源数量无效。");
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var file in Files)
        {
            ValidateRelativePath(file.Path);
            if (!paths.Add(file.Path) || file.Length < 0 || !IsHash(file.Sha256))
                throw new InvalidDataException("内嵌资源清单有重复或无效项目。");
        }
        if (!paths.Contains("CareMonitor.exe")) throw new InvalidDataException("内嵌资源缺少应用程序。");
        foreach (string path in paths)
        {
            for (int slash = path.IndexOf('/'); slash >= 0; slash = path.IndexOf('/', slash + 1))
                if (paths.Contains(path[..slash])) throw new InvalidDataException("资源文件与文件夹重名。");
        }
        _ = TotalBytes;
    }

    private static bool IsHash(string? value) => value is { Length: 64 } && value.All(Uri.IsHexDigit);

    public static void ValidateRelativePath(string? path)
    {
        if (string.IsNullOrEmpty(path) || path.Length > 1000 || path.Contains('\\') || path.StartsWith('/') || path.Contains(':'))
            throw new InvalidDataException("资源路径无效。");
        foreach (string part in path.Split('/'))
        {
            string device = part.Split('.')[0];
            if (part.Length == 0 || part is "." or ".." || part.EndsWith('.') || part.EndsWith(' ') ||
                part.Any(character => character < 32 || "<>\"|?*".Contains(character)) ||
                device.Equals("CON", StringComparison.OrdinalIgnoreCase) || device.Equals("PRN", StringComparison.OrdinalIgnoreCase) ||
                device.Equals("AUX", StringComparison.OrdinalIgnoreCase) || device.Equals("NUL", StringComparison.OrdinalIgnoreCase) ||
                System.Text.RegularExpressions.Regex.IsMatch(device, "^(COM|LPT)[0-9¹²³]$", System.Text.RegularExpressions.RegexOptions.IgnoreCase) ||
                part.StartsWith(".bootstrap-", StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("资源路径包含保留名称。");
        }
    }
}

internal sealed record InstallProgress(string Message, long Completed, long Total);
internal sealed record InstallReport(string Version, string ArchiveSha256, string Directory, bool Reused, bool Repaired, int FileCount, long TotalBytes, double ElapsedMs);
internal sealed class InstalledPayload(string directory, InstallReport report, FileStream lease) : IDisposable
{
    public string Directory { get; } = directory;
    public InstallReport Report { get; } = report;
    public void Dispose() => lease.Dispose();
}

internal sealed class PayloadInstaller(PayloadManifest manifest, Func<Stream> openArchive)
{
    private const string MarkerName = ".bootstrap-ready";
    private const string LeaseName = ".bootstrap-in-use";

    // All installation work stays on one thread: a Windows mutex must be released by its owner.
    public InstalledPayload Install(string cacheRoot, IProgress<InstallProgress>? progress = null, CancellationToken cancellation = default)
    {
        manifest.Validate();
        string root = Path.GetFullPath(cacheRoot);
        CheckNoReparsePoints(root);
        Directory.CreateDirectory(root);
        string target = ChildPath(root, manifest.CacheKey);
        string mutexHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(target.ToUpperInvariant())));
        using var mutex = new Mutex(false, "Local\\CareMonitor.Resources." + mutexHash);
        bool ownsMutex = false;
        string? staging = null;
        var watch = Stopwatch.StartNew();
        try
        {
            progress?.Report(new("正在等待资源准备…", 0, manifest.TotalBytes));
            while (!ownsMutex)
            {
                cancellation.ThrowIfCancellationRequested();
                try { ownsMutex = mutex.WaitOne(100); }
                catch (AbandonedMutexException) { ownsMutex = true; }
            }
            CheckNoReparsePoints(root);
            CleanupAbandonedStages(root);
            bool existed = Directory.Exists(target);
            if (CacheIsValid(target, progress, cancellation))
                return Result(target, true, false, watch);

            // Never replace resource files while another launcher is using this version.
            if (existed)
            {
                CheckNoReparsePoints(target);
                try { using var exclusive = OpenLease(target, FileShare.None); }
                catch (IOException exception) { throw new IOException("本版本正在运行，资源暂时无法修复。请关闭康养监测后重新打开。", exception); }
            }
            cancellation.ThrowIfCancellationRequested();
            CheckFreeSpace(root);
            staging = ChildPath(root, "." + manifest.CacheKey + ".stage." + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(staging);
            Extract(staging, progress, cancellation);
            cancellation.ThrowIfCancellationRequested();
            using (var marker = new FileStream(Path.Combine(staging, MarkerName), FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                marker.Write(Encoding.UTF8.GetBytes(manifest.ArchiveSha256));
                marker.Flush(true);
            }
            string? previous = null;
            if (existed)
            {
                previous = ChildPath(root, "." + manifest.CacheKey + ".old." + Guid.NewGuid().ToString("N"));
                Directory.Move(target, previous);
            }
            try { Directory.Move(staging, target); staging = null; }
            catch
            {
                if (previous is not null && !Directory.Exists(target)) Directory.Move(previous, target);
                throw;
            }
            if (previous is not null) TryDeleteOwnedDirectory(root, previous);
            return Result(target, false, existed, watch);
        }
        finally
        {
            if (staging is not null) TryDeleteOwnedDirectory(root, staging);
            if (ownsMutex) mutex.ReleaseMutex();
        }
    }

    private InstalledPayload Result(string target, bool reused, bool repaired, Stopwatch watch)
    {
        CheckNoReparsePoints(target);
        var lease = OpenLease(target, FileShare.ReadWrite);
        var report = new InstallReport(manifest.Version, manifest.ArchiveSha256, target, reused, repaired, manifest.Files.Length, manifest.TotalBytes, watch.Elapsed.TotalMilliseconds);
        return new(target, report, lease);
    }

    private static FileStream OpenLease(string target, FileShare sharing)
    {
        string path = Path.Combine(target, LeaseName);
        CheckNoReparsePoints(path);
        return new(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, sharing);
    }

    private bool CacheIsValid(string target, IProgress<InstallProgress>? progress, CancellationToken cancellation)
    {
        if (!Directory.Exists(target)) return false;
        CheckNoReparsePoints(target);
        string marker = Path.Combine(target, MarkerName);
        CheckNoReparsePoints(marker);
        if (!File.Exists(marker) || new FileInfo(marker).Length != 64 || File.ReadAllText(marker) != manifest.ArchiveSha256) return false;
        long verified = 0;
        foreach (var entry in manifest.Files)
        {
            cancellation.ThrowIfCancellationRequested();
            string file = ChildPath(target, entry.Path);
            CheckNoReparsePoints(file);
            if (!File.Exists(file) || new FileInfo(file).Length != entry.Length) return false;
            using var stream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read, 131072, FileOptions.SequentialScan);
            string hash = HashStream(stream, cancellation);
            if (!hash.Equals(entry.Sha256, StringComparison.OrdinalIgnoreCase)) return false;
            verified += entry.Length;
            progress?.Report(new("正在检查运行资源…", verified, manifest.TotalBytes));
        }
        // Reject unexpected executables/libraries rather than let native probing load unverified code.
        var expected = manifest.Files.Select(file => file.Path).ToHashSet(StringComparer.OrdinalIgnoreCase);
        foreach (var file in SafeFiles(target))
        {
            cancellation.ThrowIfCancellationRequested();
            string relative = Path.GetRelativePath(target, file).Replace('\\', '/');
            string extension = Path.GetExtension(file);
            if ((extension.Equals(".dll", StringComparison.OrdinalIgnoreCase) || extension.Equals(".exe", StringComparison.OrdinalIgnoreCase) || extension.Equals(".onnx", StringComparison.OrdinalIgnoreCase)) && !expected.Contains(relative))
                return false;
        }
        return true;
    }

    private void Extract(string destination, IProgress<InstallProgress>? progress, CancellationToken cancellation)
    {
        using var source = openArchive();
        if (!source.CanSeek) throw new InvalidDataException("内嵌资源流不可定位。");
        progress?.Report(new("正在校验内嵌资源…", 0, manifest.TotalBytes));
        if (!HashStream(source, cancellation).Equals(manifest.ArchiveSha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("程序内嵌资源校验失败，请重新获取完整安装文件。");
        source.Position = 0;
        using var zip = new ZipArchive(source, ZipArchiveMode.Read, leaveOpen: true);
        var files = manifest.Files.ToDictionary(file => file.Path, StringComparer.OrdinalIgnoreCase);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (zip.Entries.Count != manifest.Files.Length) throw new InvalidDataException("内嵌资源数量与清单不一致。");
        long completed = 0;
        byte[] buffer = new byte[131072];
        foreach (var entry in zip.Entries)
        {
            cancellation.ThrowIfCancellationRequested();
            PayloadManifest.ValidateRelativePath(entry.FullName);
            // A payload must contain regular files only, with a one-to-one match to the embedded manifest.
            if (!seen.Add(entry.FullName) || !files.TryGetValue(entry.FullName, out var expected) || entry.Length != expected.Length ||
                ((entry.ExternalAttributes >> 16) & 0xF000) is 0xA000 or 0x4000)
                throw new InvalidDataException("内嵌资源项目与清单不一致。");
            string path = ChildPath(destination, expected.Path);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            CheckNoReparsePoints(path);
            using var input = entry.Open();
            using var output = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None, buffer.Length, FileOptions.SequentialScan);
            using var digest = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            long count = 0;
            int read;
            while ((read = input.Read(buffer, 0, buffer.Length)) != 0)
            {
                cancellation.ThrowIfCancellationRequested();
                count = checked(count + read);
                if (count > expected.Length) throw new InvalidDataException("内嵌资源长度异常。");
                output.Write(buffer, 0, read);
                digest.AppendData(buffer, 0, read);
                completed += read;
                progress?.Report(new("首次启动，正在准备运行资源…", completed, manifest.TotalBytes));
            }
            if (count != expected.Length || !Convert.ToHexString(digest.GetHashAndReset()).Equals(expected.Sha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("内嵌资源文件校验失败：" + expected.Path);
        }
    }

    private void CheckFreeSpace(string root)
    {
        var drive = new DriveInfo(Path.GetPathRoot(root)!);
        if (drive.IsReady && drive.AvailableFreeSpace < manifest.TotalBytes + 64L * 1024 * 1024)
            throw new IOException($"运行资源需要约 {Math.Ceiling(manifest.TotalBytes / 1048576d)} MB 空间，请清理缓存所在磁盘后再启动。");
    }

    internal static string HashStream(Stream stream, CancellationToken cancellation = default)
    {
        using var digest = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] buffer = new byte[131072];
        int count;
        while ((count = stream.Read(buffer, 0, buffer.Length)) != 0)
        {
            cancellation.ThrowIfCancellationRequested();
            digest.AppendData(buffer, 0, count);
        }
        return Convert.ToHexString(digest.GetHashAndReset()).ToLowerInvariant();
    }

    private void CleanupAbandonedStages(string root)
    {
        foreach (string directory in Directory.EnumerateDirectories(root, "." + manifest.CacheKey + ".stage.*", SearchOption.TopDirectoryOnly))
            TryDeleteOwnedDirectory(root, directory);
    }

    private void TryDeleteOwnedDirectory(string root, string path)
    {
        string full = Path.GetFullPath(path);
        string name = Path.GetFileName(full);
        if (!string.Equals(Path.GetDirectoryName(full), root.TrimEnd(Path.DirectorySeparatorChar), StringComparison.OrdinalIgnoreCase) ||
            !(name.StartsWith("." + manifest.CacheKey + ".stage.", StringComparison.Ordinal) || name.StartsWith("." + manifest.CacheKey + ".old.", StringComparison.Ordinal)))
            throw new InvalidDataException("拒绝清理资源缓存以外的目录。");
        try
        {
            if (!Directory.Exists(full)) return;
            CheckNoReparsePoints(full);
            _ = SafeFiles(full).ToArray();
            Directory.Delete(full, recursive: true);
        }
        catch (IOException) { /* A locked old cache can be removed after the user closes its process. */ }
        catch (UnauthorizedAccessException) { }
    }

    private static IEnumerable<string> SafeFiles(string directory)
    {
        foreach (string path in Directory.EnumerateFileSystemEntries(directory))
        {
            var attributes = File.GetAttributes(path);
            if (attributes.HasFlag(FileAttributes.ReparsePoint)) throw new IOException("资源缓存包含目录链接，请使用普通本地文件夹。");
            if (attributes.HasFlag(FileAttributes.Directory))
                foreach (string file in SafeFiles(path)) yield return file;
            else yield return path;
        }
    }

    internal static string ChildPath(string root, string relative)
    {
        string fullRoot = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
        string full = Path.GetFullPath(Path.Combine(fullRoot, relative.Replace('/', Path.DirectorySeparatorChar)));
        if (!full.StartsWith(fullRoot, StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("资源路径超出目标目录。");
        return full;
    }

    internal static void CheckNoReparsePoints(string path)
    {
        string? current = Path.GetFullPath(path);
        while (!string.IsNullOrEmpty(current))
        {
            try
            {
                if (File.GetAttributes(current).HasFlag(FileAttributes.ReparsePoint))
                    throw new IOException("资源路径包含目录链接，请使用普通本地文件夹。");
            }
            catch (FileNotFoundException) { }
            catch (DirectoryNotFoundException) { }
            current = Path.GetDirectoryName(current);
        }
    }
}
