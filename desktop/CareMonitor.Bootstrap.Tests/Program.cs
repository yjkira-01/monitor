using CareMonitor.Bootstrap;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

string root = Path.GetFullPath(args.Length == 0 ? "artifacts/embedded_desktop_20261010/unit-" + Guid.NewGuid().ToString("N") : args[0]);
Directory.CreateDirectory(root);
var checks = new List<object>();
var contents = new Dictionary<string, byte[]> { ["CareMonitor.exe"] = Encoding.UTF8.GetBytes("test executable"), ["Assets/model.onnx"] = Enumerable.Range(0, 300000).Select(number => (byte)(number % 251)).ToArray(), ["说明.md"] = Encoding.UTF8.GetBytes("测试资源") };
byte[] zip = Archive(contents.Select(pair => (pair.Key, pair.Value)).ToArray());
var manifest = new PayloadManifest("0.2.1", Hash(zip), contents.Select(pair => new PayloadFile(pair.Key, pair.Value.LongLength, Hash(pair.Value))).ToArray());
var installer = new PayloadInstaller(manifest, () => new MemoryStream(zip, writable: false));
try
{
    string basic = Path.Combine(root, "basic");
    string installed;
    using (var first = installer.Install(basic))
    {
        installed = first.Directory;
        Assert(!first.Report.Reused && !first.Report.Repaired, "first extraction");
        Verify(first.Directory);
    }
    DateTime timestamp = File.GetLastWriteTimeUtc(Path.Combine(installed, "Assets/model.onnx"));
    using (var repeat = installer.Install(basic)) Assert(repeat.Report.Reused && File.GetLastWriteTimeUtc(Path.Combine(installed, "Assets/model.onnx")) == timestamp, "verified reuse without rewriting");

    File.Delete(Path.Combine(installed, "说明.md"));
    using (var repair = installer.Install(basic)) { Assert(repair.Report.Repaired, "missing file repair"); Verify(repair.Directory); }
    byte[] damaged = contents["Assets/model.onnx"].ToArray(); damaged[100] ^= 1;
    File.WriteAllBytes(Path.Combine(installed, "Assets/model.onnx"), damaged);
    File.SetLastWriteTimeUtc(Path.Combine(installed, "Assets/model.onnx"), timestamp);
    using (var repair = installer.Install(basic)) { Assert(repair.Report.Repaired, "same-length and same-time corruption repair"); Verify(repair.Directory); }
    File.WriteAllText(Path.Combine(installed, "unlisted.dll"), "unexpected native module");
    using (var repair = installer.Install(basic)) Assert(repair.Report.Repaired && !File.Exists(Path.Combine(installed, "unlisted.dll")), "unlisted executable rejection");
    File.Delete(Path.Combine(installed, ".bootstrap-ready"));
    using (var repair = installer.Install(basic)) Assert(repair.Report.Repaired, "incomplete commit recovery");

    using (var active = installer.Install(basic))
    {
        File.Delete(Path.Combine(installed, "说明.md"));
        Expect<IOException>(() => installer.Install(basic), "active lease prevents replacement");
        Assert(File.Exists(Path.Combine(installed, "CareMonitor.exe")), "active install remains intact");
    }
    using (var repair = installer.Install(basic)) Assert(repair.Report.Repaired, "repair after lease closes");

    string parallel = Path.Combine(root, "parallel");
    var tasks = Enumerable.Range(0, 4).Select(_ => Task.Run(() => installer.Install(parallel))).ToArray();
    Task.WaitAll(tasks);
    Assert(tasks.Count(task => !task.Result.Report.Reused) == 1 && tasks.Select(task => task.Result.Directory).Distinct().Count() == 1, "concurrent install extracts exactly once");
    foreach (var task in tasks) { Verify(task.Result.Directory); task.Result.Dispose(); }

    string cancelled = Path.Combine(root, "cancelled");
    using (var stop = new CancellationTokenSource())
    {
        Expect<OperationCanceledException>(() => installer.Install(cancelled, new InlineProgress(value => { if (value.Completed > 0) stop.Cancel(); }), stop.Token), "cancel midway");
        Assert(!Directory.EnumerateDirectories(cancelled).Any(), "cancel removes partial installation");
    }
    using (var restart = installer.Install(cancelled)) Assert(!restart.Report.Reused, "restart after cancellation");

    string abandoned = Path.Combine(root, "abandoned");
    string leftover = Path.Combine(abandoned, "." + manifest.CacheKey + ".stage.interrupted");
    Directory.CreateDirectory(leftover); File.WriteAllText(Path.Combine(leftover, "partial"), "not a complete payload");
    using (var recovered = installer.Install(abandoned)) Assert(!Directory.Exists(leftover), "abandoned stage cleanup");

    var next = new PayloadInstaller(manifest with { Version = "0.2.2" }, () => new MemoryStream(zip, false));
    using (var version = next.Install(basic)) Assert(version.Directory != installed && File.Exists(Path.Combine(installed, "CareMonitor.exe")), "version isolation");

    string sentinel = Path.Combine(root, "resident-data.json"); File.WriteAllText(sentinel, "user profile remains");
    using (var repeat = installer.Install(basic)) Assert(File.ReadAllText(sentinel) == "user profile remains", "user data untouched");

    foreach (string path in new[] { "../escape", "/absolute", "C:/absolute", "Assets\\file", "Assets//file", "A/../file", "CON.txt", "dir/NUL", "dir/COM1.dll", "dir/file. ", ".bootstrap-ready" })
        Expect<InvalidDataException>(() => PayloadManifest.ValidateRelativePath(path), "reject path: " + path);
    var duplicateManifest = manifest with { Files = [new("CareMonitor.exe", 0, Hash([])), new("CAREMONITOR.EXE", 0, Hash([]))] };
    Expect<InvalidDataException>(duplicateManifest.Validate, "case-insensitive manifest duplicates");
    Expect<InvalidDataException>(() => PayloadInstaller.ChildPath(root, "../outside"), "path containment");

    var wrongArchive = manifest with { ArchiveSha256 = new string('a', 64) };
    Expect<InvalidDataException>(() => new PayloadInstaller(wrongArchive, () => new MemoryStream(zip, false)).Install(Path.Combine(root, "wrong-archive")), "archive checksum failure");
    var badContents = contents.ToDictionary(pair => pair.Key, pair => pair.Value); badContents["CareMonitor.exe"] = Encoding.UTF8.GetBytes("fake executable");
    byte[] badZip = Archive(badContents.Select(pair => (pair.Key, pair.Value)).ToArray());
    var badManifest = manifest with { ArchiveSha256 = Hash(badZip) };
    Expect<InvalidDataException>(() => new PayloadInstaller(badManifest, () => new MemoryStream(badZip, false)).Install(Path.Combine(root, "wrong-file")), "individual file checksum failure");
    byte[] unsafeZip = Archive([("CareMonitor.exe", contents["CareMonitor.exe"]), ("../escaped.onnx", contents["Assets/model.onnx"]), ("说明.md", contents["说明.md"])]);
    Expect<InvalidDataException>(() => new PayloadInstaller(manifest with { ArchiveSha256 = Hash(unsafeZip) }, () => new MemoryStream(unsafeZip, false)).Install(Path.Combine(root, "unsafe-zip")), "zip traversal rejection");
    Assert(!File.Exists(Path.Combine(root, "escaped.onnx")), "zip escape never written");
    byte[] duplicateZip = Archive([("CareMonitor.exe", contents["CareMonitor.exe"]), ("CAREMONITOR.EXE", contents["CareMonitor.exe"]), ("说明.md", contents["说明.md"])]);
    Expect<InvalidDataException>(() => new PayloadInstaller(manifest with { ArchiveSha256 = Hash(duplicateZip) }, () => new MemoryStream(duplicateZip, false)).Install(Path.Combine(root, "duplicate-zip")), "zip duplicate rejection");
    using var json = new MemoryStream(JsonSerializer.SerializeToUtf8Bytes(manifest, PayloadManifest.Json));
    Assert(PayloadManifest.Read(json).CacheKey == manifest.CacheKey, "embedded manifest JSON roundtrip");
    File.WriteAllText(Path.Combine(root, "result.json"), JsonSerializer.Serialize(new { passed = true, checks }, PayloadManifest.Json));
    Console.WriteLine($"PASS {checks.Count} checks; {root}");
    return 0;
}
catch (Exception exception)
{
    File.WriteAllText(Path.Combine(root, "result.json"), JsonSerializer.Serialize(new { passed = false, checks, error = exception.ToString() }, PayloadManifest.Json));
    Console.Error.WriteLine(exception);
    return 1;
}

void Assert(bool condition, string name)
{
    if (!condition) throw new InvalidOperationException("FAIL " + name);
    checks.Add(new { name, passed = true });
}
void Expect<T>(Action action, string name) where T : Exception
{
    try { action(); } catch (T) { Assert(true, name); return; }
    throw new InvalidOperationException("Expected " + typeof(T).Name + ": " + name);
}
void Verify(string directory)
{
    foreach (var file in manifest.Files)
        if (Hash(File.ReadAllBytes(PayloadInstaller.ChildPath(directory, file.Path))) != file.Sha256) throw new InvalidDataException("Extracted bytes differ");
}
static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
static byte[] Archive((string Path, byte[] Bytes)[] files)
{
    using var stream = new MemoryStream();
    using (var archive = new ZipArchive(stream, ZipArchiveMode.Create, true))
        foreach (var file in files) { using var output = archive.CreateEntry(file.Path).Open(); output.Write(file.Bytes); }
    return stream.ToArray();
}
sealed class InlineProgress(Action<InstallProgress> callback) : IProgress<InstallProgress>
{
    public void Report(InstallProgress value) => callback(value);
}
