using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace CareMonitor.Bootstrap;

internal sealed record BootstrapOptions(string CacheRoot, string? ReportPath, bool ExtractOnly, bool Quiet, string[] ApplicationArguments)
{
    public static BootstrapOptions Parse(string[] args)
    {
        string root = Path.Combine(AppContext.BaseDirectory, "Runtime");
        string? report = null;
        bool extractOnly = false, quiet = false;
        var forwarded = new List<string>();
        for (int index = 0; index < args.Length; index++)
        {
            switch (args[index])
            {
                case "--bootstrap-cache": root = Value(args, ref index); break;
                case "--bootstrap-report": report = Value(args, ref index); break;
                case "--bootstrap-extract-only": extractOnly = true; break;
                case "--bootstrap-quiet": quiet = true; break;
                default: forwarded.Add(args[index]); break;
            }
        }
        return new(Path.GetFullPath(root), report is null ? null : Path.GetFullPath(report), extractOnly, quiet, forwarded.ToArray());
    }

    private static string Value(string[] args, ref int index)
    {
        if (++index >= args.Length || string.IsNullOrWhiteSpace(args[index]) || args[index].StartsWith("--bootstrap-", StringComparison.Ordinal))
            throw new ArgumentException("启动参数缺少文件夹或文件路径。");
        return args[index];
    }
}

internal static class Program
{
    [STAThread]
    private static int Main(string[] args)
    {
        BootstrapOptions? options = null;
        InstallReport? report = null;
        try
        {
            options = BootstrapOptions.Parse(args);
            using var metadata = Resource("CareMonitor.Payload.json");
            var manifest = PayloadManifest.Read(metadata);
            var installer = new PayloadInstaller(manifest, () => Resource("CareMonitor.Payload.zip"));
            InstalledPayload installed;
            if (options.Quiet) installed = installer.Install(options.CacheRoot);
            else
            {
                ApplicationConfiguration.Initialize();
                using var startup = new StartupForm(installer, options.CacheRoot);
                Application.Run(startup);
                if (startup.Cancelled) return 2;
                if (startup.Failure is not null) throw startup.Failure;
                installed = startup.Installed ?? throw new InvalidOperationException("运行资源尚未就绪。");
            }
            using (installed)
            {
                report = installed.Report;
                WriteReport(options, report, null, null);
                if (options.ExtractOnly) return 0;
                string executable = Path.Combine(installed.Directory, "CareMonitor.exe");
                var start = new ProcessStartInfo(executable) { WorkingDirectory = installed.Directory, UseShellExecute = false };
                foreach (string argument in options.ApplicationArguments) start.ArgumentList.Add(argument);
                using var child = Process.Start(start) ?? throw new IOException("康养监测进程未能启动。");
                // Resource streams are closed. Release the mapped ZIP's resident pages while waiting.
                GC.Collect();
                EmptyWorkingSet(GetCurrentProcess());
                child.WaitForExit();
                WriteReport(options, report, child.ExitCode, null);
                return child.ExitCode;
            }
        }
        catch (Exception exception)
        {
            if (options is not null)
            {
                try { WriteReport(options, report, 1, exception.ToString()); }
                catch { /* Keep the original installation error visible even if the report is unwritable. */ }
            }
            if (options?.Quiet != true)
            {
                string message = exception is UnauthorizedAccessException
                    ? "程序所在文件夹没有写入权限，请将 EXE 放到可以写入的文件夹后重新打开。"
                    : exception.Message;
                MessageBox.Show("康养监测无法启动。\n\n" + message, "青禾 · 康养监测", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            return 1;
        }
    }

    private static Stream Resource(string name) => Assembly.GetExecutingAssembly().GetManifestResourceStream(name)
        ?? throw new InvalidDataException("程序缺少内嵌资源，请重新获取完整发布文件。");

    private static void WriteReport(BootstrapOptions options, InstallReport? installation, int? exitCode, string? error)
    {
        string path = options.ReportPath ?? Path.Combine(options.CacheRoot, "bootstrap-last.json");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        PayloadInstaller.CheckNoReparsePoints(path);
        string temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            File.WriteAllText(temporary, JsonSerializer.Serialize(new { installed = installation is not null, installation, exitCode, error, recordedAt = DateTimeOffset.Now }, PayloadManifest.Json));
            File.Move(temporary, path, overwrite: true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    [DllImport("kernel32.dll")] private static extern IntPtr GetCurrentProcess();
    [DllImport("psapi.dll")] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool EmptyWorkingSet(IntPtr process);
}
