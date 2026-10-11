using System.Diagnostics;

namespace CareMonitor.Bootstrap;

internal sealed class StartupForm : Form, IProgress<InstallProgress>
{
    private readonly CancellationTokenSource cancellation = new();
    private readonly Label status;
    private readonly Label detail;
    private readonly ProgressBar progress;
    private readonly Button cancel;
    private readonly System.Windows.Forms.Timer reveal;
    private readonly Stopwatch throttle = Stopwatch.StartNew();
    private bool finished;
    public InstalledPayload? Installed { get; private set; }
    public Exception? Failure { get; private set; }
    public bool Cancelled { get; private set; }

    public StartupForm(PayloadInstaller installer, string cacheRoot)
    {
        Text = "青禾 · 康养监测";
        ClientSize = new Size(480, 235);
        BackColor = Color.FromArgb(246, 249, 246);
        Font = new Font("Microsoft YaHei UI", 10);
        FormBorderStyle = FormBorderStyle.FixedSingle;
        StartPosition = FormStartPosition.CenterScreen;
        MaximizeBox = false;
        MinimizeBox = false;
        Opacity = 0;
        ShowInTaskbar = false;
        if (Environment.ProcessPath is { } path) Icon = Icon.ExtractAssociatedIcon(path);
        Controls.Add(new Label { Text = "青禾  /  康养监测", Location = new(28, 24), Size = new(420, 35), Font = new Font(Font.FontFamily, 17, FontStyle.Bold), ForeColor = Color.FromArgb(31, 91, 66) });
        status = new Label { Text = "正在准备运行资源…", Location = new(30, 79), Size = new(420, 26), ForeColor = Color.FromArgb(37, 54, 44) };
        progress = new ProgressBar { Location = new(30, 114), Size = new(420, 8), Maximum = 1000, Style = ProgressBarStyle.Marquee, MarqueeAnimationSpeed = 20 };
        detail = new Label { Text = "首次启动会自动释放资源，完成后打开应用。", Location = new(30, 137), Size = new(420, 30), ForeColor = Color.FromArgb(93, 111, 98), Font = new Font(Font.FontFamily, 9) };
        cancel = new Button { Text = "取消", Location = new(372, 184), Size = new(78, 29), FlatStyle = FlatStyle.Flat, BackColor = Color.White };
        cancel.FlatAppearance.BorderColor = Color.FromArgb(208, 219, 210);
        cancel.Click += (_, _) => RequestCancel();
        Controls.AddRange([status, progress, detail, cancel]);
        reveal = new() { Interval = 500 };
        reveal.Tick += (_, _) => { reveal.Stop(); if (!finished) { Opacity = 1; ShowInTaskbar = true; } };
        FormClosing += (_, e) => { if (!finished) { RequestCancel(); e.Cancel = true; } };
        Shown += async (_, _) =>
        {
            reveal.Start();
            try { Installed = await Task.Run(() => installer.Install(cacheRoot, this, cancellation.Token)); }
            catch (OperationCanceledException) { Cancelled = true; }
            catch (Exception exception) { Failure = exception; }
            finally { finished = true; reveal.Stop(); Close(); }
        };
    }

    private void RequestCancel()
    {
        cancellation.Cancel();
        status.Text = "正在取消…";
        cancel.Enabled = false;
    }

    public void Report(InstallProgress value)
    {
        if (throttle.ElapsedMilliseconds < 80) return;
        throttle.Restart();
        if (IsDisposed || !IsHandleCreated) return;
        BeginInvoke((Action)(() =>
        {
            if (finished || cancellation.IsCancellationRequested) return;
            status.Text = value.Message;
            if (value.Completed > 0 && value.Total > 0)
            {
                progress.Style = ProgressBarStyle.Continuous;
                progress.Value = Math.Clamp((int)(1000d * value.Completed / value.Total), 0, 1000);
                detail.Text = $"{100d * value.Completed / value.Total:0}%   ·   {value.Completed / 1048576d:0} / {value.Total / 1048576d:0} MB";
            }
            else { progress.Style = ProgressBarStyle.Marquee; detail.Text = "完成后自动打开应用，请稍候。"; }
        }));
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) { reveal.Dispose(); cancellation.Dispose(); }
        base.Dispose(disposing);
    }
}
