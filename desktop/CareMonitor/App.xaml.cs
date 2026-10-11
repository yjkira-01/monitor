using System.IO;
using System.Windows;

namespace CareMonitor;
public partial class App : Application
{
    private System.Threading.Mutex? instance;
    private void OnStartup(object sender, StartupEventArgs e)
    {
        string mutex=e.Args.Contains("--self-test")?"Local\\CareMonitor.NativeDesktop.SelfCheck":e.Args.Contains("--ui-smoke")?"Local\\CareMonitor.NativeDesktop.UiSmoke":"Local\\CareMonitor.NativeDesktop";
        instance = new(true, mutex, out bool first);
        if (!first) { MessageBox.Show("康养监测已在运行，请从任务栏打开。", "康养监测"); Shutdown(); return; }
        DispatcherUnhandledException += (_, args) =>
        {
            Services.AppLog.Write("unhandled-ui",args.Exception);
            if(e.Args.Contains("--ui-smoke"))
            {File.WriteAllText(Path.Combine(AppContext.BaseDirectory,"ui-validation/error.txt"),args.Exception.ToString());args.Handled=true;Shutdown(1);return;}
            MessageBox.Show("操作失败：" + args.Exception.Message, "康养监测", MessageBoxButton.OK, MessageBoxImage.Error);
            args.Handled = true;
        };
        TaskScheduler.UnobservedTaskException+=(_,args)=>Services.AppLog.Write("unobserved-task",args.Exception);
        if (e.Args.Contains("--self-test"))
        {
            try { SelfCheck.Run(e.Args); Shutdown(0); }
            catch (Exception ex) { File.WriteAllText(Path.Combine(AppContext.BaseDirectory, "self-test-error.txt"), ex.ToString()); Shutdown(1); }
            return;
        }
        try { new MainWindow().Show(); }
        catch(Exception ex)
        {
            if(e.Args.Contains("--ui-smoke"))File.WriteAllText(Path.Combine(AppContext.BaseDirectory,"ui-startup-error.txt"),ex.ToString());
            else MessageBox.Show("应用无法启动："+ex.Message,"康养监测",MessageBoxButton.OK,MessageBoxImage.Error);
            Shutdown(1);
        }
    }
}
