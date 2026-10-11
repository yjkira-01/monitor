using System.IO;
namespace CareMonitor.Services;
public static class AppLog
{
    private static readonly object gate=new();
    public static void Write(string operation,Exception error)
    {
        // Do not log source URLs, credentials, or private observations.
        try{lock(gate){string directory=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"CareMonitor","logs");Directory.CreateDirectory(directory);
            string path=Path.Combine(directory,$"diagnostic-{DateTime.Now:yyyyMMdd}.log");
            File.AppendAllText(path,$"{DateTimeOffset.Now:O} {operation} {error.GetType().Name}\n{error.StackTrace}\n");}}
        catch(IOException){}catch(UnauthorizedAccessException){}
    }
}
