using System.Globalization;
using System.IO;
using System.IO.Ports;
using System.Diagnostics;
using System.Text.RegularExpressions;
using CareMonitor.Models;
namespace CareMonitor.Services;
public sealed record RadarFirmware(string Version,string Help)
{
    public bool Supported=>Version.Contains("IWR68XX",StringComparison.OrdinalIgnoreCase)&&Version.Contains("AOP",StringComparison.OrdinalIgnoreCase)&&Version.Contains("03.05.00.04")&&Help.Contains("vitalsign:")&&Help.Contains("VSRangeIdxCfg:");
}
public static class RadarSetup
{
    public static string Configuration(string assets,AppSettings settings)
    {
        if(!double.IsFinite(settings.RadarHeight)||settings.RadarHeight is <.1 or >4||!double.IsFinite(settings.RadarAzimuth)||Math.Abs(settings.RadarAzimuth)>90||!double.IsFinite(settings.RadarElevation)||Math.Abs(settings.RadarElevation)>90)
            throw new ArgumentException("雷达安装高度须在 0.1–4 米，方向角须在 -90–90 度。");
        string cfg=File.ReadAllText(Path.Combine(assets,"Radar/sr112-aop-vitals.cfg"));
        string position=FormattableString.Invariant($"sensorPosition {settings.RadarHeight} {settings.RadarAzimuth} {settings.RadarElevation}");
        return Regex.Replace(cfg,@"(?m)^sensorPosition[^\r\n]*",position);
    }
    public static async Task<RadarFirmware> IdentifyAsync(string control,CancellationToken token)
    {
        using var port=Open(control);await Task.Delay(100,token);port.DiscardInBuffer();
        string version=await CommandAsync(port,"version",token);
        string help=await CommandAsync(port,"help",token);
        return new(version,help);
    }
    internal static SerialPort Open(string name)
    {var port=new SerialPort(name,115200){ReadTimeout=1000,WriteTimeout=1000};try{port.Open();return port;}catch{port.Dispose();throw;}}
    internal static async Task<string> CommandAsync(SerialPort port,string command,CancellationToken token)
    {
        token.ThrowIfCancellationRequested();port.DiscardInBuffer();port.Write(command+"\n");
        string response="";var watch=Stopwatch.StartNew();
        while(watch.Elapsed<TimeSpan.FromSeconds(command=="sensorStart"?15:4))
        {
            await Task.Delay(50,token);response+=port.ReadExisting();
            if(response.Contains("Error",StringComparison.OrdinalIgnoreCase)||response.Contains("not recognized",StringComparison.OrdinalIgnoreCase))throw new IOException("设备拒绝命令 "+command.Split(' ')[0]+"："+response);
            if(response.Contains("Done"))return response;
        }
        throw new IOException("设备未确认命令 "+command.Split(' ')[0]+"。请检查控制串口和固件。");
    }
    public static async Task<int> ConfigureAsync(string control,string cfg,CancellationToken token,Action<int,string>? progress=null)
    {
        if(string.IsNullOrWhiteSpace(cfg))throw new ArgumentException("配置文件为空。");
        using var port=Open(control);int count=0;
        try
        {
            foreach(string line in cfg.Split('\n').Select(l=>l.Trim()).Where(l=>l.Length>0&&!l.StartsWith('%')&&!l.StartsWith('#')))
            {await CommandAsync(port,line,token);count++;progress?.Invoke(count,line.Split(' ')[0]);await Task.Delay(line=="sensorStop"?400:100,token);}
        }
        catch{try{port.Write("sensorStop\n");}catch(IOException){}throw;}
        return count;
    }
}
