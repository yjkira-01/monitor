using System.IO;
using System.IO.Ports;
using CareMonitor.Models;

namespace CareMonitor.Services;
public sealed class RadarService : IAsyncDisposable
{
    private CancellationTokenSource? cancellation;
    private Task? worker;
    private SerialPort? port;
    private readonly SemaphoreSlim lifecycle=new(1,1);
    private readonly RadarAnalytics analytics=new();
    private RadarZone[] zones=[];
    private readonly System.Diagnostics.Stopwatch clock=new();
    private long receivedBytes;private int frames,missed;private uint? lastFrame;
    private readonly HashSet<uint> seenTypes=[];
    public RadarObservation? Observation { get; private set; }
    public RadarDiagnostics Diagnostics { get; private set; }=new(0,0,0,0,[],[]);
    public void ConfigureRules(AppSettings settings)=>zones=settings.RadarZones.ToArray();
    public event Action<MonitorEvent>? Event;
    public RadarSnapshot? Latest { get; private set; }
    public string Status { get; private set; } = "未连接雷达";
    public event Action<VitalSample>? Vital;
    public async Task ConnectAsync(string name,int baud)
    {
        await lifecycle.WaitAsync();try{
        await StopCoreAsync();
        if(!SerialPort.GetPortNames().Contains(name) || baud<9600||baud>3000000) throw new ArgumentException("请选择有效数据串口与波特率。");
        port=new(name,baud,Parity.None,8,StopBits.One){ ReadTimeout=500,WriteTimeout=1000,ReadBufferSize=1024*1024 };
        try{port.Open();}catch{port.Dispose();port=null;throw;}
        cancellation=new();var token=cancellation.Token;var active=port;
        Status="已连接 "+name+" · 等待监测数据";
        worker=Task.Run(()=>
        {
            var parser=new RadarParser();byte[] bytes=new byte[16384];
            try
            {
                while(!token.IsCancellationRequested)
                {
                    int count;try{count=active.Read(bytes,0,bytes.Length);}catch(TimeoutException){continue;}
                    receivedBytes+=count;foreach(var frame in parser.Feed(bytes.AsSpan(0,count))) Accept(frame,name,false);
                }
            }
            catch(Exception ex) when(ex is IOException or InvalidOperationException or UnauthorizedAccessException)
            { if(!token.IsCancellationRequested){Status="雷达连接中断："+ex.Message;Latest=null;AppLog.Write("radar-read",ex);} }
        },token);
        }finally{lifecycle.Release();}
    }
    public async Task ReplayAsync(string path)
    {
        await lifecycle.WaitAsync();try{
        await StopCoreAsync();cancellation=new();var token=cancellation.Token;
        worker=Task.Run(async ()=>
        {
            try
            {
                using var stream=File.OpenRead(path);var parser=new RadarParser();byte[] bytes=new byte[4096];int count;
                while(!token.IsCancellationRequested && (count=await stream.ReadAsync(bytes,token))>0)
                {
                    receivedBytes+=count;
                    foreach(var frame in parser.Feed(bytes.AsSpan(0,count))) { Accept(frame,Path.GetFileName(path),true);await Task.Delay(90,token); }
                }
                if(!token.IsCancellationRequested) Status="数据回放结束";
            }
            catch(OperationCanceledException){}
            catch(Exception ex){Status="数据回放失败："+ex.Message;}
        },token);
        }finally{lifecycle.Release();}
    }
    private void Accept(RadarFrame frame,string source,bool replay)
    {
        Latest=new(DateTimeOffset.Now,source,replay,frame);Status=$"{(replay?"数据回放":"雷达采集")} · 帧 {frame.Number} · {frame.Points.Length} 点 / {frame.Tracks.Length} 轨迹";
        if(!clock.IsRunning)clock.Start();
        if(lastFrame.HasValue){uint delta=unchecked(frame.Number-lastFrame.Value);if(delta>1&&delta<1000)missed+=(int)delta-1;}
        lastFrame=frame.Number;frames++;foreach(uint type in frame.TlvTypes??[])seenTypes.Add(type);
        Diagnostics=new(receivedBytes,frames,missed,clock.Elapsed.TotalSeconds>0?frames/clock.Elapsed.TotalSeconds:0,seenTypes.Order().ToArray(),seenTypes.Where(t=>t is not (1 or 7 or 1010 or 1011 or 1012 or 1020 or 1021 or 1030 or 1040)).Order().ToArray());
        Observation=analytics.Update(Latest,replay?frames*.09:clock.Elapsed.TotalSeconds,zones,e=>Event?.Invoke(e));
        foreach(var vital in frame.Vitals)
        {
            var track=frame.Tracks.FirstOrDefault(t=>t.Id==vital.Id);
            if(track is null||vital.Deviation<=0)continue;
            // shortcut: 0.1 m/s motion threshold follows the existing provider and needs site calibration.
            Vital?.Invoke(new(){Target=vital.Id,HeartRate=vital.HeartRate>0?vital.HeartRate:null,
                BreathRate=vital.Deviation>=.02 && vital.BreathRate>0?vital.BreathRate:null,
                Moving=track.Velocity.Length()>=.1,IsReplay=replay,HeartWave=vital.HeartWave,BreathWave=vital.BreathWave});
        }
    }
    public static async Task ConfigureAsync(string controlPort,string configuration,CancellationToken token)
    {
        await RadarSetup.ConfigureAsync(controlPort,configuration,token);
    }
    public async Task StopAsync()
    {
        await lifecycle.WaitAsync();try{await StopCoreAsync();}finally{lifecycle.Release();}
    }
    private async Task StopCoreAsync()
    {
        cancellation?.Cancel();port?.Close();if(worker is not null)try{await worker;}catch(OperationCanceledException){}
        port?.Dispose();port=null;cancellation?.Dispose();cancellation=null;worker=null;Latest=null;Observation=null;Status="未连接雷达";
        analytics.Reset();clock.Reset();receivedBytes=0;frames=missed=0;lastFrame=null;seenTypes.Clear();Diagnostics=new(0,0,0,0,[],[]);
    }
    public async ValueTask DisposeAsync()=>await StopAsync();
}
