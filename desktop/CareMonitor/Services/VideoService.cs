using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using CareMonitor.Models;
using OpenCvSharp;
using ByteTrack;

namespace CareMonitor.Services;
public sealed class VideoService : IAsyncDisposable
{
    [DllImport("winmm.dll")] private static extern uint timeBeginPeriod(uint milliseconds);
    [DllImport("winmm.dll")] private static extern uint timeEndPeriod(uint milliseconds);
    private readonly object sync=new();
    private readonly SemaphoreSlim lifecycle=new(1,1);
    private static readonly object ffmpegOpenGate=new();
    private static readonly object cameraOpenGate=new();
    private readonly VideoService? modelOwner;
    private readonly bool observationOnly;
    private TimedObservation? observation;
    public TimedObservation? Observation { get {lock(sync)return observation;} }
    public bool StereoCapable=>models?.OneStage==true;
    private readonly string assets;
    private readonly bool useGpu;
    private readonly bool preview;
    private readonly string fallModel;
    private Task<NativeModels>? initialization;
    private NativeModels? models;
    private CancellationTokenSource? cancellation;
    private Task[] workers=[];
    private VideoSnapshot? latest;
    private AnalysisSnapshot? analysis;
    private long generation,sequence;
    private volatile bool paused;
    private double requestedSeek=-1;
    private RgbJob? pending;
    private readonly Queue<(long Generation,double Time,byte[] Frame)> capturedScene=[];
    public bool ReplayAnalysisComplete {get{lock(sync)return latest?.Ended==true&&latest.Replay&&(observationOnly||latest.Time<2||analysis is { } a&&a.RgbTime>=latest.Time-.001);}}
    private readonly Dictionary<int,(float Risk,string Activity,float Confidence,int Consecutive,double Time,bool WholeScene,float? ShortRisk,int ShortConsecutive)> rgbResults=[];
    private readonly Dictionary<int,(string State,double Time)> lastAlerts=[];
    private readonly MonitoringAnalytics analytics=new();
    private sealed record RuleConfig(MonitoringZone[] Zones,int Inactivity);
    private volatile RuleConfig rules=new([],120);
    public void ConfigureRules(AppSettings settings)
    {
        lock(sync)
        {
            bool changed=!rules.Zones.SequenceEqual(settings.Zones);rules=new(settings.Zones.ToArray(),settings.InactivitySeconds);
            if(changed)
            {
                analytics.InvalidateZones();
                if(analysis is not null)analysis=analysis with{People=analysis.People.Select(p=>p.Telemetry is null?p:p with{Telemetry=p.Telemetry with{Zone="区域外",DwellSeconds=0}}).ToArray()};
            }
        }
    }
    public string ModelStatus { get; private set; } = "正在初始化分析服务…";
    public string CaptureStatus { get; private set; } = "请选择视频文件或连接摄像头";
    public event Action<MonitorEvent>? Event;
    public VideoSnapshot? Latest { get { lock(sync)return latest; } }
    public AnalysisSnapshot? Analysis { get { lock(sync)return analysis; } }
    private sealed record RgbJob(long Generation,double Time,List<byte[]> Scene,List<byte[]> ShortScene,(int Id,List<byte[]> Frames,List<byte[]> ShortFrames,bool Partial,bool ShortEligible)[] People,int? ScenePersonId);
    private sealed class Track
    {
        public int Id;public Box Box;public float Score;public double Seen;
        public Joint[] Image=[];public Joint[] World=[];
        public readonly List<(double Time,Joint[] Pose)> Poses=[];
        public readonly List<(double Time,byte[] Frame)> Frames=[];
        public float? PoseRisk;public double LastPose=-1;public string Detector="SSD";
        public string Posture="unknown";public float PostureConfidence;public ActionCandidate[] Candidates=[];
        public readonly PostureTimeline Timeline=new();
    }
    public VideoService(string assets,bool useGpu,bool preview=false,string fallModel="gmd",VideoService? modelOwner=null)
    {this.assets=assets;this.useGpu=useGpu;this.preview=preview;this.fallModel=fallModel;this.modelOwner=modelOwner;observationOnly=modelOwner is not null;}
    public Task InitializeAsync()
    {
        lock(sync) initialization ??= Task.Run(async ()=>
        {
            try
            {
                if(modelOwner is not null)
                {await modelOwner.InitializeAsync();models=modelOwner.models!;if(!models.OneStage)throw new InvalidOperationException("双目监测需要在系统设置选择 RTMO 实时方案并重启应用。");ModelStatus=modelOwner.ModelStatus;return models;}
                models=new(assets,useGpu,preview,fallModel);ModelStatus="正在预热人员识别与视频分析…";
                using var empty=new Mat(480,640,MatType.CV_8UC3,Scalar.All(0));
                models.ObservePeople(empty);var prepared=models.PrepareVideoFrame(empty);models.PredictVideo(Enumerable.Repeat(prepared,6).ToArray());
                ModelStatus=models.DetectorName+" · "+models.VideoName+" · "+(models.VideoGpuEnabled?"硬件加速已启用":"使用 CPU 分析");return models;
            }
            catch(Exception ex){ModelStatus="分析服务不可用："+ex.Message;throw;}
        });
        return initialization;
    }
    public async Task StartFileAsync(string path)
    {
        if(!File.Exists(path))throw new FileNotFoundException("找不到视频文件",path);
        await StartAsync(path,null);
    }
    public async Task StartCameraAsync(AppSettings settings)=>await StartAsync(null,settings);
    public async Task StartNetworkAsync(string address)
    {
        if(!Uri.TryCreate(address,UriKind.Absolute,out var uri)||uri.Scheme is not ("rtsp" or "http" or "https"))throw new ArgumentException("请输入 RTSP、HTTP 或 HTTPS 视频流地址。");
        await StartAsync(address,null,true);
    }
    private async Task StartAsync(string? path,AppSettings? settings,bool network=false)
    {
        await InitializeAsync();
        await lifecycle.WaitAsync();try
        {
        await StopCoreAsync();cancellation=new();var token=cancellation.Token;long session=Interlocked.Increment(ref generation);
        paused=false;requestedSeek=-1;
        CaptureStatus="正在连接视频源";
        lock(sync){latest=null;analysis=null;observation=null;pending=null;capturedScene.Clear();rgbResults.Clear();lastAlerts.Clear();}
        var ready=new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var captureWorker=Task.Factory.StartNew(()=>Capture(path,settings,session,token,network,ready),token,TaskCreationOptions.LongRunning,TaskScheduler.Default).Unwrap();
        workers=observationOnly?[captureWorker,Task.Run(()=>Observe(session,token),token)]:
            [captureWorker,Task.Run(()=>Observe(session,token),token),Task.Run(()=>Classify(session,token),token)];
        try{await ready.Task.WaitAsync(TimeSpan.FromSeconds(15),token);}catch{await StopCoreAsync();throw;}
        }finally{lifecycle.Release();}
    }
    private async Task Capture(string? path,AppSettings? settings,long session,CancellationToken token,bool network,TaskCompletionSource ready)
    {
        using var capture=new VideoCapture();
        bool preciseTimer=false,exposureLimited=false;int slowExposureSamples=0;
        try
        {
            if(path is not null)
            {
                lock(ffmpegOpenGate)
                {
                string? previous=Environment.GetEnvironmentVariable("OPENCV_FFMPEG_CAPTURE_OPTIONS");
                try
                {
                    if(network)Environment.SetEnvironmentVariable("OPENCV_FFMPEG_CAPTURE_OPTIONS","rtsp_transport;tcp|timeout;3000000|rw_timeout;3000000");
                    capture.Open(path,VideoCaptureAPIs.FFMPEG);
                }
                finally{if(network)Environment.SetEnvironmentVariable("OPENCV_FFMPEG_CAPTURE_OPTIONS",previous);}
                }
            }
            else
            {
                lock(cameraOpenGate)
                {
                capture.Open(settings!.CameraIndex,VideoCaptureAPIs.DSHOW);
                capture.Set(VideoCaptureProperties.FrameWidth,settings.CameraWidth);capture.Set(VideoCaptureProperties.FrameHeight,settings.CameraHeight);capture.Set(VideoCaptureProperties.Fps,settings.CameraFps);
                // DirectShow renegotiates the subtype when dimensions/FPS change. Apply compression last.
                capture.Set(VideoCaptureProperties.FourCC,VideoWriter.FourCC('M','J','P','G'));
                if(settings.ManualExposure){capture.Set(VideoCaptureProperties.AutoExposure,0);capture.Set(VideoCaptureProperties.Exposure,settings.Exposure);if(settings.ManualGain)capture.Set(VideoCaptureProperties.Gain,settings.Gain);}
                else capture.Set(VideoCaptureProperties.AutoExposure,1);
                }
            }
            if(!capture.IsOpened())throw new IOException("视频源无法打开，摄像头可能被其他应用占用。");
            preciseTimer=timeBeginPeriod(1)==0;
            double fps=capture.Get(VideoCaptureProperties.Fps);if(!double.IsFinite(fps)||fps<1||fps>240)fps=30;
            bool replay=path is not null&&!network;
            double total=replay?capture.Get(VideoCaptureProperties.FrameCount)/fps:0;
            var clock=Stopwatch.StartNew();double measuredStart=0;int measuredFrames=0;double measuredFps=0;
            while(!token.IsCancellationRequested)
            {
                if(paused){await Task.Delay(30,token);continue;}
                double seek;lock(sync){seek=requestedSeek;requestedSeek=-1;}
                if(seek>=0 && replay)
                {
                    capture.Set(VideoCaptureProperties.PosMsec,seek*1000);session=Interlocked.Increment(ref generation);
                    lock(sync){analysis=null;pending=null;capturedScene.Clear();rgbResults.Clear();lastAlerts.Clear();}
                }
                var started=Stopwatch.GetTimestamp();using var frame=new Mat();
                bool read=capture.Read(frame);long captured=Stopwatch.GetTimestamp();
                if(!read||frame.Empty())
                {
                    if(!replay)throw new IOException("视频源已断开，请检查设备或网络后重新连接。");
                    lock(sync){if(latest is not null)latest=latest with{Ended=true,Paused=true,Sequence=Interlocked.Increment(ref sequence)};}paused=true;CaptureStatus="视频回放结束 · 正在完成末段分析";
                    await Task.Delay(30,token);continue;
                }
                double time=replay?Math.Max(0,(capture.Get(VideoCaptureProperties.PosFrames)-1)/fps):clock.Elapsed.TotalSeconds;
                using var display=new Mat();
                if(frame.Width>960)Cv2.Resize(frame,display,new(960,(int)Math.Round(frame.Height*960d/frame.Width)),0,0,InterpolationFlags.Area);else frame.CopyTo(display);
                if(display.Channels()==4)Cv2.CvtColor(display,display,ColorConversionCodes.BGRA2BGR);
                if(display.Channels()==1)Cv2.CvtColor(display,display,ColorConversionCodes.GRAY2BGR);
                measuredFrames++;double elapsed=clock.Elapsed.TotalSeconds;
                if(elapsed-measuredStart>=1)
                {
                    measuredFps=measuredFrames/(elapsed-measuredStart);measuredFrames=0;measuredStart=elapsed;
                    if(path is null&&settings is {ManualExposure:false,LimitSlowExposure:true}&&!exposureLimited&&elapsed>2)
                    {
                        slowExposureSamples=measuredFps<settings.CameraFps*.55?slowExposureSamples+1:0;
                        if(slowExposureSamples>=2)
                        {capture.Set(VideoCaptureProperties.AutoExposure,0);capture.Set(VideoCaptureProperties.Exposure,settings.Exposure);if(settings.ManualGain)capture.Set(VideoCaptureProperties.Gain,settings.Gain);capture.Set(VideoCaptureProperties.Fps,settings.CameraFps);exposureLimited=true;}
                    }
                }
                var snapshot=new VideoSnapshot(NativeModels.Bytes(display),display.Width,display.Height,time,total,measuredFps,replay,false,false,
                    network?"网络摄像头 "+new Uri(path!).Host:path is not null?Path.GetFileName(path):"USB 摄像头 "+settings!.CameraIndex,session,Interlocked.Increment(ref sequence),captured);
                byte[]? prepared=!observationOnly?models!.PrepareVideoFrame(frame):null;
                lock(sync)
                {
                    latest=snapshot;
                    if(prepared is not null)
                    {
                        capturedScene.Enqueue((session,time,prepared));
                        while(capturedScene.Count>320||capturedScene.Count>0&&(capturedScene.Peek().Generation!=session||time-capturedScene.Peek().Time>5))capturedScene.Dequeue();
                    }
                }
                ready.TrySetResult();
                CaptureStatus=$"{(network?"网络采集":replay?"文件回放":"摄像头采集")} · {snapshot.Width} × {snapshot.Height} · {measuredFps:F1} FPS"+(exposureLimited?" · 曝光已限时":"");
                if(replay){double delay=1000/fps-Stopwatch.GetElapsedTime(started).TotalMilliseconds;if(delay>0)await Task.Delay((int)delay,token);}
            }
        }
        catch(OperationCanceledException){ready.TrySetCanceled(token);}
        catch(Exception ex){ready.TrySetException(ex);CaptureStatus="采集失败："+ex.Message;lock(sync){if(latest is not null)latest=latest with{Paused=true,Ended=true};analysis=null;observation=null;pending=null;rgbResults.Clear();}AppLog.Write("video-capture",ex);}
        finally
        {
            if(preciseTimer)timeEndPeriod(1);
            if(path is null&&(settings?.ManualExposure==true||exposureLimited)&&capture.IsOpened())capture.Set(VideoCaptureProperties.AutoExposure,1);
        }
    }
    private async Task Observe(long session,CancellationToken token)
    {
        try
        {
            await InitializeAsync().WaitAsync(token);var engine=models!;
            int detectionFps=engine.OneStage?8:4;
            ByteTracker tracker=new(new(){TrackThresh=.35f,MatchThresh=.82f,TrackBuffer=60,Mot20=engine.OneStage},detectionFps);
            StablePeopleTracker identities=new();
            List<Track> tracks=[];Dictionary<int,Track> retained=[];List<(double Time,int? Person)> occupancy=[];long lastSeq=-1;double lastDetection=-10,lastJob=-10;long active=-1;
            while(!token.IsCancellationRequested)
            {
                var snap=Latest;if(snap is null||snap.Sequence==lastSeq){await Task.Delay(20,token);continue;}lastSeq=snap.Sequence;
                if(snap.Ended&&!snap.Replay){await Task.Delay(50,token);continue;}
                if(snap.Generation!=active){active=snap.Generation;tracks.Clear();retained.Clear();occupancy.Clear();identities=new();lastDetection=-10;lastJob=-10;analytics.Reset();tracker=new(new(){TrackThresh=.35f,MatchThresh=.82f,TrackBuffer=60,Mot20=engine.OneStage},detectionFps);}
                using var source=Mat.FromPixelData(snap.Height,snap.Width,MatType.CV_8UC3,snap.Pixels);var started=Stopwatch.StartNew();
                bool newDetection=false;
                if(snap.Ended||snap.Time-lastDetection>=(engine.OneStage?.12:.25))
                {
                    PeopleObservation[] observations;
                    try{observations=engine.ObservePeople(source);}
                    catch(Exception ex) when(ex is InvalidDataException or Microsoft.ML.OnnxRuntime.OnnxRuntimeException)
                    {
                        ModelStatus="人员分析暂未完成，正在重试："+ex.Message;
                        lock(sync){analysis=null;}tracks.Clear();await Task.Delay(200,token);continue;
                    }
                    var boxes=observations.Select(o=>(o.Box,o.Score)).ToArray();(Joint[] Image,Joint[] World)? fallback=null;
                    if(!engine.Realtime&&boxes.Length==0&&tracks.Count<=1)
                    {
                        var candidate=engine.Pose(source,new(0,0,1,1),tracks.FirstOrDefault()?.Image);
                        var visible=candidate.Image.Where(j=>j.Confidence>=.5&&j.X>=0&&j.X<=1&&j.Y>=0&&j.Y<=1).ToArray();
                        if(visible.Length>=6&&visible.Average(j=>j.Confidence)>.6)
                        {
                            var box=new Box(Math.Clamp(visible.Min(j=>j.X)-.035f,0,1),Math.Clamp(visible.Min(j=>j.Y)-.035f,0,1),Math.Clamp(visible.Max(j=>j.X)+.035f,0,1),Math.Clamp(visible.Max(j=>j.Y)+.035f,0,1));
                            if(box.Width>.03&&box.Height>.04){boxes=[(box,visible.Average(j=>j.Confidence))];fallback=candidate;}
                        }
                    }
                    var stable=engine.OneStage?identities.Update(observations.Select(o=>new StablePeopleTracker.Observation(o.Box,o.Score,o.Joints,Appearance(source,o.Box))).ToArray(),snap.Time):[];
                    var tracked=engine.OneStage?[]:tracker.Update(boxes.Select(b=>new Detection(b.Box.X1*1000,b.Box.Y1*1000,b.Box.X2*1000,b.Box.Y2*1000,b.Score)).ToArray());
                    List<Track> fresh=[];
                    var unmatched=observations.ToList();
                    var assigned=engine.OneStage?stable.Select(p=>(Id:p.Id,Box:p.Observation.Box,Score:p.Observation.Score,Image:p.Observation.Joints)):
                        tracked.Take(6).Select(d=>{var b=d.Tlbr;return (Id:d.TrackId,Box:new Box(Math.Clamp((float)b[0]/1000,0,1),Math.Clamp((float)b[1]/1000,0,1),Math.Clamp((float)b[2]/1000,0,1),Math.Clamp((float)b[3]/1000,0,1)),Score:d.Score,Image:Array.Empty<Joint>());});
                    foreach(var detection in assigned)
                    {
                        if(!retained.TryGetValue(detection.Id,out var match)){match=new(){Id=detection.Id};retained[match.Id]=match;}
                        match.Box=detection.Box;
                        if(match.Box.Width<=.01||match.Box.Height<=.01)continue;
                        if(snap.Time-match.Seen>2.5){match.Poses.Clear();match.Frames.Clear();match.PoseRisk=null;match.Candidates=[];match.Timeline.Reset();}
                        if(engine.OneStage)
                        {
                            match.Image=detection.Image;
                        }
                        match.Score=detection.Score;match.Seen=snap.Time;match.Detector=fallback.HasValue?"BlazePose辅助":engine.DetectorName;fresh.Add(match);
                    }
                    foreach(var id in retained.Keys.Where(id=>snap.Time-retained[id].Seen>2.5).ToArray())retained.Remove(id);
                    tracks=fresh;lastDetection=snap.Time;newDetection=true;
                }
                foreach(var track in tracks)
                {
                    var previous=track.Image;var pose=engine.OneStage?(Image:track.Image,World:Array.Empty<Joint>()):engine.Pose(source,track.Box,previous);track.Image=pose.Image;track.World=[];
                    if(engine.Realtime&&(!engine.OneStage||newDetection))
                    {track.Timeline.Update(pose.Image,track.Box,snap.Time,snap.Width/(float)snap.Height);track.Posture=track.Timeline.Activity;track.PostureConfidence=track.Timeline.Confidence;}
                    Joint[] observed=pose.Image.Length==(engine.Realtime?17:33)?pose.Image:new Joint[engine.Realtime?17:33];
                    track.Poses.Add((snap.Time,observed));track.Poses.RemoveAll(p=>snap.Time-p.Time>5);if(track.Poses.Count>96)track.Poses.RemoveAt(0);
                    if(!observationOnly)
                    {
                    using var crop=Crop(source,track.Box);
                    track.Frames.Add((snap.Time,engine.PrepareVideoFrame(crop)));track.Frames.RemoveAll(p=>snap.Time-p.Time>5);if(track.Frames.Count>60)track.Frames.RemoveAt(0);
                    }
                    if(track.Poses.Count>=6 && snap.Time-track.Poses[0].Time>=1 && snap.Time-track.LastPose>=.5)
                    {
                        if(engine.Realtime)
                        {
                            var window=track.Poses.Where(p=>snap.Time-p.Time<=4).Select(p=>p.Pose).ToArray();
                            float coverage=window.Count(p=>new[]{5,6,11,12}.All(j=>p[j].Confidence>=.3))/(float)window.Length;
                            var scores=coverage>=.4&&window.Length>=6?engine.ActionScores(window):null;
                            track.Candidates=scores is null?[]:PretrainedActions.Top(scores);
                            // Generic NTU pretraining is shown as supplementary evidence; it does not trigger safety alerts.
                            track.PoseRisk=null;
                        }
                        else
                        {
                            var sampled=Enumerable.Range(0,48).Select(i=>track.Poses[(int)Math.Round(i*(track.Poses.Count-1)/47d)].Pose).ToArray();
                            float coverage=sampled.Count(p=>new[]{11,12,23,24}.All(j=>p[j].Confidence>=.3))/48f;
                            track.PoseRisk=coverage>=.2?engine.PoseRisk(sampled,snap.Width/(float)snap.Height):null;
                        }
                        track.LastPose=snap.Time;
                    }
                }
                occupancy.Add((snap.Time,tracks.Count==1?tracks[0].Id:tracks.Count>1?-1:null));occupancy.RemoveAll(p=>snap.Time-p.Time>5);
                (long Generation,double Time,byte[] Frame)[] scene;lock(sync)scene=capturedScene.Where(p=>p.Generation==active&&p.Time<=snap.Time).ToArray();
                // Avoid interpreting a repeated one-second warm-up as a complete action clip.
                if(!observationOnly&&scene.Length>=8&&snap.Time-scene[0].Time>=2&&(snap.Ended||snap.Time-lastJob>=.75))
                {
                    var individual=tracks.Where(t=>t.Frames.Count>=8&&snap.Time-t.Frames[0].Time>=2).Select(t=>(t.Id,t.Frames.Select(f=>f.Frame).ToList(),t.Frames.Where(f=>f.Time>=snap.Time-2).Select(f=>f.Frame).ToList(),t.Box.Clipped||t.Image.Length!=(engine.Realtime?17:33),engine.Realtime&&(t.Timeline.FallMotion||t.Timeline.ShortWindowSupported&&t.PostureConfidence>=.65f&&(t.Posture is "lying" or "fallen")))).ToArray();
                    int? soleId=tracks.Count==1?tracks[0].Id:null;
                    bool stableSole=soleId.HasValue&&occupancy.All(f=>f.Person is null||f.Person==soleId)&&occupancy.Count(f=>f.Person==soleId)>=6;
                    lock(sync){pending=new(active,snap.Time,scene.Select(f=>f.Frame).ToList(),scene.Where(f=>f.Time>=snap.Time-2).Select(f=>f.Frame).ToList(),individual,stableSole?soleId:null);}lastJob=snap.Time;
                }
                PersonState[] people=tracks.Select(t=>
                {
                    (float Risk,string Activity,float Confidence,int Consecutive,double Time,bool WholeScene,float? ShortRisk,int ShortConsecutive)? rgb;
                    lock(sync){rgb=rgbResults.TryGetValue(t.Id,out var value)&&snap.Time-value.Time<3?value:null;}
                    bool partial=t.Box.Clipped||t.Image.Length!=(engine.Realtime?17:33)||(engine.Realtime?new[]{5,6,11,12,13,14,15,16}:new[]{11,12,23,24,25,26,27,28}).Any(i=>t.Image[i].Confidence<.4);
                    string activity=engine.Realtime?t.Posture:rgb?.Activity??"";
                    string status=Assessment(rgb?.Risk,rgb?.Consecutive??0,t.PoseRisk,partial,activity,engine.Realtime&&t.Timeline.FallMotion,engine.Realtime?t.PostureConfidence:0,rgb?.ShortRisk,rgb?.ShortConsecutive??0);
                    return new PersonState(t.Id,t.Box,t.Score,t.Image,[],snap.Time,rgb?.Risk,t.PoseRisk,activity,engine.Realtime?t.PostureConfidence:rgb?.Confidence??0,status,partial,rgb?.Consecutive??0,rgb?.Time??-1,t.Detector,engine.Realtime?engine.DetectorName+" 姿态与时序观测":"R3D 时序模型",Alternatives:t.Candidates,WholeSceneInput:rgb?.WholeScene,FallMotion:engine.Realtime&&t.Timeline.FallMotion,MotionEvidence:t.Timeline.MotionEvidence,ShortRgbScore:rgb?.ShortRisk,ShortConsecutive:rgb?.ShortConsecutive??0);
                }).ToArray();
                lock(sync)
                {
                    if(active==generation&&!(latest?.Ended==true&&!latest.Replay))
                    {
                        bool freshRgb=analysis is not null&&analysis.RgbTime>=0&&snap.Time-analysis.RgbTime<3;
                        analysis=new(active,snap.Time,people,freshRgb?analysis?.SceneScore:null,freshRgb?analysis?.Activity??"":"",
                            people.Length==0?"等待人员识别":"人员状态随视频更新",started.Elapsed.TotalMilliseconds,engine.Device,
                            freshRgb?analysis?.Attention:null,freshRgb?analysis?.FeatureChange:null,freshRgb?analysis?.RgbTime??-1:-1,SceneConsecutive:freshRgb?analysis?.SceneConsecutive??0:0);
                        var config=rules;analysis=analytics.Update(analysis,snap,config.Zones,config.Inactivity,e=>Event?.Invoke(e));
                        observation=new(snap,analysis);
                        if(!observationOnly)PublishEvents(analysis,snap);
                    }
                }
                int sleep=Math.Max(1,100-(int)started.ElapsedMilliseconds);await Task.Delay(sleep,token);
            }
        }
        catch(OperationCanceledException){}
        catch(Exception ex){ModelStatus="识别失败："+ex.Message;AppLog.Write("video-observe",ex);lock(sync){analysis=null;observation=null;}}
    }
    private async Task Classify(long session,CancellationToken token)
    {
        try
        {
            await InitializeAsync().WaitAsync(token);var engine=models!;int cursor=0,sceneConsecutive=0;double lastSceneTime=-1;float[]? previousEmbedding=null;long active=-1;
            while(!token.IsCancellationRequested)
            {
                RgbJob? job;lock(sync){job=pending;pending=null;}
                if(job is null){await Task.Delay(30,token);continue;}
                if(job.Generation!=Interlocked.Read(ref generation))continue;
                if(active!=job.Generation){active=job.Generation;previousEmbedding=null;sceneConsecutive=0;lastSceneTime=-1;}
                try
                {
                var diagnostic=engine.PredictVideo(job.Scene);var scores=diagnostic.Scores;
                float? change=null;
                if(previousEmbedding is not null)
                {
                    double dot=previousEmbedding.Zip(diagnostic.Embedding,(a,b)=>(double)a*b).Sum(),a=previousEmbedding.Sum(x=>(double)x*x),b=diagnostic.Embedding.Sum(x=>(double)x*x);
                    change=(float)Math.Clamp(1-dot/Math.Max(1e-9,Math.Sqrt(a*b)),0,2);
                }
                previousEmbedding=diagnostic.Embedding;float risk=engine.Risk(scores);string activity=engine.Classes[Array.IndexOf(scores,scores.Max())];
                sceneConsecutive=risk>=.5?(lastSceneTime>=0&&job.Time-lastSceneTime<3?sceneConsecutive:0)+1:0;lastSceneTime=job.Time;
                int count=job.People.Length;var results=new List<(int Id,float Risk,string Activity,float Confidence,bool WholeScene,float? ShortRisk)>();
                for(int i=0;i<Math.Min(2,count);i++)
                {
                    var person=job.People[(cursor+i)%count];
                    // Preserve the training framing for a single visible person. A scene score
                    // must never be assigned to an individual in a multi-person observation.
                    bool wholeScene=count==1&&job.ScenePersonId==person.Id;
                    var p=wholeScene?scores:engine.VideoScores(person.Frames);
                    // A recent motion window can recover a brief fall diluted by earlier upright
                    // frames. It is requested only with measured horizontal/motion support.
                    var shortFrames=wholeScene?job.ShortScene:person.ShortFrames;
                    float? shortRisk=person.ShortEligible&&shortFrames.Count>=6&&shortFrames.Count<(wholeScene?job.Scene.Count:person.Frames.Count)
                        ?engine.Risk(engine.VideoScores(shortFrames)):null;
                    results.Add((person.Id,engine.Risk(p),engine.Classes[Array.IndexOf(p,p.Max())],p.Max(),wholeScene,shortRisk));
                }
                cursor+=2;
                lock(sync)
                {
                    if(job.Generation!=generation||latest is null||latest.Time-job.Time>=3||(latest.Ended&&!latest.Replay))continue;
                    foreach(int id in rgbResults.Keys.Where(id=>job.Time-rgbResults[id].Time>3).ToArray())rgbResults.Remove(id);
                    foreach(var r in results)
                    {
                        rgbResults.TryGetValue(r.Id,out var previous);int consecutive=r.Risk>=.5?(job.Time-previous.Time<3?previous.Consecutive:0)+1:0;
                        int shortConsecutive=r.ShortRisk>=.5?(job.Time-previous.Time<3?previous.ShortConsecutive:0)+1:0;
                        rgbResults[r.Id]=(r.Risk,r.Activity,r.Confidence,consecutive,job.Time,r.WholeScene,r.ShortRisk,shortConsecutive);
                    }
                    if(analysis is not null)
                    {
                        var people=analysis.People.Select(p=>
                        {
                            if(!rgbResults.TryGetValue(p.Id,out var r))return p;
                            string currentActivity=engine.Realtime?p.Activity:r.Activity;
                            return p with{RgbScore=r.Risk,Activity=currentActivity,Confidence=engine.Realtime?p.Confidence:r.Confidence,Consecutive=r.Consecutive,PredictionTime=r.Time,WholeSceneInput=r.WholeScene,
                                ActivitySource=engine.Realtime?engine.DetectorName+" 姿态与时序观测":"R3D 时序模型",
                                ShortRgbScore=r.ShortRisk,ShortConsecutive=r.ShortConsecutive,
                                Status=Assessment(r.Risk,r.Consecutive,p.PoseScore,p.Partial,currentActivity,p.FallMotion,engine.Realtime?p.Confidence:0,r.ShortRisk,r.ShortConsecutive)};
                        }).ToArray();
                        analysis=analysis with{SceneScore=risk,Activity=activity,People=people,Attention=diagnostic.Attention,FeatureChange=change,RgbTime=job.Time,SceneConsecutive=sceneConsecutive};
                        if(latest.Ended&&latest.Replay&&job.Time>=latest.Time-.001)CaptureStatus="视频回放结束 · 分析已完成";
                        if(latest is not null)PublishEvents(analysis,latest);
                    }
                }
                }
                catch(Exception ex) when(ex is InvalidDataException or Microsoft.ML.OnnxRuntime.OnnxRuntimeException)
                {
                    ModelStatus="视频复核暂未完成，正在重试："+ex.Message;
                    lock(sync){rgbResults.Clear();if(analysis is not null&&analysis.Generation==job.Generation)analysis=analysis with{SceneScore=null,Attention=null,FeatureChange=null,RgbTime=-1};}
                    await Task.Delay(250,token);continue;
                }
                ModelStatus=engine.DetectorName+" · "+engine.VideoName+" · "+(engine.VideoGpuEnabled?"硬件加速已启用":"使用 CPU 分析");
            }
        }
        catch(OperationCanceledException){}
        catch(Exception ex){ModelStatus="视频时序识别失败："+ex.Message;AppLog.Write("video-classify",ex);}
    }
    public static string Assessment(float? rgb,int consecutive,float? pose,bool partial,string activity,bool fallMotion=false,float postureConfidence=0,float? shortRgb=null,int shortConsecutive=0)
    {
        if(rgb>=.5&&!fallMotion&&postureConfidence>=.65f&&(activity is "standing" or "walk"))return Labels.Activity(activity)+" · 视频结果待复核";
        if(rgb>=.5)return consecutive>=2||fallMotion||postureConfidence>=.65f&&(activity is "lying" or "fallen")?"疑似跌倒 · 请核查":"视频异常 · 复核中";
        if(shortRgb>=.5&&(fallMotion||postureConfidence>=.65f&&(activity is "lying" or "fallen")))
            return shortConsecutive>=2||fallMotion?"疑似跌倒 · 请核查":"近期动作异常 · 复核中";
        if(fallMotion)return "动作异常 · 请核查";
        string value=pose>=.5?"姿态异常 · 请核查":activity.Length>0?Labels.Activity(activity):"识别中";
        return value;
    }
    public static bool IsFallAlert(PersonState p)=>p.Status.StartsWith("疑似跌倒",StringComparison.Ordinal)||p.FallMotion||p.PoseScore>=.5;
    public static bool UprightDisagreement(AnalysisSnapshot result)=>result.SceneScore>=.5&&result.People.Length>0&&result.People.All(p=>p.WholeSceneInput==true&&p.Confidence>=.65f&&(p.Activity is "standing" or "walk")&&!p.FallMotion);
    public static string SceneAssessment(AnalysisSnapshot result)=>UprightDisagreement(result)?"人员直立 · 视频结果待复核":
        result.People.Any(p=>p.Status.StartsWith("疑似跌倒",StringComparison.Ordinal))?"疑似跌倒，请核查":result.People.Any(p=>p.FallMotion)?"动作异常，请核查":
        result.People.Any(p=>p.PoseScore>=.5)?"姿态异常，请核查":result.SceneScore>=.5?(result.SceneConsecutive>=2?"疑似跌倒，请核查":"视频异常 · 复核中"):result.SceneScore.HasValue?"持续监测中":"正在分析画面";
    private void PublishEvents(AnalysisSnapshot result,VideoSnapshot source)
    {
        foreach(var person in result.People.Where(IsFallAlert))
            Publish(person.Id,person.Status,$"视频位置 {source.Time:F2} 秒；长窗口视频分值 {person.RgbScore:F3}；"+(person.ShortRgbScore.HasValue?$"最近2秒视频分值 {person.ShortRgbScore:F3}（连续{person.ShortConsecutive}窗）；":"")+$"姿态分析分值 {person.PoseScore:F3}；画面{(person.Partial?"不完整":"完整")}。"+person.MotionEvidence+(person.PoseScore>=.5&&person.RgbScore<.5?"两项分析结果不一致，请核查画面。":"")+"分析结果仅用于辅助核查。",source);
        if(result.SceneScore>=.5&&result.SceneConsecutive>=2&&!UprightDisagreement(result)&&result.People.All(p=>!IsFallAlert(p)))
            Publish(0,"疑似跌倒 · 待定位",$"整体画面分析分值 {result.SceneScore:F3}，尚未定位到对应风险人员，请核查监控画面。",source);
    }
    private void Publish(int id,string state,string evidence,VideoSnapshot source)
    {
        if(lastAlerts.TryGetValue(id,out var previous)&&previous.State==state&&source.Time-previous.Time<30)return;
        lastAlerts[id]=(state,source.Time);
        Event?.Invoke(new(){Source=source.Source,Subject=id==0?"未定位":"P"+id.ToString("00"),Kind=state,Evidence=evidence,IsReplay=source.Replay,VideoPositionSeconds=source.Time});
    }
    private static Mat Crop(Mat frame,Box b)
    {
        float dx=b.Width*.12f,dy=b.Height*.12f;
        int x1=Math.Clamp((int)((b.X1-dx)*frame.Width),0,frame.Width-1),x2=Math.Clamp((int)((b.X2+dx)*frame.Width),x1+1,frame.Width);
        int y1=Math.Clamp((int)((b.Y1-dy)*frame.Height),0,frame.Height-1),y2=Math.Clamp((int)((b.Y2+dy)*frame.Height),y1+1,frame.Height);
        return new(frame,new Rect(x1,y1,x2-x1,y2-y1));
    }
    public void TogglePause(){paused=!paused;lock(sync){if(latest is not null)latest=latest with{Paused=paused};}}
    public void Seek(double seconds)
    {
        var snap=Latest;if(snap?.Replay!=true)return;
        lock(sync){requestedSeek=Math.Clamp(seconds,0,Math.Max(0,snap.Duration-.1));}paused=false;
    }
    private static float[] Appearance(Mat source,Box box)
    {
        using var crop=Crop(source,box);using var thumb=new Mat();Cv2.Resize(crop,thumb,new(16,24),0,0,InterpolationFlags.Area);
        var bins=new float[48];
        for(int y=4;y<20;y++)for(int x=3;x<13;x++){var pixel=thumb.At<Vec3b>(y,x);bins[pixel.Item0/16]++;bins[16+pixel.Item1/16]++;bins[32+pixel.Item2/16]++;}
        float norm=(float)Math.Sqrt(bins.Sum(v=>v*v));return bins.Select(v=>v/Math.Max(1,norm)).ToArray();
    }
    public void ResetStatistics(){lock(sync){analytics.Reset();if(analysis is not null)analysis=analysis with{Analytics=null};}}
    public async Task StopAsync()
    {
        await lifecycle.WaitAsync();try{await StopCoreAsync();}finally{lifecycle.Release();}
    }
    private async Task StopCoreAsync()
    {
        cancellation?.Cancel();
        Interlocked.Increment(ref generation);
        lock(sync){latest=null;analysis=null;observation=null;pending=null;capturedScene.Clear();rgbResults.Clear();lastAlerts.Clear();}
        if(workers.Length>0)try{await Task.WhenAll(workers).WaitAsync(TimeSpan.FromSeconds(5));}catch(OperationCanceledException){}catch(TimeoutException){throw new IOException("视频设备没有及时释放，请检查驱动后关闭应用。");}
        workers=[];cancellation?.Dispose();cancellation=null;Interlocked.Increment(ref generation);
        lock(sync){latest=null;analysis=null;pending=null;rgbResults.Clear();lastAlerts.Clear();}CaptureStatus="视频已停止";
    }
    public async ValueTask DisposeAsync(){await StopAsync();if(initialization is not null)try{models=await initialization;}catch{}if(modelOwner is null)models?.Dispose();}
}
