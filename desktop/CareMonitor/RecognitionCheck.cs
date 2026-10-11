using System.Diagnostics;
using System.IO;
using System.Text.Json;
using CareMonitor.Models;
using CareMonitor.Services;

namespace CareMonitor;

/// <summary>Uses the application's actual capture/observation/classification workers.</summary>
public static class RecognitionCheck
{
    public static void Run(string[] args)=>Task.Run(()=>RunAsync(args)).GetAwaiter().GetResult();
    private static async Task RunAsync(string[] args)
    {
        string listPath=args[Array.IndexOf(args,"--recognition-replay")+1];
        string output=args[Array.IndexOf(args,"--eval-output")+1];
        using var list=JsonDocument.Parse(File.ReadAllText(listPath));
        string model=args.Contains("--fall-model")?args[Array.IndexOf(args,"--fall-model")+1]:"stage_c";
        await using var service=new VideoService(Path.Combine(AppContext.BaseDirectory,"Assets"),args.Contains("--gpu"),true,model);
        await service.InitializeAsync();var rows=new List<object>();
        foreach(var item in list.RootElement.EnumerateArray())
        {
            string file=item.GetProperty("file").GetString()!;bool truth=item.GetProperty("fall").GetBoolean();
            var events=new List<MonitorEvent>();object eventSync=new();
            void OnEvent(MonitorEvent e){lock(eventSync)events.Add(e);}
            service.Event+=OnEvent;
            await service.StartFileAsync(file);var watch=Stopwatch.StartNew();
            var observations=new List<AnalysisSnapshot>();var sequences=new HashSet<long>();long lastGeneration=-1;double lastTime=-1,lastRgb=-1;
            bool ended=false;
            while(watch.Elapsed<TimeSpan.FromSeconds(90))
            {
                await Task.Delay(25);
                var video=service.Latest;var result=service.Analysis;
                if(video is not null)sequences.Add(video.Sequence);
                if(result is not null&&(result.Generation!=lastGeneration||result.Time!=lastTime||result.RgbTime!=lastRgb))
                {observations.Add(result with{Attention=null,Analytics=null});lastGeneration=result.Generation;lastTime=result.Time;lastRgb=result.RgbTime;}
                if(video?.Ended==true&&service.ReplayAnalysisComplete)
                {var final=service.Analysis;if(final is not null)observations.Add(final with{Attention=null,Analytics=null});ended=true;break;}
                if(service.ModelStatus.Contains("失败")||service.CaptureStatus.Contains("失败"))throw new InvalidOperationException(service.ModelStatus+" / "+service.CaptureStatus);
            }
            if(!ended)throw new TimeoutException("Public replay did not end: "+Path.GetFileName(file));
            service.Event-=OnEvent;
            MonitorEvent[] eventCopy;lock(eventSync)eventCopy=events.ToArray();
            rows.Add(new{file,truth_fall=truth,video_frames=sequences.Count,observations=observations.Count,
                detected_observations=observations.Count(o=>o.People.Length>0),max_people=observations.Count==0?0:observations.Max(o=>o.People.Length),
                track_ids=observations.SelectMany(o=>o.People).Select(p=>p.Id).Distinct().ToArray(),
                predicted_fall=eventCopy.Any(e=>e.Kind.Contains("跌倒")),events=eventCopy,
                final=observations.LastOrDefault(),timeline=observations.Select(o=>new{o.Time,o.SceneScore,o.RgbTime,o.InferenceMs,
                    people=o.People.Select(p=>new{p.Id,p.Box,p.Activity,p.Status,p.Confidence,p.RgbScore,p.ShortRgbScore,p.ShortConsecutive,p.PredictionTime,p.WholeSceneInput,p.FallMotion,p.MotionEvidence,p.Partial,
                        image_joints=args.Contains("--pose-diagnostics")?p.ImageJoints:null})}).ToArray()});
            await service.StopAsync();
            File.WriteAllText(output,JsonSerializer.Serialize(new{state="running",completed=rows.Count,total=list.RootElement.GetArrayLength(),rows},new JsonSerializerOptions{WriteIndented=true}));
        }
        File.WriteAllText(output,JsonSerializer.Serialize(new{state="complete",model_status=service.ModelStatus,rows,
            notes="Actual VideoService realtime replay, 5-second timestamp windows and application event rules. Known public development data; no independent clinical accuracy claim."},new JsonSerializerOptions{WriteIndented=true}));
    }
}
