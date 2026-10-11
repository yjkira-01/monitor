using System.Diagnostics;
using System.IO;
using System.Text.Json;
using CareMonitor.Services;
using OpenCvSharp;

namespace CareMonitor;

/// <summary>Deterministic offline window comparison; never starts devices or saves frames.</summary>
public static class TemporalCheck
{
    public static void Run(string[] args)
    {
        string listPath=args[Array.IndexOf(args,"--temporal-check")+1];
        string output=args[Array.IndexOf(args,"--eval-output")+1];
        string model=args.Contains("--fall-model")?args[Array.IndexOf(args,"--fall-model")+1]:"gmd";
        using var list=JsonDocument.Parse(File.ReadAllText(listPath));
        using var engine=new NativeModels(Path.Combine(AppContext.BaseDirectory,"Assets"),args.Contains("--gpu"),true,model);
        var rows=new List<object>();
        double[] scales=[2,3.5,5];
        foreach(var item in list.RootElement.EnumerateArray())
        {
            string file=item.GetProperty("file").GetString()!;bool truth=item.GetProperty("fall").GetBoolean();
            using var capture=new VideoCapture(file);
            if(!capture.IsOpened())throw new InvalidDataException("不能读取对比视频："+file);
            double fps=capture.Fps;long count=(long)capture.FrameCount;
            if(!double.IsFinite(fps)||fps<=0||count<2)throw new InvalidDataException("视频时基无效。");
            // Decode once, sample at 12 Hz and keep resized bytes only in memory.
            var frames=new List<(double Time,byte[] Bytes)>();using var mat=new Mat();double next=0;
            for(long index=0;capture.Read(mat)&&!mat.Empty();index++)
            {
                double time=index/fps;
                if(time+1e-6<next&&index<count-1)continue;
                frames.Add((time,engine.PrepareVideoFrame(mat)));next=time+1/12d;
            }
            if(frames.Count<6)throw new InvalidDataException("视频没有足够有效帧。");
            double duration=frames[^1].Time;
            var ends=Enumerable.Range(0,(int)Math.Floor(Math.Max(0,duration-1.5)/.8)+1).Select(i=>1.5+i*.8).Where(t=>t<duration-.01).Append(duration).ToArray();
            var timeline=new List<object>();double totalMs=0;int predictions=0;
            var scoresByScale=scales.ToDictionary(s=>s,s=>new List<(double Time,float Score)>());
            foreach(double end in ends)
            {
                var current=new Dictionary<string,float>();
                foreach(double scale in scales)
                {
                    var window=frames.Where(f=>f.Time>=Math.Max(0,end-scale)-1e-6&&f.Time<=end+1e-6).ToArray();
                    if(window.Length<6)continue;
                    var watch=Stopwatch.StartNew();float risk=engine.Risk(engine.VideoScores(window.Select(f=>f.Bytes).ToArray()));
                    totalMs+=watch.Elapsed.TotalMilliseconds;predictions++;
                    current[scale.ToString(System.Globalization.CultureInfo.InvariantCulture)]=risk;
                    scoresByScale[scale].Add((end,risk));
                }
                timeline.Add(new{end,window_scores=current});
            }
            var perScale=scoresByScale.ToDictionary(p=>p.Key.ToString(System.Globalization.CultureInfo.InvariantCulture),p=>new{
                peak=p.Value.Max(v=>v.Score),
                consecutive=p.Value.Zip(p.Value.Skip(1),(a,b)=>a.Score>=.5&&b.Score>=.5).Any(v=>v),
                first_candidate=p.Value.Where(v=>v.Score>=.5).Select(v=>(double?)v.Time).FirstOrDefault()});
            rows.Add(new{file,truth_fall=truth,duration,scales=perScale,timeline,mean_inference_ms=totalMs/Math.Max(1,predictions)});
            File.WriteAllText(output,JsonSerializer.Serialize(new{state="running",model,completed=rows.Count,total=list.RootElement.GetArrayLength(),rows},new JsonSerializerOptions{WriteIndented=true}));
        }
        File.WriteAllText(output,JsonSerializer.Serialize(new{state="complete",model,device=engine.Device,scales,rows,
            notes="Fixed 0.8-second endpoints; raw temporal model scores only, without pose or GUI alert rules. Known development videos. No training, camera capture or image export."},new JsonSerializerOptions{WriteIndented=true}));
    }
}
