using System.Diagnostics;
using System.IO;
using System.Text.Json;
using CareMonitor.Models;
using CareMonitor.Services;
using OpenCvSharp;

namespace CareMonitor;

public static class PeopleCheck
{
    public static void Run(string[] args)
    {
        using var fixtures=JsonDocument.Parse(File.ReadAllText(args[Array.IndexOf(args,"--people-fixtures")+1]));
        using var models=new NativeModels(Path.Combine(AppContext.BaseDirectory,"Assets"),args.Contains("--gpu"),true);
        var rows=new List<object>();
        string output=args[Array.IndexOf(args,"--eval-output")+1];
        foreach(var fixture in fixtures.RootElement.EnumerateArray())
        {
            string file=fixture.GetProperty("file").GetString()!;double time=fixture.GetProperty("time").GetDouble();
            using var cap=new VideoCapture(file);cap.Set(VideoCaptureProperties.PosMsec,time*1000);using var decoded=new Mat();
            if(!cap.Read(decoded))throw new IOException("Fixture could not decode");
            using var source=new Mat();Cv2.Resize(decoded,source,new(640,(int)(decoded.Height*640d/decoded.Width)));
            foreach(string variant in new[]{"normal","dark_18pct","partial_bottom_crop","two_side_by_side"})
            {
                File.WriteAllText(output,JsonSerializer.Serialize(new{state="running",current_file=file,current_variant=variant,rows}));
                using var transformed=new Mat();
                if(variant=="dark_18pct")source.ConvertTo(transformed,MatType.CV_8UC3,.18);
                else if(variant=="partial_bottom_crop")
                {using var region=new Mat(source,new Rect(0,0,source.Width,(int)(source.Height*.65)));region.CopyTo(transformed);}
                else if(variant=="two_side_by_side")Cv2.HConcat(new[]{source,source},transformed);
                else source.CopyTo(transformed);
                var watch=Stopwatch.StartNew();var people=models.ObservePeople(transformed);double elapsed=watch.Elapsed.TotalMilliseconds;
                // The old two-stage path provides the pose in a separate call.
                if(!models.OneStage)people=people.Select(p=>p with{Joints=models.Pose(transformed,p.Box).Image}).ToArray();
                float aspect=transformed.Width/(float)transformed.Height;
                rows.Add(new{file,time,variant,expected_people=variant=="two_side_by_side"?2:1,detected_people=people.Count(p=>p.Score>=.35f),elapsed_ms=elapsed,
                    people=people.Select(p=>{var posture=PostureObservation.Classify(p.Joints,p.Box,aspect:aspect);return new{p.Box,p.Score,joints=p.Joints,posture=posture.Activity,posture_confidence=posture.Confidence};}).ToArray()});
            }
        }
        File.WriteAllText(output,JsonSerializer.Serialize(new{state="complete",model=models.DetectorName,rows,
            notes="Public frames with synthetic brightness, crop and duplicated-scene transforms. These are regression fixtures, not real night/occlusion/multi-person accuracy."},new JsonSerializerOptions{WriteIndented=true}));
    }
}
