using System.Diagnostics;
using System.IO;
using System.Text.Json;
using CareMonitor.Services;
using OpenCvSharp;
namespace CareMonitor;
public static class CameraCheck
{
    public static void Run(string[] args)
    {
        string output=args[Array.IndexOf(args,"--eval-output")+1];var rows=new List<object>();
        if(args.Contains("--dual-matrix")){DualMatrix(output);return;}
        foreach(var camera in CameraDevices.List().Where(c=>c.Name.Contains("C60E")&&!args.Contains("--dual-only")))
            foreach(var mode in new[]{(640,480,30,false),(640,360,30,false),(640,480,30,true),(320,240,30,true)})
            {
                using var capture=new VideoCapture(camera.Index,VideoCaptureAPIs.DSHOW);using var mat=new Mat();
                bool codec=capture.Set(VideoCaptureProperties.FourCC,VideoWriter.FourCC('M','J','P','G'));
                bool width=capture.Set(VideoCaptureProperties.FrameWidth,mode.Item1),height=capture.Set(VideoCaptureProperties.FrameHeight,mode.Item2),fps=capture.Set(VideoCaptureProperties.Fps,mode.Item3);
                codec=capture.Set(VideoCaptureProperties.FourCC,VideoWriter.FourCC('M','J','P','G'));
                bool auto=capture.Set(VideoCaptureProperties.AutoExposure,mode.Item4?0:1),exposure=mode.Item4?capture.Set(VideoCaptureProperties.Exposure,-5):true;
                var clock=Stopwatch.StartNew();int count=0;while(clock.Elapsed.TotalSeconds<4&&capture.Read(mat)&&!mat.Empty())count++;
                rows.Add(new{camera,requested=mode,actual_width=capture.FrameWidth,actual_height=capture.FrameHeight,reported_fps=capture.Fps,measured_fps=count/clock.Elapsed.TotalSeconds,
                    codec_accepted=codec,width_accepted=width,height_accepted=height,fps_accepted=fps,auto_accepted=auto,exposure_accepted=exposure,
                    exposure=capture.Get(VideoCaptureProperties.Exposure),auto=capture.Get(VideoCaptureProperties.AutoExposure),fourcc=capture.Get(VideoCaptureProperties.FourCC)});
                capture.Set(VideoCaptureProperties.AutoExposure,1);
                File.WriteAllText(output,JsonSerializer.Serialize(new{rows,no_images_saved=true},new JsonSerializerOptions{WriteIndented=true,IncludeFields=true}));
            }
        var devices=CameraDevices.List().Where(c=>c.Name.Contains("C60E")).ToArray();
        if(devices.Length==2)
        {
            VideoCapture Open(int index)
            {var c=new VideoCapture(index,VideoCaptureAPIs.DSHOW);c.Set(VideoCaptureProperties.FrameWidth,640);c.Set(VideoCaptureProperties.FrameHeight,480);c.Set(VideoCaptureProperties.Fps,30);c.Set(VideoCaptureProperties.FourCC,VideoWriter.FourCC('M','J','P','G'));c.Set(VideoCaptureProperties.AutoExposure,0);c.Set(VideoCaptureProperties.Exposure,-5);return c;}
            using var l=Open(devices[0].Index);using var r=Open(devices[1].Index);
            object Measure(VideoCapture c,int index)
            {using var mat=new Mat();var clock=Stopwatch.StartNew();int count=0;while(clock.Elapsed.TotalSeconds<8&&c.Read(mat)&&!mat.Empty())count++;return new{index,measured_fps=count/clock.Elapsed.TotalSeconds,frames=count};}
            var a=Task.Factory.StartNew(()=>Measure(l,devices[0].Index),CancellationToken.None,TaskCreationOptions.LongRunning,TaskScheduler.Default);
            var b=Task.Factory.StartNew(()=>Measure(r,devices[1].Index),CancellationToken.None,TaskCreationOptions.LongRunning,TaskScheduler.Default);
            Task.WhenAll(a,b).GetAwaiter().GetResult();l.Set(VideoCaptureProperties.AutoExposure,1);r.Set(VideoCaptureProperties.AutoExposure,1);
            File.WriteAllText(output,JsonSerializer.Serialize(new{rows,dual_raw=new[]{a.Result,b.Result},no_images_saved=true},new JsonSerializerOptions{WriteIndented=true,IncludeFields=true}));
        }
    }
    private static void DualMatrix(string output)
    {
        var cameras=CameraDevices.List().Where(c=>c.Name.Contains("C60E")).ToArray();
        if(cameras.Length!=2)throw new InvalidOperationException("Two cameras required.");
        var rows=new List<object>();
        foreach(var mode in new[]{(VideoCaptureAPIs.DSHOW,640,480,"MJPG",false),(VideoCaptureAPIs.DSHOW,640,480,"MJPG",true),(VideoCaptureAPIs.DSHOW,640,360,"MJPG",false)})
        {
            var first=mode.Item5?cameras[1]:cameras[0];var second=mode.Item5?cameras[0]:cameras[1];
            VideoCapture Open(int index)
            {
                var capture=new VideoCapture(index,mode.Item1);
                capture.Set(VideoCaptureProperties.FourCC,VideoWriter.FourCC(mode.Item4));
                capture.Set(VideoCaptureProperties.FrameWidth,mode.Item2);capture.Set(VideoCaptureProperties.FrameHeight,mode.Item3);capture.Set(VideoCaptureProperties.Fps,30);
                capture.Set(VideoCaptureProperties.FourCC,VideoWriter.FourCC(mode.Item4));
                capture.Set(VideoCaptureProperties.AutoExposure,0);capture.Set(VideoCaptureProperties.Exposure,-5);return capture;
            }
            using var left=Open(first.Index);using var right=Open(second.Index);
            object Measure(VideoCapture capture,int index)
            {
                using var frame=new Mat();var clock=Stopwatch.StartNew();int count=0;
                while(clock.Elapsed.TotalSeconds<6&&capture.Read(frame)&&!frame.Empty())count++;
                return new{index,opened=capture.IsOpened(),frames=count,measured_fps=count/clock.Elapsed.TotalSeconds,width=capture.FrameWidth,height=capture.FrameHeight,fourcc=capture.Get(VideoCaptureProperties.FourCC)};
            }
            var a=Task.Factory.StartNew(()=>Measure(left,first.Index),CancellationToken.None,TaskCreationOptions.LongRunning,TaskScheduler.Default);
            var b=Task.Factory.StartNew(()=>Measure(right,second.Index),CancellationToken.None,TaskCreationOptions.LongRunning,TaskScheduler.Default);
            Task.WhenAll(a,b).GetAwaiter().GetResult();
            rows.Add(new{backend=mode.Item1.ToString(),requested_width=mode.Item2,requested_height=mode.Item3,codec=mode.Item4,reversed=mode.Item5,first=a.Result,second=b.Result});
            File.WriteAllText(output,JsonSerializer.Serialize(new{rows,no_images_saved=true},new JsonSerializerOptions{WriteIndented=true}));
            left.Set(VideoCaptureProperties.AutoExposure,1);right.Set(VideoCaptureProperties.AutoExposure,1);
        }
    }
}
