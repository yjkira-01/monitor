using System.Diagnostics;
using System.IO;
using System.Text.Json;
using CareMonitor.Models;
using CareMonitor.Services;
namespace CareMonitor;
public static class DeviceCheck
{
    public static void Run(string[] args)=>Task.Run(()=>RunAsync(args)).GetAwaiter().GetResult();
    private static async Task RunAsync(string[] args)
    {
        string output=args[Array.IndexOf(args,"--eval-output")+1];string assets=Path.Combine(AppContext.BaseDirectory,"Assets");
        var devices=CameraDevices.List();var cameras=devices.Where(c=>c.Name.Contains("C60E")).ToArray();
        if(cameras.Length!=2||cameras[0].DeviceId==cameras[1].DeviceId)throw new InvalidOperationException("Two uniquely identified C60E cameras required.");
        var ports=RadarDevices.Detect()??throw new InvalidOperationException("SR112 port pair not found.");
        var firmware=await RadarSetup.IdentifyAsync(ports.Control,CancellationToken.None);if(!firmware.Supported)throw new InvalidOperationException("Radar firmware unsupported.");
        var settings=new AppSettings{CameraIndex=cameras[0].Index,CameraDeviceId=cameras[0].DeviceId,CameraDeviceName=cameras[0].Name,SecondaryCameraIndex=cameras[1].Index,SecondaryCameraId=cameras[1].DeviceId,RadarControlPort=ports.Control,RadarPort=ports.Data,UseGpu=args.Contains("--gpu")};
        settings.ManualExposure=true;
        await using var radar=new RadarService();await radar.ConnectAsync(ports.Data,921600);
        int commands=0;if(args.Contains("--configure-radar"))commands=await RadarSetup.ConfigureAsync(ports.Control,RadarSetup.Configuration(assets,settings),CancellationToken.None);
        await using var left=new VideoService(assets,settings.UseGpu,true,"gmd");await left.InitializeAsync();
        await using var right=new VideoService(assets,settings.UseGpu,true,"gmd",left);await right.InitializeAsync();
        await left.StartCameraAsync(settings);await right.StartCameraAsync(new(){CameraIndex=cameras[1].Index,CameraWidth=640,CameraHeight=480,CameraFps=30,ManualExposure=true,Exposure=-5,Gain=64});
        var pairer=new StereoPairing();var watch=Stopwatch.StartNew();int pairs=0,leftObservations=0,rightObservations=0;long ls=-1,rs=-1,ps=-1;var skews=new List<double>();var observedTypes=new HashSet<uint>();int maxPoints=0,maxTracks=0,vitalFrames=0,heightFrames=0,presenceFrames=0;uint? radarSeq=null;
        while(watch.Elapsed.TotalSeconds<18)
        {
            await Task.Delay(20);var l=left.Observation;var r=right.Observation;
            if(l is not null&&l.Frame.Sequence!=ls){ls=l.Frame.Sequence;leftObservations++;}
            if(r is not null&&r.Frame.Sequence!=rs){rs=r.Frame.Sequence;rightObservations++;}
            var p=pairer.Update(l,r);if(p is not null&&p.LeftSequence!=ps){ps=p.LeftSequence;pairs++;skews.Add(p.PairDeltaMs);}
            if(radar.Latest is {} frame&&radarSeq!=frame.Frame.Number){radarSeq=frame.Frame.Number;foreach(uint type in frame.Frame.TlvTypes??[])observedTypes.Add(type);maxPoints=Math.Max(maxPoints,frame.Frame.Points.Length);maxTracks=Math.Max(maxTracks,frame.Frame.Tracks.Length);if(frame.Frame.Vitals.Length>0)vitalFrames++;if(frame.Frame.Heights?.Length>0)heightFrames++;if(frame.Frame.Presence.HasValue)presenceFrames++;}
        }
        var lf=left.Latest;var rf=right.Latest;bool passed=lf is {Ended:false}&&rf is {Ended:false}&&lf.Fps>15&&rf.Fps>15&&leftObservations>10&&rightObservations>10&&pairs>=3&&radar.Diagnostics.Frames>10;
        if(passed&&args.Contains("--persist-device-settings"))
        {
            var store=new AppStore();var s=store.Data.Settings;s.ModelProfile="realtime";s.FallModel="gmd";s.CameraDeviceId=cameras[0].DeviceId;s.CameraDeviceName=cameras[0].Name;s.CameraIndex=cameras[0].Index;s.SecondaryCameraId=cameras[1].DeviceId;s.SecondaryCameraIndex=cameras[1].Index;s.CameraWidth=640;s.CameraHeight=480;s.CameraFps=30;s.ManualExposure=true;s.Exposure=-5;s.Gain=64;s.DeviceSetupRevision=1;s.RadarPort=ports.Data;s.RadarControlPort=ports.Control;s.RadarBaud=921600;store.Save();
        }
        File.WriteAllText(output,JsonSerializer.Serialize(new{passed,cameras,ports,firmware_compatible=firmware.Supported,configuration_commands=commands,live_seconds=watch.Elapsed.TotalSeconds,
            left_fps=lf?.Fps,right_fps=rf?.Fps,left_capture_status=left.CaptureStatus,right_capture_status=right.CaptureStatus,leftObservations,rightObservations,software_pairs=pairs,pair_skew_ms=skews,models=left.ModelStatus,radar=radar.Diagnostics,
            observed_tlv=observedTypes.Order().ToArray(),maxPoints,maxTracks,vitalFrames,heightFrames,presenceFrames,no_images_saved=true,calibrated_3d=false,
            shared_person_model=true},new JsonSerializerOptions{WriteIndented=true}));
        if(!passed)throw new InvalidOperationException("Physical-device throughput check failed; see evidence.");
    }
}
