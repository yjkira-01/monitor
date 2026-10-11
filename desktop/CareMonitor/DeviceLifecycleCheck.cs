using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using CareMonitor.Services;
using OpenCvSharp;
namespace CareMonitor;
public static class DeviceLifecycleCheck
{
    public static void Run(string[] args)=>Task.Run(()=>RunAsync(args)).GetAwaiter().GetResult();
    private static async Task RunAsync(string[] args)
    {
        string path=args[Array.IndexOf(args,"--lifecycle-video")+1],output=args[Array.IndexOf(args,"--eval-output")+1];
        await using var service=new VideoService(Path.Combine(AppContext.BaseDirectory,"Assets"),args.Contains("--gpu"),true,"gmd");await service.InitializeAsync();
        for(int i=0;i<3;i++)
        {
            await Task.WhenAll(service.StartFileAsync(path),service.StopAsync(),service.StartFileAsync(path),service.StopAsync());
            if(service.Latest is not null||service.Analysis is not null||service.Observation is not null)throw new InvalidOperationException("Concurrent source switch retained old data.");
        }
        using var source=new VideoCapture(path);using var frame=new Mat();source.Set(VideoCaptureProperties.PosMsec,2000);if(!source.Read(frame))throw new IOException("Cannot decode public test frame.");Cv2.ImEncode(".jpg",frame,out byte[] jpeg);
        using var listener=new TcpListener(IPAddress.Loopback,0);listener.Start();int port=((IPEndPoint)listener.LocalEndpoint).Port;
        var producer=Task.Run(async()=>
        {
            using var client=await listener.AcceptTcpClientAsync();using var stream=client.GetStream();
            await stream.WriteAsync(Encoding.ASCII.GetBytes("HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\nConnection: close\r\n\r\n"));
            for(int i=0;i<45;i++)
            {
                await stream.WriteAsync(Encoding.ASCII.GetBytes($"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {jpeg.Length}\r\n\r\n"));await stream.WriteAsync(jpeg);await stream.WriteAsync(Encoding.ASCII.GetBytes("\r\n"));await Task.Delay(200);
            }
        });
        await service.StartNetworkAsync($"http://127.0.0.1:{port}/public-test.mjpg");bool observed=false,ended=false;var clock=Stopwatch.StartNew();
        while(clock.Elapsed.TotalSeconds<18)
        {
            await Task.Delay(30);observed|=service.Analysis?.SceneScore.HasValue==true;
            if(service.Latest?.Ended==true){ended=true;break;}
        }
        await producer.WaitAsync(TimeSpan.FromSeconds(5));
        bool cleared=service.Analysis is null&&service.Observation is null;
        await service.StopAsync();bool stopped=service.Latest is null&&service.Analysis is null;
        bool passed=observed&&ended&&cleared&&stopped;
        File.WriteAllText(output,JsonSerializer.Serialize(new{passed,concurrent_switch_cycles=3,network_results_updated=observed,network_disconnect_detected=ended,old_analysis_cleared=cleared,stop_clears_source=stopped,public_fixture_only=true},new JsonSerializerOptions{WriteIndented=true}));
        if(!passed)throw new InvalidOperationException("Source lifecycle regression failed.");
    }
}
