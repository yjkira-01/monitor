using System.Buffers.Binary;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using CareMonitor.Models;
using CareMonitor.Services;
using CareMonitor.Views;
using OpenCvSharp;

namespace CareMonitor;
public static class SelfCheck
{
    private static void Assert(bool value,string message){if(!value)throw new InvalidOperationException(message);}
    public static void Run(string[] args)
    {
        if(args.Contains("--temporal-check")){TemporalCheck.Run(args);return;}
        if(args.Contains("--recognition-replay")){RecognitionCheck.Run(args);return;}
        if(args.Contains("--device-check")){DeviceCheck.Run(args);return;}
        if(args.Contains("--lifecycle-video")){DeviceLifecycleCheck.Run(args);return;}
        if(args.Contains("--camera-capabilities")){CameraCheck.Run(args);return;}
        if(args.Contains("--people-fixtures")){PeopleCheck.Run(args);return;}
        if(args.Contains("--evaluation-list")){EvaluateList(args);return;}
        string assets=Path.Combine(AppContext.BaseDirectory,"Assets");var checks=new Dictionary<string,object>();
        DeviceRegression.Run(checks);
        RepairRegression.Run(checks);
        byte[] packet=new byte[64];new byte[]{2,1,4,3,6,5,8,7}.CopyTo(packet,0);
        void U(int at,uint value)=>BinaryPrimitives.WriteUInt32LittleEndian(packet.AsSpan(at),value);
        void F(int at,float value)=>BinaryPrimitives.WriteSingleLittleEndian(packet.AsSpan(at),value);
        U(12,64);U(20,9);U(28,1);U(32,1);U(40,1);U(44,16);F(48,1);F(52,2);F(56,3);F(60,0);
        var parser=new RadarParser();Assert(parser.Feed([0,9,0]).Count==0,"noise parser");Assert(parser.Feed(packet.AsSpan(0,23)).Count==0,"partial parser");
        var frame=parser.Feed(packet.AsSpan(23));Assert(frame.Count==1&&frame[0].Number==9&&frame[0].Points[0]==new Vector3(1,2,3),"fragmented parser");
        packet[44]=255;Assert(RadarParser.Parse(packet) is null,"bad TLV rejected");checks["radar_fragment_and_corruption"]=true;
        var store=new AppStore(Path.Combine(AppContext.BaseDirectory,"self-test-data"));store.Data.Residents.Clear();
        var resident=new Resident{Name="离线自检档案",Room="自检",Bed="A"};store.SaveResident(resident);
        var reopened=new AppStore(store.DirectoryPath);Assert(reopened.Data.Residents.Count==1,"persistence roundtrip");
        bool duplicate=false;try{reopened.SaveResident(new(){Name="重复测试",Room="自检",Bed="A"});}catch(ArgumentException){duplicate=true;}Assert(duplicate,"duplicate bed validation");
        string secret=AppStore.ProtectKey("self-check-only");Assert(secret!="self-check-only"&&AppStore.UnprotectKey(secret)=="self-check-only","DPAPI encryption");checks["store_and_key"]=true;
        Assert(VideoService.Assessment(.1f,0,.9f,true,"adl")=="姿态异常 · 请核查","pose evidence is separate from binary classifier");
        Assert(VideoService.Assessment(null,0,.9f,false,"")=="姿态异常 · 请核查","pose warning before RGB window");
        var coordinates=new float[195];Array.Fill(coordinates,128);var heatmap=new float[64*64*39];Array.Fill(heatmap,-80);
        heatmap[(33*64+34)*39]=80;NativeModels.Refine(coordinates,heatmap);Assert(Math.Abs(coordinates[0]-136)<.01&&Math.Abs(coordinates[1]-132)<.01,"heatmap landmark refinement");
        byte[] vitalPacket=new byte[304];new byte[]{2,1,4,3,6,5,8,7}.CopyTo(vitalPacket,0);
        void Vu(int at,uint v)=>BinaryPrimitives.WriteUInt32LittleEndian(vitalPacket.AsSpan(at),v);
        void Vf(int at,float v)=>BinaryPrimitives.WriteSingleLittleEndian(vitalPacket.AsSpan(at),v);
        Vu(12,304);Vu(20,11);Vu(32,2);Vu(40,1010);Vu(44,112);Vu(48,7);Vf(52,1);Vf(56,2);Vf(60,3);Vf(156,.8f);Vu(160,1040);Vu(164,136);
        BinaryPrimitives.WriteUInt16LittleEndian(vitalPacket.AsSpan(168),7);Vf(172,.3f);Vf(176,71);Vf(180,15);
        var vitalFrame=RadarParser.Parse(vitalPacket);Assert(vitalFrame?.Tracks.Single().Id==7&&vitalFrame.Vitals.Single().HeartRate==71,"track and vital TLV schema");
        Vf(176,float.NaN);Assert(RadarParser.Parse(vitalPacket) is null,"nonfinite vital rejected");checks["tracks_and_vitals"]=true;
        foreach(bool withKey in new[]{true,false})
        {
            using var listener=new TcpListener(IPAddress.Loopback,0);
            listener.Start();int port=((IPEndPoint)listener.LocalEndpoint).Port;
            var server=Task.Run(async()=>
            {
                using var client=await listener.AcceptTcpClientAsync();using var stream=client.GetStream();using var timeout=new CancellationTokenSource(TimeSpan.FromSeconds(5));
                var all=new List<byte>();byte[] chunk=new byte[4096];int headerEnd=-1,length=0;
                while(true)
                {
                    int read=await stream.ReadAsync(chunk,timeout.Token);if(read==0)break;all.AddRange(chunk.Take(read));
                    string request=Encoding.UTF8.GetString(all.ToArray());headerEnd=request.IndexOf("\r\n\r\n",StringComparison.Ordinal);
                    if(headerEnd>=0)
                    {
                        string header=request[..headerEnd];string contentLength=header.Split("\r\n").First(l=>l.StartsWith("Content-Length:",StringComparison.OrdinalIgnoreCase));length=int.Parse(contentLength.Split(':')[1]);
                        if(all.Count>=Encoding.UTF8.GetByteCount(request[..(headerEnd+4)])+length)break;
                    }
                }
                string text=Encoding.UTF8.GetString(all.ToArray());Assert(text.StartsWith("POST /v1/chat/completions"),"compatible API route");
                Assert((withKey?text.Contains("Bearer self-check-only"):!text.Contains("Authorization:",StringComparison.OrdinalIgnoreCase))&&!text.Contains("离线自检档案"),"API auth and no identity leakage");
                byte[] body=Encoding.UTF8.GetBytes("{\"choices\":[{\"message\":{\"content\":\"测试报告：无有效来源\"}}]}");
                byte[] headerBytes=Encoding.ASCII.GetBytes($"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {body.Length}\r\nConnection: close\r\n\r\n");
                await stream.WriteAsync(headerBytes);await stream.WriteAsync(body);
            });
            store.Data.Settings.ApiUrl=$"http://127.0.0.1:{port}/v1";store.Data.Settings.ProtectedApiKey=withKey?secret:"";
            using var service=new ReportService();var report=Task.Run(()=>service.GenerateAsync(store.Data,null,null,null,true,CancellationToken.None)).GetAwaiter().GetResult();
            server.GetAwaiter().GetResult();Assert(report.Text.Contains("测试报告")&&report.Evidence.Contains("暂无视频识别结果"),"API response and observation evidence");checks[withKey?"compatible_api_loopback":"keyless_local_api_loopback"]=true;
        }
        using var fixture=JsonDocument.Parse(File.ReadAllText(Path.Combine(assets,"selfcheck.json")));var root=fixture.RootElement;
        var poses=root.GetProperty("poses").EnumerateArray().Select(t=>t.EnumerateArray().Select(j=>new Joint(j[0].GetSingle(),j[1].GetSingle(),0,j[2].GetSingle())).ToArray()).ToArray();
        var expected=root.GetProperty("features").EnumerateArray().Select(x=>x.GetSingle()).ToArray();var features=PoseFeatures.Build(poses,root.GetProperty("aspect").GetSingle());
        float error=features.Zip(expected,(a,b)=>Math.Abs(a-b)).Max();Assert(error<2e-5,"pose feature Python/C# parity: "+error);checks["pose_features_max_error"]=error;
        CheckTrackingAndAnalytics(checks);
        CheckPostureTimeline(checks);
        var cameraDevices=CameraDevices.List();checks["camera_names"]=cameraDevices.Select(c=>c.ToString()).ToArray();
        string fallModel=args.Contains("--fall-model")?args[Array.IndexOf(args,"--fall-model")+1]:"stage_c";
        using var binaryFixture=fallModel=="gmd"?JsonDocument.Parse(File.ReadAllText(Path.Combine(assets,"Binary/gmd/selfcheck.json"))):null;
        var videoFixture=binaryFixture?.RootElement??root;
        using var models=new NativeModels(assets,args.Contains("--gpu"),args.Contains("--realtime"),fallModel);
        var clips=Enumerable.Range(0,16).Select(t=>Enumerable.Range(0,models.FrameWidth*models.FrameHeight*3).Select(i=>(byte)((i+t*7)%251)).ToArray()).ToArray();
        var inferenceWatch=Stopwatch.StartNew();var scores=models.VideoScores(clips);checks["video_first_inference_ms"]=inferenceWatch.Elapsed.TotalMilliseconds;
        var expectedScores=videoFixture.GetProperty("video_scores").EnumerateArray().Select(x=>x.GetSingle()).ToArray();
        Assert(scores.Length==expectedScores.Length,"video class count parity");
        float videoError=scores.Zip(expectedScores,(a,b)=>Math.Abs(a-b)).Max();Assert(videoError<.005,"video ONNX/C# parity: "+videoError);checks["video_max_error"]=videoError;checks["device"]=models.Device;
        checks["video_model"]=models.VideoName;checks["detector"]=models.DetectorName;
        inferenceWatch.Restart();var diagnostic=models.PredictVideo(clips);checks["video_warm_inference_ms"]=inferenceWatch.Elapsed.TotalMilliseconds;
        Assert(diagnostic.Embedding.Length==models.EmbeddingSize&&diagnostic.Attention.Length>0&&diagnostic.Attention.All(v=>v>=0&&v<=1),"model spatial evidence");checks["attention_cells"]=diagnostic.Attention.Length;
        var dark=clips.Select((b,t)=>b.Select(v=>(byte)(v*(.08+.02*t))).ToArray()).ToArray();var night=models.VideoScores(dark);
        float nightError=night.Zip(videoFixture.GetProperty("night_scores").EnumerateArray().Select(x=>x.GetSingle()),(a,b)=>Math.Abs(a-b)).Max();Assert(nightError<.005,"dark preprocessing parity: "+nightError);checks["night_preprocessing_max_error"]=nightError;
        checks["app_local_vc_runtime"]=Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Where(m=>m.ModuleName.StartsWith("vcruntime140",StringComparison.OrdinalIgnoreCase)).Select(m=>m.FileName).ToArray();
        using var empty=new Mat(360,640,MatType.CV_8UC3,Scalar.All(0));var boxes=models.Detect(empty);Assert(boxes.All(b=>b.Box.X1>=0&&b.Box.X2<=1),"detector bounds");
        var prepared=models.PrepareVideoFrame(empty);Assert(prepared.Length==models.FrameWidth*models.FrameHeight*3,"video frame dimensions");
        if(models.Preview)Assert(prepared[0]==104&&prepared[1]==116&&prepared[2]==124&&prepared[(112*224+112)*3]==0,"training letterbox geometry and BGR fill");
        var poseWatch=Stopwatch.StartNew();var joints=models.Pose(empty,new(0,0,1,1));checks["pose_inference_ms"]=poseWatch.Elapsed.TotalMilliseconds;
        Assert(joints.Image.Length==0||joints.Image.Length==(models.Realtime?17:33),"pose output contract");
        if(models.Realtime)
        {
            var synthetic=Enumerable.Range(0,16).Select(t=>Enumerable.Range(0,17).Select(j=>new Joint(.2f+j*.03f,.1f+j*.035f,0,.8f)).ToArray()).ToArray();
            var actionWatch=Stopwatch.StartNew();var actionScores=models.ActionScores(synthetic);checks["action_inference_ms"]=actionWatch.Elapsed.TotalMilliseconds;
            if(actionScores is not null){Assert(actionScores.Length==60&&Math.Abs(actionScores.Sum()-1)<1e-4,"action output contract");checks["action_candidates"]=PretrainedActions.Top(actionScores);}
        }
        float poseScore=models.PoseRisk(poses,root.GetProperty("aspect").GetSingle());Assert(float.IsFinite(poseScore),"TCN output");checks["pose_score"]=poseScore;
        if(args.Contains("--video"))
        {
            int index=Array.IndexOf(args,"--video");string path=args[index+1];using var capture=new VideoCapture(path);Assert(capture.IsOpened(),"video opens");
            List<byte[]> frames=[];var observations=new List<object>();var sw=Stopwatch.StartNew();
            for(int i=0;i<120;i++)
            {
                using var mat=new Mat();if(!capture.Read(mat))break;
                frames.Add(models.PrepareVideoFrame(mat));
                if(i%20==0)
                {
                    var people=models.Detect(mat);foreach(var b in people.Take(4)){var pose=models.Pose(mat,b.Box);observations.Add(new{frame=i,b.Box,b.Score,joints=pose.Image.Length});}
                }
            }
            Assert(frames.Count>=16,"video decoded");var predictions=models.VideoScores(frames.TakeLast(48).ToList());
            checks["public_video"]=new{frames=frames.Count,elapsed_ms=sw.Elapsed.TotalMilliseconds,predictions,observations};
        }
        var providers=models.KernelProviders();checks["kernel_providers"]=providers;
        if(args.Contains("--profile-models")&&args.Contains("--gpu"))Assert(providers.Values.All(counts=>counts.Keys.Any(name=>name.Contains("Dml",StringComparison.OrdinalIgnoreCase))),"DirectML kernels executed");
        checks["passed"]=true;File.WriteAllText(Path.Combine(AppContext.BaseDirectory,"self-test-result.json"),JsonSerializer.Serialize(checks,new JsonSerializerOptions{WriteIndented=true}));
    }
    private static void CheckPostureTimeline(Dictionary<string,object> checks)
    {
        Joint[] Upright(float shift=0)
        {
            var j=new Joint[17];
            foreach(int i in new[]{5,6})j[i]=new(.5f+shift,.2f,0,.9f);
            foreach(int i in new[]{11,12})j[i]=new(.5f+shift,.4f,0,.9f);
            foreach(int i in new[]{13,14})j[i]=new(.5f+shift,.6f,0,.9f);
            foreach(int i in new[]{15,16})j[i]=new(.5f+shift,.8f,0,.9f);
            return j;
        }
        var box=new Box(.3f,.1f,.7f,.9f);var standing=Upright();
        Assert(PostureObservation.Classify(standing,box,aspect:16f/9).Activity=="standing","upright posture");
        var sitting=Upright();foreach(int i in new[]{13,14})sitting[i]=new(.65f,.45f,0,.9f);foreach(int i in new[]{15,16})sitting[i]=new(.65f,.7f,0,.9f);
        Assert(PostureObservation.Classify(sitting,box,aspect:16f/9).Activity=="sitting","flexed knee sitting");
        var bending=Upright();foreach(int i in new[]{5,6})bending[i]=new(.35f,.35f,0,.9f);
        Assert(PostureObservation.Classify(bending,box,aspect:16f/9).Activity=="bending","bending with upright legs is not lying");
        var lying=Upright();foreach(int i in new[]{5,6})lying[i]=new(.3f,.65f,0,.9f);foreach(int i in new[]{11,12})lying[i]=new(.5f,.7f,0,.9f);
        foreach(int i in new[]{13,14})lying[i]=new(.65f,.7f,0,.9f);foreach(int i in new[]{15,16})lying[i]=new(.8f,.7f,0,.9f);
        Assert(PostureObservation.Classify(lying,new(.2f,.55f,.9f,.8f),aspect:16f/9).Activity=="lying","horizontal body");
        var partial=(Joint[])lying.Clone();foreach(int i in new[]{6,12,13,14,15,16})partial[i]=default;
        Assert(PostureObservation.Classify(partial,new(.2f,.55f,.9f,.8f),aspect:16f/9).Activity=="lying","one visible torso side can support lying");
        var close=new PeopleObservation(box,.8f,standing);var other=new PeopleObservation(new(.45f,.1f,.85f,.9f),.9f,Upright(.15f));
        Assert(RealtimePeople.SamePose(close,new(new(.4f,.1f,.6f,.9f),.7f,Upright(.001f)),16f/9),"nested duplicate boxes with matching joints are suppressed");
        Assert(!RealtimePeople.SamePose(close,other,16f/9),"nearby distinct bodies are not merged");
        foreach(double interval in new[]{.1,.2})
        {
            var timeline=new PostureTimeline();for(double t=0;t<1;t+=interval)timeline.Update(standing,box,t,16f/9);
            Assert(timeline.Activity=="standing","pose stabilization is time-based");
            timeline.Update(lying,new(.2f,.55f,.9f,.8f),1.05,16f/9);
            Assert(timeline.FallMotion,"rapid torso rotation and descent retained as separate fall evidence");
            timeline.Update(new Joint[17],box,2.0,16f/9);
            Assert(timeline.Activity=="unknown","missing pose expires instead of freezing old state");
            timeline.Reset();timeline.Update(standing,box,0,16f/9);
            Assert(!timeline.FallMotion&&timeline.Activity=="unknown","seek/reset clears prior motion evidence");
        }
        checks["posture_aspect_bending_partial_time_missing_duplicate_distinct"]=true;
    }
    private static void CheckTrackingAndAnalytics(Dictionary<string,object> checks)
    {
        var tracker=new ByteTrack.ByteTracker(new(){TrackThresh=.35f,TrackBuffer=60},4);
        ByteTrack.Detection D(double x,float score=.9f)=>new(x,100,x+150,400,score);
        var first=tracker.Update([D(100),D(600)]);Assert(first.Count==2,"two distinct tracks");
        int id=first.OrderBy(t=>t.Tlbr[0]).First().TrackId;
        var low=tracker.Update([D(108,.2f),D(596)]);Assert(low.Count==2&&low.Any(t=>t.TrackId==id),"low-score detection retains existing identity");
        tracker.Update([D(590)]);var returned=tracker.Update([D(116),D(585)]);
        Assert(returned.Count==2&&returned.Any(t=>t.TrackId==id),"short occlusion retains identity");
        var noGhost=new ByteTrack.ByteTracker(new(){TrackThresh=.35f},4);Assert(noGhost.Update([D(100,.2f)]).Count==0,"low-score detection cannot start a ghost track");
        var analytics=new MonitoringAnalytics();var events=new List<MonitorEvent>();
        var zone=new MonitoringZone(Guid.NewGuid(),"门口","public-fixture",new(0,0,.5f,1),2);
        var source=new VideoSnapshot([],640,480,0,0,30,true,false,false,"public-fixture",1,1);
        var person=new PersonState(5,new(.1f,.1f,.3f,.8f),.9f,[],[],0,Activity:"standing");
        AnalysisSnapshot Snap(double time,PersonState p)=>new(1,time,[p with{Time=time}],null,"","",0,"test");
        for(int i=0;i<12;i++)analytics.Update(Snap(i*.5,person),source with{Time=i*.5},[zone],2,events.Add);
        Assert(events.Count(e=>e.Kind=="进入区域")==1&&events.Count(e=>e.Kind=="区域停留提醒")==1&&events.Count(e=>e.Kind=="长时间静止提醒")==1,"zone and inactivity reminders deduplicated");
        var outside=analytics.Update(Snap(6,person with{Box=new(.6f,.1f,.8f,.8f)}),source with{Time=6},[zone],2,events.Add);
        Assert(outside.Analytics!.Entries==1&&outside.Analytics.Exits==1&&outside.Analytics.CurrentZoneOccupancy==0,"zone exit and occupancy");
        var gap=analytics.Update(Snap(20,person),source with{Time=20},[zone],2,events.Add);
        Assert(gap.People[0].Telemetry!.StillSeconds==0&&gap.People[0].Telemetry!.DwellSeconds==0,"occlusion is not counted as observed stillness or dwell");
        Assert(events.All(e=>e.IsReplay),"replay events stay separate");analytics.Reset();
        var reset=analytics.Update(Snap(0,person),source,[zone],2,events.Add);Assert(reset.Analytics!.ObservedSeconds==0&&reset.Analytics.UniquePeople==1,"generation statistics reset");
        Assert(PostureObservation.Classify(new Joint[17],new(0,0,1,1)).Activity=="unknown","missing keypoints do not produce a fabricated posture");
        Assert(NativeModels.ValidUnit(1.0000001f)&&!NativeModels.ValidUnit(1.01f)&&!NativeModels.ValidUnit(float.NaN)&&!NativeModels.ValidUnit(float.PositiveInfinity),"float32 rounding is tolerated while invalid output remains rejected");
        checks["tracking_low_confidence_occlusion_no_ghost"]=true;checks["zone_dwell_inactivity_reset"]=true;
    }
    private static void EvaluateList(string[] args)
    {
        string path=args[Array.IndexOf(args,"--evaluation-list")+1];using var list=JsonDocument.Parse(File.ReadAllText(path));
        string fallModel=args.Contains("--fall-model")?args[Array.IndexOf(args,"--fall-model")+1]:"stage_c";
        using var models=new NativeModels(Path.Combine(AppContext.BaseDirectory,"Assets"),args.Contains("--gpu"),true,fallModel);
        var rows=new List<object>();string output=args.Contains("--eval-output")?args[Array.IndexOf(args,"--eval-output")+1]:Path.Combine(AppContext.BaseDirectory,"self-test-evaluation.json");
        foreach(var item in list.RootElement.EnumerateArray())
        {
            string file=item.GetProperty("file").GetString()!;bool truth=item.GetProperty("fall").GetBoolean();
            using var cap=new VideoCapture(file);Assert(cap.IsOpened(),"public evaluation video opens");
            double fps=cap.Get(VideoCaptureProperties.Fps);int stride=Math.Max(1,(int)Math.Round(fps/8));
            var clock=Stopwatch.StartNew();int frame=0,samples=0,located=0,visible=0,maxPeople=0,actionHigh=0,rgbHigh=0;
            float? actionPeak=null,rgbPeak=null;var candidates=new List<object>();var scene=new List<byte[]>();
            var tracker=new ByteTrack.ByteTracker(new(){TrackThresh=.35f,TrackBuffer=60},8);
            var histories=new Dictionary<int,List<Joint[]>>();
            while(true)
            {
                using var decoded=new Mat();if(!cap.Read(decoded))break;
                if(frame++%stride!=0)continue;
                using var source=new Mat();if(decoded.Width>640)Cv2.Resize(decoded,source,new(640,(int)(decoded.Height*640d/decoded.Width)));else decoded.CopyTo(source);
                double time=(frame-1)/fps;samples++;
                var detections=args.Contains("--rgb-only")?Array.Empty<(Box Box,float Score)>():models.Detect(source);
                var tracks=tracker.Update(detections.Select(b=>new ByteTrack.Detection(b.Box.X1*1000,b.Box.Y1*1000,b.Box.X2*1000,b.Box.Y2*1000,b.Score)).ToArray());
                maxPeople=Math.Max(maxPeople,tracks.Count);if(tracks.Count>0)located++;
                foreach(var track in tracks.Take(4))
                {
                    var b=track.Tlbr;var box=new Box(Math.Clamp((float)b[0]/1000,0,1),Math.Clamp((float)b[1]/1000,0,1),Math.Clamp((float)b[2]/1000,0,1),Math.Clamp((float)b[3]/1000,0,1));
                    var pose=models.Pose(source,box).Image;if(pose.Length!=17)continue;
                    if(!histories.TryGetValue(track.TrackId,out var history)){history=[];histories[track.TrackId]=history;}
                    history.Add(pose);if(history.Count>32)history.RemoveAt(0);
                    if(new[]{5,6,11,12}.All(j=>pose[j].Confidence>.3))visible++;
                    if(history.Count<8||samples%4!=0)continue;
                    var scores=models.ActionScores(history.ToArray());if(scores is null)continue;
                    float score=scores[42];actionPeak=Math.Max(actionPeak??0,score);if(score>=.5)actionHigh++;
                    candidates.Add(new{time,id=track.TrackId,fall_score=score,top=PretrainedActions.Top(scores)});
                }
                scene.Add(models.PrepareVideoFrame(source));if(scene.Count>32)scene.RemoveAt(0);
                if(scene.Count>=8&&samples%4==0){float risk=models.Risk(models.VideoScores(scene));rgbPeak=Math.Max(rgbPeak??0,risk);if(risk>=.5)rgbHigh++;}
            }
            rows.Add(new{file,truth_fall=truth,samples,detected_frames=located,visible_torso_observations=visible,max_people=maxPeople,
                baseline_peak=rgbPeak,baseline_predicted_fall=rgbHigh>=2,stgcn_peak=actionPeak,stgcn_predicted_fall=actionHigh>=2,
                stgcn_available=actionPeak.HasValue,elapsed_seconds=clock.Elapsed.TotalSeconds,action_observations=candidates});
            File.WriteAllText(output,JsonSerializer.Serialize(new{state="running",completed=rows.Count,total=list.RootElement.GetArrayLength(),rows},new JsonSerializerOptions{WriteIndented=true}));
        }
        File.WriteAllText(output,JsonSerializer.Serialize(new{state="complete",device=models.Device,
            model=models.VideoName,detector=models.DetectorName,rows,notes="Known public GMD validation cases; not independent elderly-scene validation. Whole-video labels and two >=0.5 windows; no event onset accuracy measured."},new JsonSerializerOptions{WriteIndented=true}));
    }
}
