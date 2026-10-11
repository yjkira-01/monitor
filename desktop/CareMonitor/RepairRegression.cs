using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using CareMonitor.Models;
using CareMonitor.Services;

namespace CareMonitor;

public static class RepairRegression
{
    private static void Require(bool condition,string message){if(!condition)throw new InvalidOperationException("Repair regression: "+message);}
    public static void Run(Dictionary<string,object> checks)
    {
        string text="**当前概况（截至某时刻）**\n- 视频为 **文件回放**。\n- 第一项\n\n## 优先事项\n1. 核查记录\n2. 复核画面\n\n建议\n<script>alert(1)</script>";
        var layout=ReportLayout.Parse(text);Require(layout.Count(s=>s.Kind=="heading")==3&&layout.Count(s=>s.Kind=="item")==4,"legacy markdown titles and lists");
        string plain=ReportLayout.PlainText(text),html=ReportLayout.Html(text);
        Require(!plain.Contains("**")&&!plain.Contains("##")&&html.Contains("<strong>文件回放</strong>")&&html.Contains("&lt;script&gt;")&&!html.Contains("<script>"),"plain copy and HTML escaping");
        Require(ReportService.Endpoint("https://example.com").AbsolutePath=="/v1/chat/completions"&&
            ReportService.Endpoint("https://example.com/api/v1/").AbsolutePath=="/api/v1/chat/completions"&&
            ReportService.Endpoint("https://example.com/v1/chat/completions").AbsolutePath=="/v1/chat/completions","API route normalization");
        Task.Run(CheckApi).GetAwaiter().GetResult();checks["report_format_api_routes_errors_and_cancellation"]=true;
        var tracker=new StablePeopleTracker();
        StablePeopleTracker.Observation Person(float x,float y,float[] color)=>new(new(x,y,x+.2f,y+.6f),.9f,[],color);
        int first=tracker.Update([Person(.2f,.1f,[1,0])],0).Single().Id;
        tracker.Update([],0.4);
        Require(tracker.Update([Person(.24f,.12f,[1,0])],1.1).Single().Id==first,"single-person disappearance keeps identity");
        var pair=new StablePeopleTracker();var start=pair.Update([Person(.1f,.1f,[1,0]),Person(.5f,.1f,[0,1])],0);
        var middle=pair.Update([Person(.24f,.1f,[1,0]),Person(.36f,.1f,[0,1])],.2);
        var crossing=pair.Update([Person(.36f,.1f,[1,0]),Person(.24f,.1f,[0,1])],.4);
        Require(start.Length==2&&middle.Length==2&&crossing.Length==2&&crossing[0].Id==start[0].Id&&crossing[1].Id==start[1].Id,"two crossing targets remain separate");
        Require(pair.Update([Person(.36f,.1f,[0,0,1])],4).Single().Id!=start[0].Id,"expired identity is not reused");checks["session_identity_gap_crossing_and_expiry"]=true;
        var shortLegs=new Joint[17];foreach(int i in new[]{5,6})shortLegs[i]=new(.5f,.1f,0,.95f);
        foreach(int i in new[]{11,12})shortLegs[i]=new(.5f,.4f,0,.95f);foreach(int i in new[]{13,14})shortLegs[i]=new(.5f,.54f,0,.95f);foreach(int i in new[]{15,16})shortLegs[i]=new(.5f,.68f,0,.95f);
        Require(PostureObservation.Classify(shortLegs,new(.35f,.02f,.65f,.8f),aspect:16f/9).Activity=="standing","perspective-shortened extended legs remain upright");
        Require(!VideoService.Assessment(.9f,3,null,false,"standing",false,.95f).Contains("跌倒")&&VideoService.Assessment(.9f,3,null,false,"lying",false,.95f).Contains("跌倒")&&VideoService.Assessment(.9f,3,null,true,"unknown").Contains("跌倒"),"upright disagreement never suppresses lying or incomplete-body candidates");
        checks["perspective_upright_counterevidence_preserves_raw_score"]=true;
        Require(!VideoService.Assessment(.51f,1,null,true,"unknown").Contains("跌倒")&&VideoService.Assessment(.51f,2,null,true,"unknown").Contains("跌倒"),"first candidate waits for temporal confirmation");
        Require(VideoService.Assessment(.6f,1,null,false,"lying",false,.9f).Contains("跌倒")&&!VideoService.Assessment(.1f,1,null,false,"lying",false,.9f).Contains("跌倒"),"agreement can confirm early but lying alone is never a fall");
        Require(VideoService.Assessment(.1f,0,null,false,"lying",false,.9f,.8f,2).Contains("跌倒"),"two recent video windows with observed horizontal posture recover a diluted long window");
        Require(!VideoService.Assessment(.1f,0,null,false,"standing",false,.9f,.9f,3).Contains("跌倒")&&
            !VideoService.Assessment(.1f,0,null,true,"unknown",false,0,.9f,3).Contains("跌倒")&&
            !VideoService.Assessment(.1f,0,null,false,"lying",false,.9f,.8f,1).Contains("跌倒"),"short window alone, unsupported or unconfirmed never becomes a fall");
        checks["supported_short_window_confirmation_without_upright_false_alarm"]=true;
        Joint[] upright=(Joint[])shortLegs.Clone();foreach(int i in new[]{13,14})upright[i]=new(.5f,.65f,0,.95f);foreach(int i in new[]{15,16})upright[i]=new(.5f,.9f,0,.95f);
        Joint[] horizontal=(Joint[])upright.Clone();foreach(int i in new[]{5,6})horizontal[i]=new(.25f,.5f,0,.95f);foreach(int i in new[]{11,12})horizontal[i]=new(.6f,.5f,0,.95f);foreach(int i in new[]{13,14})horizontal[i]=new(.75f,.52f,0,.95f);foreach(int i in new[]{15,16})horizontal[i]=new(.8f,.6f,0,.95f);
        var ground=new PostureTimeline();ground.Update(upright,new(.35f,.02f,.65f,.95f),0,1);ground.Update(horizontal,new(.2f,.45f,.85f,.8f),.5,1);
        Require(ground.ShortWindowSupported&&!ground.FallMotion,"lateral collapse needs video verification instead of inventing a hip-drop alert");
        var bed=new PostureTimeline();bed.Update(upright,new(.35f,.02f,.65f,.95f),0,1);bed.Update(horizontal,new(.2f,.45f,.85f,.66f),.5,1);
        Require(!bed.ShortWindowSupported,"bed-level lie-down lacks previous support-foot position");
        ground.Reset();ground.Update(horizontal,new(.2f,.45f,.85f,.8f),1,1);Require(!ground.ShortWindowSupported,"lying alone and a source reset cannot invent previous motion");
        checks["recent_motion_requires_observed_support_position_and_resets"]=true;
        var zone=new MonitoringZone(Guid.NewGuid(),"床边","fixture",new(.1f,.2f,.7f,.6f),1);
        Require(MonitoringAnalytics.ZoneFor(new(.15f,.3f,.65f,.7f),[zone],"fixture")?.Id==zone.Id,"lying body overlap enters zone even when lower box edge is outside");
        Require(MonitoringAnalytics.ZoneFor(new(.15f,.3f,.65f,.7f),[zone],"other") is null,"zones isolated by source");
        var stats=new MonitoringAnalytics();var events=new List<MonitorEvent>();
        AnalysisSnapshot Snapshot(double time)=>new(1,time,[new(1,new(.2f,.1f,.4f,.5f),.9f,[],[],time,Activity:"standing")],0,"adl","",0,"fixture");
        VideoSnapshot Video(double time)=>new([],640,480,time,10,30,true,false,false,"fixture",1,(long)(time*10));
        stats.Update(Snapshot(0),Video(0),[zone],120,events.Add);
        var gap=stats.Update(Snapshot(2),Video(2),[zone],120,events.Add);
        Require(gap.People[0].Telemetry!.DwellSeconds==0&&!events.Any(e=>e.Kind=="区域停留提醒"),"unobserved gap does not accrue dwell");
        stats.InvalidateZones();var removed=stats.Update(Snapshot(2.1),Video(2.1),[],120,events.Add);
        Require(removed.People[0].Telemetry!.Zone=="区域外"&&removed.People[0].Telemetry!.DwellSeconds==0,"removed zone loses dwell state");checks["zone_overlap_gap_and_removal"]=true;
    }
    private static async Task CheckApi()
    {
        foreach(var fixture in new[]{(Status:200,Array:false),(Status:401,Array:false),(Status:200,Array:true)})
        {
            int status=fixture.Status;
            using var listener=new TcpListener(IPAddress.Loopback,0);listener.Start();int port=((IPEndPoint)listener.LocalEndpoint).Port;
            using var timeout=new CancellationTokenSource(TimeSpan.FromSeconds(5));
            var server=Task.Run(async()=>
            {
                using var client=await listener.AcceptTcpClientAsync(timeout.Token);using var stream=client.GetStream();
                var bytes=new List<byte>();byte[] buffer=new byte[4096];
                while(true)
                {
                    int n=await stream.ReadAsync(buffer,timeout.Token);if(n==0)break;bytes.AddRange(buffer.Take(n));string request=Encoding.UTF8.GetString(bytes.ToArray());int at=request.IndexOf("\r\n\r\n",StringComparison.Ordinal);
                    if(at<0)continue;var line=request[..at].Split("\r\n").First(l=>l.StartsWith("Content-Length:",StringComparison.OrdinalIgnoreCase));
                    if(bytes.Count<at+4+int.Parse(line.Split(':')[1]))continue;
                    Require(request.StartsWith("POST /v1/chat/completions ")&&!request.Contains("住户")&&!request.Contains("视频来源"),"connection test sends no monitoring data");break;
                }
                string body=status==401?"{}":!fixture.Array?"{\"choices\":[{\"message\":{\"content\":\"<think>内部思考</think>连接成功\"}}]}":"{\"choices\":[{\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"连接成功\"}]}}]}";
                byte[] encoded=Encoding.UTF8.GetBytes(body),header=Encoding.ASCII.GetBytes($"HTTP/1.1 {status} Result\r\nContent-Length: {encoded.Length}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n");
                await stream.WriteAsync(header,timeout.Token);await stream.WriteAsync(encoded,timeout.Token);
            });
            using var reports=new ReportService();var settings=new AppSettings{ApiUrl=$"http://127.0.0.1:{port}/v1/chat/completions",ApiModel="fixture"};
            bool failed=false;try{await reports.TestAsync(settings,timeout.Token);}catch(InvalidOperationException ex){failed=ex.Message.Contains("拒绝访问");}
            await server;Require(status==401?failed:!failed,"meaningful unauthorized feedback and structured content");
        }
        using var canceled=new CancellationTokenSource();canceled.Cancel();using var service=new ReportService();
        bool cancellation=false;try{await service.TestAsync(new(){ApiUrl="http://127.0.0.1:1/v1",ApiModel="fixture"},canceled.Token);}catch(OperationCanceledException){cancellation=true;}
        Require(cancellation,"user cancellation propagates");
    }
}
