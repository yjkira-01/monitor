using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net;
using System.Text;
using System.Text.Json;
using CareMonitor.Models;

namespace CareMonitor.Services;
public sealed class ReportService : IDisposable
{
    private readonly HttpClient http=new(new HttpClientHandler{AllowAutoRedirect=false}){Timeout=Timeout.InfiniteTimeSpan};
    public static bool HasCredentials(AppSettings settings)=>settings.ProtectedApiKey.Length>0||
        Uri.TryCreate(settings.ApiUrl,UriKind.Absolute,out var uri)&&uri.IsLoopback;
    public static void Validate(AppSettings settings)
    {
        if(!Uri.TryCreate(settings.ApiUrl,UriKind.Absolute,out var url)||url.UserInfo.Length>0||url.Query.Length>0||url.Fragment.Length>0 ||
            (url.Scheme!="https" && !(url.Scheme=="http" && url.IsLoopback)))
            throw new ArgumentException("API地址须使用HTTPS；本机服务可使用HTTP，不要把密钥填入地址。");
        if(string.IsNullOrWhiteSpace(settings.ApiModel)||settings.ApiModel.Length>200)throw new ArgumentException("请填写有效模型名。");
        if(settings.ReportIntervalSeconds<60||settings.ReportIntervalSeconds>3600)throw new ArgumentException("报告间隔须在60–3600秒之间。");
    }
    public static Uri Endpoint(string address)
    {
        var uri=new Uri(address.Trim());string path=uri.AbsolutePath.TrimEnd('/');
        if(path.EndsWith("/chat/completions",StringComparison.OrdinalIgnoreCase))return uri;
        if(path.Length==0)path="/v1";
        return new UriBuilder(uri){Path=path+"/chat/completions"}.Uri;
    }
    public static string Local(StoreData data,VideoSnapshot? video,AnalysisSnapshot? result,RadarSnapshot? radar)
    {
        var sb=new StringBuilder();sb.AppendLine($"健康监测报告 · {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        var pending=data.Events.Where(e=>!e.IsReplay&&e.Resolution=="待处理").ToArray();
        sb.AppendLine($"住户档案 {data.Residents.Count} 人；待处理现场告警 {pending.Length} 条，其中最近24小时 {pending.Count(e=>DateTimeOffset.Now-e.Time<TimeSpan.FromHours(24))} 条。历史待处理记录不等于当前现场异常。");
        foreach(var item in pending.OrderByDescending(e=>e.Time).Take(8))sb.AppendLine($"待处理记录：{item.Time.LocalDateTime:MM-dd HH:mm:ss}，{item.Kind}，编号 {item.Subject}。未核查记录不能认定为已发生的事件。");
        sb.AppendLine();sb.AppendLine("视频监控");
        if(video is null)sb.AppendLine("视频未连接，暂无视频识别结果。");
        else
        {
            sb.AppendLine($"视频来源：{video.Source}（{(video.Replay?"视频回放":"现场摄像头")}），播放位置 {video.Time:F2} 秒。"+(video.Paused?"播放已暂停。":""));
            if(result is null||result.Generation!=video.Generation||video.Time-result.Time>2)sb.AppendLine("识别结果更新中，暂无有效判断。");
            else
            {
                sb.AppendLine($"当前识别 {result.People.Length} 人；画面状态：{VideoService.SceneAssessment(result)}；更新位置 {result.Time:F2} 秒。");
                foreach(var person in result.People)sb.AppendLine($"P{person.Id:00}：{person.Status}；画面{(person.Partial?"不完整":"完整")}；长窗口视频分值 {person.RgbScore?.ToString("F3")??"尚未获取"}；最近2秒视频分值 {person.ShortRgbScore?.ToString("F3")??"未复核"}（连续{person.ShortConsecutive}窗）；姿态分析分值 {person.PoseScore?.ToString("F3")??"未提供"}。"+(person.PredictionTime<0?"视频分析处理中。":$"视频分析更新至 {person.PredictionTime:F2} 秒。"));
                foreach(var person in result.People.Where(p=>p.FallMotion))sb.AppendLine($"P{person.Id:00} 动作异常依据：{person.MotionEvidence}");
                if(result.People.Length==0)sb.AppendLine("当前未能定位人员，请核查监控画面。");
                if(result.Analytics is { } stats)
                {
                    sb.AppendLine($"活动观察：有效观测 {stats.ObservedSeconds:F0} 秒；累计 {stats.UniquePeople} 条轨迹（不代表独立身份）；区域进入 {stats.Entries} 次、离开 {stats.Exits} 次。");
                    foreach(var p in result.People.Where(p=>p.Telemetry is not null))
                    {
                        var t=p.Telemetry!;sb.AppendLine($"P{p.Id:00} 区域 {t.Zone}；区域停留 {t.DwellSeconds:F0} 秒；位置静止 {t.StillSeconds:F0} 秒；姿态依据 {p.ActivitySource}。");
                        sb.AppendLine("已观测姿态时长："+string.Join("；",t.ActivitySeconds.Select(v=>$"{Labels.Activity(v.Key)} {v.Value:F0} 秒")));
                    }
                }
            }
            if(video.Replay)sb.AppendLine("本次为视频回放，记录不计入现场告警。");
        }
        sb.AppendLine();sb.AppendLine("生命体征");
        if(radar is null||DateTimeOffset.Now-radar.Time>TimeSpan.FromSeconds(3))sb.AppendLine("暂无有效雷达数据，心率和呼吸率未获取。");
        else
        {
            sb.AppendLine($"雷达来源：{radar.Source}（{(radar.Replay?"数据回放":"实时采集")}）；识别 {radar.Frame.Tracks.Length} 个设备目标。");
            sb.AppendLine("雷达为非接触测量，无需佩戴或接触人体；视频与雷达编号不能自动认定为同一人。");
            if(radar.Frame.Presence is bool presence)sb.AppendLine("人员存在检测："+(presence?"检测到活动":"未触发"));
            foreach(var track in radar.Frame.Tracks)
            {
                string zone=string.Join("、",data.Settings.RadarZones.Where(z=>z.Contains(track.Position)).Select(z=>z.Name));
                sb.AppendLine($"设备目标 {track.Id}：位置 ({track.Position.X:F2}, {track.Position.Y:F2}, {track.Position.Z:F2}) m；速度 {track.Velocity.Length():F2} m/s；加速度 {track.Acceleration.Length():F2} m/s²；区域 {(zone.Length>0?zone:"区域外")}。");
                var height=radar.Frame.Heights?.FirstOrDefault(h=>h.Id==track.Id);if(height is not null)sb.AppendLine($"高度范围 {height.MinZ:F2}–{height.MaxZ:F2} m。");
            }
            var measured=radar.Frame.PointDetails?.Where(p=>p.Snr.HasValue).ToArray();
            if(measured?.Length>0)sb.AppendLine($"点云 {radar.Frame.Points.Length} 点；平均 SNR {measured.Average(p=>p.Snr):F1} dB；径向运动点 {measured.Count(p=>Math.Abs(p.Doppler)>=.1f)} 个。");
            foreach(var vital in data.Vitals.Where(v=>v.IsReplay==radar.Replay && (DateTimeOffset.Now-v.Time).TotalSeconds<3).GroupBy(v=>v.Target).Select(g=>g.Last()))
                sb.AppendLine($"设备目标 {vital.Target}：心率 {(vital.HeartRate?.ToString("F1")??"未获取")} bpm，呼吸率 {(vital.BreathRate?.ToString("F1")??"未获取")} 次/分；{(vital.Moving?"体动中，信号待确认":"静息状态")}。");
        }
        sb.AppendLine();sb.AppendLine("使用说明：识别结果用于辅助照护，疑似跌倒需人工核查；分析分值不代表风险概率。人员编号不自动关联住户身份，静息记录不等于睡眠分期。当前模型仍在现场验证阶段。");
        return sb.ToString();
    }
    public async Task<SavedReport> GenerateAsync(StoreData data,VideoSnapshot? video,AnalysisSnapshot? result,RadarSnapshot? radar,bool external,CancellationToken token)
    {
        string local=Local(data,video,result,radar);if(!external)return new(){Text=local};
        string shared=Local(data,video is null?null:video with{Source=video.Replay?"文件回放":"现场摄像头"},result,
            radar is null?null:radar with{Source=radar.Replay?"数据回放":"数据串口"});
        var payload=new{model=data.Settings.ApiModel,temperature=.2,stream=false,messages=new[]{
            new{role="system",content="你是康养值班分析助手。仅依据观测摘要，用中文写当前概况、优先事项、建议，控制在600字以内，短段落和必要列表，不输出思考过程或代码。保留回放/现场、过期、局部观测和模型分歧；历史待处理记录不等于当前告警，不编造身份、生命体征或医学诊断。雷达是非接触设备，不提出佩戴或接触检查。模型分数不是风险概率；不能把静息或躺姿当成睡眠。"},
            new{role="user",content=shared}}};
        string text=await CompleteAsync(data.Settings,payload,token);
        return new(){Provider=data.Settings.ApiModel,Text=text,Evidence=local};
    }
    public async Task TestAsync(AppSettings settings,CancellationToken token)
    {
        await CompleteAsync(settings,new{model=settings.ApiModel,stream=false,messages=new[]{new{role="user",content="连接测试，请仅回复：连接成功"}}},token,30);
    }
    private async Task<string> CompleteAsync(AppSettings settings,object payload,CancellationToken token,int seconds=120)
    {
        Validate(settings);string key;
        try{key=AppStore.UnprotectKey(settings.ProtectedApiKey);}
        catch(Exception ex) when(ex is System.Security.Cryptography.CryptographicException or FormatException){throw new InvalidOperationException("已保存密钥无法解密，请在当前Windows账户重新输入并保存密钥。");}
        if(key.Length==0&&!new Uri(settings.ApiUrl).IsLoopback)throw new InvalidOperationException("请在系统设置中保存访问密钥，或选择本机分析服务。");
        using var timeout=CancellationTokenSource.CreateLinkedTokenSource(token);timeout.CancelAfter(TimeSpan.FromSeconds(seconds));
        using var request=new HttpRequestMessage(HttpMethod.Post,Endpoint(settings.ApiUrl));
        if(key.Length>0)request.Headers.Authorization=new AuthenticationHeaderValue("Bearer",key);
        request.Content=new StringContent(JsonSerializer.Serialize(payload),Encoding.UTF8,"application/json");
        try
        {
        using var response=await http.SendAsync(request,HttpCompletionOption.ResponseHeadersRead,timeout.Token);
        if(!response.IsSuccessStatusCode)throw new InvalidOperationException(response.StatusCode switch{
            HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden=>"服务拒绝访问（"+(int)response.StatusCode+"），请检查密钥、账户权限和余额。",
            HttpStatusCode.NotFound=>"服务地址或模型不存在（404），请核对兼容API地址及模型名。",
            HttpStatusCode.TooManyRequests=>"调用达到限额（429），请检查余额或稍后重试。",
            _=>"分析服务返回 "+(int)response.StatusCode+"，请检查服务支持的模型与接口。"});
        using var stream=await response.Content.ReadAsStreamAsync(timeout.Token);
        using var limited=new MemoryStream();byte[] buffer=new byte[8192];int n;
        while((n=await stream.ReadAsync(buffer,timeout.Token))>0){if(limited.Length+n>1024*1024)throw new InvalidDataException("API响应过大。");limited.Write(buffer,0,n);}
        using var doc=JsonDocument.Parse(limited.ToArray());
        if(!doc.RootElement.TryGetProperty("choices",out var choices)||choices.ValueKind!=JsonValueKind.Array||choices.GetArrayLength()==0||
           !choices[0].TryGetProperty("message",out var message)||!message.TryGetProperty("content",out var content))
            throw new InvalidDataException("服务未返回兼容的报告内容，请使用支持 chat/completions 的模型。");
        string text=content.ValueKind==JsonValueKind.String?content.GetString()??"":content.ValueKind==JsonValueKind.Array?string.Join("",content.EnumerateArray().Where(p=>p.ValueKind==JsonValueKind.Object&&p.TryGetProperty("text",out _)).Select(p=>p.GetProperty("text").GetString())):"";
        int end=text.IndexOf("</think>",StringComparison.OrdinalIgnoreCase);if(end>=0)text=text[(end+8)..].Trim();
        if(string.IsNullOrWhiteSpace(text))throw new InvalidDataException("API没有返回有效报告。");
        return text;
        }
        catch(OperationCanceledException) when(!token.IsCancellationRequested){throw new TimeoutException($"分析服务{seconds}秒内未完成，请检查服务或选择更快的模型。可以先生成监测报告。");}
        catch(HttpRequestException){throw new InvalidOperationException("无法连接分析服务，请检查地址、网络或本机服务是否启动。");}
        catch(JsonException){throw new InvalidDataException("服务返回内容无法读取，请核对是否为兼容的分析接口。");}
    }
    public static void ExportHtml(SavedReport report,string path)
    {
        var parts=Parts(report);string body=ReportLayout.Html(parts.Text);string evidence=string.IsNullOrEmpty(parts.Evidence)?"":"<details><summary>观测依据</summary>"+ReportLayout.Html(parts.Evidence)+"</details>";
        File.WriteAllText(path,$"<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>康养健康报告</title><style>body{{font:16px/1.8 'Microsoft YaHei',sans-serif;background:#f3f5f0;color:#203b34;max-width:900px;margin:40px auto;padding:30px}}h1{{border-bottom:3px solid #28684d;padding-bottom:20px}}h2{{font-size:21px;color:#28684d;margin-top:26px}}article{{background:white;padding:30px}}.item{{margin-left:12px}}summary{{cursor:pointer;color:#28684d}}</style><h1>康养健康报告</h1><p>{report.Time.LocalDateTime:yyyy-MM-dd HH:mm:ss} · {WebUtility.HtmlEncode(report.Provider)}</p><article>{body}{evidence}</article></html>",new UTF8Encoding(false));
    }
    public static (string Text,string Evidence) Parts(SavedReport report)
    {
        if(!string.IsNullOrEmpty(report.Evidence))return(report.Text,report.Evidence);
        const string marker="\n\n观测依据\n";string text=report.Text.Replace("\r","");int at=text.LastIndexOf(marker,StringComparison.Ordinal);
        return at<0?(report.Text,""):(text[..at],text[(at+marker.Length)..]);
    }
    public void Dispose()=>http.Dispose();
}
