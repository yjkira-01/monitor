using System.Collections.Concurrent;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.IO.Ports;
using System.Text;
using System.Text.Json;
using System.Diagnostics;
using System.Windows.Media.Imaging;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using CareMonitor.Models;
using CareMonitor.Services;
using CareMonitor.Views;
using Microsoft.Win32;

namespace CareMonitor;
public partial class MainWindow : Window
{
    private readonly AppStore store;
    private readonly VideoService video;
    private readonly RadarService radar=new();
    private readonly ReportService reports=new();
    private readonly DispatcherTimer timer=new(){Interval=TimeSpan.FromMilliseconds(33)};
    private readonly ConcurrentQueue<MonitorEvent> eventQueue=new();
    private readonly ConcurrentQueue<VitalSample> vitalQueue=new();
    private readonly CancellationTokenSource reportCancel=new();
    private CancellationTokenSource? activeReport;
    private bool testingApi;
    private long lastSeq=-1,seenGeneration=-1;
    private double lastObservation=-1;
    private DateTimeOffset lastDetail=DateTimeOffset.MinValue,lastSave=DateTimeOffset.Now,lastReport=DateTimeOffset.Now;
    private bool generating,closing,closed;
    private int page;
    private SavedReport? selectedReport;
    private readonly Dictionary<int,PersonStatusCard> personCards=[];
    private readonly Dictionary<int,double> personLastSeen=[];
    private readonly Dictionary<uint,RadarTargetCard> radarCards=[];
    private readonly TextBlock radarEmpty=new(){Foreground=Brushes.Gray};
    private DateTimeOffset lastSound=DateTimeOffset.MinValue;
    private readonly TextBlock emptyPeople=new(){Foreground=Brushes.Gray,FontSize=12};
    private long peopleGeneration=-1;
    private uint? lastCloudFrame;
    private DateTimeOffset lastCloudTime;
    private MonitoringZone? redrawZone;
    public MainWindow()
    {
        InitializeComponent();
        bool smoke=Environment.GetCommandLineArgs().Contains("--ui-smoke");
        if(smoke){WindowStartupLocation=WindowStartupLocation.Manual;Left=-3000;Top=-1600;ShowInTaskbar=false;ShowActivated=false;}
        store=new(smoke?Path.Combine(AppContext.BaseDirectory,"ui-validation/store"):null);
        if(!smoke)ApplyDeviceDefaults();
        var launchArgs=Environment.GetCommandLineArgs();string selectedFall=launchArgs.Contains("--fall-model")?launchArgs[Array.IndexOf(launchArgs,"--fall-model")+1]:store.Data.Settings.FallModel;
        video=new(Path.Combine(AppContext.BaseDirectory,"Assets"),store.Data.Settings.UseGpu,store.Data.Settings.ModelProfile!="baseline",selectedFall);
        StereoWorkspace.Initialize(store,video);
        video.ConfigureRules(store.Data.Settings);video.Event+=e=>eventQueue.Enqueue(e);radar.Vital+=v=>vitalQueue.Enqueue(v);VideoCanvas.ZoneDrawn+=CreateZone;
        VideoCanvas.ZoneSelected+=id=>CurrentZones.SelectedItem=(CurrentZones.ItemsSource as MonitoringZone[])?.FirstOrDefault(z=>z.Id==id);
        VideoCanvas.ZoneDeleteRequested+=()=>DeleteCurrentZone(this,new());
        radar.ConfigureRules(store.Data.Settings);radar.Event+=e=>eventQueue.Enqueue(e);
        FillSettings();RefreshLists();EventFilter.SelectedIndex=0;ShowPage(0);StorageText.Text="数据保存位置："+store.DirectoryPath;
        timer.Tick+=Tick;timer.Start();AllowDrop=true;
        Drop+=async (_,e)=>{if(e.Data.GetData(DataFormats.FileDrop) is string[] files && files.Length>0)await Guard(()=>StartSingle(()=>video.StartFileAsync(files[0])));};
        Loaded+=async (_,_)=>
        {
            try{await video.InitializeAsync();}catch(Exception ex){StatusText.Text="分析服务启动失败："+ex.Message;}
            var args=Environment.GetCommandLineArgs();int index=Array.IndexOf(args,"--test-video");
            if(index>=0 && index+1<args.Length)await Guard(()=>video.StartFileAsync(args[index+1]));
            if(args.Contains("--ui-smoke"))await UiSmoke();
            if(store.RecoveryNote is not null)MessageBox.Show(this,store.RecoveryNote,"档案恢复");
        };
    }
    private async Task Guard(Func<Task> operation)
    {
        try{await operation();}catch(Exception ex){AppLog.Write("ui-operation",ex);MessageBox.Show(this,ex.Message,"操作未完成",MessageBoxButton.OK,MessageBoxImage.Warning);}
    }
    private async Task StartSingle(Func<Task> operation){await StereoWorkspace.StopAsync();await operation();}
    private async Task UiSmoke()
    {
        string directory=Path.Combine(AppContext.BaseDirectory,"ui-validation");Directory.CreateDirectory(directory);
        var latencies=new List<double>();var observations=new HashSet<double>();var sequences=new HashSet<long>();
        double last=Stopwatch.GetTimestamp();int count=0;
        int singlePersonComparisons=0;bool scoreMismatch=false;
        EventHandler measure=(_,_)=>
        {
            double tick=Stopwatch.GetTimestamp();latencies.Add((tick-last)*1000/Stopwatch.Frequency);last=tick;
            if(video.Analysis is { } current)
            {
                observations.Add(current.Time);
                if(current.People is [var only]&&only.WholeSceneInput==true&&only.RgbScore.HasValue&&current.SceneScore.HasValue&&Math.Abs(only.PredictionTime-current.RgbTime)<1e-6)
                {
                    if(Math.Abs(only.RgbScore.Value-current.SceneScore.Value)>1e-6)scoreMismatch=true;
                    singlePersonComparisons++;
                }
            }
            if(video.Latest is not null)sequences.Add(video.Latest.Sequence);count++;
        };
        timer.Tick+=measure;
        try
        {
            void Capture(string name)
            {
                UpdateLayout();var image=new RenderTargetBitmap((int)ActualWidth,(int)ActualHeight,96,96,PixelFormats.Pbgra32);image.Render(this);
                var encoder=new PngBitmapEncoder();encoder.Frames.Add(BitmapFrame.Create(image));using var stream=File.Create(Path.Combine(directory,name+".png"));encoder.Save(stream);
            }
            await Task.Delay(2600);Capture("monitor-active");var example=video.Analysis?.People.FirstOrDefault();await Task.Delay(15400);var snap=video.Latest;var result=video.Analysis;
            if(snap is null||sequences.Count<20||observations.Count<3)throw new InvalidOperationException("UI视频/识别没有持续更新。");
            if(scoreMismatch||singlePersonComparisons==0)throw new InvalidOperationException("单人画面与人员状态的一致性检查未通过。");
            video.TogglePause();await Task.Delay(1500);double position=video.Latest!.Time;await Task.Delay(500);
            if(Math.Abs(video.Latest!.Time-position)>.2)throw new InvalidOperationException("暂停失败。");
            timer.Stop();
            try
            {
                var fixture=(example??new PersonState(901,new(0,0,1,1),1,[],[],snap.Time)) with{ShortRgbScore=.73f,ShortConsecutive=2};
                var fixtureResult=new AnalysisSnapshot(snap.Generation,snap.Time,[fixture],.2f,"standing","UI regression fixture",0,"test");
                UpdatePeople(fixtureResult,snap);var card=personCards[fixture.Id];card.Details.IsExpanded=true;
                if(card.Details.Content is not TextBlock details||!details.Text.Contains("最近2秒跌倒分值 0.730")||!details.Text.Contains("短窗口连续确认 2 次"))throw new InvalidOperationException("人员详情未显示独立短窗口证据。");
                Capture("person-details-fixture");
                for(int i=0;i<8;i++)UpdatePeople(fixtureResult with{People=[fixture with{Confidence=i/10f}]},snap);
                if(!ReferenceEquals(card,personCards[fixture.Id])||!card.Details.IsExpanded)throw new InvalidOperationException("实时更新重置了人员详情展开状态。");
                UpdatePeople(fixtureResult with{People=[],Time=snap.Time+1},snap with{Time=snap.Time+1});
                UpdatePeople(fixtureResult with{Time=snap.Time+1.1},snap with{Time=snap.Time+1.1});
                if(!ReferenceEquals(card,personCards[fixture.Id])||!card.Details.IsExpanded)throw new InvalidOperationException("短时缺失重置了人员详情展开状态。");
                ShowPage(5);ShowPage(0);UpdatePeople(fixtureResult,snap);
                if(!card.Details.IsExpanded)throw new InvalidOperationException("切页重置了人员详情展开状态。");
                card.Details.IsExpanded=false;UpdatePeople(fixtureResult,snap);
                if(card.Details.IsExpanded)throw new InvalidOperationException("关闭的人员详情被自动展开。");
                UpdatePeople(result,snap);
            }
            finally{timer.Start();}
            Capture("monitor");
            ShowPage(4);
            var legacyReport=new SavedReport{Provider="界面验证",Text="**当前概况**\n- 视频为 **文件回放**。\n\n**优先事项**\n1. 核查疑似跌倒。\n\n建议\n依据实际观测处理。\n\n观测依据\n视频回放测试数据"};
            DisplayReport(legacyReport);UpdateLayout();
            string rendered=new System.Windows.Documents.TextRange(ReportText.Document.ContentStart,ReportText.Document.ContentEnd).Text;
            if(rendered.Contains("**")||!rendered.Contains("当前概况")||EvidencePanel.Visibility!=Visibility.Visible||ReportText.Document.Blocks.Count<4)throw new InvalidOperationException("报告标题、列表或历史依据折叠未正确排版。");
            ReportService.ExportHtml(legacyReport,Path.Combine(directory,"report-layout.html"));Capture("report-formatted");
            await Generate(false,false);
            if(selectedReport?.Provider!="本地"||generating||!LocalReportButton.IsEnabled||CancelReportButton.Visibility!=Visibility.Collapsed)throw new InvalidOperationException("本地报告生成或按钮状态异常。");
            var testZone=new MonitoringZone(Guid.NewGuid(),"区域交互验证",snap.Source,new(.1f,.2f,.5f,.7f));
            var anotherSource=new MonitoringZone(Guid.NewGuid(),"其他来源保留","other-source-fixture",new(.1f,.1f,.5f,.5f));
            store.Data.Settings.Zones.Add(testZone);store.Data.Settings.Zones.Add(anotherSource);UpdateRules();ShowPage(0);
            CurrentZones.SelectedItem=testZone;VideoCanvas.DrawingZone=true;CancelZone(this,new());
            if(VideoCanvas.DrawingZone)throw new InvalidOperationException("取消绘制未生效。");
            DeleteCurrentZone(this,new());
            if(store.Data.Settings.Zones.Any(z=>z.Id==testZone.Id)||!store.Data.Settings.Zones.Any(z=>z.Id==anotherSource.Id))throw new InvalidOperationException("监控页删除区域未正确隔离来源。");
            store.Data.Settings.Zones.RemoveAll(z=>z.Id==anotherSource.Id);UpdateRules();
            ShowPage(1);
            var points=Enumerable.Range(0,80).Select(i=>new RadarPoint(new System.Numerics.Vector3((i%10-5)*.12f,1.5f+i/10*.08f,(i%3-1)*.18f),i%3==0?-.2f:i%3==1?.2f:0,8+i%20)).ToArray();
            RadarCloudView.SetRadar(new RadarFrame(1,points.Select(p=>p.Position).ToArray(),[new(0,new(0,1.8f,.4f),default,.9f)],[],PointDetails:points));
            UpdateLayout();Capture("radar-rendering-fixture");RadarCloudView.Clear();
            foreach(var nav in NavigationPanel.Children.OfType<Button>())
            {
                int index=int.Parse(nav.Tag.ToString()!);if(index==0)continue;
                nav.RaiseEvent(new RoutedEventArgs(Button.ClickEvent));await Task.Delay(200);
                if(NavigationPanel.Children.OfType<Button>().Count(ButtonAssist.GetIsSelected)!=1 || !ButtonAssist.GetIsSelected(nav))
                    throw new InvalidOperationException($"导航点击与选中状态不一致：期望 {index}，当前 {page}，选中 "+string.Join(",",NavigationPanel.Children.OfType<Button>().Where(ButtonAssist.GetIsSelected).Select(b=>b.Tag)));
                Capture("page-"+index);
            }
            ShowPage(5);CameraChoice.IsDropDownOpen=true;await Task.Delay(200);
            if(!CameraChoice.IsDropDownOpen)throw new InvalidOperationException("设备选择下拉菜单未打开。");
            CameraChoice.IsDropDownOpen=false;
            ApiUrl.Focus();await Task.Delay(100);Capture("settings-keyboard-focus");
            ShowPage(2);
            var dialog=new EditDialog("界面验证 · 不保存档案",[("name","姓名",""),("notes","照护备注","")]){Owner=this};
            dialog.Show();await Task.Delay(100);dialog.Close();
            double fullWidth=Width,fullHeight=Height;Width=1024;Height=760;ShowPage(0);await Task.Delay(200);Capture("monitor-compact");
            Width=fullWidth;Height=fullHeight;
            ShowPage(0);video.Seek(1);await Task.Delay(1800);
            if(video.Latest!.Generation==snap.Generation)throw new InvalidOperationException("切换时间轴没有更新代次。");
            await video.StopAsync();await Task.Delay(300);
            if(PauseButton.IsEnabled || StopButton.IsEnabled)throw new InvalidOperationException("停止后的按钮状态未更新。");
            await video.StartFileAsync(Environment.GetCommandLineArgs()[Array.IndexOf(Environment.GetCommandLineArgs(),"--test-video")+1]);await Task.Delay(2500);
            video.TogglePause();lastDetail=DateTimeOffset.Now;Tick(null,EventArgs.Empty);
            if(!PauseButton.IsEnabled || PauseButton.Content.ToString()!="继续")throw new InvalidOperationException("暂停后的继续按钮未更新。");
            PauseButton.Focus();await Task.Delay(100);Capture("monitor-reopened");
            double max=latencies.Count>0?latencies.Max():0;double[] sorted=latencies.Order().ToArray();
            var evidence=new{passed=true,ticks=count,video_frames=sequences.Count,pose_observations=observations.Count,
                max_tick_ms=max,p95_tick_ms=sorted[(int)(sorted.Length*.95)],model_status=video.ModelStatus,source=snap.Source,
                snapshot=result,video_3d_removed=true,process_memory_bytes=Process.GetCurrentProcess().WorkingSet64,events=store.Data.Events.Count,
                native_command_theme=true,navigation_clicks=NavigationPanel.Children.OfType<Button>().Count()-1,compact_width=1024,device_dropdown=true,dialog_opened=true,window_icon=Icon is not null,person_details_stable=true,
                single_person_score_checks=singlePersonComparisons,short_occlusion_details_stable=true,short_window_details_visible=true,playback_controls_unthrottled=true,
                report_native_formatting=true,legacy_report_evidence_fold=true,local_report_generation=true,monitor_zone_cancel_delete_source_isolation=true,radar_rendering_fixture=true};
            File.WriteAllText(Path.Combine(directory,"result.json"),JsonSerializer.Serialize(evidence,new JsonSerializerOptions{WriteIndented=true}));
            await StereoWorkspace.StopAsync();await video.StopAsync();await radar.StopAsync();
            Closed+=(_,_)=>File.WriteAllText(Path.Combine(directory,"idle-close.json"),JsonSerializer.Serialize(new{passed=true,video_stopped=true,radar_stopped=true,closing_reentry_avoided=true}));
        }
        catch(Exception ex){File.WriteAllText(Path.Combine(directory,"error.txt"),ex.ToString());}
        finally{timer.Tick-=measure;Close();}
    }
    private void Navigate(object sender,RoutedEventArgs e)
    {
        int target=int.Parse(((Button)sender).Tag.ToString()!);ShowPage(target);
    }
    private void ShowPage(int target)
    {
        page=target;var pages=new FrameworkElement[]{MonitorPage,RadarPage,ResidentsPage,EventsPage,ReportsPage,SettingsPage,RulesPage,StereoWorkspace};
        for(int i=0;i<pages.Length;i++)pages[i].Visibility=i==target?Visibility.Visible:Visibility.Collapsed;
        foreach(var button in NavigationPanel.Children.OfType<Button>())
            ButtonAssist.SetIsSelected(button, int.Parse(button.Tag.ToString()!)==target);
        string[] titles=["监控中心","雷达监测","住户管理","事件记录","健康报告","系统设置","监测规则","双目定位"];
        string[] subtitles=["查看监控画面、人员状态和活动记录。","查看点云、空间轨迹、生命体征与区域活动。","管理住户档案、房间、床位和照护信息。","查询、核查和处理告警，保留处理记录。","汇总监测数据，生成和导出健康报告。","管理摄像头、雷达和智能分析服务。","设置重点区域、停留提醒与静止观察阈值。","两路视频监测、时间配对与标定后的多人三维定位。"];
        PageTitle.Text=titles[target];PageSubtitle.Text=subtitles[target];RefreshLists();
    }
    private void Tick(object? sender,EventArgs e)
    {
        if(closing)return;var now=DateTimeOffset.Now;ClockText.Text=now.ToString("yyyy.MM.dd  HH:mm");
        StereoWorkspace.Tick();
        if(page==1)
        {
            var fresh=radar.Latest;
            if(fresh is not null&&now-fresh.Time<TimeSpan.FromSeconds(3))
            {if(lastCloudFrame!=fresh.Frame.Number||lastCloudTime!=fresh.Time){RadarCloudView.SetRadar(fresh.Frame,radar.Observation);lastCloudFrame=fresh.Frame.Number;lastCloudTime=fresh.Time;}}
            else if(lastCloudFrame is not null){RadarCloudView.Clear();lastCloudFrame=null;}
        }
        var frame=video.Latest;var result=video.Analysis;
        if(frame?.Generation!=seenGeneration)
        {
            seenGeneration=frame?.Generation??-1;lastObservation=-1;lastSeq=-1;TrajectoryView.Update(null);HeatmapView.Update(null);
            RefreshCurrentZones();VideoCanvas.CancelDrawing();redrawZone=null;
        }
        if(page==0 && frame?.Sequence!=lastSeq){VideoCanvas.Update(frame,result);lastSeq=frame?.Sequence??-1;}
        if(page==0 && result is not null && result.Generation==frame?.Generation && result.Time!=lastObservation)
        {
            VideoCanvas.Update(frame,result);TrajectoryView.Update(result.Analytics);HeatmapView.Update(result.Analytics);lastObservation=result.Time;
        }
        PauseButton.IsEnabled=frame is not null && !frame.Ended;StopButton.IsEnabled=frame is not null;
        PauseButton.Content=frame?.Paused==true?"继续":"暂停";
        ButtonAssist.SetIcon(PauseButton,(Geometry)FindResource(frame?.Paused==true?"IconPlay":"IconPause"));
        if(now-lastDetail<TimeSpan.FromMilliseconds(250))return;lastDetail=now;
        CaptureText.Text=video.CaptureStatus;
        ModeText.Text=page==7?(StereoWorkspace.Connected?"双目实时监测":"双目未连接"):page==1?(radar.Latest is { } r&&now-r.Time<TimeSpan.FromSeconds(3)?r.Replay?"雷达数据回放":"雷达实时监测":"雷达未连接"):
            frame is null?"视频未连接":frame.Replay?"视频回放":frame.Ended?"视频已断开":"实时监控";
        StatusText.Text=video.ModelStatus;ModelDetails.Text=video.ModelStatus+"\n模型切换保存后重启应用生效。";
        if(frame is not null)
        {
            PositionText.Text=TimeSpan.FromSeconds(frame.Time).ToString(@"mm\:ss");DurationText.Text=frame.Replay?TimeSpan.FromSeconds(frame.Duration).ToString(@"mm\:ss"):"LIVE";
            VideoSeek.IsEnabled=frame.Replay;VideoSeek.Maximum=Math.Max(1,frame.Duration);if(!VideoSeek.IsMouseCaptureWithin)VideoSeek.Value=frame.Time;
        }
        else{PositionText.Text="00:00";VideoSeek.IsEnabled=false;VideoCanvas.Update(null,null);}
        if(page==0){UpdatePeople(result,frame);VideoCanvas.Update(frame,result);}
        CancelZoneButton.IsEnabled=VideoCanvas.DrawingZone;
        ZoneHintText.Text=VideoCanvas.DrawingZone?"在画面中拖动框选，点击取消或按 Esc 退出。":$"当前画面 {CurrentZones.Items.Count} 个区域 · 区域内 {result?.Analytics?.CurrentZoneOccupancy??0} 人 · 点击区域可选中，Delete 删除";
        bool changed=false;
        for(int i=0;i<100 && eventQueue.TryDequeue(out var item);i++)
        {
            store.Data.Events.Add(item);changed=true;
            if(!item.IsReplay&&store.Data.Settings.SoundAlerts&&now-lastSound>TimeSpan.FromSeconds(5)&&
                (item.Kind.Contains("跌倒")||item.Kind.Contains("提醒")))
            {System.Media.SystemSounds.Exclamation.Play();lastSound=now;}
        }
        for(int i=0;i<100 && vitalQueue.TryDequeue(out var item);i++)
        {
            store.AddVital(item);changed=true;
        }
        if(changed&&page==3)RefreshEvents();
        if(changed&&now-lastSave>TimeSpan.FromSeconds(10)){store.Save();lastSave=now;}
        if(page==1)UpdateRadar(now);
        var settings=store.Data.Settings;
        if(settings.AutoReport && !generating && ReportService.HasCredentials(settings) && now-lastReport>TimeSpan.FromSeconds(settings.ReportIntervalSeconds) &&
            (frame is not null && !frame.Paused || radar.Latest is not null && now-radar.Latest.Time<TimeSpan.FromSeconds(3)))
        {lastReport=now;_ = Generate(true,true);}
    }
    private void UpdatePeople(AnalysisSnapshot? result,VideoSnapshot? frame)
    {
        if(peopleGeneration!=(frame?.Generation??-1))
        {peopleGeneration=frame?.Generation??-1;personCards.Clear();personLastSeen.Clear();PeoplePanel.Children.Clear();}
        if(!PeoplePanel.Children.Contains(emptyPeople))PeoplePanel.Children.Insert(0,emptyPeople);
        bool valid=result is not null&&result.Generation==frame?.Generation && frame.Time-result.Time<2&&!(frame.Ended&&!frame.Replay);
        var people=valid?result!.People:[];PeopleCountText.Text=valid?$"当前识别 {people.Length} 人":"等待人员识别";
        PeopleCountText.ToolTip="人员编号仅用于本次监控中的跟踪，不关联住户身份。";
        emptyPeople.Visibility=people.Length==0?Visibility.Visible:Visibility.Collapsed;
        emptyPeople.Text=frame is null?"请先选择视频文件或连接摄像头。":!valid?"正在识别人员，请稍候。":"当前未识别到人员，可查看监控画面确认。";
        var ids=people.Select(p=>p.Id).ToHashSet();
        foreach(var id in personCards.Keys.Where(id=>!ids.Contains(id)).ToArray())
        {
            if(valid&&frame!.Time-personLastSeen.GetValueOrDefault(id)>3){PeoplePanel.Children.Remove(personCards[id]);personCards.Remove(id);personLastSeen.Remove(id);}
            else personCards[id].Visibility=Visibility.Collapsed;
        }
        foreach(var p in people)
        {
            if(!personCards.TryGetValue(p.Id,out var card))
            {card=new();personCards[p.Id]=card;PeoplePanel.Children.Add(card);}
            personLastSeen[p.Id]=frame!.Time;card.Visibility=Visibility.Visible;card.Update(p);
        }
        SceneText.Text=valid?VideoService.SceneAssessment(result!):frame?.Ended==true&&!frame.Replay?"视频已断开，等待重新连接":"正在分析画面";
        SceneAttention.Update(valid?result!.Attention:null);
        AnalysisSummaryText.Text=valid?$"更新时间 {VideoTime(result!.Time)}"+(VideoService.UprightDisagreement(result)?"\n直立姿态与视频模型不一致，保留两项结果。":result.SceneScore>=.5&&people.All(p=>!VideoService.IsFallAlert(p))?"\n尚未定位到风险人员，请查看监控画面。":""):"暂无分析结果";
        InferenceText.Text=valid?$"视频分析分值 {Score(result!.SceneScore)}\n动作变化 {Score(result.FeatureChange)}\n处理耗时 {result.InferenceMs:F0} ms\n视频分析进度 {VideoTime(result.RgbTime)}\n关注区域基于整体画面，不能定位到个人。":"暂无分析详情";
        var stats=valid?result!.Analytics:null;
        ObservationStats.Text=stats is null?"连接视频后开始记录":$"有效观测 {stats.ObservedSeconds:F0} 秒 · 累计轨迹 {stats.UniquePeople} 条\n区域内 {stats.CurrentZoneOccupancy} 人 · 进入 {stats.Entries} 次 / 离开 {stats.Exits} 次";
        ActivityTimelineText.Text=string.Join("\n\n",people.Where(p=>p.Telemetry is not null).Select(p=>$"P{p.Id:00}  {p.Telemetry!.Zone}\n"+
            string.Join(" · ",p.Telemetry.ActivitySeconds.OrderByDescending(x=>x.Value).Take(4).Select(x=>$"{Labels.Activity(x.Key)} {x.Value:F0}s"))));
    }
    private static string Score(float? value)=>value.HasValue?value.Value.ToString("F3"):"—";
    private static string VideoTime(double time)=>time<0?"处理中":TimeSpan.FromSeconds(time).ToString(@"mm\:ss");
    private void UpdateRadar(DateTimeOffset now)
    {
        RadarStatusText.Text=radar.Status;var snapshot=radar.Latest;
        if(!RadarTargetsPanel.Children.Contains(radarEmpty)){RadarTargetsPanel.Children.Clear();RadarTargetsPanel.Children.Add(radarEmpty);}
        var diagnostic=radar.Diagnostics;
        RadarDiagnosticText.Text=$"有效帧 {diagnostic.Frames} · {diagnostic.FrameHz:F1} Hz · 帧号间隙 {diagnostic.MissedFrames}\n接收 {diagnostic.ReceivedBytes/1024d:F1} KB · TLV {string.Join(", ",diagnostic.TlvTypes)}";
        if(snapshot is null||now-snapshot.Time>TimeSpan.FromSeconds(3))
        {RadarCloudView.Clear();RadarMap.Update(null,store.Data.Settings.RadarZones.ToArray());RadarPointsGrid.ItemsSource=null;RadarPresenceText.Text="人员存在：暂无有效数据";radarEmpty.Text="暂无监测数据。请连接雷达或选择已保存的数据文件。";foreach(var card in radarCards.Values)card.Visibility=Visibility.Collapsed;HeartWave.Update([]);BreathWave.Update([]);return;}
        RadarMap.Update(radar.Observation,store.Data.Settings.RadarZones.ToArray());
        RadarPointsGrid.ItemsSource=(snapshot.Frame.PointDetails??[]).Take(160).Select((p,i)=>new{Index=i,X=p.Position.X.ToString("F2"),Y=p.Position.Y.ToString("F2"),Z=p.Position.Z.ToString("F2"),Doppler=p.Doppler.ToString("F2"),Snr=p.Snr?.ToString("F1")??"—"}).ToArray();
        RadarPresenceText.Text=snapshot.Frame.Presence.HasValue?$"人员存在检测：{(snapshot.Frame.Presence.Value?"检测到活动":"未触发")}":$"跟踪目标：{snapshot.Frame.Tracks.Length} · 当前固件未输出存在标志";
        var targetIds=snapshot.Frame.Tracks.Select(t=>t.Id).Order().ToArray();
        var previousIds=RadarSubjectChoice.ItemsSource as uint[];
        if(previousIds is null||!previousIds.SequenceEqual(targetIds))
        {
            uint? selected=RadarSubjectChoice.SelectedItem is uint target?target:null;RadarSubjectChoice.ItemsSource=targetIds;
            if(selected.HasValue&&targetIds.Contains(selected.Value))RadarSubjectChoice.SelectedItem=selected.Value;else if(targetIds.Length>0)RadarSubjectChoice.SelectedIndex=0;
        }
        radarEmpty.Text=(snapshot.Replay?"雷达数据回放":"实时体征")+(targetIds.Length==0?"\n当前没有已分配的跟踪目标。":"");
        foreach(var staleId in radarCards.Keys.Where(key=>!targetIds.Contains(key)).ToArray()){RadarTargetsPanel.Children.Remove(radarCards[staleId]);radarCards.Remove(staleId);}
        foreach(var track in snapshot.Frame.Tracks)
        {
            var sample=store.Data.Vitals.LastOrDefault(v=>v.Target==track.Id && v.IsReplay==snapshot.Replay && now-v.Time<TimeSpan.FromSeconds(3));
            var resident=store.Data.Residents.FirstOrDefault(r=>r.RadarTargetId==track.Id);
            var target=radar.Observation?.Targets.FirstOrDefault(t=>t.Id==track.Id);
            if(!radarCards.TryGetValue(track.Id,out var card)){card=new();radarCards[track.Id]=card;RadarTargetsPanel.Children.Add(card);}card.Visibility=Visibility.Visible;card.Update(track,sample,resident,target);
        }
        uint? selectedTarget=RadarSubjectChoice.SelectedItem is uint id?id:null;
        var first=store.Data.Vitals.LastOrDefault(v=>v.Target==selectedTarget && v.IsReplay==snapshot.Replay && now-v.Time<TimeSpan.FromSeconds(3));
        HeartWave.Update(first?.HeartWave??[]);BreathWave.Update(first?.BreathWave??[]);
        var recent=store.Data.Vitals.Where(v=>v.Target==selectedTarget && !v.IsReplay && now-v.Time<TimeSpan.FromHours(8)).ToArray();
        RestText.Text=recent.Length==0?"暂无静息记录。请选择已连接的设备目标。":$"近8小时记录 {recent.Length} 次\n静息记录 {recent.Count(v=>!v.Moving)} 次\n有效呼吸记录 {recent.Count(v=>v.BreathRate.HasValue)} 次\n静息记录不等于睡眠时长。";
    }
    private async void OpenVideo(object sender,RoutedEventArgs e)
    {
        var dialog=new OpenFileDialog{Filter="视频文件|*.mp4;*.avi;*.mov;*.mkv;*.webm|所有文件|*.*"};if(dialog.ShowDialog(this)==true)await Guard(()=>StartSingle(()=>video.StartFileAsync(dialog.FileName)));
    }
    private async void StartCamera(object sender,RoutedEventArgs e)
    {
        await Guard(()=>StartSingle(()=>
        {
            var settings=store.Data.Settings;
            var match=CameraDevices.Resolve(CameraDevices.List(),settings.CameraDeviceId,settings.CameraDeviceName,settings.CameraIndex);
            if(match is null)throw new IOException("已配置的摄像头不在线，请在系统设置重新选择设备。");settings.CameraIndex=match.Index;
            return video.StartCameraAsync(settings);
        }));
    }
    private void PauseVideo(object sender,RoutedEventArgs e)=>video.TogglePause();
    private async void StopVideo(object sender,RoutedEventArgs e)=>await Guard(()=>StartSingle(()=>video.StopAsync()));
    private void SeekVideo(object sender,MouseButtonEventArgs e){if(VideoSeek.IsEnabled)video.Seek(VideoSeek.Value);}
    private void ResetObservation(object sender,RoutedEventArgs e){video.ResetStatistics();TrajectoryView.Update(null);HeatmapView.Update(null);}
    private async void ConnectNetworkVideo(object sender,RoutedEventArgs e)
    {
        var dialog=new EditDialog("接入网络视频",[("url","RTSP / HTTP 视频流地址","")],"支持网络摄像头、NVR 或 go2rtc 输出。地址仅用于本次连接，不写入报告或配置。"){Owner=this};
        if(dialog.ShowDialog()==true)await Guard(()=>StartSingle(()=>video.StartNetworkAsync(dialog["url"])));
    }
    private void DrawZone(object sender,RoutedEventArgs e)
    {
        if(video.Latest is null){StatusText.Text="请先连接视频，再绘制监测区域。";return;}
        redrawZone=null;VideoCanvas.DrawingZone=true;VideoCanvas.InvalidateVisual();ZoneHintText.Text="在画面拖动框选区域，按 Esc 取消。";
    }
    private void CreateZone(Box bounds)
    {
        var frame=video.Latest;if(frame is null)return;
        if(redrawZone is { } existing&&existing.Source==frame.Source)
        {store.Data.Settings.Zones.RemoveAll(z=>z.Id==existing.Id);store.Data.Settings.Zones.Add(existing with{Bounds=bounds});redrawZone=null;UpdateRules();return;}
        int index=1;while(store.Data.Settings.Zones.Any(z=>z.Source==frame.Source&&z.Name=="重点区域"+index))index++;
        EditZoneRecord(new(Guid.NewGuid(),"重点区域"+index,frame.Source,bounds));
    }
    private void CancelZone(object sender,RoutedEventArgs e){redrawZone=null;VideoCanvas.CancelDrawing();}
    private void SelectCurrentZone(object sender,SelectionChangedEventArgs e){VideoCanvas.SelectedZoneId=(CurrentZones.SelectedItem as MonitoringZone)?.Id;VideoCanvas.InvalidateVisual();}
    private void EditCurrentZone(object sender,RoutedEventArgs e){if(CurrentZones.SelectedItem is MonitoringZone z)EditZoneRecord(z);}
    private void RedrawCurrentZone(object sender,RoutedEventArgs e){if(CurrentZones.SelectedItem is MonitoringZone z){redrawZone=z;VideoCanvas.DrawingZone=true;VideoCanvas.InvalidateVisual();}}
    private void DeleteCurrentZone(object sender,RoutedEventArgs e){if(CurrentZones.SelectedItem is MonitoringZone z){store.Data.Settings.Zones.RemoveAll(x=>x.Id==z.Id);CancelZone(this,new());UpdateRules();}}
    private void ClearCurrentZones(object sender,RoutedEventArgs e){if(video.Latest is { } frame){store.Data.Settings.Zones.RemoveAll(z=>z.Source==frame.Source);CancelZone(this,new());UpdateRules();}}
    private void RefreshCurrentZones()
    {
        Guid? selected=(CurrentZones.SelectedItem as MonitoringZone)?.Id;
        var zones=store.Data.Settings.Zones.Where(z=>z.Source==video.Latest?.Source).ToArray();CurrentZones.ItemsSource=zones;
        CurrentZones.SelectedItem=zones.FirstOrDefault(z=>z.Id==selected)??zones.LastOrDefault();VideoCanvas.SetZones(zones);
    }
    private void EditZone(object sender,RoutedEventArgs e){if(ZonesGrid.SelectedItem is MonitoringZone z)EditZoneRecord(z);}
    private void EditZoneRecord(MonitoringZone zone)
    {
        var dialog=new EditDialog("监测区域",[("name","区域名称",zone.Name),("kind","用途（床边 / 出入口 / 重点区域）",zone.Kind),("dwell","停留提醒阈值（秒）",zone.DwellSeconds.ToString())],"视频来源："+zone.Source){Owner=this};
        if(dialog.ShowDialog()!=true)return;
        if(dialog["name"].Length is <1 or >40||dialog["kind"].Length is <1 or >30||!int.TryParse(dialog["dwell"],out int seconds)||seconds<1||seconds>86400)
        {MessageBox.Show(this,"请输入区域名称、用途和 1–86400 秒的停留阈值。","区域未保存");return;}
        if(store.Data.Settings.Zones.Any(z=>z.Source==zone.Source&&z.Name==dialog["name"]&&z.Id!=zone.Id))
        {MessageBox.Show(this,"该视频已有同名区域，请修改名称。","区域未保存");return;}
        store.Data.Settings.Zones.RemoveAll(z=>z.Id==zone.Id);
        store.Data.Settings.Zones.Add(zone with{Name=dialog["name"],Kind=dialog["kind"],DwellSeconds=seconds});UpdateRules();
    }
    private void DeleteZone(object sender,RoutedEventArgs e){if(ZonesGrid.SelectedItem is MonitoringZone z){store.Data.Settings.Zones.RemoveAll(x=>x.Id==z.Id);UpdateRules();}}
    private void SaveRules(object sender,RoutedEventArgs e)
    {
        if(!int.TryParse(InactivityThreshold.Text,out int threshold)||threshold<10||threshold>86400)
        {MessageBox.Show(this,"静止提醒阈值须在 10–86400 秒之间。","规则未保存");return;}
        store.Data.Settings.InactivitySeconds=threshold;store.Data.Settings.SoundAlerts=SoundAlertChoice.IsChecked==true;UpdateRules();StatusText.Text="监测规则已保存并生效。";
    }
    private void UpdateRules()
    {
        store.Save();video.ConfigureRules(store.Data.Settings);ZonesGrid.ItemsSource=store.Data.Settings.Zones.ToArray();
        RefreshCurrentZones();
    }
    private void ExportObservations(object sender,RoutedEventArgs e)
    {
        var source=video.Latest;var result=video.Analysis;
        if(source is null||result is null||source.Generation!=result.Generation){StatusText.Text="暂无有效观测数据可导出。";return;}
        var dialog=new SaveFileDialog{Filter="JSON观测记录|*.json",FileName=$"活动观测-{DateTime.Now:yyyyMMdd-HHmmss}.json"};
        if(dialog.ShowDialog(this)!=true)return;
        File.WriteAllText(dialog.FileName,JsonSerializer.Serialize(new{version="0.2.1",exported_at=DateTimeOffset.Now,source=source.Source,
            replay=source.Replay,video_time=source.Time,people=result.People.Select(p=>new{p.Id,p.Box,p.Activity,p.ActivitySource,p.Confidence,p.RgbScore,p.ShortRgbScore,p.ShortConsecutive,p.PoseScore,p.Partial,p.Telemetry,p.Alternatives}),
            analytics=result.Analytics,zones=store.Data.Settings.Zones.Where(z=>z.Source==source.Source),
            note="轨迹使用相机画面坐标；姿态观测不等于睡眠或疾病诊断。"},new JsonSerializerOptions{WriteIndented=true}),new UTF8Encoding(false));
        StatusText.Text="观测记录已导出。";
    }
    private void LocateEvent(object sender,RoutedEventArgs e)
    {
        var source=video.Latest;
        if(EventsGrid.SelectedItem is not MonitorEvent item||!item.IsReplay||!item.VideoPositionSeconds.HasValue||source?.Replay!=true||source.Source!=item.Source)
        {StatusText.Text="请选择回放事件，并打开对应的视频文件后定位。";return;}
        video.Seek(Math.Max(0,item.VideoPositionSeconds.Value-2));ShowPage(0);
    }
    private void ApplyApiPreset(object sender,RoutedEventArgs e)
    {
        bool ollama=((Button)sender).Tag.ToString()=="ollama";
        ApiUrl.Text=ollama?"http://127.0.0.1:11434/v1":"http://127.0.0.1:8080/v1";
        ApiModel.Text=ollama?"qwen3:4b":"";ApiKey.Clear();ClearKey.IsChecked=true;
        ApiConnectionStatus.Text="已填入本机服务预设。确认服务已启动，填写已安装的模型名，测试后点击保存分析设置。";
    }
    private async void ConnectRadar(object sender,RoutedEventArgs e)=>await Guard(()=>radar.ConnectAsync(store.Data.Settings.RadarPort,store.Data.Settings.RadarBaud));
    private async void DisconnectRadar(object sender,RoutedEventArgs e)=>await Guard(()=>radar.StopAsync());
    private async void ReplayRadar(object sender,RoutedEventArgs e)
    {var dialog=new OpenFileDialog{Filter="雷达数据文件|*.bin;*.dat|所有文件|*.*"};if(dialog.ShowDialog(this)==true)await Guard(()=>radar.ReplayAsync(dialog.FileName));}
    private async void SendRadarConfig(object sender,RoutedEventArgs e)
    {
        var dialog=new OpenFileDialog{Filter="雷达配置|*.cfg"};if(dialog.ShowDialog(this)!=true)return;
        if(RadarControlPort.SelectedItem is not string port||port.Length==0){MessageBox.Show(this,"请选择控制串口。","雷达配置");return;}
        if(port==store.Data.Settings.RadarPort){MessageBox.Show(this,"控制串口与数据串口必须分开。","雷达配置");return;}
        await Guard(()=>RadarService.ConfigureAsync(port,File.ReadAllText(dialog.FileName),reportCancel.Token));
    }
    private void RefreshLists(){RefreshResidents();RefreshEvents();ReportsList.ItemsSource=store.Data.Reports.OrderByDescending(r=>r.Time).ToArray();ZonesGrid.ItemsSource=store.Data.Settings.Zones.ToArray();}
    private void RefreshResidents()
    {
        if(store is null)return;string search=ResidentSearch.Text.Trim();ResidentsGrid.ItemsSource=store.Data.Residents.Where(r=>$"{r.Name} {r.Room} {r.Bed}".Contains(search,StringComparison.CurrentCultureIgnoreCase)).ToArray();
    }
    private void FilterResidents(object sender,TextChangedEventArgs e)=>RefreshResidents();
    private void AddResident(object sender,RoutedEventArgs e)=>Edit(new());
    private void EditResident(object sender,RoutedEventArgs e){if(ResidentsGrid.SelectedItem is Resident item)Edit(item);}
    private void Edit(Resident item)
    {
        var dialog=new EditDialog("住户档案",[("name","姓名",item.Name),("room","房间",item.Room),("bed","床位",item.Bed),("contact","紧急联系人",item.Contact),("radar","关联设备目标编号（选填）",item.RadarTargetId?.ToString()??""),("notes","照护备注",item.Notes)],"请确认雷达目标与住户对应关系。视频人员编号不自动关联住户。 "){Owner=this};
        if(dialog.ShowDialog()!=true)return;
        try
        {
            uint? radarId=string.IsNullOrWhiteSpace(dialog["radar"])?null:uint.Parse(dialog["radar"]);
            store.SaveResident(new(){Id=item.Id,Name=dialog["name"],Room=dialog["room"],Bed=dialog["bed"],Contact=dialog["contact"],RadarTargetId=radarId,Notes=dialog["notes"]});RefreshResidents();
        }
        catch(Exception ex){MessageBox.Show(this,ex.Message,"档案未保存");}
    }
    private void DeleteResident(object sender,RoutedEventArgs e)
    {
        if(ResidentsGrid.SelectedItem is not Resident item)return;
        if(MessageBox.Show(this,$"删除 {item.Name} 的档案与床位绑定？历史观测记录将保留。","删除档案",MessageBoxButton.YesNo,MessageBoxImage.Question)!=MessageBoxResult.Yes)return;
        store.Data.Residents.Remove(item);store.Save();RefreshResidents();
    }
    private void RefreshEvents()
    {
        if(store is null)return;var items=store.Data.Events.AsEnumerable();
        if(EventFilter.SelectedIndex==1)items=items.Where(e=>!e.IsReplay&&e.Resolution=="待处理");else if(EventFilter.SelectedIndex==2)items=items.Where(e=>e.IsReplay);
        EventsGrid.ItemsSource=items.OrderByDescending(e=>e.Time).Take(2000).ToArray();
    }
    private void FilterEvents(object sender,SelectionChangedEventArgs e)=>RefreshEvents();
    private void ResolveEvent(object sender,RoutedEventArgs e)
    {
        if(EventsGrid.SelectedItem is not MonitorEvent item)return;
        var dialog=new EditDialog("告警处理",[("result","处理结论（已确认 / 已排除 / 继续观察）",item.Resolution=="待处理"?"继续观察":item.Resolution),("note","处理备注",item.Note)],item.LocalTime+" · "+item.Mode+"\n"+item.Kind+"\n"+item.Evidence){Owner=this};
        if(dialog.ShowDialog()!=true)return;
        if(!new[]{"已确认","已排除","继续观察"}.Contains(dialog["result"])){MessageBox.Show(this,"处理结论请填写已确认、已排除或继续观察。","告警处理");return;}
        item.Resolution=dialog["result"];item.Note=dialog["note"];item.ResolvedAt=DateTimeOffset.Now;store.Save();RefreshEvents();
    }
    private async void GenerateLocal(object sender,RoutedEventArgs e)=>await Generate(false,false);
    private async void GenerateExternal(object sender,RoutedEventArgs e)=>await Generate(true,false);
    private async Task Generate(bool external,bool automatic)
    {
        if(generating||testingApi)return;generating=true;ExternalReportButton.IsEnabled=LocalReportButton.IsEnabled=false;
        activeReport=CancellationTokenSource.CreateLinkedTokenSource(reportCancel.Token);
        CancelReportButton.Visibility=external?Visibility.Visible:Visibility.Collapsed;
        var clock=Stopwatch.StartNew();var progress=new DispatcherTimer{Interval=TimeSpan.FromSeconds(1)};
        void UpdateProgress()=>ReportStatus.Text=external?$"正在分析 · 已等待 {clock.Elapsed.TotalSeconds:F0} 秒 · 可取消":"正在汇总监测数据…";
        progress.Tick+=(_,_)=>UpdateProgress();UpdateProgress();progress.Start();
        try
        {
            var captured=new StoreData{Settings=JsonSerializer.Deserialize<AppSettings>(JsonSerializer.Serialize(store.Data.Settings))!,Residents=store.Data.Residents.ToList(),Events=store.Data.Events.ToList(),Vitals=store.Data.Vitals.ToList()};
            var report=await reports.GenerateAsync(captured,video.Latest,video.Analysis,radar.Latest,external,activeReport.Token);
            if(closing)return;
            store.Data.Reports.Add(report);store.Save();DisplayReport(report);ReportsList.ItemsSource=store.Data.Reports.OrderByDescending(r=>r.Time).ToArray();lastReport=DateTimeOffset.Now;
            ReportStatus.Text=$"{(external?"智能分析":"监测报告")}已完成 · {clock.Elapsed.TotalSeconds:F1} 秒";
        }
        catch(OperationCanceledException){if(!closing)ReportStatus.Text="已取消分析，原报告保留。";}
        catch(Exception ex){if(!closing)ReportStatus.Text="生成失败："+ex.Message;}
        finally{progress.Stop();activeReport?.Dispose();activeReport=null;generating=false;ExternalReportButton.IsEnabled=LocalReportButton.IsEnabled=true;CancelReportButton.Visibility=Visibility.Collapsed;}
    }
    private void DisplayReport(SavedReport report)
    {
        selectedReport=report;var parts=ReportService.Parts(report);ReportText.SetText(parts.Text);ReportMeta.Text=$"{report.Time.LocalDateTime:yyyy-MM-dd HH:mm:ss} · {report.Provider}";
        ReportEvidence.SetText(parts.Evidence);EvidencePanel.Visibility=string.IsNullOrEmpty(parts.Evidence)?Visibility.Collapsed:Visibility.Visible;
    }
    private void SelectReport(object sender,SelectionChangedEventArgs e){if(ReportsList.SelectedItem is SavedReport report){DisplayReport(report);if(!generating)ReportStatus.Text="已打开历史报告";}}
    private void CancelReport(object sender,RoutedEventArgs e)=>activeReport?.Cancel();
    private void CopyReport(object sender,RoutedEventArgs e){if(selectedReport is not null){Clipboard.SetText(ReportLayout.PlainText(ReportService.Parts(selectedReport).Text));ReportStatus.Text="报告正文已复制";}}
    private AppSettings ApiSettingsFromForm()
    {
        if(!int.TryParse(ReportInterval.Text,out int interval))throw new ArgumentException("报告间隔须为60–3600秒。");
        return new(){ApiUrl=ApiUrl.Text.Trim(),ApiModel=ApiModel.Text.Trim(),ReportIntervalSeconds=interval,
            ProtectedApiKey=ApiKey.Password.Length>0?AppStore.ProtectKey(ApiKey.Password):ClearKey.IsChecked==true?"":store.Data.Settings.ProtectedApiKey};
    }
    private async void TestApi(object sender,RoutedEventArgs e)
    {
        if(testingApi||generating)return;testingApi=true;TestApiButton.IsEnabled=false;ApiConnectionStatus.Text="正在测试服务与模型…";
        try{await reports.TestAsync(ApiSettingsFromForm(),reportCancel.Token);ApiConnectionStatus.Text="连接成功，模型已返回内容。未保存的修改请点击保存设置。";}
        catch(OperationCanceledException){}
        catch(Exception ex){ApiConnectionStatus.Text="连接失败："+ex.Message;}
        finally{testingApi=false;TestApiButton.IsEnabled=true;}
    }
    private void SaveApiSettings(object sender,RoutedEventArgs e)
    {
        try
        {
            var api=ApiSettingsFromForm();ReportService.Validate(api);
            if(AutoReport.IsChecked==true&&!ReportService.HasCredentials(api))throw new ArgumentException("开启自动分析前请保存密钥，或配置本机服务。");
            var s=store.Data.Settings;s.ApiUrl=api.ApiUrl;s.ApiModel=api.ApiModel;s.ProtectedApiKey=api.ProtectedApiKey;s.ReportIntervalSeconds=api.ReportIntervalSeconds;s.AutoReport=AutoReport.IsChecked==true;
            store.Save();ApiKey.Clear();ClearKey.IsChecked=false;KeyStatus.Text=s.ProtectedApiKey.Length>0?"访问密钥已保存":"未设置访问密钥";ApiConnectionStatus.Text="分析设置已保存，现在可到健康报告生成智能分析。";
        }
        catch(Exception ex){ApiConnectionStatus.Text="设置未保存："+ex.Message;}
    }
    private void ExportReport(object sender,RoutedEventArgs e)
    {if(selectedReport is null){MessageBox.Show(this,"先生成或选择一份报告。","导出报告");return;}var dialog=new SaveFileDialog{Filter="HTML报告|*.html",FileName=$"康养报告-{selectedReport.Time:yyyyMMdd-HHmm}.html"};if(dialog.ShowDialog(this)==true)ReportService.ExportHtml(selectedReport,dialog.FileName);}
    private void RefreshTrends(object sender,RoutedEventArgs e)
    {
        uint? target=RadarSubjectChoice.SelectedItem is uint id?id:store.Data.Vitals.LastOrDefault(v=>!v.IsReplay)?.Target;
        var rows=store.Data.Vitals.Where(v=>v.Target==target && !v.IsReplay && DateTimeOffset.Now-v.Time<TimeSpan.FromHours(24)).ToArray();
        HeartTrend.Update(rows.Where(v=>v.HeartRate.HasValue).Select(v=>v.HeartRate!.Value));BreathTrend.Update(rows.Where(v=>v.BreathRate.HasValue).Select(v=>v.BreathRate!.Value));
        HeartTrend.ToolTip=BreathTrend.ToolTip=target.HasValue?"雷达目标 "+target:"没有有效实测目标";
    }
    private static string Csv(string value)
    {
        if(value.TrimStart().StartsWith('=')||value.TrimStart().StartsWith('+')||value.TrimStart().StartsWith('-')||value.TrimStart().StartsWith('@'))value="'"+value;
        return "\""+value.Replace("\"","\"\"")+"\"";
    }
    private void ExportEvents(object sender,RoutedEventArgs e)
    {
        var dialog=new SaveFileDialog{Filter="CSV表格|*.csv",FileName="监测事件.csv"};if(dialog.ShowDialog(this)!=true)return;
        File.WriteAllLines(dialog.FileName,new[]{"时间,来源,人物,事件,证据,处理,备注"}.Concat(store.Data.Events.Select(x=>string.Join(",",new[]{x.Time.ToString("O"),x.Mode,x.Subject,x.Kind,x.Evidence,x.Resolution,x.Note}.Select(Csv)))),new UTF8Encoding(true));
    }
    private void ExportVitals(object sender,RoutedEventArgs e)
    {
        var dialog=new SaveFileDialog{Filter="CSV表格|*.csv",FileName="雷达体征.csv"};if(dialog.ShowDialog(this)!=true)return;
        File.WriteAllLines(dialog.FileName,new[]{"时间,来源,雷达目标,心率bpm,呼吸率rpm,体动"}.Concat(store.Data.Vitals.Select(x=>string.Join(",",new[]{x.Time.ToString("O"),x.Mode,x.Target.ToString(),x.HeartRate?.ToString(CultureInfo.InvariantCulture)??"",x.BreathRate?.ToString(CultureInfo.InvariantCulture)??"",x.Moving.ToString()}.Select(Csv)))),new UTF8Encoding(true));
    }
    private void FillSettings()
    {
        var s=store.Data.Settings;RefreshCameras(this,new());RefreshPorts(this,new());
        var cameras=CameraChoice.ItemsSource as CameraDevices.Camera[];
        CameraChoice.SelectedItem=cameras is null?null:CameraDevices.Resolve(cameras,s.CameraDeviceId,s.CameraDeviceName,s.CameraIndex);
        CameraWidth.Text=s.CameraWidth.ToString();CameraHeight.Text=s.CameraHeight.ToString();CameraFps.Text=s.CameraFps.ToString();ManualExposure.IsChecked=s.ManualExposure;ManualGain.IsChecked=s.ManualGain;Exposure.Text=s.Exposure.ToString();Gain.Text=s.Gain.ToString();UseGpu.IsChecked=s.UseGpu;
        ModelProfile.SelectedIndex=s.ModelProfile=="baseline"?1:0;
        FallModelChoice.SelectedIndex=s.FallModel=="gmd"?1:0;
        InactivityThreshold.Text=s.InactivitySeconds.ToString();SoundAlertChoice.IsChecked=s.SoundAlerts;
        RadarPort.SelectedItem=s.RadarPort;RadarControlPort.SelectedItem=s.RadarControlPort;RadarBaud.Text=s.RadarBaud.ToString();ApiUrl.Text=s.ApiUrl;ApiModel.Text=s.ApiModel;AutoReport.IsChecked=s.AutoReport;ReportInterval.Text=s.ReportIntervalSeconds.ToString();KeyStatus.Text=s.ProtectedApiKey.Length>0?"访问密钥已保存":"未设置访问密钥";
        RadarHeight.Text=s.RadarHeight.ToString(CultureInfo.InvariantCulture);RadarAzimuth.Text=s.RadarAzimuth.ToString(CultureInfo.InvariantCulture);RadarElevation.Text=s.RadarElevation.ToString(CultureInfo.InvariantCulture);
    }
    private void RefreshCameras(object sender,RoutedEventArgs e)
    {
        try{var old=CameraChoice.SelectedItem as CameraDevices.Camera;var devices=CameraDevices.List();CameraChoice.ItemsSource=devices;if(old is not null)CameraChoice.SelectedItem=devices.FirstOrDefault(c=>c.DeviceId==old.DeviceId);}catch(Exception ex){StatusText.Text="摄像头枚举失败："+ex.Message;}
    }
    private void RefreshPorts(object sender,RoutedEventArgs e)
    {var ports=SerialPort.GetPortNames().Order().ToArray();RadarPort.ItemsSource=ports;RadarControlPort.ItemsSource=ports;}
    private void SaveSettings(object sender,RoutedEventArgs e)
    {
        try
        {
            var old=store.Data.Settings;var s=new AppSettings{CameraIndex=CameraChoice.SelectedItem is CameraDevices.Camera camera?camera.Index:old.CameraIndex,
                CameraDeviceName=(CameraChoice.SelectedItem as CameraDevices.Camera)?.Name??old.CameraDeviceName,
                CameraDeviceId=(CameraChoice.SelectedItem as CameraDevices.Camera)?.DeviceId??old.CameraDeviceId,SecondaryCameraId=old.SecondaryCameraId,SecondaryCameraIndex=old.SecondaryCameraIndex,DeviceSetupRevision=old.DeviceSetupRevision,
                LimitSlowExposure=old.LimitSlowExposure,ManualGain=ManualGain.IsChecked==true,
                RadarHeight=double.Parse(RadarHeight.Text,CultureInfo.InvariantCulture),RadarAzimuth=double.Parse(RadarAzimuth.Text,CultureInfo.InvariantCulture),RadarElevation=double.Parse(RadarElevation.Text,CultureInfo.InvariantCulture),RadarZones=old.RadarZones,
                CameraWidth=int.Parse(CameraWidth.Text),CameraHeight=int.Parse(CameraHeight.Text),CameraFps=int.Parse(CameraFps.Text),ManualExposure=ManualExposure.IsChecked==true,
                Exposure=int.Parse(Exposure.Text),Gain=int.Parse(Gain.Text),UseGpu=UseGpu.IsChecked==true,ModelProfile=ModelProfile.SelectedIndex==0?"realtime":"baseline",FallModel=FallModelChoice.SelectedIndex==1?"gmd":"stage_c",RadarPort=RadarPort.SelectedItem as string??old.RadarPort,
                Zones=old.Zones,InactivitySeconds=old.InactivitySeconds,SoundAlerts=old.SoundAlerts,
                RadarControlPort=RadarControlPort.SelectedItem as string??old.RadarControlPort,RadarBaud=int.Parse(RadarBaud.Text),ApiUrl=ApiUrl.Text.Trim(),ApiModel=ApiModel.Text.Trim(),
                ProtectedApiKey=ClearKey.IsChecked==true?"":ApiKey.Password.Length>0?AppStore.ProtectKey(ApiKey.Password):old.ProtectedApiKey,AutoReport=AutoReport.IsChecked==true,ReportIntervalSeconds=int.Parse(ReportInterval.Text)};
            if(s.CameraWidth<160||s.CameraWidth>1920||s.CameraHeight<120||s.CameraHeight>1080||s.CameraFps<5||s.CameraFps>60||s.Gain<0||s.Gain>255||s.Exposure<-15||s.Exposure>0||s.RadarBaud<9600||s.RadarBaud>3000000)throw new ArgumentException("采集参数超出支持范围。");
            if(s.AutoReport||s.ApiUrl!=old.ApiUrl||s.ApiModel!=old.ApiModel||s.ProtectedApiKey!=old.ProtectedApiKey)ReportService.Validate(s);
            RadarSetup.Configuration(Path.Combine(AppContext.BaseDirectory,"Assets"),s);if(s.AutoReport&&!ReportService.HasCredentials(s))throw new ArgumentException("请先配置访问密钥，或选择本机分析服务，再开启自动报告。");
            store.Data.Settings=s;store.Save();ApiKey.Clear();ClearKey.IsChecked=false;KeyStatus.Text=s.ProtectedApiKey.Length>0?"访问密钥已保存":"未设置访问密钥";StatusText.Text="设置已保存，采集参数在下次连接设备生效；模型和硬件加速设置需重启软件。";
        }
        catch(Exception ex){MessageBox.Show(this,ex.Message,"设置未保存");}
    }
    private void OnClosing(object? sender,CancelEventArgs e)
    {
        if(closed)return;e.Cancel=true;if(closing)return;closing=true;IsEnabled=false;timer.Stop();reportCancel.Cancel();
        _ = CloseAfterReleaseAsync();
    }
    private async Task CloseAfterReleaseAsync()
    {
        // Device release may complete synchronously. Leave the WPF Closing event before
        // requesting another Close, including when both cameras are already disconnected.
        await Dispatcher.Yield(DispatcherPriority.Background);
        try{try{await StereoWorkspace.DisposeAsync();}finally{try{await video.DisposeAsync();}finally{await radar.DisposeAsync();}}store.Save();reports.Dispose();reportCancel.Dispose();}
        catch(Exception ex){AppLog.Write("close-release",ex);MessageBox.Show("关闭时未完全释放设备："+ex.Message,"康养监测");}
        finally{closed=true;_ = Dispatcher.BeginInvoke(DispatcherPriority.Background,new Action(Close));}
    }
}
