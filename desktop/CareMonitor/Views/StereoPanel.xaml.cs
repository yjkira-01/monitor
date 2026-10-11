using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using CareMonitor.Models;
using CareMonitor.Services;
using Microsoft.Win32;
namespace CareMonitor.Views;
public partial class StereoPanel : UserControl
{
    private AppStore store=null!;
    private VideoService primary=null!,secondary=null!;
    private readonly StereoPairing pairing=new();
    private StereoCalibration? calibration;
    private StereoCalibrationSession? samples;
    private bool busy,active;
    public bool Connected=>active;
    private long renderedLeft=-1,renderedRight=-1;
    private DateTimeOffset cornersUntil;
    public StereoPanel(){InitializeComponent();}
    public void Initialize(AppStore data,VideoService source)
    {
        store=data;primary=source;var s=store.Data.Settings;secondary=new(Path.Combine(AppContext.BaseDirectory,"Assets"),s.UseGpu,true,s.FallModel,source);
        Refresh(this,new());string path=Path.Combine(store.DirectoryPath,"stereo-calibration.json");
        if(File.Exists(path))try{calibration=StereoCalibration.Load(path);CalibrationText.Text=$"已有标定 · 基线 {calibration.Baseline*100:F1} cm · 误差 {calibration.ReprojectionError:F2} px";}catch(Exception ex){CalibrationText.Text="已有标定无法使用："+ex.Message;}
    }
    private void Refresh(object sender,RoutedEventArgs e)
    {
        if(store is null)return;
        try
        {
            var devices=CameraDevices.List();string left=(LeftChoice.SelectedItem as CameraDevices.Camera)?.DeviceId??store.Data.Settings.CameraDeviceId,right=(RightChoice.SelectedItem as CameraDevices.Camera)?.DeviceId??store.Data.Settings.SecondaryCameraId;
            LeftChoice.ItemsSource=devices;RightChoice.ItemsSource=devices;
            LeftChoice.SelectedItem=devices.FirstOrDefault(c=>c.DeviceId==left)??devices.FirstOrDefault(c=>c.Name.Contains("C60E"));
            RightChoice.SelectedItem=devices.FirstOrDefault(c=>c.DeviceId==right)??devices.FirstOrDefault(c=>c.Name.Contains("C60E")&&c.DeviceId!=(LeftChoice.SelectedItem as CameraDevices.Camera)?.DeviceId);
        }catch(Exception ex){ConnectionText.Text="设备列表不可用："+ex.Message;}
    }
    private async Task Run(Func<Task> operation)
    {if(busy)return;busy=true;try{await operation();}catch(Exception ex){ConnectionText.Text=ex.Message;AppLog.Write("stereo-operation",ex);}finally{busy=false;}}
    private async void Start(object sender,RoutedEventArgs e)=>await Run(async()=>
    {
        if(LeftChoice.SelectedItem is not CameraDevices.Camera left||RightChoice.SelectedItem is not CameraDevices.Camera right||left.DeviceId==right.DeviceId)throw new ArgumentException("请选择两台不同的摄像头。");
        await secondary.InitializeAsync();await StopAsync();
        var s=store.Data.Settings;s.CameraDeviceId=left.DeviceId;s.CameraDeviceName=left.Name;s.CameraIndex=left.Index;s.SecondaryCameraId=right.DeviceId;s.SecondaryCameraIndex=right.Index;store.Save();
        bool compatible=calibration is not null&&calibration.LeftDeviceId==left.DeviceId&&calibration.RightDeviceId==right.DeviceId&&calibration.Width==s.CameraWidth&&calibration.Height==s.CameraHeight;
        pairing.Reset(compatible?calibration:null);
        await primary.StartCameraAsync(s);
        try{await secondary.StartCameraAsync(new(){CameraIndex=right.Index,CameraDeviceId=right.DeviceId,CameraDeviceName=right.Name,CameraWidth=s.CameraWidth,CameraHeight=s.CameraHeight,CameraFps=s.CameraFps,ManualExposure=s.ManualExposure,ManualGain=s.ManualGain,Exposure=s.Exposure,Gain=s.Gain});active=true;}
        catch{await primary.StopAsync();throw;}
        compatible=compatible&&primary.Latest is {} lf&&secondary.Latest is {} rf&&lf.Width==calibration!.Width&&lf.Height==calibration.Height&&rf.Width==calibration.Width&&rf.Height==calibration.Height;
        pairing.Reset(compatible?calibration:null);
        LeftChoice.IsEnabled=RightChoice.IsEnabled=false;ConnectionText.Text="正在打开两路画面…";if(!compatible)CalibrationText.Text="当前设备或分辨率与标定不一致，请重新标定。";
    });
    private async void Stop(object sender,RoutedEventArgs e)=>await Run(async()=>{await StopAsync();await primary.StopAsync();});
    public async Task StopAsync(){active=false;await secondary.StopAsync();pairing.Reset();samples=null;Columns.IsEnabled=Rows.IsEnabled=Square.IsEnabled=LeftChoice.IsEnabled=RightChoice.IsEnabled=true;Scene.Update([]);LeftVideo.SetCalibrationCorners([]);RightVideo.SetCalibrationCorners([]);LeftVideo.Update(null,null);RightVideo.Update(null,null);ConnectionText.Text="双目已断开";PositionText.Text="连接双摄并完成标定后显示三维定位";PeopleText.Text="";SampleProgress.Value=0;SampleCount.Text="已采集 0 / 建议20组（最少12组）";CalculateButton.IsEnabled=false;CaptureButton.IsEnabled=false;}
    public async ValueTask DisposeAsync()=>await secondary.DisposeAsync();
    public void Tick()
    {
        if(!active)return;
        var l=primary.Latest;var r=secondary.Latest;
        CaptureButton.IsEnabled=active&&!busy;CalculateButton.IsEnabled=active&&!busy&&samples?.Count>=12;
        if(DateTimeOffset.Now>cornersUntil){LeftVideo.SetCalibrationCorners([]);RightVideo.SetCalibrationCorners([]);cornersUntil=DateTimeOffset.MaxValue;}
        if(l?.Replay==true||l?.Ended==true||r?.Ended==true)
        {
            pairing.Reset();Scene.Update([]);
            if(!busy){active=false;_ = Run(async()=>{await StopAsync();await primary.StopAsync();ConnectionText.Text="双目采集已停止，请检查两台摄像头后重新连接。";});}return;
        }
        if(IsVisible){LeftVideo.Update(l,primary.Analysis);RightVideo.Update(r,secondary.Analysis);}
        var result=pairing.Update(primary.Observation,secondary.Observation);
        if(l is null||r is null){ConnectionText.Text=primary.CaptureStatus+" / "+secondary.CaptureStatus;return;}
        ConnectionText.Text=$"左 {l.Fps:F1} FPS / 右 {r.Fps:F1} FPS · "+(result is null?"等待时间配对":$"配对差 {result.PairDeltaMs:F0} ms");
        if(!IsVisible)return;
        if(result is null){Scene.Update([]);PeopleText.Text="等待有效配对；时间差超过 80 ms 或数据过期时暂停定位。";return;}
        if(result.LeftSequence==renderedLeft&&result.RightSequence==renderedRight)return;renderedLeft=result.LeftSequence;renderedRight=result.RightSequence;
        Scene.Update(result.People);PositionText.Text=result.Status;
        PeopleText.Text=result.People.Length==0?"尚无有效三维目标。请完成标定，并让两路画面都能看到对应人员。":string.Join("\n\n",result.People.Select(p=>$"P{p.Id:00} · {Labels.Activity(p.Activity)}\n位置 ({p.Position.X:F2}, {p.Position.Y:F2}, {p.Position.Z:F2}) m\n实测关节 {p.ObservedJoints}/17 · 重投影 {p.ErrorPixels:F2} px"));
    }
    private async void Capture(object sender,RoutedEventArgs e)=>await Run(async()=>
    {
        if(!active||primary.Latest is not { } l||secondary.Latest is not { } r)throw new InvalidOperationException("请先连接双摄像头。");
        samples??=new(int.Parse(Columns.Text),int.Parse(Rows.Text),double.Parse(Square.Text,CultureInfo.InvariantCulture));
        CalibrationText.Text="正在查找两路棋盘内角点，请保持棋盘静止…";
        try{CalibrationText.Text=await Task.Run(()=>samples.Add(l,r));Columns.IsEnabled=Rows.IsEnabled=Square.IsEnabled=false;}
        finally
        {
            LeftVideo.SetCalibrationCorners(samples.LeftCorners);RightVideo.SetCalibrationCorners(samples.RightCorners);cornersUntil=DateTimeOffset.Now.AddSeconds(2.5);
            SampleProgress.Value=samples.Count;SampleCount.Text=$"已采集 {samples.Count} / 建议20组（最少12组）\n{samples.Coverage}";
            CalculateButton.IsEnabled=samples.Count>=12;
        }
    });
    private async void Calibrate(object sender,RoutedEventArgs e)=>await Run(async()=>
    {
        if(samples is null||LeftChoice.SelectedItem is not CameraDevices.Camera l||RightChoice.SelectedItem is not CameraDevices.Camera r)throw new InvalidOperationException("请先采集角点样本。");
        ConnectionText.Text="正在计算标定…";var result=await Task.Run(()=>samples.Solve(l.DeviceId,r.DeviceId));
        result.Save(Path.Combine(store.DirectoryPath,"stereo-calibration.json"));calibration=result;pairing.Reset(result);
        CalibrationText.Text=$"标定已保存 · {result.Samples} 组 · 基线 {result.Baseline*100:F1} cm · 误差 {result.ReprojectionError:F2} px";
    });
    private void ClearSamples(object sender,RoutedEventArgs e){if(busy)return;samples=null;Columns.IsEnabled=Rows.IsEnabled=Square.IsEnabled=true;LeftVideo.SetCalibrationCorners([]);RightVideo.SetCalibrationCorners([]);SampleProgress.Value=0;SampleCount.Text="已采集 0 / 建议20组（最少12组）";CalculateButton.IsEnabled=false;CalibrationText.Text="标定样本已清空，已保存的标定参数保留。";}
    private void UseProvidedBoard(object sender,RoutedEventArgs e){if(busy)return;ClearSamples(sender,e);Columns.Text="9";Rows.Text="6";Square.Text="25";CalibrationText.Text="已恢复附带棋盘：10×7个方格、9×6个内角点、25mm格长。打印后请用尺量格长。";}
    private void Import(object sender,RoutedEventArgs e)
    {
        if(busy)return;var dialog=new OpenFileDialog{Filter="双目标定|*.json"};if(dialog.ShowDialog()!=true)return;
        try
        {
            var c=StereoCalibration.Load(dialog.FileName);var s=store.Data.Settings;
            string left=(LeftChoice.SelectedItem as CameraDevices.Camera)?.DeviceId??s.CameraDeviceId,right=(RightChoice.SelectedItem as CameraDevices.Camera)?.DeviceId??s.SecondaryCameraId;
            int width=active?primary.Latest?.Width??s.CameraWidth:s.CameraWidth,height=active?primary.Latest?.Height??s.CameraHeight:s.CameraHeight;
            if(c.LeftDeviceId!=left||c.RightDeviceId!=right)throw new InvalidDataException("请按标定时的左、右设备顺序选择两台相机。设备固定位置改变后需要重新采集标定。");
            if(c.Width!=width||c.Height!=height||active&&(secondary.Latest?.Width!=c.Width||secondary.Latest?.Height!=c.Height))throw new InvalidDataException($"标定分辨率 {c.Width}×{c.Height} 与当前采集 {width}×{height} 不同。请恢复对应分辨率后导入。");
            c.Save(Path.Combine(store.DirectoryPath,"stereo-calibration.json"));calibration=c;pairing.Reset(active?c:null);CalibrationText.Text=$"标定已导入 · {c.Samples} 组 · 基线 {c.Baseline*100:F1} cm · 误差 {c.ReprojectionError:F2} px";
        }
        catch(Exception ex){ConnectionText.Text=ex.Message;}
    }
    private void ExportCalibration(object sender,RoutedEventArgs e)
    {
        if(calibration is null){CalibrationText.Text="首次使用请先连接双摄像头，采集至少12组角点并计算标定，再导出保存。";return;}
        var dialog=new SaveFileDialog{Filter="双目标定参数|*.json",FileName="双目标定参数.json"};if(dialog.ShowDialog()!=true)return;
        try{calibration.Save(dialog.FileName);CalibrationText.Text="标定参数已导出，可用“导入标定”恢复；相机移动后请重新标定。";}catch(Exception ex){CalibrationText.Text=ex.Message;}
    }
    private void SaveChessboard(object sender,RoutedEventArgs e)
    {
        var dialog=new SaveFileDialog{Filter="可打印棋盘|*.svg",FileName="双目标定棋盘-25mm.svg"};if(dialog.ShowDialog()!=true)return;
        try{File.Copy(Path.Combine(AppContext.BaseDirectory,"Assets/Radar/stereo-chessboard-25mm.svg"),dialog.FileName,true);CalibrationText.Text="棋盘已保存：A4横向，按100%原尺寸打印，实测格长后填写，固定到平整硬板。";}catch(Exception ex){CalibrationText.Text=ex.Message;}
    }
    private void Export(object sender,RoutedEventArgs e)
    {
        var result=pairing.Latest;if(result is null){ConnectionText.Text="暂无有效定位结果。";return;}var dialog=new SaveFileDialog{Filter="三维定位|*.json",FileName="双目定位.json"};if(dialog.ShowDialog()!=true)return;
        try{File.WriteAllText(dialog.FileName,JsonSerializer.Serialize(new{time=DateTimeOffset.Now,result.PairDeltaMs,result.Status,coordinate_system="left_camera_x_right_y_down_z_forward_m",people=result.People.Select(p=>new{p.Id,p.RightId,position=new[]{p.Position.X,p.Position.Y,p.Position.Z},joints=p.Joints.Select(j=>j is null?null:new[]{j.Value.X,j.Value.Y,j.Value.Z}),p.ObservedJoints,p.ErrorPixels,p.Activity})},new JsonSerializerOptions{WriteIndented=true}));ConnectionText.Text="定位结果已导出。";}catch(Exception ex){ConnectionText.Text=ex.Message;}
    }
}
