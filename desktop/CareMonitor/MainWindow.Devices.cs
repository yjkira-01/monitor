using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using CareMonitor.Models;
using CareMonitor.Services;
using CareMonitor.Views;
using Microsoft.Win32;
namespace CareMonitor;
public partial class MainWindow
{
    private bool radarSetupBusy;
    private string appliedRadarConfiguration="";
    private void ApplyDeviceDefaults()
    {
        var s=store.Data.Settings;if(s.DeviceSetupRevision>=1)return;
        // One-time migration for the requested early fall model and modern person detector.
        s.ModelProfile="realtime";s.FallModel="gmd";
        var cameras=CameraDevices.List().Where(c=>c.Name.Contains("C60E")).ToArray();
        if(s.CameraDeviceId.Length==0&&cameras.Length>0)
        {var c=cameras[0];s.CameraIndex=c.Index;s.CameraDeviceName=c.Name;s.CameraDeviceId=c.DeviceId;}
        if(s.SecondaryCameraId.Length==0&&cameras.Length>1){s.SecondaryCameraId=cameras[1].DeviceId;s.SecondaryCameraIndex=cameras[1].Index;}
        var ports=RadarDevices.Detect();if(ports is not null){if(s.RadarControlPort.Length==0)s.RadarControlPort=ports.Control;if(s.RadarPort.Length==0)s.RadarPort=ports.Data;}
        s.DeviceSetupRevision=1;store.Save();
    }
    private void DetectRadar(object sender,RoutedEventArgs e)
    {
        RefreshPorts(sender,e);var ports=RadarDevices.Detect();
        if(ports is null){RadarSetupText.Text="未找到唯一的 CP2105 控制/数据端口组合，请手动选择。";return;}
        RadarControlPort.SelectedItem=ports.Control;RadarPort.SelectedItem=ports.Data;RadarBaud.Text="921600";
        RadarSetupText.Text=$"SR112：控制 {ports.Control} / 数据 {ports.Data}，点击“初始化并连接”启用。";
    }
    private AppSettings RadarInput()
    {
        if(RadarControlPort.SelectedItem is not string control||RadarPort.SelectedItem is not string data||control==data)throw new ArgumentException("请选择不同的雷达控制端口和数据端口。");
        var s=store.Data.Settings;
        double height=double.Parse(RadarHeight.Text,CultureInfo.InvariantCulture),azimuth=double.Parse(RadarAzimuth.Text,CultureInfo.InvariantCulture),elevation=double.Parse(RadarElevation.Text,CultureInfo.InvariantCulture);
        var candidate=new AppSettings{RadarHeight=height,RadarAzimuth=azimuth,RadarElevation=elevation};RadarSetup.Configuration(Path.Combine(AppContext.BaseDirectory,"Assets"),candidate);
        s.RadarControlPort=control;s.RadarPort=data;s.RadarBaud=921600;s.RadarHeight=height;s.RadarAzimuth=azimuth;s.RadarElevation=elevation;store.Save();return s;
    }
    private async void IdentifyRadar(object sender,RoutedEventArgs e)
    {
        if(radarSetupBusy)return;radarSetupBusy=true;try
        {var s=RadarInput();var info=await RadarSetup.IdentifyAsync(s.RadarControlPort,reportCancel.Token);RadarSetupText.Text=info.Supported?"IWR6843AOP · SDK 3.5 · 支持当前生命体征与跟踪配置。":"固件与内置配置不匹配，请导入与当前固件对应的配置文件。";}
        catch(Exception ex){RadarSetupText.Text="固件检查失败："+ex.Message;AppLog.Write("radar-identify",ex);}finally{radarSetupBusy=false;}
    }
    private async void InitializeRadar(object sender,RoutedEventArgs e)
    {
        if(radarSetupBusy)return;radarSetupBusy=true;
        try
        {
            var s=RadarInput();string configuration=RadarSetup.Configuration(Path.Combine(AppContext.BaseDirectory,"Assets"),s);
            if(appliedRadarConfiguration==configuration&&radar.Latest is {} current&&(DateTimeOffset.Now-current.Time).TotalSeconds<3)
            {RadarSetupText.Text="当前配置已生效，设备数据持续更新。";ShowPage(1);return;}
            await radar.StopAsync();RadarSetupText.Text="正在检查固件…";
            var firmware=await RadarSetup.IdentifyAsync(s.RadarControlPort,reportCancel.Token);
            if(!firmware.Supported)throw new InvalidOperationException("当前固件与内置 AOP 生命体征配置不兼容，未发送配置；请导入配套配置。");
            radar.ConfigureRules(s);await radar.ConnectAsync(s.RadarPort,s.RadarBaud);
            int count=await RadarSetup.ConfigureAsync(s.RadarControlPort,configuration,reportCancel.Token,(i,name)=>RadarSetupText.Text=$"初始化 {i}/37 · {name}");
            RadarSetupText.Text="配置已确认，正在检查数据帧…";
            var watch=Stopwatch.StartNew();while(radar.Latest is null&&watch.Elapsed.TotalSeconds<6)await Task.Delay(100,reportCancel.Token);
            if(radar.Latest is null)throw new IOException("配置已确认，但未收到有效数据帧。请检查数据端口、供电和设备安装。");
            appliedRadarConfiguration=configuration;
            RadarSetupText.Text=$"{count} 条命令已确认；收到实测帧 {radar.Latest.Frame.Number}。体征按设备窗口更新，需目标稳定。";ShowPage(1);
        }
        catch(Exception ex){await radar.StopAsync();RadarSetupText.Text="初始化未完成："+ex.Message+" 若检查固件也无响应，请重新给设备通电。";AppLog.Write("radar-setup",ex);}finally{radarSetupBusy=false;}
    }
    private void EditRadarZones(object sender,RoutedEventArgs e)
    {
        var s=store.Data.Settings;
        var selection=new EditDialog("雷达监测区域",[("name","区域名称","床位区域"),("kind","用途（床位 / 出入口 / 重点区域）","床位"),("x1","X 最小值（米）","-1"),("x2","X 最大值（米）","1"),("y1","Y 最近距离（米）","1"),("y2","Y 最远距离（米）","3"),("z1","Z 最低高度（米）","0"),("z2","Z 最高高度（米）","2")],"按实际床位位置填写跟踪坐标。失去目标不视为离床；再次保存同名区域可替换原范围。"){Owner=this};
        if(selection.ShowDialog()!=true)return;
        try
        {
            float N(string name)=>float.Parse(selection[name],CultureInfo.InvariantCulture);
            var z=new RadarZone(Guid.NewGuid(),selection["name"],N("x1"),N("x2"),N("y1"),N("y2"),N("z1"),N("z2"),selection["kind"]);
            if(z.Name.Length is <1 or >40||new[]{z.MinX,z.MaxX,z.MinY,z.MaxY,z.MinZ,z.MaxZ}.Any(v=>!float.IsFinite(v)||Math.Abs(v)>20)||z.MinX>=z.MaxX||z.MinY>=z.MaxY||z.MinZ>=z.MaxZ)throw new ArgumentException("请填写有效的区域名称和空间范围。");
            s.RadarZones.RemoveAll(a=>a.Name==z.Name);s.RadarZones.Add(z);store.Save();radar.ConfigureRules(s);RadarSetupText.Text="雷达区域已保存："+z.Name;
        }catch(Exception ex){MessageBox.Show(this,ex.Message,"区域未保存");}
    }
    private void ClearRadarZones(object sender,RoutedEventArgs e)
    {store.Data.Settings.RadarZones.Clear();store.Save();radar.ConfigureRules(store.Data.Settings);RadarSetupText.Text="雷达监测区域已清空。";}
    private void ExportRadar(object sender,RoutedEventArgs e)
    {
        var snapshot=radar.Latest;if(snapshot is null||(DateTimeOffset.Now-snapshot.Time).TotalSeconds>3){RadarSetupText.Text="暂无有效雷达数据。";return;}
        var dialog=new SaveFileDialog{Filter="雷达监测数据|*.json",FileName=$"雷达观测-{DateTime.Now:yyyyMMdd-HHmmss}.json"};if(dialog.ShowDialog(this)!=true)return;
        try{File.WriteAllText(dialog.FileName,JsonSerializer.Serialize(new{snapshot.Time,snapshot.Source,snapshot.Replay,frame=snapshot.Frame,observation=radar.Observation,diagnostics=radar.Diagnostics,zones=store.Data.Settings.RadarZones,
            point_targets_reference="TLV1011 belongs to the previous point-cloud frame; not assigned to current points"},new JsonSerializerOptions{WriteIndented=true,IncludeFields=true}));RadarSetupText.Text="雷达监测数据已导出。";}catch(Exception ex){MessageBox.Show(this,ex.Message,"导出未完成");}
    }
}
