using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using CareMonitor.Models;
namespace CareMonitor.Views;
public sealed class RadarTargetCard : StackPanel
{
    private readonly TextBlock title=new(){FontSize=15,FontWeight=FontWeights.SemiBold};
    private readonly TextBlock vitals=new(){Margin=new(0,10,0,0)};
    private readonly TextBlock state=new(){FontSize=12,Foreground=Brushes.Gray,Margin=new(0,8,0,0)};
    private readonly TextBlock activity=new(){FontSize=11,Foreground=Brushes.DarkSlateGray,Margin=new(0,8,0,0)};
    private readonly TextBlock details=new(){FontSize=11,Foreground=Brushes.Gray,Margin=new(0,6,0,0)};
    public Expander Details { get; }
    public RadarTargetCard()
    {Margin=new(0,0,0,22);Children.Add(title);Children.Add(vitals);Children.Add(state);Children.Add(activity);Details=new(){Header="设备详情",FontSize=11,Margin=new(0,10,0,0),Content=details};Children.Add(Details);}
    public void Update(RadarTrack track,VitalSample? sample,Resident? resident,RadarTargetObservation? target)
    {
        title.Text=$"设备目标 {track.Id} · {resident?.Name??"未关联住户"}";
        vitals.Text=$"心率 {sample?.HeartRate?.ToString("F1")??"—"} bpm    呼吸 {sample?.BreathRate?.ToString("F1")??"—"} rpm";
        state.Text=sample is null?"暂无有效体征数据":sample.Moving?"体动中 · 信号待确认":"静息状态";
        activity.Text=target is null?"":$"区域 {(target.Zone.Length>0?target.Zone:"区域外")} · 观测 {target.ObservedSeconds:F0}s\n静息 {target.RestSeconds:F0}s · 运动片段 {target.MotionEpisodes} 次\n速度 {target.Speed:F2} m/s · 加速度 {target.Acceleration:F2} m/s²"+(target.Height is null?"\n高度范围：当前未输出":$"\n高度 {target.Height.MinZ:F2}–{target.Height.MaxZ:F2} m")+(target.HeightDrop?"\n高度突降，请核查":"");
        details.Text=$"位置 ({track.Position.X:F2}, {track.Position.Y:F2}, {track.Position.Z:F2}) m\n跟踪置信度（固件值）{track.Confidence:F3}";
    }
}
