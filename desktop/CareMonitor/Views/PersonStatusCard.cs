using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using CareMonitor.Models;

namespace CareMonitor.Views;
public sealed class PersonStatusCard : StackPanel
{
    private readonly TextBlock title=new(){FontWeight=FontWeights.SemiBold,FontSize=14};
    private readonly TextBlock visibility=new(){FontSize=12,Foreground=Brushes.Gray,Margin=new(0,8,0,0)};
    private readonly TextBlock updated=new(){FontSize=11,Foreground=Brushes.Gray,Margin=new(0,6,0,0)};
    private readonly TextBlock detail=new(){FontSize=11,Foreground=Brushes.Gray,Margin=new(0,7,0,0),TextWrapping=TextWrapping.Wrap};
    public Expander Details { get; }
    public PersonStatusCard()
    {
        Margin=new(0,0,0,16);
        Details=new(){Header="识别详情",Content=detail,FontSize=11,Margin=new(0,10,0,0)};
        Children.Add(title);Children.Add(visibility);Children.Add(updated);Children.Add(Details);
        Children.Add(new Border{Height=1,Background=new SolidColorBrush(Color.FromRgb(227,233,222)),Margin=new(0,16,0,0)});
    }
    public void Update(PersonState person)
    {
        title.Text=$"P{person.Id:00}   {person.Status}";
        title.Foreground=new SolidColorBrush(CareMonitor.Services.VideoService.IsFallAlert(person)?Color.FromRgb(169,105,39):Color.FromRgb(40,104,77));
        visibility.Text=person.Partial?"画面不完整":"人员可见";
        updated.Text=$"更新时间 {TimeSpan.FromSeconds(person.Time):mm\\:ss}";
        string Score(float? value)=>value.HasValue?value.Value.ToString("F3"):"—";
        string time=person.PredictionTime<0?"处理中":TimeSpan.FromSeconds(person.PredictionTime).ToString(@"mm\:ss");
        detail.Text=$"姿态状态 {Labels.Activity(person.Activity)}\n姿态依据 {person.ActivitySource}\n姿态置信度 {person.Confidence:F2}\n长窗口跌倒分值 {Score(person.RgbScore)}\n最近2秒跌倒分值 {Score(person.ShortRgbScore)}\n短窗口连续确认 {person.ShortConsecutive} 次\n检测模型 {person.Detector}\n检测置信度 {person.DetectionScore:F2}\n分析进度 {time}\n分析分值用于辅助复核，不代表风险概率。";
        if(person.MotionEvidence.Length>0)detail.Text+="\n动作依据 "+person.MotionEvidence;
        if(person.Telemetry is { } t)
        {
            visibility.Text+=(t.Zone=="区域外"?"":$" · {t.Zone}");
            detail.Text+=$"\n已观测 {t.ObservedSeconds:F0} 秒\n位置静止 {t.StillSeconds:F0} 秒\n区域停留 {t.DwellSeconds:F0} 秒";
        }
        if(person.Alternatives is {Length:>0} candidates)detail.Text+="\n动作候选（预训练模型，待验证）\n"+string.Join("\n",candidates.Select(c=>$"{c.Name}  {c.Score:P0}"));
    }
}
