using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using CareMonitor.Services;

namespace CareMonitor.Views;

public sealed class ReportView : RichTextBox
{
    public string Text {get;private set;}="";
    public ReportView()
    {
        IsReadOnly=true;IsDocumentEnabled=false;BorderThickness=new(0);Background=Brushes.White;
        VerticalScrollBarVisibility=ScrollBarVisibility.Auto;HorizontalScrollBarVisibility=ScrollBarVisibility.Disabled;
        Padding=new(26,20,26,24);FontFamily=new("Microsoft YaHei UI");FontSize=14;
        SetText("连接监测设备后，可生成健康报告。");
    }
    public void SetText(string value)
    {
        Text=value;var doc=new FlowDocument{PagePadding=new(0),FontSize=14,Foreground=new SolidColorBrush(Color.FromRgb(38,62,52))};
        foreach(var section in ReportLayout.Parse(value))
        {
            var p=new Paragraph{Margin=new(0,0,0,12),LineHeight=25};
            if(section.Kind=="heading")
            {p.FontSize=section.Level<=1?21:18;p.FontWeight=FontWeights.SemiBold;p.Margin=new(0,14,0,12);p.Foreground=new SolidColorBrush(Color.FromRgb(30,92,64));}
            if(section.Kind=="item")
            {p.Margin=new(8,0,0,10);p.Inlines.Add(new Run(section.Number>0?$"{section.Number}. ":"• "){Foreground=new SolidColorBrush(Color.FromRgb(48,117,84))});}
            int cursor=0;
            foreach(System.Text.RegularExpressions.Match m in ReportLayout.Inline().Matches(section.Text))
            {
                if(m.Index>cursor)p.Inlines.Add(new Run(section.Text[cursor..m.Index]));
                p.Inlines.Add(new Bold(new Run(m.Groups.Cast<System.Text.RegularExpressions.Group>().Skip(1).First(g=>g.Success).Value)));
                cursor=m.Index+m.Length;
            }
            if(cursor<section.Text.Length)p.Inlines.Add(new Run(section.Text[cursor..]));doc.Blocks.Add(p);
        }
        Document=doc;ScrollToHome();
    }
}
