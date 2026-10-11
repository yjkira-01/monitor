using System.Globalization;
using System.Windows;
using System.Windows.Media;

namespace CareMonitor.Views;
public sealed class WaveView : FrameworkElement
{
    public float[] Values { get; set; }=[];
    public string EmptyText { get; set; }="等待有效信号";
    public void Update(IEnumerable<float> values){Values=values.Where(float.IsFinite).TakeLast(600).ToArray();InvalidateVisual();}
    protected override void OnRender(DrawingContext dc)
    {
        var line=new Pen(new SolidColorBrush(Color.FromRgb(226,233,222)),1);
        for(int i=1;i<5;i++)dc.DrawLine(line,new(0,ActualHeight*i/5),new(ActualWidth,ActualHeight*i/5));
        if(Values.Length<2){dc.DrawText(new FormattedText(EmptyText,CultureInfo.CurrentCulture,FlowDirection.LeftToRight,new Typeface("Microsoft YaHei UI"),12,Brushes.Gray,VisualTreeHelper.GetDpi(this).PixelsPerDip),new(12,ActualHeight/2-8));return;}
        float min=Values.Min(),max=Values.Max(),range=Math.Max(.001f,max-min);var geometry=new StreamGeometry();
        using(var c=geometry.Open())
        {
            Point P(int i)=>new(i*ActualWidth/(Values.Length-1),ActualHeight-12-(Values[i]-min)/range*Math.Max(1,ActualHeight-24));
            c.BeginFigure(P(0),false,false);for(int i=1;i<Values.Length;i++)c.LineTo(P(i),true,false);
        }
        geometry.Freeze();dc.DrawGeometry(null,new Pen(new SolidColorBrush(Color.FromRgb(47,119,81)),2),geometry);
    }
}
