using System.Globalization;
using System.Windows;
using System.Windows.Media;
using CareMonitor.Models;
namespace CareMonitor.Views;
public sealed class RadarMapView : FrameworkElement
{
    private RadarObservation? observation;
    private RadarZone[] zones=[];
    public void Update(RadarObservation? value,RadarZone[] areas){observation=value;zones=areas;InvalidateVisual();}
    protected override void OnRender(DrawingContext dc)
    {
        double w=ActualWidth,h=ActualHeight;dc.DrawRectangle(new SolidColorBrush(Color.FromRgb(239,245,237)),null,new(0,0,w,h));
        Point Project(float x,float y)=>new((x+4)*w/8,h-y*h/6);
        var pen=new Pen(new SolidColorBrush(Color.FromRgb(214,226,215)),1);for(int i=-4;i<=4;i++)dc.DrawLine(pen,Project(i,0),Project(i,6));for(int i=0;i<=6;i++)dc.DrawLine(pen,Project(-4,i),Project(4,i));
        void Text(string text,Point p,Brush brush)=>dc.DrawText(new FormattedText(text,CultureInfo.CurrentCulture,FlowDirection.LeftToRight,new Typeface("Microsoft YaHei UI"),11,brush,VisualTreeHelper.GetDpi(this).PixelsPerDip),p);
        foreach(var z in zones){var a=Project(z.MinX,z.MaxY);var b=Project(z.MaxX,z.MinY);dc.DrawRectangle(new SolidColorBrush(Color.FromArgb(35,73,132,88)),new Pen(Brushes.SeaGreen,1),new Rect(a,b));Text(z.Name,a,Brushes.DarkGreen);}
        foreach(var target in observation?.Targets??[])
        {
            var line=new Pen(Brushes.SeaGreen,1.4);var trail=target.Trail;for(int i=1;i<trail.Length;i++)dc.DrawLine(line,Project(trail[i-1].X,trail[i-1].Y),Project(trail[i].X,trail[i].Y));
            var p=Project(target.Position.X,target.Position.Y);dc.DrawEllipse(Brushes.SeaGreen,new Pen(Brushes.White,2),p,6,6);Text($"目标 {target.Id} · {target.Speed:F2} m/s",new(p.X+9,p.Y-7),Brushes.DarkGreen);
        }
        Text("雷达跟踪坐标 · X：-4–4 m / Y：0–6 m",new(12,h-24),Brushes.DarkSlateGray);
    }
}
