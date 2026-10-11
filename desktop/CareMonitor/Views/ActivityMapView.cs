using System.Globalization;
using System.Windows;
using System.Windows.Media;
using CareMonitor.Models;

namespace CareMonitor.Views;
public sealed class ActivityMapView : FrameworkElement
{
    private AnalyticsSnapshot? data;
    public bool Heatmap { get; set; }
    public void Update(AnalyticsSnapshot? value){data=value;InvalidateVisual();}
    protected override void OnRender(DrawingContext dc)
    {
        dc.DrawRectangle(new SolidColorBrush(Color.FromRgb(245,249,245)),null,new(0,0,ActualWidth,ActualHeight));
        for(int x=1;x<6;x++)dc.DrawLine(new(new SolidColorBrush(Color.FromRgb(227,234,227)),1),new(x*ActualWidth/6,0),new(x*ActualWidth/6,ActualHeight));
        for(int y=1;y<4;y++)dc.DrawLine(new(new SolidColorBrush(Color.FromRgb(227,234,227)),1),new(0,y*ActualHeight/4),new(ActualWidth,y*ActualHeight/4));
        if(data is null||data.Traces.Length==0){Text(dc,"等待人员轨迹",new(16,16));return;}
        if(Heatmap)
        {
            float max=Math.Max(1,data.Heatmap.Max());
            for(int y=0;y<data.HeatmapHeight;y++)for(int x=0;x<data.HeatmapWidth;x++)
            {
                float v=data.Heatmap[y*data.HeatmapWidth+x]/max;if(v<=0)continue;
                byte alpha=(byte)(30+190*Math.Sqrt(v));
                dc.DrawRectangle(new SolidColorBrush(Color.FromArgb(alpha,49,141,100)),null,
                    new(x*ActualWidth/data.HeatmapWidth,y*ActualHeight/data.HeatmapHeight,ActualWidth/data.HeatmapWidth+1,ActualHeight/data.HeatmapHeight+1));
            }
        }
        else foreach(var trace in data.Traces)
        {
            var brush=new SolidColorBrush(Palette(trace.Id));var points=trace.Points.TakeLast(200).ToArray();
            for(int i=1;i<points.Length;i++)if(points[i].Time-points[i-1].Time<=1.5)
                dc.DrawLine(new(brush,1.7),new(points[i-1].X*ActualWidth,points[i-1].Y*ActualHeight),new(points[i].X*ActualWidth,points[i].Y*ActualHeight));
            if(points.Length>0){var p=points[^1];dc.DrawEllipse(brush,null,new(p.X*ActualWidth,p.Y*ActualHeight),4,4);Text(dc,$"P{trace.Id:00}",new(Math.Clamp(p.X*ActualWidth+6,0,Math.Max(0,ActualWidth-40)),Math.Clamp(p.Y*ActualHeight-20,0,Math.Max(0,ActualHeight-20))));}
        }
    }
    private static Color Palette(int id)=>(id%4) switch{0=>Color.FromRgb(66,118,162),1=>Color.FromRgb(40,120,80),2=>Color.FromRgb(173,113,53),_=>Color.FromRgb(123,94,143)};
    private void Text(DrawingContext dc,string text,Point p)=>dc.DrawText(new FormattedText(text,CultureInfo.CurrentCulture,FlowDirection.LeftToRight,
        new Typeface("Microsoft YaHei UI"),11,new SolidColorBrush(Color.FromRgb(72,94,77)),VisualTreeHelper.GetDpi(this).PixelsPerDip),p);
}
