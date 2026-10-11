using System.Globalization;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Input;
using CareMonitor.Models;

namespace CareMonitor.Views;
public sealed class VideoView : FrameworkElement
{
    private WriteableBitmap? bitmap;
    private PersonState[] people=[];
    private long sequence=-1,generation=-1;
    private double videoTime,observationTime;
    private MonitoringZone[] zones=[];
    private Point? start;
    private Point end;
    public bool DrawingZone { get; set; }
    public Guid? SelectedZoneId {get;set;}
    public event Action<Box>? ZoneDrawn;
    public event Action<Guid?>? ZoneSelected;
    public event Action? ZoneDeleteRequested;
    private Joint[] calibrationCorners=[];
    public void SetCalibrationCorners(Joint[] value){calibrationCorners=value;InvalidateVisual();}
    public void CancelDrawing(){start=null;DrawingZone=false;ReleaseMouseCapture();InvalidateVisual();}
    public void SetZones(MonitoringZone[] value){zones=value;if(!zones.Any(z=>z.Id==SelectedZoneId))SelectedZoneId=null;InvalidateVisual();}
    public void Update(VideoSnapshot? frame,AnalysisSnapshot? result)
    {
        if(frame is null){bitmap=null;people=[];InvalidateVisual();return;}
        if(frame.Sequence!=sequence)
        {
            if(bitmap is null||bitmap.PixelWidth!=frame.Width||bitmap.PixelHeight!=frame.Height)bitmap=new(frame.Width,frame.Height,96,96,PixelFormats.Bgr24,null);
            bitmap.WritePixels(new Int32Rect(0,0,frame.Width,frame.Height),frame.Pixels,frame.Width*3,0);sequence=frame.Sequence;
        }
        generation=frame.Generation;videoTime=frame.Time;
        people=result?.Generation==generation?result.People:[];observationTime=result?.Time??-10;InvalidateVisual();
    }
    protected override void OnRender(DrawingContext dc)
    {
        dc.DrawRectangle(new SolidColorBrush(Color.FromRgb(19,32,28)),null,new Rect(0,0,ActualWidth,ActualHeight));
        if(bitmap is null)
        {
            DrawText(dc,"视频未连接",24,new(ActualWidth/2-115,ActualHeight/2-35),Color.FromRgb(218,231,213));
            DrawText(dc,"选择视频文件或连接摄像头",13,new(ActualWidth/2-125,ActualHeight/2+9),Color.FromRgb(135,159,143));return;
        }
        double scale=Math.Min(ActualWidth/bitmap.PixelWidth,ActualHeight/bitmap.PixelHeight),w=bitmap.PixelWidth*scale,h=bitmap.PixelHeight*scale;
        double x=(ActualWidth-w)/2,y=(ActualHeight-h)/2;dc.DrawImage(bitmap,new(x,y,w,h));
        foreach(var corner in calibrationCorners)
        {var point=new Point(x+corner.X*w,y+corner.Y*h);dc.DrawEllipse(Brushes.White,new Pen(Brushes.DarkGreen,1.3),point,3,3);}
        foreach(var zone in zones)
        {
            var b=zone.Bounds;var rect=new Rect(x+b.X1*w,y+b.Y1*h,b.Width*w,b.Height*h);
            bool selected=zone.Id==SelectedZoneId;
            dc.DrawRectangle(new SolidColorBrush(Color.FromArgb(selected?(byte)48:(byte)22,96,169,114)),new Pen(new SolidColorBrush(Color.FromRgb(172,220,183)),selected?2:1){DashStyle=selected?DashStyles.Solid:DashStyles.Dash},rect);
            string label=zone.Name+(selected?" · 已选择":"");
            var title=new FormattedText(label,CultureInfo.CurrentCulture,FlowDirection.LeftToRight,new Typeface("Microsoft YaHei UI"),11,Brushes.White,VisualTreeHelper.GetDpi(this).PixelsPerDip);
            dc.DrawRectangle(new SolidColorBrush(Color.FromArgb(230,35,81,56)),null,new Rect(rect.Left,rect.Top,title.Width+12,24));dc.DrawText(title,new(rect.Left+6,rect.Top+4));
        }
        if(DrawingZone)
        {
            DrawText(dc,"拖动框选监测区域 · Esc 取消",13,new(x+12,y+12),Colors.White);
            if(start.HasValue)dc.DrawRectangle(new SolidColorBrush(Color.FromArgb(50,78,171,116)),new Pen(Brushes.White,1.5),new Rect(start.Value,end));
        }
        if(videoTime-observationTime>1.2)return;
        foreach(var person in people)
        {
            var b=person.Box;var rect=new Rect(x+b.X1*w,y+b.Y1*h,b.Width*w,b.Height*h);
            Color color=CareMonitor.Services.VideoService.IsFallAlert(person)?Color.FromRgb(249,191,96):Color.FromRgb(151,224,184);var brush=new SolidColorBrush(color);
            dc.DrawRectangle(null,new Pen(brush,1),rect);double length=Math.Min(22,rect.Width/5);
            foreach(var (point,dx,dy) in new[]{(rect.TopLeft,1,1),(rect.TopRight,-1,1),(rect.BottomLeft,1,-1),(rect.BottomRight,-1,-1)})
            {dc.DrawLine(new(brush,3),point,new(point.X+dx*length,point.Y));dc.DrawLine(new(brush,3),point,new(point.X,point.Y+dy*length));}
            string label=$"P{person.Id:00}  {person.Status}";double top=Math.Max(y,rect.Top-25);
            var formatted=new FormattedText(label,CultureInfo.CurrentCulture,FlowDirection.LeftToRight,new Typeface("Microsoft YaHei UI"),12,new SolidColorBrush(Color.FromRgb(24,46,35)),VisualTreeHelper.GetDpi(this).PixelsPerDip);
            dc.DrawRectangle(new SolidColorBrush(Color.FromArgb(240,237,247,229)),null,new Rect(rect.Left,top,Math.Min(w-rect.Left+x,formatted.Width+16),24));dc.DrawText(formatted,new(rect.Left+8,top+4));
        }
    }
    protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
    {
        if(bitmap is null)return;Focusable=true;Focus();
        Point point=e.GetPosition(this);
        double scale=Math.Min(ActualWidth/bitmap.PixelWidth,ActualHeight/bitmap.PixelHeight),w=bitmap.PixelWidth*scale,h=bitmap.PixelHeight*scale,x=(ActualWidth-w)/2,y=(ActualHeight-h)/2;
        if(!new Rect(x,y,w,h).Contains(point))return;
        if(!DrawingZone)
        {SelectedZoneId=zones.LastOrDefault(z=>new Rect(x+z.Bounds.X1*w,y+z.Bounds.Y1*h,z.Bounds.Width*w,z.Bounds.Height*h).Contains(point))?.Id;ZoneSelected?.Invoke(SelectedZoneId);InvalidateVisual();return;}
        start=point;end=start.Value;CaptureMouse();e.Handled=true;InvalidateVisual();
    }
    protected override void OnMouseMove(MouseEventArgs e){if(start.HasValue){end=e.GetPosition(this);InvalidateVisual();}}
    protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
    {
        if(!start.HasValue||bitmap is null)return;
        Point a=start.Value,b=e.GetPosition(this);start=null;DrawingZone=false;ReleaseMouseCapture();
        double scale=Math.Min(ActualWidth/bitmap.PixelWidth,ActualHeight/bitmap.PixelHeight),w=bitmap.PixelWidth*scale,h=bitmap.PixelHeight*scale,x=(ActualWidth-w)/2,y=(ActualHeight-h)/2;
        var box=new Box((float)Math.Clamp((Math.Min(a.X,b.X)-x)/w,0,1),(float)Math.Clamp((Math.Min(a.Y,b.Y)-y)/h,0,1),
            (float)Math.Clamp((Math.Max(a.X,b.X)-x)/w,0,1),(float)Math.Clamp((Math.Max(a.Y,b.Y)-y)/h,0,1));
        if(box.Width>=.02&&box.Height>=.02)ZoneDrawn?.Invoke(box);e.Handled=true;InvalidateVisual();
    }
    protected override void OnKeyDown(KeyEventArgs e){if(e.Key==Key.Escape){CancelDrawing();e.Handled=true;}if(e.Key==Key.Delete&&!DrawingZone&&SelectedZoneId.HasValue){ZoneDeleteRequested?.Invoke();e.Handled=true;}base.OnKeyDown(e);}
    private void DrawText(DrawingContext dc,string text,double size,Point point,Color color)=>dc.DrawText(new FormattedText(text,CultureInfo.CurrentCulture,FlowDirection.LeftToRight,new Typeface("Microsoft YaHei UI"),size,new SolidColorBrush(color),VisualTreeHelper.GetDpi(this).PixelsPerDip),point);
}
