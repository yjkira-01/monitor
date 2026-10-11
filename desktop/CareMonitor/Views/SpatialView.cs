using System.Globalization;
using System.Numerics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using CareMonitor.Models;
using Vector=System.Windows.Vector;

namespace CareMonitor.Views;

public sealed class SpatialView : Grid
{
    private readonly CloudCanvas canvas=new();
    private readonly TextBlock summary=new(){Foreground=new SolidColorBrush(Color.FromRgb(49,87,68)),FontSize=11,Margin=new(16,12,12,0)};
    public bool PointMode {get;set;}
    public SpatialView()
    {
        Background=Brushes.White;ClipToBounds=true;
        RowDefinitions.Add(new(){Height=GridLength.Auto});RowDefinitions.Add(new());RowDefinitions.Add(new(){Height=GridLength.Auto});
        var toolbar=new WrapPanel{Margin=new(12,8,12,0)};
        void Command(string title,Action operation){var button=new Button{Content=title};button.Click+=(_,_)=>operation();toolbar.Children.Add(button);}
        Command("透视",()=>canvas.Preset(false));Command("俯视",()=>canvas.Preset(true));Command("适配范围",canvas.Fit);Command("复位",Reset);
        var choice=new ComboBox{ItemsSource=new[]{"按径向速度着色","按SNR着色"},SelectedIndex=0,Width=170,Margin=new(8,0,0,0)};
        choice.SelectionChanged+=(_,_)=>{canvas.Snr=choice.SelectedIndex==1;canvas.InvalidateVisual();};toolbar.Children.Add(choice);
        Children.Add(toolbar);var pane=new Grid();SetRow(pane,1);Children.Add(pane);pane.Children.Add(canvas);pane.Children.Add(summary);summary.VerticalAlignment=VerticalAlignment.Top;summary.IsHitTestVisible=false;
        var legend=new TextBlock{Text="拖动旋转 · 滚轮缩放 · 坐标单位 m · X 横向 / Y 前向 / Z 高度\n速度：蓝色负向 / 绿色静止 / 橙色正向；SNR：浅绿低 / 深绿高；圆环与标签为设备跟踪目标。",FontSize=10,Foreground=new SolidColorBrush(Color.FromRgb(80,108,94)),Margin=new(16,6,16,10),TextWrapping=TextWrapping.Wrap};
        SetRow(legend,2);Children.Add(legend);Reset();
    }
    public void Reset()=>canvas.Preset(false);
    public void SetRadar(RadarFrame frame,RadarObservation? observation=null)
    {
        canvas.Frame=frame;canvas.Observation=observation;canvas.InvalidateVisual();
        summary.Text=$"当前帧 {frame.Number} · {frame.Points.Length} 个散射点 · {frame.Tracks.Length} 个跟踪目标\n雷达输出坐标 · Z=0为参考平面";
    }
    public void SetRadar(Vector3[] points)=>SetRadar(new RadarFrame(0,points,[],[]));
    public void Clear(){canvas.Frame=null;canvas.Observation=null;canvas.InvalidateVisual();summary.Text="等待点云数据";}
    private sealed class CloudCanvas : FrameworkElement
    {
        public RadarFrame? Frame;
        public RadarObservation? Observation;
        public bool Snr;
        private double yaw=.4,pitch=.45,zoom=1;
        private Vector3 center=new(0,2,0);
        private Point? drag;
        private Vector3 right,up,depth;
        private double scale;
        public CloudCanvas()
        {
            ClipToBounds=true;Focusable=true;
            MouseLeftButtonDown+=(_,e)=>{drag=e.GetPosition(this);CaptureMouse();e.Handled=true;};
            MouseLeftButtonUp+=(_,_)=>{drag=null;ReleaseMouseCapture();};LostMouseCapture+=(_,_)=>drag=null;
            MouseMove+=(_,e)=>{if(drag is not Point previous)return;var p=e.GetPosition(this);yaw+=(p.X-previous.X)*.008;pitch=Math.Clamp(pitch+(p.Y-previous.Y)*.006,.08,Math.PI/2);drag=p;InvalidateVisual();};
            MouseWheel+=(_,e)=>{zoom=Math.Clamp(zoom*Math.Exp(e.Delta/120d*.13),.3,4);InvalidateVisual();e.Handled=true;};
        }
        public void Preset(bool top){yaw=top?0:.4;pitch=top?Math.PI/2:.45;zoom=1;center=new(0,2,0);InvalidateVisual();}
        public void Fit()
        {
            var points=(Frame?.Points??[]).Concat(Frame?.Tracks.Select(t=>t.Position)??[]).Where(Finite).Append(Vector3.Zero).ToArray();
            if(points.Length<=1)return;
            var min=points.Aggregate(Vector3.Min);var max=points.Aggregate(Vector3.Max);center=(min+max)/2;
            zoom=Math.Clamp(5/Math.Max(2,(max-min).Length()),.3,3);InvalidateVisual();
        }
        private Point Project(Vector3 p){var v=p-center;return new(ActualWidth/2+Vector3.Dot(v,right)*scale,ActualHeight*.55-Vector3.Dot(v,up)*scale);}
        protected override void OnRender(DrawingContext dc)
        {
            dc.DrawRectangle(Brushes.White,null,new Rect(0,0,ActualWidth,ActualHeight));
            scale=Math.Min(ActualWidth/8,ActualHeight/5)*zoom;
            depth=new((float)(Math.Sin(yaw)*Math.Cos(pitch)),(float)(-Math.Cos(yaw)*Math.Cos(pitch)),(float)Math.Sin(pitch));
            right=new((float)Math.Cos(yaw),(float)Math.Sin(yaw),0);up=Vector3.Normalize(Vector3.Cross(depth,right));
            var grid=new Pen(new SolidColorBrush(Color.FromRgb(223,233,226)),1);
            for(int i=-4;i<=4;i++)dc.DrawLine(grid,Project(new(i,0,0)),Project(new(i,6,0)));
            for(int i=0;i<=6;i++)dc.DrawLine(grid,Project(new(-4,i,0)),Project(new(4,i,0)));
            void Axis(Vector3 end,string label,Color color){Point p=Project(end);dc.DrawLine(new(new SolidColorBrush(color),1.8),Project(Vector3.Zero),p);Text(dc,label,p+new Vector(5,-10),color);}
            Axis(new(2,0,0),"X 2 m",Color.FromRgb(177,105,87));Axis(new(0,4,0),"Y 4 m",Color.FromRgb(77,125,172));Axis(new(0,0,2),"Z 2 m",Color.FromRgb(44,132,99));
            for(int i=1;i<=5;i++)Text(dc,i+" m",Project(new(-.1f,i,0)),Color.FromRgb(128,145,134),9);
            Point origin=Project(Vector3.Zero);dc.DrawEllipse(new SolidColorBrush(Color.FromRgb(45,96,70)),null,origin,5,5);Text(dc,"雷达原点",origin+new Vector(8,8),Color.FromRgb(49,87,68));
            if(Frame is not { } frame){Text(dc,"等待点云数据",new(20,50),Color.FromRgb(110,133,120));return;}
            foreach(var target in Observation?.Targets??[])
            {
                var path=target.Trail.Where(Finite).TakeLast(90).ToArray();var pen=new Pen(new SolidColorBrush(Color.FromArgb(120,43,123,85)),1.4);
                for(int i=1;i<path.Length;i++)dc.DrawLine(pen,Project(path[i-1]),Project(path[i]));
            }
            var details=frame.PointDetails??frame.Points.Select(p=>new RadarPoint(p,0)).ToArray();
            foreach(var p in details.Where(p=>Finite(p.Position)).Take(1600).OrderBy(p=>Vector3.Dot(p.Position,depth)))
            {
                Color color=frame.PointDetails is null?Color.FromRgb(145,155,147):Snr?p.Snr is float snr?Color.FromRgb((byte)Math.Clamp(177-snr*3,33,177),(byte)Math.Clamp(207-snr*2,105,207),(byte)Math.Clamp(179-snr*3,73,179)):Color.FromRgb(145,155,147):p.Doppler<-.1?Color.FromRgb(73,137,196):p.Doppler>.1?Color.FromRgb(206,135,65):Color.FromRgb(54,142,96);
                var brush=new SolidColorBrush(color);brush.Freeze();Point projected=Project(p.Position);double radius=p.Snr is float s?Math.Clamp(2+s/18,2,4):2.7;
                dc.DrawEllipse(new SolidColorBrush(Color.FromArgb(30,color.R,color.G,color.B)),null,projected,radius+2,radius+2);dc.DrawEllipse(brush,null,projected,radius,radius);
            }
            foreach(var t in frame.Tracks.Where(t=>Finite(t.Position)).Take(16))
            {
                Point p=Project(t.Position);var color=Color.FromRgb(28,94,61);dc.DrawEllipse(null,new(new SolidColorBrush(color),2),p,10,10);
                Text(dc,$"R{t.Id:00} · {t.Position.X:F2}, {t.Position.Y:F2}, {t.Position.Z:F2} m",p+new Vector(14,-10),color);
                if(t.Velocity.Length()>.03)dc.DrawLine(new(new SolidColorBrush(color),2),p,Project(t.Position+t.Velocity));
            }
        }
        private static bool Finite(Vector3 v)=>float.IsFinite(v.X)&&float.IsFinite(v.Y)&&float.IsFinite(v.Z)&&v.Length()<40;
        private void Text(DrawingContext dc,string text,Point p,Color color,double size=11)=>dc.DrawText(new FormattedText(text,CultureInfo.CurrentCulture,FlowDirection.LeftToRight,new Typeface("Microsoft YaHei UI"),size,new SolidColorBrush(color),VisualTreeHelper.GetDpi(this).PixelsPerDip),p);
    }
}
