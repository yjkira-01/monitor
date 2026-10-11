using System.Windows;
using System.Windows.Media;

namespace CareMonitor.Views;
public sealed class AttentionView : FrameworkElement
{
    private float[] values=[];
    public void Update(float[]? data){values=data??[];InvalidateVisual();}
    protected override void OnRender(DrawingContext dc)
    {
        int side=(int)Math.Sqrt(values.Length);if(side<1||side*side!=values.Length)return;
        for(int row=0;row<side;row++)for(int col=0;col<side;col++)
        {
            float value=Math.Clamp(values[row*side+col],0,1);
            var color=Color.FromRgb((byte)(230-180*value),(byte)(238-126*value),(byte)(221-145*value));
            dc.DrawRectangle(new SolidColorBrush(color),null,new(col*ActualWidth/side,row*ActualHeight/side,ActualWidth/side+.5,ActualHeight/side+.5));
        }
    }
}
