using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace CareMonitor.Views;
public sealed class EditDialog : Window
{
    private readonly Dictionary<string,TextBox> fields=[];
    public string this[string name]=>fields[name].Text.Trim();
    public EditDialog(string title,IEnumerable<(string Key,string Label,string Value)> inputs,string? evidence=null)
    {
        Style=(Style)Application.Current.FindResource(typeof(Window));Title=title;Width=520;SizeToContent=SizeToContent.Height;MaxHeight=850;WindowStartupLocation=WindowStartupLocation.CenterOwner;ResizeMode=ResizeMode.NoResize;
        var content=new StackPanel{Margin=new(28)};Content=new ScrollViewer{Content=content,VerticalScrollBarVisibility=ScrollBarVisibility.Auto};
        content.Children.Add(new TextBlock{Text=title,FontSize=22,FontWeight=FontWeights.SemiBold,Margin=new(0,0,0,12)});
        if(evidence is not null)content.Children.Add(new TextBlock{Text=evidence,Margin=new(0,0,0,18),Foreground=Brushes.DimGray});
        foreach(var (key,label,value) in inputs)
        {
            content.Children.Add(new TextBlock{Text=label,Margin=new(0,12,0,6),Foreground=Brushes.DimGray});
            var box=new TextBox{Text=value,MinHeight=36,MaxLength=key=="notes"?4000:500};
            if(key is "notes" or "note"){box.Height=90;box.AcceptsReturn=true;box.TextWrapping=TextWrapping.Wrap;}
            fields[key]=box;content.Children.Add(box);
        }
        var buttons=new StackPanel{Orientation=Orientation.Horizontal,HorizontalAlignment=HorizontalAlignment.Right,Margin=new(0,24,0,0)};
        var save=new Button{Content="保存",IsDefault=true,Style=(Style)Application.Current.FindResource("Primary")};save.Click+=(_,_)=>{DialogResult=true;};
        ButtonAssist.SetIcon(save,(Geometry)Application.Current.FindResource("IconSave"));
        var cancel=new Button{Content="取消",IsCancel=true,Style=(Style)Application.Current.FindResource("Subtle")};
        ButtonAssist.SetIcon(cancel,(Geometry)Application.Current.FindResource("IconDismiss"));
        buttons.Children.Add(cancel);buttons.Children.Add(save);content.Children.Add(buttons);
    }
}
