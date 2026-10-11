using System.Net;
using System.Text;
using System.Text.RegularExpressions;

namespace CareMonitor.Services;

// A bounded, local report format. No HTML execution, remote images or external links.
public static partial class ReportLayout
{
    public sealed record Section(string Kind,string Text,int Level=0,int Number=0);
    [GeneratedRegex(@"^\s*(#{1,6})\s+(.+?)\s*#*\s*$")] private static partial Regex Heading();
    [GeneratedRegex(@"^\s*(?:[-+*]\s+|(?<number>\d{1,3})[.)、]\s+)(?<text>.+)$")] private static partial Regex Item();
    [GeneratedRegex(@"\*\*(.+?)\*\*|__(.+?)__|`([^`]+)`")] public static partial Regex Inline();
    public static Section[] Parse(string? text)
    {
        text??="";if(text.Length>150000)text=text[..150000];
        var result=new List<Section>();var paragraph=new StringBuilder();
        void Flush(){if(paragraph.Length==0)return;result.Add(new("paragraph",paragraph.ToString()));paragraph.Clear();}
        foreach(string line in text.Replace("\r","").Split('\n').Take(2500))
        {
            string value=line.Trim();if(value.Length==0){Flush();continue;}
            var heading=Heading().Match(value);var item=Item().Match(value);
            bool boldTitle=value.Length<140&&value.StartsWith("**")&&value.EndsWith("**")&&value.Count(c=>c=='*')==4;
            bool plainTitle=value is "视频监控" or "生命体征" or "当前概况" or "优先事项" or "建议" or "观测依据";
            if(heading.Success||boldTitle||plainTitle)
            {Flush();result.Add(new("heading",heading.Success?heading.Groups[2].Value:boldTitle?value[2..^2]:value,heading.Success?heading.Groups[1].Length:2));}
            else if(item.Success)
            {Flush();result.Add(new("item",item.Groups["text"].Value,Number:int.TryParse(item.Groups["number"].Value,out int number)?number:0));}
            else{if(paragraph.Length>0)paragraph.Append('\n');paragraph.Append(value);}
        }
        Flush();return result.ToArray();
    }
    public static string PlainInline(string value)=>Inline().Replace(value,m=>m.Groups.Cast<Group>().Skip(1).First(g=>g.Success).Value);
    public static string PlainText(string text)=>string.Join("\n\n",Parse(text).Select(s=>(s.Kind=="item"?(s.Number>0?$"{s.Number}. ":"• "):"")+PlainInline(s.Text)));
    public static string Html(string text)
    {
        var sb=new StringBuilder();
        foreach(var s in Parse(text))
        {
            string value=WebUtility.HtmlEncode(s.Text);
            value=Inline().Replace(value,m=>$"<strong>{m.Groups.Cast<Group>().Skip(1).First(g=>g.Success).Value}</strong>").Replace("\n","<br>");
            string tag=s.Kind=="heading"?"h2":"p";
            sb.Append('<').Append(tag).Append(s.Kind=="item"?" class=\"item\"":"").Append('>');
            if(s.Kind=="item")sb.Append(s.Number>0?$"{s.Number}. ":"• ");
            sb.Append(value).Append("</").Append(tag).Append('>');
        }
        return sb.ToString();
    }
}
