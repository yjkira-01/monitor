using CareMonitor.Models;

namespace CareMonitor.Services;

/// <summary>Timestamp-based observation records, bounded in memory. No estimated vital signs.</summary>
public sealed class MonitoringAnalytics
{
    private sealed class Record
    {
        public double Last=-1, Observed, Still, ZoneStart,Dwell;
        public float X,Y;
        public Guid? Zone;
        public bool DwellSent,StillSent;
        public List<TracePoint> Trace=[];
        public List<ActivityInterval> Activities=[];
        public Dictionary<string,double> Durations=[];
    }
    private readonly Dictionary<int,Record> records=[];
    private readonly HashSet<int> unique=[];
    private readonly float[] heat=new float[24*16];
    private double observed,lastTime=-1;
    private int entries,exits;
    public void Reset(){records.Clear();unique.Clear();Array.Clear(heat);observed=0;lastTime=-1;entries=exits=0;}
    public void InvalidateZones(){foreach(var r in records.Values){r.Zone=null;r.Dwell=0;r.DwellSent=false;}}
    public static MonitoringZone? ZoneFor(Box box,MonitoringZone[] zones,string source)
    {
        float x=(box.X1+box.X2)/2,y=box.Y2;
        return zones.Where(z=>z.Source==source).Select(z=>
        {
            float area=Math.Max(0,Math.Min(box.X2,z.Bounds.X2)-Math.Max(box.X1,z.Bounds.X1))*Math.Max(0,Math.Min(box.Y2,z.Bounds.Y2)-Math.Max(box.Y1,z.Bounds.Y1));
            return (Zone:z,Overlap:area/Math.Max(.0001f,box.Width*box.Height),Foot:x>=z.Bounds.X1&&x<=z.Bounds.X2&&y>=z.Bounds.Y1&&y<=z.Bounds.Y2);
        }).Where(z=>z.Foot||z.Overlap>=.2f).OrderByDescending(z=>z.Overlap).Select(z=>z.Zone).FirstOrDefault();
    }
    public AnalysisSnapshot Update(AnalysisSnapshot result, VideoSnapshot source, MonitoringZone[] zones, int inactivity,
        Action<MonitorEvent> publish)
    {
        if(lastTime>=0&&result.Time<lastTime)Reset();
        double sceneDelta=lastTime<0?0:Math.Clamp(result.Time-lastTime,0,1);
        observed+=sceneDelta;lastTime=result.Time;
        var people=result.People.Select(p=>
        {
            if(!records.TryGetValue(p.Id,out var r)){r=new();records[p.Id]=r;unique.Add(p.Id);}
            float x=(p.Box.X1+p.Box.X2)/2, y=p.Box.Y2;
            double gap=r.Last<0?0:result.Time-r.Last;
            double dt=gap>.65?0:Math.Clamp(gap,0,.65);
            if(gap>.65){r.Still=0;r.StillSent=false;r.Zone=null;r.Dwell=0;}
            double moved=Math.Sqrt(Math.Pow(x-r.X,2)+Math.Pow(y-r.Y,2));
            r.Observed+=dt;
            if(r.Last>=0&&moved<.012)r.Still+=dt;else {r.Still=0;r.StillSent=false;}
            var zone=ZoneFor(p.Box,zones,source.Source);
            if(r.Zone!=zone?.Id)
            {
                if(r.Zone.HasValue&&gap<=1.5)
                {
                    exits++;var old=zones.FirstOrDefault(z=>z.Id==r.Zone);
                    if(old is not null)Emit("离开区域",$"离开 {old.Name}",p,source,publish);
                }
                r.Zone=zone?.Id;r.ZoneStart=result.Time;r.Dwell=0;r.DwellSent=false;
                if(zone is not null){entries++;Emit("进入区域",$"进入 {zone.Name}（{zone.Kind}）",p,source,publish);}
            }
            else if(zone is not null)r.Dwell+=dt;
            double dwell=zone is null?0:r.Dwell;
            if(zone is not null&&dwell>=zone.DwellSeconds&&!r.DwellSent)
            {r.DwellSent=true;Emit("区域停留提醒",$"在 {zone.Name} 连续观测 {dwell:F0} 秒；设置阈值 {zone.DwellSeconds} 秒。",p,source,publish);}
            if(inactivity>0&&r.Still>=inactivity&&!r.StillSent)
            {r.StillSent=true;Emit("长时间静止提醒",$"画面位置连续 {r.Still:F0} 秒变化较小，请核查。静止不代表失去意识或睡眠。",p,source,publish);}
            string activity=string.IsNullOrEmpty(p.Activity)?"unknown":p.Activity;
            r.Durations[activity]=r.Durations.GetValueOrDefault(activity)+dt;
            if(r.Activities.Count>0&&r.Activities[^1].Activity==activity&&gap<=1.5)
                r.Activities[^1]=r.Activities[^1] with{End=result.Time};
            else r.Activities.Add(new(result.Time,result.Time,activity));
            if(r.Activities.Count>120)r.Activities.RemoveAt(0);
            if(r.Trace.Count==0||result.Time-r.Trace[^1].Time>=.4)r.Trace.Add(new(result.Time,x,y));
            r.Trace.RemoveAll(v=>result.Time-v.Time>1800);
            if(r.Trace.Count>1800)r.Trace.RemoveRange(0,r.Trace.Count-1800);
            int hx=Math.Clamp((int)(x*24),0,23),hy=Math.Clamp((int)(y*16),0,15);
            heat[hy*24+hx]+= (float)dt;
            r.Last=result.Time;r.X=x;r.Y=y;
            return p with{Telemetry=new(r.Observed,r.Still,zone?.Name??"区域外",dwell,new Dictionary<string,double>(r.Durations))};
        }).ToArray();
        foreach(var id in records.Keys.Where(id=>result.Time-records[id].Last>300).ToArray())records.Remove(id);
        foreach(var id in records.OrderByDescending(p=>p.Value.Last).Skip(24).Select(p=>p.Key).ToArray())records.Remove(id);
        return result with{People=people,Analytics=new(records.Select(p=>new PersonTrace(p.Key,p.Value.Trace.ToArray(),p.Value.Activities.ToArray())).ToArray(),
            (float[])heat.Clone(),24,16,observed,unique.Count,entries,exits,people.Count(p=>p.Telemetry?.Zone!="区域外"))};
    }
    private static void Emit(string kind,string detail,PersonState person,VideoSnapshot source,Action<MonitorEvent> publish)=>
        publish(new(){Source=source.Source,Subject=$"P{person.Id:00}",Kind=kind,IsReplay=source.Replay,VideoPositionSeconds=source.Time,
            Resolution=kind is "进入区域" or "离开区域"?"已记录":"待处理",
            Evidence=$"视频位置 {source.Time:F2} 秒；{detail} 依据为视频观测，不自动关联住户身份。"});
}
