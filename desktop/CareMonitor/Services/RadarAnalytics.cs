using System.Numerics;
using CareMonitor.Models;
namespace CareMonitor.Services;
public sealed class RadarAnalytics
{
    private sealed class State
    {
        public double Seen,Observed,Rest,LastAlert=-100;
        public bool Moving;
        public int Motion;
        public string Zone="";
        public readonly List<(double Time,float Height)> Heights=[];
        public readonly Queue<Vector3> Trail=new();
    }
    private readonly Dictionary<uint,State> targets=[];
    private double previous=-1,total;
    private int enters,exits;
    public void Reset(){targets.Clear();previous=-1;total=0;enters=exits=0;}
    public RadarObservation Update(RadarSnapshot snapshot,double seconds,RadarZone[] zones,Action<MonitorEvent>? publish=null)
    {
        double dt=previous>=0&&seconds-previous is >0 and <=.5?seconds-previous:0;previous=seconds;total+=dt;
        var result=new List<RadarTargetObservation>();
        foreach(var track in snapshot.Frame.Tracks)
        {
            bool existed=targets.TryGetValue(track.Id,out var state);state??=new();targets[track.Id]=state;
            bool continuous=existed&&seconds-state.Seen is >0 and <=.5;
            string zone=string.Join("、",zones.Where(z=>z.Contains(track.Position)).Select(z=>z.Name));
            float speed=track.Velocity.Length();bool moving=speed>=.1;
            if(continuous)
            {
                state.Observed+=dt;if(!moving)state.Rest+=dt;if(moving&&!state.Moving)state.Motion++;
                if(zone!=state.Zone)
                {
                    if(state.Zone.Length>0){exits++;Emit("离开区域",state.Zone);}
                    if(zone.Length>0){enters++;Emit("进入区域",zone);}
                }
            }
            else state.Heights.Clear();
            var height=snapshot.Frame.Heights?.FirstOrDefault(h=>h.Id==track.Id);bool drop=false;
            if(height is not null)
            {
                state.Heights.Add((seconds,height.MaxZ));state.Heights.RemoveAll(h=>seconds-h.Time>2.5);
                var past=state.Heights.Where(h=>seconds-h.Time>=.5).ToArray();
                drop=past.Length>=3&&height.MaxZ<.6f*past.Max(h=>h.Height)&&past.Max(h=>h.Height)-height.MaxZ>.45f;
                if(drop&&seconds-state.LastAlert>30){state.LastAlert=seconds;Emit("雷达高度突降 · 待核查",$"当前最高点 {height.MaxZ:F2} m；2.5 秒内观测高度明显下降，需要核查画面。");}
            }
            state.Zone=zone;state.Seen=seconds;state.Moving=moving;
            state.Trail.Enqueue(track.Position);while(state.Trail.Count>240)state.Trail.Dequeue();
            result.Add(new(track.Id,track.Position,zone,state.Observed,state.Rest,state.Motion,speed,track.Acceleration.Length(),height,drop,state.Trail.ToArray()));
            void Emit(string kind,string evidence)=>publish?.Invoke(new(){Source=snapshot.Source,Subject="雷达 "+track.Id,Kind=kind,Evidence=evidence,IsReplay=snapshot.Replay});
        }
        foreach(uint id in targets.Keys.Where(id=>seconds-targets[id].Seen>3).ToArray())targets.Remove(id);
        return new(snapshot.Frame.Number,snapshot.Frame.Presence,result.ToArray(),total,enters,exits);
    }
}
