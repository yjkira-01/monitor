using CareMonitor.Models;

namespace CareMonitor.Services;

/// <summary>Per-track measured-time observations, separate from trained video scores.</summary>
public sealed class PostureTimeline
{
    private sealed record Sample(double Time,Joint[] Joints,string Activity,float Confidence,float HipX,float HipY,float ShoulderY,float Torso,float Tilt,float? FootY);
    private readonly List<Sample> history=[];
    private string stable="unknown",candidate="unknown",transition="";
    private double candidateSince=-1,lastValid=-1,transitionUntil=-1,fallUntil=-1;
    private float stableConfidence;
    private double recentCollapseUntil=-1;
    private Sample? supportReference;
    private bool supportLocked;
    public string Activity { get; private set; }="unknown";
    public float Confidence { get; private set; }
    public bool FallMotion { get; private set; }
    public bool ShortWindowSupported { get; private set; }
    public string MotionEvidence { get; private set; }="";
    public void Reset(){history.Clear();stable=candidate=Activity="unknown";transition="";candidateSince=lastValid=transitionUntil=fallUntil=recentCollapseUntil=-1;Confidence=stableConfidence=0;FallMotion=ShortWindowSupported=supportLocked=false;supportReference=null;MotionEvidence="";}
    public void Update(Joint[] joints,Box box,double time,float aspect)
    {
        if(!double.IsFinite(time)||history.Count>0&&(time<=history[^1].Time||time-history[^1].Time>1))Reset();
        var raw=PostureObservation.Classify(joints,box,aspect:aspect);
        var sides=joints.Length==17?new[]{(Shoulder:5,Hip:11),(Shoulder:6,Hip:12)}.Where(s=>joints[s.Shoulder].Confidence>=.4f&&joints[s.Hip].Confidence>=.4f).ToArray():[];
        var shoulders=sides.Select(s=>s.Shoulder).ToArray();var hips=sides.Select(s=>s.Hip).ToArray();
        if(shoulders.Length>0&&hips.Length>0)
        {
            float sx=shoulders.Average(i=>joints[i].X*aspect),sy=shoulders.Average(i=>joints[i].Y);
            float hx=hips.Average(i=>joints[i].X*aspect),hy=hips.Average(i=>joints[i].Y);
            float torso=MathF.Sqrt((hx-sx)*(hx-sx)+(hy-sy)*(hy-sy));
            float tilt=MathF.Atan2(Math.Abs(hx-sx),Math.Abs(hy-sy))*180/MathF.PI;
            var feet=new[]{15,16}.Where(i=>joints[i].Confidence>=.6f&&float.IsFinite(joints[i].Y)&&joints[i].Y>=hy+torso*.65f).Select(i=>joints[i].Y).ToArray();
            var current=new Sample(time,joints,raw.Activity,raw.Confidence,hx,hy,sy,torso,tilt,feet.Length>0?feet.Max():null);
            history.RemoveAll(s=>time-s.Time>4);history.Add(current);if(history.Count>64)history.RemoveAt(0);
            // Freeze support before lying down. A transient seated-looking pose on a bed
            // must not redefine the support height after the movement has started.
            if(raw.Activity is "standing" or "walk"&&stable is "standing" or "walk"&&time-candidateSince>=.8&&feet.Length>0&&tilt<35)
            {supportLocked=false;supportReference=null;}
            if(!supportLocked&&sides.Max(s=>Math.Min(joints[s.Shoulder].Confidence,joints[s.Hip].Confidence))>=.65f&&torso>=.025f&&current.FootY.HasValue&&
                (supportReference is null||current.FootY>supportReference.FootY))supportReference=current;
            if(raw.Activity=="lying")supportLocked=true;
            // Require two physical changes and a subsequent horizontal observation.
            // This is motion evidence, never a fabricated classifier probability.
            if(raw.Activity=="lying"&&torso>=.025f)
            {
                var before=history.Where(s=>time-s.Time is >=.2 and <=1.2&&s.Activity is "standing" or "walk"&&s.Torso>=.025f)
                    .FirstOrDefault(s=>(hy-s.HipY)/s.Torso>.65f&&(sy-s.ShoulderY)/s.Torso>.65f&&tilt-s.Tilt>30&&(hy-s.HipY)/s.Torso/(time-s.Time)>1.2f);
                if(before is not null)
                {fallUntil=time+2.5;MotionEvidence=$"{time-before.Time:F2} 秒内髋部与肩部快速下降，随后躯干转为水平；二维动作证据待核查。";}
                // A lateral fall may rotate around the hips without a large hip drop.
                // Require the earlier support-foot observation to distinguish a bed-level
                // lie-down; this is an image-space cue, not a calibrated floor plane.
                var collapse=history.Where(s=>time-s.Time is >=.2 and <=2.5&&s.Activity is "standing" or "walk" or "sitting"&&s.Confidence>=.65f&&s.Torso>=.025f&&s.FootY.HasValue)
                    .FirstOrDefault(s=>supportReference?.FootY is float support&&(sy-s.ShoulderY)/s.Torso>.4f&&tilt-s.Tilt>35&&box.Y2>=support-supportReference.Torso*.35f);
                if(collapse is not null)recentCollapseUntil=time+6;
            }
            if(raw.Activity is "standing" or "walk"&&raw.Confidence>=.65f)recentCollapseUntil=-1;
            if(raw.Activity=="standing"&&history.Count>=4)
            {
                var start=history.FirstOrDefault(s=>time-s.Time is >=.5 and <=1.2&&s.Activity=="standing"&&s.Torso>=.025f);
                if(start is not null&&new[]{15,16}.All(i=>joints[i].Confidence>=.4f&&start.Joints[i].Confidence>=.4f))
                {
                    float speed=Math.Abs(hx-start.HipX)/start.Torso/(float)(time-start.Time);
                    float footChange=Math.Abs((joints[15].X-joints[16].X)-(start.Joints[15].X-start.Joints[16].X))*aspect/start.Torso;
                    if(speed>.25f&&footChange>.18f)raw=("walk",raw.Confidence);
                }
            }
        }
        if(raw.Activity!="unknown")
        {
            lastValid=time;
            if(candidate!=raw.Activity){candidate=raw.Activity;candidateSince=time;}
            if(time-candidateSince>=(candidate=="lying"?.25:.35))
            {
                if(stable!=candidate)
                {
                    transition=stable is "standing" or "walk"&&candidate=="sitting"?"sit_down":stable=="sitting"&&candidate=="standing"?"stand_up":"";
                    transitionUntil=time+.65;stable=candidate;
                }
                stableConfidence=raw.Confidence;
            }
        }
        else {candidate="unknown";candidateSince=time;if(time-lastValid>.65){stable="unknown";stableConfidence=0;transition="";}}
        Activity=time<transitionUntil&&transition.Length>0?transition:stable;
        Confidence=stable=="unknown"?0:stableConfidence;
        FallMotion=time<=fallUntil;
        ShortWindowSupported=time<=recentCollapseUntil;
        if(!FallMotion)MotionEvidence=ShortWindowSupported?"躯干转为水平、上身下降，人物下沿接近此前足部位置；二维位置线索用于视频复核。":"";
    }
}
