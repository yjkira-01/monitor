using CareMonitor.Models;

namespace CareMonitor.Services;
public static class PostureObservation
{
    // Geometric observations remain explicitly distinct from trained action classes.
    public static (string Activity,float Confidence) Classify(Joint[] joints,Box box,Joint[]? previous=null,float aspect=1)
    {
        // Work in isotropic image units; normalized x and y have different pixel scales.
        if(joints.Length!=17||!float.IsFinite(aspect)||aspect<=0)return("unknown",0);
        bool Visible(int i)=>float.IsFinite(joints[i].X)&&float.IsFinite(joints[i].Y)&&joints[i].Confidence>=.4f;
        (float X,float Y) Point(int i)=>(joints[i].X*aspect,joints[i].Y);
        var sides=new[]{(Shoulder:5,Hip:11),(Shoulder:6,Hip:12)}.Where(s=>Visible(s.Shoulder)&&Visible(s.Hip)).ToArray();
        var shoulders=sides.Select(s=>s.Shoulder).ToArray();var hips=sides.Select(s=>s.Hip).ToArray();
        if(shoulders.Length==0||hips.Length==0)return("unknown",0);
        float sx=shoulders.Average(i=>Point(i).X),sy=shoulders.Average(i=>Point(i).Y);
        float hx=hips.Average(i=>Point(i).X),hy=hips.Average(i=>Point(i).Y);
        float dx=hx-sx,dy=hy-sy,torso=MathF.Sqrt(dx*dx+dy*dy);
        if(torso<.025f)return("unknown",0);
        // One reliable visible side can support the torso; an occluded far shoulder
        // should not erase a complete near-side observation.
        float confidence=sides.Max(s=>Math.Min(joints[s.Shoulder].Confidence,joints[s.Hip].Confidence));
        bool horizontal=Math.Abs(dx)>Math.Abs(dy)*1.2f;
        var legs=new List<(float Knee,float ThighVertical,float AnkleBelow,float Confidence)>();
        foreach(var (hip,knee,ankle) in new[]{(11,13,15),(12,14,16)})
        {
            if(!Visible(hip)||!Visible(knee)||!Visible(ankle))continue;
            var h=Point(hip);var k=Point(knee);var a=Point(ankle);
            float ux=h.X-k.X,uy=h.Y-k.Y,vx=a.X-k.X,vy=a.Y-k.Y;
            float thigh=MathF.Sqrt(ux*ux+uy*uy),shin=MathF.Sqrt(vx*vx+vy*vy);
            if(thigh<torso*.2f||shin<torso*.2f)continue;
            float angle=MathF.Acos(Math.Clamp((ux*vx+uy*vy)/(thigh*shin),-1,1))*180/MathF.PI;
            legs.Add((angle,(k.Y-h.Y)/thigh,(a.Y-h.Y)/torso,new[]{hip,knee,ankle}.Min(i=>joints[i].Confidence)));
        }
        // A horizontal torso alone also occurs when bending to pick something up.
        // Extended knees and downward thighs tolerate perspective shortening of the legs.
        bool uprightLegs=legs.Any(l=>l.Knee>150&&l.ThighVertical>.6f&&l.AnkleBelow>.65f);
        if(horizontal&&!uprightLegs&&(box.Width*aspect>box.Height*.85f||legs.Any(l=>l.AnkleBelow<.8f)))return("lying",confidence);
        if(legs.Count==0)return("unknown",0);
        confidence=Math.Min(confidence,legs.Max(l=>l.Confidence));
        if(uprightLegs)return(horizontal||dy<torso*.65f?("bending",confidence):("standing",confidence));
        if(dy>torso*.55f&&legs.Any(l=>l.Knee<145&&l.ThighVertical<.55f&&l.AnkleBelow>.15f))return("sitting",confidence);
        if(dy>torso*.3f&&legs.All(l=>l.Knee<135&&l.AnkleBelow<1.2f)&&legs.Any(l=>l.ThighVertical>.45f))return("crouching",confidence);
        return("unknown",0);
    }
}
