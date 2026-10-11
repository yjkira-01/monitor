using CareMonitor.Models;

namespace CareMonitor.Services;

/// <summary>Session identities with bounded, one-to-one recovery. Not biometric identification.</summary>
public sealed class StablePeopleTracker
{
    public sealed record Observation(Box Box,float Score,Joint[] Joints,float[] Appearance);
    private sealed record Entry(int Id,Observation Last,double Time);
    private readonly Dictionary<int,Entry> entries=[];
    private int nextId=1;
    public (int Id,Observation Observation)[] Update(Observation[] observations,double time)
    {
        foreach(int id in entries.Keys.Where(id=>time-entries[id].Time>2.5).ToArray())entries.Remove(id);
        var input=observations.Where(o=>o.Score>=.25f&&o.Box.Width>.01&&o.Box.Height>.01).OrderByDescending(o=>o.Score).Take(6).ToArray();
        var prior=entries.Values.OrderByDescending(e=>e.Time).Take(12).ToArray();
        var costs=new double[input.Length,prior.Length];
        for(int i=0;i<input.Length;i++)for(int j=0;j<prior.Length;j++)costs[i,j]=Cost(input[i],prior[j].Last,time-prior[j].Time);
        // Reject uncertain recovery after a gap; simultaneous detections always use separate IDs.
        for(int i=0;i<input.Length;i++)
        {
            var ranked=Enumerable.Range(0,prior.Length).OrderBy(j=>costs[i,j]).ToArray();
            if(ranked.Length>1&&costs[i,ranked[1]]-costs[i,ranked[0]]<.12)
                foreach(int j in ranked.Where(j=>time-prior[j].Time>.45))costs[i,j]=99;
        }
        int[] chosen=Enumerable.Repeat(-1,input.Length).ToArray(),current=(int[])chosen.Clone();double best=double.MaxValue;
        var visited=new Dictionary<(int,int),double>();
        void Search(int i,int used,double score)
        {
            if(score>=best)return;
            if(visited.TryGetValue((i,used),out double previous)&&previous<=score)return;visited[(i,used)]=score;
            if(i==input.Length){best=score;chosen=(int[])current.Clone();return;}
            current[i]=-1;Search(i+1,used,score+1.1);
            for(int j=0;j<prior.Length;j++)if((used&(1<<j))==0&&costs[i,j]<1.05)
            {current[i]=j;Search(i+1,used|(1<<j),score+costs[i,j]);}
        }
        Search(0,0,0);var result=new List<(int,Observation)>();
        for(int i=0;i<input.Length;i++)
        {
            int match=chosen[i];if(match<0&&input[i].Score<.35f)continue;
            int id=match>=0?prior[match].Id:nextId++;entries[id]=new(id,input[i],time);result.Add((id,input[i]));
        }
        return result.ToArray();
    }
    private static double Cost(Observation a,Observation b,double gap)
    {
        if(gap<0||gap>2.5)return 99;
        double scale=Math.Max(.08,Math.Max(Math.Sqrt(a.Box.Width*a.Box.Width+a.Box.Height*a.Box.Height),Math.Sqrt(b.Box.Width*b.Box.Width+b.Box.Height*b.Box.Height)));
        double dx=(a.Box.X1+a.Box.X2-b.Box.X1-b.Box.X2)/2,dy=(a.Box.Y1+a.Box.Y2-b.Box.Y1-b.Box.Y2)/2;
        double center=Math.Sqrt(dx*dx+dy*dy)/scale,iou=Box.Iou(a.Box,b.Box);
        double appearance=0;bool colour=a.Appearance.Length>0&&a.Appearance.Length==b.Appearance.Length;
        if(colour)
        {
            double dot=a.Appearance.Zip(b.Appearance,(x,y)=>(double)x*y).Sum();
            appearance=1-dot/Math.Max(1e-9,Math.Sqrt(a.Appearance.Sum(x=>(double)x*x)*b.Appearance.Sum(x=>(double)x*x)));
        }
        if(center>(gap>.6?.95:.8)||colour&&appearance>.55)return 99;
        var common=Enumerable.Range(0,Math.Min(a.Joints.Length,b.Joints.Length)).Where(j=>a.Joints[j].Confidence>=.45&&b.Joints[j].Confidence>=.45).ToArray();
        double joints=common.Length>=4?common.Select(j=>Math.Sqrt(Math.Pow(a.Joints[j].X-b.Joints[j].X,2)+Math.Pow(a.Joints[j].Y-b.Joints[j].Y,2))/scale).Order().ElementAt(common.Length/2):center;
        if(iou<.05&&center>.5&&(!colour||appearance>.25)&&joints>.5)return 99;
        return .33*(1-iou)+.34*center+.15*Math.Min(joints,1)+.45*appearance+.04*gap;
    }
}
