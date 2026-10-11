using System.Diagnostics;
using CareMonitor.Models;
namespace CareMonitor.Services;
public sealed class StereoPairing
{
    private readonly List<TimedObservation> left=[],right=[];
    private long leftGeneration=-1,rightGeneration=-1;
    private (long,long) lastPair=(-1,-1);
    private long matchedTick;
    private StereoGeometry? geometry;
    public StereoSnapshot? Latest { get; private set; }
    public void Reset(StereoCalibration? calibration=null)
    {left.Clear();right.Clear();lastPair=(-1,-1);Latest=null;geometry=calibration is null?null:new(calibration);}
    public StereoSnapshot? Update(TimedObservation? l,TimedObservation? r)
    {
        if(l is null||r is null||l.Frame.Ended||r.Frame.Ended||l.Frame.Replay||r.Frame.Replay){Latest=null;return null;}
        if(geometry is not null&&(!geometry.MatchesResolution(l.Frame.Width,l.Frame.Height)||!geometry.MatchesResolution(r.Frame.Width,r.Frame.Height)))
        {Latest=null;return null;}
        if(Stopwatch.GetElapsedTime(l.Frame.CaptureTimestamp).TotalSeconds>.65||Stopwatch.GetElapsedTime(r.Frame.CaptureTimestamp).TotalSeconds>.65){Latest=null;return null;}
        if(l.Frame.Generation!=leftGeneration||r.Frame.Generation!=rightGeneration)
        {left.Clear();right.Clear();lastPair=(-1,-1);Latest=null;leftGeneration=l.Frame.Generation;rightGeneration=r.Frame.Generation;}
        void Add(List<TimedObservation> buffer,TimedObservation sample)
        {
            if(!buffer.Any(s=>s.Frame.Sequence==sample.Frame.Sequence))buffer.Add(sample);
            buffer.RemoveAll(s=>Stopwatch.GetElapsedTime(s.Frame.CaptureTimestamp).TotalSeconds>.65);while(buffer.Count>8)buffer.RemoveAt(0);
        }
        Add(left,l);Add(right,r);
        var pairs=left.SelectMany(a=>right.Select(b=>new{Left=a,Right=b,Delta=Math.Abs(a.Frame.CaptureTimestamp-b.Frame.CaptureTimestamp)*1000d/Stopwatch.Frequency}))
            .Where(p=>p.Delta<=80&&p.Left.Frame.Sequence>lastPair.Item1&&p.Right.Frame.Sequence>lastPair.Item2).OrderByDescending(p=>Math.Min(p.Left.Frame.CaptureTimestamp,p.Right.Frame.CaptureTimestamp)).ToArray();
        var pair=pairs.FirstOrDefault();if(pair is null){if(Latest is not null&&Stopwatch.GetElapsedTime(matchedTick).TotalSeconds>.65)Latest=null;return Latest;}
        lastPair=(pair.Left.Frame.Sequence,pair.Right.Frame.Sequence);
        matchedTick=Math.Min(pair.Left.Frame.CaptureTimestamp,pair.Right.Frame.CaptureTimestamp);
        var people=geometry?.Reconstruct(pair.Left.Analysis.People,pair.Right.Analysis.People,Math.Max(pair.Left.Frame.CaptureTimestamp,pair.Right.Frame.CaptureTimestamp))??[];
        Latest=new(lastPair.Item1,lastPair.Item2,pair.Delta,people,geometry is null?"两路监测已连接 · 完成标定后提供三维定位":"双目三维定位 · 左相机坐标系，单位：米");return Latest;
    }
}
