using System.IO;
using System.Numerics;
using System.Text.Json;
using CareMonitor.Models;
using OpenCvSharp;
namespace CareMonitor.Services;

public sealed class StereoCalibration
{
    public int Version { get; set; }=1;
    public string LeftDeviceId { get; set; }="";
    public string RightDeviceId { get; set; }="";
    public int Width { get; set; }
    public int Height { get; set; }
    public double[] LeftK { get; set; }=[];
    public double[] RightK { get; set; }=[];
    public double[] LeftD { get; set; }=[];
    public double[] RightD { get; set; }=[];
    public double[] Rotation { get; set; }=[];
    public double[] Translation { get; set; }=[];
    public double ReprojectionError { get; set; }
    public int Samples { get; set; }
    public double SquareMillimeters { get; set; }
    public DateTimeOffset Created { get; set; }=DateTimeOffset.Now;
    public double Baseline=>Math.Sqrt(Translation.Sum(v=>v*v));
    public void Validate()
    {
        if(Version!=1||LeftDeviceId.Length==0||RightDeviceId.Length==0||LeftDeviceId==RightDeviceId||Width<160||Height<120||LeftK.Length!=9||RightK.Length!=9||LeftD.Length!=5||RightD.Length!=5||Rotation.Length!=9||Translation.Length!=3||
            LeftK.Concat(RightK).Concat(LeftD).Concat(RightD).Concat(Rotation).Concat(Translation).Any(v=>!double.IsFinite(v))||
            LeftK[0]<=0||LeftK[4]<=0||RightK[0]<=0||RightK[4]<=0||Baseline is <.02 or >2||!double.IsFinite(ReprojectionError)||ReprojectionError is <0 or >1.5||Samples<12)
            throw new InvalidDataException("双目标定参数无效，需重新采集标定。");
        for(int i=0;i<3;i++)for(int j=0;j<3;j++)
            if(Math.Abs(Enumerable.Range(0,3).Sum(k=>Rotation[i*3+k]*Rotation[j*3+k])-(i==j?1:0))>1e-3)throw new InvalidDataException("双目旋转矩阵无效。");
        double determinant=Rotation[0]*(Rotation[4]*Rotation[8]-Rotation[5]*Rotation[7])-Rotation[1]*(Rotation[3]*Rotation[8]-Rotation[5]*Rotation[6])+Rotation[2]*(Rotation[3]*Rotation[7]-Rotation[4]*Rotation[6]);
        if(Math.Abs(determinant-1)>1e-3)throw new InvalidDataException("双目旋转矩阵方向无效。");
    }
    public static StereoCalibration Load(string path)
    {
        if(new FileInfo(path).Length>1024*1024)throw new InvalidDataException("标定文件过大，请选择导出的双目标定JSON文件。");
        string text=File.ReadAllText(path);using var document=JsonDocument.Parse(text);
        if(document.RootElement.ValueKind!=JsonValueKind.Object||!document.RootElement.TryGetProperty("LeftK",out _))throw new InvalidDataException("这不是双目标定文件。请使用“导出标定”保存的JSON；“导出定位”文件不能导入标定。");
        var data=JsonSerializer.Deserialize<StereoCalibration>(text)??throw new InvalidDataException("标定文件为空。");data.Validate();return data;
    }
    public void Save(string path)
    {Validate();string temporary=path+".tmp";File.WriteAllText(temporary,JsonSerializer.Serialize(this,new JsonSerializerOptions{WriteIndented=true}));File.Move(temporary,path,true);}
}
public sealed record StereoPerson(int Id,int RightId,Vector3?[] Joints,Vector3 Position,int ObservedJoints,double ErrorPixels,string Activity);
public sealed record StereoSnapshot(long LeftSequence,long RightSequence,double PairDeltaMs,StereoPerson[] People,string Status);

public sealed class StereoGeometry
{
    private readonly StereoCalibration calibration;
    private readonly Dictionary<int,(int Right,Vector3?[] Joints,long Tick)> history=[];
    public StereoGeometry(StereoCalibration calibration){calibration.Validate();this.calibration=calibration;}
    public bool MatchesResolution(int width,int height)=>width==calibration.Width&&height==calibration.Height;
    private Point2f[] Undistort(Joint[] joints,double[] k,double[] d)
    {
        var points=joints.Select(j=>new Point2f(j.X*calibration.Width,j.Y*calibration.Height)).ToArray();
        using var src=Mat.FromPixelData(points.Length,1,MatType.CV_32FC2,points);
        using var matrix=Mat.FromPixelData(3,3,MatType.CV_64FC1,k);using var distortion=Mat.FromPixelData(1,5,MatType.CV_64FC1,d);using var dst=new Mat();
        Cv2.UndistortPoints(src,dst,matrix,distortion);
        return Enumerable.Range(0,points.Length).Select(i=>{var p=dst.At<Vec2f>(i);return new Point2f(p.Item0,p.Item1);}).ToArray();
    }
    public StereoPerson[] Reconstruct(PersonState[] left,PersonState[] right,long tick)
    {
        left=left.Where(p=>p.ImageJoints.Length==17).Take(6).ToArray();right=right.Where(p=>p.ImageJoints.Length==17).Take(6).ToArray();
        var ln=left.Select(p=>Undistort(p.ImageJoints,calibration.LeftK,calibration.LeftD)).ToArray();var rn=right.Select(p=>Undistort(p.ImageJoints,calibration.RightK,calibration.RightD)).ToArray();
        double[,] first=new double[3,4]{{1,0,0,0},{0,1,0,0},{0,0,1,0}},second=new double[3,4];
        for(int i=0;i<3;i++){for(int j=0;j<3;j++)second[i,j]=calibration.Rotation[i*3+j];second[i,3]=calibration.Translation[i];}
        using var firstMatrix=Mat.FromPixelData(3,4,MatType.CV_64FC1,first.Cast<double>().ToArray());
        using var secondMatrix=Mat.FromPixelData(3,4,MatType.CV_64FC1,second.Cast<double>().ToArray());
        double focal=(calibration.LeftK[0]+calibration.LeftK[4]+calibration.RightK[0]+calibration.RightK[4])/4;
        var candidates=new Dictionary<(int,int),(Vector3?[] Joints,double Cost,int Count,Vector3 Center)>();
        for(int a=0;a<left.Length;a++)for(int b=0;b<right.Length;b++)
        {
            int[] common=Enumerable.Range(0,17).Where(j=>left[a].ImageJoints[j].Confidence>=.45&&right[b].ImageJoints[j].Confidence>=.45&&InFrame(left[a].ImageJoints[j])&&InFrame(right[b].ImageJoints[j])).ToArray();
            if(common.Length<6||common.Count(j=>j is 5 or 6 or 11 or 12)<2)continue;
            double[] lp=common.Select(j=>(double)ln[a][j].X).Concat(common.Select(j=>(double)ln[a][j].Y)).ToArray();
            double[] rp=common.Select(j=>(double)rn[b][j].X).Concat(common.Select(j=>(double)rn[b][j].Y)).ToArray();
            using var leftPoints=Mat.FromPixelData(2,common.Length,MatType.CV_64FC1,lp);using var rightPoints=Mat.FromPixelData(2,common.Length,MatType.CV_64FC1,rp);using var homogeneous=new Mat();
            Cv2.TriangulatePoints(firstMatrix,secondMatrix,leftPoints,rightPoints,homogeneous);
            var joints=new Vector3?[17];var errors=new List<double>();
            for(int i=0;i<common.Length;i++)
            {
                double w=homogeneous.At<double>(3,i);if(!double.IsFinite(w)||Math.Abs(w)<1e-9)continue;
                var p=new Vector3((float)(homogeneous.At<double>(0,i)/w),(float)(homogeneous.At<double>(1,i)/w),(float)(homogeneous.At<double>(2,i)/w));
                var q=Transform(p);if(!Finite(p)||p.Z is <.3f or >12||q.Z<=.1)continue;
                int j=common[i];double e1=Math.Sqrt(Math.Pow(p.X/p.Z-ln[a][j].X,2)+Math.Pow(p.Y/p.Z-ln[a][j].Y,2))*focal;
                double e2=Math.Sqrt(Math.Pow(q.X/q.Z-rn[b][j].X,2)+Math.Pow(q.Y/q.Z-rn[b][j].Y,2))*focal;
                if(Math.Max(e1,e2)>5)continue;joints[j]=p;errors.Add((e1+e2)/2);
            }
            if(errors.Count<6)continue;
            var torso=new[]{5,6,11,12}.Where(j=>joints[j].HasValue).Select(j=>joints[j]!.Value).ToArray();if(torso.Length<2)continue;
            // Bone-length gates reject plausible epipolar matches between unrelated people.
            if(Bones.Any(edge=>joints[edge.A].HasValue&&joints[edge.B].HasValue&&Vector3.Distance(joints[edge.A]!.Value,joints[edge.B]!.Value)>1.05))continue;
            var center=torso.Aggregate(Vector3.Zero,(p,q)=>p+q)/torso.Length;
            double cost=errors.Average();if(cost>3.5)continue;
            candidates[(a,b)]=(joints,cost,errors.Count,center);
        }
        // Ambiguous associations produce no measurement instead of an invented identity.
        var competing=candidates.ToArray();
        foreach(var key in candidates.Keys.ToArray())
        {
            var value=candidates[key];double alternative=competing.Where(c=>c.Key!=key&&(c.Key.Item1==key.Item1||c.Key.Item2==key.Item2)).Select(c=>c.Value.Cost).DefaultIfEmpty(99).Min();
            bool same=history.TryGetValue(left[key.Item1].Id,out var prior)&&prior.Right==right[key.Item2].Id&&System.Diagnostics.Stopwatch.GetElapsedTime(prior.Tick,tick).TotalSeconds<.7;
            if(!same&&Math.Abs(alternative-value.Cost)<.5)candidates.Remove(key);
        }
        var chosen=new List<(int A,int B)>();double best=double.MaxValue;var current=new List<(int,int)>();
        void Search(int a,int used,double score)
        {
            if(score>=best)return;
            if(a==left.Length){best=score;chosen=current.ToList();return;}
            Search(a+1,used,score+4);
            for(int b=0;b<right.Length;b++)if((used&(1<<b))==0&&candidates.TryGetValue((a,b),out var c))
            {current.Add((a,b));Search(a+1,used|(1<<b),score+c.Cost);current.RemoveAt(current.Count-1);}
        }
        Search(0,0,0);var people=new List<StereoPerson>();
        foreach(var (a,b) in chosen)
        {
            var c=candidates[(a,b)];int id=left[a].Id;
            if(history.TryGetValue(id,out var previous)&&previous.Right==right[b].Id&&System.Diagnostics.Stopwatch.GetElapsedTime(previous.Tick,tick).TotalSeconds<.5)
                for(int j=0;j<17;j++)if(c.Joints[j] is Vector3 p&&previous.Joints[j] is Vector3 q&&Vector3.Distance(p,q)<.6)c.Joints[j]=Vector3.Lerp(q,p,.65f);
            history[id]=(right[b].Id,c.Joints,tick);
            people.Add(new(id,right[b].Id,c.Joints,c.Center,c.Count,c.Cost,left[a].Activity));
        }
        foreach(int id in history.Keys.Where(id=>System.Diagnostics.Stopwatch.GetElapsedTime(history[id].Tick,tick).TotalSeconds>1).ToArray())history.Remove(id);
        return people.ToArray();
    }
    private Vector3 Transform(Vector3 p)
    {var r=calibration.Rotation;var t=calibration.Translation;return new((float)(r[0]*p.X+r[1]*p.Y+r[2]*p.Z+t[0]),(float)(r[3]*p.X+r[4]*p.Y+r[5]*p.Z+t[1]),(float)(r[6]*p.X+r[7]*p.Y+r[8]*p.Z+t[2]));}
    private static bool InFrame(Joint j)=>j.X is >=0 and <=1&&j.Y is >=0 and <=1&&float.IsFinite(j.X)&&float.IsFinite(j.Y);
    private static bool Finite(Vector3 p)=>float.IsFinite(p.X)&&float.IsFinite(p.Y)&&float.IsFinite(p.Z);
    public static readonly (int A,int B)[] Bones=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)];
}
