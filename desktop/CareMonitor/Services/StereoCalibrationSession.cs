using System.IO;
using CareMonitor.Models;
using OpenCvSharp;
namespace CareMonitor.Services;
public sealed class StereoCalibrationSession
{
    private readonly List<Point2f[]> left=[],right=[];
    private readonly List<(float X,float Y,float Area)> views=[];
    private readonly Size board;
    private readonly double square;
    private Size image;
    public int Count=>left.Count;
    public Joint[] LeftCorners {get;private set;}=[];
    public Joint[] RightCorners {get;private set;}=[];
    public string Coverage=>views.Count==0?"尚无样本":$"水平覆盖 {views.Max(v=>v.X)-views.Min(v=>v.X):P0} / 至少20%；垂直覆盖 {views.Max(v=>v.Y)-views.Min(v=>v.Y):P0} / 至少15%";
    public StereoCalibrationSession(int columns=9,int rows=6,double squareMillimeters=25)
    {if(columns is <4 or >16||rows is <4 or >12||!double.IsFinite(squareMillimeters)||squareMillimeters is <5 or >100)throw new ArgumentException("棋盘内角点须为 4–16 列、4–12 行，格长须为 5–100 毫米。");board=new(columns,rows);square=squareMillimeters;}
    public string Add(VideoSnapshot l,VideoSnapshot r)
    {
        LeftCorners=[];RightCorners=[];
        if(l.Width!=r.Width||l.Height!=r.Height||l.Replay||r.Replay||l.Ended||r.Ended)throw new InvalidOperationException("标定需两台摄像头在线，并使用相同分辨率。");
        if(Math.Abs(l.CaptureTimestamp-r.CaptureTimestamp)*1000d/System.Diagnostics.Stopwatch.Frequency>50)throw new InvalidOperationException("两路时间差超过 50 毫秒，请保持棋盘静止后重新采集。");
        if(left.Count>0&&(image.Width!=l.Width||image.Height!=l.Height))throw new InvalidOperationException("分辨率已改变，请清空标定样本。");image=new(l.Width,l.Height);
        Point2f[] Find(VideoSnapshot snap)
        {
            using var src=Mat.FromPixelData(snap.Height,snap.Width,MatType.CV_8UC3,snap.Pixels);using var gray=new Mat();Cv2.CvtColor(src,gray,ColorConversionCodes.BGR2GRAY);
            if(!Cv2.FindChessboardCorners(gray,board,out Point2f[] corners,ChessboardFlags.AdaptiveThresh|ChessboardFlags.NormalizeImage))throw new InvalidOperationException($"{(ReferenceEquals(snap,l)?"左":"右")}画面未检测到完整 {board.Width} × {board.Height} 内角点，请移动棋盘。");
            Cv2.CornerSubPix(gray,corners,new Size(5,5),new Size(-1,-1),new TermCriteria(CriteriaTypes.Count|CriteriaTypes.Eps,30,.01));return corners;
        }
        var lc=Find(l);LeftCorners=lc.Select(p=>new Joint(p.X/l.Width,p.Y/l.Height,0,1)).ToArray();
        var rc=Find(r);RightCorners=rc.Select(p=>new Joint(p.X/r.Width,p.Y/r.Height,0,1)).ToArray();
        float x=lc.Average(p=>p.X)/image.Width,y=lc.Average(p=>p.Y)/image.Height,area=(lc.Max(p=>p.X)-lc.Min(p=>p.X))*(lc.Max(p=>p.Y)-lc.Min(p=>p.Y))/(image.Width*(float)image.Height);
        if(area<.025)throw new InvalidOperationException("棋盘过小，请移近一些。");
        if(views.Any(v=>Math.Abs(v.X-x)<.06&&Math.Abs(v.Y-y)<.06&&Math.Abs(v.Area-area)<.025))throw new InvalidOperationException("当前棋盘位置与已有样本重复，请改变位置、距离和倾斜角度。");
        if(Count>=50)throw new InvalidOperationException("样本已达 50 组，请计算标定。");left.Add(lc);right.Add(rc);views.Add((x,y,area));return $"已采集 {Count} 组角点，图像不保存";
    }
    public StereoCalibration Solve(string leftId,string rightId)
    {
        if(Count<12||views.Max(v=>v.X)-views.Min(v=>v.X)<.2||views.Max(v=>v.Y)-views.Min(v=>v.Y)<.15)throw new InvalidOperationException("至少采集 12 组，建议 20 组；棋盘位置需要覆盖画面的左、右、上、下，并改变距离和倾斜角度。");
        var obj=Enumerable.Range(0,board.Height).SelectMany(y=>Enumerable.Range(0,board.Width).Select(x=>new Point3f((float)(x*square/1000),(float)(y*square/1000),0))).ToArray();
        var objects=Enumerable.Repeat(obj,Count).ToArray();double[,] kl=new double[3,3],kr=new double[3,3];double[] dl=new double[5],dr=new double[5];
        double le=Cv2.CalibrateCamera(objects,left,image,kl,dl,out Vec3d[] _,out Vec3d[] _,CalibrationFlags.FixK3);
        double re=Cv2.CalibrateCamera(objects,right,image,kr,dr,out Vec3d[] _,out Vec3d[] _,CalibrationFlags.FixK3);
        using var rotation=new Mat();using var translation=new Mat();using var essential=new Mat();using var fundamental=new Mat();
        double error=Cv2.StereoCalibrate(objects,left,right,kl,dl,kr,dr,image,rotation,translation,essential,fundamental,CalibrationFlags.FixIntrinsic,new TermCriteria(CriteriaTypes.Count|CriteriaTypes.Eps,100,1e-6));
        if(le>1.5||re>1.5||error>1.5)throw new InvalidDataException($"重投影误差偏大（左 {le:F2} / 右 {re:F2} / 双目 {error:F2} px），请清空并补充清晰、多角度的样本。");
        var result=new StereoCalibration{Width=image.Width,Height=image.Height,LeftDeviceId=leftId,RightDeviceId=rightId,LeftK=kl.Cast<double>().ToArray(),RightK=kr.Cast<double>().ToArray(),LeftD=dl,RightD=dr,
            Rotation=Enumerable.Range(0,9).Select(i=>rotation.At<double>(i/3,i%3)).ToArray(),Translation=Enumerable.Range(0,3).Select(i=>translation.At<double>(i)).ToArray(),Samples=Count,SquareMillimeters=square,ReprojectionError=error};
        result.Validate();return result;
    }
    internal static StereoCalibration SyntheticCheck()
    {
        // Exercise corner detection, sample collection and the full calibration solver on
        // generated chessboards. These are synthetic geometry checks, not device accuracy.
        var session=new StereoCalibrationSession();long timestamp=System.Diagnostics.Stopwatch.GetTimestamp();int number=0;
        foreach(double depth in new[]{.72,.9,1.08})foreach(double x in new[]{-.1,.1,.3})foreach(double y in new[]{-.22,-.04,.14})
        {
            double rx=.18+number*.007,ry=-.24+number*.015;number++;
            VideoSnapshot Render(bool rightCamera)
            {
                using var mat=new Mat(480,640,MatType.CV_8UC3,Scalar.All(255));
                Point Project(double px,double py)
                {
                    double xx=Math.Cos(ry)*px+Math.Sin(ry)*Math.Sin(rx)*py+x-(rightCamera?.2:0);
                    double yy=Math.Cos(rx)*py+y,zz=-Math.Sin(ry)*px+Math.Cos(ry)*Math.Sin(rx)*py+depth;
                    return new((int)Math.Round(600*xx/zz+320),(int)Math.Round(600*yy/zz+240));
                }
                for(int row=-1;row<6;row++)for(int col=-1;col<9;col++)
                    if((row+col)%2==0)Cv2.FillConvexPoly(mat,new[]{Project(col*.025,row*.025),Project((col+1)*.025,row*.025),Project((col+1)*.025,(row+1)*.025),Project(col*.025,(row+1)*.025)},Scalar.All(0));
                return new(NativeModels.Bytes(mat),640,480,number,0,30,false,false,false,"synthetic",1,number,timestamp);
            }
            try{session.Add(Render(false),Render(true));}catch(InvalidOperationException){}
        }
        var result=session.Solve("synthetic-left","synthetic-right");
        if(Math.Abs(result.Baseline-.2)>.02)throw new InvalidDataException("Synthetic stereo calibration baseline check failed.");return result;
    }
}
