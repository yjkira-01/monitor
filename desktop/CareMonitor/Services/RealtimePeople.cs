using System.IO;
using System.Security.Cryptography;
using System.Text.Json;
using CareMonitor.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace CareMonitor.Services;

public sealed record PeopleObservation(Box Box,float Score,Joint[] Joints);

/// <summary>Official RTMO body7 ONNX deployment: BGR 0..255, top-left resize/pad.</summary>
public sealed class RealtimePeople : IDisposable
{
    private readonly InferenceSession session;
    internal InferenceSession Session=>session;
    public RealtimePeople(string directory,Func<string,InferenceSession> create)
    {
        string path=Path.Combine(directory,"rtmo.onnx");
        using var metadata=JsonDocument.Parse(File.ReadAllText(Path.Combine(directory,"rtmo.json")));
        using var stream=File.OpenRead(path);
        if(Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant()!=metadata.RootElement.GetProperty("sha256").GetString())
            throw new InvalidDataException("RTMO 模型完整性检查失败。");
        session=create(path);
    }
    public PeopleObservation[] Predict(Mat frame,float threshold=.15f)
    {
        float ratio=Math.Min(640f/frame.Width,640f/frame.Height);
        using var enhanced=NativeModels.Enhance(frame);using var resized=new Mat();
        Cv2.Resize(enhanced,resized,new((int)(frame.Width*ratio),(int)(frame.Height*ratio)),0,0,InterpolationFlags.Linear);
        using var padded=new Mat(640,640,MatType.CV_8UC3,Scalar.All(114));
        using(var region=new Mat(padded,new Rect(0,0,resized.Width,resized.Height)))resized.CopyTo(region);
        byte[] bytes=NativeModels.Bytes(padded);float[] input=new float[3*640*640];
        for(int i=0;i<640*640;i++)for(int c=0;c<3;c++)input[c*640*640+i]=bytes[3*i+c];
        using var outputs=session.Run([NamedOnnxValue.CreateFromTensor("input",new DenseTensor<float>(input,[1,3,640,640]))]);
        var dets=outputs.Single(v=>v.Name=="dets").AsTensor<float>();
        var pose=outputs.Single(v=>v.Name=="keypoints").AsTensor<float>();
        if(dets.Rank!=3||dets.Dimensions[2]!=5||pose.Rank!=4||pose.Dimensions[1]!=dets.Dimensions[1]||pose.Dimensions[2]!=17||pose.Dimensions[3]!=3)
            throw new InvalidDataException("RTMO 输出结构不兼容。");
        var people=new List<PeopleObservation>();
        foreach(int i in Enumerable.Range(0,dets.Dimensions[1]).OrderByDescending(i=>dets[0,i,4]))
        {
            float score=dets[0,i,4];if(!NativeModels.ValidUnit(score))throw new InvalidDataException("RTMO 检测输出异常。");
            if(score<threshold)continue;
            var b=new Box(dets[0,i,0]/ratio/frame.Width,dets[0,i,1]/ratio/frame.Height,dets[0,i,2]/ratio/frame.Width,dets[0,i,3]/ratio/frame.Height);
            if(new[]{b.X1,b.Y1,b.X2,b.Y2}.Any(v=>!float.IsFinite(v)))continue;
            b=new(Math.Clamp(b.X1,0,1),Math.Clamp(b.Y1,0,1),Math.Clamp(b.X2,0,1),Math.Clamp(b.Y2,0,1));
            if(b.Width<.02f||b.Height<.025f)continue;
            var joints=new Joint[17];
            for(int j=0;j<17;j++)
            {
                float x=pose[0,i,j,0]/ratio/frame.Width,y=pose[0,i,j,1]/ratio/frame.Height,confidence=pose[0,i,j,2];
                if(!float.IsFinite(x)||!float.IsFinite(y)||!NativeModels.ValidUnit(confidence))throw new InvalidDataException("RTMO 关键点输出异常。");
                joints[j]=new(x,y,0,x<0||x>1||y<0||y>1?0:Math.Clamp(confidence,0,1));
            }
            // Some exported queries have a very loose body box but the same precise pose.
            // Derive a stable extent from sufficiently observed body points before tracking.
            var visible=joints.Where(j=>j.Confidence>=.5f).ToArray();
            if(visible.Length>=8&&new[]{5,6,11,12}.Count(j=>joints[j].Confidence>=.5f)>=3&&new[]{13,14,15,16}.Any(j=>joints[j].Confidence>=.5f))
            {
                float x1=visible.Min(j=>j.X),x2=visible.Max(j=>j.X),y1=visible.Min(j=>j.Y),y2=visible.Max(j=>j.Y);
                float marginX=Math.Max(.015f,(y2-y1)*frame.Height/frame.Width*.08f),marginY=Math.Max(.015f,(y2-y1)*.05f);
                b=new(Math.Clamp(x1-marginX,0,1),Math.Clamp(y1-marginY,0,1),Math.Clamp(x2+marginX,0,1),Math.Clamp(y2+marginY,0,1));
            }
            var candidate=new PeopleObservation(b,Math.Clamp(score,0,1),joints);
            if(people.Any(p=>SamePose(p,candidate,frame.Width/(float)frame.Height)))continue;
            people.Add(candidate);if(people.Count==12)break;
        }
        return people.ToArray();
    }
    public static bool SamePose(PeopleObservation a,PeopleObservation b,float aspect)
    {
        float overlap=Box.Iou(a.Box,b.Box);if(overlap<=0)return false;
        var common=Enumerable.Range(0,17).Where(j=>a.Joints[j].Confidence>=.4f&&b.Joints[j].Confidence>=.4f).ToArray();
        if(common.Length<6)return overlap>.65f;
        float scale=MathF.Sqrt(Math.Min(a.Box.Width*a.Box.Height,b.Box.Width*b.Box.Height)*aspect);
        if(scale<=0)return false;
        var distances=common.Select(j=>MathF.Sqrt(MathF.Pow((a.Joints[j].X-b.Joints[j].X)*aspect,2)+MathF.Pow(a.Joints[j].Y-b.Joints[j].Y,2))/scale).ToArray();
        return distances.Average()<.06f&&distances.Count(d=>d<.08f)>=common.Length*.7f;
    }
    public void Dispose()=>session.Dispose();
}
