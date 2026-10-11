using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using CareMonitor.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace CareMonitor.Services;
public sealed class NativeModels : IDisposable
{
    private readonly InferenceSession? detector,landmarks;
    private readonly InferenceSession pose,video;
    private readonly RealtimePose? realtimePose;
    private readonly RealtimePeople? realtimePeople;
    private readonly PretrainedActions? actions;
    public string[] Classes { get; }
    public bool LowLightTrained { get; }
    public bool PartialTrained { get; }
    public string DetectorName { get; }
    public string VideoName { get; }
    public int FrameWidth { get; }
    public int FrameHeight { get; }
    public int ClipLength { get; }
    public int EmbeddingSize { get; }
    public bool Preview { get; }
    public bool Realtime { get; }
    public bool OneStage { get; }
    public bool VideoGpuEnabled { get; }
    private readonly int detectorSize;
    private readonly bool preserveAspect;
    private readonly bool profileModels=Environment.GetCommandLineArgs().Contains("--profile-models");
    public string Device { get; private set; } = "CPU";
    public NativeModels(string assets,bool useGpu,bool preview=false,string fallModel="stage_c")
    {
        Preview=false;Realtime=preview;string profile=Realtime?Path.Combine(assets,"Realtime"):assets;
        OneStage=Realtime&&File.Exists(Path.Combine(profile,"rtmo.onnx"))&&!Environment.GetCommandLineArgs().Contains("--legacy-pose");
        using var meta=JsonDocument.Parse(File.ReadAllText(Path.Combine(profile,"models.json")));
        if(fallModel is not ("gmd" or "stage_c"))throw new InvalidDataException("跌倒模型选择无效。");
        string videoPath=fallModel=="gmd"?Path.Combine(assets,"Binary/gmd/video.onnx"):Path.Combine(assets,"video.onnx");
        using var binaryMeta=fallModel=="gmd"?JsonDocument.Parse(File.ReadAllText(Path.Combine(assets,"Binary/gmd/models.json"))):null;
        var videoInfo=binaryMeta?.RootElement.GetProperty("video")??meta.RootElement.GetProperty("video");
        Classes=videoInfo.GetProperty("classes").EnumerateArray().Select(e=>e.GetString()!).ToArray();
        LowLightTrained=videoInfo.GetProperty("low_light_trained").GetBoolean();
        PartialTrained=videoInfo.GetProperty("partial_body_trained").GetBoolean();
        if(Classes.Length<2||!Classes.Contains("fall"))throw new InvalidDataException("视频模型类别不兼容。");
        var shape=videoInfo.GetProperty("input").EnumerateArray().Select(e=>e.GetInt32()).ToArray();
        if(shape.Length!=5||shape[0]!=1||shape[4]!=3)throw new InvalidDataException("视频模型输入格式不兼容。");
        ClipLength=shape[1];FrameHeight=shape[2];FrameWidth=shape[3];EmbeddingSize=512;
        preserveAspect=false;
        detectorSize=meta.RootElement.GetProperty("people").GetProperty("input")[2].GetInt32();
        if(detectorSize!=(Realtime?576:320))throw new InvalidDataException("人员检测模型输入规格不兼容。");
        DetectorName=OneStage?"RTMO-m":Realtime?"RF-DETR Medium":"SSD";
        VideoName=fallModel=="gmd"?"R3D 跌倒二分类 · GMD":"R3D 跌倒二分类 · Stage C";
        foreach(var (key,file) in new[]{("video","video.onnx"),("people","people.onnx"),("pose_tcn","pose_tcn.onnx"),("landmarks","pose_landmarks_detector_full.onnx")})
        {
            if(OneStage&&key is "people" or "landmarks")continue;
            using var stream=File.OpenRead(key=="video"?videoPath:Path.Combine(key=="people"?profile:assets,file));
            var hash=Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
            if(hash!=(key=="video"?videoInfo:meta.RootElement.GetProperty(key)).GetProperty("sha256").GetString())throw new InvalidDataException("模型完整性检查失败："+file);
        }
        if(!OneStage){detector=Create(Path.Combine(profile,"people.onnx"),useGpu);landmarks=Create(Path.Combine(assets,"pose_landmarks_detector_full.onnx"),useGpu);}
        pose=Create(Path.Combine(assets,"pose_tcn.onnx"),false);
        video=Create(videoPath,useGpu);
        VideoGpuEnabled=Device=="DirectML / GPU";
        if(OneStage)realtimePeople=new(profile,p=>Create(p,useGpu));
        else if(Realtime)realtimePose=new(profile,p=>Create(p,useGpu));
        if(Realtime&&Environment.GetCommandLineArgs().Contains("--research-actions")&&File.Exists(Path.Combine(profile,"actions.onnx")))actions=new(profile);
    }
    private InferenceSession Create(string path,bool gpu)
    {
        if(gpu)
        {
            try
            {
                using var dml=new SessionOptions{EnableMemoryPattern=false,ExecutionMode=ExecutionMode.ORT_SEQUENTIAL};
                ConfigureProfile(dml,path,"dml");
                dml.AppendExecutionProvider_DML(0);var session=new InferenceSession(path,dml);Device="DirectML / GPU";return session;
            }
            catch(OnnxRuntimeException){Device="CPU（GPU不可用，已回退）";}
        }
        using var cpu=new SessionOptions{IntraOpNumThreads=3,InterOpNumThreads=1};ConfigureProfile(cpu,path,"cpu");return new(path,cpu);
    }
    private void ConfigureProfile(SessionOptions options,string path,string provider)
    {
        if(!profileModels)return;
        string directory=Path.Combine(AppContext.BaseDirectory,"self-test-data");Directory.CreateDirectory(directory);
        options.ProfileOutputPathPrefix=Path.Combine(directory,Path.GetFileNameWithoutExtension(path)+"_"+provider+"_"+Guid.NewGuid().ToString("N"));
        options.EnableProfiling=true;
    }
    public Dictionary<string,Dictionary<string,int>> KernelProviders()
    {
        var result=new Dictionary<string,Dictionary<string,int>>();if(!profileModels)return result;
        foreach(var (name,session) in new[]{("video",video),("detector",detector),("people_rtmo",realtimePeople?.Session)})
        {
            if(session is null)continue;
            using var events=JsonDocument.Parse(File.ReadAllText(session.EndProfiling()));var counts=new Dictionary<string,int>();
            foreach(var item in events.RootElement.EnumerateArray())
                if(item.TryGetProperty("args",out var details)&&details.TryGetProperty("provider",out var provider))
                {string key=provider.GetString()!;counts[key]=counts.GetValueOrDefault(key)+1;}
            result[name]=counts;
        }
        return result;
    }
    private static float[] Run(InferenceSession session,float[] input,int[] shape,string? output=null)
    {
        using var values=session.Run([NamedOnnxValue.CreateFromTensor(session.InputMetadata.Keys.First(),new DenseTensor<float>(input,shape))],
            output is null?session.OutputMetadata.Keys.ToArray():[output]);
        return values.First().AsTensor<float>().ToArray();
    }
    public (Box Box,float Score)[] Detect(Mat frame,float threshold=.12f)
    {
        if(realtimePeople is not null)return realtimePeople.Predict(frame,threshold).Select(p=>(p.Box,p.Score)).ToArray();
        using var enhanced=Enhance(frame);using var resized=new Mat();Cv2.Resize(enhanced,resized,new(detectorSize,detectorSize),0,0,Realtime?InterpolationFlags.Linear:InterpolationFlags.Area);
        var bytes=Bytes(resized);float[] input=new float[3*detectorSize*detectorSize];int plane=detectorSize*detectorSize;
        for(int i=0;i<plane;i++){input[i]=bytes[3*i+2]/255f;input[plane+i]=bytes[3*i+1]/255f;input[plane*2+i]=bytes[3*i]/255f;}
        using var result=detector!.Run([NamedOnnxValue.CreateFromTensor("input",new DenseTensor<float>(input,[1,3,detectorSize,detectorSize]))]);
        var boxes=result.First(v=>v.Name=="boxes").AsTensor<float>().ToArray();var scores=result.First(v=>v.Name=="scores").AsTensor<float>().ToArray();
        var labels=result.First(v=>v.Name=="labels").AsTensor<long>().ToArray();List<(Box,float)> people=[];
        for(int i=0;i<scores.Length;i++)
        {
            if(labels[i]!=1||!float.IsFinite(scores[i])||scores[i]<threshold||boxes.Skip(i*4).Take(4).Any(v=>!float.IsFinite(v)))continue;
            var b=new Box(Math.Clamp(boxes[i*4]/320,0,1),Math.Clamp(boxes[i*4+1]/320,0,1),Math.Clamp(boxes[i*4+2]/320,0,1),Math.Clamp(boxes[i*4+3]/320,0,1));
            if(b.Width<.02||b.Height<.04||people.Any(p=>Box.Iou(p.Item1,b)>.45))continue;
            people.Add((b,scores[i]));if(people.Count==12)break;
        }
        return people.ToArray();
    }
    public (Joint[] Image,Joint[] World) Pose(Mat frame,Box box,Joint[]? previous=null)
    {
        if(realtimePeople is not null)
        {
            var match=realtimePeople.Predict(frame).OrderByDescending(p=>Box.Iou(box,p.Box)).FirstOrDefault();
            return(match is not null&&Box.Iou(box,match.Box)>.2f?match.Joints:[],[]);
        }
        if(realtimePose is not null)return(realtimePose.Predict(frame,box),[]);
        float cx=(box.X1+box.X2)*frame.Width*.5f,cy=(box.Y1+box.Y2)*frame.Height*.5f;
        float side=Math.Max(box.Width*frame.Width,box.Height*frame.Height)*1.25f,rotation=0;
        if(previous is {Length:33} && new[]{11,12,23,24}.All(i=>previous[i].Confidence>.5))
        {
            float hx=(previous[23].X+previous[24].X)*frame.Width*.5f,hy=(previous[23].Y+previous[24].Y)*frame.Height*.5f;
            float sx=(previous[11].X+previous[12].X)*frame.Width*.5f,sy=(previous[11].Y+previous[12].Y)*frame.Height*.5f;
            rotation=MathF.Atan2(sx-hx,hy-sy);
        }
        double scale=256/Math.Max(24,side),cos=Math.Cos(rotation),sin=Math.Sin(rotation);
        using var affine=Mat.FromArray(new double[,]{{scale*cos,-scale*sin,128-scale*(cos*cx-sin*cy)},{scale*sin,scale*cos,128-scale*(sin*cx+cos*cy)}});
        using var crop=new Mat();using var enhanced=Enhance(frame);Cv2.WarpAffine(enhanced,crop,affine,new(256,256),InterpolationFlags.Linear,BorderTypes.Constant,Scalar.Black);
        var bytes=Bytes(crop);float[] input=new float[256*256*3];for(int i=0;i<input.Length;i+=3){input[i]=bytes[i+2]/255f;input[i+1]=bytes[i+1]/255f;input[i+2]=bytes[i]/255f;}
        using var output=landmarks!.Run([NamedOnnxValue.CreateFromTensor("input_1",new DenseTensor<float>(input,[1,256,256,3]))],["Identity","Identity_1","Identity_3","Identity_4"]);
        float presence=output.First(v=>v.Name=="Identity_1").AsTensor<float>()[0,0];if(!float.IsFinite(presence)||presence<.5)return([],[]);
        var raw=output.First(v=>v.Name=="Identity").AsTensor<float>().ToArray();var world=output.First(v=>v.Name=="Identity_4").AsTensor<float>().ToArray();
        var heatmap=output.First(v=>v.Name=="Identity_3").AsTensor<float>().ToArray();Refine(raw,heatmap);
        Joint[] imageJoints=new Joint[33],worldJoints=new Joint[33];
        for(int j=0;j<33;j++)
        {
            float dx=(raw[j*5]-128)/(float)scale,dy=(raw[j*5+1]-128)/(float)scale;
            float x=(float)(cx+cos*dx+sin*dy)/frame.Width,y=(float)(cy-sin*dx+cos*dy)/frame.Height;
            float conf=Sigmoid(raw[j*5+3]);if(x<0||x>1||y<0||y>1)conf=0;
            imageJoints[j]=new(x,y,raw[j*5+2]/(float)scale/frame.Width,conf);
            worldJoints[j]=new((float)(cos*world[3*j]+sin*world[3*j+1]),(float)(-sin*world[3*j]+cos*world[3*j+1]),world[3*j+2],conf);
            if(!float.IsFinite(x)||!float.IsFinite(y)||!float.IsFinite(world[3*j])||!float.IsFinite(world[3*j+1])||!float.IsFinite(world[3*j+2])||!float.IsFinite(conf))return([],[]);
        }
        return(imageJoints,worldJoints);
    }
    public static void Refine(float[] coordinates,float[] heatmap)
    {
        if(coordinates.Length!=195||heatmap.Length!=64*64*39)throw new ArgumentException("BlazePose热图尺寸不兼容。");
        for(int j=0;j<33;j++)
        {
            int x=(int)(coordinates[j*5]/4),y=(int)(coordinates[j*5+1]/4);if(x<0||x>=64||y<0||y>=64)continue;
            float sum=0,wx=0,wy=0,max=0;
            for(int row=Math.Max(0,y-3);row<Math.Min(64,y+4);row++)for(int col=Math.Max(0,x-3);col<Math.Min(64,x+4);col++)
            {
                float value=Sigmoid(heatmap[(row*64+col)*39+j]);sum+=value;wx+=col*value;wy+=row*value;max=Math.Max(max,value);
            }
            if(max>=.5&&sum>0){coordinates[j*5]=wx/sum*4;coordinates[j*5+1]=wy/sum*4;}
        }
    }
    public float PoseRisk(Joint[][] frames,float aspect)
    {
        var result=Run(pose,PoseFeatures.Build(frames,aspect),[1,169,48]);float score=1/(1+MathF.Exp(result[0]-result[1]));
        if(!float.IsFinite(score))throw new InvalidDataException("姿态模型输出异常。");return score;
    }
    public float[]? ActionScores(Joint[][] frames)=>actions?.Predict(frames);
    public float[] VideoScores(IReadOnlyList<byte[]> bgrFrames)=>PredictVideo(bgrFrames).Scores;
    public byte[] PrepareVideoFrame(Mat frame)
    {
        using var resized=new Mat();
        if(!preserveAspect){Cv2.Resize(frame,resized,new(FrameWidth,FrameHeight),0,0,InterpolationFlags.Area);return Bytes(resized);}
        double scale=Math.Min(FrameWidth/(double)frame.Width,FrameHeight/(double)frame.Height);
        int width=Math.Max(1,(int)Math.Round(frame.Width*scale)),height=Math.Max(1,(int)Math.Round(frame.Height*scale));
        Cv2.Resize(frame,resized,new(width,height),0,0,InterpolationFlags.Area);
        using var canvas=new Mat(FrameHeight,FrameWidth,MatType.CV_8UC3,new Scalar(104,116,124));
        using(var region=new Mat(canvas,new Rect((FrameWidth-width)/2,(FrameHeight-height)/2,width,height)))resized.CopyTo(region);
        return Bytes(canvas);
    }
    public VideoEvidence PredictVideo(IReadOnlyList<byte[]> bgrFrames)
    {
        if(bgrFrames.Count<6)throw new ArgumentException("有效视频窗口不足。");int pixels=FrameWidth*FrameHeight;
        if(bgrFrames.Any(b=>b.Length!=pixels*3))throw new ArgumentException("视频帧尺寸无效。");float[] input=new float[ClipLength*pixels*3];
        var selected=Enumerable.Range(0,ClipLength).Select(t=>bgrFrames[(int)Math.Round(t*(bgrFrames.Count-1)/(double)(ClipLength-1))]).ToArray();
        float gamma=1;
        if(LowLightTrained)
        {
            var means=selected.Select(b=>
            {
                using var mat=Mat.FromPixelData(FrameHeight,FrameWidth,MatType.CV_8UC3,b);using var resized=new Mat();using var gray=new Mat();
                Cv2.Resize(mat,resized,new(160,90),0,0,InterpolationFlags.Linear);Cv2.CvtColor(resized,gray,ColorConversionCodes.BGR2GRAY);return Cv2.Mean(gray).Val0;
            }).Order().ToArray();
            double mean=(means[(ClipLength-1)/2]+means[ClipLength/2])*.5;if(mean>=1&&mean<65)gamma=(float)Math.Clamp(Math.Log(80/255d)/Math.Log(mean/255d),.3,1);
        }
        for(int t=0;t<ClipLength;t++)
        {
            var b=selected[t];if(b.Length!=pixels*3)throw new ArgumentException("视频帧尺寸无效。");
            for(int i=0;i<pixels;i++)for(int c=0;c<3;c++)input[t*pixels*3+3*i+c]=gamma<1?(byte)(255*Math.Pow(b[3*i+2-c]/255d,gamma)):b[3*i+2-c];
        }
        using var output=video.Run([NamedOnnxValue.CreateFromTensor("input",new DenseTensor<float>(input,[1,ClipLength,FrameHeight,FrameWidth,3]))]);
        var values=output.First(v=>v.Name=="probabilities").AsTensor<float>().ToArray();
        var embedding=output.First(v=>v.Name=="embedding").AsTensor<float>().ToArray();var attention=output.First(v=>v.Name=="attention").AsTensor<float>().ToArray();
        if(values.Length!=Classes.Length||embedding.Length!=EmbeddingSize||values.Any(v=>!ValidUnit(v))||embedding.Any(v=>!float.IsFinite(v))||attention.Any(v=>!ValidUnit(v)))
            throw new InvalidDataException($"视频模型输出异常：分类范围 [{values.Min():R}, {values.Max():R}]；关注图范围 [{attention.Min():R}, {attention.Max():R}]；特征有限 {embedding.All(float.IsFinite)}。");
        // DirectML may round normalized attention to 1 + one float32 ULP.
        return new(values.Select(v=>Math.Clamp(v,0,1)).ToArray(),embedding,attention.Select(v=>Math.Clamp(v,0,1)).ToArray());
    }
    public static bool ValidUnit(float value)=>float.IsFinite(value)&&value>=-1e-5f&&value<=1+1e-5f;
    public float Risk(float[] values)=>Classes.Select((name,i)=>(name,i)).Where(p=>p.name is "fall" or "fallen").Sum(p=>values[p.i]);
    public static float Sigmoid(float x)=>1/(1+MathF.Exp(-Math.Clamp(x,-80,80)));
    public static Mat Enhance(Mat frame)
    {
        using var gray=new Mat();Cv2.CvtColor(frame,gray,ColorConversionCodes.BGR2GRAY);double brightness=Cv2.Mean(gray).Val0;
        if(brightness>=65)return frame.Clone();
        byte[] table=Enumerable.Range(0,256).Select(i=>(byte)(255*Math.Pow(i/255d,Math.Max(.35,brightness/65)))).ToArray();
        using var lookup=Mat.FromArray(table);var result=new Mat();Cv2.LUT(frame,lookup,result);return result;
    }
    public static byte[] Bytes(Mat frame)
    {
        if(!frame.IsContinuous()) { using var copy=frame.Clone();return Bytes(copy); }
        byte[] bytes=new byte[frame.Rows*frame.Cols*frame.ElemSize()];Marshal.Copy(frame.Data,bytes,0,bytes.Length);return bytes;
    }
    private readonly object observationGate=new();
    public PeopleObservation[] ObservePeople(Mat frame)
    {lock(observationGate)return realtimePeople?.Predict(frame)??Detect(frame).Select(p=>new PeopleObservation(p.Box,p.Score,[])).ToArray();}
    public void Dispose(){actions?.Dispose();realtimePeople?.Dispose();realtimePose?.Dispose();detector?.Dispose();landmarks?.Dispose();pose.Dispose();video.Dispose();}
}
