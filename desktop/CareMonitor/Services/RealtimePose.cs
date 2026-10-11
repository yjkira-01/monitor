using System.IO;
using System.Security.Cryptography;
using System.Text.Json;
using CareMonitor.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace CareMonitor.Services;

/// <summary>RTMPose SimCC deployment, following Tau-J/rtmlib (Apache-2.0).</summary>
public sealed class RealtimePose : IDisposable
{
    private readonly InferenceSession session;
    public RealtimePose(string directory,Func<string,InferenceSession> create)
    {
        string path=Path.Combine(directory,"rtmpose.onnx");
        using var meta=JsonDocument.Parse(File.ReadAllText(Path.Combine(directory,"rtmpose.json")));
        using var stream=File.OpenRead(path);
        if(Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant()!=meta.RootElement.GetProperty("sha256").GetString())
            throw new InvalidDataException("RTMPose 权重校验失败。");
        session=create(path);
    }
    public Joint[] Predict(Mat frame,Box box,bool enhance=true)
    {
        float cx=(box.X1+box.X2)*frame.Width/2,cy=(box.Y1+box.Y2)*frame.Height/2;
        float width=box.Width*frame.Width*1.25f,height=box.Height*frame.Height*1.25f;
        if(width>height*.75f)height=width/.75f;else width=height*.75f;
        if(width<1||height<1)return [];
        using var affine=Mat.FromArray(new double[,]{{192d/width,0,96-192d*cx/width},{0,256d/height,128-256d*cy/height}});
        using var image=enhance?NativeModels.Enhance(frame):frame.Clone();using var crop=new Mat();
        Cv2.WarpAffine(image,crop,affine,new(192,256),InterpolationFlags.Linear,BorderTypes.Constant,Scalar.Black);
        var b=NativeModels.Bytes(crop);var input=new float[3*192*256];float[] mean=[123.675f,116.28f,103.53f],std=[58.395f,57.12f,57.375f];
        // Official rtmlib normalizes the OpenCV BGR crop directly; do not swap channels.
        for(int i=0;i<192*256;i++)for(int c=0;c<3;c++)input[c*192*256+i]=(b[i*3+c]-mean[c])/std[c];
        using var values=session.Run([NamedOnnxValue.CreateFromTensor(session.InputMetadata.Keys.First(),new DenseTensor<float>(input,[1,3,256,192]))]);
        var outputs=values.Select(v=>v.AsTensor<float>()).ToArray();
        var sx=outputs.Single(t=>t.Dimensions[^1]==384);var sy=outputs.Single(t=>t.Dimensions[^1]==512);
        var joints=new Joint[17];
        for(int j=0;j<17;j++)
        {
            int xi=0,yi=0;
            for(int k=1;k<384;k++)if(sx[0,j,k]>sx[0,j,xi])xi=k;
            for(int k=1;k<512;k++)if(sy[0,j,k]>sy[0,j,yi])yi=k;
            float x=(xi/2f/192*width+cx-width/2)/frame.Width,y=(yi/2f/256*height+cy-height/2)/frame.Height;
            float score=(sx[0,j,xi]+sy[0,j,yi])/2;
            if(!float.IsFinite(x)||!float.IsFinite(y)||!float.IsFinite(score))return [];
            joints[j]=new(x,y,0,x<0||x>1||y<0||y>1?0:Math.Clamp(score,0,1));
        }
        return joints;
    }
    public void Dispose()=>session.Dispose();
}
