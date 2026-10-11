using System.IO;
using System.Security.Cryptography;
using System.Text.Json;
using CareMonitor.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CareMonitor.Services;
public sealed class PretrainedActions : IDisposable
{
    private readonly InferenceSession session;
    public PretrainedActions(string directory)
    {
        string path=Path.Combine(directory,"actions.onnx");
        using var meta=JsonDocument.Parse(File.ReadAllText(Path.Combine(directory,"actions.json")));
        using var stream=File.OpenRead(path);
        if(Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant()!=meta.RootElement.GetProperty("sha256").GetString())
            throw new InvalidDataException("动作模型权重校验失败。");
        using var options=new SessionOptions{IntraOpNumThreads=2,InterOpNumThreads=1};session=new(path,options);
    }
    public float[] Predict(Joint[][] frames)
    {
        if(frames.Length<6||frames.Any(p=>p.Length!=17))throw new ArgumentException("动作窗口需包含 COCO 17 点时序。");
        var input=new float[2*100*17*3];
        for(int t=0;t<100;t++)
        {
            var pose=frames[(int)Math.Round(t*(frames.Length-1)/99d)];
            for(int j=0;j<17;j++)
            {
                var p=pose[j];int at=(t*17+j)*3;
                input[at]=p.Confidence>0?p.X*2-1:-1;input[at+1]=p.Confidence>0?p.Y*2-1:-1;input[at+2]=p.Confidence;
            }
        }
        using var values=session.Run([NamedOnnxValue.CreateFromTensor("poses",new DenseTensor<float>(input,[1,2,100,17,3]))]);
        var result=values.First().AsTensor<float>().ToArray();
        if(result.Length!=60||result.Any(p=>!NativeModels.ValidUnit(p)))throw new InvalidDataException("动作模型输出异常。");
        return result.Select(v=>Math.Clamp(v,0,1)).ToArray();
    }
    private static readonly string[] names=["喝水","进食","刷牙","梳头","物品掉落","拾取物品","投掷","坐下","起身","拍手","阅读","书写","撕纸","穿外套","脱外套","穿鞋","脱鞋","戴眼镜","摘眼镜","戴帽子","摘帽子","举手","挥手","踢物品","手伸入口袋","单脚跳","跳跃","打电话","使用手机","使用键盘","指向物体","自拍","查看时间","搓手","点头或低头","摇头","擦脸","敬礼","双手合十","交叉双手","咳嗽或打喷嚏","身体摇晃","跌倒动作","触碰头部","触碰胸部","触碰背部","触碰颈部","弯身不适动作","扇风","挥拳动作","踢人动作","推人动作","拍背","指向他人","拥抱","递送物品","触碰他人口袋","握手","相向行走","相背行走"];
    public static ActionCandidate[] Top(float[] values)=>values.Select((score,index)=>(score,index)).OrderByDescending(x=>x.score).Take(3).Select(x=>new ActionCandidate(names[x.index],x.score)).ToArray();
    public void Dispose()=>session.Dispose();
}
