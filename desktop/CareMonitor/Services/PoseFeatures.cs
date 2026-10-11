using CareMonitor.Models;

namespace CareMonitor.Services;
public static class PoseFeatures
{
    public static float[] Build(IReadOnlyList<Joint[]> poses,float aspect)
    {
        if(poses.Count!=48||poses.Any(p=>p.Length!=33)||!float.IsFinite(aspect)||aspect<=0)throw new ArgumentException("需要48帧33关节与有效画面比例。");
        const int T=48;var output=new float[169*T];var centers=new (float X,float Y)[T];float[] scales=new float[T];bool[] torso=new bool[T];
        List<float> validScale=[];
        for(int t=0;t<T;t++)
        {
            var p=poses[t];torso[t]=new[]{11,12,23,24}.All(j=>Valid(p[j])&&p[j].Confidence>=.3);
            centers[t]=((Clean(p[23].X)+Clean(p[24].X))*aspect*.5f,(Clean(p[23].Y)+Clean(p[24].Y))*.5f);
            float sx=(Clean(p[11].X)+Clean(p[12].X))*aspect*.5f-centers[t].X;
            float sy=(Clean(p[11].Y)+Clean(p[12].Y))*.5f-centers[t].Y;
            scales[t]=MathF.Sqrt(sx*sx+sy*sy);if(torso[t])validScale.Add(scales[t]);
        }
        validScale.Sort();float scale=Math.Max(.02f,validScale.Count>0?validScale[(validScale.Count-1)/2]:1);
        for(int t=0;t<T;t++)
        {
            for(int j=0;j<33;j++)
            {
                var p=poses[t][j];float visibility=Valid(p)?Math.Clamp(p.Confidence,0,1):0;
                bool good=visibility>=.3 && torso[t];
                output[j*T+t]=good?Math.Clamp((p.X*aspect-centers[t].X)/scale,-6,6):0;
                output[(33+j)*T+t]=good?Math.Clamp((p.Y-centers[t].Y)/scale,-6,6):0;
                output[(132+j)*T+t]=visibility;
                if(t>0 && good && torso[t-1] && Valid(poses[t-1][j]) && poses[t-1][j].Confidence>=.3)
                {
                    output[(66+j)*T+t]=output[j*T+t]-output[j*T+t-1];
                    output[(99+j)*T+t]=output[(33+j)*T+t]-output[(33+j)*T+t-1];
                }
            }
            if(t>0&&torso[t]&&torso[t-1])
            {
                output[165*T+t]=Math.Clamp((centers[t].X-centers[t-1].X)/scale,-3,3);
                output[166*T+t]=Math.Clamp((centers[t].Y-centers[t-1].Y)/scale,-3,3);
            }
            output[167*T+t]=torso[t]?Math.Clamp(scales[t]/scale,0,6):0;output[168*T+t]=torso[t]?1:0;
        }
        return output;
    }
    private static bool Valid(Joint p)=>float.IsFinite(p.X)&&float.IsFinite(p.Y)&&float.IsFinite(p.Confidence);
    private static float Clean(float value)=>float.IsFinite(value)?value:0;
}
