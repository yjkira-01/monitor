using System.Diagnostics;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

if(args.Length!=4)throw new ArgumentException("Usage: model.onnx input-f32.bin reference.json output.json");
var raw=File.ReadAllBytes(args[1]);
if(raw.Length!=3*576*576*sizeof(float))throw new InvalidDataException("Invalid detector input size.");
float[] image=new float[raw.Length/sizeof(float)];Buffer.BlockCopy(raw,0,image,0,raw.Length);
if(image.Any(v=>!float.IsFinite(v)||v<0||v>1))throw new InvalidDataException("Invalid RGB values.");
using var reference=JsonDocument.Parse(File.ReadAllText(args[2]));
float[] expectedBoxes=reference.RootElement.GetProperty("boxes").EnumerateArray().Select(e=>e.GetSingle()).ToArray();
float[] expectedScores=reference.RootElement.GetProperty("scores").EnumerateArray().Select(e=>e.GetSingle()).ToArray();
var results=new List<object>();
bool passed=true;
foreach(bool gpu in new[]{false,true})
{
    try
    {
        using var options=new SessionOptions{IntraOpNumThreads=3,InterOpNumThreads=1,
            EnableProfiling=true,ProfileOutputPathPrefix=Path.Combine(Path.GetDirectoryName(args[3])!,gpu?"dml":"cpu")};
        if(gpu){options.EnableMemoryPattern=false;options.ExecutionMode=ExecutionMode.ORT_SEQUENTIAL;options.AppendExecutionProvider_DML(0);}
        var watch=Stopwatch.StartNew();using var session=new InferenceSession(args[0],options);
        double loadMs=watch.Elapsed.TotalMilliseconds;
        var input=NamedOnnxValue.CreateFromTensor("input",new DenseTensor<float>(image,[1,3,576,576]));
        float[] boxes=[],scores=[];long[] labels=[];
        double[] timings=new double[10];
        for(int i=-2;i<10;i++)
        {
            watch.Restart();using var output=session.Run([input]);
            boxes=output.First(o=>o.Name=="boxes").AsTensor<float>().ToArray();
            scores=output.First(o=>o.Name=="scores").AsTensor<float>().ToArray();
            labels=output.First(o=>o.Name=="labels").AsTensor<long>().ToArray();
            if(i>=0)timings[i]=watch.Elapsed.TotalMilliseconds;
        }
        if(scores.Length!=300||boxes.Length!=1200||labels.Any(v=>v!=1)||boxes.Concat(scores).Any(v=>!float.IsFinite(v)))
            throw new InvalidDataException("Invalid detector output.");
        // Check the public frame's accepted person, avoiding nearly tied unused queries.
        int count=scores.Count(v=>v>=.35f);int referenceCount=expectedScores.Count(v=>v>=.35f);
        if(count==0||count!=referenceCount)throw new InvalidDataException("Detection count differs from the CPU reference.");
        float scoreError=0,boxError=0;
        for(int i=0;i<count;i++)
        {
            scoreError=Math.Max(scoreError,Math.Abs(scores[i]-expectedScores[i]));
            for(int j=0;j<4;j++)boxError=Math.Max(boxError,Math.Abs(boxes[i*4+j]-expectedBoxes[i*4+j]));
        }
        if(scoreError>.01||boxError>1)throw new InvalidDataException("Accepted boxes differ from the reference.");
        string profile=session.EndProfiling();using var events=JsonDocument.Parse(File.ReadAllText(profile));
        var providers=new Dictionary<string,int>();
        foreach(var item in events.RootElement.EnumerateArray())
            if(item.TryGetProperty("args",out var details)&&details.TryGetProperty("provider",out var value))
            {string name=value.GetString()!;providers[name]=providers.GetValueOrDefault(name)+1;}
        if(gpu&&!providers.Keys.Any(p=>p.Contains("Dml",StringComparison.OrdinalIgnoreCase)))
            throw new InvalidOperationException("No DirectML kernel was executed.");
        results.Add(new{device=gpu?"DirectML adapter 0":"CPU",load_ms=loadMs,mean_ms=timings.Average(),
            p95_ms=timings.Order().ElementAt(9),timings_ms=timings,accepted_queries=count,
            max_score_error=scoreError,max_box_error=boxError,providers,passed=true});
    }
    catch(Exception ex){passed=false;results.Add(new{device=gpu?"DirectML adapter 0":"CPU",passed=false,error=ex.Message});}
}
File.WriteAllText(args[3],JsonSerializer.Serialize(new{runtime="C# / ONNX Runtime DirectML 1.20.1",results},new JsonSerializerOptions{WriteIndented=true}));
Console.WriteLine(File.ReadAllText(args[3]));
Environment.ExitCode=passed?0:1;
