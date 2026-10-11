using System.IO;
using System.Numerics;
using System.Text.Json;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using CareMonitor.Models;
using Quaternion = System.Numerics.Quaternion;

namespace CareMonitor.Views;
public sealed class GlbAvatar
{
    private sealed class Bone
    {
        public string Name=""; public int Parent=-1;public Vector3 Translation;public Vector3 Scale=Vector3.One;
        public Quaternion Rotation,RestRotation;public Matrix4x4 World,RestWorld;
    }
    private sealed class Surface
    {
        public Vector3[] Vertices=[],Normals=[];public Vector4[] Weights=[];public int[][] Joints=[];
        public MeshGeometry3D Geometry=new();
    }
    private readonly Bone[] bones;
    private readonly int[] jointNodes;
    private readonly Matrix4x4[] inverseBind;
    private readonly List<Surface> surfaces=[];
    private readonly Dictionary<string,int> names;
    private readonly List<(int Node,int A,int B,Vector3 Direction,Quaternion BindRotation)> segments=[];
    private readonly Vector3 origin;
    private readonly Quaternion restPelvis,restSpine,restNeck,restHead;
    private Quaternion body=Quaternion.Identity;
    private float bodyScale=1;
    public Model3DGroup Model { get; }=new();
    public GlbAvatar(string path)
    {
        byte[] file=File.ReadAllBytes(path);if(BitConverter.ToUInt32(file,0)!=0x46546c67)throw new InvalidDataException("人物资源不是GLB。");
        int jsonLength=BitConverter.ToInt32(file,12);using var document=JsonDocument.Parse(file.AsMemory(20,jsonLength));var root=document.RootElement;
        int dataStart=20+jsonLength+8;byte[] binary=file[dataStart..];
        var nodes=root.GetProperty("nodes").EnumerateArray().ToArray();bones=new Bone[nodes.Length];names=new(StringComparer.Ordinal);
        Vector3 Vec(JsonElement node,string key,Vector3 fallback)=>node.TryGetProperty(key,out var v)?new(v[0].GetSingle(),v[1].GetSingle(),v[2].GetSingle()):fallback;
        for(int i=0;i<nodes.Length;i++)
        {
            var n=nodes[i];Quaternion rotation=n.TryGetProperty("rotation",out var r)?new(r[0].GetSingle(),r[1].GetSingle(),r[2].GetSingle(),r[3].GetSingle()):Quaternion.Identity;
            bones[i]=new(){Name=n.TryGetProperty("name",out var name)?name.GetString()!:"",Translation=Vec(n,"translation",Vector3.Zero),Scale=Vec(n,"scale",Vector3.One),Rotation=rotation,RestRotation=rotation};
            if(bones[i].Name.Length>0)names[bones[i].Name]=i;
        }
        for(int i=0;i<nodes.Length;i++)if(nodes[i].TryGetProperty("children",out var children))foreach(var child in children.EnumerateArray())bones[child.GetInt32()].Parent=i;
        UpdateWorld();foreach(var bone in bones)bone.RestWorld=bone.World;
        var skin=root.GetProperty("skins")[0];jointNodes=skin.GetProperty("joints").EnumerateArray().Select(v=>v.GetInt32()).ToArray();
        var matrices=Read(skin.GetProperty("inverseBindMatrices").GetInt32(),root,binary);
        inverseBind=Enumerable.Range(0,jointNodes.Length).Select(i=>Matrix(matrices.Skip(i*16).Take(16).ToArray())).ToArray();
        foreach(var mesh in root.GetProperty("meshes").EnumerateArray())foreach(var primitive in mesh.GetProperty("primitives").EnumerateArray())
        {
            var attr=primitive.GetProperty("attributes");float[] p=Read(attr.GetProperty("POSITION").GetInt32(),root,binary),n=Read(attr.GetProperty("NORMAL").GetInt32(),root,binary),j=Read(attr.GetProperty("JOINTS_0").GetInt32(),root,binary),w=Read(attr.GetProperty("WEIGHTS_0").GetInt32(),root,binary);
            var surface=new Surface{Vertices=Enumerable.Range(0,p.Length/3).Select(i=>new Vector3(p[3*i],p[3*i+1],p[3*i+2])).ToArray(),
                Normals=Enumerable.Range(0,n.Length/3).Select(i=>new Vector3(n[3*i],n[3*i+1],n[3*i+2])).ToArray(),
                Joints=Enumerable.Range(0,j.Length/4).Select(i=>j.Skip(i*4).Take(4).Select(x=>(int)x).ToArray()).ToArray(),
                Weights=Enumerable.Range(0,w.Length/4).Select(i=>new Vector4(w[4*i],w[4*i+1],w[4*i+2],w[4*i+3])).ToArray()};
            surface.Geometry.TriangleIndices=new Int32Collection(Read(primitive.GetProperty("indices").GetInt32(),root,binary).Select(i=>(int)i));
            string meshName=mesh.TryGetProperty("name",out var mn)?mn.GetString()!:"";
            var color=meshName=="Face"?Color.FromRgb(44,63,52):meshName=="Face.001"?Color.FromRgb(225,236,220):Color.FromRgb(115,159,137);
            var material=new MaterialGroup();material.Children.Add(new DiffuseMaterial(new SolidColorBrush(color)));material.Children.Add(new SpecularMaterial(new SolidColorBrush(Color.FromRgb(76,101,85)),18));material.Freeze();
            Model.Children.Add(new GeometryModel3D(surface.Geometry,material){BackMaterial=material});surfaces.Add(surface);
        }
        int pelvis=names["pelvis"];origin=bones[pelvis].RestWorld.Translation;restPelvis=Rotation(bones[pelvis].RestWorld);restSpine=Rotation(bones[names["spine_03"]].RestWorld);
        restNeck=Rotation(bones[names["neck_01"]].RestWorld);restHead=Rotation(bones[names["Head"]].RestWorld);
        foreach(var (name,child,a,b) in new[]{("upperarm_l","lowerarm_l",11,13),("lowerarm_l","hand_l",13,15),("upperarm_r","lowerarm_r",12,14),("lowerarm_r","hand_r",14,16),
            ("thigh_l","calf_l",23,25),("calf_l","foot_l",25,27),("foot_l","ball_l",29,31),("thigh_r","calf_r",24,26),("calf_r","foot_r",26,28),("foot_r","ball_r",30,32)})
        {
            int node=names[name];segments.Add((node,a,b,Vector3.Normalize(bones[names[child]].RestWorld.Translation-bones[node].RestWorld.Translation),Rotation(bones[node].RestWorld)));
        }
        Skin(1,Vector3.Zero);
    }
    private static float[] Read(int id,JsonElement root,byte[] binary)
    {
        var accessor=root.GetProperty("accessors")[id];var view=root.GetProperty("bufferViews")[accessor.GetProperty("bufferView").GetInt32()];
        int type=accessor.GetProperty("componentType").GetInt32(),size=type is 5120 or 5121?1:type is 5122 or 5123?2:4;
        int components=accessor.GetProperty("type").GetString() switch{"SCALAR"=>1,"VEC2"=>2,"VEC3"=>3,"VEC4"=>4,"MAT4"=>16,_=>throw new InvalidDataException("人物资源格式不支持")};
        int count=accessor.GetProperty("count").GetInt32(),offset=(view.TryGetProperty("byteOffset",out var vo)?vo.GetInt32():0)+(accessor.TryGetProperty("byteOffset",out var ao)?ao.GetInt32():0);
        int stride=view.TryGetProperty("byteStride",out var s)?s.GetInt32():components*size;float[] values=new float[count*components];
        for(int i=0;i<count;i++)for(int c=0;c<components;c++)
        {
            int at=offset+i*stride+c*size;values[i*components+c]=type switch{5121=>binary[at],5123=>BitConverter.ToUInt16(binary,at),5125=>BitConverter.ToUInt32(binary,at),5126=>BitConverter.ToSingle(binary,at),_=>throw new InvalidDataException("人物资源数据类型不支持")};
        }
        return values;
    }
    private static Matrix4x4 Matrix(float[] m)=>new(m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[8],m[9],m[10],m[11],m[12],m[13],m[14],m[15]);
    private static Quaternion Rotation(Matrix4x4 m){Matrix4x4.Decompose(m,out _,out var q,out _);return q;}
    private void UpdateWorld()
    {
        bool[] done=new bool[bones.Length];
        void Update(int i)
        {
            if(done[i])return;var b=bones[i];if(b.Parent>=0)Update(b.Parent);
            b.World=Matrix4x4.CreateScale(b.Scale)*Matrix4x4.CreateFromQuaternion(b.Rotation)*Matrix4x4.CreateTranslation(b.Translation);
            if(b.Parent>=0)b.World*=bones[b.Parent].World;done[i]=true;
        }
        for(int i=0;i<bones.Length;i++)Update(i);
    }
    private void Orient(int i,Quaternion world)
    {
        var b=bones[i];var parent=b.Parent<0?Quaternion.Identity:Rotation(bones[b.Parent].World);
        b.Rotation=Quaternion.Normalize(Quaternion.Inverse(parent)*world);UpdateWorld();
    }
    private static Quaternion FromTo(Vector3 a,Vector3 b)
    {
        a=Vector3.Normalize(a);b=Vector3.Normalize(b);float dot=Vector3.Dot(a,b);
        if(dot<-.9999f)return Quaternion.CreateFromAxisAngle(Vector3.Normalize(Vector3.Cross(a,Math.Abs(a.X)<.9?Vector3.UnitX:Vector3.UnitY)),MathF.PI);
        return Quaternion.Normalize(new Quaternion(Vector3.Cross(a,b),1+dot));
    }
    public void SetPose(Joint[] joints,float horizontalOffset)
    {
        Vector3[] p=joints.Select(j=>new Vector3(j.X,-j.Y,-j.Z)).ToArray();
        if(joints.Length==33 && new[]{11,12,23,24}.All(i=>joints[i].Confidence>=.5))
        {
            Vector3 y=(p[11]+p[12]-p[23]-p[24])*.5f,x=p[23]-p[24],z=Vector3.Cross(x,y);
            if(y.LengthSquared()>.01&&z.LengthSquared()>1e-6)
            {
                bodyScale=Math.Clamp(y.Length()/.51f,.65f,1.35f);body=Basis(x,y,body);
                var upper=Basis(p[11]-p[12],y,body);var face=upper;
                if(joints[7].Confidence>=.5&&joints[8].Confidence>=.5)face=Basis(p[7]-p[8],(p[7]+p[8]-p[11]-p[12])*.5f,upper);
                Orient(names["pelvis"],body*restPelvis);Orient(names["spine_03"],upper*restSpine);
                Orient(names["neck_01"],Quaternion.Slerp(upper,face,.5f)*restNeck);Orient(names["Head"],face*restHead);
            }
        }
        if(joints.Length==33)foreach(var rig in segments)
        {
            if(Math.Min(joints[rig.A].Confidence,joints[rig.B].Confidence)<.4)continue;
            Vector3 direction=p[rig.B]-p[rig.A];if(direction.LengthSquared()<1e-6)continue;
            var delta=FromTo(Vector3.Transform(rig.Direction,body),Vector3.Normalize(direction));Orient(rig.Node,delta*body*rig.BindRotation);
        }
        Skin(bodyScale,new(horizontalOffset,0,0));
    }
    private static Quaternion Basis(Vector3 across,Vector3 up,Quaternion fallback)
    {
        Vector3 z=Vector3.Cross(across,up);if(up.LengthSquared()<.001||z.LengthSquared()<1e-6)return fallback;
        up=Vector3.Normalize(up);z=Vector3.Normalize(z);var x=Vector3.Normalize(Vector3.Cross(up,z));
        return Quaternion.CreateFromRotationMatrix(new(x.X,x.Y,x.Z,0,up.X,up.Y,up.Z,0,z.X,z.Y,z.Z,0,0,0,0,1));
    }
    private void Skin(float scale,Vector3 placement)
    {
        var matrices=jointNodes.Select((node,i)=>inverseBind[i]*bones[node].World).ToArray();
        float bottom=surfaces.SelectMany(s=>s.Vertices.Select((p,i)=>Skinned(p,s.Weights[i],s.Joints[i],matrices))).Min(p=>p.Y);
        foreach(var surface in surfaces)
        {
            var positions=new Point3DCollection(surface.Vertices.Length);var normals=new Vector3DCollection(surface.Vertices.Length);
            for(int i=0;i<surface.Vertices.Length;i++)
            {
                Vector3 value=Skinned(surface.Vertices[i],surface.Weights[i],surface.Joints[i],matrices);value=(value-origin)*scale+placement;value.Y+=(origin.Y-bottom)*scale-.95f;
                if(!float.IsFinite(value.X)||!float.IsFinite(value.Y)||!float.IsFinite(value.Z))throw new InvalidDataException("人物变换结果无效。");
                positions.Add(new(value.X,value.Y,value.Z));Vector3 normal=Vector3.Zero;Vector4 weights=surface.Weights[i];
                for(int k=0;k<4;k++)normal+=Vector3.TransformNormal(surface.Normals[i],matrices[surface.Joints[i][k]])*weights[k];
                normal=normal.LengthSquared()>1e-8?Vector3.Normalize(normal):Vector3.UnitY;normals.Add(new(normal.X,normal.Y,normal.Z));
            }
            surface.Geometry.Positions=positions;surface.Geometry.Normals=normals;
        }
    }
    private static Vector3 Skinned(Vector3 p,Vector4 weights,int[] joints,Matrix4x4[] matrices)
    {
        Vector3 value=Vector3.Zero;for(int k=0;k<4;k++)value+=Vector3.Transform(p,matrices[joints[k]])*weights[k];return value;
    }
    public IEnumerable<Vector3> Vertices()=>surfaces.SelectMany(s=>s.Geometry.Positions.Select(p=>new Vector3((float)p.X,(float)p.Y,(float)p.Z)));
}
