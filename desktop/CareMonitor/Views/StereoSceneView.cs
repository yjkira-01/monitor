using System.Numerics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using CareMonitor.Services;
using Quaternion=System.Windows.Media.Media3D.Quaternion;
namespace CareMonitor.Views;
public sealed class StereoSceneView : Grid
{
    private readonly Model3DGroup people=new();
    private readonly Dictionary<int,GeometryModel3D[]> models=[];
    private readonly PerspectiveCamera camera=new(){FieldOfView=42,UpDirection=new(0,1,0)};
    private static readonly MeshGeometry3D sphere=Sphere();
    private readonly Color[] colors=[Color.FromRgb(36,117,79),Color.FromRgb(49,110,160),Color.FromRgb(159,114,61),Color.FromRgb(112,100,172),Color.FromRgb(161,75,96),Color.FromRgb(66,135,131)];
    private double yaw=.2,pitch=.2,distance=7;
    private Point? drag;
    public StereoSceneView()
    {
        ClipToBounds=true;Background=new SolidColorBrush(Color.FromRgb(234,240,231));
        var scene=new Model3DGroup();scene.Children.Add(new AmbientLight(Color.FromRgb(175,186,175)));scene.Children.Add(new DirectionalLight(Colors.White,new(-1,-1,-1)));scene.Children.Add(people);
        var viewport=new Viewport3D{Camera=camera};viewport.Children.Add(new ModelVisual3D{Content=scene});Children.Add(viewport);
        MouseLeftButtonDown+=(_,e)=>{drag=e.GetPosition(this);CaptureMouse();};MouseLeftButtonUp+=(_,_)=>{drag=null;ReleaseMouseCapture();};
        MouseMove+=(_,e)=>{if(drag is not Point p)return;var q=e.GetPosition(this);yaw+=(q.X-p.X)*.008;pitch=Math.Clamp(pitch+(q.Y-p.Y)*.005,-.4,1.2);drag=q;Camera();};
        MouseWheel+=(_,e)=>{distance=Math.Clamp(distance-e.Delta*.004,2,16);Camera();};Camera();
    }
    private void Camera(){camera.Position=new(distance*Math.Sin(yaw)*Math.Cos(pitch),distance*Math.Sin(pitch),3-distance*Math.Cos(yaw)*Math.Cos(pitch));camera.LookDirection=new(-camera.Position.X,-camera.Position.Y,3-camera.Position.Z);}
    private static MeshGeometry3D Sphere()
    {
        var m=new MeshGeometry3D();for(int y=0;y<=8;y++)for(int x=0;x<=12;x++){double a=y*Math.PI/8,b=x*2*Math.PI/12;m.Positions.Add(new(Math.Sin(a)*Math.Cos(b),Math.Cos(a),Math.Sin(a)*Math.Sin(b)));}
        for(int y=0;y<8;y++)for(int x=0;x<12;x++){int i=y*13+x;foreach(int k in new[]{i,i+13,i+1,i+1,i+13,i+14})m.TriangleIndices.Add(k);}m.Freeze();return m;
    }
    private static Point3D Point(Vector3 p)=>new(p.X,-p.Y,p.Z);
    public void Update(StereoPerson[] observations)
    {
        foreach(int id in models.Keys.Where(id=>observations.All(p=>p.Id!=id)).ToArray())
        {foreach(var model in models[id])people.Children.Remove(model);models.Remove(id);}
        foreach(var person in observations)
        {
            if(!models.TryGetValue(person.Id,out var figure))
            {
                var material=new DiffuseMaterial(new SolidColorBrush(colors[Math.Abs(person.Id)%colors.Length]));material.Freeze();
                figure=Enumerable.Range(0,17+StereoGeometry.Bones.Length).Select(_=>new GeometryModel3D(sphere,material){BackMaterial=material}).ToArray();
                models[person.Id]=figure;foreach(var model in figure)people.Children.Add(model);
            }
            void Place(int i,Point3D p,double sx,double sy,double sz,Quaternion? rotation=null)
            {var group=new Transform3DGroup();group.Children.Add(new ScaleTransform3D(sx,sy,sz));if(rotation.HasValue)group.Children.Add(new RotateTransform3D(new QuaternionRotation3D(rotation.Value)));group.Children.Add(new TranslateTransform3D(p.X,p.Y,p.Z));figure[i].Transform=group;}
            for(int j=0;j<17;j++)
            {if(person.Joints[j] is Vector3 p)Place(j,Point(p),j==0?.07:.025,j==0?.09:.025,j==0?.07:.025);else Place(j,new(),0,0,0);}
            for(int i=0;i<StereoGeometry.Bones.Length;i++)
            {
                var (a,b)=StereoGeometry.Bones[i];
                if(person.Joints[a] is not Vector3 av||person.Joints[b] is not Vector3 bv){Place(17+i,new(),0,0,0);continue;}
                Point3D ap=Point(av),bp=Point(bv);Vector3D delta=bp-ap;double length=delta.Length;if(length<1e-6){Place(17+i,new(),0,0,0);continue;}delta.Normalize();var axis=Vector3D.CrossProduct(new(0,1,0),delta);
                var q=axis.Length>1e-8?new Quaternion(axis,Math.Acos(Math.Clamp(delta.Y,-1,1))*180/Math.PI):new Quaternion(new Vector3D(1,0,0),delta.Y<0?180:0);
                Place(17+i,new((ap.X+bp.X)/2,(ap.Y+bp.Y)/2,(ap.Z+bp.Z)/2),.017,length/2,.017,q);
            }
        }
    }
}
