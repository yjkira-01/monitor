using System.Buffers.Binary;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using System.Text.Json;
using CareMonitor.Models;
using CareMonitor.Services;
namespace CareMonitor;
public static class DeviceRegression
{
    private static void Require(bool value,string message){if(!value)throw new InvalidOperationException("Device regression: "+message);}
    public static void Run(Dictionary<string,object> result)
    {
        var devices=new[]{new CameraDevices.Camera(0,"C60E","left"),new CameraDevices.Camera(1,"C60E","right")};
        Require(CameraDevices.Resolve(devices,"right","C60E",0)?.Index==1,"same-name unique identity");
        Require(CameraDevices.Resolve(devices,"gone","C60E",0) is null,"missing identity never falls back to other camera");
        Require(CameraDevices.Resolve(devices,"","C60E",7) is null,"ambiguous legacy selection rejected");result["camera_unique_identity"]=true;
        string directory=Path.Combine(AppContext.BaseDirectory,"self-test-data/device-regression");Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory,"monitor.json"),"{");File.WriteAllText(Path.Combine(directory,"monitor.json.bak"),"{");
        var recovered=new AppStore(directory);Require(recovered.RecoveryNote is not null&&recovered.Data.Events.Count==0,"damaged archive and damaged backup do not crash startup");result["corrupt_backup_recovery"]=true;
        var calibration=new StereoCalibration{LeftDeviceId="left",RightDeviceId="right",Width=640,Height=480,LeftK=[600,0,320,0,600,240,0,0,1],RightK=[600,0,320,0,600,240,0,0,1],LeftD=new double[5],RightD=new double[5],Rotation=[1,0,0,0,1,0,0,0,1],Translation=[-.2,0,0],ReprojectionError=.2,Samples=20,SquareMillimeters=25};
        calibration.Save(Path.Combine(directory,"calibration.json"));calibration=StereoCalibration.Load(Path.Combine(directory,"calibration.json"));
        PersonState Person(int id,float offset,float depth,bool right)
        {
            var pose=Enumerable.Range(0,17).Select(j=>new Joint((600*((j%2==0?-.12f:.12f)+offset-(right?.2f:0))/depth+320)/640,(600*(j*.042f+offset*.22f-.4f)/depth+240)/480,0,.95f)).ToArray();
            return new(id,new(.1f,.1f,.8f,.9f),.95f,pose,[],0,Activity:"standing");
        }
        var left=Person(1,-.5f,3,false);var right=Person(9,-.5f,3,true);
        var reconstructed=new StereoGeometry(calibration).Reconstruct([left],[right],Stopwatch.GetTimestamp());
        Require(reconstructed.Length==1&&reconstructed[0].ObservedJoints==17&&Math.Abs(reconstructed[0].Position.Z-3)<1e-4,"triangulation metric depth");
        var multiple=new StereoGeometry(calibration).Reconstruct([left,Person(2,.8f,4,false)],[Person(10,.8f,4,true),right],Stopwatch.GetTimestamp());
        Require(multiple.Length==2&&multiple.Single(p=>p.Id==1).RightId==9&&multiple.Single(p=>p.Id==2).RightId==10,"cross-view multi-person association");
        var impossible=right with{ImageJoints=right.ImageJoints.Select(j=>j with{Y=j.Y+.3f}).ToArray()};
        Require(new StereoGeometry(calibration).Reconstruct([left],[impossible],Stopwatch.GetTimestamp()).Length==0,"wrong epipolar association rejected");
        var partial=right with{ImageJoints=right.ImageJoints.Select((j,i)=>i>=11?j with{Confidence=0}:j).ToArray()};
        var incomplete=new StereoGeometry(calibration).Reconstruct([left],[partial],Stopwatch.GetTimestamp());
        Require(incomplete.Length==1&&incomplete[0].ObservedJoints==11&&incomplete[0].Joints[15] is null,"missing joints not invented");
        result["stereo_geometry_and_partial_rejection"]=true;
        var solved=StereoCalibrationSession.SyntheticCheck();result["synthetic_chessboard_calibration"]=new{solved.Samples,solved.Baseline,solved.ReprojectionError};
        TimedObservation Sample(int seq,long tick)=>new(new([],640,480,0,0,30,false,false,false,"fixture",1,seq,tick),new(1,0,[],null,"","",0,"fixture"));
        var pairing=new StereoPairing();long now=Stopwatch.GetTimestamp();
        Require(pairing.Update(Sample(1,now),Sample(1,now-(long)(Stopwatch.Frequency*.12))) is null,"120ms skew rejected");
        Require(pairing.Update(Sample(2,now),Sample(2,now-(long)(Stopwatch.Frequency*.02)))?.PairDeltaMs is >19 and <21,"20ms skew paired");
        Require(pairing.Update(Sample(3,now-Stopwatch.Frequency),Sample(3,now-Stopwatch.Frequency)) is null,"expired observation cleared");
        Require(pairing.Update(null,null) is null,"disconnect cleared");result["stereo_pairing_and_expiry"]=true;
        pairing.Reset(calibration);
        Require(pairing.Update(Sample(4,now) with{Frame=Sample(4,now).Frame with{Width=320}},Sample(4,now)) is null,"driver-negotiated resolution mismatch rejects calibration");result["stereo_actual_resolution_binding"]=true;
        var packet=new byte[88];new byte[]{2,1,4,3,6,5,8,7}.CopyTo(packet,0);
        void U(int offset,uint v)=>BinaryPrimitives.WriteUInt32LittleEndian(packet.AsSpan(offset),v);
        void F(int offset,float v)=>BinaryPrimitives.WriteSingleLittleEndian(packet.AsSpan(offset),v);
        U(12,88);U(20,12);U(32,3);U(40,1012);U(44,12);U(48,5);F(52,1.7f);F(56,.1f);U(60,1021);U(64,4);U(68,1);U(72,1011);U(76,3);packet[80]=5;packet[81]=253;packet[82]=255;
        var radar=RadarParser.Parse(packet);Require(radar?.Heights?.Single().MaxZ==1.7f&&radar.Presence==true&&radar.PreviousPointTargets!.SequenceEqual(new byte[]{5,253,255}),"additional TLV independent-layout fixture");
        F(52,float.NaN);Require(RadarParser.Parse(packet) is null,"invalid radar height rejected");result["radar_height_presence_previous_frame_tlv"]=true;
        var rules=new RadarAnalytics();var zone=new RadarZone(Guid.NewGuid(),"bed",-1,1,1,3,0,2);
        var events=new List<MonitorEvent>();RadarObservation? state=null;
        RadarSnapshot Snapshot(uint frame,float x,float height)=>new(DateTimeOffset.Now,"fixture",true,new(frame,[],[new(5,new(x,2,1),Vector3.Zero,.9f)],[],Heights:[new(5,height,0)]));
        for(int i=0;i<25;i++)state=rules.Update(Snapshot((uint)i,0,i<20?1.6f:.6f),i*.09,[zone],events.Add);
        Require(state!.Targets.Single().HeightDrop&&events.Any(e=>e.Kind.Contains("高度突降")),"time-based height-drop observation");
        rules.Update(new(DateTimeOffset.Now,"fixture",true,new(40,[],[],[])),4,[zone],events.Add);
        rules.Update(Snapshot(50,2,1.6f),5,[zone],events.Add);
        Require(!events.Any(e=>e.Kind=="离开区域"),"loss of tracking never means bed exit");
        result["radar_activity_height_drop_and_tracking_loss"]=true;
        string configuration=RadarSetup.Configuration(Path.Combine(AppContext.BaseDirectory,"Assets"),new(){RadarHeight=1.2,RadarAzimuth=5,RadarElevation=-10});
        Require(configuration.Contains("sensorPosition 1.2 5 -10")&&configuration.Contains("trackingCfg 1 2 800 1"),"embedded single-target firmware configuration");result["radar_embedded_config"]=true;
        Require(VideoService.Assessment(.7f,2,.8f,false,"lying",true)=="疑似跌倒 · 请核查"&&VideoService.Assessment(.1f,0,null,false,"lying",true)=="动作异常 · 请核查","binary and motion evidence separated");
        result["binary_status_separated_from_motion_evidence"]=true;
    }
}
