using System.Numerics;
using System.Text.Json.Serialization;

namespace CareMonitor.Models;

public sealed class AppSettings
{
    public int DeviceSetupRevision { get; set; }
    public int CameraIndex { get; set; } = 1;
    public string CameraDeviceName { get; set; } = "EMEET SmartCam C60E 4K";
    public string CameraDeviceId { get; set; } = "";
    public string SecondaryCameraId { get; set; } = "";
    public int SecondaryCameraIndex { get; set; } = -1;
    public double RadarHeight { get; set; } = 1.8;
    public double RadarAzimuth { get; set; }
    public double RadarElevation { get; set; }
    public List<RadarZone> RadarZones { get; set; }=[];
    public int CameraWidth { get; set; } = 640;
    public int CameraHeight { get; set; } = 480;
    public int CameraFps { get; set; } = 30;
    public bool ManualExposure { get; set; }
    public bool ManualGain { get; set; }
    public bool LimitSlowExposure { get; set; }=true;
    public int Exposure { get; set; } = -5;
    public int Gain { get; set; } = 64;
    public bool UseGpu { get; set; } = true;
    public string ModelProfile { get; set; } = "realtime";
    public string FallModel { get; set; } = "gmd";
    public List<MonitoringZone> Zones { get; set; } = [];
    public int InactivitySeconds { get; set; } = 120;
    public bool SoundAlerts { get; set; }
    public string RadarPort { get; set; } = "";
    public int RadarBaud { get; set; } = 921600;
    public string RadarControlPort { get; set; } = "";
    public string ApiUrl { get; set; } = "https://api.openai.com/v1";
    public string ApiModel { get; set; } = "gpt-4o-mini";
    public string ProtectedApiKey { get; set; } = "";
    public bool AutoReport { get; set; }
    public int ReportIntervalSeconds { get; set; } = 120;
}
public sealed class Resident
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string Name { get; set; } = "";
    public string Room { get; set; } = "";
    public string Bed { get; set; } = "";
    public string Contact { get; set; } = "";
    public string Notes { get; set; } = "";
    public uint? RadarTargetId { get; set; }
}
public sealed class MonitorEvent
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public DateTimeOffset Time { get; set; } = DateTimeOffset.Now;
    public string Source { get; set; } = "";
    public string Subject { get; set; } = "";
    public string Kind { get; set; } = "";
    public string Evidence { get; set; } = "";
    public bool IsReplay { get; set; }
    public double? VideoPositionSeconds { get; set; }
    public string Resolution { get; set; } = "待处理";
    public string Note { get; set; } = "";
    public DateTimeOffset? ResolvedAt { get; set; }
    [JsonIgnore] public string Mode => IsReplay ? "回放" : "现场";
    [JsonIgnore] public string LocalTime => Time.LocalDateTime.ToString("MM-dd HH:mm:ss");
}
public sealed class VitalSample
{
    public DateTimeOffset Time { get; set; } = DateTimeOffset.Now;
    public uint Target { get; set; }
    public float? HeartRate { get; set; }
    public float? BreathRate { get; set; }
    public bool Moving { get; set; }
    public bool IsReplay { get; set; }
    public float[] HeartWave { get; set; } = [];
    public float[] BreathWave { get; set; } = [];
    [JsonIgnore] public string Mode => IsReplay ? "数据回放" : "雷达实测";
    [JsonIgnore] public string LocalTime => Time.LocalDateTime.ToString("MM-dd HH:mm:ss");
}
public sealed class SavedReport
{
    public DateTimeOffset Time { get; set; } = DateTimeOffset.Now;
    public string Provider { get; set; } = "本地";
    public string Text { get; set; } = "";
    public string Evidence { get; set; } = "";
    [JsonIgnore] public string DisplayTitle => Time.LocalDateTime.ToString("MM-dd HH:mm")+" · "+(Provider=="本地"?"监测报告":"智能分析");
}
public readonly record struct Joint(float X, float Y, float Z, float Confidence);
public readonly record struct Box(float X1, float Y1, float X2, float Y2)
{
    public float Width => X2-X1;
    public float Height => Y2-Y1;
    public bool Clipped => X1 < .005 || Y1 < .005 || X2 > .995 || Y2 > .995;
    public static float Iou(Box a, Box b)
    {
        float intersection = Math.Max(0, Math.Min(a.X2,b.X2)-Math.Max(a.X1,b.X1))*Math.Max(0,Math.Min(a.Y2,b.Y2)-Math.Max(a.Y1,b.Y1));
        return intersection / Math.Max(1e-6f,a.Width*a.Height+b.Width*b.Height-intersection);
    }
}
public sealed record PersonState(int Id, Box Box, float DetectionScore, Joint[] ImageJoints, Joint[] WorldJoints,
    double Time, float? RgbScore = null, float? PoseScore = null, string Activity = "", float Confidence = 0,
    string Status = "识别中", bool Partial = false, int Consecutive = 0, double PredictionTime = -1, string Detector = "SSD",
    string ActivitySource = "视频模型", PersonTelemetry? Telemetry = null, ActionCandidate[]? Alternatives = null, bool? WholeSceneInput = null,
    bool FallMotion=false,string MotionEvidence="",float? ShortRgbScore=null,int ShortConsecutive=0);
public sealed record VideoSnapshot(byte[] Pixels, int Width, int Height, double Time, double Duration, double Fps,
    bool Replay, bool Paused, bool Ended, string Source, long Generation, long Sequence, long CaptureTimestamp=0);
public sealed record TimedObservation(VideoSnapshot Frame, AnalysisSnapshot Analysis);
public sealed record AnalysisSnapshot(long Generation, double Time, PersonState[] People, float? SceneScore,
    string Activity, string Note, double InferenceMs, string Device, float[]? Attention=null, float? FeatureChange=null, double RgbTime=-1,
    AnalyticsSnapshot? Analytics=null,int SceneConsecutive=0);
public sealed record MonitoringZone(Guid Id, string Name, string Source, Box Bounds, int DwellSeconds=60, string Kind="重点区域");
public sealed record ActionCandidate(string Name, float Score);
public sealed record TracePoint(double Time, float X, float Y);
public sealed record ActivityInterval(double Start, double End, string Activity);
public sealed record PersonTelemetry(double ObservedSeconds, double StillSeconds, string Zone, double DwellSeconds,
    IReadOnlyDictionary<string,double> ActivitySeconds);
public sealed record PersonTrace(int Id, TracePoint[] Points, ActivityInterval[] Activities);
public sealed record AnalyticsSnapshot(PersonTrace[] Traces, float[] Heatmap, int HeatmapWidth, int HeatmapHeight,
    double ObservedSeconds, int UniquePeople, int Entries, int Exits, int CurrentZoneOccupancy);
public sealed record VideoEvidence(float[] Scores,float[] Embedding,float[] Attention);
public sealed record RadarTrack(uint Id, Vector3 Position, Vector3 Velocity, float Confidence,Vector3 Acceleration=default);
public sealed record RadarVital(uint Id, float Deviation, float HeartRate, float BreathRate, float[] HeartWave, float[] BreathWave,ushort RangeBin=0);
public sealed record RadarPoint(Vector3 Position,float Doppler,float? Snr=null,float? Noise=null);
public sealed record RadarHeight(uint Id,float MaxZ,float MinZ);
public sealed record RadarFrame(uint Number, Vector3[] Points, RadarTrack[] Tracks, RadarVital[] Vitals,
    RadarPoint[]? PointDetails=null,RadarHeight[]? Heights=null,byte[]? PreviousPointTargets=null,bool? Presence=null,uint[]? TlvTypes=null,uint? OccupancyMask=null);
public sealed record RadarSnapshot(DateTimeOffset Time, string Source, bool Replay, RadarFrame Frame);
public sealed record RadarZone(Guid Id,string Name,float MinX,float MaxX,float MinY,float MaxY,float MinZ,float MaxZ,string Kind="重点区域")
{public bool Contains(Vector3 p)=>p.X>=MinX&&p.X<=MaxX&&p.Y>=MinY&&p.Y<=MaxY&&p.Z>=MinZ&&p.Z<=MaxZ;}
public sealed record RadarTargetObservation(uint Id,Vector3 Position,string Zone,double ObservedSeconds,double RestSeconds,int MotionEpisodes,
    float Speed,float Acceleration,RadarHeight? Height,bool HeightDrop,Vector3[] Trail);
public sealed record RadarObservation(uint Frame,bool? Presence,RadarTargetObservation[] Targets,double ObservedSeconds,int ZoneEntries,int ZoneExits);
public sealed record RadarDiagnostics(long ReceivedBytes,int Frames,int MissedFrames,double FrameHz,uint[] TlvTypes,uint[] UnsupportedTypes);

public static class Labels
{
    public static string Activity(string key) => key switch
    {
        "adl" => "日常活动", "fall" => "疑似跌倒", "fallen" => "倒地状态", "walk" => "行走",
        "sit_down" => "坐下", "sitting" => "坐姿", "lie_down" => "躺下", "lying" => "躺姿",
        "stand_up" => "起身", "standing" => "站姿", "bending" => "弯腰", "crouching" => "蹲姿", "other" => "其他活动", "unknown" => "姿态待确认", _ => "正在分析"
    };
}
