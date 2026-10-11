using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using CareMonitor.Models;

namespace CareMonitor.Services;
public sealed class StoreData
{
    public int Schema { get; set; } = 1;
    public AppSettings Settings { get; set; } = new();
    public List<Resident> Residents { get; set; } = [];
    public List<MonitorEvent> Events { get; set; } = [];
    public List<VitalSample> Vitals { get; set; } = [];
    public List<SavedReport> Reports { get; set; } = [];
}
public sealed class AppStore
{
    public string DirectoryPath { get; }
    public StoreData Data { get; private set; }
    public string? RecoveryNote { get; private set; }
    private readonly object sync = new();
    private readonly JsonSerializerOptions json = new() { WriteIndented = true };
    public AppStore(string? directory = null)
    {
        DirectoryPath = directory ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "CareMonitor");
        Directory.CreateDirectory(DirectoryPath);
        string path = Path.Combine(DirectoryPath,"monitor.json");
        Data = new();
        if (!File.Exists(path)) return;
        try { Data = Read(path); }
        catch (Exception ex) when (ex is JsonException or IOException or InvalidDataException)
        {
            File.Copy(path,Path.Combine(DirectoryPath,$"monitor-damaged-{DateTime.Now:yyyyMMddHHmmss}.json"),true);
            bool restored=false;
            if (File.Exists(path+".bak"))
                try{Data=Read(path+".bak");restored=true;}
                catch(Exception backupError) when(backupError is JsonException or IOException or InvalidDataException){ }
            RecoveryNote = restored?"档案文件异常，已保留原文件并恢复可读取的备份，请检查历史记录。":"档案及备份无法读取，原文件已保留，已创建空白档案。";
        }
    }
    private StoreData Read(string path)
    {
        var data = JsonSerializer.Deserialize<StoreData>(File.ReadAllText(path),json) ?? throw new InvalidDataException("档案为空");
        if (data.Schema != 1 || data.Settings is null || data.Residents is null || data.Events is null || data.Vitals is null || data.Reports is null)
            throw new InvalidDataException("档案格式不兼容");
        data.Settings.Zones??=[];data.Settings.RadarZones??=[];
        data.Settings.CameraDeviceId??="";data.Settings.SecondaryCameraId??="";data.Settings.CameraDeviceName??="";
        data.Settings.ProtectedApiKey??="";data.Settings.RadarPort??="";data.Settings.RadarControlPort??="";
        data.Settings.ModelProfile??="realtime";data.Settings.FallModel??="gmd";
        return data;
    }
    public void Save()
    {
        lock(sync)
        {
            if(Data.Events.Count>10000){Archive("events",Data.Events.Take(5000));Data.Events.RemoveRange(0,5000);}
            if(Data.Reports.Count>2000){Archive("reports",Data.Reports.Take(500));Data.Reports.RemoveRange(0,500);}
            string path = Path.Combine(DirectoryPath,"monitor.json"), temp = path+".tmp";
            File.WriteAllText(temp,JsonSerializer.Serialize(Data,json),new UTF8Encoding(false));
            using(var stream = new FileStream(temp,FileMode.Open,FileAccess.ReadWrite,FileShare.None)) stream.Flush(true);
            if (File.Exists(path)) File.Replace(temp,path,path+".bak"); else File.Move(temp,path);
        }
    }
    public void SaveResident(Resident resident)
    {
        if (string.IsNullOrWhiteSpace(resident.Name) || resident.Name.Length>100 || resident.Notes.Length>4000)
            throw new ArgumentException("请填写姓名（100字以内），备注不超过4000字。");
        if (resident.RadarTargetId.HasValue && Data.Residents.Any(r=>r.Id!=resident.Id && r.RadarTargetId==resident.RadarTargetId))
            throw new ArgumentException("该雷达目标已绑定其他档案，请先解除原绑定。");
        if (!string.IsNullOrEmpty(resident.Bed) && Data.Residents.Any(r=>r.Id!=resident.Id && r.Room==resident.Room && r.Bed==resident.Bed))
            throw new ArgumentException("该房间床位已被使用。");
        int index = Data.Residents.FindIndex(r=>r.Id==resident.Id);
        if(index<0) Data.Residents.Add(resident); else Data.Residents[index]=resident;
        Save();
    }
    public void AddEvent(MonitorEvent item) { Data.Events.Add(item); Save(); }
    public void AddVital(VitalSample sample)
    {
        Data.Vitals.Add(sample);
        // shortcut: keep the last 20,000 waveform records locally; larger deployments need an indexed database.
        if(Data.Vitals.Count>20000) { Archive("vitals",Data.Vitals.Take(5000)); Data.Vitals.RemoveRange(0,5000); }
    }
    private void Archive<T>(string prefix,IEnumerable<T> rows) => File.WriteAllText(Path.Combine(DirectoryPath,$"{prefix}-{DateTime.UtcNow:yyyyMMddHHmmssfff}.json"),JsonSerializer.Serialize(rows,json));
    public static string ProtectKey(string plain) => plain.Length==0 ? "" : Convert.ToBase64String(ProtectedData.Protect(Encoding.UTF8.GetBytes(plain),null,DataProtectionScope.CurrentUser));
    public static string UnprotectKey(string encrypted) => encrypted.Length==0 ? "" : Encoding.UTF8.GetString(ProtectedData.Unprotect(Convert.FromBase64String(encrypted),null,DataProtectionScope.CurrentUser));
}
