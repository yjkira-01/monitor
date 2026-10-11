using Microsoft.Win32;
namespace CareMonitor.Services;
public sealed record RadarPorts(string Control,string Data);
public static class RadarDevices
{
    public static RadarPorts? Detect()
    {
        using var usb=Registry.LocalMachine.OpenSubKey(@"SYSTEM\CurrentControlSet\Enum\USB");
        if(usb is null)return null;
        var entries=new List<(string Parent,string Port,bool Control)>();
        foreach(var name in usb.GetSubKeyNames().Where(n=>n.StartsWith("VID_10C4&PID_EA70&MI_",StringComparison.OrdinalIgnoreCase)))
        {
            using var device=usb.OpenSubKey(name);if(device is null)continue;
            foreach(var instance in device.GetSubKeyNames())
            {
                using var key=device.OpenSubKey(instance);using var parameters=key?.OpenSubKey("Device Parameters");
                string port=parameters?.GetValue("PortName") as string??"";
                string parent=key?.GetValue("ParentIdPrefix") as string??instance[..Math.Max(0,instance.LastIndexOf('&'))];
                if(port.Length>0)entries.Add((parent,port,name.EndsWith("MI_00",StringComparison.OrdinalIgnoreCase)));
            }
        }
        var available=System.IO.Ports.SerialPort.GetPortNames().ToHashSet(StringComparer.OrdinalIgnoreCase);
        var pairs=entries.Where(e=>e.Control&&available.Contains(e.Port)).SelectMany(c=>entries.Where(d=>!d.Control&&d.Parent==c.Parent&&available.Contains(d.Port)).Select(d=>new RadarPorts(c.Port,d.Port))).Distinct().ToArray();
        return pairs.Length==1?pairs[0]:null;
    }
}
