using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

namespace CareMonitor.Services;
public static class CameraDevices
{
    [ComImport,Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"),InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface DeviceEnum { [PreserveSig] int CreateClassEnumerator(ref Guid category,out IEnumMoniker? enumerator,int flags); }
    [ComImport,Guid("55272A00-42CB-11CE-8135-00AA004BB851"),InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface PropertyBag
    {
        [PreserveSig] int Read([MarshalAs(UnmanagedType.LPWStr)] string name,[MarshalAs(UnmanagedType.Struct)] out object value,IntPtr errorLog);
        [PreserveSig] int Write([MarshalAs(UnmanagedType.LPWStr)] string name,ref object value);
    }
    public sealed record Camera(int Index,string Name,string DeviceId=""){public override string ToString()=>$"{Index} · {Name}";}
    public static Camera? Resolve(Camera[] devices,string deviceId,string name,int index)
    {
        if(deviceId.Length>0)return devices.FirstOrDefault(c=>c.DeviceId==deviceId);
        return devices.FirstOrDefault(c=>c.Index==index&&c.Name==name)
            ?? devices.Where(c=>c.Name==name).SingleOrDefaultSafe();
    }
    private static Camera? SingleOrDefaultSafe(this IEnumerable<Camera> values)
    {var list=values.Take(2).ToArray();return list.Length==1?list[0]:null;}
    public static Camera[] List()
    {
        var type=Type.GetTypeFromCLSID(new("62BE5D10-60EB-11D0-BD3B-00A0C911CE86"))!;
        object source=Activator.CreateInstance(type)!;IEnumMoniker? iterator=null;var list=new List<Camera>();
        try
        {
            var category=new Guid("860BB310-5D01-11D0-BD3B-00A0C911CE86");
            if(((DeviceEnum)source).CreateClassEnumerator(ref category,out iterator,0)!=0||iterator is null)return [];
            var monikers=new IMoniker[1];
            while(iterator.Next(1,monikers,IntPtr.Zero)==0)
            {
                object? bag=null;
                try
                {
                    var id=typeof(PropertyBag).GUID;monikers[0].BindToStorage(null!,null!,ref id,out bag!);
                    ((PropertyBag)bag).Read("FriendlyName",out var value,IntPtr.Zero);
                    ((PropertyBag)bag).Read("DevicePath",out var path,IntPtr.Zero);
                    string identity=path?.ToString()??"";
                    if(identity.Length==0){monikers[0].GetDisplayName(null!,null!,out identity);}
                    list.Add(new(list.Count,value?.ToString()??"USB 摄像头",identity));
                }
                finally{if(bag is not null)Marshal.ReleaseComObject(bag);Marshal.ReleaseComObject(monikers[0]);}
            }
            return list.ToArray();
        }
        finally{if(iterator is not null)Marshal.ReleaseComObject(iterator);Marshal.ReleaseComObject(source);}
    }
}
