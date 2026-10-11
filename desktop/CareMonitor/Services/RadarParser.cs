using System.Buffers.Binary;
using System.Numerics;
using CareMonitor.Models;

namespace CareMonitor.Services;
public sealed class RadarParser
{
    private static readonly byte[] Magic = [2,1,4,3,6,5,8,7];
    private byte[] tail = [];
    public List<RadarFrame> Feed(ReadOnlySpan<byte> bytes)
    {
        byte[] data = new byte[tail.Length+bytes.Length];tail.CopyTo(data,0);bytes.CopyTo(data.AsSpan(tail.Length));
        var frames = new List<RadarFrame>(); int position=0;
        while(true)
        {
            int found = data.AsSpan(position).IndexOf(Magic);
            if(found<0) { tail=data.AsSpan(Math.Max(position,data.Length-7)).ToArray();return frames; }
            int start=position+found;
            if(data.Length-start<40) { tail=data[start..];return frames; }
            uint length=U(data,start+12);
            if(length<40 || length>1024*1024) { position=start+1;continue; }
            if(data.Length-start<length) { tail=data[start..];return frames; }
            var frame=Parse(data.AsSpan(start,(int)length));
            if(frame is null) { position=start+1;continue; }
            frames.Add(frame);position=start+(int)length;
        }
    }
    private static uint U(ReadOnlySpan<byte> b,int offset) => BinaryPrimitives.ReadUInt32LittleEndian(b[offset..]);
    private static float F(ReadOnlySpan<byte> b,int offset) => BinaryPrimitives.ReadSingleLittleEndian(b[offset..]);
    private static Vector3 V(ReadOnlySpan<byte> b,int offset) => new(F(b,offset),F(b,offset+4),F(b,offset+8));
    private static bool Finite(Vector3 v) => float.IsFinite(v.X) && float.IsFinite(v.Y) && float.IsFinite(v.Z);
    public static RadarFrame? Parse(ReadOnlySpan<byte> b)
    {
        if(b.Length<40 || !b[..8].SequenceEqual(Magic) || U(b,12)!=b.Length) return null;
        uint count=U(b,28), ntlv=U(b,32);int offset=40;
        if(ntlv>(b.Length-40)/8) return null;
        List<Vector3> points=[]; List<RadarTrack> tracks=[];List<RadarVital> vitals=[];List<RadarPoint> details=[];List<RadarHeight> heights=[];List<uint> types=[];
        byte[]? targets=null;bool? presence=null;uint? occupancy=null;ReadOnlySpan<byte> sideInfo=default;
        for(int t=0;t<ntlv;t++)
        {
            if(offset+8>b.Length) return null;
            uint type=U(b,offset),length=U(b,offset+4);offset+=8;
            types.Add(type);
            if(length>b.Length-offset) return null;
            var payload=b.Slice(offset,(int)length);offset+=(int)length;
            if(type==1)
            {
                if(length!=count*16L) return null;
                for(int i=0;i<payload.Length;i+=16) { var p=V(payload,i);if(!Finite(p)||!float.IsFinite(F(payload,i+12)))return null;points.Add(p);details.Add(new(p,F(payload,i+12))); }
            }
            else if(type==1020)
            {
                if(length<20 || length-20!=count*8L) return null;
                var units=new float[5];for(int i=0;i<5;i++) { units[i]=F(payload,i*4);if(!float.IsFinite(units[i])||units[i]<=0)return null; }
                for(int i=20;i<payload.Length;i+=8)
                {
                    float el=(sbyte)payload[i]*units[0],az=(sbyte)payload[i+1]*units[1],r=BinaryPrimitives.ReadUInt16LittleEndian(payload[(i+4)..])*units[3];
                    var p=new Vector3(r*MathF.Cos(el)*MathF.Sin(az),r*MathF.Cos(el)*MathF.Cos(az),r*MathF.Sin(el));
                    float doppler=BinaryPrimitives.ReadInt16LittleEndian(payload[(i+2)..])*units[2],snr=BinaryPrimitives.ReadUInt16LittleEndian(payload[(i+6)..])*units[4];
                    if(!Finite(p)||!float.IsFinite(doppler)||!float.IsFinite(snr))return null;points.Add(p);details.Add(new(p,doppler,snr));
                }
            }
            else if(type==1010)
            {
                if(length%112!=0) return null;
                for(int i=0;i<payload.Length;i+=112)
                {
                    for(int f=i+4;f<i+112;f+=4) if(!float.IsFinite(F(payload,f)))return null;
                    tracks.Add(new(U(payload,i),V(payload,i+4),V(payload,i+16),F(payload,i+108),V(payload,i+28)));
                }
            }
            else if(type==1040)
            {
                if(length!=136) return null;
                for(int i=4;i<136;i+=4) if(!float.IsFinite(F(payload,i)))return null;
                float dev=F(payload,4),hr=F(payload,8),br=F(payload,12);
                if(dev<0||hr<0||br<0)return null;
                float[] heart=new float[15],breath=new float[15];for(int i=0;i<15;i++){heart[i]=F(payload,16+4*i);breath[i]=F(payload,76+4*i);}
                vitals.Add(new(BinaryPrimitives.ReadUInt16LittleEndian(payload),dev,hr,br,heart,breath,BinaryPrimitives.ReadUInt16LittleEndian(payload[2..])));
            }
            else if(type==1011){targets=payload.ToArray();}
            else if(type==1012)
            {
                if(length%12!=0)return null;
                for(int i=0;i<payload.Length;i+=12){float max=F(payload,i+4),min=F(payload,i+8);if(!float.IsFinite(max)||!float.IsFinite(min)||max<min)return null;heights.Add(new(U(payload,i),max,min));}
            }
            else if(type==1021){if(length!=4||U(payload,0)>1)return null;presence=U(payload,0)==1;}
            else if(type==1030){if(length!=4)return null;occupancy=U(payload,0);}
            else if(type==7){if(length!=count*4L)return null;sideInfo=payload;}
        }
        if(!sideInfo.IsEmpty){if(details.Count!=count)return null;for(int i=0;i<details.Count;i++)details[i]=details[i] with{Snr=BinaryPrimitives.ReadUInt16LittleEndian(sideInfo[(i*4)..])*.1f,Noise=BinaryPrimitives.ReadUInt16LittleEndian(sideInfo[(i*4+2)..])*.1f};}
        return new(U(b,20),points.ToArray(),tracks.ToArray(),vitals.ToArray(),details.ToArray(),heights.ToArray(),targets,presence,types.ToArray(),occupancy);
    }
}
