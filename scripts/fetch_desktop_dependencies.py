"""Populate an ordinary local NuGet feed when the SDK's TLS transport is unavailable."""
from pathlib import Path
import io
import re
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
FEED=ROOT/'desktop/packages';FEED.mkdir(parents=True,exist_ok=True)
SEEDS=[('Microsoft.ML.OnnxRuntime.DirectML','1.20.1'),('OpenCvSharp4','4.10.0.20241108'),
       ('OpenCvSharp4.runtime.win','4.10.0.20241108'),('System.IO.Ports','8.0.0'),
       ('System.Security.Cryptography.ProtectedData','8.0.0'),
       ('Microsoft.NET.ILLink.Tasks','8.0.14')]


def fetch(name,version,seen):
    name=name.lower();key=(name,version)
    if key in seen:return
    seen.add(key);path=FEED/f'{name}.{version}.nupkg'
    if not path.is_file():
        url=f'https://api.nuget.org/v3-flatcontainer/{name}/{version}/{name}.{version}.nupkg'
        with urllib.request.urlopen(url,timeout=90) as r:path.write_bytes(r.read())
    with zipfile.ZipFile(path) as archive:
        # Validate every CRC before relying on the feed.
        assert archive.testzip() is None
        meta=ET.fromstring(archive.read(next(n for n in archive.namelist() if n.endswith('.nuspec'))))
    ns={'n':'http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd'}
    dependencies=next((e for e in meta.iter() if e.tag.split('}')[-1]=='dependencies'),None)
    if dependencies is None:return
    groups=[e for e in dependencies if e.tag.split('}')[-1]=='group']
    if groups:
        priorities=['net8.0-windows','net8.0','net7.0','net6.0','.NETCoreApp3.1','.NETStandard2.1','.NETStandard2.0','netstandard2.0','']
        selected=next((g for p in priorities for g in groups if g.get('targetFramework','')==p),None)
        dependencies=[] if selected is None else selected
    print('feed',name,version,flush=True)
    for dep in dependencies:
        if dep.tag.split('}')[-1]!='dependency':continue
        version=re.search(r'\d+\.\d+(?:\.\d+)?(?:[-\w.]*)',dep.get('version',''))
        if version:fetch(dep.get('id'),version[0],seen)


if __name__=='__main__':
    import sys
    seeds=SEEDS if len(sys.argv)==1 else [(sys.argv[1],sys.argv[2])]
    seen=set()
    for name,version in seeds:fetch(name,version,seen)
