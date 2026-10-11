"""App-local VC runtime from Microsoft's signed Desktop framework distribution."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE='https://aka.ms/Microsoft.VCLibs.x64.14.00.Desktop.appx'
archive=ROOT/'desktop/packages/Microsoft.VCLibs.x64.14.00.Desktop.appx'
archive.parent.mkdir(parents=True,exist_ok=True)
if not archive.is_file():
    with urllib.request.urlopen(SOURCE,timeout=60) as response:archive.write_bytes(response.read())
output=ROOT/'desktop/CareMonitor/Runtime';output.mkdir(parents=True,exist_ok=True)
names=['concrt140.dll','msvcp140.dll','msvcp140_1.dll','msvcp140_2.dll','msvcp140_atomic_wait.dll',
       'msvcp140_codecvt_ids.dll','vcruntime140.dll','vcruntime140_1.dll']
files=[]
with zipfile.ZipFile(archive) as package:
    assert package.testzip() is None
    for name in names:
        data=package.read(name);assert data[:2]==b'MZ'
        (output/name).write_bytes(data)
        files.append({'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
(ROOT/'desktop/CareMonitor/Assets/vc-runtime-provenance.json').write_text(json.dumps({
    'source':SOURCE,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':files,
    'note':'Microsoft VC runtime; Microsoft license terms apply. App-local copies; no system installation.'},indent=2),encoding='utf-8')
print('prepared official app-local VC runtime',len(files))
