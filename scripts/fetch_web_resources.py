"""Download the pinned web resource bundle and restore it without overwriting local files."""
import json
from pathlib import Path
import urllib.request

from project_archive import _bounded, _sha, restore

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'web-resources.json').read_text(encoding='utf-8'))
    if not manifest['url'].startswith('https://github.com/yjkira-01/monitor/releases/download/'):
        raise ValueError('Resources must come from the project release')
    directory = ROOT/'archive'
    directory.mkdir(exist_ok=True)
    package = _bounded('archive/'+manifest['archive'])
    if package.exists():
        with package.open('rb') as stream:
            valid = _sha(stream) == manifest['archiveSha256']
    else:
        valid = False
    if not valid:
        temporary = package.with_suffix('.zip.part')
        request = urllib.request.Request(manifest['url'], headers={'User-Agent': 'CareMonitor-Setup'})
        try:
            with urllib.request.urlopen(request, timeout=60) as source, temporary.open('wb') as output:
                size = 0
                while block := source.read(1024*1024):
                    size += len(block)
                    if size > manifest['archiveBytes']:
                        raise ValueError('Resource download exceeds the pinned length')
                    output.write(block)
            with temporary.open('rb') as stream:
                if temporary.stat().st_size != manifest['archiveBytes'] or _sha(stream) != manifest['archiveSha256']:
                    raise ValueError('Resource checksum mismatch; existing resources were preserved')
            temporary.replace(package)
        finally:
            temporary.unlink(missing_ok=True)
    (directory/'web-resources.manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    for target in manifest['targets']:
        print(json.dumps(restore(target), ensure_ascii=False))


if __name__ == '__main__':
    main()
