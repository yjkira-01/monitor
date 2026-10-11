"""Check the distributed single EXE and confirm that packaging preserved model assets."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'desktop/release/CareMonitor-0.3.2-embedded-win-x64.build.json'
OUT = ROOT / 'artifacts/model_optimization_20261011/integrity.json'
ASSET_BASELINE = ROOT / 'desktop/model-assets.sha256.json'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-report', type=Path, default=REPORT)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--asset-baseline', type=Path, default=ASSET_BASELINE)
    args = parser.parse_args()
    build = json.loads(args.build_report.read_text(encoding='utf-8-sig'))
    exe = Path(build['executable'])
    assert list(exe.parent.iterdir()) == [exe], 'Delivery must contain only the EXE'
    assert sha256(exe) == build['executableSha256'], 'EXE checksum mismatch'
    manifest = json.loads(Path(build['embeddedManifest']).read_text(encoding='utf-8-sig'))
    assert sha256(build['embeddedArchive']) == manifest['archiveSha256']
    current = {file['path'].replace('\\', '/'): file for file in manifest['files']}
    assert len(current) == len(manifest['files']), 'Duplicate resource paths'
    with zipfile.ZipFile(build['embeddedArchive']) as archive:
        assert set(archive.namelist()) == set(current), 'Embedded entries differ from manifest'
        for key, file in current.items():
            assert archive.getinfo(key).file_size == file['length'], f'Resource size: {key}'
            with archive.open(key) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == file['sha256'], f'Resource hash: {key}'
        assert archive.testzip() is None, 'Embedded ZIP CRC failure'
    assert not any(key.startswith(('ui-validation/', 'self-test-data/', 'Assets/Preview/'))
                   or key in ('Assets/Realtime/actions.onnx', 'Assets/Realtime/people.onnx', 'Assets/Realtime/rtmpose.onnx')
                   or Path(key).name.startswith(('self-test-', 'ui-startup-error')) for key in current), 'Unexpected research/test files'
    assets = json.loads(args.asset_baseline.read_text(encoding='utf-8-sig'))
    for file in assets:
        key = file['file'].replace('\\', '/')
        assert current[key]['sha256'] == file['sha256'], f'Changed model or asset: {key}'
    with zipfile.ZipFile(build['zip']) as archive:
        assert archive.namelist() == ['CareMonitor.exe'], 'Portable ZIP has external resources'
        assert archive.testzip() is None, 'ZIP CRC failure'
        with archive.open('CareMonitor.exe') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == build['executableSha256']
    result = dict(passed=True, executable_bytes=exe.stat().st_size,
                  executable_sha256=build['executableSha256'],
                  resource_files=len(current), resource_bytes=build['resourceBytes'],
                  preserved_asset_files=len(assets), zip_entries=1,
                  zip_bytes=Path(build['zip']).stat().st_size,
                  zip_sha256=sha256(build['zip']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
