"""Restore a selected, SHA256-verified project archive without overwriting files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def _sha(stream) -> str:
    digest = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def _bounded(relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Unsafe archive path: {relative}')
    result = ROOT.joinpath(*path.parts)
    if not result.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f'Path leaves the project: {relative}')
    return result


def ensure_dependency(name: str) -> Path:
    target = ROOT / 'artifacts' / name
    if not target.is_dir():
        restore(f'artifacts/{name}')
    return target


def restore(selected: str) -> dict:
    selected = selected.replace('\\', '/').rstrip('/')
    _bounded(selected)
    for manifest_file in (ROOT / 'archive').glob('*.manifest.json'):
        manifest = json.loads(manifest_file.read_text(encoding='utf-8-sig'))
        if selected not in manifest['targets']:
            continue
        archive = manifest_file.with_name(manifest['archive'])
        # The manifest is local, but never allow a changed manifest to address another location.
        if archive.parent != manifest_file.parent or archive.name != manifest['archive']:
            raise ValueError('Invalid archive filename')
        with archive.open('rb') as stream:
            if _sha(stream) != manifest['archiveSha256']:
                raise ValueError(f'Archive SHA256 mismatch: {archive.name}')
        records = [r for r in manifest['files'] if r['path'] == selected or r['path'].startswith(selected + '/')]
        directories = [d for d in manifest['directories'] if d.rstrip('/') == selected or d.startswith(selected + '/')]
        with zipfile.ZipFile(archive) as package:
            # Validate every selected entry and existing destination before making any changes.
            for record in records:
                destination = _bounded(record['path'])
                entry = package.getinfo(record['path'])
                if entry.file_size != record['bytes']:
                    raise ValueError(f'Entry length mismatch: {record["path"]}')
                with package.open(entry) as stream:
                    if _sha(stream) != record['sha256']:
                        raise ValueError(f'Entry SHA256 mismatch: {record["path"]}')
                if destination.exists():
                    with destination.open('rb') as stream:
                        if _sha(stream) != record['sha256']:
                            raise FileExistsError(f'Refusing to overwrite changed file: {record["path"]}')
            for directory in directories:
                _bounded(directory).mkdir(parents=True, exist_ok=True)
            for record in records:
                destination = _bounded(record['path'])
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation protects existing content if it appeared during validation.
                with package.open(record['path']) as source, destination.open('xb') as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                with destination.open('rb') as stream:
                    if _sha(stream) != record['sha256']:
                        raise ValueError(f'Restored file SHA256 mismatch: {record["path"]}')
        return {'target': selected, 'files': len(records), 'verified': True}
    raise FileNotFoundError(f'No project archive contains {selected}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', nargs='?', help='Original project-relative path to restore')
    args = parser.parse_args()
    if args.target:
        print(json.dumps(restore(args.target), ensure_ascii=False))
    else:
        for path in (ROOT / 'archive').glob('*.manifest.json'):
            manifest = json.loads(path.read_text(encoding='utf-8-sig'))
            print(json.dumps({'archive': manifest['archive'], 'targets': manifest['targets'], 'files': len(manifest['files'])}, ensure_ascii=False))
